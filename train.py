
import torch
import torch.nn as nn
from pathlib import Path
import warnings
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

#tokenizer is used to split sentence into single words or tokens. There are different types of tokenizer. We are using Word by word here.
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

#import model
from model import build_transformer

#import necessary functions from dataset and config file required for training loop.
from dataset import MutilingualDataset, causal_mask
from config import get_weights_file_path, get_config
from validation import greedy_decode, run_validation




#training this on Opus Dataset. It's quite famous one in translation world :)
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang): #we'll define configuration of our model later.
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print("Tokenizer Created")

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print("Tokenizer Existed already")

    return tokenizer

#code to get datasets
def get_ds(config):
    ds_raw = load_dataset("opus_books",f"{config['lang_src']}-{config['lang_tgt']}", split = "train")

    #building tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    print("Both Source and target tokenizer build done")

    # splitting data set in training, and validation set. Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    print("Data Successfully Splitted into Train and Validation Set")
    #making our dataset as described in dataset file.
    train_ds = MutilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = MutilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    #here we also want to watch what is the maximum seq length of each input. 
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
    
    print(f'Max length of source sentence : {max_len_src}')
    print(f'Max length of target sentence : {max_len_tgt}')
    
    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle= True)
    val_dataloader = DataLoader(train_ds, batch_size=1, shuffle= True)
    print("Data Loaded Successfully")
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])    
    print("Transformer Model Architecture Called ")
    return model

#we have defined our configurations in config.py file. Now we can define our trianing loop.



def train_model(config):
    print("Training Started .....")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device : {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True) #making sure weights folder exists

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    #starting Tensorboard to visualize loss and other charts

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_epoch = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading Model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_epoch = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch : 02d}")
        
        
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) # (B, Seq_len)
            decoder_input = batch['encoder_input'].to(device) #(b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1,1,seq_len) #here we are only tellin it hide padding tokens
            decoder_mask = batch['decoder_mask'].to(device) #(b,1,seq_len, seq_len) 

            #let's run the tensors through the transformer

            encoder_output = model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (b, seq_len, d_model) but now we want to project it back it into vocabulary. So we need to change it's dimension again
            proj_output = model.project(decoder_output) #(b, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) #(b, seq_len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

            #log the loss
            writer.add_scalar('train_loss', loss.item(), global_epoch)
            writer.flush()

            #backpropagate the loss
            loss.backward()

            #update the weigths
            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_epoch, writer)
            
            global_epoch+=1



        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'eppch' : epoch,
            'model_state_dict' : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            'global_epoch' : global_epoch
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)