import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MutilingualDataset(Dataset):

    #defining constructor and passing input to it. 
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds) #returns total length of the dataset
    
    
    
    def __getitem__(self, index: any) -> any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        #converting each text into tokens. 
        #first it'll split sentence into words. and then map each word to a corresponding number in the vocabulary
    
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids #the output will be array
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids #the output will be array

       #since model works on fixed length of tokens each time. many a times we might not have same words in the sentence. in that case we use padding(PAD) tokens until it reaches sequence length.
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2 #-2 coz we also have SOS and EOS.
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # we only have EOS in decoder side. while both EOS and POS in encoder side.

        # since these values should never become negative we add a check here.

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long bruh')
        
        #adding SOS and EOS to the encoder input. Note THAT we are increasing size of encoder input by adding padding tokens but we do not want it to be used during training. We'll create a mask for this later.
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        #only adding SOS in decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len


        return { "encoder_input" : encoder_input, #(of size seq_len)
                "Decoder_input" : decoder_input,
                "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #this is now (1,1,seq_len) size. It'll be used in self attention mechanism.
                "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  #this is (1,Seq_len) & (1, seq_len, seq_len)
                "label" : label,
                "src_text" : src_text,
                "tgt_text" : tgt_text
                } #this finally makes out dataset
    



def causal_mask(size):
        mask = torch.triu(torch.ones(1, size,size), diagonal=1).type(torch.int) #triu in torch means give me all the values above the diagonal that i'm providing. 
        return mask == 0