
import torch
from dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_index = tokenizer_tgt.token_to_id('[SOS]')
    eos_index = tokenizer_tgt.token_to_id('[EOS]')

    #precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    #initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_index).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        #build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #get the next token
        probab = model.project(out[:,-1])

        #select the token with the maximum probability
        _, next_word = torch.max(probab, dim=1)

        #the moment we get the next word we will append it to the decoder input which will be used for the enxt word.
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)


        if next_word == eos_index: #if next word is end of token we'll stop the search
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_epoch, writer, num_example = 2):
    #function to run

    model.eval()
    count = 0


    # source_texts = []
    # expected = []
    # predicted =[]

    #size of the control window
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) ==1, "Batch size must be one for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]

            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            #print to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE : {source_text}')
            print_msg(f'TARGET : {target_text}')
            print_msg(f'PREDICTED : {model_out_text}')

            if count == num_example:
                break

