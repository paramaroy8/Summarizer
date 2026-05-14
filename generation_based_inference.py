'''
Generation Based Inference

Decoder will generate human readable texts from generated tokens.

Token IDs:
           • BOS = 0
           • EOS = 2 
           • PAD = 1
'''

import os, sys, torch
from torch.utils.data import DataLoader
import time
from transformers import BartTokenizer

from datasets import load_from_disk 
from building_blocks import model_comps, load_checkpoint
from utils import Logger



def infer_block(tokenizer, max_len, batch, emb, enc, dec, out_proj, vocab_size, device, DEBUG):
    '''
    Decoder Mask will not be needed since there is abstract the decoder has to compare with.
    So, for each generated token, it can only see that token and all tokens generated so far.
    '''
    ''' Encoder input '''
    enc_input = batch["input_ids"].to(device)

    enc_pad   = batch["attention_mask"].bool().to(device)  # (batch_size, enc_seq_len)
    # (batch_size, 1, enc_seq_len) & (batch_size, enc_seq_len, 1) = (batch_size, enc_seq_len, seq_len)
    enc_mask = (enc_pad.unsqueeze(1) & enc_pad.unsqueeze(2)).unsqueeze(1)   # (batch_size, 1, enc_seq_len, enc_seq_len)

    ''' Decoder Input Parameters '''
    # BOS token ID
    BOS_token = tokenizer.bos_token_id  # 0
    # EOS token ID
    EOS_token = tokenizer.eos_token_id  # 2
    # initially, the decoder has only the BOS token
    dec_input = torch.tensor([[BOS_token]], dtype = torch.long).to(device)  # (batch_size = 1, dec_seq_len = 1 (only BOS token initally))
    
    dec_mask = None # no decoder mask needed since it can only see itself

    cross_mask = enc_pad.unsqueeze(1).unsqueeze(1)    # (batch_size, 1, 1, enc_seq_len)

    # encoder will have source embedding but decoder will have to start from scratch
    source_emb = emb(token_ids = enc_input)

    # encoder output
    enc_out = enc(x = source_emb, att_mask = enc_mask)
    del source_emb                   # free memory

    while (dec_input.shape[1] < max_len):
        # embed the input : (batch_size = 1, dec_seq_len = current number of tokens being processed, d_model)
        dec_emb = emb(token_ids = dec_input) 
        
        # decoder raw output : (batch_size, dec_seq_len, d_model)
        dec_out = dec(
                  x        = dec_emb, 
                  enc_out  = enc_out, 
                  dec_mask = dec_mask, 
                  enc_mask = cross_mask
                )    
        
        # compute logits (hidden states) by projecting decoder output to tokenizer vocab_size
        logits = out_proj(dec_out)   # (batch_size = 1, current_len, vocab_size)

        '''
        Last state among the hidden states is the new token.
        '''
        last_token = logits[:, -1, :]  # (batch_size = 1, vocab_size)

        '''
        Based on all comparison scores, choose the token with highest comparison score across vocab_size.

        argmax returns single value in as 1D tensor. To keep it in 2D, we use keepdim = True
        '''
        new_token = torch.argmax(last_token, dim = -1, keepdim = True) # (batch_size = 1, dec_seq_len = 1)

        '''
        If the new token is end token, then summarization is complete.
        Else, we add it back to decoder input for the next token generation.
        '''
        if new_token.item() == EOS_token:
            # if the new token is the end of summary, then generation loop can stop
            break

        # add new token to the decoder input for the next iteration
        dec_input = torch.cat([dec_input, new_token], dim = 1) # (batch_size = 1, dec_seq_len = current input len + 1)
    
    # remove the BOS token and return
    return dec_input[:, 1:]   # (batch_size = 1, dec_seq_len = num of generated tokens)



def generate_inference(
              device, dataloader, model_path, res_test_path, run_id, temperature, max_len, DEBUG,
              d_model = 512, d_ffn_enc = 1408, d_ffn_dec = 1408, enc_heads = 8, dec_heads = 8, 
              cross_heads = 8, enc_blocks = 5, dec_blocks = 10,
              vocab_size = 50265, emb_drop = 0, enc_att_drop = 0, enc_ffn_drop = 0, mask_att_drop = 0, 
              cross_att_drop = 0, dec_ffn_drop = 0, learning_rate = 0, weight_decay = 0, 
              sch_mode = "min", sch_factor = 0, sch_patience = 0
    ):

    '''
    Generate decoder output without seeing the abstract (target label).
    '''

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    start_time = time.time()
    # initialize model component objects
    emb, enc, dec, out_proj, _, _, _, _ = model_comps(
                                device, d_model, d_ffn_enc, d_ffn_dec, vocab_size, 
                                enc_heads, dec_heads, cross_heads, enc_blocks, dec_blocks, 
                                emb_drop, enc_att_drop, enc_ffn_drop, mask_att_drop, 
                                cross_att_drop, dec_ffn_drop, DEBUG, learning_rate, weight_decay, 
                                sch_mode, sch_factor, sch_patience
                                )
    # load_checkpoint loads the model and returns start epoch
    # inference does not have epoch, so we will ignore it
    _ = load_checkpoint(device, model_path, emb, enc, dec, out_proj, optimizer = None, scheduler = None)

    emb.eval()
    enc.eval()
    dec.eval()
    out_proj.eval()

    with torch.no_grad():

        for batch_index, batch in enumerate(dataloader, 1):
            # generated summary of the current batch (paper)
            generated_ids = infer_block(tokenizer, max_len, batch, emb, enc, dec, out_proj, vocab_size, device, DEBUG)

            # decode the generated IDs and tokenized abstract
            # tokenizer.decode() returns string; expects dim (seq_len,)
            summary = tokenizer.decode(
                                        generated_ids[0].cpu().tolist(), 
                                        skip_special_tokens = True
                                    )
            labels = batch["labels"][0]
            # get rid of pad tokens
            labels = labels[labels != -100]

            reference = tokenizer.decode(
                                        labels.cpu().tolist(), 
                                        skip_special_tokens = True
                                    )

            print(
                f"\nPaper = {batch_index}\n"
                f"\nSummary = {summary}\n"
                f"\nReference = {reference}\n"
            )


            # save as CSV for later analysis
            with open(res_test_path, "a") as f:
                f.write(f"{run_id}, {batch_index}, {summary}, {reference}\n")

    return 
    


def main(test_data_path = "./tokenized-dataset/test"):
    device = torch.device(
                            "mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available()
                            else "cpu"
                         )

    ''' set up directories '''

    os.makedirs("./console_logs/", exist_ok = True)
    write_choice = "a"       # "w" to overwrite, "a" to keep everything

    os.makedirs("./visualization/", exist_ok = True)

    # set up console output saving
    sys.stdout = Logger(filepath = "./console_logs/infer_console.txt")

    print(f"\ninference on device = {device}")

    ''' ------ Set up experiment parameters ------ '''

    model_path        = "./checkpoints/latest_model.pt"  # load model
    test_samples      = 1    # number of test data to be used for inference
    batch_size        = 1       # how many samples to process at a time per batch
    temperature       = 0.1     # how deterministic we want the model to be
    max_len           = 500     # maximum length the summary can have

    ''' Set up inference data '''
    # load test data; data will be loaded on device during forward pass
    full_test_data    = load_from_disk(test_data_path)
    test_data         = full_test_data.select(range(test_samples))
    test_data         = test_data.with_format("torch")   # convert Python list into torch tensor

    dataloader        = DataLoader(test_data, batch_size = batch_size)
    res_test_path     = "./visualization/summaries_0.csv"      # path for loss logging for visualization
    run_id            = f"test_samples{test_samples}"         # run ID in the parameter output

    DEBUG             = False


    generate_inference(
              device, dataloader, model_path, res_test_path, run_id, temperature, max_len, DEBUG,
              d_model = 512, d_ffn_enc = 1408, d_ffn_dec = 1408, enc_heads = 8, dec_heads = 8, 
              cross_heads = 8, enc_blocks = 5, dec_blocks = 10,
              vocab_size = 50265, emb_drop = 0, enc_att_drop = 0, enc_ffn_drop = 0, mask_att_drop = 0, 
              cross_att_drop = 0, dec_ffn_drop = 0, learning_rate = 0, weight_decay = 0, 
              sch_mode = "min", sch_factor = 0, sch_patience = 0
            )


    return 


if __name__ == "__main__":
    # load test data
    test_data_path = "./tokenized-dataset/test"
    main(test_data_path)
