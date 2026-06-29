'''
Model Inference Pipeline

Test Data features: ['input_ids', 'labels', 'attention_mask', 'decoder_attention_mask']

Higher perplexity and losses are expected if the model was trained on smaller dataset. 
Also, some papers are harder to infer and thus may cause higher losses and perplexities.

Infer block in this case is same as val_block. So, we will simply call that function to compute inference loss.
'''

import os, sys, torch
from torch.utils.data import DataLoader
import time

from datasets import load_from_disk 
from building_blocks import model_comps, load_checkpoint, forward_pass, val_block
from utils import Logger


def loss_inference(
              device, dataloader, model_path, res_test_path, run_id, DEBUG,
              d_model = 512, d_ffn_enc = 1408, d_ffn_dec = 1408, enc_heads = 8, dec_heads = 8, 
              cross_heads = 8, enc_blocks = 5, dec_blocks = 10,
              vocab_size = 50265, emb_drop = 0, enc_att_drop = 0, enc_ffn_drop = 0, mask_att_drop = 0, 
              cross_att_drop = 0, dec_ffn_drop = 0, learning_rate = 0, weight_decay = 0, 
              sch_mode = None, sch_factor = 0, sch_patience = 0
 ):
    '''
    Compute loss based inference for the model.
    '''
    start_time = time.time()
    # initialize model component objects
    emb, enc, dec, out_proj, loss_func, _, _, _ = model_comps(
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

    infer_losses = [] # all inference losses for all batches
    plexes = [] # perplexity per inference loss

    for batch_index, batch in enumerate(dataloader, 1):
        batch_loss = val_block(batch, emb, enc, dec, out_proj, loss_func, vocab_size, device, DEBUG)
        infer_losses.append(batch_loss)  # batch loss is not in tensor format

        perplexity = torch.exp(torch.tensor(batch_loss))
        plexes.append(perplexity.item())

        # save batch wise (per paper) loss and perplexity
        with open(res_test_path, "a") as f:
            f.write(f"{run_id}, {batch_index}, {batch_loss}, {perplexity.item()}\n")
        
        print(f"\nbatch = {batch_index} | loss = {batch_loss} | perplexity = {perplexity.item()}")

    print(f"\naverage inference loss       = {sum(infer_losses) / batch_index}\n"
          f"\naverage inference perplexity = {sum(plexes) / batch_index}\n"
          f"\ntotal time                   = {time.time() - start_time} seconds\n"
          )

    return 



def main(test_data_path = "./tokenized-dataset/test"):
    ''' set up device '''

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
    test_samples      = 1000    # number of test data to be used for inference
    batch_size        = 1       # how many samples to process at a time per batch


    ''' Set up inference data '''
    # load test data; data will be loaded on device during forward pass
    full_test_data    = load_from_disk(test_data_path)
    test_data         = full_test_data.select(range(test_samples))
    test_data         = test_data.with_format("torch")   # convert Python list into torch tensor

    dataloader        = DataLoader(test_data, batch_size = batch_size)
    res_test_path     = "./visualization/test_out_0.csv"      # path for loss logging for visualization
    run_id            = f"test_samples{test_samples}"         # run ID in the parameter output

    DEBUG             = False


    loss_inference(
              device, dataloader, model_path, res_test_path, run_id, DEBUG,
              d_model = 512, d_ffn_enc = 1408, d_ffn_dec = 1408, enc_heads = 8, dec_heads = 8, 
              cross_heads = 8, enc_blocks = 5, dec_blocks = 10,
              vocab_size = 50265, emb_drop = 0, enc_att_drop = 0, enc_ffn_drop = 0, mask_att_drop = 0, 
              cross_att_drop = 0, dec_ffn_drop = 0, learning_rate = 0, weight_decay = 0, 
              sch_mode = "min", sch_factor = 0, sch_patience = 0
            )


if __name__ == "__main__":
    # load test data
    test_data_path = "./tokenized-dataset/test"
    main(test_data_path)


    
