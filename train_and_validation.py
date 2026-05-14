'''
Modularized training pipeline with validation.

In this file, I introduced chunk size training data so that we don't have to train on the same set every time.

Multiple epochs can be simulated by calling the train_val function several times and saving the model parameters each time.
'''
import os 
import sys
import torch 
from torch.utils.data import DataLoader 

from datasets import load_from_disk
from building_blocks import model_comps, load_checkpoint, train_block, val_block 
from utils import Logger 

import time 


def train_val(
            device, train_path, val_path, d_model, d_ffn_enc, d_ffn_dec, vocab_size, DEBUG, batch_size, val_samples,
            enc_blocks, dec_blocks, emb_drop, enc_att_drop, enc_ffn_drop, mask_att_drop, cross_att_drop, dec_ffn_drop,
            enc_heads, dec_heads, cross_heads, checkpoint_path, 
            learning_rate, weight_decay, epochs, sch_mode, sch_factor, sch_patience, grad_acc, 
            res_train_path, run_id, res_val_path, console_path, train_chunk_size
         
        ):
    
    try:
        # load data
        train_data = load_from_disk(train_path)
        # huggingface stores data in list format; convert it into torch tensor
        train_data = train_data.with_format("torch")

        # load data
        val_data = load_from_disk(val_path)
        # huggingface stores data in list format; convert it into torch tensor
        val_data = val_data.with_format("torch")

        # get model compoenent class objects
        emb, enc, dec, out_proj, loss_func, all_params, optimizer, scheduler = model_comps(device, d_model, d_ffn_enc, d_ffn_dec, vocab_size, enc_heads, dec_heads, cross_heads, enc_blocks, dec_blocks, 
                emb_drop, enc_att_drop, enc_ffn_drop, mask_att_drop, cross_att_drop, dec_ffn_drop, DEBUG,
                learning_rate, weight_decay, sch_mode, sch_factor, sch_patience
                )
        
        # count paramters
        total_params = sum(p.numel() for p in all_params)
        learnable_params = sum(p.numel() for p in all_params if p.requires_grad)
        print(f"total model parameters = {total_params}, total learnable parameters = {learnable_params}\n")
        
        ''' Train '''
        start_epoch, train_start_index = load_checkpoint(device, checkpoint_path, emb, enc, dec, out_proj, optimizer, scheduler)
        
        train_end_index = min((train_start_index + train_chunk_size), len(train_data))

        if train_end_index == len(train_data):
            print("\n\ntraining with last portion of complete training data\n\n")
        
        train_sample = train_data.select(range(train_start_index, train_end_index))

        print(f"\ntrain start index = {train_start_index} , train end index = {train_end_index}\n")

        # load desired amount of data for TRAINING
        train_dataloader = DataLoader(train_sample, batch_size = batch_size, shuffle = True)
        data_len = len(train_dataloader)
        
        # select val samples
        val_sample = val_data.select(range(val_samples)) # first two elements in the tokenized train data
        # no need to shuffle validation data
        val_dataloader = DataLoader(val_sample, batch_size = batch_size, shuffle = False)
        
        # start time for training
        train_start = time.time()

        # begin training
        for ep in range(start_epoch, start_epoch + epochs):
            train_epoch_loss_sum = 0

            for batch_index, batch in enumerate(train_dataloader, 1):        # 1 to make starting index as 1 to avoid 0 division
                batch_start = time.time()
                # add all batches losses in the current epoch
                train_batch_loss = train_block(batch_index, data_len, batch, emb, enc, dec, out_proj, loss_func, optimizer, all_params, vocab_size, device, grad_acc, DEBUG)
                train_epoch_loss_sum += train_batch_loss
                print(f"Epoch = {ep} | Batch = {batch_index} | Batch Loss = {train_batch_loss:.4f}")

                batch_time = time.time() - batch_start
                print(f"batch time = {batch_time:.4f} seconds\n")

                # if mps device is being used, empty its cache
                if device.type == "mps":
                    torch.mps.empty_cache() # release unused cached memory
                
            # average of all batch losses in the current epoch
            train_epoch_loss = train_epoch_loss_sum / batch_index  # batch_index starts with 1 so no 0 division possible
            # compute perplexity
            train_perplexity = torch.exp(torch.tensor(train_epoch_loss)) 

            total_train_time = time.time() - train_start

            print(f"\nEpoch {ep} | Train Loss = {train_epoch_loss:.4f} | Train Perplexity = {train_perplexity:.4f}\n")

            print(f"\ntotal train time = {total_train_time} seconds\n")

            # save outputs for visualization: epoch, average loss per epoch
            with open(res_train_path, "a") as f:
                f.write(f"{run_id}, {ep}, {train_epoch_loss}, {train_perplexity.item()}\n") # perplexity score is torch tensor, keep the value only


            ''' Validate '''
            # set them to eval mode to prevent gradient computation
            emb.eval()
            enc.eval()
            dec.eval()
            out_proj.eval()
            
            val_loss_sum = 0

            for batch_index, batch in enumerate(val_dataloader, 1):
                val_loss = val_block(batch, emb, enc, dec, out_proj, loss_func, vocab_size, device, DEBUG)
                val_loss_sum += val_loss
            
                print(f"\nEpoch = {ep} | Batch = {batch_index} | Val Batch Loss = {val_loss}\n")
            
            val_epoch_loss = val_loss_sum / batch_index
            
            # compute perplexity
            val_perplexity = torch.exp(torch.tensor(val_epoch_loss)) 

            print(f"\nEpoch = {ep} | Validation Loss = {val_epoch_loss} | Validation Perplexity = {val_perplexity}\n")

            # save outputs for visualization: epoch, average loss per epoch
            with open(res_val_path, "a") as f:
                f.write(f"{run_id}, {ep}, {val_epoch_loss}, {val_perplexity.item()}\n") # perplexity score is torch tensor, keep the value only

            ''' Resume Training '''
            emb.train()
            enc.train()
            dec.train()
            out_proj.train()

            total_time = time.time() - train_start
            print(f"\ntrain + val time = {total_time} seconds\n")

            next_epoch = ep # for next iteration

            if train_end_index == len(train_data):
                # we have reached the end of current epoch, and time to start new epoch
                # in that case, training index starts from 0
                next_epoch = ep + 1
                train_end_index = 0
            
            scheduler.step(val_epoch_loss)

            # save model after this epoch
            torch.save({
                            "epoch"               : next_epoch,                  # next starting epoch 
                            "train_end_index"     : train_end_index,             # act as the starting index for next iteration
                            "emb"                 : emb.state_dict(),
                            "enc"                 : enc.state_dict(),
                            "dec"                 : dec.state_dict(),
                            "out_proj"            : out_proj.state_dict(),
                            "optimizer"           : optimizer.state_dict(),
                            "scheduler"           : scheduler.state_dict()
                        }, checkpoint_path)
            

            print(f"saved model for epoch = {ep}")
            

    
    except Exception as e:
        print(f"\nERROR = {e}\n")
    
    return 
