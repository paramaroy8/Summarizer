'''
Various functions to facilitate training, validation and testing of the model.
'''
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from embeddings import FullEmbedding 
from full_encoder import Encoder
from full_decoder import Decoder 

def model_comps(device, d_model, d_ffn_enc, d_ffn_dec, vocab_size, enc_heads, dec_heads, cross_heads, enc_blocks, dec_blocks, 
                emb_drop, enc_att_drop, enc_ffn_drop, mask_att_drop, cross_att_drop, dec_ffn_drop, DEBUG,
                learning_rate, weight_decay, sch_mode, sch_factor, sch_patience
                ):
           
    '''
    Initialize component objects and return.
    '''
    # initialize full embedding instance and load it to device
    emb = FullEmbedding(
                        d_model    = d_model, 
                        vocab_size = vocab_size, 
                        debug      = DEBUG, 
                        dropout    = emb_drop
                        ).to(device)

    # initialize encoder instance and load it to device
    enc = Encoder(
                  d_model       = d_model, 
                  d_ffn         = d_ffn_enc, 
                  num_of_heads  = enc_heads, 
                  num_of_layers = enc_blocks, 
                  att_drop      = enc_att_drop,
                  ffn_drop      = enc_ffn_drop,
                  debug         = DEBUG
                ).to(device)
    
    # initalize decoder instance and load it to device
    dec = Decoder(
                  d_model        = d_model, 
                  d_ffn          = d_ffn_dec, 
                  dec_heads      = dec_heads,
                  cross_heads    = cross_heads, 
                  num_of_layers  = dec_blocks, 
                  debug          = DEBUG,
                  mask_att_drop  = mask_att_drop,
                  cross_att_drop = cross_att_drop,
                  ffn_drop       = dec_ffn_drop
                ).to(device)
    
    # output projection to convert decoder output to vocabulary
    out_proj = nn.Linear(d_model, vocab_size).to(device)

    ''' initialize loss function instance; Cross Entropy Loss '''
    # label smoothing adds uncertainty
    loss_func = nn.CrossEntropyLoss(ignore_index = -100, label_smoothing = 0.1)    # pad_token_ID = -100   

    ''' initialize optimizer '''
    # collect list of parameters
    all_params = (
                  list(emb.parameters()) +
                  list(enc.parameters()) +
                  list(dec.parameters()) +
                  list(out_proj.parameters())
                )
    # initialize optimizer and its scheduler
    optimizer = torch.optim.AdamW(all_params, lr = learning_rate, weight_decay = weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode = sch_mode, factor = sch_factor, patience = sch_patience)
    
    return emb, enc, dec, out_proj, loss_func, all_params, optimizer, scheduler


def load_checkpoint(device, checkpoint_path, emb, enc, dec, out_proj, optimizer, scheduler):
    '''
    Load previously trained and saved model if it exists, and initialize parameters.

    Return start_epoch of that model if it exists, else return 0.
    '''
    ''' choose starting point '''
    if os.path.exists(checkpoint_path):
        # if previously trained models exist, get those states as starting point
        # with map_location, tensors are loaded to the current "device"
        # map_location is not important here since we are using mps for everything but its a good 
        print(f"loading model {checkpoint_path}\n")

        checkpoint = torch.load(checkpoint_path, map_location = device)
        emb.load_state_dict(checkpoint["emb"])
        enc.load_state_dict(checkpoint["enc"])
        dec.load_state_dict(checkpoint["dec"])
        out_proj.load_state_dict(checkpoint["out_proj"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        train_start_index = checkpoint["train_end_index"]
        
    else:
        print(f"{checkpoint_path} model not found, starting fresh!\n")
        os.makedirs("./checkpoints/", exist_ok = True)
        start_epoch = 0         
        train_start_index = 0        
    
    print(f"starting at epoch = {start_epoch}\n")
        
    return start_epoch, train_start_index


# forward pass – reusable by train, val, test
def forward_pass(batch, emb, enc, dec, out_proj, loss_func, vocab_size, device, DEBUG):
    '''
    batch       : current batch of data 
    emb         : embedding instance to convert tokens to embeddings
    enc         : encoder class object
    dec         : decoder class object
    out_proj    : output projection for decoder output to match vocab_size
    device      : device used for computation
    DEBUG       : show statistics
    '''
    ''' Encoder Input Parameters '''
    enc_input = batch["input_ids"].to(device)              # (batch_size, enc_seq_len)

    enc_pad   = batch["attention_mask"].bool().to(device)  # (batch_size, enc_seq_len)
    # (batch_size, 1, enc_seq_len) & (batch_size, enc_seq_len, 1) = (batch_size, enc_seq_len, seq_len)
    enc_mask = (enc_pad.unsqueeze(1) & enc_pad.unsqueeze(2)).unsqueeze(1)   # (batch_size, 1, enc_seq_len, enc_seq_len)
    
    ''' Decoder Input Parameters '''
    # BOS already exists in the lables
    # dropping last token in each sequence
    dec_input = batch["labels"][:, :-1].to(device)  # (batch_size, dec_seq_len = 499)

    # ensure pad tokens are back from -100 to 1 (without it error out of bound occurs)
    dec_input = dec_input.masked_fill(dec_input == -100, 1) # tokenizer's default pad token ID

    dec_pad = batch["decoder_attention_mask"][:, :-1].bool().to(device)  # (batch_size, dec_seq_len = 499)
    _, dec_seq_len = dec_pad.shape

    causal_mask = torch.tril(torch.ones(dec_seq_len, dec_seq_len, device = device)).bool() # (dec_seq_len, dec_seq_len)
    # (batch_size, 1, dec_seq_len) & (batch_size, dec_seq_len, 1) = (batch_size, dec_seq_len, dec_seq_len)
    # PyTorch aligns dimensions from the right, and expands size 1 as needed
    dec_mask = (dec_pad.unsqueeze(1) & dec_pad.unsqueeze(2)).unsqueeze(1) & causal_mask # (batch_size, 1, dec_seq_len, dec_seq_len)

    cross_mask = enc_pad.unsqueeze(1).unsqueeze(1)    # (batch_size, 1, 1, enc_seq_len)

    # labels will be used to compare the logits
    # we take 499 of the tokens out of 500 to match the logit
    labels = batch["labels"][:, 1:].to(device)                 #  (batch_size, dec_seq_len = 499)
    
    # embeddings
    source_emb = emb(token_ids = enc_input)
    target_emb = emb(token_ids = dec_input)

    # encoder output
    enc_out = enc(x = source_emb, att_mask = enc_mask)
    del source_emb                   # free memory

    # decoder output : (batch_size, dec_seq_len, d_model)
    dec_out = dec(
                  x        = target_emb, 
                  enc_out  = enc_out, 
                  dec_mask = dec_mask, 
                  enc_mask = cross_mask
                )
    
    del target_emb, enc_out          # free memory
   
    # final model output
    logits = out_proj(dec_out)       # (batch_size, dec_seq_len, vocab_size)
    del dec_out                      # free memory

    # average cross entropy loss for the current batch, ignoring positions at pad tokens = -100
    # .view() reshapes and returns the same tensor without changing underlying memory            
    # loss = loss_func(
    #                 logits.view(-1, vocab_size),  # (batch_size * 499, vocab_size)
    #                 labels.view(-1)               # (batch_size * 499)
    #                 )    
    loss = loss_func(
                    logits.reshape(-1, vocab_size),  # (batch_size * 499, vocab_size)
                    labels.reshape(-1)               # (batch_size * 499)
                    )    
 
    
    del logits

    return loss   # as tensor


def train_block(batch_index, data_len, batch, emb, enc, dec, out_proj, loss_func, optimizer, all_params, 
                vocab_size, device,  grad_acc, DEBUG):
    '''
    Train a given batch: forward pass + backward pass + optimizer step
    '''
    # perform forward pass and compute batch loss
    batch_loss = forward_pass(batch, emb, enc, dec, out_proj, loss_func, vocab_size, device, DEBUG)

    scaled_loss = batch_loss / grad_acc  # scale batch_loss caused by accumulated gradients

    # compute gradients based on this batch loss
    scaled_loss.backward()

    # end of current accumulation cycle or lend of available data, update gradients
    if (batch_index % grad_acc == 0) or (batch_index == data_len):  
        # clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(all_params, max_norm = 1.0) # maximum gradient norm = 1

        # update optmizer
        optimizer.step()
        
        # clear previous gradients after updating parameters
        optimizer.zero_grad()
    
    return batch_loss.item() # return only magnitude of the tensor for epoch loss computation


def val_block(batch, emb, enc, dec, out_proj, loss_func, vocab_size, device, DEBUG):
    '''
    Validate trained model using current validation data batch, compute validation batch loss.
    '''
    with torch.no_grad():
        # compute validation loss
        val_batch_loss = forward_pass(batch, emb, enc, dec, out_proj, loss_func, vocab_size, device, DEBUG)
    
    return val_batch_loss.item()  # only magnitude
    
