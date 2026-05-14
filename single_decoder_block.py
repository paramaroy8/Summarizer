'''
Single Decoder Block
'''

import torch
import torch.nn as nn

from attention_block import MultiHeadAttention
from normalize_layer import LayerNorm
from feed_forward_network import FeedForwardNetwork
from utils import debug_tensor


class SingleDecoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, dec_heads, cross_heads, debug, mask_att_drop = 0.1, cross_att_drop = 0.1, ffn_drop = 0.1):
        
        super().__init__()

        # initialize parameters
        self.d_model        = d_model
        self.d_ffn          = d_ffn
        self.dec_heads      = dec_heads
        self.cross_heads    = cross_heads 
        self.DEBUG          = debug
        self.mask_att_drop  = mask_att_drop
        self.cross_att_drop = cross_att_drop
        self.ffn_drop       = ffn_drop 

        # normalize target input
        self.input_norm = LayerNorm(
            d_model = self.d_model, 
            debug = self.DEBUG
            )
        
        # masked attention block : Q, K, V -> same source
        self.masked_att = MultiHeadAttention(
            d_model = self.d_model, 
            num_of_heads = self.dec_heads, 
            debug = self.DEBUG,
            dropout = self.mask_att_drop
            )
       
        self.masked_att_norm = LayerNorm(
            d_model = self.d_model, 
            debug = self.DEBUG
            )
        
        # cross attention block : Q -> target ; K, V -> Encoder Output
        self.cross_att = MultiHeadAttention(
            d_model = self.d_model, 
            num_of_heads = self.cross_heads, 
            debug = self.DEBUG,
            dropout = self.cross_att_drop 
            )
        
        self.cross_att_norm = LayerNorm(
            d_model = self.d_model, 
            debug = self.DEBUG
            )
        
        # feedforward network
        self.ffn = FeedForwardNetwork(
            d_model = self.d_model, 
            d_ffn = self.d_ffn, 
            debug = self.DEBUG,
            dropout = self.ffn_drop
            )
        self.ffn_norm = LayerNorm(
            d_model = self.d_model, 
            debug = self.DEBUG
            )
    
    
    def forward(self, target, enc_out, dec_mask, enc_mask):
        '''
        target      : target input embedding that the model wants to learn
        enc_out     : encoder output 
        dec_mask    : (causal mask + padded mask) used for masked self-attention
        enc_mask    : encoder mask for pad tokens
        '''

        # normalize target
        target_norm = self.input_norm(target) # (batch_size, dec_seq_len, d_model)

        '''masked attention'''
        masked_out  = self.masked_att(x = target_norm, mask = dec_mask, x_kv = None) # (batch_size, dec_seq_len, d_model)
        masked_norm = self.masked_att_norm(masked_out + target) # (batch_size, dec_seq_len, d_model)

        '''cross attention'''
        cross_out  = self.cross_att(x = masked_norm, mask = enc_mask, x_kv = enc_out) # (batch_size, dec_seq_len, d_model)
        cross_norm = self.cross_att_norm(cross_out + masked_norm) # (batch_size, dec_seq_len, d_model)

        '''feedforward network'''
        ffn_out  = self.ffn(cross_norm) # (batch_size, dec_seq_len, d_model)
        ffn_norm = self.ffn_norm(ffn_out + cross_norm) # (batch_size, dec_seq_len, d_model)

        if self.DEBUG:
            debug_tensor("normalized target embedding", target_norm)
            debug_tensor("masked attention output", masked_out)
            debug_tensor("normalized masked attention", masked_norm)
            debug_tensor("cross attention output", cross_out)
            debug_tensor("normalized cross attention", cross_norm)
            debug_tensor("ffn output", ffn_out)
            debug_tensor("normalized ffn", ffn_norm)


        return ffn_norm  # output : (batch_size, dec_seq_len, d_model)


if __name__ == "__main__":
    # hyperparameters
    batch_size     = 2
    dec_seq_len    = 10    # seq len for target input
    enc_seq_len    = 20    # seq len for source input for encoder
    d_model        = 1024
    d_ffn          = 2048
    dec_heads      = 8     # num of heads for decoder self attention 
    cross_heads    = 8     # num of heads for decoder cross attention 
    mask_att_drop  = 0.1
    cross_att_drop = 0.01
    ffn_drop       = 0.2
    DEBUG          = True

    target = torch.randn(batch_size, dec_seq_len, d_model)   # (batch_size, dec_seq_len, d_model)
    enc_out = torch.randn(batch_size, enc_seq_len, d_model)  # (batch_size, enc_seq_len, d_model)

    '''
    torch.ones() give you float32 data type be default.
    So, we want to convert it into bool so that we only have 0 or 1 as mask element
    '''
    dec_mask = torch.tril(torch.ones(dec_seq_len, dec_seq_len)).bool()    # (dec_seq_len, dec_seq_len)
    # correct dimensions
    dec_mask = dec_mask.unsqueeze(0).unsqueeze(1)                         #  (1, 1, dec_seq_len, dec_seq_len)

    # pytorch will broadcast it
    enc_mask = torch.ones(batch_size, 1, 1, enc_seq_len).bool() # (batch_size, 1 (num of heads), 1(dec_seq_len), enc_seq_len)

    # decoder block instance
    dec_block = SingleDecoderBlock(
                                    d_model        = d_model, 
                                    d_ffn          = d_ffn, 
                                    dec_heads      = dec_heads,
                                    cross_heads    = cross_heads,
                                    debug          = DEBUG, 
                                    mask_att_drop  = mask_att_drop, 
                                    cross_att_drop = cross_att_drop, 
                                    ffn_drop       = ffn_drop
                                )

    output = dec_block(target = target, enc_out = enc_out, dec_mask = dec_mask, enc_mask = enc_mask)

    print(f"\noutput shape = {output.shape}\n")


