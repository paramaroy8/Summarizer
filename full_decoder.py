'''
Decoder Class based on multiple decoder blocks.
'''

import torch
import torch.nn as nn

from single_decoder_block import SingleDecoderBlock
from utils import debug_tensor


class Decoder(nn.Module):
    def __init__(self, d_model, d_ffn, dec_heads, cross_heads, num_of_layers, debug, mask_att_drop = 0.1, cross_att_drop = 0.1, ffn_drop = 0.1):
        super().__init__()

        # parameters
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dec_heads = dec_heads
        self.cross_heads = cross_heads
        self.num_of_layers = num_of_layers
        self.DEBUG = debug
        self.mask_att_drop = mask_att_drop
        self.cross_att_drop = cross_att_drop
        self.ffn_drop = ffn_drop 

        # create module list for decoder blocks instances
        self.decoder_layers = nn.ModuleList([
            SingleDecoderBlock(
                d_model = self.d_model, 
                d_ffn = self.d_ffn, 
                dec_heads = self.dec_heads,
                cross_heads = self.cross_heads, 
                debug = self.DEBUG,
                mask_att_drop = self.mask_att_drop,    # masked attention dropout
                cross_att_drop = self.cross_att_drop,  # cross attention dropout
                ffn_drop = self.ffn_drop               # ffn dropout
                )
            for _ in range(self.num_of_layers)
        ])


    def forward(self, x, enc_out, dec_mask, enc_mask):
        '''
        x          : target embedding
        dec_mask   : mask for decoder
        enc_out    : source for K and V in cross attention
        enc_mask   : encoder mask for pad tokens
        '''

        for index, layer in enumerate(self.decoder_layers, 1): # 1 is to ensure index counting starts from 1
            if self.DEBUG:
                print(f"\n\n~~~~~~~~~~~~~~~~  layer {index}  ~~~~~~~~~~~~~~~~~~")
            
            x = layer(target = x, enc_out = enc_out, dec_mask = dec_mask, enc_mask = enc_mask)
        
        if self.DEBUG:
            debug_tensor("full decoder output", x)
        
        return x


if __name__ == "__main__":
    # parameters
    # hyperparameters
    batch_size     = 2
    dec_seq_len    = 10    # seq len for target input
    enc_seq_len    = 20    # seq len for source input for encoder
    d_model        = 1024
    d_ffn          = 2816  # divisible by 128
    dec_heads      = 8
    cross_heads    = 8
    num_of_layers  = 3
    mask_att_drop  = 0.01  # masked attention dropout
    cross_att_drop = 0.02  # cross attention dropout
    ffn_drop       = 0.001 # ffn dropout

    target = torch.randn(batch_size, dec_seq_len, d_model) # (batch_size, dec_seq_len, d_model)
    enc_out = torch.randn(batch_size, enc_seq_len, d_model)  # (batch_size, enc_seq_len, d_model)

    '''
    torch.ones() give you float32 data type be default.
    So, we want to convert it into bool so that we only have 0 or 1 as mask element
    '''
    dec_mask = torch.tril(torch.ones(dec_seq_len, dec_seq_len)).bool()    # (dec_seq_len, dec_seq_len)
    # correct dimensions
    dec_mask = dec_mask.unsqueeze(0).unsqueeze(1)                         #  (1, 1, dec_seq_len, dec_seq_len)

    enc_mask = torch.ones(batch_size, 1, 1, enc_seq_len).bool()           # (batch_size, 1, 1, enc_seq_len)

    # decoder block instance
    dec = Decoder(
        d_model = d_model, 
        d_ffn = d_ffn, 
        dec_heads = dec_heads,
        cross_heads = cross_heads,
        num_of_layers = num_of_layers, 
        debug = False,
        mask_att_drop = mask_att_drop,
        cross_att_drop = cross_att_drop,
        ffn_drop = ffn_drop
        )

    output = dec(x = target, enc_out = enc_out, dec_mask = dec_mask, enc_mask = enc_mask)

    print(f"\noutput shape = {output.shape}\n") # (batch_size, dec_seq_len, d_model)
