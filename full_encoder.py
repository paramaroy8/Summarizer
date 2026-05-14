'''
Encoder class containing mutiple encoder blocks.
'''

import torch
import torch.nn as nn

from single_encoder_block import SingleEncoderBlock
from utils import debug_tensor


class Encoder(nn.Module):
    def __init__(self, d_model, d_ffn, num_of_heads, num_of_layers, att_drop, ffn_drop, debug):
        super().__init__()

        # initialize parameters
        self.d_model       = d_model
        self.d_ffn         = d_ffn
        self.num_of_heads  = num_of_heads
        self.num_of_layers = num_of_layers
        self.att_drop      = att_drop       # attention dropout
        self.ffn_drop      = ffn_drop       # ffn dropout
        self.DEBUG         = debug

        # stack encoder block to be executed one after the other
        self.encoder_layers = nn.ModuleList([
            SingleEncoderBlock(d_model = d_model, 
            d_ffn = d_ffn, 
            num_of_heads = num_of_heads, 
            debug = self.DEBUG, 
            att_drop = self.att_drop,
            ffn_drop = self.ffn_drop
            )
            for _ in range(self.num_of_layers)
        ])


    def forward(self, x, att_mask):
        '''
        x        : embedding
        att_mask : attention mask for encoder
        '''

        # each layer is one encoder block
        for index, layer in enumerate(self.encoder_layers, 1): # 1 is the starting layer index
            if self.DEBUG:
                print(f"\n\n~~~~~~~~~~~~~~~~  layer {index}  ~~~~~~~~~~~~~~~~~~")
            
            x = layer(x, att_mask) # output of prev layer become input of next layer
        
        if self.DEBUG:
            debug_tensor("full encoder output", x)
        
        return x # output of the encoder


if __name__ == "__main__":
    batch_size = 2
    d_model = 1024
    d_ffn = 2048 
    num_of_heads = 2
    debug = True
    att_drop = 0.1 # attention dropout
    ffn_drop = 0.2 # encoder dropout
    seq_len = 20
    num_of_layers = 2

    X = torch.randn(batch_size, seq_len, d_model)  # (batch_size, enc_seq_len, d_model)

    # PyTorch will broadcass across num of heads automatically
    att_mask = torch.ones(batch_size, 1, seq_len, seq_len).bool()  # (batch_size, num_of_heads, seq_len, seq_len)

    enc = Encoder(
        d_model = d_model, 
        d_ffn = d_ffn, 
        num_of_heads = num_of_heads, 
        num_of_layers = num_of_layers, 
        att_drop = att_drop, 
        ffn_drop = ffn_drop, 
        debug = debug
        )

    enc_out = enc(X, att_mask)

    print(f"\nenc out shape = {enc_out.shape}")

    print("\n\nEncoder Complete!\n\n")
