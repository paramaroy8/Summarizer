'''
Single Encoder Block
'''

import torch 
import torch.nn as nn 

from attention_block import MultiHeadAttention
from normalize_layer import LayerNorm
from feed_forward_network import FeedForwardNetwork
from utils import debug_tensor

class SingleEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_of_heads, debug, att_drop = 0.01, ffn_drop = 0.01):
        super().__init__()

        # initialize parameters
        self.d_model          = d_model
        self.d_ffn            = d_ffn
        self.num_of_heads     = num_of_heads
        self.DEBUG            = debug
        self.att_drop         = att_drop
        self.ffn_drop         = ffn_drop

        # create instances of each sub block
        self.input_norm = LayerNorm(
            d_model = d_model, 
            debug = self.DEBUG
            ) # encoder block input
        
        self.att = MultiHeadAttention(
            d_model = self.d_model, 
            num_of_heads = self.num_of_heads, 
            debug = self.DEBUG, 
            dropout = self.att_drop
            ) # attention

        self.att_norm = LayerNorm(
            d_model = self.d_model, 
            debug = self.DEBUG
            ) # layer norm for attention
        
        self.ffn = FeedForwardNetwork(
            d_model = self.d_model, 
            d_ffn = self.d_ffn, 
            debug = self.DEBUG, 
            dropout = self.ffn_drop
            ) # feedforward network
        
        self.ffn_norm = LayerNorm(
            d_model = self.d_model, 
            debug = self.DEBUG
            )


    def forward(self, data, att_mask):
        '''
        initially, data = embedding
        '''  
        # normalize input to prevent layer 1 instability -> Pre LN
        norm_out1 = self.input_norm(data)
        # attention + layer norm
        att_out = self.att(x = norm_out1, mask = att_mask)
        norm_out2 = self.att_norm(att_out + data)

        # feedforward network + layer norm
        ffn_out = self.ffn(norm_out2)
        norm_out3 = self.ffn_norm(ffn_out + norm_out2)

        if self.DEBUG:
            debug_tensor("input norm", norm_out1)
            debug_tensor("attention output", att_out)
            debug_tensor("normalized attention", norm_out2)
            debug_tensor("ffn output", ffn_out)
            debug_tensor("normalized ffn", norm_out3)

        return norm_out3 # output of single encoder block


if __name__ == "__main__":

    batch_size = 2
    d_model = 1024
    d_ffn = 2048 
    num_of_heads = 2
    debug = True
    att_drop = 0.1 # attention dropout
    ffn_drop = 0.2 # encoder dropout
    seq_len = 20

    X = torch.randn(batch_size, seq_len, d_model)  # (batch_size, enc_seq_len, d_model)

    # PyTorch will broadcass across num of heads automatically
    att_mask = torch.ones(batch_size, 1, seq_len, seq_len).bool()  # (batch_size = 1, num_of_heads = 1, seq_len = 2, seq_len = 2)

    enc = SingleEncoderBlock(d_model, d_ffn, num_of_heads, debug, att_drop = att_drop, ffn_drop = ffn_drop)
    output = enc(X, att_mask)

    print(f"output shape = {output.shape}")
