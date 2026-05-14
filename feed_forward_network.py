'''
Feedforward Network: SwiGLU
'''

import torch
import torch.nn as nn
from utils import debug_tensor


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ffn, debug, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ffn   = d_ffn
        self.DEBUG   = debug
        self.dropout = nn.Dropout(dropout)

        # initialize weights
        # batch_size and seq_len do not matter here; PyTorch handles these automatically
        self.W1 = nn.Linear(in_features = d_model, out_features = d_ffn, bias = False)   # d_model -> d_ffn
        self.W2 = nn.Linear(in_features = d_model, out_features = d_ffn, bias = False)   # d_model -> d_ffn
        self.W3 = nn.Linear(in_features = d_ffn, out_features = d_model, bias = False)   # d_ffn   -> d_model

    def forward(self, x):
        '''
        x = ffn input = (batch_size, seq_len, d_model)
        '''
        # SwiGLU Feedforward: FFN(x) = W3( SiLU(W1(x)) * W2(x) )

        # linear layers can be called as general function
        w1_out = self.W1(x)
        path1  = w1_out * torch.sigmoid(w1_out)     # activation branch
        path2  = self.W2(x)                         # gating branch (modulation signal)
        
        # for elementwise multiplication, use torch.mul(x, y) or simply, x * y
        path_output = path1 * path2                 # gating

        # add dropout before projecting back
        path_output = self.dropout(path_output)

        ffn_output = self.W3(path_output)           # project feature dimension back to d_model

        if self.DEBUG:
            debug_tensor("feedforward network output", ffn_output)

        return ffn_output 

if __name__ == "__main__":
    
    X = torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.5, 0.6, 0.7]
        ]
    ]) # batch_size = 1, seq_len = 2, d_model = 4

    _, _, d_model = X.shape 
    
    d_ffn = 128
    dropout = 0.001

    ffn = FeedForwardNetwork(d_model = d_model, d_ffn = d_ffn, debug = True, dropout = dropout)
    output = ffn(X)

    print(f"\noutput shape = {output.shape}")  # expect: (1, 2, 4)

    print("\nfeed forward complete!\n")
