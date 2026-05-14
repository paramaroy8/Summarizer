'''
multi-head attention class 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import debug_tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_of_heads, debug, dropout = 0.1):
        # import all parent class functions
        super().__init__()

        # ensure d_model is divisible by num_of_heads
        assert d_model % num_of_heads == 0, "d_model not divisible by num_of_heads"

        self.d_model      = d_model
        self.num_of_heads = num_of_heads
        self.d_k          = d_model // num_of_heads
        self.dropout      = nn.Dropout(p = dropout)
        self.DEBUG        = debug

        # initialize weights for each attention vector for all heads
        # w(input) = input @ w.T + b ; both w and b are learnable parameters
        # input @ w.T : (seq_len, d_model) ; b : (d_model,)
        self.w_q     = nn.Linear(d_model, d_model)
        self.w_k     = nn.Linear(d_model, d_model)
        self.w_v     = nn.Linear(d_model, d_model)
        self.w_combo = nn.Linear(d_model, d_model) # output projection for feature combination

    
    def split_heads(self, x):
        '''
        x          : (batch_size, seq_len, d_model)
        batch_size : how many chunks from different papers get processed at the same time
        seq_len    : different for "abstract" and "article"
        output     : (batch_size, num_of_heads, seq_len, d_k)
        '''
        batch_size, seq_len, _ = x.shape

        x = x.reshape(batch_size, seq_len, self.num_of_heads, self.d_k) # (batch_size, seq_len, num_of_heads, d_k)
        
        output = x.transpose(2, 1) # (batch_size, num_of_heads, seq_len, d_k)

        return output
    
    
    def forward(self, x, mask = None, x_kv = None):
        '''
        x = embedded tokens or output of previous layer : (batch_size, seq_len, d_model)

        mask  = None for encoder : (batch_size, 1, seq_len, seq_len)

        x_kv  = source for K and V vectors during cross attention for decoder block

        dropout = default value is 0.1
        '''

        batch_size, seq_len, _ = x.shape # encoder and decoder have different seq_len
        
        kv_source = x_kv if x_kv is not None else x     # K and V depends on x for encoder and encoder output for decoder

        Q = self.split_heads(self.w_q(x))
        K = self.split_heads(self.w_k(kv_source))
        V = self.split_heads(self.w_v(kv_source))

        attention_scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5) # (batch_size, num_of_heads, seq_len, seq_len)

        if mask is not None:
            # all padded elements will be replaced with -inf so that softmax would turn those elements to 0
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        

        # row-wise softmax -> how each token relates to other tokens
        attention_weights = F.softmax(attention_scores, dim = -1) 

        # guard against NaN values; convert NaN to 0.0
        attention_weights = torch.nan_to_num(attention_weights, nan = 0.0)

        # add dropout
        attention_weights = self.dropout(attention_weights)

        # attention output
        M = attention_weights @ V

        # reshape output per token
        # transpose() creates non-contiguous memory; reshape will fail without contiguous memory
        M = M.transpose(2, 1).contiguous()  # (batch_size, seq_len, num_of_heads, d_k)

        # join each head output
        M = M.reshape(batch_size, seq_len, self.d_model) # (batch_size, seq_len, d_model)

        # combine the features 
        output = self.w_combo(M)  # (batch_size, seq_len, d_model)

        #---------LOGGING----------
        if self.DEBUG:
            debug_tensor("attention scores", attention_scores)
            debug_tensor("attention weights", attention_weights, True)
            debug_tensor("attention output", output)

        return output


if __name__ == "__main__":
    DEBUG = True 

    torch.manual_seed(0)

    # input 
    X = torch.tensor([
        [[1., 0., 1., 0.],
        [0., 2., 0., 2.],
        [1., 1., 1., 1.]]
    ])  # (batch_size = 1, seq_len = 3, d_model = 4)

    batch_size, seq_len, d_model = X.shape

    num_of_heads = 2
    dropout = 0.2

    # create attention object
    m = MultiHeadAttention(d_model = d_model, num_of_heads = num_of_heads, debug = DEBUG, dropout = dropout)

    m_output = m(X) # attention output; forward function is called internally

    # view all learnable parameters
    print("\nLearnable Attention Parameters =\n")

    for name, parameter in m.named_parameters():
        print(f"\nname = {name}, \n\n{parameter}\n")
    
