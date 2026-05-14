'''
LayerNorm : Normalization + Scale + Shift
'''
import torch 
import torch.nn as nn 
from utils import debug_tensor


class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-6, debug = True):
        super().__init__()

        self.DEBUG = debug
        self.epsilon = epsilon

        # learnable scaling and shifting parameters; shape = (d_model, )
        self.gamma = nn.Parameter(torch.ones(d_model)) # scaling parameter
        self.beta = nn.Parameter(torch.zeros(d_model)) # shifting parameter

    def forward(self, input):
        '''
        input : (batch_size, seq_len, d_model)
        '''

        # token-wise mean and variance
        # keepdim keeps reduced dimension as 1 instead of removing it completely
        # population variance used
        mean = input.mean(dim = -1, keepdim = True)                   # (batch_size, seq_len, 1)
        var = input.var(dim = -1, keepdim = True, correction = 0)     # (batch_size, seq_len, 1)

        # normalize; mean and variance is broadcasted across d_model
        input_norm = (input - mean) / torch.sqrt(var + self.epsilon)  # (batch_size, seq_len, d_model)

        # scale and shift; torch left-pads the gamma and beta => (d_model, ) into (1, 1, d_model) 
        # then, broadcasts across (batch_size, seq_len)
        output = (input_norm * self.gamma) + self.beta                # (batch_size, seq_len, d_model)

        if self.DEBUG:
            debug_tensor("layer norm input", input)
            debug_tensor("layer norm output", output)
        
        return output 


if __name__ == "__main__":
    debug = True
    
    X = torch.tensor([
        [[1, 0, 1, 0],
        [0, 2, 0, 2],
        [1, 1, 1, 1]]
    ]).float() # (1, 3, 4)

    batch_size, seq_len, d_model = X.shape
    
    # initialize LayerNorm class
    layer_norm = LayerNorm(d_model = d_model, debug = debug)
    # compute layer norm output
    layer_norm_output = layer_norm(X)

    # in the example, the last token has all same feature values, meaning no variance
    print("\n\n", layer_norm_output.std(dim=-1, correction=0))  # std per token

    print("\nlayer normalization done!\n")


