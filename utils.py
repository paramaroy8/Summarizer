'''
Utility functions such as logging functions.
'''

import torch
import sys
from datetime import datetime 

# calculate and print statistics of various torch tensors 

def debug_tensor(name, x, is_probs=False):
    '''
    name = name of the tensor we are handling

    x = tensor

    is_probs = if the tensor comes after passing through softmax, that is, if x is tensor of probabilities
                only then, we can calculate entropy
    '''
    shape        = x.shape                     # dimensions
    min_val      = x.min().item()              # minimum value
    max_val      = x.max().item()              # maximum value
    mean_val     = x.mean().item()             # mean
    standard_dev = x.std().item()              # standard deviation
    has_nan      = torch.isnan(x).sum().item() # check for NaN value
    has_inf      = torch.isinf(x).sum().item() # check for inf value [decoder will have inf values in mask positions]

    print(
          f"\n======= {name} =======\n"
          # f"\ndebug input = \n{x}\n" 
          f"\nshape = {shape}"
          f"\nminimum value = {min_val:.4f}, \nmaximum value = {max_val:.4f}"
          f"\nmean = {mean_val:.4f}, \nstandard deviation = {standard_dev:.4f}" 
          f"\nNaN count = {has_nan}, \nInf count = {has_inf}"
        )
    
    # if x is probability values, we can also calculate average entropy actoss heads or tokens
    if is_probs:
        # for probability values, row wise sum should be 1
        print("\nRow wise sum =\n", x.sum(dim = -1))
        
        av_entropy = -(x * torch.log(x + 1e-9)).sum(dim = -1).mean().item()
        print(f"\naverage entropy = {av_entropy:.4f}")

        # row wise maximum probability
        # max_prob = x.max(dim = -1) 
    

# class to save console output in a file

class Logger:
    '''
    Save all outputs in a file as well as console for analysis.
    '''
    def __init__(self, filepath, write_choice = "a"):
        self.console = sys.__stdout__      # console output
        # open file for writing, 
        # write_choice can be "w" for overwrite or "a" for keeping everything
        self.file    = open(filepath, write_choice) 
        # use timestamp before each run
        self.file.write(f"\n\n\ndate and time = {datetime.now()}\n\n")
    
    def write(self, message):
        self.console.write(message)        # print to terminal
        self.file.write(message)           # writes to file
    
    def flush(self):
        '''
        recover printed outputs after a crash
        '''
        self.console.flush()               # flush to console
        if not self.file.closed:
            self.file.flush()              # flush to file as long as the file remains open

    def close(self):
        sys.stdout = self.console
        if not self.file.closed:
            self.file.close()

