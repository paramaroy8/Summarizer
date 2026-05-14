'''
Run this file to train the model.
'''
import os
import sys
import torch 

from train_and_validation import train_val
from utils import Logger 
from visualize import read_log, plots 

# plot curves of output
def plot(plot_name, data_path, run_id):
    # get epochs, corresponding epoch losses and perplexities
    epochs, losses, plexes = read_log(data_path, run_id)

    print(
        f"\nepochs = {epochs}\n"
        f"\nlosses = {losses}\n"
        f"\nperplexities = {plexes}\n"
    )

    # plot epoch loss 
    plots(
        folder_name = "visualization",
        epochs      = epochs,
        losses      = losses,
        title       = f"{plot_name} : Epoch Losses Per Epoch",
        x_label     = "Epochs",
        y_label     = "Losses"
    )

    # plot perplexities
    plots(
        folder_name = "visualization",
        epochs      = epochs,
        losses      = plexes,
        title       = f"{plot_name} : Perplexities Per Epoch",
        x_label     = "Epochs",
        y_label     = "Perplexities"
    )


if __name__ == "__main__":
    # set up device
    device = torch.device(
                            "mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available()
                            else "cpu"
                        )
    
    print(f"\nTraining on Device = {device}\n\n")
    os.makedirs("./console_logs/", exist_ok = True)
    # folder to store parameter output and plots
    os.makedirs("./visualization/", exist_ok = True)

    '''------------------------------- Parameters --------------------------------------------------'''

    # train_samples     = 500       # how many samples of train data 
    train_chunk_size  = 1000        # num of training samples to train with at a time


    val_samples       = 1000
    num_main          = 4        # instead of multiple epochs, call the the train_main multiple times

    d_model           = 512

    # divisible by 128, appropriate for calculations on GPU
    d_ffn_enc         = 1408     # encoder ffn dim
    d_ffn_dec         = 1408     # decoder ffn dim

    vocab_size        = 50265

    enc_heads         = 8        # num of heads in encoder self attention
    dec_heads         = 8        # num of heads in decoder self attention
    cross_heads       = 8        # num of heads in decoder cross attention

    batch_size        = 5        # how many samples to process at a time per batch
    grad_acc          = 1        # accumulate gradients for this many batches; simulating batch_size *= grad_acc

    enc_blocks        = 5        # num of encoder blocks
    dec_blocks        = 10       # num of decoder blocks

    emb_drop          = 0.01     # embedding dropout

    enc_att_drop      = 0.01     # encoder attention dropout
    enc_ffn_drop      = 0.01     # encoder FFN dropout

    mask_att_drop     = 0.01     # decoder masked attention dropout
    cross_att_drop    = 0.01     # decoder cross attention dropout
    dec_ffn_drop      = 0.01     # decoder FFN dropout

    epochs            = 1        # nums of times to train the model

    checkpoint_path   = "./checkpoints/latest_model_0.pt" # last trained model

    learning_rate     = 3e-4     # optimizer learning rate starting value for gradient descent
    weight_decay      = 1e-2     # optimizer regularization to prevent large gradient

    sch_mode          = "min"    # scheduler mode = "min" for monitoring loss (lower is better)
    sch_factor        = 0.5      # scheduler updates learning rate by scheduler factor
    sch_patience      = 2        # scheduler epoch patience for loss improvement before reducing learning rate
    
    res_train_path    = "./visualization/train_out_0.csv"      # path for loss logging for visualization
    run_id            = "0"                                    # run ID in the parameter output; keep it in string format

    res_val_path      = "./visualization/val_out_0.csv"        # path for loss logging for visualization

    DEBUG             = False

    TRAIN             = True     # True to activate training
    
    train_curve       = True     # True to plot training output
    val_curve         = True     # True to plot validation output

    # tokenized train data path
    train_path        = "./tokenized-dataset/train"
    val_path          = "./tokenized-dataset/validation"
    
    write_choice      = "a"    # "w" for overwrite and "a" for keeping everything in log file

    console_path      = "./console_logs/console.txt"    # save console trainig output

    '''---------------------------------------------------------------------------------'''

   
    sys.stdout = Logger(filepath = console_path, write_choice = write_choice)

    # print some of the parampeters
    print(
          f"num of train chunk size = {train_chunk_size}\n"
          f"num of val samples      = {val_samples}\n"
          f"d_model                 = {d_model}\n"
          f"batch_size              = {batch_size}\n"
          f"gradient accumulation   = {grad_acc}\n"
          f"encoder d_ffn           = {d_ffn_enc}\n"
          f"decoder d_ffn           = {d_ffn_dec}\n"
          f"enc heads               = {enc_heads}\n"
          f"dec heads               = {dec_heads}\n"
          f"cross heads             = {cross_heads}\n"
          f"calling train function  = {num_main} time"
          )

    if TRAIN:
        for i in range(num_main):
            # num of times model is being trained -> simulating epochs
            print(f"\n~~ begin training = {i + 1} ~~\n")

            train_val(
            device, train_path, val_path, d_model, d_ffn_enc, d_ffn_dec, vocab_size, DEBUG, batch_size, val_samples,
            enc_blocks, dec_blocks, emb_drop, enc_att_drop, enc_ffn_drop, mask_att_drop, cross_att_drop, dec_ffn_drop,
            enc_heads, dec_heads, cross_heads, checkpoint_path, 
            learning_rate, weight_decay, epochs, sch_mode, sch_factor, sch_patience, grad_acc, 
            res_train_path, run_id, res_val_path, console_path, train_chunk_size
            )
        
        if train_curve:
            plot(plot_name = "train", data_path = res_train_path, run_id = run_id)
        if val_curve:
            plot(plot_name = "validation", data_path = res_val_path, run_id = run_id)
        
    
    # finally, close console and file
    sys.stdout.close()
