'''
Visualization
'''
import os
import matplotlib.pyplot as plt
import csv


def read_log(data_path, run_id):
    '''
    Read log file and return each data separately.

    data_path : path of the log
    run_id    : which runs to include in the current plots
    '''
    # train epochs, epoch losses, perplexities
    epochs, losses, plexes = [], [], []

    with open(data_path, "r") as f:                  # "r" activates read only mode
        reader = csv.reader(f)                       # initialize reader instance to read each row
        for index, row in enumerate(reader):
            # print(f"row = {row}")
            if row[0] == run_id:
                epochs.append(int(row[1]))           # second column => epoch
                losses.append(float(row[2]))         # third column  => corresponding loss
                plexes.append(float(row[3]))         # fourth column => corresponding perplexities
        
    return epochs, losses, plexes


def plots(folder_name, epochs, losses, title, x_label, y_label):
    '''
    Plot epoch loss per epoch.
    '''
    os.makedirs(folder_name, exist_ok = True)
    save_path = "./" + folder_name + "/" + title + ".png"
    plt.figure()
    plt.plot(epochs, losses, marker = 'o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)      # grid argument should be boolean
    plt.savefig(save_path)
    plt.close()



if __name__ == "__main__":
    data_path = "./visualization/out_log.csv"   # path to collect plotting data 
    run_id = "sample_10_lr_0.0003"

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
           title       = "Epoch Losses Per Epoch",
           x_label     = "Epochs",
           y_label     = "Losses"
    )

    # plot perplexities
    plots(
           folder_name = "visualization",
           epochs      = epochs,
           losses      = plexes,
           title       = "Perplexities Per Epoch",
           x_label     = "Epochs",
           y_label     = "Perplexities"
    )
