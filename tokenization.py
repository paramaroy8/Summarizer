'''
Tokenize and save dataset. 

To load, use:

    from datasets import load_from_disk

    train_path = "./tokenized-dataset/train" # path for train tokenized data
    train_data = load_from_disk(train_path)
'''
import os
from transformers import BartTokenizer # vocab size = 50265
from get_full_dataset import load_data

# initialize tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenize(example):
    # input tokenized data 
    source = tokenizer(
        example["article"],
        max_length = 1024,
        truncation = True,
        padding = "max_length"
    )

    # output tokenized data
    target = tokenizer(
        example["abstract"],
        max_length = 500, # abstracts are smaller in size
        truncation = True,
        padding = "max_length"
    )

    # create labels of target token IDs
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in each_sample]
        for each_sample in target["input_ids"]
        ]

    output = {
        "input_ids"              : source["input_ids"],
        "labels"                 : labels,
        "attention_mask"         : source["attention_mask"],
        "decoder_attention_mask" : target["attention_mask"]
    }

    return output


def main(datatype):
    # path to load data from
    load_path = "./raw_data/arxiv" 
    # load data
    data = load_data(load_path, datatype)

    # sample = data.select(range(2))

    tokenized_data = data.map(tokenize, batched = True, batch_size = 1000)

    print("\n", datatype, "tokenization complete!\n")

    # remove columns from tokenized data
    tokenized_data = tokenized_data.remove_columns(["article", "abstract"])

    print("\nraw text columns removed")

    # save path
    save_path = "./tokenized-dataset"

    # create path for current datatype
    current_path = os.path.join(save_path, datatype)

    # save tokenized data
    tokenized_data.save_to_disk(current_path)

    print("\n", datatype, "tokenized data saved!\n")


if __name__ == "__main__":
    for datatype in ["train", "validation", "test"]:
        main(datatype)
    
    print("\n Tokenization Complete! \n")
