'''
Collect 2 papers to build the tokenizer pipeline first.

Once the pipeline is complete, we can download the complete data and complete the tokenization, and save them.
'''

from datasets import load_dataset, load_from_disk
import os # joining path
import shutil # deleting cache

def download_data(cache_path):
    '''
    Download dataset.
    '''

    train_cache_path = os.path.join(cache_path, "train")
    validation_cache_path = os.path.join(cache_path, "validation")
    test_cache_path = os.path.join(cache_path, "test")

    # download 10 JSON files and store them in arxiv_dataset folder
    # currently it is set to download 2 of each type
    raw_train_data = load_dataset("ccdv/arxiv-summarization", "section", split = "train[:2]", cache_dir = train_cache_path) 
    print("\ntraining dataset downloaded!\n")
    raw_validation_data = load_dataset("ccdv/arxiv-summarization", "section", split = "validation[:2]", cache_dir = validation_cache_path)
    print("\nvalidation dataset downloaded!\n")
    raw_test_data = load_dataset("ccdv/arxiv-summarization", "section", split = "test[:2]", cache_dir = test_cache_path)
    print("\ntest dataset downloaded!\n")

    return raw_train_data, raw_validation_data, raw_test_data


def save_data(save_path, raw_train_data, raw_validation_data, raw_test_data):
    '''
    Save dataset to the computer.

    Takes the variables from loaded cache which is then used to save to disk.
    '''

    train_save_path = os.path.join(save_path, "train")
    validation_save_path = os.path.join(save_path, "validation")
    test_save_path = os.path.join(save_path, "test")

    raw_train_data.save_to_disk(train_save_path)
    raw_validation_data.save_to_disk(validation_save_path)
    raw_test_data.save_to_disk(test_save_path)
    
    print("\ncompleted saving to disk!\n")

def delete_cache(cache_dir):
    '''
    Delete cached dataset.
    Ensure dataset is saved to computer before deleting cache.
    '''

    shutil.rmtree(cache_dir)
    
    print("\ncache deleted!\n")


def load_data(load_path, data_type):
    '''
    Load dataset from the given path.

    Path will determine whether we are getting training, test or validation dataset.
    '''
    # form the correct path based on the data type
    path = os.path.join(load_path, data_type)
    # load from the disk
    data = load_from_disk(path)

    # we can choose only a subset of the total downloaded dataset
    # here, we chose 2 papers
    subset = data.select(range(2)) 

    print("\ncompleted loading!\n")

    # each dictionary elements
    print("fields available =", data[0].keys())

    # preview abstract
    print("\n\nabstract preview =", data[0]["abstract"][:200], "\n")

    return data

if __name__ == "__main__":
    '''
    Ensure correct flags have been 
    '''
    print("Ensure correct flags have been activated!")

    cache_path = "./raw_data/cache"
    save_path = "./raw_data/arxiv"
    load_path = "./raw_data/arxiv" 

    raw_train_data = raw_validation_data = raw_test_data = None

    download = False # if we want to download dataset instead of simply loading, we can turn this flag on
    save = False # True if we want to save the downloaded data to the computer
    del_cache = False # True to delete cached data
    load = True # False if we don't want to load

    if download:
        # download dataset
        raw_train_data, raw_validation_data, raw_test_data = download_data(cache_path) 
    
    if save:
        # save to disk
        save_data(save_path, raw_train_data, raw_validation_data, raw_test_data)
    
    if del_cache:
        # delete cached data
        delete_cache(cache_path)

    if load:
        # load data
        data_type = "train" # or "validation" or "test"
        data = load_data(load_path, data_type)