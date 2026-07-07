'''
Clean dataset before tokenization.
'''
from datasets import load_from_disk
import re 
import os
import html

def data_cleaning(batch):
    '''
    batch : dictionary of each paper section as keys and the corresponding texts as values

    Clean artifacts such as @xcite, @xmath etc.
    Each paper data has two sections: abstract and article.
    Clean both sections.
    '''
    for section in ["article", "abstract"]: 
        clean_texts = [] # store clean sections
        for text in batch[section]:
            ''' 
            HTML tags
            '''
            # convert tags into actual characters the tags represent
            text = html.unescape(text) 
            # convert non-breaking spaces (\xa0) to standard spaces
            text = re.sub(r'\xa0', ' ', text)  
            # strip explicit HTML tags if remain, e.g. <sub>, <html> etc.
            # ^> = any character that is not >
            text = re.sub(r'<[^>]*>', ' ', text)


            '''
            Instead of removing citation and math artifacts completely, replacing them with
            placeholders help model understand paper structure better.
            '''
            text = re.sub(r'@xcite\w*', ' [CITATION] ', text) # citation artifacts e.g. @xcite, @xcite123, @xcite_error etc.
            text = re.sub(r'@xmath\w*', ' [MATH] ', text)     # math equation artifacts
            text = re.sub(r'@\w+', '', text)                  # any other artifacts e.g. @xfootnote

            '''
            strip table descriptors
            '''
            text = re.sub(r'\[\s*cols?\s*=.*?\]', ' ', text)         # remove [cols= " >, >, >, > ",]

            '''
            Math leftovers
            '''
            text = re.sub(r'\\[,;!{}]', '', text)
            text = re.sub(r'~+', '', text)
            text = re.sub(r'\{?\w*\s*\}', '', text)
            text = re.sub(r'\s*\^\s*\w*', '', text)            # remove certain artifacts e.g., (^ 2, ^s etc)

            '''
            word and punctuation formatting spacing rules
            '''
            text = re.sub(r'\s+-\s+', '-', text)            # any space around - within single words
            text = re.sub(r'\s*_\s*', ' ', text)            # remove isolated underscores
            text = re.sub(r'\s+([.,:;])', r'\1', text)      # keep group 1 pattern removing excess space before it

            '''
            special characters
            '''
            text = re.sub(r'[$#&\\|`}{?*]', '', text)        # get rid of unnecessary special characters

            '''
            spaces inside brackets
            '''
            text = re.sub(r'\(\s+', '(', text)
            text = re.sub(r'\s+\)', ')', text)

            text = re.sub(r'\[\s+', '[', text)
            text = re.sub(r'\s+\]', ']', text)

            '''
            clean messy bracket remnants and LaTEX commands
            '''
            text = re.sub(r'\\?\[\s*\]\s*\](?:\s*\])+', ' ', text)   # remove [ ] ], ] ] padded with space
            text = re.sub(r'\\?\]\s*\]+', ' ', text)                 # remove .\]] or loose ] ] artifacts
            text = re.sub(r'\\[a-zA-Z0-9]+', ' ', text)              # remove commands such as, \bigl, \mbox, \c67
            text = re.sub(r'(\[MATH\]\s*)\]+', r'\1', text)          # convert [MATH]]].. to [MATH]
            
            '''
            collapse two or more spaces into a single space; remove trailing spaces
            '''
            text = re.sub(r'\s{2,}', ' ', text).strip()
            
            # store the cleaned text of the current section
            clean_texts.append(text)
        
        batch[section] = clean_texts                       # update batch section with cleaned section
    
    return batch

def preview(raw_folder_path, clean_folder_path, data_type):
    '''
    Preview clean data.
    '''
    raw_data_path = os.path.join(raw_folder_path, data_type)
    # get the raw data
    raw_data = load_from_disk(raw_data_path)

    clean_data_path = os.path.join(clean_folder_path, data_type)
    # get the raw data
    clean_data = load_from_disk(clean_data_path)

    print(f"\n\nraw {data_type} first paper article preview =", raw_data[5]["article"][:], "\n")

    print(f"\n\nclean {data_type} first paper article preview =", clean_data[5]["article"][:], "\n")
    

def main(raw_folder_path, clean_folder_path, data_type):
    '''
    folder_path : path of the folder where the complete raw dataset have been stored
    data_type   : train, validation, test
    '''
    # create path for raw data
    raw_data_path = os.path.join(raw_folder_path, data_type)

    # get the raw data
    raw_data = load_from_disk(raw_data_path)
    print(f"Completed loading raw data!")

    # clean data batchwise
    clean_data = raw_data.map(data_cleaning, batched = True, batch_size = 1000)

    # save clean data separately in the raw data folder
    clean_data_path = os.path.join(clean_folder_path, data_type)
    clean_data.save_to_disk(clean_data_path)

    print(f"\nclean {data_type} data saved to disk!\n")



if __name__ == '__main__':
    raw_folder_path = "./raw_data/arxiv"
    clean_folder_path = "./raw_data/clean_arxiv"

    CLEAN = False
    PREVIEW = True

    if CLEAN:
        for data_type in ["train", "validation", "test"]:
            main(raw_folder_path, clean_folder_path, data_type)
    
    if PREVIEW:
        data_type = "train"
        preview(raw_folder_path, clean_folder_path, data_type)

        
