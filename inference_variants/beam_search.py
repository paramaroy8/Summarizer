'''
Generation Based Inference : Beam Search with Repetition Penalty and N-gram Sequence Blocking
'''
import os, sys, torch, time

from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import BartTokenizer

from building_blocks import model_comps, load_checkpoint
from utils import Logger 


def r_penalty(penalty, seen_tokens, last_token):
    """ 
    Repetition Penalty 

    penalty     : penalty value for the last token of the current beam
    seen_tokens : tokens seen for the current beam so far
    last_token  : comparison scores for the last token generated for the current beam
    """
    if penalty != 1.0 and len(seen_tokens) > 0:
        for seen_token in seen_tokens:
            # scale seen token ID probabilities to prevent repetition
            if last_token[seen_token] > 0:
                last_token[seen_token] /= penalty    # positive probability gets penalized by division
            else:
                last_token[seen_token] *= penalty    # negative value get smaller during multiplication
    return last_token


def n_gram(dec_input, n_size, last_token):
    """
    N-gram Sequence Blocking

    dec_input  : decoder input for the current beam, dimension : (num of tokens generated so far for this beam, )
    n_size     : num of tokens for each sequence 
    last_token : comparison scores for the last token generated for the current beam
    """

    prev_tokens = dec_input.cpu().tolist()     # convert decoder input from tensor to python list

    '''
    If (n - 1) token sequence is found, then the nth token must be blocked to prevent sequence repetition.
    '''
    if len(prev_tokens) >= n_size - 1:
        '''
        We take one less token because we want to block the final token if the previous tokens in this sequence have already 
        occurred. As a result, if n_size is too small, the language structure breaks.
        '''
        context = prev_tokens[-(n_size - 1) :]  # sequence that should not appear

        # slide through all generated tokens to ensure the context sequence has not already appeared
        for i in range(len(prev_tokens) - n_size + 1):
            current_window = prev_tokens[i : i + n_size - 1]

            if current_window == context:
                '''
                Block reoccurrence of this sequence by setting the score of the final token of the sequence 
                to a very small value such that softmax would convert it to 0, ensuring this token never gets chosen again. 
                '''
                last_token[prev_tokens[i + n_size - 1]] = float("-inf")

    return last_token


def beam_search(min_len, beam_width, alpha, n_size, penalty, temperature, tokenizer, max_len, batch, emb, enc, dec, out_proj, vocab_size, device, DEBUG):
    """
    Decoding with Beam Search with n-gram and repetition penalty.
    """
    # BOS token ID
    BOS_token = tokenizer.bos_token_id  # 0
    # EOS token ID
    EOS_token = tokenizer.eos_token_id  # 2

    enc_input = batch["input_ids"].to(device)
    enc_pad   = batch["attention_mask"].bool().to(device) # (batch_size, enc_seq_len)

    # (batch_size, 1, enc_seq_len) & (batch_size, enc_seq_len, 1) = (batch_size, enc_seq_len, enc_seq_len)
    enc_mask = (enc_pad.unsqueeze(1) & enc_pad.unsqueeze(2)).unsqueeze(1) # (batch_size, 1, enc_seq_len, enc_seq_len)

    cross_mask = enc_pad.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, enc_seq_len)

    # input embedding for encoder
    source_emb = emb(token_ids = enc_input)

    # encoder output
    enc_out = enc(x = source_emb, att_mask = enc_mask) # (1, enc_seq_len, d_model)

    # decoder self-attention mask
    dec_mask = None

    ''' ---------- Beam Search ---------- '''

    # shape : (beam_width, 1) ––– initially, each decoder beam has only the BOS token
    # each row corresponds to each beam (hypothesis)
    # each batch processes only one paper at a time
    dec_input = torch.full((beam_width, 1), BOS_token, dtype = torch.long).to(device) 

    # initialize cumulative log-prob score for each beam
    beam_scores = torch.zeros(beam_width, device = device) # (beam_width, )

    '''
    To prevent identical beams, 
    the first generated tokens for each beam will be chosen from the top K tokens of the first beam.
    '''
    if beam_width > 1:
        beam_scores[1 :] = float("-inf")
    
    '''
    The current encoder mask and cross attention mask are set for only one beam.
    So, we will repeat for all beams.
    '''
    enc_out = enc_out.repeat(beam_width, 1, 1) # (beam_width, enc_seq_len, d_model)
    cross_mask = cross_mask.repeat(beam_width, 1, 1, 1) # (beam_width, 1, 1, enc_seq_len)

    '''
    Each beam has its own seen_tokens set.
    '''
    all_seen_tokens = [set() for _ in range(beam_width)]

    '''
    All beams need to reach ending before the beam search stops. 
    Some beams may end faster than others.
    We will need to keep track of beams that reached their ending and ensure no unnecessary tokens are generated further
    for these beams while other beams are still reaching their ends.

    Active Mask tells us which beam_index is still generating.
    '''
    completed_beams = [] # store the beams that have reached ending

    active_mask = torch.ones(beam_width, dtype = torch.bool, device = device)

    '''
    dec_input.shape[1] is the sequence length of each beam. 
    Each sequence has same number of generated tokens since they grow toether.
    '''
    while (dec_input.shape[1] < max_len):
        # embed the decoder input, dimension : (beam_width, batch_size = 1, d_model)
        dec_emb = emb(token_ids = dec_input)

        # decoder raw output
        dec_out = dec(
                        x = dec_emb, 
                        enc_out = enc_out,
                        dec_mask = dec_mask,
                        enc_mask = cross_mask
                    )
        # logits (hidden states) of all beams
        # current_len = current decoder input length used to generate the logits
        logits = out_proj(dec_out) # (beam_width, current_len, vocab_size)

        '''
        Last hidden state is the new generated token ID.
        Since we are selecting a single row using -1 that corresponds to last row, this dimension gets dropped.

        Detach from computation graph so that any in-place updates do not affect it.
        Clone it to create a new tensor to separate the memory allocation.
        '''
        last_token = logits[:, -1, :].detach().clone() # (beam_width, vocab_size)

        # scale with temperature
        if temperature > 0 :
            last_token /= temperature
        
        ''' ----- Per Beam Loop ----- '''

        for beam_index in range(beam_width):

            '''
            To ensure changes in one beam does not affect the others, 
            we will use a seaparate variable for applying repetition penalty and n-gram sequence blocking.
            '''
            current_beam_logits = last_token[beam_index].clone()  # (vocab_size, )

            '''
            If active mask is False, then it has already completed generating.
            '''
            if not active_mask[beam_index]:
               continue
            
            # apply repetition penalty 
            current_beam_logits = r_penalty(penalty, all_seen_tokens[beam_index], current_beam_logits)    # (vocab_size, )

            # apply n-gram sequence blocking
            current_beam_logits = n_gram(dec_input[beam_index], n_size, current_beam_logits)   # (vocab_size, )

            # add it back to last_token tensor
            last_token[beam_index] = current_beam_logits   # (vocab_size, )

        
        '''
        For each hypothesis, probability of each chosen tokens get multiplied to compute total scores of the given hypothesis.
        Log-probs help prevent underflow that occurs in case of multiplication of probabilities.
        '''
        log_probs = torch.log_softmax(last_token, dim = -1)     # (beam_width, vocab_size)

        '''
        Accumulation Score of each beam represents the quality of that beam.
        It is the cumulative scores of the tokens generated so far for each beam.
        '''
        acc_scores = beam_scores.unsqueeze(1) + log_probs       # (beam_width, vocab_size)

        '''
        Flatten the accumulation scores before choosing top beam_width number of beams.
        '''
        acc_flat = acc_scores.view(1, -1)       # (1, beam_width * vocab_size)

        # get top scores and corresponding indices from the flattened scores
        top_scores, top_indices = torch.topk(acc_flat, beam_width, dim = -1)

        # update dimensions
        top_scores  = top_scores.squeeze(0)      # (beam_width, )
        top_indices = top_indices.squeeze(0)     # (beam_width, )

        '''
        Recover beam indices and corresponding chosen token IDs.

        Some beams may be chosen more than once, some beams may be discarded.
        '''
        beam_indices = top_indices // vocab_size
        new_tokens   = top_indices % vocab_size   # (beam_width, )

        '''
        Based on the chosen beam indices, 
        create new decoder input by  first extracting corresponding rows from the previous decoder input,
        and then, add new chosen tokens to the decoder input.
        '''
        dec_input  = dec_input[beam_indices]                     # (beam_width, current_beam_length)
        new_tokens = new_tokens.unsqueeze(1)                     # (beam_width, 1)
        dec_input  = torch.cat([dec_input, new_tokens], dim = 1) # columnwise concatenation

        '''
        Update encoder output and cross attention mask based on new token IDs.
        '''
        enc_out = enc_out[beam_indices]
        cross_mask = cross_mask[beam_indices]

        '''
        Update seen_tokens of each beam with the newly generated tokens.
        '''
        # first, get the beams that survived
        all_seen_tokens = [all_seen_tokens[b].copy() for b in beam_indices.cpu().tolist()]
        # finally, add new tokens to these beams
        for i, token in enumerate(new_tokens.squeeze(1).cpu().tolist()):
            all_seen_tokens[i].add(token)
        
        '''
        Update active masks using beam indices.
        '''
        active_mask = active_mask[beam_indices]
        
        # top scores become the new beam scores
        beam_scores = top_scores

        '''
        Check each hypothesis for EOS token.
        Each beam ends as soon as it generates EOS token AND the beam is at least as long as minimum length.
        We use (Minimum Length + 2) because each beam contains BOS and EOS tokens as well that should not be considered.
        '''
        for beam_index, token in enumerate(new_tokens.squeeze(1).cpu().tolist()):
            if token == EOS_token and dec_input.shape[1] >= min_len + 2:
                if active_mask[beam_index]:
                    '''
                    Normalize raw scores using length normalization.
                    '''
                    raw_scores = beam_scores[beam_index].item()
                    beam_length = dec_input[beam_index].shape[0]
                    length_norm_factor = ((5 + beam_length) ** alpha) / (10 ** alpha)
                    normalized_scores = raw_scores / length_norm_factor


                    # current beam index reached ending
                    # add the beam and its normalized score to completed_beams list
                    # we clone the score to avoid in-place modification to avoid any effect on other values in dec_input
                    completed_beams.append((dec_input[beam_index].clone(), normalized_scores))
                    # set the current active mask to False
                    active_mask[beam_index] = False 
                    '''
                    Set corresponding beam_index free by setting the index value to negative infinity because from now on
                    any addition of log-probs would still result to negative infinity.
                    '''
                    beam_scores[beam_index] = float("-inf")
        
        
        if not active_mask.any():
            # all beams reached ending
            print(f"\nBeam Search Complete!\n\nAll Beam Scores = {beam_scores}\n")
            break
        

    '''
    Sort completed beams based on scores.
    '''
    completed_beams.sort(key = lambda x : x[1], reverse = True)

    return completed_beams  # all generated beams sorted by score 


def generate_inference(
    min_len, beam_width, alpha, n_size, penalty, temperature, tokenizer, max_len, 
    emb, enc, dec, out_proj, vocab_size, device, dataloader, DEBUG = False
):
    """
    Generate Summary without seeing the abstract.
    """
    # start_time = time.time()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, 1):
            print(f"\nPaper = {batch_index}\n")

            # get several beams
            completed_beams = beam_search(min_len, beam_width, alpha, n_size, penalty, temperature, tokenizer, max_len, batch, 
                                              emb, enc, dec, out_proj, vocab_size, device, DEBUG)
            
            for index, (beam, score) in enumerate(completed_beams, 1):
                # decode the token IDs to generate final summary of the paper
                summary = tokenizer.decode(beam.cpu().tolist(), skip_special_tokens = True)
                
                print(f"\nRank = {index} | Score = {score:.4f}\nSummary = \n{summary}\n\n")

            # decode abstract of the paper for reference
            # since we are using one paper per batch, [0] makes sense
            label = batch["labels"][0]
            # get rid of pad tokens
            label = label[label != -100]
            
            reference = tokenizer.decode(label.cpu().tolist(), skip_special_tokens = True)

            print(f"\nReference =\n{reference}\n")

    return 


def main():
    device = torch.device(
                          "mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu"
                        )

    ''' Set up directories '''
    os.makedirs("./console_logs/", exist_ok = True)
    
    ''' Experiment Parameters '''
    model_path = "./checkpoints/latest_model_0.pt"

    model_description = "Training Data = 200K+, Epoch = 2"
    
    console_path      = "./console_logs/beam_console_3.txt"    # save console training output

    penalties     = [1.4]      # repetition penalties

    n_gram_sizes  = [4]        # n-gram context sizes

    beam_nums     = [4]        # number of hypotheses

    alphas        = [1.3]      # value for beam length normalization

    temperatures  = [1.0]      # control how closely the output should be aligned to abstract
    
    ''' Number of Allowed Tokens in Summary '''

    min_len_list  = [5]       # list of possible minimum lengths of the output
    max_len       = 500       # maximum length of the output  

    DEBUG         = False 
    vocab_size    = 50265
   
    test_data_path  = "./tokenized-dataset/test"
    full_test_data  = load_from_disk(test_data_path)
    test_samples    = 1        # number of papers to be used during inference
    test_data       = full_test_data.select(range(test_samples))
    test_data       = test_data.with_format("torch")  # convert Python list to PyTorch tensor
    batch_size      = 1        # number of papers per batch
    dataloader      = DataLoader(test_data, batch_size = batch_size)

    write_choice    = "a"      # "w" for overwrite and "a" for keeping everything in log file

    sys.stdout = Logger(filepath = console_path, write_choice = write_choice)
    
    print(
           f"Inference on device = {device}\n"
           f"\nModel Description: {model_description}\n"
        )

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # initialize model components
    emb, enc, dec, out_proj, _, _, _, _ = model_comps(
                            device, d_model = 512, d_ffn_enc = 1408, d_ffn_dec = 1408, vocab_size = vocab_size, 
                            enc_heads = 8, dec_heads = 8, cross_heads = 8, enc_blocks = 5, dec_blocks = 10, 
                            emb_drop = 0, enc_att_drop = 0,  enc_ffn_drop = 0, mask_att_drop = 0, cross_att_drop = 0,
                            dec_ffn_drop = 0, DEBUG = DEBUG, learning_rate = 0, weight_decay = 0, sch_mode = "min", 
                            sch_factor = 0, sch_patience = 0
                            )
    
    # load checkpoint, ignore returned epoch number
    _ = load_checkpoint(device, model_path, emb, enc, dec, out_proj, optimizer = None, scheduler = None)

    emb.eval()
    enc.eval()
    dec.eval()
    out_proj.eval()

    for beam_width in beam_nums:
        for alpha in alphas:
            for n_size in n_gram_sizes:
                for penalty in penalties:
                    for temperature in temperatures:
                        for min_len in min_len_list:
                            print(
                                f"\nExperiment Params:\n"
                                f"Beam Width            = {beam_width}\n"
                                f"Alpha                 = {alpha}\n"
                                f"N-Gram size           = {n_size}\n" 
                                f"Repetition Penalty    = {penalty}\n"
                                f"Temperature           = {temperature}\n"
                                f"Minimum Length        = {min_len}\n\n"
                            )
                            generate_inference(  
                                                min_len, beam_width, alpha, n_size, penalty, temperature, tokenizer, max_len, 
                                                emb, enc, dec, out_proj, vocab_size, device, dataloader, DEBUG = False
                                            )
    
    # finally, close console and file
    sys.stdout.close()

if __name__ == "__main__":
    main()
