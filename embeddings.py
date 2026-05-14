'''
Create Embedding Class: Learnable Token Embedding and Fixed Positional Encoding.
'''
import torch
import torch.nn as nn
import math

from utils import debug_tensor

class FullEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size = 50265, debug = False, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.DEBUG = debug
        self.dropout = nn.Dropout(p = dropout)

        # create embedding layer that gets updated from each batch
        self.embedding_layer = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.d_model)

    def forward(self, token_ids):
        '''
        Token Embedding

        token_ids : (batch_size, seq_len)
        '''
        _, seq_len = token_ids.shape

        # create token embedding
        token_emb = self.embedding_layer(token_ids)   # (batch_size, seq_len, d_model)

        '''
        Position Embedding
        '''
        pos     = torch.arange(seq_len, device = token_ids.device).unsqueeze(1)    # (seq_len, 1)
        indices = torch.arange(0, self.d_model, 2, device = token_ids.device).float()   # even indices -> (d_model / 2, )
        div     = torch.pow(10000, indices / self.d_model)   # divisor term -> (d_model / 2, )
        angle   = pos / div 

        pos_emb          = torch.zeros(seq_len, self.d_model, device = token_ids.device)
        pos_emb[:, 0::2] = torch.sin(angle)             # even indices
        pos_emb[:, 1::2] = torch.cos(angle)             # odd indices

        # add batch dimension
        pos_emb = pos_emb.unsqueeze(0)                  # (1, seq_len, d_model)

        '''
        Final Embedding
        '''
        final_emb = (token_emb * math.sqrt(self.d_model)) + pos_emb 

        # add dropouts
        final_emb = self.dropout(final_emb)

        if self.DEBUG:
            print(f"\ntoken ID shape = {token_ids.shape}")
            print(f"\nvocab size = {self.vocab_size}")
            debug_tensor("token embeddings", token_emb)
            debug_tensor("positional encoding", pos_emb)
            debug_tensor("final embedding", final_emb)
    

        return final_emb 

if __name__ == "__main__":
    # example input
    d_model = 4
    vocab_size = 10
    DEBUG = True
    dropout = 0.2 
    
    token_ids = torch.tensor([
        [1, 7, 5, 6]
    ]) # four token IDs, one batch

    fe = FullEmbedding(d_model = d_model, vocab_size = vocab_size, debug = DEBUG, dropout = dropout)

    output = fe(token_ids) 

    print(f"\nfinal embedding = {output}")
