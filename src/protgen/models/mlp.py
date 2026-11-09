import torch 
import torch.nn as nn
from utils import windowed_batch




class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, embedding_dim = 8, hidden_dim = 32, vocab_size = 23):
        super().__init__()
        #Embedding of tokens
        self.input_dim = input_dim
        self.padding_value = 0
        self.vocab_size = len(vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, self.vocab_size)
        )

    
    def forward(self, token_ids):
        windows = windowed_batch(token_ids, self.input_dim, self.padding_value)
        embeddings = self.embedding(windows)
        embeddings = embeddings.flatten(-2)

        return self.layers(embeddings)


    