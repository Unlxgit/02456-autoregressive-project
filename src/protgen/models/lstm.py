import torch
import torch.nn as nn

from protgen.data import total_tokens

# Fixed project vocabulary: <PAD>, <BOS>, <EOS>, then the 20 residues. Same ids as ProteinDataset.
vocab_size = len(total_tokens)


class MyRecurrentNet(nn.Module):
    def __init__(self, embedding_dim=50, hidden_dim=50):
        super(MyRecurrentNet, self).__init__()
        #input size = (30,129)

        # embed token ids
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch dimension comes first
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True)

        # Output layer
        self.l_out = nn.Linear(in_features=hidden_dim,
                            out_features=vocab_size,
                            bias=True)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x, (h, c) = self.lstm(x)
        # score for each token in the vocab
        return self.l_out(x)
