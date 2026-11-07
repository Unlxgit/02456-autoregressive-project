import torch
import torch.nn as nn

from protgen.data import total_tokens

# Fixed project vocabulary: <PAD>, <BOS>, <EOS>, then the 20 residues. Same ids as ProteinDataset.
vocab_size = len(total_tokens)


class MyRecurrentNet(nn.Module):
    def __init__(self):
        super(MyRecurrentNet, self).__init__()

        # embed token ids
        self.embedding = nn.Embedding(vocab_size, 50)
        # batch dimension comes first
        self.lstm = nn.LSTM(input_size=50,
                            hidden_size=50,
                            num_layers=1,
                            batch_first=True)

        # Output layer
        self.l_out = nn.Linear(in_features=50,
                            out_features=vocab_size,
                            bias=False)

    def forward(self, token_ids):
        ws = self.embedding(token_ids)

        # shift right for teacher forcing
        w0 = torch.zeros((ws.shape[0], 1, ws.shape[2]), device=ws.device)
        ws_shifted = torch.cat([w0, ws[:, :-1]], dim=1)

        hidden, (h, c) = self.lstm(ws_shifted)
        # score for each token in the vocab
        return self.l_out(hidden)
