import torch 
import torch.nn as nn
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from protgen.data import total_tokens
from protgen.models import MyRecurrentNet
import numpy as np


def windowed_batch(batch, sequence_length=3, padding_value=0):

    left_padding = torch.full((batch.size(0), sequence_length-1),
                              padding_value, dtype=batch.dtype, 
                              device=batch.device)
    batch_with_padding = torch.cat([left_padding, batch], dim=1)

    return batch_with_padding.unfold(1,sequence_length,1)

test_tensor = torch.tensor(np.array([[1, 2, 3, 5], [4, 5, 6, 10]]))

print(windowed_batch(test_tensor))

NUMBER_OF_AMINO_ACIDS = 20
BATCH_SIZE = 32
vocab_size = len(total_tokens)

class MLP(nn.Module):
    def __init__(self, sequence, input_dim=3, output_dim=1):
        super().__init__()
        #Embedding of tokens
        self.embedding = nn.Embedding(vocab_size, 50)
        self.sequence_length = len(sequence)

        self.layers = nn.Sequential(
            nn.Linear(input_dim * NUMBER_OF_AMINO_ACIDS, 32),
            nn.ReLU(),
            nn.Linear(32,32), 
            nn.ReLU(),
            nn.Linear(NUMBER_OF_AMINO_ACIDS,output_dim)
        )
    
    def forward(self, token_ids):
        # chop up here

        # loop over chops
        self.layers

        # merge chops here

        return # return sequences

    with torch.no_grad():
        for x, x_shifted, y, length in val_loader:

            # next-token loss, skip the start step
            logits = net(x)
            loss = criterion(logits[:, 1:].transpose(1, 2), x_shifted[:, :-1])

            # Update loss
            epoch_validation_loss += loss.detach().numpy()

def train(net, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        ## Train
        epoch_training_loss = 0
        epoch_validation_loss = 0


        # For each batch in the validation set
        net.eval()
        with torch.no_grad():
            for x, x_shifted, y, length in val_loader:

                # next-token loss, skip the start step
                logits = net(x)
                loss = criterion()

        net.train()
        for _, x_shifted, y, length in train_loader:
            optimizer.zero_grad()

            logits = net(x)
            loss = criterion()
            
            batch_loss = criterion(output, target_batch)
            batch_loss.backward()
            optimizer.step()
            
            cur_loss += batch_loss   


                

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
            
        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                    epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

train_data = ProteinDataset(path_to_data="data/train.csv")
val_data = ProteinDataset(path_to_data="data/val.csv")
test_data = ProteinDataset(path_to_data="data/test.csv")

print(f"Data class: {train_data}")

# Get tensors
X_train = train_data.tokens
y_train_labels = train_data.labels          # classification task
y_train_shifted = train_data.tokens_shifted # generative task

print(f"# Train data: \n Tokens shape: {X_train.shape}\n Labels (shape): {y_train_labels.shape} \n Shifted shape: {y_train_shifted.shape}")


# Inddexing
x, x_shifted, y, length = train_data[0]
print(f"# Sample 0: \nTokens: \n {x}\n Shifted: \n {x_shifted}\n Label: {y}\n Length: {length}")

# DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(train_loader)

net = MyRecurrentNet()
net = MLP()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train(net, train_loader, val_loader, test_loader, optimizer, criterion)
