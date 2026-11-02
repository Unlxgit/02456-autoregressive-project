import torch 
import torch.nn as nn
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
import torch.optim as optim


NUMBER_OF_AMINO_ACIDS = 20
BATCH_SIZE = 32

class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim * NUMBER_OF_AMINO_ACIDS, 32),
            nn.ReLU(),
            nn.Linear(32,32), 
            nn.ReLU(),
            nn.Linear(NUMBER_OF_AMINO_ACIDS,output_dim)
        )
    
    def forward(self, x):
        return self.layers



def train(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        ## Train
        cur_loss = 0
        model.train()
        for i in range(num_batches_train):
                optimizer.zero_grad()
                slce = get_slice(i, batch_size)
                output = net(x_train[slce])
                
                # compute gradients given loss
                target_batch = targets_train[slce]
                batch_loss = criterion(output, target_batch)
                batch_loss.backward()
                optimizer.step()
                
                cur_loss += batch_loss   
            losses.append(cur_loss / batch_size)

            net.eval()
            ### Evaluate training
            train_preds, train_targs = [], []
            for i in range(num_batches_train):
                slce = get_slice(i, batch_size)
                output = net(x_train[slce])
                
                preds = torch.max(output, 1)[1]
                
                train_targs += list(targets_train[slce].numpy())
                train_preds += list(preds.data.numpy())

            ### Evaluate validation
            val_preds, val_targs = [], []
            for i in range(num_batches_valid):
                slce = get_slice(i, batch_size)
                
                output = net(x_valid[slce])
                preds = torch.max(output, 1)[1]
                val_targs += list(targets_valid[slce].numpy())
                val_preds += list(preds.data.numpy())
                

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

net = MLP()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train(net, train_loader, val_loader, test_loader, optimizer, criterion)
