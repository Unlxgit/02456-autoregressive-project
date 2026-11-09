import torch 
import torch.nn as nn
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from protgen.data import total_tokens
from protgen.models import MyRecurrentNet
import numpy as np
from src.protgen.models.mlp import MLP
import torch.nn.functional as F
import copy

#Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
EMBEDDING_SIZE = 25
HIDDEN_DIM = 64

#Load data
train_data = ProteinDataset(path_to_data="data/train.csv")
val_data = ProteinDataset(path_to_data="data/val.csv")
test_data = ProteinDataset(path_to_data="data/test.csv")

# DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

#Model 
model = MLP(vocab_size=total_tokens, embedding_dim=EMBEDDING_SIZE, hidden_dim=HIDDEN_DIM)

#Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def compute_loss(model, input, target):
    raw_prediction = model(input)                                                   #Of size (B, 129, vocab_size)
    return F.cross_entropy(
                            raw_prediction.reshape(-1, raw_prediction.size(-1)),    #Reshapes to (B*129, vocab_size)
                            target.reshape(-1),                                     # Reshaped to (B*129,)
                            ignore_index=-1                                         # Skips positions after the end
                           )

@torch.no_grad()
def evaluate(model , dataloader):
    "Computes loss and different metrics such as accuracy"

    #Put model into eval mode
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0

    for input_tokens, target_tokens, _, _ in dataloader:
        pred = model(input_tokens)

        flat_pred = pred.reshape(-1, pred.size(-1))
        flat_target = target_tokens.reshape(-1) 
        valid_targets = flat_target != -1

        total_loss += F.cross_entropy(flat_pred, flat_target, ignore_index=-1, reduction="sum").item()
        total_correct += (flat_pred.argmax(-1)[valid_targets] == flat_target[valid_targets]).sum().item()
        total_tokens += valid_targets.sum().item()

    mean_loss = total_loss / total_tokens
    return mean_loss, total_correct / total_tokens

best_val_loss = float("inf")
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    training_loss = 0

    for input_tokens, target_tokens, _, _ in train_loader:
        loss = compute_loss(model, input_tokens, target_tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()


    #Set model to eval 
    model.eval()

    validation_loss = 0

    with torch.no_grad():
        for input_tokens, target_tokens, _, _ in val_loader:
            loss = compute_loss(model, input_tokens, target_tokens)
            validation_loss += loss.item()


    mean_training_loss = training_loss / len(train_loader)
    mean_validation_loss = validation_loss / len(val_loader)

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, "best_mlp.pt")



    #Print progress and loss

    print(f"epoch {epoch}  train loss {mean_training_loss}  val loss {mean_validation_loss}")


model.load_state_dict(best_model_state)
test_loss, test_accuracy = evaluate(model, test_loader)

print(f"\nbest val loss: {best_val_loss:.4f}")
print(f"test loss:     {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.1%}")
print(f"test perplexity: {torch.exp(torch.tensor(test_loss)):.2f}")





# def train(net, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs=20):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     net.to(device)

#     for epoch in range(num_epochs):
#         # Forward -> Backprob -> Update params
#         ## Train
#         epoch_training_loss = 0
#         epoch_validation_loss = 0


#         # For each batch in the validation set
#         net.eval()
#         with torch.no_grad():
#             for x, x_shifted, y, length in val_loader:

#                 # next-token loss, skip the start step
#                 logits = net(x)
#                 loss = criterion()

#         net.train()
#         for _, x_shifted, y, length in train_loader:
#             optimizer.zero_grad()

#             logits = net(x)
#             loss = criterion()
            
#             batch_loss = criterion(output, target_batch)
#             batch_loss.backward()
#             optimizer.step()
            
#             cur_loss += batch_loss   


                

#         train_acc_cur = accuracy_score(train_targs, train_preds)
#         valid_acc_cur = accuracy_score(val_targs, val_preds)
        
#         train_acc.append(train_acc_cur)
#         valid_acc.append(valid_acc_cur)
            
#         if epoch % 10 == 0:
#             print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
#                     epoch+1, losses[-1], train_acc_cur, valid_acc_cur))



# print(f"Data class: {train_data}")

# # Get tensors
# X_train = train_data.tokens
# y_train_labels = train_data.labels          # classification task
# y_train_shifted = train_data.tokens_shifted # generative task

# print(f"# Train data: \n Tokens shape: {X_train.shape}\n Labels (shape): {y_train_labels.shape} \n Shifted shape: {y_train_shifted.shape}")


# # Inddexing
# x, x_shifted, y, length = train_data[0]
# print(f"# Sample 0: \nTokens: \n {x}\n Shifted: \n {x_shifted}\n Label: {y}\n Length: {length}")



# print(train_loader)

# # net = MyRecurrentNet()
# # net = MLP()
# # net =   
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# train(net, train_loader, val_loader, test_loader, optimizer, criterion)
