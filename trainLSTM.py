import torch
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models import MyRecurrentNet
import matplotlib.pyplot as plt
import numpy as np

#Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 40
EMBEDDING_SIZE = 60
HIDDEN_DIM = 130

#Load data
train_data = ProteinDataset(path_to_data="data/train.csv")
val_data = ProteinDataset(path_to_data="data/val.csv")
test_data = ProteinDataset(path_to_data="data/test.csv")

# DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

#Model from lstm.py. At each position it has seen the whole sequence up to there.
model = MyRecurrentNet(embedding_dim=EMBEDDING_SIZE, hidden_dim=HIDDEN_DIM)

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

train_losses = []
val_losses = []

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

    train_losses.append(mean_training_loss)
    val_losses.append(mean_validation_loss)

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, "best_lstm.pt")


    #Print progress and loss

    print(f"epoch {epoch}  train loss {mean_training_loss}  val loss {mean_validation_loss}")


model.load_state_dict(best_model_state)
test_loss, test_accuracy = evaluate(model, test_loader)

print(f"\nbest val loss: {best_val_loss:.4f}")
print(f"test loss:     {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.1%}")
print(f"test perplexity: {torch.exp(torch.tensor(test_loss)):.2f}")

#Plot training and validation losses per epoch
epoch = np.arange(len(train_losses))
plt.plot(epoch, train_losses, 'ro-', label='Training loss')
plt.plot(epoch, val_losses, 'bo-', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
