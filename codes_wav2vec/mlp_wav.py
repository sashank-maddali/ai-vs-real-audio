import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set seed for reproducibility
torch.manual_seed(42)

# Load embeddings and labels
real_embeds = np.load("/home/sashank/Desktop/project/wav_2vec/embeds_attention/real_embeddings.npy")
ai_embeds = np.load("/home/sashank/Desktop/project/wav_2vec/embeds_attention/ai_embeddings.npy")
X = np.vstack([real_embeds, ai_embeds])
y = np.array([0] * len(real_embeds) + [1] * len(ai_embeds))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define MLP Classifier Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary classification: 2 outputs
        )
    def forward(self, x):
        return self.net(x)

input_dim = X.shape[1]
model = MLPClassifier(input_dim)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluate
model.eval()
all_preds = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.numpy())
        
y_pred_all = np.concatenate(all_preds)
print("MLP Classification Report after attention pooling:")
print(classification_report(y_test, y_pred_all))

