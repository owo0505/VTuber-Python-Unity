import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from utils import *

# === Step 1: Load CSVs ===
train_df = pd.read_csv("emotion_train2.csv")
test_df  = pd.read_csv("emotion_test2.csv")

train_df = train_df.apply(align_row, axis=1) ## align face angle by eyes
train_df.columns = [f"f{i}" for i in range(956)] + ["label"]
test_df = test_df.apply(align_row, axis=1)
test_df.columns = [f"f{i}" for i in range(956)] + ["label"]

X_train = select_columns(train_df).iloc[:, :].values.astype(np.float32)
y_train = train_df.iloc[:, -1].values.astype(np.int64)

X_test  = select_columns(test_df).iloc[:, :].values.astype(np.float32)
y_test  = test_df.iloc[:, -1].values.astype(np.int64)


# === Step 2: Create PyTorch Dataset ===
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

print("Input shape:", X_train.shape)
train_ds = LandmarkDataset(X_train, y_train)
test_ds  = LandmarkDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)


model = EmotionNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Step 4: Training ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

# === Step 5: Evaluation ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = model(Xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")
torch.save(model.state_dict(), "emotion_model2.pth")
