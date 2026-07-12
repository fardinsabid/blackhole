"""
BlackHole Optimizer - MNIST CNN Training Demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from blackhole import BlackHole

# ============================================================
# 1. CNN MODEL
# ============================================================

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ============================================================
# 2. LOAD DATA
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ============================================================
# 3. MODEL & OPTIMIZER
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTCNN().to(device)
print(f"🔥 Using device: {device}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = BlackHole(
    model.parameters(),
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    G=0.01,
    c=10.0,
    hbar=0.01,
    k_B=0.01,
    Lambda=0.001,
    alpha=0.05,
    spin=0.5,
    extra_dim_strength=0.01
)

# ============================================================
# 4. TRAINING
# ============================================================

def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy

def test():
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# ============================================================
# 5. TRAINING LOOP
# ============================================================

print("\n🚀 Training MNIST CNN with BlackHole...")
print("="*60)

epochs = 5

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    
    print(f"Epoch {epoch}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print("-"*60)

print("\n✅ BlackHole MNIST Training Complete!")

# ============================================================
# 6. SAMPLE PREDICTIONS
# ============================================================

print("\n🔮 Sample Predictions:")
print("="*60)

model.eval()
data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)

with torch.no_grad():
    output = model(data[:5])
    pred = output.argmax(dim=1)

print("Actual:  ", target[:5].cpu().numpy())
print("Predicted:", pred.cpu().numpy())