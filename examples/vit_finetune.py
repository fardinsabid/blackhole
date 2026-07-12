"""
BlackHole Optimizer - Vision Transformer (ViT) Fine-Tuning on CIFAR-10 Demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from blackhole import BlackHole

# ============================================================
# 1. LOAD DATA
# ============================================================

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# ============================================================
# 2. LOAD PRETRAINED VIT
# ============================================================

model_name = "google/vit-base-patch16-224-in21k"

# Load processor for preprocessing
processor = ViTImageProcessor.from_pretrained(model_name)

# Load model with custom classifier head for CIFAR-10 (10 classes)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=10,
    ignore_mismatched_sizes=True
)

# ============================================================
# 3. DEVICE & MODEL INFO
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"🔥 Using device: {device}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Freeze backbone (optional - comment out to fine-tune entire model)
# for param in model.vit.parameters():
#     param.requires_grad = False

# Only train classifier head (optional)
# for param in model.classifier.parameters():
#     param.requires_grad = True

# ============================================================
# 4. BLACKHOLE OPTIMIZER
# ============================================================

optimizer = BlackHole(
    model.parameters(),
    lr=2e-5,                # Lower LR for fine-tuning pretrained models
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

criterion = nn.CrossEntropyLoss()

# ============================================================
# 5. TRAINING
# ============================================================

def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=data)
        loss = outputs.loss
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
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
            outputs = model(pixel_values=data)
            test_loss += outputs.loss.item()
            pred = outputs.logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# ============================================================
# 6. TRAINING LOOP
# ============================================================

print("\n🚀 Fine-Tuning ViT on CIFAR-10 with BlackHole...")
print("="*60)

epochs = 5

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test()
    
    print(f"Epoch {epoch}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print("-"*60)

print("\n✅ BlackHole ViT Fine-Tuning Complete!")

# ============================================================
# 7. SAMPLE PREDICTIONS
# ============================================================

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("\n🔮 Sample Predictions:")
print("="*60)

model.eval()
data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)

with torch.no_grad():
    outputs = model(pixel_values=data[:5])
    pred = outputs.logits.argmax(dim=1)

print("Actual:  ", [classes[t] for t in target[:5].cpu().numpy()])
print("Predicted:", [classes[p] for p in pred.cpu().numpy()])

print("\n✅ BlackHole ViT Demo Complete!")