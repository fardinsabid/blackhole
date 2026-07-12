"""
BlackHole Optimizer - Distributed Data Parallel (DDP) Training Demo
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from blackhole import BlackHole

# ============================================================
# 1. SIMPLE MODEL
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# ============================================================
# 2. DDP TRAINING FUNCTION
# ============================================================

def ddp_train(rank, world_size, epochs=5):
    """
    DDP training function run on each GPU.
    
    Args:
        rank: Current GPU rank
        world_size: Total number of GPUs
        epochs: Number of training epochs
    """
    
    # ============================================================
    # 2a. Initialize DDP
    # ============================================================
    
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12355',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"🔥 GPU {rank} initialized")
    
    # ============================================================
    # 2b. Load Data with Distributed Sampler
    # ============================================================
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Distributed sampler ensures each GPU gets different data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # ============================================================
    # 2c. Model & Optimizer
    # ============================================================
    
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[rank])
    
    # BlackHole optimizer (each GPU has its own optimizer state)
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
    # 2d. Training Loop
    # ============================================================
    
    print(f"🚀 GPU {rank}: Starting training...")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"GPU {rank} - Epoch {epoch}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print("-"*40)
    
    # ============================================================
    # 2e. Cleanup
    # ============================================================
    
    print(f"✅ GPU {rank}: Training complete!")
    dist.destroy_process_group()

# ============================================================
# 3. MAIN ENTRY POINT
# ============================================================

def main():
    """
    Main function to launch DDP training.
    """
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available! DDP requires multiple GPUs.")
        return
    
    world_size = torch.cuda.device_count()
    print(f"🔥 Found {world_size} GPUs")
    
    if world_size < 2:
        print("⚠️ Only 1 GPU found. Falling back to single GPU training.")
        print("   For DDP, use 2 or more GPUs.")
        
        # Fallback: Single GPU training
        device = torch.device('cuda')
        model = SimpleCNN().to(device)
        
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
        
        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        print("\n🚀 Single GPU Training...")
        print("="*50)
        
        for epoch in range(5):
            model.train()
            total_loss = 0
            correct = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / len(train_loader.dataset)
            
            print(f"Epoch {epoch+1}/5:")
            print(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print("-"*40)
        
        print("\n✅ Single GPU Training Complete!")
        return
    
    # ============================================================
    # 4. Launch DDP with multiple processes
    # ============================================================
    
    print(f"🚀 Launching DDP training with {world_size} GPUs...")
    print("="*50)
    
    mp.spawn(
        ddp_train,
        args=(world_size, 5),  # (world_size, epochs)
        nprocs=world_size,
        join=True
    )
    
    print("\n✅ DDP Training Complete!")

# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":
    # Set environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    main()