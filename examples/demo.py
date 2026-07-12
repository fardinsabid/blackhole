"""
BlackHole Optimizer - Simple Usage Demo
"""

import torch
import torch.nn as nn
from blackhole import BlackHole

# 1. Create a simple model
model = nn.Linear(10, 1)

# 2. Initialize BlackHole optimizer with custom parameters
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

# 3. Dummy data
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# 4. Training step
criterion = nn.MSELoss()

optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
print("✅ BlackHole is working!")