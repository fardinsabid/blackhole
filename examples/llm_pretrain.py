"""
BlackHole Optimizer - LLM Pretraining from Scratch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from blackhole import BlackHole
import math

# ============================================================
# 1. SIMPLE TRANSFORMER MODEL FROM SCRATCH
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.embed(input_ids)
        x = x + self.pos_embed(positions)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

# ============================================================
# 2. SAMPLE DATASET
# ============================================================

class PretrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze()

# Training data
training_texts = [
    "The universe is expanding at an accelerating rate.",
    "Black holes are regions of spacetime where gravity is so strong.",
    "Dark matter makes up approximately 85% of the matter in the universe.",
    "Quantum entanglement allows particles to be connected across vast distances.",
    "The cosmic microwave background radiation is evidence of the Big Bang.",
    "Neural networks are inspired by the structure of the human brain.",
    "Transformer architecture revolutionized natural language processing.",
    "Gradient descent is the backbone of modern deep learning.",
    "The attention mechanism allows models to focus on relevant information.",
    "Large language models are trained on massive text corpora.",
    "Deep learning has transformed computer vision and natural language processing.",
    "Reinforcement learning enables agents to learn through interaction.",
    "The future of AI lies in general intelligence and reasoning.",
    "Data is the new oil in the era of artificial intelligence.",
    "Ethics and fairness are crucial considerations in AI development.",
    "The singularity refers to the hypothetical future emergence of superintelligence.",
    "GPT-3 demonstrated the power of scaling language models.",
    "Self-attention is the key innovation in transformer architectures.",
    "Transfer learning allows models to leverage pre-trained knowledge.",
    "The loss landscape of neural networks is highly non-convex.",
]

# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

print(f"📚 Vocabulary size: {vocab_size}")

# ============================================================
# 4. DATALOADER
# ============================================================

dataset = PretrainingDataset(training_texts, tokenizer, max_length=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ============================================================
# 5. MODEL
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=4,
    max_seq_len=128
)
model.to(device)

print(f"🔥 Using device: {device}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 6. BLACKHOLE OPTIMIZER
# ============================================================

optimizer = BlackHole(
    model.parameters(),
    lr=1e-3,                # Higher LR for training from scratch
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
# 7. TRAINING LOOP
# ============================================================

criterion = nn.CrossEntropyLoss()
epochs = 10

print("\n🚀 Starting LLM Pretraining from scratch with BlackHole...")
print("="*60)

for epoch in range(epochs):
    total_loss = 0
    steps = 0
    
    for batch in dataloader:
        input_ids = batch.to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = criterion(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 10 == 0:
            print(f"  Step {steps}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / steps
    perplexity = math.exp(avg_loss)
    print(f"\n📊 Epoch {epoch+1}/{epochs}:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")
    print("-"*60)

# ============================================================
# 8. GENERATE SAMPLE TEXT
# ============================================================

print("\n🔮 Generating sample text...")
print("="*60)

model.eval()
prompt = "The future of"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    # Autoregressive generation
    for _ in range(50):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

print("\n✅ BlackHole LLM Pretraining Demo Complete!")