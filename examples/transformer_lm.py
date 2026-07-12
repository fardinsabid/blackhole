"""
BlackHole Optimizer - Custom Transformer Language Model Training Demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from blackhole import BlackHole
import math

# ============================================================
# 1. SIMPLE TRANSFORMER LANGUAGE MODEL
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

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
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_layers=4,
        max_seq_len=128,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, mask=None):
        x = self.token_embed(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

# ============================================================
# 2. SAMPLE DATASET
# ============================================================

class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_length=128):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.vocab.encode(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.vocab.pad_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

# Simple vocabulary
class SimpleVocab:
    def __init__(self):
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self._token_to_id = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3
        }
        self._id_to_token = {
            0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'
        }
        self.next_id = 4
    
    def add_word(self, word):
        if word not in self._token_to_id:
            self._token_to_id[word] = self.next_id
            self._id_to_token[self.next_id] = word
            self.next_id += 1
    
    def build_vocab(self, texts):
        for text in texts:
            for word in text.split():
                self.add_word(word.lower())
    
    def encode(self, text):
        return [self._token_to_id.get(word.lower(), self.unk_id) for word in text.split()]
    
    def decode(self, ids):
        return ' '.join([self._id_to_token.get(id, '<unk>') for id in ids])
    
    @property
    def vocab_size(self):
        return len(self._token_to_id)

# Training data
training_texts = [
    "The universe is expanding at an accelerating rate.",
    "Black holes are regions of spacetime where gravity is so strong.",
    "Dark matter makes up approximately 85 percent of the matter in the universe.",
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
# 3. BUILD VOCABULARY
# ============================================================

vocab = SimpleVocab()
vocab.build_vocab(training_texts)
print(f"📚 Vocabulary size: {vocab.vocab_size}")

# ============================================================
# 4. DATALOADER
# ============================================================

dataset = TextDataset(training_texts, vocab, max_length=32)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ============================================================
# 5. MODEL & OPTIMIZER
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerLM(
    vocab_size=vocab.vocab_size,
    embed_dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=3,
    max_seq_len=32,
    dropout=0.1
)
model.to(device)

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

criterion = nn.CrossEntropyLoss()

# ============================================================
# 6. TRAINING LOOP
# ============================================================

def create_causal_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)

print("\n🚀 Training Transformer LM with BlackHole...")
print("="*60)

epochs = 20

for epoch in range(1, epochs + 1):
    total_loss = 0
    steps = 0
    
    for batch in dataloader:
        input_ids = batch.to(device)
        
        # Create causal mask for autoregressive language modeling
        mask = create_causal_mask(input_ids.size(1), device)
        
        # Forward pass
        logits = model(input_ids, mask)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = criterion(
            shift_logits.view(-1, vocab.vocab_size),
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
    
    avg_loss = total_loss / steps
    perplexity = math.exp(avg_loss)
    
    print(f"Epoch {epoch}/{epochs}:")
    print(f"  Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    print("-"*60)

print("\n✅ BlackHole Transformer LM Training Complete!")

# ============================================================
# 7. GENERATE SAMPLE TEXT
# ============================================================

print("\n🔮 Generating sample text...")
print("="*60)

model.eval()
prompt = "The universe is"

# Tokenize prompt
prompt_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    # Autoregressive generation
    for _ in range(20):
        # Get logits for current sequence
        mask = create_causal_mask(prompt_ids.size(1), device)
        logits = model(prompt_ids, mask)
        
        # Get next token prediction
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to sequence
        prompt_ids = torch.cat([prompt_ids, next_token], dim=1)

generated_text = vocab.decode(prompt_ids[0].cpu().numpy())
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

print("\n✅ BlackHole Transformer LM Demo Complete!")