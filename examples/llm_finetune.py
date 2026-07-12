"""
BlackHole Optimizer - LLM Fine-Tuning Demo
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from blackhole import BlackHole

# ============================================================
# 1. SAMPLE DATASET (Replace with your data)
# ============================================================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
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
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

# Sample training data
sample_texts = [
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
]

# ============================================================
# 2. LOAD MODEL
# ============================================================

model_name = "gpt2"  # or "microsoft/DialoGPT-medium", "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.train()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"🔥 Using device: {device}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 3. DATALOADER
# ============================================================

dataset = TextDataset(sample_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ============================================================
# 4. BLACKHOLE OPTIMIZER
# ============================================================

optimizer = BlackHole(
    model.parameters(),
    lr=5e-5,                # Typical LLM learning rate
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    G=0.01,                 # Gravitational constant (tune for LLMs)
    c=10.0,
    hbar=0.01,
    k_B=0.01,
    Lambda=0.001,
    alpha=0.05,
    spin=0.5,
    extra_dim_strength=0.01
)

# ============================================================
# 5. TRAINING LOOP
# ============================================================

criterion = nn.CrossEntropyLoss()
epochs = 3

print("\n🚀 Starting LLM Fine-Tuning with BlackHole...")
print("="*60)

for epoch in range(epochs):
    total_loss = 0
    steps = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (recommended for LLMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        print(f"  Step {steps}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / steps
    print(f"\n📊 Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.4f}")
    print("-"*60)

# ============================================================
# 6. GENERATE SAMPLE TEXT
# ============================================================

print("\n🔮 Generating sample text...")
print("="*60)

model.eval()
prompt = "The universe is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

print("\n✅ BlackHole LLM Fine-Tuning Demo Complete!")