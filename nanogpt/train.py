import torch
import torch.nn as nn
from model import MiniGPT

# Load data
with open("data/nanogpt_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

data = encode(text)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

block_size = 64
batch_size = 32

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Training loop
for step in range(2000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

print("NanoGPT training completed.")
