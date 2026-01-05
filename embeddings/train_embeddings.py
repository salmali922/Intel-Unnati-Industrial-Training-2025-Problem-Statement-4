from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path

# Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = Path("data/chunks/size_256")
MODEL_SAVE_PATH = Path("models/embedding_model")
if not MODEL_SAVE_PATH.exists():
    MODEL_SAVE_PATH.mkdir(parents=True)


# Create training examples
train_examples = []

files = list(DATA_DIR.glob("*.txt"))

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Self-supervised: positive pair = same text twice
    train_examples.append(InputExample(texts=[text, text]))

# DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=50,
    show_progress_bar=True
)

# Save model
model.save(str(MODEL_SAVE_PATH))

print("Embedding model training completed and saved.")
