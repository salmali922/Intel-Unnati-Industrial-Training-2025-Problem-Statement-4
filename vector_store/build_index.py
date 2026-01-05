import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNK_DIR = Path("data/chunks/size_256")
MODEL_PATH = Path("models/embedding_model")
INDEX_DIR = Path("vector_store")
INDEX_DIR.mkdir(exist_ok=True)

# Load trained embedding model
model = SentenceTransformer(str(MODEL_PATH))

texts = []
metadata = []

for file in CHUNK_DIR.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()
        texts.append(text)
        metadata.append(file.name)

print(f"Total chunks: {len(texts)}")

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

with open(INDEX_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Vector index built and saved successfully.")
