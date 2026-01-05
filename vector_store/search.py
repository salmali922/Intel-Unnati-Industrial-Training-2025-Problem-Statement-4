import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

INDEX_DIR = Path("vector_store")
CHUNK_DIR = Path("data/chunks/size_256")
MODEL_PATH = Path("models/embedding_model")

model = SentenceTransformer(str(MODEL_PATH))
index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        chunk_file = metadata[idx]
        with open(CHUNK_DIR / chunk_file, "r", encoding="utf-8") as f:
            text = f.read()

        results.append({
            "text": text,
            "chunk_id": chunk_file,
            "score": float(distances[0][rank])
        })

    return results
