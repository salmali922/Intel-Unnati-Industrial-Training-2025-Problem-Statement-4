import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

INDEX_DIR = Path("vector_store")
CHUNK_DIR = Path("data/chunks/size_256")

# Multilingual embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

with open(INDEX_DIR / "metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search_multilingual(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        with open(CHUNK_DIR / metadata[idx], "r", encoding="utf-8") as f:
            results.append(f.read())

    return results
