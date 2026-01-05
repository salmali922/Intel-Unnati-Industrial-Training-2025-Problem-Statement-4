import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_images(query, threshold=0.45, top_k=3):
    with open("data/images_metadata.json", "r", encoding="utf-8") as f:
        images = json.load(f)

    if not images:
        return []

    query_emb = model.encode([query])

    relevant = []

    for img in images:
        context = img.get("context_text", "")
        if not context.strip():
            continue

        context_emb = model.encode([context])
        score = cosine_similarity(query_emb, context_emb)[0][0]

        if score >= threshold:
            img["score"] = float(score)
            relevant.append(img)

    relevant.sort(key=lambda x: x["score"], reverse=True)
    return relevant[:top_k]
