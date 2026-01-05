from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langdetect import detect

from vector_store.search import search  # reuse your Phase 5 logic


# ----------------------------
# Load local LLM
# ----------------------------
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ----------------------------
# Helper: format answer nicely
# ----------------------------
def format_answer(text):
    lines = text.split("\n")
    points = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Keep numbered points if model generates them
        if line[0].isdigit():
            points.append(line)
        else:
            points.append(f"- {line}")

    return "\n".join(points)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ----------------------------
# Helper: remove prompt leakage
# ----------------------------
def strip_prompt_leakage(text):
    blacklist = [
        "You are an enterprise document assistant",
        "Answer the question strictly",
        "Follow these rules",
        "SYSTEM:",
        "RULES:",
        "USER QUESTION:",
        "ASSISTANT ANSWER:",
        "Question:",
    ]

    for phrase in blacklist:
        text = text.replace(phrase, "")

    return text.strip()


# ----------------------------
# Helper: detect nonsense output
# ----------------------------
def is_nonsense(text):
    words = text.split()
    if len(words) < 3:
        return True
    # repeated single word (clock clock clock)
    if len(set(words)) <= 2:
        return True
    return False

# ----------------------------
# Main RAG function
# ----------------------------
def generate_answer(query, top_k=3):
    query_lang = detect_language(query)

    retrieved = search(query, top_k=top_k)

    context = "\n\n".join([r["text"] for r in retrieved])

    prompt = f"""
You are an enterprise document assistant.
Answer strictly from the context.

Context:
{context}

Question:
{query}

Answer in bullet points:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=700
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_len:]

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = strip_prompt_leakage(answer)

    # 🔹 Confidence calculation
    avg_score = sum(r["score"] for r in retrieved) / len(retrieved)
    if avg_score < 0.3:
        confidence = "High"
    elif avg_score < 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "answer": format_answer(answer),
        "confidence": confidence,
        "language": query_lang,
        "sources": retrieved
    }


# ----------------------------
# CLI test (optional)
# ----------------------------
if __name__ == "__main__":
    query = input("Ask a question: ")
    answer = generate_answer(query)

    print("\n--- Generated Answer ---\n")
    print(answer)
