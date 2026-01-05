import spacy
from pathlib import Path
from tqdm import tqdm
import re

# Load English NLP model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

INPUT_DIR = Path("data/extracted_text")
OUTPUT_DIR = Path("data/cleaned_text")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase
    text = text.lower()

    return text.strip()


def preprocess_document(text):
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if len(token.text) < 3:
            continue
        tokens.append(token.lemma_)

    return " ".join(tokens)


def process_all_documents():
    files = list(INPUT_DIR.glob("*.txt"))

    for file_path in tqdm(files, desc="Preprocessing text"):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)
        processed = preprocess_document(cleaned)

        output_file = OUTPUT_DIR / file_path.name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(processed)


if __name__ == "__main__":
    process_all_documents()
