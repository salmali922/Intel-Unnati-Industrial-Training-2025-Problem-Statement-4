from pathlib import Path

INPUT_DIR = Path("data/cleaned_text")
OUTPUT_FILE = Path("data/nanogpt_corpus.txt")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for file in INPUT_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            out.write(f.read())
            out.write("\n")

print("NanoGPT corpus prepared.")
