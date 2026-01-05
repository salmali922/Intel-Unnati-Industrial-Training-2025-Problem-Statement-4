from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/cleaned_text")
OUTPUT_DIR = Path("data/chunks")

CHUNK_SIZES = [128, 256, 512]
OVERLAP_RATIO = 0.2  # 20% overlap


def chunk_tokens(tokens, chunk_size, overlap):
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) < chunk_size // 2:
            break
        chunks.append(chunk)

    return chunks


def process_all_files():
    files = list(INPUT_DIR.glob("*.txt"))

    for size in CHUNK_SIZES:
        overlap = int(size * OVERLAP_RATIO)
        size_dir = OUTPUT_DIR / f"size_{size}"
        size_dir.mkdir(parents=True, exist_ok=True)

        for file_path in tqdm(files, desc=f"Chunking size {size}"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = text.split()
            chunks = chunk_tokens(tokens, size, overlap)

            for idx, chunk in enumerate(chunks):
                chunk_text = " ".join(chunk)
                out_file = size_dir / f"{file_path.stem}_chunk_{idx}.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(chunk_text)


if __name__ == "__main__":
    process_all_files()
