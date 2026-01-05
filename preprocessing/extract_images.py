import fitz  # PyMuPDF
import json
from pathlib import Path

PDF_DIR = Path("data/raw_pdfs")
OUTPUT = Path("data/images_metadata.json")

all_images = []

for pdf in PDF_DIR.glob("*.pdf"):
    doc = fitz.open(pdf)
    for page_num in range(len(doc)):
        images = doc[page_num].get_images(full=True)
        if images:
            all_images.append({
                "source_pdf": pdf.name,
                "page": page_num + 1,
                "description": "Image or chart related to document content"
            })

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_images, f, indent=2)

print("Image and chart metadata extracted.")
