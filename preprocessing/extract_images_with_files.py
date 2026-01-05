import fitz  # PyMuPDF
import json
from pathlib import Path

PDF_DIR = Path("data/raw_pdfs")
IMG_DIR = Path("data/images")
META_FILE = Path("data/images_metadata.json")

IMG_DIR.mkdir(parents=True, exist_ok=True)

all_images = []
img_count = 0

for pdf in PDF_DIR.glob("*.pdf"):
    doc = fitz.open(pdf)

    for page_num in range(len(doc)):
        images = doc[page_num].get_images(full=True)

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            img_name = f"{pdf.stem}_page{page_num+1}_{img_count}.{image_ext}"
            img_path = IMG_DIR / img_name

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            page_text = doc[page_num].get_text()

            all_images.append({
                 "image_file": str(img_path),
                 "source_pdf": pdf.name,
                 "page": page_num + 1,
                 "context_text": page_text[:1500],  # limit size
                 "description": "Extracted image or chart from document"
            })


            img_count += 1

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(all_images, f, indent=2)

print(f"Image extraction completed. {img_count} images saved.")
