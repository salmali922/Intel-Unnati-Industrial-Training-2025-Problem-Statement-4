import pdfplumber
from pathlib import Path
from tqdm import tqdm
from preprocessing.ocr_fallback import ocr_pdf


RAW_PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/extracted_text")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


def process_all_pdfs():
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            print(f"OCR applied to {pdf_file.name}")
            text = ocr_pdf(pdf_file)

        output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    process_all_pdfs()
