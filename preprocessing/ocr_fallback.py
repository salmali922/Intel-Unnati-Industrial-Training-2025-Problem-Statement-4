import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

# Set this path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

def ocr_pdf(pdf_path):
     images = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH
    )
    
     text = []

     for img in images:
        text.append(pytesseract.image_to_string(img))

     return "\n".join(text)
