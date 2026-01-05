import cv2
import pytesseract
import unicodedata
import re

def clean_ocr_text(text):
    # Step 1: normalize whitespace
    text = " ".join(text.split())

    # Step 2: remove phrase repetition (English)
    text = remove_repeated_phrases(text)

    # Step 3: remove excessive single-word repetition
    tokens = text.split()
    cleaned = []
    prev = None

    for token in tokens:
        if token != prev:
            cleaned.append(token)
        prev = token

    return " ".join(cleaned)


def remove_repeated_phrases(text, max_ngram=4):
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        found_repeat = False
        for n in range(max_ngram, 1, -1):
            if i + 2*n <= len(words):
                phrase1 = words[i:i+n]
                phrase2 = words[i+n:i+2*n]
                if phrase1 == phrase2:
                    result.extend(phrase1)
                    i += n
                    found_repeat = True
                    break
        if not found_repeat:
            result.append(words[i])
            i += 1

    return " ".join(result)


def remove_repeated_tokens(text):
    tokens = text.split()
    cleaned = []

    prev = None
    repeat_count = 0

    for token in tokens:
        if token == prev:
            repeat_count += 1
            if repeat_count < 2:
                cleaned.append(token)
        else:
            repeat_count = 0
            cleaned.append(token)
            prev = token

    return " ".join(cleaned)

def is_valid_indic_word(word):
    # Remove combining marks
    base = "".join(
        c for c in word if unicodedata.category(c) != "Mn"
    )

    # Very short or symbol-heavy tokens are noise
    if len(base) < 3:
        return False

    # Too many rare Unicode chars = OCR noise
    weird = sum(
        1 for c in base
        if not ("DEVANAGARI" in unicodedata.name(c, "") or c.isalpha())
    )

    return weird <= 1


def extract_chart_text(image_path, lang="eng+hin+ben"):
    """
    Extract text from charts / tables / diagrams
    Supports English, Hindi, and Bengali
    """

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")

    # 🔹 Upscale image (VERY important for Indic scripts)
    img = cv2.resize(
        img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )

    # 🔹 Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔹 Gentle preprocessing (better than thresholding for Hindi/Bengali)
    gray = cv2.medianBlur(gray, 3)

    # 🔹 Multilingual OCR
    text = pytesseract.image_to_string(
        gray,
        lang=lang,
        config="--psm 6"
    )

    return text


def chart_text_to_json(text):
    """
    Convert extracted chart text into structured JSON
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    return {
        "type": "chart_or_table",
        "extracted_lines": lines
    }
