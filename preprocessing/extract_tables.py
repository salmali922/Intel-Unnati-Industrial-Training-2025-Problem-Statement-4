import camelot
import json
from pathlib import Path

PDF_DIR = Path("data/raw_pdfs")
OUTPUT_FILE = Path("data/tables/tables.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

all_tables = []

for pdf in PDF_DIR.glob("*.pdf"):
    try:
        tables = camelot.read_pdf(str(pdf), pages="1-end")
        for i, table in enumerate(tables):
            all_tables.append({
                "source_pdf": pdf.name,
                "table_id": i,
                "rows": table.df.values.tolist()
            })
    except Exception as e:
        print(f"Skipping {pdf.name}: {e}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_tables, f, indent=2)

print("Table extraction completed.")
