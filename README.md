# Intel-Unnati-Industrial-Training-2025-Problem-Statement-4
# 📄 Enterprise PDF Knowledge Assistant  
---

## 🚀 Project Overview

The **Enterprise PDF Knowledge Assistant** is an AI-powered system that converts unstructured enterprise PDF documents — including scanned images, charts, manuals, and policies — into a **searchable knowledge base**. Users can query the system in **natural language** (English, Hindi, Bengali), and get precise, factual answers grounded strictly in the provided documents.

This repository contains the **full source code**, demo artifacts, performance benchmarks, and documentation for the project.

---

## ✨ Key Features

### 📄 Data Extraction
- Reads digital text from PDF
- Applies **OCR (Tesseract)** for scanned PDFs and images
- Extracts text from charts and diagrams

### 🧠 Knowledge Retrieval
- Chunk-based semantic embedding
- Vector similarity search using **FAISS**
- Automatic language detection (English, Hindi, Bengali)

### 🔎 Answer Generation
- **Retrieval-Augmented Generation (RAG)**
- Answers grounded strictly from uploaded documents
- Includes **confidence score**

### 🖼️ Multimodal Querying
- Text query
- Image (chart/diagram) query
- Multilingual support

### 🧪 Evaluation & Benchmarks
- OCR accuracy metrics
- Retrieval time
- Hallucination control
- End-to-end response latency
- Comparative analysis

---

## 📂 Repository Structure

```plaintext
Intel-Unnati-Industrial-Training-2025-Problem-Statement-4/
│
├── src/                            # Main source code
│   ├── app.py
│   ├── preprocessing/
│   ├── rag/
│   ├── vector_store/
│   ├── vision/
│   ├── embeddings/
│   ├── data/
│   └── requirements.txt
│
├── demo/                           # Demo video or link
│   └── Enterprise_PDF_Knowledge_Assistant_Demo.mp4
│
├── benchmarks/                     # Performance evaluation
│   ├── Performance_Evaluation.md
│   └── screenshots/
│
├── report/                         # Final written report
│   └── Project_Report.pdf
│
├── .gitignore
└── README.md
```
---

### 🛠️ Technologies Used


| Category         | Tools                    |
| ---------------- | ------------------------ |
| Language         | Python                   |
| OCR              | Tesseract OCR            |
| Semantic Search  | Sentence Transformers    |
| Vector DB        | FAISS                    |
| Language Model   | HuggingFace Transformers |
| UI               | Streamlit                |
| PDF Parsing      | pdfplumber, camelot      |
| Image Processing | OpenCV, PIL              |


### 🧪 How to Run the Project (Quick Start)

1️⃣ Clone the Repo
git clone https://github.com/salmali922/Intel-Unnati-Industrial-Training-2025-Problem-Statement-4.git
cd Intel-Unnati-Industrial-Training-2025-Problem-Statement-4

2️⃣ Install Dependencies
pip install -r src/requirements.txt

3️⃣ Run the App
streamlit run src/app.py

Once started, navigate to:
http://localhost:8501

### 💬 Demo Video & Showcase

🎥 The working demo video is in the demo/ folder.

If the file is too large, a link to watch the demo is provided there.
Drive link : https://drive.google.com/file/d/1UH5w4pWI3k08w0FN4Q-k7cpFMNt0XdJR/view?usp=sharing

### 📊 Performance Evaluation

Detailed performance evaluation including OCR accuracy, retrieval precision, and response times is available in:
benchmarks/Performance_Evaluation.md


Screenshots demonstrating the pipeline and metrics are provided under:

benchmarks/screenshots/




