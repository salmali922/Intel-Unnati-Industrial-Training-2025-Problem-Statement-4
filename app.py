import streamlit as st
import json


from rag.rag_pipeline import generate_answer
from vector_store.search import search
from vector_store.search_multilingual import search_multilingual
from vector_store.image_relevance import get_relevant_images
from PIL import Image
from vision.chart_extraction import extract_chart_text, clean_ocr_text

st.info(
    "⚠️ This system answers strictly from uploaded documents. "
    "If information is missing, it will be clearly stated."
)


# Page config
st.set_page_config(
    page_title="Enterprise PDF Knowledge Assistant",
    layout="wide"
)

st.title("📄 Enterprise PDF Knowledge Assistant")

st.markdown(
    """
    🔍 **Search, analyze, and reason over enterprise documents**  
    Supports **text queries**, **image-based queries**, and **multilingual OCR**
    """
)

st.divider()
st.subheader("✨ Key Capabilities")

st.markdown("""
- 📄 Converts unstructured PDFs into searchable knowledge  
- 🔎 Uses semantic search with vector embeddings  
- 🧠 Retrieval-Augmented Generation (RAG) for accurate answers  
- 🌐 Multilingual OCR (English, Hindi, Bengali)  
- 📷 Image-based querying for charts and scanned documents  
""")
with st.sidebar:
    st.header("📌 Project Info")
    st.write("**Project:** Convert Enterprise PDFs into Searchable Knowledge")
    st.write("**Tech:** Python, OCR, RAG, Vector DB")
    st.write("**Features:** Text + Image Query")



# Input
query = st.text_input("🔍 Enter your question:")


result = generate_answer(query)

st.subheader("✅ AI Generated Answer")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📌 Answer")
    st.markdown(result["answer"])

with col2:
    st.markdown("### 📊 Metadata")
    st.write(f"**Detected Language:** {result['language']}")
    st.write(f"**Confidence Level:** {result['confidence']}")



st.subheader("📌 Source Evidence")

for src in result["sources"]:
    with st.expander(f"📄 {src['chunk_id']} (score: {src['score']:.2f})"):
        st.write(src["text"][:800])


st.divider()
st.subheader("🖼️ Relevant Images & Charts")

if query:
    relevant_images = get_relevant_images(query)

    if not relevant_images:
        st.info("No relevant images found for this query.")
    else:
        for img in relevant_images:
            st.markdown(
                f"📄 **{img['source_pdf']}** — Page {img['page']} "
                f"(relevance score: {img['score']:.2f})"
            )

            image = Image.open(img["image_file"])
            st.image(image, use_column_width=True)

            st.caption(img["description"])
            st.markdown("---")
st.divider()
st.subheader("📷 Image-based Query (Chart / Diagram)")
uploaded_image = st.file_uploader(
    "Upload an image (chart, diagram, scanned page)",
    type=["png", "jpg", "jpeg"]
)
if uploaded_image:
    import tempfile
    from vision.chart_extraction import extract_chart_text

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_image.read())
        image_path = tmp.name

    raw_text = extract_chart_text(image_path)
    extracted_text = clean_ocr_text(raw_text)

    st.subheader("📝 Extracted text from image")
    st.code(extracted_text)

    if extracted_text.strip():
     result = generate_answer(extracted_text)

    st.subheader("✅ AI Answer from Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 Answer")
        st.markdown(result["answer"])

    with col2:
        st.markdown("### 📊 Metadata")
        st.write(f"**Detected Language:** {result['language']}")
        st.write(f"**Confidence Level:** {result['confidence']}")

    st.subheader("📌 Source Evidence")
    for i, src in enumerate(result["sources"], 1):
        with st.expander(f"📄 Source {i} — {src['chunk_id']}"):
            st.write(src["text"][:600])

else:
    st.warning("Could not extract readable text from image.")

