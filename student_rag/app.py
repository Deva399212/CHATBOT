import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load model and index
model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"
PDF_DIR = "data/pdfs"

st.set_page_config(page_title="Student Question Paper Finder")
st.title("📚 Student Search Assistant")
st.markdown("Type your query to find question papers and notes!")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load chunk metadata
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# Search function
def search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

# UI input
query = st.text_input("🔎 Enter your query")

if query:
    with st.spinner("Searching..."):
        results = search(query)

    st.subheader("🔍 Top Results")
    if results:
        for res in results:
            st.markdown(f"**📄 Source:** `{res['source']}`")
            st.markdown(f"📝 **Excerpt:** {res['text'][:500]}...")
            pdf_path = os.path.join(PDF_DIR, res['source'])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(label="⬇️ Download PDF", data=f, file_name=res['source'], mime="application/pdf")
            st.markdown("---")
    else:
        st.warning("No results found. Try a different query.")
