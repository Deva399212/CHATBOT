# preprocess_batch.py
import os
import pickle
import numpy as np
import faiss
from utils.embedding import extract_text, chunk_text, embed_chunks

PDF_DIR = "data/pdfs"
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

chunks_all = []
metadata_all = []
dimension = 384  # for MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

# Loop through all PDFs
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"📄 Processing {filename}")

        text = extract_text(pdf_path)
        chunks = chunk_text(text)

        embeddings = embed_chunks(chunks)

        # Add embeddings to FAISS
        index.add(np.array(embeddings))

        # Save text + metadata
        for chunk in chunks:
            metadata_all.append({
                "text": chunk,
                "source": filename
            })

# Save the FAISS index
faiss.write_index(index, INDEX_PATH)

# Save metadata chunks
with open(CHUNKS_PATH, 'wb') as f:
    pickle.dump(metadata_all, f)

print("✅ Finished! Index and chunks saved.")
