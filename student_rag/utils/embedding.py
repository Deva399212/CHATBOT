# utils/embedding.py
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Load embedding model (you can replace with any other)
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split long text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    """Generate embeddings for each text chunk."""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings
