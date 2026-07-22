import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import warnings
import fitz
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
PDF_PATH = BASE_DIR / "raw_data" / "과실도표" / "과실도표" / "230630_자동차사고 과실비율 인정기준_최종 (1).pdf"
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb_fault"

def build_index():
    print("=== Fault Standards Indexing Started (Optimized) ===")
    
    # 1. Load PDF and filter pages
    doc = fitz.open(str(PDF_PATH))
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        # Only index pages that contain "기본 과실비율"
        if "기본 과실비율" not in text or len(text) < 100:
            continue
        chunks.append({
            "text": text,
            "page": i + 1,
            "source": "과실도표"
        })
        
    print(f"Filtered to {len(chunks)} relevant pages containing '기본 과실비율'.")
    
    if not chunks:
        print("No pages matched the filter criteria.")
        return
        
    # 2. Initialize Chroma client and collection
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        client.delete_collection("fault_standards")
    except Exception:
        pass
    collection = client.create_collection("fault_standards", metadata={"hnsw:space": "cosine"})
    
    # 3. Load Embedding Model
    print("Loading embedding model BAAI/bge-m3...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    
    # 4. Generate Embeddings & Save
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks (using batch_size=8 for CPU efficiency)...")
    embeddings = embed_model.encode(texts, batch_size=8, show_progress_bar=True)
    embeddings = [e.tolist() for e in embeddings]
    
    ids = [f"fault_{idx}" for idx in range(len(chunks))]
    metadatas = [{"page": c["page"], "source": c["source"]} for c in chunks]
    
    print("Saving to ChromaDB...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=texts[i:end_idx]
        )
        
    print("=== Fault Standards Indexing Completed Successfully ===")

if __name__ == "__main__":
    build_index()
