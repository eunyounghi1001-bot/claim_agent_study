import os
import pickle
import json
import chromadb
import faiss
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CLAUSE_INDEX_PATH = BASE_DIR / "index" / "clause_index.pkl"
VERDICT_INDEX_PATH = BASE_DIR / "index" / "verdict_index.pkl"
STANDARD_INDEX_PATH = BASE_DIR / "index" / "standard_index.pkl"
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb"

def migrate():
    print("=== ChromaDB Migration Started ===")
    
    # 1. Initialize Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # 2. Migrate clauses
    print("\nMigrating clauses...")
    with open(CLAUSE_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    idx = data["index"]
    chunks = data["chunks"]
    embeddings = idx.reconstruct_n(0, idx.ntotal).tolist()
    
    try:
        client.delete_collection("clauses")
    except Exception:
        pass
    clauses_col = client.create_collection("clauses", metadata={"hnsw:space": "cosine"})
    
    ids = [f"clause_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "article": c.get("article", ""),
            "title": c.get("title", ""),
            "page": int(c.get("page", 0)),
            "coverage": ",".join(c.get("coverage", [])), # list to comma-separated
            "is_exemption": bool(c.get("is_exemption", False)),
            "source": c.get("source", "약관")
        }
        metadatas.append(meta)
        
    clauses_col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print(f"  - Clauses migration completed! Added {len(chunks)} items.")
    
    # 3. Migrate verdicts
    print("\nMigrating verdicts...")
    with open(VERDICT_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    idx = data["index"]
    chunks = data["chunks"]
    embeddings = idx.reconstruct_n(0, idx.ntotal).tolist()
    
    try:
        client.delete_collection("verdicts")
    except Exception:
        pass
    verdicts_col = client.create_collection("verdicts", metadata={"hnsw:space": "cosine"})
    
    ids = [f"verdict_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "full_text": c.get("full_text", "")[:3000],
            "case_name": c.get("case_name", ""),
            "case_num": c.get("case_num", ""),
            "court": c.get("court", ""),
            "date": c.get("date", ""),
            "issue": c.get("issue", "")[:300] if c.get("issue") else "",
            "summary": c.get("summary", "")[:500] if c.get("summary") else "",
            "keywords": ",".join(c.get("keywords", [])), # list to comma-separated
            "filename": c.get("filename", ""),
            "source": c.get("source", "판결문")
        }
        metadatas.append(meta)
        
    # Batch add in chunks of 500 to avoid any limits
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        verdicts_col.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx]
        )
    print(f"  - Verdicts migration completed! Added {len(chunks)} items.")
    
    # 4. Migrate standards
    if STANDARD_INDEX_PATH.exists():
        print("\nMigrating standards...")
        with open(STANDARD_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        idx = data["index"]
        chunks = data["chunks"]
        embeddings = idx.reconstruct_n(0, idx.ntotal).tolist()
        
        try:
            client.delete_collection("standards")
        except Exception:
            pass
        standards_col = client.create_collection("standards", metadata={"hnsw:space": "cosine"})
        
        ids = [f"standard_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "accident_type": c.get("accident_type", ""),
                "standard_json": json.dumps(c.get("standard", {}))
            }
            metadatas.append(meta)
            
        standards_col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        print(f"  - Standards migration completed! Added {len(chunks)} items.")
        
    print("\n=== ChromaDB Migration Finished Successfully ===")

if __name__ == "__main__":
    migrate()
