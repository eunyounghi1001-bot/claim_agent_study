import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
torch.set_num_threads(1)
import re
import json
import warnings
import fitz
import chromadb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
PDF_PATH = BASE_DIR / "raw_data" / "약관" / "약관" / "개인용자동차보험약관(2026.06.10).pdf"
VERDICT_DIR = BASE_DIR / "raw_data" / "판례" / "판례" / "판결문"
STANDARDS_PATH = BASE_DIR / "data" / "standards.json"
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb"

COVERAGE_MAP = {
    "대인배상I":     ["제3조", "제4조", "제5조"],
    "대인배상II":    ["제6조", "제7조", "제8조", "제9조", "제10조"],
    "대물배상":      ["제6조", "제7조", "제8조", "제9조", "제10조"],
    "사고부담금":    ["제11조"],
    "자기신체사고":  ["제12조", "제13조", "제14조", "제15조"],
    "무보험차상해":  ["제17조", "제18조", "제19조", "제20조"],
    "자기차량손해":  ["제21조", "제22조", "제23조", "제24조"],
}

EXEMPTION_MAP = {
    "대인배상I":    "제5조",
    "대인배상II":   "제8조",
    "대물배상":     "제8조",
    "사고부담금":   "제11조",
    "자기신체사고": "제14조",
    "무보험차상해": "제19조",
    "자기차량손해": "제23조",
}

# ── 1. 약관 조항 청킹 ───────────────────────────
def load_clause_chunks(pdf_path: Path) -> list:
    doc = fitz.open(str(pdf_path))
    full_text = ""
    page_offsets = []
    for i, page in enumerate(doc):
        offset = len(full_text)
        text = page.get_text()
        full_text += text
        page_offsets.append((offset, i + 1))

    def get_page(pos):
        page = 1
        for offset, pnum in page_offsets:
            if offset <= pos:
                page = pnum
            else:
                break
        return page

    pattern = re.compile(r'(제\d+조)\s*[\(（]?([^\)\n\r]{0,40})')
    matches = list(pattern.finditer(full_text))

    chunks = []
    for idx, m in enumerate(matches):
        art_num = m.group(1)
        art_title = m.group(2).strip()
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()

        if len(text) < 50 or len(text) > 8000:
            continue

        coverage = []
        for cov, arts in COVERAGE_MAP.items():
            if art_num in arts:
                coverage.append(cov)

        is_exemption = any(
            EXEMPTION_MAP.get(c) == art_num for c in coverage
        ) or "보상하지" in art_title

        chunks.append({
            "text": text,
            "article": art_num,
            "title": art_title,
            "page": get_page(start),
            "coverage": ",".join(coverage), # Stringified for Chroma metadata
            "is_exemption": is_exemption,
            "source": "약관",
        })
    return chunks

# ── 2. 판결문 청킹 ───────────────────────────────────
def parse_verdict_file(filepath: Path) -> dict | None:
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    def extract(pattern, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    case_name   = extract(r'사건명[:\s]+(.+)')
    case_num    = extract(r'사건번호[:\s]+(.+)')
    court       = extract(r'법원명[:\s]+(.+)')
    date        = extract(r'선고일자[:\s]+(\d+)')
    issue       = extract(r'\[판시사항\]\s*([\s\S]*?)(?=\[판결요지\]|\[참조조문\]|\[판례내용\]|$)')
    summary     = extract(r'\[판결요지\]\s*([\s\S]*?)(?=\[참조조문\]|\[판례내용\]|$)')

    fname = filepath.stem
    if not case_num:
        m = re.search(r'(\d{2,4}[다도가나]\d+)', fname)
        if m:
            case_num = m.group(1)

    keywords = []
    trigger_words = ["음주운전", "무면허", "무단운전", "유상운송", "고의", "절취",
                     "가족", "고지의무", "운전자범위", "사고부담금", "면책", "구상"]
    for kw in trigger_words:
        if kw in text:
            keywords.append(kw)

    search_text = f"{case_name} {issue} {summary}".strip()
    if len(search_text) < 30:
        search_text = text[:1000]

    return {
        "text": search_text,
        "full_text": text[:3000],
        "case_name": case_name,
        "case_num": case_num,
        "court": court,
        "date": date,
        "issue": issue[:300] if issue else "",
        "summary": summary[:500] if summary else "",
        "keywords": ",".join(keywords), # Stringified for Chroma metadata
        "filename": filepath.name,
        "source": "판결문",
    }

def load_verdict_chunks(verdict_dir: Path) -> list:
    chunks = []
    txt_files = list(verdict_dir.glob("*.txt"))
    print(f"  판결문 TXT: {len(txt_files)}개 파싱 중...")
    for f in txt_files:
        parsed = parse_verdict_file(f)
        if parsed and len(parsed["text"]) > 30:
            chunks.append(parsed)
    print(f"  파싱 완료: {len(chunks)}개")
    return chunks

# ── 3. 판단기준 청킹 ─────────────────────────────────
def load_standard_chunks(standards_path: Path) -> list:
    with open(standards_path, encoding="utf-8") as f:
        standards = json.load(f)
    chunks = []
    for s in standards:
        text = (
            f"{s['accident_type']} 사고 판단 기준. "
            f"{s['description']}. "
            f"핵심 쟁점: {', '.join(s['key_issues'])}. "
            f"검색어: {s['search_query']}"
        )
        for alias in [s['accident_type']] + s.get('aliases', []):
            alias_text = f"{alias} " + text
            chunks.append({
                "text": alias_text,
                "accident_type": s['accident_type'],
                "standard_json": json.dumps(s) # Stringified JSON
            })
    return chunks

# ── 4. Chroma 인코딩 & 저장 ────────────────────────
def populate_collection(collection, chunks, embed_model):
    texts = [c["text"] for c in chunks]
    print(f"  임베딩 생성 중... ({len(texts)}개)")
    
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = [e.tolist() for e in embeddings]
    
    ids = [f"id_{idx}" for idx in range(len(chunks))]
    metadatas = []
    documents = []
    
    for c in chunks:
        documents.append(c["text"])
        meta = {k: v for k, v in c.items() if k != "text"}
        metadatas.append(meta)
        
    print(f"  ChromaDB 저장 중...")
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
            documents=documents[i:end_idx]
        )
    print(f"  완료!")

if __name__ == "__main__":
    print("=== ChromaDB 인덱스 구축 시작 ===\n")
    
    print("[1/4] 임베딩 모델 로딩 중...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    print("      완료!\n")
    
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # 1. Clauses
    print("[2/4] 약관 조항 인덱싱 중...")
    try:
        client.delete_collection("clauses")
    except Exception:
        pass
    clauses_col = client.create_collection("clauses", metadata={"hnsw:space": "cosine"})
    clause_chunks = load_clause_chunks(PDF_PATH)
    populate_collection(clauses_col, clause_chunks, embed_model)
    print()
    
    # 2. Verdicts
    print("[3/4] 판결문 인덱싱 중...")
    try:
        client.delete_collection("verdicts")
    except Exception:
        pass
    verdicts_col = client.create_collection("verdicts", metadata={"hnsw:space": "cosine"})
    verdict_chunks = load_verdict_chunks(VERDICT_DIR)
    populate_collection(verdicts_col, verdict_chunks, embed_model)
    print()
    
    # 3. Standards
    print("[4/4] 판단기준 인덱싱 중...")
    try:
        client.delete_collection("standards")
    except Exception:
        pass
    standards_col = client.create_collection("standards", metadata={"hnsw:space": "cosine"})
    standard_chunks = load_standard_chunks(STANDARDS_PATH)
    populate_collection(standards_col, standard_chunks, embed_model)
    print()
    
    print("=== ChromaDB 구축 완료 ===")
