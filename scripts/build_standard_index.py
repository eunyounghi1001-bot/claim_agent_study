"""
사고 유형별 판단 기준 데이터를 벡터 인덱스로 빌드.
python build_standard_index.py
"""
import json, pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_PATH  = Path("data/standards.json")
INDEX_PATH = Path("index/standard_index.pkl")

print("bge-m3 로딩 중...")
model = SentenceTransformer("BAAI/bge-m3")

with open(DATA_PATH, encoding="utf-8") as f:
    standards = json.load(f)

chunks = []
texts  = []

for s in standards:
    # 각 사고 유형을 검색 가능한 청크로 변환
    text = (
        f"{s['accident_type']} 사고 판단 기준. "
        f"{s['description']}. "
        f"핵심 쟁점: {', '.join(s['key_issues'])}. "
        f"검색어: {s['search_query']}"
    )
    # alias도 동일 청크로 추가 (검색 커버리지 확대)
    for alias in [s['accident_type']] + s.get('aliases', []):
        alias_text = f"{alias} " + text
        chunks.append({"standard": s, "accident_type": s['accident_type'], "text": alias_text})
        texts.append(alias_text)

print(f"청크 {len(chunks)}개 임베딩 중...")
vecs = model.encode(texts, show_progress_bar=True).astype("float32")
faiss.normalize_L2(vecs)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)

INDEX_PATH.parent.mkdir(exist_ok=True)
with open(INDEX_PATH, "wb") as f:
    pickle.dump({"index": index, "chunks": chunks}, f)

print(f"완료: {INDEX_PATH} ({len(chunks)}개 기준 저장)")
