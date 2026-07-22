import json
import warnings
import chromadb
import numpy as np
import faiss
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb_fault"

class FaultRAG:
    def __init__(self, embed_model, llm_call_fn):
        self.embed_model = embed_model
        self._llm = llm_call_fn
        self._load_index()

    def _load_index(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.client.get_collection("fault_standards")

    def _embed(self, query: str) -> list:
        vec = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        return vec[0].tolist()

    def search(self, query: str, top_k: int = 3) -> list:
        query_vec = self._embed(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            print(f"[FaultRAG] Search query failed: {e}. Reloading index and retrying...")
            self._load_index()
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][idx]
                doc = results["documents"][0][idx]
                distance = results["distances"][0][idx]

                chunk = {
                    "text": doc,
                    "page": int(metadata.get("page", 0)),
                    "source": metadata.get("source", "과실도표"),
                    "score": float(1.0 - distance)
                }
                retrieved.append(chunk)
        return retrieved

    def answer(self, query: str, context_chunks: list = None, top_k: int = 3) -> dict:
        if context_chunks is None:
            context_chunks = self.search(query, top_k=top_k)

        context = "\n\n".join([
            f"[과실도표 page {c.get('page','')}]\n{c['text'][:1500]}"
            for c in context_chunks
        ])

        system = "자동차사고 과실비율 판단 전문가. 아래 과실도표 인정기준만을 근거로 질문에 답하세요. JSON만 출력."
        user = f"""[질문]
{query}

[관련 과실도표 기준]
{context}

JSON만 출력:
{{
  "accident_type": "식별된 사고 유형 (예: 후진 사고, 횡단보도 보행자 사고 등)",
  "fault_code": "해당 도표/도표번호 (예: 도표 244, 차12-1 등)",
  "base_fault": {{ "피보험차": 기본과실비율_숫자, "상대차": 기본과실비율_숫자 }},
  "modifiers": [
    {{ "factor": "수정요소 요약", "delta": 가감비율_숫자_양수또는음수, "target": "피보험차" 또는 "상대차", "basis": "적용 근거" }}
  ],
  "final_fault_ratio": "최종 과실비율 (예: 피보험차 80% : 상대차 20%)"
}}"""
        raw = self._llm(system, user)
        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
        except Exception:
            result = {
                "accident_type": "판단 불가",
                "fault_code": "도표 확인 불가",
                "base_fault": {"피보험차": 0, "상대차": 0},
                "modifiers": [],
                "final_fault_ratio": "판단 불가"
            }

        result["chunks"] = context_chunks
        return result
