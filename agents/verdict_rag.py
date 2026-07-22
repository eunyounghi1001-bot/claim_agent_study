"""
판결문 RAG Agent (ChromaDB 버전)
- ChromaDB에서 유사 판결문 검색
- 검색 결과를 LLM이 해석해서 자연어 답변 생성
"""

import json
import warnings
import chromadb
import numpy as np
import faiss
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb"

class VerdictRAG:
    def __init__(self, embed_model, llm_call_fn, chroma_client=None):
        self.embed_model = embed_model
        self._llm = llm_call_fn
        self.client = chroma_client
        self._load_index()

    def _load_index(self):
        if self.client is None:
            self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.client.get_collection("verdicts")

    def _embed(self, query: str) -> list:
        vec = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        return vec[0].tolist()

    def _calculate_score(self, query: str, chunk_text: str) -> float:
        query_vec = self.embed_model.encode([query]).astype("float32")
        chunk_vec = self.embed_model.encode([chunk_text]).astype("float32")
        faiss.normalize_L2(query_vec)
        faiss.normalize_L2(chunk_vec)
        return float(np.dot(query_vec[0], chunk_vec[0]))

    def search(self, query: str, top_k: int = 5) -> list:
        query_vec = self._embed(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k * 3, # Query a bit more to deduplicate cases
            include=["metadatas", "documents", "distances"]
        )

        retrieved, seen = [], set()
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][idx]
                doc = results["documents"][0][idx]
                distance = results["distances"][0][idx]
                case_num = metadata.get("case_num", "")

                if case_num not in seen:
                    seen.add(case_num)
                    keywords_str = metadata.get("keywords", "")
                    keywords = keywords_str.split(",") if keywords_str else []

                    chunk = {
                        "text": doc,
                        "full_text": metadata.get("full_text", ""),
                        "case_name": metadata.get("case_name", ""),
                        "case_num": case_num,
                        "court": metadata.get("court", ""),
                        "date": metadata.get("date", ""),
                        "issue": metadata.get("issue", ""),
                        "summary": metadata.get("summary", ""),
                        "keywords": keywords,
                        "filename": metadata.get("filename", ""),
                        "source": metadata.get("source", "판결문"),
                        "score": float(1.0 - distance)
                    }
                    retrieved.append(chunk)
                if len(retrieved) >= top_k:
                    break
        return retrieved

    def search_by_triggers(self, triggers: list, top_k: int = 3) -> list:
        query = " ".join(triggers) + " 자동차보험 면책 보상"
        query_vec = self._embed(query)

        # Retrieve a candidate set (e.g. 50 cases) and filter by keywords in Python
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=50,
            include=["metadatas", "documents", "distances"]
        )

        retrieved, seen = [], set()
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][idx]
                doc = results["documents"][0][idx]
                distance = results["distances"][0][idx]
                case_num = metadata.get("case_num", "")

                if case_num not in seen:
                    keywords_str = metadata.get("keywords", "")
                    keywords = keywords_str.split(",") if keywords_str else []

                    has_trigger = any(t in keywords for t in triggers)
                    if has_trigger:
                        seen.add(case_num)
                        chunk = {
                            "text": doc,
                            "full_text": metadata.get("full_text", ""),
                            "case_name": metadata.get("case_name", ""),
                            "case_num": case_num,
                            "court": metadata.get("court", ""),
                            "date": metadata.get("date", ""),
                            "issue": metadata.get("issue", ""),
                            "summary": metadata.get("summary", ""),
                            "keywords": keywords,
                            "filename": metadata.get("filename", ""),
                            "source": metadata.get("source", "판결문"),
                            "score": float(1.0 - distance)
                        }
                        retrieved.append(chunk)
                        if len(retrieved) >= top_k:
                            return retrieved

        if len(retrieved) < top_k:
            for chunk in self.search(query, top_k=top_k):
                if chunk.get("case_num") not in seen:
                    retrieved.append(chunk)
                    seen.add(chunk["case_num"])
                    if len(retrieved) >= top_k:
                        break
        return retrieved[:top_k]

    def answer(self, query: str, context_chunks: list = None, top_k: int = 3) -> dict:
        if context_chunks is None:
            context_chunks = self.search(query, top_k=top_k)

        context = "\n\n".join([
            f"[{c.get('case_num','')} {c.get('court','')} {c.get('date','')}]\n"
            f"판시: {c.get('issue','')[:300]}\n"
            f"요지: {c.get('summary','')[:300]}"
            for c in context_chunks
        ])

        system = "자동차보험 판례 전문가. 아래 판결문만을 근거로 질문에 답하세요. JSON만 출력."
        user = f"""[질문]
{query}

[관련 판결문]
{context}

JSON만 출력:
{{
  "answer": "판례 기반 답변 (2~3문장)",
  "case_nums": ["근거 판례번호 목록"],
  "fault_ratio": "과실비율 (예: 좌회전차 70% : 직진차 30%, 언급 없으면 빈 문자열)"
}}"""
        raw = self._llm(system, user)
        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
        except Exception:
            result = {"answer": raw, "case_nums": [], "fault_ratio": ""}

        result["chunks"] = context_chunks
        return result
