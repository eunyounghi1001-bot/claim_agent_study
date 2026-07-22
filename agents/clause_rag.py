"""
약관 RAG Agent (ChromaDB 버전)
- ChromaDB에서 관련 약관 조항 검색
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

class ClauseRAG:
    def __init__(self, embed_model, llm_call_fn, chroma_client=None):
        self.embed_model = embed_model
        self._llm = llm_call_fn
        self.client = chroma_client
        self._load_index()

    def _load_index(self):
        if self.client is None:
            self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.client.get_collection("clauses")

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

    def search(self, query: str, top_k: int = 5, exemption_only: bool = False) -> list:
        query_vec = self._embed(query)
        where_filter = {}
        if exemption_only:
            where_filter = {"is_exemption": True}

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["metadatas", "documents", "distances"]
        )

        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][idx]
                doc = results["documents"][0][idx]
                distance = results["distances"][0][idx]

                coverage_str = metadata.get("coverage", "")
                coverage = coverage_str.split(",") if coverage_str else []

                chunk = {
                    "text": doc,
                    "article": metadata.get("article", ""),
                    "title": metadata.get("title", ""),
                    "page": int(metadata.get("page", 0)),
                    "coverage": coverage,
                    "is_exemption": bool(metadata.get("is_exemption", False)),
                    "source": metadata.get("source", "약관"),
                    "score": float(1.0 - distance) # Cosine similarity (1 - distance)
                }
                retrieved.append(chunk)
        return retrieved

    def search_by_triggers(self, triggers: list) -> list:
        query = " ".join(triggers) + " 보상하지 않는 손해 면책"
        trigger_article_map = {
            "음주운전":     ["제11조", "제23조"],
            "무면허":       ["제11조", "제23조"],
            "무면허운전":   ["제11조", "제23조"],
            "마약약물":     ["제11조"],
            "마약약물운전": ["제11조"],
            "조치의무위반": ["제11조"],
            "고의":         ["제5조", "제8조", "제14조", "제19조", "제23조"],
            "유상운송":     ["제8조", "제14조", "제19조", "제23조"],
            "가족피해":     ["제8조"],
            "무단운전":     ["제8조"],
            "절취운전":     ["제8조"],
            "운전자범위":   ["제8조"],
            "사기":         ["제23조"],
            "사기횡령":     ["제23조"],
            "고지의무":     ["제44조", "제53조"],
        }
        
        target_arts = []
        for trigger in triggers:
            target_arts.extend(trigger_article_map.get(trigger, []))
            
        retrieved = []
        if target_arts:
            target_arts = list(set(target_arts))
            results = self.collection.query(
                query_embeddings=[self._embed(query)],
                n_results=100,
                where={"article": {"$in": target_arts}},
                include=["metadatas", "documents", "distances"]
            )
            seen_articles = set()
            if results and results["ids"] and results["ids"][0]:
                for idx in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][idx]
                    doc = results["documents"][0][idx]
                    distance = results["distances"][0][idx]
                    art = metadata.get("article", "")
                    if art not in seen_articles:
                        seen_articles.add(art)
                        coverage_str = metadata.get("coverage", "")
                        coverage = coverage_str.split(",") if coverage_str else []
                        
                        chunk = {
                            "text": doc,
                            "article": art,
                            "title": metadata.get("title", ""),
                            "page": int(metadata.get("page", 0)),
                            "coverage": coverage,
                            "is_exemption": bool(metadata.get("is_exemption", False)),
                            "source": metadata.get("source", "약관"),
                            "score": float(1.0 - distance)
                        }
                        retrieved.append(chunk)
                        
        if not retrieved:
            retrieved = self.search(query, top_k=5, exemption_only=True)
        return retrieved

    def answer(self, query: str, context_chunks: list = None, top_k: int = 5) -> dict:
        if context_chunks is None:
            context_chunks = self.search(query, top_k=top_k)

        context = "\n\n".join([
            f"[{c['article']} {c.get('title','')}]\n{c['text'][:400]}"
            for c in context_chunks
        ])

        system = "자동차보험 약관 전문가. 아래 약관 조항만을 근거로 질문에 답하세요. JSON만 출력."
        user = f"""[질문]
{query}

[관련 약관 조항]
{context}

JSON만 출력:
{{
  "answer": "약관 조항 기반 답변 (2~3문장)",
  "articles": ["근거 조항 번호 목록"],
  "is_exempt": true 또는 false
}}"""
        raw = self._llm(system, user)
        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
        except Exception:
            result = {"answer": raw, "articles": [], "is_exempt": False}

        result["chunks"] = context_chunks
        return result
