"""
약관 RAG Agent
- FAISS 벡터 검색으로 관련 약관 조항 검색
- 검색 결과를 LLM이 해석해서 자연어 답변 생성
"""

import json
import pickle
import warnings

import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR          = Path(__file__).parent.parent
CLAUSE_INDEX_PATH = BASE_DIR / "index" / "clause_index.pkl"


class ClauseRAG:
    def __init__(self, embed_model: SentenceTransformer, llm_call_fn):
        self.embed_model = embed_model
        self._llm = llm_call_fn
        self._load_index()

    def _load_index(self):
        with open(CLAUSE_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.index  = data["index"]
        self.chunks = data["chunks"]

    def _embed(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = 5, exemption_only: bool = False) -> list:
        vec = self._embed(query)
        _, indices = self.index.search(vec, top_k * 3)
        results = []
        for i in indices[0]:
            if i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            if exemption_only and not chunk.get("is_exemption"):
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def search_by_triggers(self, triggers: list) -> list:
        trigger_article_map = {
            "음주운전":     ["제11조"],
            "무면허":       ["제11조"],
            "마약약물":     ["제11조"],
            "조치의무위반": ["제11조"],
            "고의":         ["제5조", "제8조", "제14조", "제19조", "제23조"],
            "유상운송":     ["제8조", "제14조", "제19조", "제23조"],
            "가족피해":     ["제8조"],
            "무단운전":     ["제8조"],
            "절취운전":     ["제8조"],
            "운전자범위":   ["제8조"],
            "사기":         ["제23조"],
            "고지의무":     ["제44조", "제53조"],
        }
        results, seen = [], set()
        for trigger in triggers:
            for art in trigger_article_map.get(trigger, []):
                for chunk in self.chunks:
                    if chunk["article"] == art and art not in seen:
                        results.append(chunk)
                        seen.add(art)
        if not results:
            query = " ".join(triggers) + " 보상하지 않는 손해 면책"
            results = self.search(query, top_k=5, exemption_only=True)
        return results

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
