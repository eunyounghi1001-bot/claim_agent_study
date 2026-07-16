"""
판례 RAG Agent
- FAISS 벡터 검색으로 유사 판결문 검색
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

BASE_DIR           = Path(__file__).parent.parent
VERDICT_INDEX_PATH = BASE_DIR / "index" / "verdict_index.pkl"


class VerdictRAG:
    def __init__(self, embed_model: SentenceTransformer, llm_call_fn):
        self.embed_model = embed_model
        self._llm = llm_call_fn
        self._load_index()

    def _load_index(self):
        with open(VERDICT_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.index  = data["index"]
        self.chunks = data["chunks"]

    def _embed(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = 5) -> list:
        vec = self._embed(query)
        _, indices = self.index.search(vec, top_k * 3)
        results, seen = [], set()
        for i in indices[0]:
            if i >= len(self.chunks):
                continue
            chunk = self.chunks[i]
            if chunk.get("case_num") not in seen:
                results.append(chunk)
                seen.add(chunk.get("case_num"))
            if len(results) >= top_k:
                break
        return results

    def search_by_triggers(self, triggers: list, top_k: int = 3) -> list:
        results, seen = [], set()
        for trigger in triggers:
            for chunk in self.chunks:
                if trigger in chunk.get("keywords", []) and chunk.get("case_num") not in seen:
                    results.append(chunk)
                    seen.add(chunk["case_num"])
                    if len(results) >= top_k:
                        return results
        if len(results) < top_k:
            query = " ".join(triggers) + " 자동차보험 면책 보상"
            for chunk in self.search(query, top_k=top_k):
                if chunk.get("case_num") not in seen:
                    results.append(chunk)
                    seen.add(chunk["case_num"])
                    if len(results) >= top_k:
                        break
        return results[:top_k]

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
