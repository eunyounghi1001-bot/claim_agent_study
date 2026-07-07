# ── search_agent.py ──────────────────────────────────
# 약관 조항 검색 Agent + 판결문 검색 Agent
# ─────────────────────────────────────────────────────

import pickle, warnings
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR            = Path(__file__).parent.parent
CLAUSE_INDEX_PATH   = BASE_DIR / "index" / "clause_index.pkl"
VERDICT_INDEX_PATH  = BASE_DIR / "index" / "verdict_index.pkl"
STANDARD_INDEX_PATH = BASE_DIR / "index" / "standard_index.pkl"


class SearchAgent:
    """약관 조항 + 판결문 통합 검색 Agent"""

    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model
        self._load_indices()

    def _load_indices(self):
        print("  검색 Agent: 인덱스 로딩 중...")

        with open(CLAUSE_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.clause_index  = data["index"]
        self.clause_chunks = data["chunks"]

        with open(VERDICT_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.verdict_index  = data["index"]
        self.verdict_chunks = data["chunks"]

        if STANDARD_INDEX_PATH.exists():
            with open(STANDARD_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self.standard_index  = data["index"]
            self.standard_chunks = data["chunks"]
        else:
            self.standard_index  = None
            self.standard_chunks = []

        print(f"  약관 조항: {len(self.clause_chunks)}개, "
              f"판결문: {len(self.verdict_chunks)}개, "
              f"판단기준: {len(self.standard_chunks)}개 로드 완료")

    def _embed(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        return vec

    # ── 약관 조항 검색 ───────────────────────────────
    def search_clauses(self, query: str, top_k: int = 5,
                       coverage_filter: str = None,
                       exemption_only: bool = False) -> list:
        """약관 조항 검색. coverage_filter로 담보 필터링 가능."""
        vec = self._embed(query)
        _, indices = self.clause_index.search(vec, top_k * 3)  # 넉넉하게 가져와서 필터

        results = []
        for i in indices[0]:
            if i >= len(self.clause_chunks):
                continue
            chunk = self.clause_chunks[i]
            if exemption_only and not chunk["is_exemption"]:
                continue
            if coverage_filter and coverage_filter not in chunk["coverage"]:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    def search_exemption_clauses(self, triggers: list) -> list:
        """면책 트리거 키워드로 관련 면책 조항 직접 검색"""
        results = []
        seen_articles = set()

        # 트리거 키워드별 검색
        trigger_article_map = {
            "음주운전":   ["제11조"],
            "무면허":     ["제11조"],
            "마약약물":   ["제11조"],
            "조치의무위반": ["제11조"],
            "고의":       ["제5조", "제8조", "제14조", "제19조", "제23조"],
            "유상운송":   ["제8조", "제14조", "제19조", "제23조"],
            "가족피해":   ["제8조"],
            "무단운전":   ["제8조"],
            "절취운전":   ["제8조"],
            "운전자범위": ["제8조"],
            "사기":       ["제23조"],
            "고지의무":   ["제44조", "제53조"],
        }

        for trigger in triggers:
            target_arts = trigger_article_map.get(trigger, [])
            for chunk in self.clause_chunks:
                if chunk["article"] in target_arts and chunk["article"] not in seen_articles:
                    results.append(chunk)
                    seen_articles.add(chunk["article"])

        # 없으면 벡터 검색으로 보완
        if not results:
            query = " ".join(triggers) + " 보상하지 않는 손해 면책"
            results = self.search_clauses(query, top_k=5, exemption_only=True)

        return results

    # ── 판결문 검색 ──────────────────────────────────
    def search_verdicts(self, query: str, top_k: int = 5,
                        keyword_filter: str = None) -> list:
        """판결문 검색. keyword_filter로 쟁점 키워드 필터링 가능."""
        vec = self._embed(query)
        _, indices = self.verdict_index.search(vec, top_k * 3)

        results = []
        for i in indices[0]:
            if i >= len(self.verdict_chunks):
                continue
            chunk = self.verdict_chunks[i]
            if keyword_filter and keyword_filter not in chunk["keywords"]:
                continue
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    # ── 판단 기준 검색 ──────────────────────────────────
    def search_standard(self, accident_type: str) -> dict | None:
        """사고 유형으로 판단 기준 검색. 없으면 None 반환."""
        if self.standard_index is None or not self.standard_chunks:
            return None
        vec = self._embed(accident_type + " 사고 판단 기준")
        _, indices = self.standard_index.search(vec, 1)
        i = indices[0][0]
        if i < len(self.standard_chunks):
            return self.standard_chunks[i]["standard"]
        return None

    def search_verdicts_by_triggers(self, triggers: list, top_k: int = 3) -> list:
        """면책 트리거로 관련 판결문 검색"""
        query = " ".join(triggers) + " 자동차보험 면책 보상"
        results = []
        seen = set()

        # 트리거 키워드가 있는 판결문 우선
        for trigger in triggers:
            for chunk in self.verdict_chunks:
                if trigger in chunk["keywords"] and chunk["case_num"] not in seen:
                    results.append(chunk)
                    seen.add(chunk["case_num"])
                    if len(results) >= top_k:
                        return results

        # 부족하면 벡터 검색 보완
        if len(results) < top_k:
            vec_results = self.search_verdicts(query, top_k=top_k)
            for r in vec_results:
                if r["case_num"] not in seen:
                    results.append(r)
                    seen.add(r["case_num"])
                    if len(results) >= top_k:
                        break

        return results[:top_k]
