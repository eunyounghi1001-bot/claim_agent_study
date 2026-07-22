# ── search_agent.py ──────────────────────────────────
# 약관 조항 검색 Agent + 판결문 검색 Agent (ChromaDB 버전)
# ─────────────────────────────────────────────────────

import warnings
import chromadb
import numpy as np
import os
import json
import urllib.request
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb"


def _call_llm(system: str, user: str, max_tokens: int = 256) -> str:
    """Ollama 또는 Groq API를 호출하는 통합 헬퍼 함수"""
    use_ollama = os.environ.get("USE_OLLAMA", "False").lower() in ("true", "1", "yes")
    
    if not use_ollama:
        api_key = os.environ.get("GROQ_API_KEY")
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        if api_key:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"      [SearchAgent Groq Error] '{model_name}' 호출 실패: {e}. 'llama-3.1-8b-instant' 모델로 2차 시도합니다.")
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user}
                        ],
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                    print("      [SearchAgent Groq] 'llama-3.1-8b-instant' 2차 시도 성공!")
                    return resp.choices[0].message.content.strip()
                except Exception as e2:
                    print(f"      [SearchAgent Groq Error] 2차 시도 실패: {e2}. 로컬 Ollama 폴백 진행.")
            
    # 로컬 Ollama 호출
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens
        }
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["message"]["content"].strip()
    except Exception as e:
        print(f"      [SearchAgent Ollama Error] {e}")
        return ""


class SearchAgent:
    """ChromaDB 기반 약관 조항 + 판결문 통합 검색 Agent"""

    def __init__(self, embed_model: SentenceTransformer, chroma_client=None):
        self.embed_model = embed_model
        self.client = chroma_client
        self._load_indices()

    def _load_indices(self):
        print("  검색 Agent: ChromaDB 로딩 중...")
        if self.client is None:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.clauses_col = self.client.get_collection("clauses")
        self.verdicts_col = self.client.get_collection("verdicts")
        try:
            self.standards_col = self.client.get_collection("standards")
            has_standards = True
        except Exception:
            self.standards_col = None
            has_standards = False

        print(f"  ChromaDB 로드 완료 (약관, 판례, 판단기준: {has_standards})")

    def _embed(self, query: str) -> list:
        vec = self.embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        return vec[0].tolist()

    def _rewrite_query(self, query: str) -> list:
        """구어체 진술을 RAG 조항 검색용 핵심 키워드로 변환"""
        system = (
            "교통사고 및 보험 약관 검색 전문가. 사용자의 일상어 사고 진술을 분석하여, "
            "약관이나 법률 판례를 찾기 위한 대표적인 명사형 키워드 3~4개를 쉼표로 구분하여 출력하십시오. "
            "예: '음주운전, 보행자 보호의무, 사고부담금'. 부연설명 없이 키워드만 출력하세요."
        )
        user = f"사고 진술: '{query}'"
        raw = _call_llm(system, user)
        
        keywords = [q.strip() for q in raw.split(",") if q.strip()]
        if not keywords:
            keywords = [query]
        return keywords[:3]

    def _llm_rerank(self, query: str, candidates: list, top_k: int = 5, is_verdict: bool = False) -> list:
        """LLM을 이용한 정밀 리랭킹 및 연관도 선별"""
        if not candidates:
            return []

        doc_type = "판례" if is_verdict else "약관 조항"
        cand_desc = []
        for idx, c in enumerate(candidates):
            title = c.get("case_num") if is_verdict else f"{c.get('article')} {c.get('title')}"
            snippet = c.get("text")[:200].replace('\n', ' ')
            cand_desc.append(f"[{idx+1}] {title}: {snippet}")

        cand_text = "\n".join(cand_desc)

        system = (
            f"당신은 보험 및 법률 전문가입니다. 질의어와 가장 일치하는 {doc_type} 후보 목록을 분석하여, "
            f"가장 연관성이 높은 순서대로 후보 번호들만 쉼표로 구분해 나열하십시오. "
            f"관련성이 현저히 떨어지는 후보는 제외해도 됩니다. 답변에는 숫자와 쉼표 이외의 텍스트를 포함하지 마십시오. "
            f"예: '3,1,4'"
        )
        user = f"[질문/팩트]\n{query}\n\n[후보 목록]\n{cand_text}\n\n출력:"
        raw = _call_llm(system, user)

        try:
            order = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            reranked = []
            seen = set()
            for idx in order:
                real_idx = idx - 1
                if 0 <= real_idx < len(candidates):
                    c = candidates[real_idx]
                    key = c.get("case_num") if is_verdict else c.get("article")
                    if key not in seen:
                        reranked.append(c)
                        seen.add(key)
            # 순위에 없는 후보도 후순위로 보존
            for c in candidates:
                key = c.get("case_num") if is_verdict else c.get("article")
                if key not in seen:
                    reranked.append(c)
                    seen.add(key)
        except Exception:
            reranked = candidates

        return reranked[:top_k]

    def search_clauses(self, query: str, top_k: int = 5,
                       coverage_filter: str = None,
                       exemption_only: bool = False) -> list:
        """약관 조항 하이브리드 검색 및 LLM 리랭킹 적용"""
        print(f"    [SearchAgent] 약관 ChromaDB 검색 시작: '{query}'")
        
        query_vec = self._embed(query)
        where_filter = {}
        if exemption_only:
            where_filter["is_exemption"] = True

        # Chroma Query
        results = self.clauses_col.query(
            query_embeddings=[query_vec],
            n_results=12,
            where=where_filter if where_filter else None,
            include=["metadatas", "documents", "distances"]
        )

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][idx]
                doc = results["documents"][0][idx]
                distance = results["distances"][0][idx]

                coverage_str = metadata.get("coverage", "")
                coverage = coverage_str.split(",") if coverage_str else []

                # 필터링
                if coverage_filter and coverage_filter not in coverage:
                    continue

                chunk = {
                    "text": doc,
                    "article": metadata.get("article", ""),
                    "title": metadata.get("title", ""),
                    "page": int(metadata.get("page", 0)),
                    "coverage": coverage,
                    "is_exemption": bool(metadata.get("is_exemption", False)),
                    "source": metadata.get("source", "약관"),
                    "score": float(1.0 - distance)
                }
                candidates.append(chunk)

        results = self._llm_rerank(query, candidates, top_k=top_k, is_verdict=False)
        return results

    def search_exemption_clauses(self, triggers: list) -> list:
        """면책 트리거 키워드로 관련 면책 조항 직접 검색"""
        query = " ".join(triggers) + " 보상하지 않는 손해 면책"
        trigger_article_map = {
            "음주운전":   ["제11조", "제23조"],
            "무면허":     ["제11조", "제23조"],
            "무면허운전": ["제11조", "제23조"],
            "마약약물":   ["제11조"],
            "마약약물운전": ["제11조"],
            "조치의무위반": ["제11조"],
            "고의":       ["제5조", "제8조", "제14조", "제19조", "제23조"],
            "유상운송":   ["제8조", "제14조", "제19조", "제23조"],
            "가족피해":   ["제8조"],
            "무단운전":   ["제8조"],
            "절취운전":   ["제8조"],
            "운전자범위": ["제8조"],
            "사기":       ["제23조"],
            "사기횡령":   ["제23조"],
            "고지의무":   ["제44조", "제53조"],
        }

        target_arts = []
        for trigger in triggers:
            target_arts.extend(trigger_article_map.get(trigger, []))

        results = []
        if target_arts:
            target_arts = list(set(target_arts))
            query_results = self.clauses_col.query(
                query_embeddings=[self._embed(query)],
                n_results=50,
                where={"article": {"$in": target_arts}},
                include=["metadatas", "documents", "distances"]
            )
            seen_articles = set()
            if query_results and query_results["ids"] and query_results["ids"][0]:
                for idx in range(len(query_results["ids"][0])):
                    metadata = query_results["metadatas"][0][idx]
                    doc = query_results["documents"][0][idx]
                    distance = query_results["distances"][0][idx]
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
                        results.append(chunk)

        if not results:
            results = self.search_clauses(query, top_k=5, exemption_only=True)

        return results

    def search_verdicts(self, query: str, top_k: int = 5,
                         keyword_filter: str = None) -> list:
        """판결문 하이브리드 검색 및 LLM 리랭킹 적용"""
        print(f"    [SearchAgent] 판례 ChromaDB 검색 시작: '{query}'")
        
        query_vec = self._embed(query)
        results = self.verdicts_col.query(
            query_embeddings=[query_vec],
            n_results=15,
            include=["metadatas", "documents", "distances"]
        )

        candidates, seen = [], set()
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

                    # 필터
                    if keyword_filter and keyword_filter not in keywords:
                        continue

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
                    candidates.append(chunk)

        results = self._llm_rerank(query, candidates, top_k=top_k, is_verdict=True)
        return results

    def search_standard(self, accident_type: str) -> dict | None:
        """사고 유형으로 판단 기준 검색. 없으면 None 반환."""
        if self.standards_col is None:
            return None
        query = f"{accident_type} 사고 판단 기준"
        query_vec = self._embed(query)
        
        results = self.standards_col.query(
            query_embeddings=[query_vec],
            n_results=1,
            include=["metadatas"]
        )
        if results and results["ids"] and results["ids"][0]:
            meta = results["metadatas"][0][0]
            standard_json = meta.get("standard_json", "")
            if standard_json:
                return json.loads(standard_json)
        return None

    def search_verdicts_by_triggers(self, triggers: list, top_k: int = 3) -> list:
        """면책 트리거로 관련 판결문 검색"""
        query = " ".join(triggers) + " 자동차보험 면책 보상"
        query_vec = self._embed(query)
        
        results = self.verdicts_col.query(
            query_embeddings=[query_vec],
            n_results=50,
            include=["metadatas", "documents", "distances"]
        )

        candidates, seen = [], set()
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
                        candidates.append(chunk)
                        if len(candidates) >= top_k:
                            return candidates

        if len(candidates) < top_k:
            vec_results = self.search_verdicts(query, top_k=top_k)
            for r in vec_results:
                if r["case_num"] not in seen:
                    candidates.append(r)
                    seen.add(r["case_num"])
                    if len(candidates) >= top_k:
                        break

        return candidates[:top_k]
