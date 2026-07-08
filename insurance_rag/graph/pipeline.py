# ── pipeline.py ──────────────────────────────────────
# LangGraph 통합 파이프라인
# P1(자료수집·사실확정) + P2(검색·면부책)를 단일 그래프로
#
# 그래프 구조:
#   text_parse → media_analyze → normalize → fact_check
#                                                ↙        ↘
#                                           (충분)         (부족)
#                                              ↓              ↓
#                                      detect_triggers  extra_request
#                                              ↓         (interrupt)
#                                      search_clauses
#                                              ↓
#                                      search_verdicts
#                                              ↓
#                                      liability_judge
#                                              ↓
#                                            END
# ─────────────────────────────────────────────────────

import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer

from graph.state import PipelineState
from graph.nodes import (
    node_text_parse,
    node_media_analyze,
    node_normalize,
    node_fact_check,
    node_extra_request,
    node_detect_triggers,
    node_search_clauses,
    node_search_verdicts,
    node_liability_judge,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.search_agent import SearchAgent

MAX_RETRIES = 2

# ── 싱글톤 모델 (서버 시작 시 1회 로드) ─────────────
_embed_model:  SentenceTransformer | None = None
_search_agent: SearchAgent | None    = None
_checkpointer  = MemorySaver()


def get_models():
    global _embed_model, _search_agent
    if _embed_model is None:
        print("bge-m3 로딩 중...")
        _embed_model = SentenceTransformer("BAAI/bge-m3")
        print("임베딩 완료!")
    if _search_agent is None:
        _search_agent = SearchAgent(_embed_model)
    return None, _embed_model, _search_agent


# ── 조건 엣지 ────────────────────────────────────────
def edge_after_fact(state: PipelineState) -> str:
    if state["fact"].get("status") == "충분":
        return "confirmed"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "exceeded"
    return "need_more"


# ── 그래프 빌드 ──────────────────────────────────────
def build_graph():
    llm, _, search_agent = get_models()

    g = StateGraph(PipelineState)

    # 노드 등록 (partial로 의존성 주입)
    g.add_node("text_parse",       partial(node_text_parse,      llm=llm))
    g.add_node("media_analyze",    partial(node_media_analyze,   llm=llm))
    g.add_node("normalize",        partial(node_normalize,        llm=llm))
    g.add_node("fact_check",       partial(node_fact_check,       llm=llm, search_agent=search_agent))
    g.add_node("extra_request",    partial(node_extra_request,    llm=llm))
    g.add_node("detect_triggers",  node_detect_triggers)
    g.add_node("search_clauses",   partial(node_search_clauses,   search_agent=search_agent))
    g.add_node("search_verdicts",  partial(node_search_verdicts,  search_agent=search_agent))
    g.add_node("liability_judge",  partial(node_liability_judge,  llm=llm, search_agent=search_agent))

    # 엣지 (선형 흐름)
    g.set_entry_point("text_parse")
    g.add_edge("text_parse",    "media_analyze")
    g.add_edge("media_analyze", "normalize")
    g.add_edge("normalize",     "fact_check")

    # 사실확정 분기
    g.add_conditional_edges(
        "fact_check",
        edge_after_fact,
        {
            "confirmed":  "detect_triggers",  # P2로 진행
            "need_more":  "extra_request",     # 추가자료 요청
            "exceeded":   END,                 # 최대 재시도 초과
        },
    )

    # extra_request → fact_check 루프 (interrupt_after로 중단)
    g.add_edge("extra_request", "fact_check")

    # P2 흐름
    g.add_edge("detect_triggers", "search_clauses")
    g.add_edge("search_clauses",  "search_verdicts")
    g.add_edge("search_verdicts", "liability_judge")
    g.add_edge("liability_judge", END)

    # extra_request 후 interrupt → 고객 자료 대기
    return g.compile(
        checkpointer=_checkpointer,
        interrupt_after=["extra_request"],
    )


# ── 실행 함수 ─────────────────────────────────────────
def run(
    user_input:  str,
    case_id:     str  = "case-001",
    media_paths: list = None,
    youtube_url: str  = "",
    extra_data:  str  = "",
) -> dict:
    """1회차 실행. extra_request 노드 후 자동 interrupt."""
    graph  = build_graph()
    config = {"configurable": {"thread_id": case_id}}

    init_state: PipelineState = {
        "case_id":      case_id,
        "user_input":   user_input,
        "media_paths":  media_paths or [],
        "youtube_url":  youtube_url,
        "extra_data":   extra_data,
        "text_result":   {},
        "media_result":  "",
        "normalized":    {},
        "keyframe_urls": [],
        "fact":          {},
        "extra_request": None,
        "retry_count":   0,
        "clause_hits":   [],
        "verdict_hits":  [],
        "triggers":      [],
        "liability":     {},
        "final_status":  "",
        "error":         "",
    }

    result = graph.invoke(init_state, config)

    if not result.get("final_status"):
        result["final_status"] = (
            "완료" if result.get("liability")
            else "추가자료 대기 중"
        )
    return result


def resume(case_id: str, extra_data: str, media_paths: list = None) -> dict:
    """고객 추가자료 재제출 후 처음부터 재실행."""
    import sqlite3, json as _json
    graph  = build_graph()
    # 새 thread_id로 체크포인트 충돌 방지
    import time
    config = {"configurable": {"thread_id": f"{case_id}-r{int(time.time())}"}}

    conn = sqlite3.connect("cases.db", check_same_thread=False)
    row  = conn.execute(
        "SELECT result_json, user_input FROM cases WHERE case_id=?", (case_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"{case_id} 케이스를 찾을 수 없습니다.", "final_status": "오류"}

    prev = _json.loads(row[0])
    init_state: PipelineState = {
        "case_id":       case_id,
        "user_input":    row[1],
        "media_paths":   media_paths or [],
        "youtube_url":   prev.get("youtube_url", ""),
        "extra_data":    extra_data,
        "text_result":   {},
        "media_result":  "",
        "normalized":    {},
        "keyframe_urls": [],
        "fact":          {},
        "extra_request": None,
        "retry_count":   0,
        "clause_hits":   [],
        "verdict_hits":  [],
        "triggers":      [],
        "liability":     {},
        "final_status":  "",
        "error":         "",
    }
    result = graph.invoke(init_state, config)

    if not result.get("final_status"):
        result["final_status"] = (
            "완료" if result.get("liability")
            else "추가자료 대기 중"
        )
    return result
