# ── pipeline.py ──────────────────────────────────────
# LangGraph 통합 파이프라인
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
#                                      format_output
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
    node_normalize,
    node_fact_check,
    node_extra_request,
    node_detect_triggers,
    node_search_clauses,
    node_search_verdicts,
    node_liability_judge,
    node_format_output,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.search_agent import SearchAgent
from agents.clause_rag import ClauseRAG
from agents.verdict_rag import VerdictRAG

MAX_RETRIES = 2

_embed_model:  SentenceTransformer | None = None
_search_agent: SearchAgent | None    = None
_clause_rag:   ClauseRAG | None      = None
_verdict_rag:  VerdictRAG | None     = None
_checkpointer  = MemorySaver()


def get_models():
    global _embed_model, _search_agent, _clause_rag, _verdict_rag
    if _embed_model is None:
        print("bge-m3 로딩 중...")
        _embed_model = SentenceTransformer("BAAI/bge-m3")
        print("임베딩 완료!")
    if _search_agent is None:
        _search_agent = SearchAgent(_embed_model)
    if _clause_rag is None:
        from graph.nodes import _call
        _clause_rag  = ClauseRAG(_embed_model, _call)
        _verdict_rag = VerdictRAG(_embed_model, _call)
        print("약관 RAG / 판례 RAG 로드 완료!")
    return _embed_model, _search_agent, _clause_rag, _verdict_rag


def edge_after_fact(state: PipelineState) -> str:
    if state["fact"].get("status") == "충분":
        return "confirmed"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "exceeded"
    return "need_more"


def build_graph():
    _, search_agent, clause_rag, verdict_rag = get_models()

    g = StateGraph(PipelineState)

    g.add_node("text_parse",      node_text_parse)
    g.add_node("normalize",       node_normalize)
    g.add_node("fact_check",      node_fact_check)
    g.add_node("extra_request",   node_extra_request)
    g.add_node("detect_triggers", node_detect_triggers)
    g.add_node("search_clauses",  partial(node_search_clauses,  clause_rag=clause_rag))
    g.add_node("search_verdicts", partial(node_search_verdicts, verdict_rag=verdict_rag))
    g.add_node("liability_judge", partial(node_liability_judge, clause_rag=clause_rag, verdict_rag=verdict_rag))
    g.add_node("format_output",   node_format_output)

    g.set_entry_point("text_parse")
    g.add_edge("text_parse", "normalize")
    g.add_edge("normalize",     "fact_check")

    g.add_conditional_edges(
        "fact_check",
        edge_after_fact,
        {
            "confirmed": "detect_triggers",
            "need_more": "extra_request",
            "exceeded":  END,
        },
    )

    g.add_edge("extra_request",   "fact_check")
    g.add_edge("detect_triggers", "search_clauses")
    g.add_edge("search_clauses",  "search_verdicts")
    g.add_edge("search_verdicts", "liability_judge")
    g.add_edge("liability_judge", "format_output")
    g.add_edge("format_output",   END)

    return g.compile(
        checkpointer=_checkpointer,
        interrupt_after=["extra_request"],
    )


def run(
    user_input: str,
    case_id:    str = "case-001",
    video_url:  str = "",
    extra_data: str = "",
) -> dict:
    graph  = build_graph()
    config = {"configurable": {"thread_id": case_id}}

    init_state: PipelineState = {
        "case_id":      case_id,
        "user_input":   user_input,
        "video_url":    video_url,
        "extra_data":   extra_data,
        "text_result":   {},
        "normalized":    {},
        "fact":          {},
        "extra_request":      None,
        "retry_count":        0,
        "clause_hits":        [],
        "clause_answer":      "",
        "clause_articles":    [],
        "verdict_hits":       [],
        "verdict_answer":     "",
        "verdict_fault_ratio": "",
        "triggers":           [],
        "liability":          {},
        "report":             "",
        "trace":              [],
        "final_status":       "",
        "error":              "",
    }

    result = graph.invoke(init_state, config)

    if not result.get("final_status"):
        result["final_status"] = (
            "완료" if result.get("liability")
            else "추가자료 대기 중"
        )
    return result


def resume(case_id: str, extra_data: str) -> dict:
    import sqlite3, json as _json, time
    graph  = build_graph()
    config = {"configurable": {"thread_id": f"{case_id}-r{int(time.time())}"}}

    conn = sqlite3.connect("cases.db", check_same_thread=False)
    row  = conn.execute(
        "SELECT result_json, user_input FROM cases WHERE case_id=?", (case_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"{case_id} 케이스를 찾을 수 없습니다.", "final_status": "오류"}

    prev = _json.loads(row[0])

    # 이전 accident_type을 user_input에 힌트로 포함 (LLM 재파싱 오류 방지)
    prev_type = (prev.get("normalized") or {}).get("accident_type", "")
    base_input = row[1]
    if prev_type and prev_type not in base_input:
        base_input = f"[사고유형: {prev_type}] {base_input}"

    # 이전 제출 히스토리 누적
    prev_history = prev.get("extra_history", [])
    new_entry = {
        "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extra_data": extra_data,
    }
    extra_history = prev_history + [new_entry]

    init_state: PipelineState = {
        "case_id":       case_id,
        "user_input":    base_input,
        "video_url":     prev.get("video_url", ""),
        "extra_data":    extra_data,
        "text_result":   {},
        "normalized":    {},
        "fact":          {},
        "extra_request":      None,
        "retry_count":        0,
        "clause_hits":        [],
        "clause_answer":      "",
        "clause_articles":    [],
        "verdict_hits":       [],
        "verdict_answer":     "",
        "verdict_fault_ratio": "",
        "triggers":           [],
        "liability":          {},
        "report":             "",
        "trace":              [],
        "final_status":       "",
        "error":              "",
    }
    result = graph.invoke(init_state, config)

    if not result.get("final_status"):
        result["final_status"] = (
            "완료" if result.get("liability")
            else "추가자료 대기 중"
        )
    result["extra_history"] = extra_history
    return result
