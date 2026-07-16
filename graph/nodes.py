# ── nodes.py ─────────────────────────────────────────
# LangGraph 노드 함수 모음
# 각 노드는 PipelineState를 받아 업데이트할 딕셔너리를 반환
# ─────────────────────────────────────────────────────

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from groq import Groq

from graph.state import PipelineState

load_dotenv()
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── 데이터 로드 ───────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"

def _load_standards() -> list:
    with open(_DATA_DIR / "standards.json", encoding="utf-8") as f:
        return json.load(f)

def _load_exemptions() -> dict:
    with open(_DATA_DIR / "exemptions.json", encoding="utf-8") as f:
        return json.load(f)

_STANDARDS  = _load_standards()
_EXEMPTIONS = _load_exemptions()


# ── 사고 유형 매칭 ────────────────────────────────────
def _find_standard(accident_type: str) -> dict | None:
    at = accident_type.strip()
    for s in _STANDARDS:
        if at == s["accident_type"] or at in s["aliases"]:
            return s
    # 부분 문자열 매칭 (Qwen이 표현을 다르게 뱉을 때 대비)
    for s in _STANDARDS:
        candidates = [s["accident_type"]] + s["aliases"]
        if any(c in at or at in c for c in candidates):
            return s
    return None


# ── LLM 호출 헬퍼 ────────────────────────────────────
def _call(system: str, user: str, max_tokens: int = 1024) -> str:
    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        try:
            idx = clean.rfind("}")
            if idx != -1:
                return json.loads(clean[:idx + 1])
        except Exception:
            pass
        return {"raw_output": clean}


# ══════════════════════════════════════════════════════
#  P1 노드: 자료수집 Agent
# ══════════════════════════════════════════════════════

def node_text_parse(state: PipelineState, **_) -> dict:
    user = f"""콜센터 접수 사고 내용을 파싱하세요.

[접수 내용]
{state['user_input']}

JSON만 출력:
{{
  "content"          : "사고 핵심 요약 1~2문장",
  "dispatch"         : "O 또는 X (언급 없으면 X)",
  "accident_type"    : "후진/횡단보도/비보호좌회전/차선변경/추돌/교차로/기타",
  "first_request_msg": "dispatch=X이면 고객에게 보낼 1차 자료 요청문, 있으면 빈 문자열"
}}"""
    raw    = _call("자동차 사고 자료수집 전문가. JSON만 출력.", user)
    result = _parse_json(raw)
    return {"text_result": result}


def node_normalize(state: PipelineState, **_) -> dict:
    video_url = state.get("video_url", "") or ""
    video_note = f"영상 URL 제출됨: {video_url}" if video_url else "영상 없음"

    user = f"""아래 결과를 하나의 JSON으로 정리하세요.

[텍스트 파싱]
{json.dumps(state['text_result'], ensure_ascii=False)}

[영상 자료]
{video_note}
(영상 판독은 담당자가 직접 수행)

JSON만 출력:
{{
  "accident_type"    : "사고 유형",
  "dispatch_yn"      : "O 또는 X",
  "text_summary"     : "접수 내용 요약",
  "video_url"        : "영상 URL (없으면 빈 문자열)",
  "first_request_msg": "1차 자료 요청문 (없으면 빈 문자열)",
  "metadata"         : {{"source": "콜센터 접수 및 고객 제출 자료", "reliability": "상 또는 하"}}
}}"""
    raw    = _call("자동차 사고 데이터 정규화. JSON만 출력.", user)
    result = _parse_json(raw)
    return {"normalized": result}


# ══════════════════════════════════════════════════════
#  P1 노드: 사실확정 — 룰 엔진
# ══════════════════════════════════════════════════════

def node_fact_check(state: PipelineState, **_) -> dict:
    normalized   = state.get("normalized", {})
    media_result = state.get("media_result", "")
    extra_data   = state.get("extra_data", "") or ""
    accident_type = normalized.get("accident_type", "") or state.get("text_result", {}).get("accident_type", "")

    standard = _find_standard(accident_type)

    # 증거 텍스트 통합
    evidence_text = "\n".join([
        normalized.get("text_summary", ""),
        normalized.get("media_summary", ""),
        media_result,
        extra_data,
    ]).lower()

    # 사실확정 충분/부족 판단
    if standard:
        criteria = standard["fact_check_criteria"]
        sufficient   = any(kw in evidence_text for kw in criteria["sufficient_keywords"])
        insufficient = any(kw in evidence_text for kw in criteria["insufficient_keywords"])
        status = "충분" if sufficient and not insufficient else "부족"

        # 충분 조건에 해당하는 증거 항목 수집
        matched_evidence = [
            kw for kw in criteria["sufficient_keywords"] if kw in evidence_text
        ]

        key_issue   = standard["key_issues"][0]
        legal_basis = criteria["legal_basis"]

        # 미제출 증거 (우선순위 상위 2개 중 언급 없는 것)
        missing_info = []
        for ev in sorted(standard["required_evidence"], key=lambda x: x["priority"])[:2]:
            if ev["item"].replace(" ", "").lower() not in evidence_text.replace(" ", ""):
                missing_info.append({
                    "item":        ev["item"],
                    "reason":      ev["reason"],
                    "decisiveness": ev["decisiveness"],
                })
    else:
        # 기준 없는 사고 유형 — 기본값
        status        = "부족"
        key_issue     = "사고 유형 불명확 — 추가 확인 필요"
        legal_basis   = ""
        matched_evidence = []
        missing_info  = [{"item": "블랙박스 또는 CCTV 영상", "reason": "사고 경위 객관적 확인", "decisiveness": 9}]

    fact_summary = (
        f"{accident_type} 사고. "
        f"확인된 증거: {', '.join(matched_evidence) if matched_evidence else '없음'}. "
        f"법적 근거: {legal_basis}"
    ) if status == "충분" else ""

    trace_entry = {
        "node":             "fact_check",
        "accident_type":    accident_type,
        "standard_found":   standard is not None,
        "evidence_text":    evidence_text[:300],
        "matched_keywords": matched_evidence,
        "status":           status,
    }

    return {
        "fact": {
            "key_issue":    key_issue,
            "status":       status,
            "fact_summary": fact_summary,
            "confidence":   "상" if status == "충분" else "",
            "contradiction": "",
            "unresolved":   [] if status == "충분" else [key_issue],
            "missing_info": missing_info,
        },
        "trace": state.get("trace", []) + [trace_entry],
    }


def node_extra_request(state: PipelineState, **_) -> dict:
    missing = sorted(
        state["fact"].get("missing_info", []),
        key=lambda x: x.get("decisiveness", 0),
        reverse=True,
    )[:2]

    user = f"""사실확정에 필요한 추가 자료를 고객에게 요청하는 안내문을 작성하세요.

[핵심 쟁점]
{state['fact'].get('key_issue', '')}

[필요 자료 (최대 2개)]
{json.dumps(missing, ensure_ascii=False)}

JSON만 출력:
{{
  "request_items": [{{"priority": 1, "item": "자료명", "reason": "이유", "decisiveness": 9}}],
  "request_msg"  : "고객에게 전달할 정중한 안내문 (한국어)"
}}"""
    raw    = _call("자동차 사고 추가자료 요청 담당자. JSON만 출력.", user)
    result = _parse_json(raw)
    return {
        "extra_request": result,
        "retry_count":   state.get("retry_count", 0) + 1,
    }


# ══════════════════════════════════════════════════════
#  P2 노드: 검색 Agent
# ══════════════════════════════════════════════════════

def node_detect_triggers(state: PipelineState) -> dict:
    """면책 트리거 감지 — exemptions.json 키워드 매칭"""
    text     = json.dumps(state["fact"], ensure_ascii=False).lower()
    triggers = []
    matched_details = {}
    for trigger, data in _EXEMPTIONS["triggers"].items():
        matched_kws = [kw for kw in data["keywords"] if kw in text]
        if matched_kws:
            triggers.append(trigger)
            matched_details[trigger] = matched_kws

    trace_entry = {
        "node":            "detect_triggers",
        "scanned_text":    text[:300],
        "matched_triggers": matched_details,
    }
    return {
        "triggers": triggers,
        "trace":    state.get("trace", []) + [trace_entry],
    }


def node_search_clauses(state: PipelineState, clause_rag) -> dict:
    triggers = state.get("triggers", [])
    fact     = state.get("fact", {})
    query    = fact.get("fact_summary", "") + " " + fact.get("key_issue", "")

    if triggers:
        chunks = clause_rag.search_by_triggers(triggers)
        rag_result = clause_rag.answer(
            f"다음 면책 트리거 관련 약관 조항을 해석하세요: {', '.join(triggers)}",
            context_chunks=chunks,
        )
    else:
        rag_result = clause_rag.answer(query, top_k=5)
        chunks     = rag_result.get("chunks", [])

    return {
        "clause_hits":   chunks,
        "clause_answer": rag_result.get("answer", ""),
        "clause_articles": rag_result.get("articles", []),
    }


def node_search_verdicts(state: PipelineState, verdict_rag) -> dict:
    triggers = state.get("triggers", [])
    fact     = state.get("fact", {})
    query    = fact.get("fact_summary", "") + " " + fact.get("key_issue", "") + " 자동차보험 보상"

    if triggers:
        chunks = verdict_rag.search_by_triggers(triggers, top_k=3)
        rag_result = verdict_rag.answer(
            f"다음 면책 트리거 관련 판례를 해석하세요: {', '.join(triggers)}",
            context_chunks=chunks,
        )
    else:
        rag_result = verdict_rag.answer(query, top_k=3)
        chunks     = rag_result.get("chunks", [])

    return {
        "verdict_hits":   chunks,
        "verdict_answer": rag_result.get("answer", ""),
        "verdict_fault_ratio": rag_result.get("fault_ratio", ""),
    }


# ══════════════════════════════════════════════════════
#  P2 노드: 면부책 — 룰 엔진
# ══════════════════════════════════════════════════════

def node_liability_judge(state: PipelineState, clause_rag=None, verdict_rag=None, **_) -> dict:
    normalized    = state.get("normalized", {})
    fact          = state.get("fact", {})
    triggers      = state.get("triggers", [])
    verdict_hits  = state.get("verdict_hits", [])
    accident_type = normalized.get("accident_type", "") or state.get("text_result", {}).get("accident_type", "")

    standard = _find_standard(accident_type)

    # ── 과실비율 계산 ─────────────────────────────────
    if standard:
        ls      = standard["liability_standard"]
        base    = dict(ls["기본_과실"])  # {"좌회전차": 70, "직진차": 30}
        evidence_text = json.dumps(fact, ensure_ascii=False).lower()

        applied_modifiers = []
        for mod in ls["수정_요소"]:
            if any(kw in evidence_text for kw in mod["keywords"]):
                base[mod["target"]] = base.get(mod["target"], 0) + mod["delta"]
                applied_modifiers.append(mod["description"])

        # 합계가 100이 되도록 보정
        total = sum(base.values())
        if total != 100:
            keys = list(base.keys())
            base[keys[0]] += 100 - total

        fault_ratio = " : ".join(f"{k} {v}%" for k, v in base.items())
        근거_약관   = ls["근거_약관"]
    else:
        fault_ratio       = "산정 불가 (기준 없음)"
        applied_modifiers = []
        근거_약관         = []

    # ── 담보별 판단 ───────────────────────────────────
    coverage_rules = _EXEMPTIONS["coverage_rules"]
    담보별판단 = {}
    면책사유   = []

    for coverage, rule in coverage_rules.items():
        exempt = any(t in triggers for t in rule["exempt_triggers"])
        if exempt:
            담보별판단[coverage] = "면책"
            for t in triggers:
                if t in rule["exempt_triggers"] and t not in 면책사유:
                    면책사유.append(t)
        else:
            담보별판단[coverage] = rule["default"]

    # ── 사고부담금 ────────────────────────────────────
    surcharge = {"대인I": 0, "대인II": 0, "대물": 0}
    for t in triggers:
        if t in _EXEMPTIONS["surcharge"]:
            for k, v in _EXEMPTIONS["surcharge"][t].items():
                surcharge[k] = max(surcharge[k], v)

    # ── 전체 면/부책 결정 ─────────────────────────────
    # 대인배상I이 항상 부책이므로 면책사유가 있어도 "부책(일부 담보 면책)"으로 표현
    any_exempt = any(v == "면책" for v in 담보별판단.values())
    any_liable = any(v == "부책" for v in 담보별판단.values())

    if any_exempt and any_liable:
        liability_verdict = "부책(일부 담보 면책)"
    elif any_exempt:
        liability_verdict = "면책(미지급)"
    else:
        liability_verdict = "부책(보상)"

    # ── 약관·판례 근거 수집 ───────────────────────────
    약관근거 = list(근거_약관)
    for t in triggers:
        약관근거 += _EXEMPTIONS["triggers"][t]["articles"]

    # RAG 답변에서 추가 근거 보강
    clause_answer  = state.get("clause_answer", "")
    verdict_answer = state.get("verdict_answer", "")
    rag_articles   = state.get("clause_articles", [])
    rag_fault      = state.get("verdict_fault_ratio", "")

    약관근거 = sorted(set(약관근거 + rag_articles))
    판례근거 = [v["case_num"] for v in verdict_hits[:3] if "case_num" in v]

    # RAG 과실비율이 있으면 우선 반영
    if rag_fault:
        fault_ratio = rag_fault

    coverage_trace = {
        coverage: {
            "verdict": verdict,
            "reason":  f"면책 트리거 {[t for t in triggers if t in coverage_rules[coverage]['exempt_triggers']]} 감지" if verdict == "면책" else coverage_rules[coverage]["note"],
        }
        for coverage, verdict in 담보별판단.items()
    }

    trace_entry = {
        "node":              "liability_judge",
        "standard_found":    standard is not None,
        "base_fault":        dict(ls["기본_과실"]) if standard else {},
        "applied_modifiers": applied_modifiers,
        "final_fault_ratio": fault_ratio,
        "triggers_detected": triggers,
        "coverage_trace":    coverage_trace,
        "surcharge":         surcharge if any(surcharge.values()) else {},
    }

    result = {
        "liability":    liability_verdict,
        "면책사유":     면책사유,
        "약관근거":     약관근거,
        "판례근거":     판례근거,
        "담보별판단":   담보별판단,
        "과실비율":     fault_ratio,
        "수정요소적용": applied_modifiers,
        "사고부담금":   surcharge if any(surcharge.values()) else {},
        "분기":         "협상(면책통보)" if liability_verdict == "면책(미지급)" else "P3 과실판단",
        "약관RAG해석":  clause_answer,
        "판례RAG해석":  verdict_answer,
    }
    return {
        "liability":    result,
        "final_status": "완료",
        "trace":        state.get("trace", []) + [trace_entry],
    }


# ══════════════════════════════════════════════════════
#  P3 노드: 포매팅 전용 LLM
# ══════════════════════════════════════════════════════

def node_format_output(state: PipelineState, **_) -> dict:
    liability  = state.get("liability", {})
    fact       = state.get("fact", {})
    normalized = state.get("normalized", {})
    trace      = state.get("trace", [])

    user = f"""아래 보상 판단 결과를 담당자가 읽기 좋은 한국어 보고서로 작성하세요.
숫자와 판단값은 절대 바꾸지 말고, 문장만 자연스럽게 정리하세요.

[사고 개요]
{json.dumps(normalized, ensure_ascii=False, indent=2)}

[사실관계]
{json.dumps(fact, ensure_ascii=False, indent=2)}

[면부책 판단]
{json.dumps(liability, ensure_ascii=False, indent=2)}

[판단 근거 히스토리]
{json.dumps(trace, ensure_ascii=False, indent=2)}

다음 형식으로 출력:
## 사고 개요
(1~2문장)

## 사실관계
(확정된 사실 또는 미확정 이유)

## 면부책 판단
- 결론: 부책/면책
- 과실비율: ...
- 담보별 판단: (표 형식)
- 근거 약관: ...
- 참고 판례: ...

## 판단 근거
(각 단계별로 어떤 키워드/데이터에 의해 결정됐는지 간결하게)

## 특이사항
(면책사유, 사고부담금, 수정요소 등. 없으면 '없음')"""

    report = _call("자동차보험 보상 보고서 작성 전문가. 숫자와 판단값은 절대 변경 금지.", user, max_tokens=1500)
    return {"report": report}
