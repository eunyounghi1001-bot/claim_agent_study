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
_groq_client = None
try:
    if os.environ.get("GROQ_API_KEY"):
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception:
    pass
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


# ── LLM 호출 헬퍼 (Groq 및 Ollama 폴백) ──────────────────
def _detect_ollama_model() -> str:
    import urllib.request
    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        return env_model
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            for candidate in ["Llama-3-Open-Ko-8B-Q5_K_M.gguf:latest", "llama3:latest", "qwen2.5:3b", "qwen3.6:latest"]:
                if candidate in models:
                    return candidate
            if models:
                return models[0]
    except Exception:
        pass
    return "llama3:latest"


def _call_ollama(system: str, user: str, max_tokens: int = 1024) -> str:
    import urllib.request
    model_name = _detect_ollama_model()
    print(f"    [Ollama] {model_name} 로컬 모델 호출 중...")
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
        with urllib.request.urlopen(req, timeout=90) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["message"]["content"].strip()
    except Exception as e:
        print(f"    [Ollama] API 호출 실패: {e}")
        raise e

def _call(system: str, user: str, max_tokens: int = 1024) -> str:
    use_ollama = os.environ.get("USE_OLLAMA", "False").lower() in ("true", "1", "yes")
    
    if not use_ollama and _groq_client is not None:
        try:
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
        except Exception as e:
            print(f"    [Groq] '{GROQ_MODEL}' API 호출 실패: {e}. 'llama-3.1-8b-instant' 모델로 2차 시도합니다.")
            try:
                resp = _groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                print("    [Groq] 'llama-3.1-8b-instant' 2차 시도 성공!")
                return resp.choices[0].message.content.strip()
            except Exception as e2:
                print(f"    [Groq] 2차 시도 실패: {e2}. Ollama 로컬 모델로 폴백합니다.")
            
    try:
        return _call_ollama(system, user, max_tokens)
    except Exception as oe:
        return f"Error: LLM 호출 실패 (Groq 및 Ollama 모두 실패). 상세: {oe}"


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

    # 증거 텍스트 통합 및 라벨링
    evidence_text = "\n".join([
        f"사고 상황 기술: {normalized.get('text_summary', '')}",
        f"미디어 분석 결과: {normalized.get('media_summary', '')}",
        f"미디어 상세 정보: {media_result}",
        f"고객/담당자 추가 제출 자료: {extra_data}",
    ]).strip()

    # 기준 정보가 있는 경우 가이드라인으로 활용
    standard_guideline = ""
    if standard:
        standard_guideline = f"""
[참고 표준 판단 기준]
- 주요 쟁점 목록: {json.dumps(standard.get('key_issues', []), ensure_ascii=False)}
- 필수 요구 증거: {json.dumps(standard.get('required_evidence', []), ensure_ascii=False)}
- 기본 법적 근거: {standard.get('fact_check_criteria', {}).get('legal_basis', '')}
"""

    system_prompt = (
        "교통사고 사실관계 확인(Fact Checking) 전문가. "
        "제출된 블랙박스 영상 분석 결과, 현장 진술, 추가 제출물 등을 종합 분석하여 "
        "사고의 과실 비율과 면부책을 판단하기에 사실관계 정보가 충분한지 판정합니다. "
        "반드시 JSON 포맷으로만 응답해야 합니다."
    )

    user_prompt = f"""제출된 증거 및 진술을 분석하여 사실관계가 충분히 확인(확정)되었는지 판단해 주세요.
만약 목격자 진술이나 고객의 일방적 주장만 있고, 사고 경위를 증명할 객관적 증거(블랙박스, CCTV 영상, 목격 차량 블박, 현장 사진 등)가 없다면 상태를 반드시 '부족'으로 판정해야 합니다.

[분석할 사고 유형]
{accident_type}
{standard_guideline}
[제출된 증거 및 진술 정보]
{evidence_text}

아래 JSON 스키마 형식으로만 출력하세요. 다른 설명 텍스트는 절대 포함하지 마세요.
{{
  "status": "충분" 또는 "부족" (사고 상황이 객관적으로 명확히 입증되면 '충분', 주장이 엇갈리거나 핵심 증거가 없으면 '부족'),
  "key_issue": "가장 핵심이 되는 핵심 쟁점 한 줄 요약",
  "fact_summary": "충분 상태일 때 작성할 확정된 사고 사실관계 요약 (부족 상태면 빈 문자열)",
  "confidence": "상", "중", "하" 중 선택,
  "matched_evidence": ["확인에 사용된 핵심 객관적 증거 목록"],
  "missing_info": [
    {{
      "item": "추가로 확보해야 할 필수 자료명 (예: 상대 차량 블랙박스 영상)",
      "reason": "해당 자료가 사실관계 확인에 필요한 구체적 이유",
      "decisiveness": 1에서 10 사이 중요도 점수 (10이 가장 높음)
    }}
  ] (충분 상태면 빈 배열)
}}"""

    raw = _call(system_prompt, user_prompt)
    result = _parse_json(raw)

    status = result.get("status", "부족")
    key_issue = result.get("key_issue", "사고 경위 불명확 - 추가 자료 확인 필요")
    fact_summary = result.get("fact_summary", "")
    confidence = result.get("confidence", "중")
    matched_evidence = result.get("matched_evidence", [])
    missing_info = result.get("missing_info", [])

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
            "confidence":   confidence,
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
    import re
    # 사실관계 요약뿐만 아니라 사용자 입력 및 추가 데이터도 함께 스캔하여 키워드를 누락 없이 감지
    texts = [
        json.dumps(state.get("fact", {}), ensure_ascii=False),
        state.get("user_input", ""),
        state.get("extra_data", "")
    ]
    text = "\n".join(texts).lower()
    
    # 이미 설정된 수동 트리거 가져오기
    triggers = list(state.get("triggers", []))
    matched_details = {t: ["수동 설정"] for t in triggers}
    
    def match_keyword(kw: str, txt: str) -> bool:
        if kw == "고의":
            # '사고의', '피고의', '원고의', '보고의', '경고의', '최고의' 등 조사 매칭으로 인한 오탐 방지
            pattern = r'(?<![사피원창보최경제금연])고의'
            return bool(re.search(pattern, txt))
        elif kw == "사기":
            # '수사기관', '복사기' 등 오탐 방지
            pattern = r'(?<![수복검])사기'
            return bool(re.search(pattern, txt))
        return kw in txt

    for trigger, data in _EXEMPTIONS["triggers"].items():
        if trigger in triggers:
            continue
        matched_kws = [kw for kw in data["keywords"] if match_keyword(kw, text)]
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

    # RAG LLM이 누락시킨 근거 조항이 있을 수 있으므로, 감지된 트리거에 매핑된 약관 조항도 함께 포함
    articles = list(rag_result.get("articles", []))
    for t in triggers:
        if t in _EXEMPTIONS["triggers"]:
            articles.extend(_EXEMPTIONS["triggers"][t]["articles"])
        # '무면허'와 '무면허운전' 이름 차이 호환 처리
        elif t == "무면허운전" and "무면허" in _EXEMPTIONS["triggers"]:
            articles.extend(_EXEMPTIONS["triggers"]["무면허"]["articles"])
        elif t == "무면허" and "무면허운전" in _EXEMPTIONS["triggers"]:
            articles.extend(_EXEMPTIONS["triggers"]["무면허운전"]["articles"])

    return {
        "clause_hits":   chunks,
        "clause_answer": rag_result.get("answer", ""),
        "clause_articles": sorted(list(set(articles))),
    }


def node_search_verdicts(state: PipelineState, verdict_rag) -> dict:
    triggers = state.get("triggers", [])
    fact     = state.get("fact", {})

    # 사고 팩트 요약, 주요 쟁점, 그리고 면책 트리거를 병합하여 정교한 검색 쿼리 작성
    query_parts = []
    if fact.get("fact_summary"):
        query_parts.append(fact.get("fact_summary"))
    if fact.get("key_issue"):
        query_parts.append(fact.get("key_issue"))
    if triggers:
        query_parts.append(" ".join(triggers))
    query_parts.append("자동차보험 보상")

    query = " ".join(query_parts)

    # 면책 트리거가 있어도 사고 정황(Semantic) 벡터 검색을 기본적으로 수행
    rag_result = verdict_rag.answer(query, top_k=3)
    chunks     = rag_result.get("chunks", [])

    # 만약 벡터 검색으로 판례가 검색되지 않거나 부족한 경우, 트리거 기반 판례 검색으로 보강
    if len(chunks) < 3 and triggers:
        fallback_chunks = verdict_rag.search_by_triggers(triggers, top_k=3)
        # 중복 제거하며 병합
        seen = {c.get("case_num") for c in chunks if c.get("case_num")}
        for fc in fallback_chunks:
            if fc.get("case_num") not in seen:
                chunks.append(fc)
                seen.add(fc.get("case_num"))
        chunks = chunks[:3]
        # 판례 해석 다시 수행
        rag_result = verdict_rag.answer(query, context_chunks=chunks)

    return {
        "verdict_hits":   chunks,
        "verdict_answer": rag_result.get("answer", ""),
        "verdict_fault_ratio": rag_result.get("fault_ratio", ""),
    }



# ══════════════════════════════════════════════════════
#  P2 노드: 면부책 — 룰 엔진
# ══════════════════════════════════════════════════════

def node_liability_judge(state: PipelineState, clause_rag=None, verdict_rag=None, fault_rag=None, **_) -> dict:
    normalized    = state.get("normalized", {})
    fact          = state.get("fact", {})
    triggers      = state.get("triggers", [])
    verdict_hits  = state.get("verdict_hits", [])
    accident_type = normalized.get("accident_type", "") or state.get("text_result", {}).get("accident_type", "")

    standard = _find_standard(accident_type)

    fault_ratio = "산정 불가"
    applied_modifiers = []
    근거_약관 = []
    과실도표_페이지 = []

    # ── RAG 기반 과실비율 계산 ────────────────────────
    if fault_rag and accident_type:
        query = f"사고 유형: {accident_type}. 상황: {fact.get('fact_summary') or state.get('user_input', '')}. 추가정보: {state.get('extra_data', '')}"
        try:
            # 1. 과실도표 RAG 질의
            fault_result = fault_rag.answer(query, top_k=3)
            fault_ratio = fault_result.get("final_fault_ratio", "산정 불가")
            
            # 매칭된 페이지 정보 수집
            for chunk in fault_result.get("chunks", []):
                p = chunk.get("page")
                if p:
                    과실도표_페이지.append(f"p.{p} (유사도: {chunk.get('score', 0.0)*100:.1f}%)")
            
            if fault_ratio and fault_ratio != "산정 불가":
                # 2. 기본과실 파싱 및 수정요소 포맷팅
                base_fault = fault_result.get("base_fault", {})
                base_str = " / ".join(f"{k}: {v}%" for k, v in base_fault.items())
                applied_modifiers.append(f"기본 과실비율: {base_str}")
                
                for mod in fault_result.get("modifiers", []):
                    factor = mod.get("factor", "")
                    delta = mod.get("delta", 0)
                    try:
                        delta = int(str(delta).replace("%", "").strip())
                    except Exception:
                        delta = 0
                    target = mod.get("target", "")
                    direction = "증가" if delta > 0 else "감소"
                    applied_modifiers.append(f"수정요소 ({target} {factor}): {abs(delta)}% {direction} (근거: {mod.get('basis', '')})")
                    
                fault_code = fault_result.get("fault_code", "")
                if fault_code:
                    applied_modifiers.append(f"과실도표 근거: {fault_code}")
                
                # RAG 판례 교차검증 결론이 있다면 추가
                if state.get("verdict_answer"):
                    applied_modifiers.append(f"판례 교차검증: {state.get('verdict_answer')[:120]}...")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Fault RAG Error] {e}. Falling back to rule-based logic.")
            fault_ratio = "산정 불가"

    # ── Rule-based Fallback ───────────────────────────
    if fault_ratio == "산정 불가" or not fault_ratio:
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
        # "base_fault":        dict(ls["기본_과실"]) if standard else {},
        "base_fault": dict(standard["liability_standard"]["기본_과실"]) if (standard and "liability_standard" in standard) else {},
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
        "과실도표근거": 과실도표_페이지,
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
- 과실도표 근거 페이지:
  • 과실도표 책자 페이지: [과실도표 책자 페이지 번호] (실제 PDF [실제 PDF 페이지 번호]페이지)
  • 사고 유형 코드: [과실도표 코드 예: 차41-1 또는 차41] [사고 유형 명칭 예: 양 차량 주행 중 후방 추돌]
  • 기본 과실비율: [가해차/피해차 기본 과실비율]
  (과실도표근거 및 수정요소적용 정보를 바탕으로 위의 상세 형식에 맞추어 작성하고, 유사도가 표시된 다른 페이지들도 함께 기술하세요.)
- 담보별 판단: (표 형식)
- 근거 약관: ...
- 참고 판례: ...

## 판단 근거
(각 단계별로 어떤 키워드/데이터에 의해 결정됐는지 간결하게)

## 특이사항
(면책사유, 사고부담금, 수정요소 등. 없으면 '없음')"""

    report = _call("자동차보험 보상 보고서 작성 전문가. 숫자와 판단값은 절대 변경 금지.", user, max_tokens=1500)
    return {"report": report}


# ══════════════════════════════════════════════════════
#  P2 노드: 면책 요건 리뷰 대기 노드
# ══════════════════════════════════════════════════════

def node_trigger_review(state: PipelineState, **_) -> dict:
    """면책 조건 수동 리뷰를 위해 파이프라인 일시 대기"""
    return {"final_status": "면책 검토 대기"}
