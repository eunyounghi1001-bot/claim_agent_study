# ── nodes.py ─────────────────────────────────────────
# LangGraph 노드 함수 모음
# 각 노드는 PipelineState를 받아 업데이트할 딕셔너리를 반환
# ─────────────────────────────────────────────────────

import json
import os
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from graph.state import PipelineState

load_dotenv()
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# 이미지/영상 처리
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 면책 트리거 키워드 매핑
TRIGGER_KEYWORDS = {
    "음주운전":     ["음주", "음주운전", "혈중알코올"],
    "무면허운전":   ["무면허", "면허 없", "면허없"],
    "마약약물운전": ["마약", "약물운전"],
    "조치의무위반": ["조치의무", "도주", "뺑소니"],
    "고의":         ["고의", "의도적", "일부러"],
    "유상운송":     ["유상운송", "대리운전", "배달업"],
    "가족피해":     ["배우자", "부모", "자녀", "가족 동승"],
    "무단운전":     ["무단운전", "허락 없", "허락없"],
    "절취운전":     ["절취", "훔친 차", "도난차"],
    "운전자범위":   ["운전자 범위", "연령 한정", "연령한정"],
    "사기횡령":     ["사기", "횡령", "허위 사고"],
    "고지의무":     ["고지의무", "알릴의무"],
}

TRIGGER_ARTICLE_MAP = {
    "음주운전":     ["제11조"],
    "무면허운전":   ["제11조"],
    "마약약물운전": ["제11조"],
    "조치의무위반": ["제11조"],
    "고의":         ["제5조", "제8조", "제14조", "제19조", "제23조"],
    "유상운송":     ["제8조", "제14조", "제19조", "제23조"],
    "가족피해":     ["제8조"],
    "무단운전":     ["제8조"],
    "절취운전":     ["제8조", "제23조"],
    "운전자범위":   ["제8조"],
    "사기횡령":     ["제23조"],
    "고지의무":     ["제44조", "제53조"],
}

SURCHARGE_TABLE = {
    "음주운전":     {"대인I": 300, "대인II": 300, "대물": 100},
    "무면허운전":   {"대인I": 150, "대인II": 150, "대물": 50},
    "마약약물운전": {"대인I": 300, "대인II": 300, "대물": 100},
    "조치의무위반": {"대인I": 150, "대인II": 150, "대물": 50},
}


# ── LLM 호출 헬퍼 ────────────────────────────────────
def _call(llm, system: str, user: str, max_tokens: int = 1024) -> str:
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


# ── 보조 도구 ─────────────────────────────────────────
def _exif_time(path: str) -> str:
    if not HAS_PIL:
        return ""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return ""
        for tid, val in exif.items():
            if TAGS.get(tid) == "DateTimeOriginal":
                return str(val)
    except Exception:
        pass
    return ""


def _keyframes(video_path: str, n: int = 4) -> List[str]:
    import tempfile, os
    if not HAS_CV2:
        return []
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    tmp   = tempfile.mkdtemp()
    paths = []
    for idx in [int(total * i / n) for i in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            p = os.path.join(tmp, f"frame_{idx}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
    cap.release()
    return paths


GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _image_to_b64(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


FRAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)


def _vision_analyze(image_paths: List[str], context: str) -> str:
    """키프레임 이미지를 Groq vision API로 분석."""
    content = [{"type": "text", "text": f"다음은 자동차 사고 관련 영상/이미지입니다.\n\n[사고 접수 내용]\n{context}\n\n파손 부위, 충격 방향, 도로 상황, 사고 정황을 구체적으로 분석하세요."}]
    for p in image_paths[:4]:
        b64 = _image_to_b64(p)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    resp = _groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _save_frames(case_id: str, image_paths: List[str]) -> List[str]:
    """키프레임을 static/frames/에 저장하고 URL 목록 반환."""
    import shutil
    urls = []
    for i, src in enumerate(image_paths[:4]):
        fname = f"{case_id}_frame{i}.jpg"
        dst   = os.path.join(FRAMES_DIR, fname)
        shutil.copy2(src, dst)
        urls.append(f"/static/frames/{fname}")
    return urls


def _describe_media(media_paths: List[str], youtube_url: str, user_input: str = "", case_id: str = "") -> tuple:
    """(media_result: str, keyframe_urls: List[str]) 반환."""
    all_images = []
    meta_lines = []

    for path in media_paths:
        suffix = Path(path).suffix.lower()
        size   = Path(path).stat().st_size // 1024
        if suffix in (".jpg", ".jpeg", ".png", ".webp"):
            t = _exif_time(path)
            meta_lines.append(f"[이미지] {Path(path).name} ({size}KB)" + (f" 촬영: {t}" if t else ""))
            all_images.append(path)
        elif suffix in (".mp4", ".mov", ".avi"):
            frames = _keyframes(path)
            meta_lines.append(f"[영상] {Path(path).name} ({size}KB) → 키프레임 {len(frames)}장 추출")
            all_images.extend(frames)
    if youtube_url:
        meta_lines.append(f"[YouTube] {youtube_url}")

    if not meta_lines:
        return "제출된 자료 없음", []

    if all_images:
        keyframe_urls = _save_frames(case_id, all_images) if case_id else []
        vision_result = _vision_analyze(all_images, user_input)
        text = "\n".join(meta_lines) + "\n\n[Vision 분석 결과]\n" + vision_result
        return text, keyframe_urls

    return "\n".join(meta_lines), []


# ══════════════════════════════════════════════════════
#  P1 노드: 자료수집 Agent
# ══════════════════════════════════════════════════════

def node_text_parse(state: PipelineState, llm=None) -> dict:
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
    raw    = _call(llm, "자동차 사고 자료수집 전문가. JSON만 출력.", user)
    result = _parse_json(raw)
    return {"text_result": result}


def node_media_analyze(state: PipelineState, llm=None) -> dict:
    media_desc, keyframe_urls = _describe_media(
        state.get("media_paths") or [],
        state.get("youtube_url") or "",
        state.get("user_input") or "",
        state.get("case_id") or "",
    )
    if media_desc == "제출된 자료 없음":
        return {"media_result": "제출된 자료 없음 — 자료 요청 필요", "keyframe_urls": []}

    return {"media_result": media_desc, "keyframe_urls": keyframe_urls}


def node_normalize(state: PipelineState, llm=None) -> dict:
    media_result = state['media_result']
    has_vision = "[Vision 분석 결과]" in media_result

    user = f"""아래 두 결과를 하나의 JSON으로 통합하세요.

[텍스트 파싱]
{json.dumps(state['text_result'], ensure_ascii=False)}

[미디어 판독]
{media_result}

{"⚠️ Vision AI가 실제 영상/이미지를 분석한 결과가 포함되어 있습니다. media_summary에 분석 내용을 상세히 포함하고 reliability를 '상'으로 설정하세요." if has_vision else ""}

JSON만 출력:
{{
  "accident_type"    : "사고 유형",
  "dispatch_yn"      : "O 또는 X",
  "text_summary"     : "접수 내용 요약",
  "media_summary"    : "미디어 판독 요약 (없으면 '제출 자료 없음')",
  "first_request_msg": "1차 자료 요청문 (없으면 빈 문자열)",
  "media_analyzed"   : {"true" if has_vision else "false"},
  "metadata"         : {{"source": "콜센터 접수 및 고객 제출 자료", "reliability": "상 또는 하"}}
}}"""
    raw    = _call(llm, "자동차 사고 데이터 정규화. JSON만 출력.", user)
    result = _parse_json(raw)
    return {"normalized": result}


# ══════════════════════════════════════════════════════
#  P1 노드: 사실확정 Agent
# ══════════════════════════════════════════════════════

def node_fact_check(state: PipelineState, llm=None, search_agent=None) -> dict:
    media_result = state.get('media_result', '')
    has_vision = "[Vision 분석 결과]" in media_result
    vision_note = "\n⚠️ Vision AI가 실제 영상/이미지를 분석한 결과가 있습니다. 이는 블랙박스·CCTV급 객관 증거로 간주하고 status를 '충분'으로 판단하세요." if has_vision else ""

    # 사고 유형별 판단 기준 조회
    standard = None
    if search_agent:
        accident_type = state.get('normalized', {}).get('accident_type', '') or state.get('text_result', {}).get('accident_type', '')
        if accident_type:
            standard = search_agent.search_standard(accident_type)

    standard_ctx = ""
    if standard:
        import json as _j
        standard_ctx = f"""
[사고 유형별 판단 기준]
- 핵심 쟁점: {', '.join(standard['key_issues'])}
- 사실확정 충분 조건: {standard['fact_check_criteria']['충분_조건']}
- 사실확정 부족 조건: {standard['fact_check_criteria']['부족_조건']}
- 판단 기준: {standard['fact_check_criteria']['판단_기준']}
- 필요 증거 (우선순위): {', '.join(f"①{e['item']}({e['reason']})" for e in sorted(standard['required_evidence'], key=lambda x: x['priority'])[:2])}
"""

    user = f"""수집된 사고 데이터로 사실관계를 확정하세요.{vision_note}
{standard_ctx}
[사고 데이터]
{json.dumps(state['normalized'], ensure_ascii=False, indent=2)}

[Vision 영상 분석 원문]
{media_result if has_vision else '없음'}

[추가 제출 자료]
{state.get('extra_data') or '없음'}

[증거 우선순위]
객관기록(블랙박스·CCTV·EXIF·Vision분석) > 현장사진 > 현장출동 보고서 > 진술

JSON만 출력:
{{
  "key_issue"    : "핵심 쟁점 1문장",
  "status"       : "충분 또는 부족",
  "fact_summary" : "확정된 사실관계 2~3문장 (부족이면 빈 문자열)",
  "confidence"   : "상/중/하 (충분일 때만)",
  "contradiction": "모순 기술 또는 빈 문자열",
  "unresolved"   : ["미확정 쟁점"],
  "missing_info" : [{{"item": "자료명", "reason": "필요 이유", "decisiveness": 9}}]
}}"""
    raw    = _call(llm, "자동차 사고 사실확정 전문가. JSON만 출력.", user, max_tokens=1024)
    result = _parse_json(raw)
    return {"fact": result}


def node_extra_request(state: PipelineState, llm=None) -> dict:
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
    raw    = _call(llm, "자동차 사고 추가자료 요청 담당자. JSON만 출력.", user)
    result = _parse_json(raw)
    return {
        "extra_request": result,
        "retry_count":   state.get("retry_count", 0) + 1,
    }


# ══════════════════════════════════════════════════════
#  P2 노드: 검색 Agent
# ══════════════════════════════════════════════════════

def node_detect_triggers(state: PipelineState) -> dict:
    """사실관계에서 면책 트리거 감지 (LLM 불필요 — 키워드 매칭)"""
    text     = json.dumps(state["fact"], ensure_ascii=False).lower()
    triggers = []
    for trigger, keywords in TRIGGER_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            triggers.append(trigger)
    return {"triggers": triggers}


def node_search_clauses(state: PipelineState, search_agent) -> dict:
    """약관 조항 검색"""
    triggers = state.get("triggers", [])
    if triggers:
        chunks = search_agent.search_exemption_clauses(triggers)
    else:
        query  = state["fact"].get("fact_summary", "") + " " + state["fact"].get("key_issue", "")
        chunks = search_agent.search_clauses(query, top_k=5)
    return {"clause_hits": chunks}


def node_search_verdicts(state: PipelineState, search_agent) -> dict:
    """판결문 검색"""
    triggers = state.get("triggers", [])
    if triggers:
        chunks = search_agent.search_verdicts_by_triggers(triggers, top_k=3)
    else:
        query  = state["fact"].get("fact_summary", "") + " 자동차보험 보상"
        chunks = search_agent.search_verdicts(query, top_k=3)
    return {"verdict_hits": chunks}


# ══════════════════════════════════════════════════════
#  P2 노드: 면부책 Agent
# ══════════════════════════════════════════════════════

def node_liability_judge(state: PipelineState, llm=None, search_agent=None) -> dict:
    """약관 + 판례 + 판단기준 기반 면부책 판단"""
    triggers      = state.get("triggers", [])
    clause_hits   = state.get("clause_hits", [])
    verdict_hits  = state.get("verdict_hits", [])

    # 사고부담금 산정
    surcharge = {"대인I": 0, "대인II": 0, "대물": 0}
    for t in triggers:
        if t in SURCHARGE_TABLE:
            for k, v in SURCHARGE_TABLE[t].items():
                surcharge[k] = max(surcharge[k], v)

    clause_ctx  = "\n\n".join(
        f"[{c['article']} {c['title']}]\n{c['text'][:500]}"
        for c in clause_hits
    )
    verdict_ctx = "\n\n".join(
        f"[{v['case_num']} {v['court']}]\n판시: {v['issue'][:200]}\n요지: {v['summary'][:300]}"
        for v in verdict_hits[:3]
    )

    # 사고 유형별 표준 판단 기준 주입
    standard = None
    if search_agent:
        accident_type = state.get('normalized', {}).get('accident_type', '') or state.get('text_result', {}).get('accident_type', '')
        if accident_type:
            standard = search_agent.search_standard(accident_type)

    standard_ctx = ""
    if standard:
        ls = standard['liability_standard']
        standard_ctx = f"""
[사고 유형별 표준 과실 기준]
- 기본 과실: {ls['기본_과실']}
- 수정 요소: {json.dumps(ls['수정_요소'], ensure_ascii=False)}
- 담보별 표준 판단: {json.dumps(ls['담보별_판단'], ensure_ascii=False)}
- 근거 약관: {', '.join(ls['근거_약관'])}
"""

    user = f"""확정된 사실관계와 약관·판례를 바탕으로 면부책을 판단하세요.
{standard_ctx}
[확정 사실관계]
{json.dumps(state['fact'], ensure_ascii=False, indent=2)}

[감지된 면책 트리거]
{triggers}

[관련 약관 조항]
{clause_ctx}

[관련 판례]
{verdict_ctx}

⚠️ 판단 원칙:
- 음주·무면허·마약·조치의무위반: 부책(보상)이나 사고부담금 부과 (제11조)
- 고의·유상운송·가족피해·무단운전 등: 해당 담보 면책(미지급)
- 대인배상I: 고의 외 면책 불가 — 항상 부책

JSON만 출력:
{{
  "liability"   : "부책(보상) 또는 면책(미지급)",
  "면책사유"    : ["면책(미지급) 사유만. 사고부담금 대상 제외"],
  "약관근거"    : ["제8조①1 형식"],
  "판례근거"    : ["사건번호"],
  "담보별판단"  : {{"대인배상I": "부책/면책", "대인배상II": "부책/면책", "대물배상": "부책/면책", "자기신체": "부책/면책"}},
  "과실비율"    : "좌회전차 X% : 직진차 Y% (해당되는 경우)",
  "근거요약"    : "2~3문장. 사고부담금 대상이면 부책이지만 부담금 부과됨 명시"
}}"""

    raw    = _call(llm, "자동차보험 면부책 전문가. JSON만 출력.", user, max_tokens=1024)
    result = _parse_json(raw)
    result["사고부담금"] = surcharge if any(surcharge.values()) else {}
    result["분기"]       = "협상(면책통보)" if result.get("liability") == "면책(미지급)" else "P3 과실판단"

    return {"liability": result, "final_status": "완료"}
