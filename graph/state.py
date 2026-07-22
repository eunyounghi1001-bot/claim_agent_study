# ── state.py ─────────────────────────────────────────
# 전체 파이프라인 공유 상태 정의
# ─────────────────────────────────────────────────────

from typing import List, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    # ── 입력 ─────────────────────────────────────────
    case_id:      str
    user_input:   str
    video_url:    str          # 영상 URL (블랙박스 등)
    extra_data:   str          # 고객 추가 제출 자료

    # ── P1: 자료수집 Agent ───────────────────────────
    text_result:   dict        # text_parse 결과
    normalized:    dict        # normalize 결과

    # ── P1: 사실확정 Agent ───────────────────────────
    fact:          dict        # fact_check 결과
    extra_request: Optional[dict]  # 추가자료 요청 내용
    retry_count:   int

    # ── P2: 검색 Agent ───────────────────────────────
    clause_hits:     List[dict]       # 약관 조항 검색 결과
    clause_answer:   str              # 약관 RAG LLM 해석
    clause_articles: List[str]        # 약관 RAG 근거 조항
    verdict_hits:    List[dict]       # 판결문 검색 결과
    verdict_answer:  str              # 판례 RAG LLM 해석
    verdict_fault_ratio: str          # 판례 RAG 과실비율
    triggers:        List[str]        # 감지된 면책 트리거

    # ── P2: 면부책 Agent ─────────────────────────────
    liability:     dict        # 최종 면부책 판단 결과

    # ── P3: 포매팅 ───────────────────────────────────
    report:        str         # 자연어 보고서

    # ── 판단 근거 히스토리 ────────────────────────────
    trace:         List[dict]  # 노드별 판단 근거 누적

    # ── 파이프라인 제어 ──────────────────────────────
    final_status:  str         # 사실관계 확정 / 추가자료 대기 / 완료
    error:         str         # 오류 메시지
    reviewed:      bool        # 면책 검토 완료 여부

