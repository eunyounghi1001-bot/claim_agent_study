# ── state.py ─────────────────────────────────────────
# 전체 파이프라인 공유 상태 정의
# ─────────────────────────────────────────────────────

from typing import List, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    # ── 입력 ─────────────────────────────────────────
    case_id:      str
    user_input:   str
    media_paths:  List[str]
    youtube_url:  str
    extra_data:   str          # 고객 추가 제출 자료

    # ── P1: 자료수집 Agent ───────────────────────────
    text_result:   dict        # text_parse 결과
    media_result:  str         # media_analyze 결과
    normalized:    dict        # normalize 결과
    keyframe_urls: List[str]   # Vision 분석에 사용된 키프레임 URL 목록

    # ── P1: 사실확정 Agent ───────────────────────────
    fact:         dict         # fact_check 결과
    extra_request: Optional[dict]  # 추가자료 요청 내용
    retry_count:  int

    # ── P2: 검색 Agent ───────────────────────────────
    clause_hits:  List[dict]   # 약관 조항 검색 결과
    verdict_hits: List[dict]   # 판결문 검색 결과
    triggers:     List[str]    # 감지된 면책 트리거

    # ── P2: 면부책 Agent ─────────────────────────────
    liability:    dict         # 최종 면부책 판단 결과

    # ── 파이프라인 제어 ──────────────────────────────
    final_status: str          # 사실관계 확정 / 추가자료 대기 / 완료
    error:        str          # 오류 메시지
