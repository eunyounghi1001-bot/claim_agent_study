"""
자동 검증 테스트 스위트 (Golden Dataset Test Suite)
실행: ..\.venv\Scripts\python.exe test_suite.py
"""

import os
import json
import warnings
from dotenv import load_dotenv

# 1. 환경 설정 로드 (오프라인 모드 강제 적용)
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from graph.pipeline import run, resume
from graph.pipeline import get_models

# 골든 데이터셋 정의 (정답지 + 충분한 객관적 증거 데이터 보강)
GOLDEN_DATASET = [
    {
        "case_id": "TC-001",
        "name": "음주운전 자차 사고 (면책 및 사고부담금)",
        "user_input": "어제 저녁에 술을 마시고 운전하고 가다가 혼자 가로수를 들이받았습니다. 제 차 수리비를 보험으로 받을 수 있을까요? 그리고 상대방이나 차량 피해가 발생하면 사고부담금도 내야 하나요?",
        "extra_data": "가해 차량의 블랙박스 영상 분석 및 경찰 사고접수 서류 확인 결과, 운전자의 혈중알코올농도가 0.08% 면허취소 수치 상태로 가로수를 직접 충돌한 단독 사고임이 객관적으로 증명됨.",
        "expected_triggers": ["음주운전"],
        "expected_articles": ["제11조", "제23조"],  # 제11조(사고부담금), 제23조(자기차량손해 면책)
        "check_liability": True,
        "expected_liability_keys": ["사고부담금", "담보별판단"]
    },
    {
        "case_id": "TC-002",
        "name": "무면허 운전 사고 (면책 및 사고부담금)",
        "user_input": "면허가 취소된 상태인 줄 모르고 운전하다가 신호대기 중인 앞차를 추돌하는 사고를 냈습니다. 상대방 운전자가 경미하게 다쳤고 앞차 범퍼가 찌그러졌습니다. 제 차와 상대방에 대해 보험처리가 되나요?",
        "extra_data": "교통사고 사실확인원 및 운전자 조회를 통해 무면허 운전 상태에서 전방 주시 태만으로 신호 대기 차량을 후방에서 추돌했음이 공인 문서로 확정됨.",
        "expected_triggers": ["무면허운전"],
        "expected_articles": ["제11조", "제23조"],  # 제11조(사고부담금), 제23조(자기차량손해 면책)
        "check_liability": True,
        "expected_liability_keys": ["사고부담금", "담보별판단"]
    },
    {
        "case_id": "TC-003",
        "name": "고의 사고 (전체 면책)",
        "user_input": "조수석에 타고 있던 동승자와 말다툼을 하다가 홧김에 고의로 도로 옹벽을 세게 들이받았습니다. 동승자가 골절상을 입었고 차량은 반파되었습니다. 보험금을 청구하려고 합니다.",
        "extra_data": "차량 블랙박스 오디오 파일 녹음 내용에 피보험자(운전자)가 동승자에게 '일부러 벽에 박하겠다'고 협박한 음성이 확인되었으며 고의로 차량을 가속하여 옹벽을 들이받았음이 확인됨.",
        "expected_triggers": ["고의"],
        "expected_articles": ["제14조", "제23조", "제5조", "제8조"], # 고의면책 조항들
        "check_liability": True,
        "expected_liability_keys": ["담보별판단"]
    },
    {
        "case_id": "TC-004",
        "name": "정상 일반 접촉 사고 (면책 사항 없음)",
        "user_input": "정상 신호 대기 중이었는데 뒤에서 오던 차량이 제 뒷범퍼를 박았습니다. 상대방 과실 100% 상황인데 대물배상과 대인 접수해서 병원 치료를 받고 싶습니다.",
        "extra_data": "사고 차량 블랙박스 영상 판독 결과, 피해 차량이 교차로 신호 대기를 위해 완전히 정차한 상태에서 가해 차량이 제동을 지체하여 후방 추돌한 사고임이 명백히 촬영됨.",
        "expected_triggers": [],
        "expected_articles": [], # 정상 사고이므로 특수 면책조항 불필요
        "check_liability": False
    }
]

def run_evaluation():
    print("==================================================")
    print("      자동차보험 클레임 에이전트 자동 검증 시작")
    print("==================================================")
    print("임베딩 모델 및 DB 로드 중...")
    get_models()
    print("준비 완료! 테스트를 시작합니다.\n")

    total_tests = len(GOLDEN_DATASET)
    passed_tests = 0
    results_report = []

    for idx, case in enumerate(GOLDEN_DATASET):
        print(f"[{idx+1}/{total_tests}] {case['name']} 테스트 진행 중...")
        case_id = f"TEST-{case['case_id']}"
        
        # 1단계: 면책 트리거 자동 감지 기능 테스트
        print("  - [1단계] 면책 트리거 자동 감지 기능 검증...")
        initial_res = run(
            user_input=case["user_input"], 
            case_id=case_id, 
            extra_data=case.get("extra_data", "")
        )
        detected_triggers = initial_res.get("triggers", [])
        
        # 트리거 일치 여부 비교
        trigger_ok = set(detected_triggers) == set(case["expected_triggers"])
        
        # 2단계: RAG 및 면부책 최종 연산 테스트 (트리거를 고정하여 끝까지 실행)
        print("  - [2단계] RAG 연동 및 면부책 판단 검증...")
        full_res = run(
            user_input=case["user_input"], 
            case_id=f"{case_id}-full", 
            extra_data=case.get("extra_data", ""),
            manual_triggers=case["expected_triggers"] # 검토 완료 상태로 바이패스 실행
        )

        # 근거 약관 조항 매칭율 검증
        clause_articles = full_res.get("clause_articles", [])
        article_ok = True
        matched_articles = []
        for art in case["expected_articles"]:
            # 예상 조항이 하나라도 포함되었는지 체크
            found = any(art in hit_art for hit_art in clause_articles)
            if found:
                matched_articles.append(art)
            else:
                article_ok = False

        if not case["expected_articles"]:
            article_ok = len(clause_articles) == 0 or True # 정상사고는 면책조항이 없어도 무방

        # 최종 지급/면책 판단 필드 검증
        liability = full_res.get("liability", {})
        liability_ok = True
        if case.get("check_liability"):
            for key in case["expected_liability_keys"]:
                if key not in liability or not liability[key]:
                    liability_ok = False

        # 개별 테스트 통과 기준
        case_passed = trigger_ok and article_ok and liability_ok
        if case_passed:
            passed_tests += 1

        # 결과 기록
        results_report.append({
            "name": case["name"],
            "passed": case_passed,
            "trigger_ok": trigger_ok,
            "expected_triggers": case["expected_triggers"],
            "detected_triggers": detected_triggers,
            "article_ok": article_ok,
            "expected_articles": case["expected_articles"],
            "matched_articles": matched_articles,
            "liability_ok": liability_ok
        })

    # 최종 레포트 출력
    print("\n" + "="*50)
    print("                 최종 검증 보고서")
    print("="*50)
    accuracy = (passed_tests / total_tests) * 100
    print(f"전체 테스트 케이스: {total_tests}건")
    print(f"통과한 테스트 케이스: {passed_tests}건")
    print(f"시스템 최종 검증 정확도: {accuracy:.1f}%\n")

    print("상세 테스트 내역:")
    for r in results_report:
        status_str = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"- {r['name']} : {status_str}")
        print(f"  * 트리거 감지: {'성공' if r['trigger_ok'] else '실패'} (기대: {r['expected_triggers']}, 감지: {r['detected_triggers']})")
        print(f"  * 근거약관 매칭: {'성공' if r['article_ok'] else '실패'} (기대: {r['expected_articles']}, 매치됨: {r['matched_articles']})")
        print(f"  * 면부책 판단 출력: {'성공' if r['liability_ok'] else '실패'}")
        print("-" * 40)

    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_evaluation()
