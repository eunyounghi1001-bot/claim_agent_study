import os
import sys
import json
import uuid
import warnings

# 프로젝트 경로를 python path에 추가하여 내부 모듈 임포트 가능하도록 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from graph.pipeline import build_graph, get_models
from graph.state import PipelineState

def print_separator():
    print("\n" + "=" * 80 + "\n")

def print_node_header(node_name, file_path, function_name, description):
    print(f"\n[▶ 노드 실행] : \033[1;36m{node_name}\033[0m")
    print(f"  - 소스 파일: \033[1;32m{file_path}\033[0m")
    print(f"  - 실행 함수: \033[1;32m{function_name}()\033[0m")
    print(f"  - 역할 설명: {description}")
    print("-" * 50)

def main():
    print("=" * 80)
    print("      Claim Agent Study - 파이프라인 흐름 추적 & 디버깅 도구")
    print("=" * 80)
    
    print("임베딩 모델 및 RAG 데이터베이스 로딩 중... 잠시만 기다려주세요...")
    get_models()
    print("로딩 완료!\n")
    
    # 1. 사용자로부터 사고 접수 내용 입력받기
    print("[단계 1] 사고 내용 입력")
    print("예시: 어제 저녁에 술을 마시고 운전하고 가다가 혼자 가로수를 들이받았습니다. 제 차 수리비를 보험으로 받을 수 있을까요?")
    user_input = input("사고 내용을 입력해 주세요: ").strip()
    if not user_input:
        print("입력된 내용이 없어 종료합니다.")
        return
        
    video_url = input("영상 URL (선택사항, 없을 시 Enter): ").strip()
    extra_data = input("추가 정보/증거 데이터 (선택사항, 없을 시 Enter): ").strip()
    
    case_id = "DEBUG-" + uuid.uuid4().hex[:8].upper()
    print(f"\n생성된 케이스 ID: {case_id}")
    
    # 그래프 빌드
    graph = build_graph()
    config = {"configurable": {"thread_id": case_id}}
    
    # 초기 상태 설정
    init_state = {
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
        "reviewed":           False,
    }
    
    # 노드별 메타데이터 매핑
    node_metadata = {
        "text_parse": {
            "file": "graph/nodes.py",
            "func": "node_text_parse",
            "desc": "사용자가 입력한 원본 접수 내용을 LLM을 통해 분석하여 핵심 요약, 현장 출동 여부, 사고 유형을 추출합니다."
        },
        "normalize": {
            "file": "graph/nodes.py",
            "func": "node_normalize",
            "desc": "텍스트 파싱 결과와 제출된 미디어 자료 등을 정형화된 JSON 형태로 정리합니다."
        },
        "fact_check": {
            "file": "graph/nodes.py",
            "func": "node_fact_check",
            "desc": "사고 상황 및 제출된 증거들을 토대로 과실과 면부책을 판단하기에 사실관계 정보가 충분한지 검증합니다."
        },
        "extra_request": {
            "file": "graph/nodes.py",
            "func": "node_extra_request",
            "desc": "사실관계 확인 결과 정보가 부족한 경우, 고객에게 필요한 추가 자료를 정중하게 요청하는 문서를 생성합니다."
        },
        "detect_triggers": {
            "file": "graph/nodes.py",
            "func": "node_detect_triggers",
            "desc": "사고 텍스트에서 면책 사유(예: 음주운전, 무면허운전, 고의 등)와 관련된 키워드를 스캔하여 트리거합니다."
        },
        "trigger_review": {
            "file": "graph/nodes.py",
            "func": "node_trigger_review",
            "desc": "[대기 상태] 자동 감지된 면책 트리거 항목을 사용자(담당자)가 직접 확인 및 수정할 수 있게 인터럽트합니다."
        },
        "search_clauses": {
            "file": "graph/nodes.py (내부적으로 agents/clause_rag.py -> ClauseRAG 사용)",
            "func": "node_search_clauses",
            "desc": "감지된 면책 트리거 또는 사고 유형에 대응하는 보험 약관 조항을 ChromaDB에서 검색하고 LLM으로 보상 여부를 분석합니다."
        },
        "search_verdicts": {
            "file": "graph/nodes.py (내부적으로 agents/verdict_rag.py -> VerdictRAG 사용)",
            "func": "node_search_verdicts",
            "desc": "사고 유형 및 상황 정보를 활용하여 유사 법원 판결문을 ChromaDB에서 검색하고 판례 기반 과실비율 등을 분석합니다."
        },
        "liability_judge": {
            "file": "graph/nodes.py (내부적으로 agents/fault_rag.py -> FaultRAG 사용)",
            "func": "node_liability_judge",
            "desc": "약관 해석 결과, 유사 판결문, 그리고 과실도표 RAG 검색 결과를 종합하여 최종 면부책 판단 및 과실비율을 산정합니다."
        },
        "format_output": {
            "file": "graph/nodes.py",
            "func": "node_format_output",
            "desc": "전체 처리된 중간 결과와 판단 근거를 종합하여 사람이 읽기 좋은 최종 사고보상 보고서를 작성합니다."
        }
    }
    
    def run_graph_stream(state_to_run):
        # LangGraph 실행 및 스트리밍
        events = graph.stream(state_to_run, config, stream_mode="updates")
        for event in events:
            for node_name, state_update in event.items():
                meta = node_metadata.get(node_name, {"file": "알 수 없음", "func": "알 수 없음", "desc": "설명 없음"})
                print_node_header(node_name, meta["file"], meta["func"], meta["desc"])
                
                # 노드가 리턴한 주요 상태 변화값들을 예쁘게 출력
                if isinstance(state_update, dict):
                    for key, val in state_update.items():
                        print(f"  └─ 업데이트된 상태 필드: \033[1;33m{key}\033[0m")
                        if isinstance(val, dict):
                            print(json.dumps(val, ensure_ascii=False, indent=4))
                        elif isinstance(val, list):
                            print(f"     목록 개수: {len(val)}개")
                            if len(val) > 0 and isinstance(val[0], dict):
                                print(json.dumps(val[:2], ensure_ascii=False, indent=4))
                                if len(val) > 2:
                                    print(f"     ... 외 {len(val)-2}개 생략")
                            else:
                                print(f"     값: {val}")
                        else:
                            print(f"     값: {val}")
                else:
                    print(f"  └─ 업데이트 값 ({type(state_update).__name__}): {state_update}")
                print("-" * 50)
                
    # 첫 실행 시작
    print_separator()
    print("▶ 파이프라인 최초 실행 중...")
    run_graph_stream(init_state)
    
    # 인터럽트 상태 체크 및 루프 돌기
    while True:
        state_info = graph.get_state(config)
        next_nodes = state_info.next
        
        if not next_nodes:
            # 더 이상 실행할 노드가 없음 (종료됨)
            break
            
        current_values = state_info.values
        print_separator()
        print(f"⚠️ [대기 상태 (Interrupt)] 파이프라인 실행 중지 상태입니다. 다음 예정 노드: {next_nodes}")
        
        # 1. extra_request 노드 직후 대기 (추가 자료 입력 필요)
        if "fact_check" in next_nodes and current_values.get("extra_request"):
            req_msg = current_values["extra_request"].get("request_msg", "추가 자료가 필요합니다.")
            print(f"\n[고객 추가자료 요청 메시지]:\n\033[1;35m{req_msg}\033[0m\n")
            print("사실관계 확정을 위해 추가 자료/증거 텍스트를 입력해 주세요. (예: 경찰 사고접수 서류 내용 등)")
            extra_input = input("추가 자료 입력 (종료하려면 'exit' 입력): ").strip()
            
            if extra_input.lower() == 'exit':
                print("디버깅을 종료합니다.")
                break
                
            # 상태 업데이트 후 재개
            graph.update_state(
                config,
                {
                    "extra_data": extra_input,
                    "retry_count": current_values.get("retry_count", 0)
                },
                as_node="extra_request"
            )
            print("\n▶ 추가 데이터를 적용하고 파이프라인을 재개합니다...")
            run_graph_stream(None)
            
        # 2. trigger_review 노드 직후 대기 (면책 트리거 직접 검토)
        elif "search_clauses" in next_nodes:
            detected_triggers = current_values.get("triggers", [])
            print(f"\n[자동 감지된 면책 트리거 목록]: \033[1;31m{detected_triggers}\033[0m")
            print("이 트리거 목록을 수정하거나 그대로 진행하시겠습니까?")
            print("1. 그대로 진행 (Enter 입력)")
            print("2. 트리거 직접 입력/수정 (쉼표로 구분하여 입력, 예: 음주운전, 고의)")
            user_choice = input("선택: ").strip()
            
            final_triggers = detected_triggers
            if user_choice and user_choice != "1":
                final_triggers = [t.strip() for t in user_choice.split(",") if t.strip()]
                
            graph.update_state(
                config,
                {
                    "triggers": final_triggers,
                    "reviewed": True
                },
                as_node="trigger_review"
            )
            print(f"\n▶ 면책 트리거 검토 완료 (최종 트리거: {final_triggers}). 파이프라인을 재개합니다...")
            run_graph_stream(None)
        else:
            # 그 외의 인터럽트 처리
            print("기타 대기 지점입니다. 재개합니다.")
            run_graph_stream(None)
            
    # 최종 상태 출력
    final_state = graph.get_state(config).values
    print_separator()
    print("\033[1;32m[★ 최종 처리 완료] 최종 보상 보고서 출력\033[0m")
    print("-" * 80)
    print(final_state.get("report", "레포트 생성 실패"))
    print("-" * 80)
    print_separator()

if __name__ == "__main__":
    main()
