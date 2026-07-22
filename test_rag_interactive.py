import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 프로젝트 경로를 python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.nodes import _call
from agents.clause_rag import ClauseRAG
from agents.verdict_rag import VerdictRAG

def main():
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        print("에러: .env 파일에 GROQ_API_KEY가 설정되지 않았습니다.")
        return

    print("임베딩 모델(BAAI/bge-m3) 및 RAG 인덱스 로드 중...")
    model = SentenceTransformer('BAAI/bge-m3')
    clause_rag = ClauseRAG(model, _call)
    verdict_rag = VerdictRAG(model, _call)
    print("로드 완료!\n")

    while True:
        print("="*60)
        print(" 테스트할 RAG 번호를 선택하세요 (종료하려면 exit 입력)")
        print(" [1] 약관 RAG (보험사 면책 및 보상 여부)")
        print(" [2] 판례 RAG (사고 상황별 과실비율 및 판례)")
        print("="*60)
        
        choice = input("선택 (1 또는 2): ").strip()
        if choice.lower() == 'exit':
            print("테스트를 종료합니다.")
            break
            
        if choice not in ['1', '2']:
            print("잘못된 선택입니다. 1, 2, 또는 exit를 입력하세요.\n")
            continue
            
        query = input("검색할 질문을 입력하세요: ").strip()
        if not query:
            print("질문이 비어있습니다. 다시 입력해주세요.\n")
            continue
            
        print("\nRAG 검색 및 Groq LLM 추론 진행 중...")
        try:
            if choice == '1':
                res = clause_rag.answer(query)
                print("\n" + "-"*40)
                print("[약관 RAG 결과]")
                print(f"답변: {res.get('answer')}")
                print(f"면책(보상 제외) 여부: {'면책 (보상 불가)' if res.get('is_exempt') else '부책 (보상 대상)'}")
                print(f"근거 약관 조항: {res.get('articles')}")
                print("-"*40)
            elif choice == '2':
                res = verdict_rag.answer(query)
                print("\n" + "-"*40)
                print("[판례 RAG 결과]")
                print(f"답변: {res.get('answer')}")
                print(f"도출 과실비율: {res.get('fault_ratio') if res.get('fault_ratio') else '언급 없음'}")
                print(f"근거 판례번호: {res.get('case_nums')}")
                print("-"*40)
        except Exception as e:
            print(f"\n오류 발생: {e}")
        print()

if __name__ == "__main__":
    main()
