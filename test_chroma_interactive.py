"""
ChromaDB RAG 검색 터미널 검증 스크립트
실행: ..\.venv\Scripts\python.exe test_chroma_interactive.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import warnings
import chromadb
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = BASE_DIR / "index" / "chromadb"

def main():
    print("==================================================")
    # 1. 모델 로딩
    print("임베딩 모델 (bge-m3) 로딩 중... (오프라인 모드)")
    model = SentenceTransformer("BAAI/bge-m3")
    print("임베딩 모델 로드 완료!")

    # 2. ChromaDB 로딩
    print(f"ChromaDB 연결 중: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    try:
        clauses_col = client.get_collection("clauses")
        verdicts_col = client.get_collection("verdicts")
        print("ChromaDB 컬렉션 로드 완료!")
    except Exception as e:
        print(f"[오류] 컬렉션을 찾을 수 없습니다. 마이그레이션이 정상 완료되었는지 확인해 주세요. 상세: {e}")
        return
    print("==================================================\n")

    while True:
        print("------ RAG 검증 메뉴 ------")
        print("1. 약관 (Clauses) 검색 검증")
        print("2. 판례 (Verdicts) 검색 검증")
        print("3. 종료")
        choice = input("선택 (1~3): ").strip()

        if choice == '3':
            print("검증 프로그램을 종료합니다.")
            break
        elif choice not in ['1', '2']:
            print("잘못된 선택입니다. 다시 입력해 주세요.\n")
            continue

        query = input("\n검색할 질의어(예: 음주운전 면책, 사고부담금 등)를 입력하세요:\n> ").strip()
        if not query:
            print("질의어가 비어 있습니다.\n")
            continue

        print("\n검색 벡터 연산 및 코사인 유사도 계산 중...")
        # 쿼리 임베딩
        vec = model.encode([query]).astype("float32")
        faiss.normalize_L2(vec)
        query_vec = vec[0].tolist()

        if choice == '1':
            # 약관 검색
            results = clauses_col.query(
                query_embeddings=[query_vec],
                n_results=3,
                include=["metadatas", "documents", "distances"]
            )
            print(f"\n=== 약관 RAG 검색 결과 (상위 3개) ===")
            if results and results["ids"] and results["ids"][0]:
                for idx in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][idx]
                    doc = results["documents"][0][idx]
                    dist = results["distances"][0][idx]
                    score = 1.0 - dist  # 코사인 유사도
                    
                    print(f"\n[{idx+1}] {meta.get('article', '')} {meta.get('title', '')}")
                    print(f"    - 페이지: {meta.get('page', '')}p")
                    print(f"    - 면책조항 여부: {meta.get('is_exemption', False)}")
                    print(f"    - 유사도 점수: {score*100:.2f}%")
                    print(f"    - 텍스트 요약: {doc[:200]}...")
            else:
                print("검색 결과가 없습니다.")
        else:
            # 판례 검색
            results = verdicts_col.query(
                query_embeddings=[query_vec],
                n_results=3,
                include=["metadatas", "documents", "distances"]
            )
            print(f"\n=== 판례 RAG 검색 결과 (상위 3개) ===")
            if results and results["ids"] and results["ids"][0]:
                for idx in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][idx]
                    doc = results["documents"][0][idx]
                    dist = results["distances"][0][idx]
                    score = 1.0 - dist
                    
                    print(f"\n[{idx+1}] {meta.get('case_num', '')} [{meta.get('court', '')}]")
                    print(f"    - 사건명: {meta.get('case_name', '')}")
                    print(f"    - 선고일자: {meta.get('date', '')}")
                    print(f"    - 유사도 점수: {score*100:.2f}%")
                    print(f"    - 주요 키워드: {meta.get('keywords', '')}")
                    print(f"    - 판결요약: {meta.get('summary', '')[:250]}...")
            else:
                print("검색 결과가 없습니다.")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
