# ── indexer.py ───────────────────────────────────────
# 약관(조항 단위) + 판결문 인덱스 구축 모듈
# 실행: python agents/indexer.py  (최초 1회)
# ─────────────────────────────────────────────────────

import os, re, pickle, warnings
import fitz
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).parent.parent
PDF_PATH     = Path("/home/u313370/다운로드/개인용자동차보험약관(2026.06.10).pdf")
VERDICT_DIR  = BASE_DIR / "판결문"
INDEX_DIR    = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

CLAUSE_INDEX_PATH  = INDEX_DIR / "clause_index.pkl"    # 약관 조항 인덱스
VERDICT_INDEX_PATH = INDEX_DIR / "verdict_index.pkl"   # 판결문 인덱스

# 담보별 핵심 조항 매핑
COVERAGE_MAP = {
    "대인배상I":     ["제3조", "제4조", "제5조"],
    "대인배상II":    ["제6조", "제7조", "제8조", "제9조", "제10조"],
    "대물배상":      ["제6조", "제7조", "제8조", "제9조", "제10조"],
    "사고부담금":    ["제11조"],
    "자기신체사고":  ["제12조", "제13조", "제14조", "제15조"],
    "무보험차상해":  ["제17조", "제18조", "제19조", "제20조"],
    "자기차량손해":  ["제21조", "제22조", "제23조", "제24조"],
}

# 면책 조항 바로 참조
EXEMPTION_MAP = {
    "대인배상I":    "제5조",
    "대인배상II":   "제8조",
    "대물배상":     "제8조",
    "사고부담금":   "제11조",
    "자기신체사고": "제14조",
    "무보험차상해": "제19조",
    "자기차량손해": "제23조",
}


# ── 1. 약관 조항 단위 청킹 ───────────────────────────
def load_clause_chunks(pdf_path: Path) -> list:
    """약관 PDF를 조항(제X조) 단위로 청킹"""
    doc = fitz.open(str(pdf_path))

    # 전체 텍스트 + 페이지 위치 정보 수집
    full_text = ""
    page_offsets = []  # (offset, page_num)
    for i, page in enumerate(doc):
        offset = len(full_text)
        text = page.get_text()
        full_text += text
        page_offsets.append((offset, i + 1))

    def get_page(pos):
        page = 1
        for offset, pnum in page_offsets:
            if offset <= pos:
                page = pnum
            else:
                break
        return page

    # 제X조 패턴으로 분할
    pattern = re.compile(r'(제\d+조)\s*[\(（]?([^\)\n\r]{0,40})')
    matches = list(pattern.finditer(full_text))

    chunks = []
    for idx, m in enumerate(matches):
        art_num = m.group(1)       # 제8조
        art_title = m.group(2).strip()  # 보상하지 않는 손해
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()

        # 너무 짧거나(목차 등) 너무 긴 경우 건너뜀
        if len(text) < 50 or len(text) > 8000:
            continue

        # 담보 분류
        coverage = []
        for cov, arts in COVERAGE_MAP.items():
            if art_num in arts:
                coverage.append(cov)

        # 면책 여부
        is_exemption = any(
            EXEMPTION_MAP.get(c) == art_num for c in coverage
        ) or "보상하지" in art_title

        chunks.append({
            "text": text,
            "article": art_num,
            "title": art_title,
            "page": get_page(start),
            "coverage": coverage,
            "is_exemption": is_exemption,
            "source": "약관",
        })

    return chunks


# ── 2. 판결문 청킹 ───────────────────────────────────
def parse_verdict_file(filepath: Path) -> dict | None:
    """판결문 TXT 파일 파싱"""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    def extract(pattern, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    case_name   = extract(r'사건명[:\s]+(.+)')
    case_num    = extract(r'사건번호[:\s]+(.+)')
    court       = extract(r'법원명[:\s]+(.+)')
    date        = extract(r'선고일자[:\s]+(\d+)')
    issue       = extract(r'\[판시사항\]\s*([\s\S]*?)(?=\[판결요지\]|\[참조조문\]|\[판례내용\]|$)')
    summary     = extract(r'\[판결요지\]\s*([\s\S]*?)(?=\[참조조문\]|\[판례내용\]|$)')
    content     = extract(r'\[판례내용\]\s*([\s\S]*?)$')

    # 파일명에서 사건번호 추출 (파싱 실패 시 보완)
    fname = filepath.stem
    if not case_num:
        m = re.search(r'(\d{2,4}[다도가나]\d+)', fname)
        if m:
            case_num = m.group(1)

    # 핵심 쟁점 키워드 추출
    keywords = []
    trigger_words = ["음주운전", "무면허", "무단운전", "유상운송", "고의", "절취",
                     "가족", "고지의무", "운전자범위", "사고부담금", "면책", "구상"]
    for kw in trigger_words:
        if kw in text:
            keywords.append(kw)

    # 검색용 통합 텍스트
    search_text = f"{case_name} {issue} {summary}".strip()
    if len(search_text) < 30:
        search_text = text[:1000]

    return {
        "text": search_text,
        "full_text": text[:3000],  # 답변 생성용 (처음 3000자)
        "case_name": case_name,
        "case_num": case_num,
        "court": court,
        "date": date,
        "issue": issue[:300] if issue else "",
        "summary": summary[:500] if summary else "",
        "keywords": keywords,
        "filename": filepath.name,
        "source": "판결문",
    }


def load_verdict_chunks(verdict_dir: Path) -> list:
    """판결문 디렉토리에서 모든 TXT 파일 파싱"""
    chunks = []
    txt_files = list(verdict_dir.glob("*.txt"))
    print(f"  판결문 TXT: {len(txt_files)}개 파싱 중...")

    for f in txt_files:
        parsed = parse_verdict_file(f)
        if parsed and len(parsed["text"]) > 30:
            chunks.append(parsed)

    print(f"  파싱 완료: {len(chunks)}개")
    return chunks


# ── 3. FAISS 인덱스 빌드 ────────────────────────────
def build_faiss_index(chunks: list, embed_model: SentenceTransformer, save_path: Path):
    texts = [c["text"] for c in chunks]
    print(f"  임베딩 생성 중... ({len(texts)}개)")
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    with open(save_path, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)
    print(f"  저장 완료: {save_path}")
    return index


# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 인덱스 구축 시작 ===\n")

    print("[1/3] 임베딩 모델 로딩 중...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    print("      완료!\n")

    # 약관 조항 인덱스
    print("[2/3] 약관 조항 인덱스 구축 중...")
    if CLAUSE_INDEX_PATH.exists():
        print("      기존 인덱스 존재 → 건너뜀 (삭제 후 재실행하면 재생성)")
    else:
        clause_chunks = load_clause_chunks(PDF_PATH)
        print(f"  조항 청크: {len(clause_chunks)}개")
        exemption_chunks = [c for c in clause_chunks if c["is_exemption"]]
        print(f"  면책 조항: {len(exemption_chunks)}개")
        build_faiss_index(clause_chunks, embed_model, CLAUSE_INDEX_PATH)
    print()

    # 판결문 인덱스
    print("[3/3] 판결문 인덱스 구축 중...")
    if VERDICT_INDEX_PATH.exists():
        print("      기존 인덱스 존재 → 건너뜀 (삭제 후 재실행하면 재생성)")
    else:
        verdict_chunks = load_verdict_chunks(VERDICT_DIR)
        build_faiss_index(verdict_chunks, embed_model, VERDICT_INDEX_PATH)
    print()

    print("=== 인덱스 구축 완료 ===")
