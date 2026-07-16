"""
사고보상 Agent 서버
===================
POST /api/cases          → 새 사건 접수
POST /api/cases/{id}/resume → 추가자료 재제출
GET  /api/cases          → 사건 목록
GET  /api/cases/{id}     → 사건 상세
DELETE /api/cases/{id}   → 사건 삭제
GET  /                   → 웹 UI
"""

import json
import os
import uuid
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List

UPLOAD_DIR = "static/uploads"
LOG_DIR    = "logs"
DB_PATH    = "cases.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)


def _init_db():
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id       TEXT PRIMARY KEY,
            accident_type TEXT,
            final_status  TEXT DEFAULT '처리중',
            created_at    TEXT,
            updated_at    TEXT,
            user_input    TEXT,
            result_json   TEXT
        )
    """)
    _conn.commit()

_init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from graph.pipeline import get_models
    print("\n임베딩 모델 로딩 중...")
    get_models()
    print("준비 완료! (LLM: Groq API)\n")
    yield
    _conn.close()


app = FastAPI(title="사고보상 Agent API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── 유틸 ──────────────────────────────────────────────
def _save_case(case_id, user_input, result):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized   = result.get("normalized") or {}
    final_status = result.get("final_status", "처리중")
    result_json  = json.dumps(result, ensure_ascii=False)

    row = _conn.execute("SELECT case_id FROM cases WHERE case_id=?", (case_id,)).fetchone()
    if row:
        _conn.execute("""
            UPDATE cases SET accident_type=?, final_status=?, updated_at=?, result_json=?
            WHERE case_id=?
        """, (normalized.get("accident_type",""), final_status, now, result_json, case_id))
    else:
        _conn.execute("""
            INSERT INTO cases VALUES (?,?,?,?,?,?,?)
        """, (case_id, normalized.get("accident_type",""), final_status,
              now, now, user_input[:200], result_json))
    _conn.commit()

    # logs/<case_id>.jsonl 에 타임스탬프와 함께 순서대로 추가
    log_entry = {"timestamp": now, "case_id": case_id, **result}
    log_path  = os.path.join(LOG_DIR, f"{case_id}.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ── API ───────────────────────────────────────────────
@app.post("/api/cases")
async def create_case(
    user_input: str              = Form(...),
    video_url:  str              = Form(""),
    extra_data: str              = Form(""),
    files:      List[UploadFile] = File(default=[]),
):
    case_id = "CASE-" + uuid.uuid4().hex[:8].upper()

    image_urls = []
    for f in files:
        if f.filename:
            ext  = os.path.splitext(f.filename)[1].lower()
            dest = os.path.join(UPLOAD_DIR, f"{case_id}_{uuid.uuid4().hex[:6]}{ext}")
            with open(dest, "wb") as fp:
                fp.write(await f.read())
            image_urls.append("/" + dest)

    from graph.pipeline import run
    result = run(
        user_input=user_input,
        case_id=case_id,
        video_url=video_url,
        extra_data=extra_data,
    )
    result["image_urls"] = image_urls
    _save_case(case_id, user_input, result)
    return {"case_id": case_id, **result}


@app.post("/api/cases/{case_id}/resume")
async def resume_case(
    case_id:    str,
    extra_data: str              = Form(""),
    files:      List[UploadFile] = File(default=[]),
):
    if not extra_data.strip():
        raise HTTPException(400, "추가 자료 내용을 입력하세요.")

    image_urls = []
    for f in files:
        if f.filename:
            ext  = os.path.splitext(f.filename)[1].lower()
            dest = os.path.join(UPLOAD_DIR, f"{case_id}_{uuid.uuid4().hex[:6]}{ext}")
            with open(dest, "wb") as fp:
                fp.write(await f.read())
            image_urls.append("/" + dest)

    from graph.pipeline import resume
    result = resume(case_id=case_id, extra_data=extra_data)

    # 기존 이미지 목록에 합산
    prev_images = result.get("image_urls") or []
    result["image_urls"] = prev_images + image_urls
    _save_case(case_id, "", result)
    return {"case_id": case_id, **result}


@app.get("/api/cases")
async def list_cases():
    rows = _conn.execute("""
        SELECT case_id, accident_type, final_status, created_at, updated_at
        FROM cases ORDER BY created_at DESC
    """).fetchall()
    return [{"case_id":r[0],"accident_type":r[1],"final_status":r[2],
             "created_at":r[3],"updated_at":r[4]} for r in rows]


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    row = _conn.execute(
        "SELECT result_json, user_input, created_at FROM cases WHERE case_id=?",
        (case_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404, f"{case_id} 없음")
    result = json.loads(row[0])
    result["user_input"]  = row[1]
    result["created_at"]  = row[2]
    result["case_id"]     = case_id
    return result


@app.delete("/api/cases/{case_id}")
async def delete_case(case_id: str):
    _conn.execute("DELETE FROM cases WHERE case_id=?", (case_id,))
    _conn.commit()
    return {"deleted": case_id}


# ── 웹 UI ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  🚗 사고보상 Agent 서버")
    print("  http://localhost:8080")
    print("═"*50 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)
