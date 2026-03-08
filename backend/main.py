"""
DevPilot API — FastAPI backend
Embedding model is pre-loaded at startup via lifespan event.
"""
import base64
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from embeddings import load_model_at_startup
from ingestion import index_github_repo, index_uploaded_files
from ai_engine import (
    explain_code, explain_code_commented,
    debug_analyze,
    multimodal_debug,
    decode_minified_stacktrace,
    rag_chat,
    analyze_image_for_debug,
)
from vector_index import get_index_stats, clear_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: pre-load embedding model so first request is fast ──
    logger.info("Server starting — loading embedding model...")
    load_model_at_startup()
    logger.info("Server ready ✓")
    yield
    # ── Shutdown ──
    logger.info("Server shutting down")


app = FastAPI(
    title="DevPilot API",
    description="AI-powered developer productivity",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class GithubIngestRequest(BaseModel):
    repo_url: str
    github_token: Optional[str] = None
    session_id: Optional[str] = None

class FileIngestRequest(BaseModel):
    files: List[Dict[str, str]]
    session_id: Optional[str] = None

class ExplainCodeRequest(BaseModel):
    code: str
    language: str = "python"

class DebugRequest(BaseModel):
    stacktrace: str = ""
    error_message: str = ""
    code_context: str = ""
    console_logs: str = ""

class MultimodalDebugRequest(BaseModel):
    stacktrace: str = ""
    console_logs: str = ""
    screenshot_base64: Optional[str] = None

class DecodeStacktraceRequest(BaseModel):
    stacktrace: str
    source_map: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    session_id: str
    conversation_history: Optional[List[Dict]] = None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_configured": bool(settings.groq_api_key),
        "model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }


# ── Session ───────────────────────────────────────────────────────────────────

@app.post("/session/create")
def create_session():
    return {"session_id": str(uuid.uuid4())}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    clear_session(session_id)
    return {"message": "Session cleared"}

@app.get("/session/{session_id}/stats")
def session_stats(session_id: str):
    return get_index_stats(session_id)


# ── Ingestion ─────────────────────────────────────────────────────────────────

# Store repo info per session for later file fetching
_session_repos: dict = {}

@app.post("/ingest/github")
async def ingest_github(req: GithubIngestRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = await index_github_repo(
        session_id=session_id,
        repo_url=req.repo_url,
        github_token=req.github_token or settings.github_token or None,
    )
    # Store repo metadata for file fetching
    if "repo" in result and "error" not in result:
        parts = req.repo_url.rstrip("/").split("/")
        idx = parts.index("github.com")
        _session_repos[session_id] = {
            "owner": parts[idx + 1],
            "repo": parts[idx + 2],
            "branch": result.get("branch", "main"),
            "token": req.github_token or settings.github_token or None,
        }
    return {"session_id": session_id, **result}

@app.get("/fetch-file")
async def fetch_file(session_id: str, filepath: str):
    """Fetch raw file content from GitHub for documentation."""
    import httpx
    info = _session_repos.get(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="No GitHub repo indexed for this session")
    owner, repo, branch = info["owner"], info["repo"], info["branch"]
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}"
    headers = {}
    if info.get("token"):
        headers["Authorization"] = f"token {info['token']}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    return {"content": r.text, "filepath": filepath}

@app.post("/ingest/files")
async def ingest_files(req: FileIngestRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = await index_uploaded_files(session_id, req.files)
    return {"session_id": session_id, **result}

@app.post("/ingest/upload")
async def ingest_upload(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    file_data = []
    for f in files:
        try:
            content = (await f.read()).decode("utf-8", errors="replace")
            file_data.append({"filename": f.filename, "content": content})
        except Exception as e:
            logger.warning(f"Could not read {f.filename}: {e}")
    result = await index_uploaded_files(session_id, file_data)
    return {"session_id": session_id, **result}


# ── Documentation ─────────────────────────────────────────────────────────────

@app.post("/docs/explain")
async def explain_endpoint(req: ExplainCodeRequest):
    if not req.code.strip():
        raise HTTPException(400, "Code cannot be empty")
    return await explain_code(req.code, req.language)

@app.post("/docs/commented-code")
async def commented_code_endpoint(req: ExplainCodeRequest):
    if not req.code.strip():
        raise HTTPException(400, "Code cannot be empty")
    return await explain_code_commented(req.code, req.language)


# ── Debugging ─────────────────────────────────────────────────────────────────

@app.post("/debug/analyze")
async def debug_analyze_endpoint(req: DebugRequest):
    return await debug_analyze(
        stacktrace=req.stacktrace,
        error_message=req.error_message,
        code_context=req.code_context,
        console_logs=req.console_logs,
    )

@app.post("/debug/multimodal")
async def multimodal_debug_endpoint(req: MultimodalDebugRequest):
    screenshot_desc = ""
    if req.screenshot_base64:
        screenshot_desc = await analyze_image_for_debug(req.screenshot_base64)
    result = await multimodal_debug(
        stacktrace=req.stacktrace,
        console_logs=req.console_logs,
        screenshot_desc=screenshot_desc,
    )
    result["screenshot_interpretation"] = screenshot_desc
    return result

@app.post("/debug/decode-stacktrace")
async def decode_stacktrace_endpoint(req: DecodeStacktraceRequest):
    return await decode_minified_stacktrace(req.stacktrace, req.source_map)

@app.post("/debug/upload-screenshot")
async def upload_screenshot_debug(
    screenshot: UploadFile = File(...),
    stacktrace: str = Form(default=""),
    console_logs: str = Form(default=""),
):
    img_bytes = await screenshot.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    screenshot_desc = await analyze_image_for_debug(img_b64)
    result = await multimodal_debug(
        stacktrace=stacktrace,
        console_logs=console_logs,
        screenshot_desc=screenshot_desc,
    )
    result["screenshot_interpretation"] = screenshot_desc
    return result


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    response = await rag_chat(
        session_id=req.session_id,
        query=req.query,
        conversation_history=req.conversation_history,
    )
    return {"response": response, "session_id": req.session_id}
