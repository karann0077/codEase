"""
GitHub repository ingestion - clones and indexes repo files.
Inspired by CodeRAG's file watching and indexing approach.
"""
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx

from embeddings import generate_embeddings
from vector_index import add_to_index, clear_session, get_index_stats

logger = logging.getLogger(__name__)

# File extensions and their languages
LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".md": "markdown", ".txt": "text", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".html": "html", ".css": "css", ".sh": "bash",
}

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next", "target"}
MAX_FILE_SIZE = 100_000  # 100KB per file


def detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return LANGUAGE_MAP.get(ext, "unknown")


async def index_github_repo(session_id: str, repo_url: str, github_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch and index a GitHub repository via the GitHub API (no git clone needed).
    Returns indexing stats.
    """
    clear_session(session_id)

    # Parse owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    if "github.com" in repo_url:
        idx = parts.index("github.com")
        owner, repo_name = parts[idx + 1], parts[idx + 2].replace(".git", "")
    else:
        return {"error": "Invalid GitHub URL. Use format: https://github.com/owner/repo"}

    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    indexed = 0
    skipped = 0
    errors = []

    async def fetch_tree(path: str = ""):
        nonlocal indexed, skipped
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                errors.append(f"Failed to fetch {path}: {resp.status_code}")
                return
            items = resp.json()
            if isinstance(items, dict):
                items = [items]

            for item in items:
                if item["type"] == "dir":
                    if item["name"] not in IGNORE_DIRS:
                        await fetch_tree(item["path"])
                elif item["type"] == "file":
                    lang = detect_language(item["name"])
                    if lang == "unknown" and not item["name"].endswith(tuple(LANGUAGE_MAP.keys())):
                        skipped += 1
                        continue
                    if item.get("size", 0) > MAX_FILE_SIZE:
                        skipped += 1
                        continue

                    # Fetch file content
                    file_resp = await client.get(item["download_url"], timeout=20)
                    if file_resp.status_code != 200:
                        skipped += 1
                        continue

                    try:
                        content = file_resp.text
                    except Exception:
                        skipped += 1
                        continue

                    # Generate embeddings and index
                    embedding = generate_embeddings(content)
                    if embedding is not None:
                        add_to_index(
                            session_id=session_id,
                            embeddings=embedding,
                            content=content,
                            filename=item["name"],
                            filepath=item["path"],
                            language=lang,
                        )
                        indexed += 1
                    else:
                        skipped += 1

    await fetch_tree()

    stats = get_index_stats(session_id)
    return {
        "repo": f"{owner}/{repo_name}",
        "indexed_files": indexed,
        "skipped_files": skipped,
        "errors": errors[:5],
        "files": stats["files"],
    }


async def index_uploaded_files(session_id: str, files_content: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Index a list of uploaded files: [{"filename": ..., "content": ...}]
    """
    indexed = 0
    skipped = 0

    for f in files_content:
        filename = f.get("filename", "unknown")
        content = f.get("content", "")

        if not content.strip():
            skipped += 1
            continue

        lang = detect_language(filename)
        embedding = generate_embeddings(content)
        if embedding is not None:
            add_to_index(
                session_id=session_id,
                embeddings=embedding,
                content=content,
                filename=filename,
                filepath=filename,
                language=lang,
            )
            indexed += 1
        else:
            skipped += 1

    return {
        "indexed_files": indexed,
        "skipped_files": skipped,
        "files": get_index_stats(session_id)["files"],
    }
