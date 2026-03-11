
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import httpx

from embeddings import generate_embeddings_async
from vector_index import add_to_index, clear_session, get_index_stats

logger = logging.getLogger(__name__)


LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".md": "markdown", ".txt": "text", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".html": "html", ".css": "css", ".sh": "bash",
}


IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "target", ".idea", ".vscode",
    "coverage", ".pytest_cache", "tests", "test",
}

ALLOWED_EXTENSIONS = set(LANGUAGE_MAP.keys())
MAX_FILE_SIZE  = 50_000   # 50KB
MAX_FILES      = 40       # cap for Render free RAM
MAX_CONCURRENT = 10       # parallel downloads
CHUNK_SIZE     = 1500     


def detect_language(filepath: str) -> str:
    return LANGUAGE_MAP.get(Path(filepath).suffix.lower(), "unknown")


def should_index(path: str, size: int) -> bool:
    """Mirror CodeRAG's should_ignore_path logic."""
    p = Path(path)
    if any(part in IGNORE_DIRS for part in p.parts):
        return False
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    if size > MAX_FILE_SIZE:
        return False
    if p.name.lower() in {
        "package-lock.json", "yarn.lock", "poetry.lock",
        "pipfile.lock", "composer.lock", "pnpm-lock.yaml",
    }:
        return False
    return True


def chunk_code(content: str, filepath: str, chunk_size: int = CHUNK_SIZE) -> List[Tuple[str, str]]:
    """
    Code-aware chunking — inspired by CodeRAG's approach.
    Splits on function/class boundaries for Python/JS/TS,
    falls back to paragraph/character chunking for other files.
    Returns list of (chunk_text, chunk_label) tuples.
    """
    content = content.strip()
    if not content:
        return []

    ext = Path(filepath).suffix.lower()
    chunks = []

    if ext == ".py":
        # Split on top-level def/class boundaries
        pattern = re.compile(r'\n(?=(?:def |class |async def ))', re.MULTILINE)
        parts = pattern.split(content)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # If a single function is still huge, sub-chunk it
            if len(part) > chunk_size:
                for i in range(0, len(part), chunk_size):
                    sub = part[i:i + chunk_size].strip()
                    if sub:
                        chunks.append(sub)
            else:
                chunks.append(part)

    elif ext in {".js", ".ts", ".jsx", ".tsx"}:
        # Split on function/arrow function/class boundaries
        pattern = re.compile(
            r'\n(?=(?:function |class |const \w+ = |export (?:default )?(?:function|class)|async function))',
            re.MULTILINE
        )
        parts = pattern.split(content)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > chunk_size:
                for i in range(0, len(part), chunk_size):
                    sub = part[i:i + chunk_size].strip()
                    if sub:
                        chunks.append(sub)
            else:
                chunks.append(part)

    else:
        # For other files: split on blank lines (paragraphs), then by size
        paragraphs = re.split(r'\n\s*\n', content)
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current += "\n\n" + para
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para
        if current.strip():
            chunks.append(current.strip())

        # If any chunk is still too large, sub-split
        final = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                for i in range(0, len(chunk), chunk_size):
                    sub = chunk[i:i + chunk_size].strip()
                    if sub:
                        final.append(sub)
            else:
                final.append(chunk)
        chunks = final

    return [(c, f"{Path(filepath).name} chunk {i+1}/{len(chunks)}") for i, c in enumerate(chunks)] if chunks else [(content[:chunk_size], Path(filepath).name)]


async def index_github_repo(
    session_id: str,
    repo_url: str,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full RAG ingestion — mirrors CodeRAG's full_reindex() but over GitHub API.
    Each file is chunked at function/class boundaries, each chunk gets its own embedding.
    """
    clear_session(session_id)

    parts = repo_url.rstrip("/").split("/")
    if "github.com" not in repo_url:
        return {"error": "Invalid GitHub URL. Use: https://github.com/owner/repo"}
    idx      = parts.index("github.com")
    owner    = parts[idx + 1]
    repo_name = parts[idx + 2].replace(".git", "")

    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    indexed_files   = 0
    indexed_chunks  = 0
    skipped         = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, read=20.0),
        headers=headers,
        follow_redirects=True,
    ) as client:

        # ── 1. Get default branch ─────────────────────────────────────
        repo_resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo_name}"
        )
        if repo_resp.status_code == 404:
            return {"error": f"Repo not found or private: {owner}/{repo_name}"}
        if repo_resp.status_code == 403:
            return {"error": "GitHub rate limit hit. Add a GitHub token."}
        if repo_resp.status_code != 200:
            return {"error": f"GitHub API error {repo_resp.status_code}"}

        default_branch = repo_resp.json().get("default_branch", "main")
        logger.info(f"Indexing {owner}/{repo_name} @ {default_branch}")

        # ── 2. Full file tree in one call (CodeRAG does os.walk, we do this) ──
        tree_resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if tree_resp.status_code != 200:
            return {"error": f"Could not fetch file tree: {tree_resp.status_code}"}

        all_blobs = [
            item for item in tree_resp.json().get("tree", [])
            if item["type"] == "blob"
            and should_index(item["path"], item.get("size", 0))
        ]

        if len(all_blobs) > MAX_FILES:
            logger.info(f"Capping {len(all_blobs)} → {MAX_FILES} files")
            all_blobs = all_blobs[:MAX_FILES]

        if not all_blobs:
            return {
                "repo": f"{owner}/{repo_name}",
                "indexed_files": 0, "indexed_chunks": 0, "skipped_files": 0,
                "errors": ["No indexable code files found."],
                "files": [],
            }

        logger.info(f"Downloading {len(all_blobs)} files...")

        # ── 3. Concurrent downloads ───────────────────────────────────
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def fetch_one(item: dict) -> Optional[Dict]:
            async with sem:
                url = (
                    f"https://raw.githubusercontent.com"
                    f"/{owner}/{repo_name}/{default_branch}/{item['path']}"
                )
                try:
                    r = await client.get(url, timeout=15)
                    if r.status_code == 200:
                        return {
                            "path":     item["path"],
                            "name":     Path(item["path"]).name,
                            "content":  r.text,
                            "language": detect_language(item["path"]),
                        }
                except Exception as e:
                    logger.warning(f"Download failed {item['path']}: {e}")
                return None

        downloads = await asyncio.gather(*[fetch_one(f) for f in all_blobs])
        files_ok  = [f for f in downloads if f is not None]
        skipped  += len(all_blobs) - len(files_ok)

        logger.info(f"Downloaded {len(files_ok)} files. Chunking and embedding...")

        # ── 4. Chunk + embed each file — mirrors CodeRAG's full_reindex loop ──
        async def process_file(file_data: dict):
            nonlocal indexed_files, indexed_chunks, skipped

            # Code-aware chunking
            chunks = chunk_code(file_data["content"], file_data["path"])
            if not chunks:
                skipped += 1
                return

            file_indexed = False
            for chunk_text, chunk_label in chunks:
                if not chunk_text.strip():
                    continue
                try:
                    # Each chunk gets its own embedding — true RAG like CodeRAG
                    emb = await generate_embeddings_async(chunk_text)
                    if emb is not None:
                        add_to_index(
                            session_id=session_id,
                            embeddings=emb,
                            content=chunk_text,           # chunk, not whole file
                            filename=chunk_label,
                            filepath=file_data["path"],
                            language=file_data["language"],
                        )
                        indexed_chunks += 1
                        file_indexed = True
                except Exception as e:
                    logger.error(f"Embed failed {file_data['path']}: {e}")

            if file_indexed:
                indexed_files += 1
            else:
                skipped += 1

        # Process all files concurrently
        await asyncio.gather(*[process_file(f) for f in files_ok])
        logger.info(f"Done. files={indexed_files} chunks={indexed_chunks} skipped={skipped}")

    return {
        "repo":           f"{owner}/{repo_name}",
        "branch":         default_branch,
        "indexed_files":  indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files":  skipped,
        "errors":         [],
        "files":          get_index_stats(session_id)["files"],
    }


async def index_uploaded_files(
    session_id: str,
    files_content: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Mirror CodeRAG's full_reindex for uploaded files."""
    indexed_files  = 0
    indexed_chunks = 0
    skipped        = 0

    async def process_one(f: dict):
        nonlocal indexed_files, indexed_chunks, skipped
        content  = f.get("content", "")
        filename = f.get("filename", "unknown")
        if not content.strip():
            skipped += 1
            return

        chunks = chunk_code(content, filename)
        file_indexed = False
        for chunk_text, chunk_label in chunks:
            if not chunk_text.strip():
                continue
            emb = await generate_embeddings_async(chunk_text)
            if emb is not None:
                add_to_index(
                    session_id=session_id,
                    embeddings=emb,
                    content=chunk_text,
                    filename=chunk_label,
                    filepath=filename,
                    language=detect_language(filename),
                )
                indexed_chunks += 1
                file_indexed = True
        if file_indexed:
            indexed_files += 1
        else:
            skipped += 1

    await asyncio.gather(*[process_one(f) for f in files_content])

    return {
        "indexed_files":  indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files":  skipped,
        "files":          get_index_stats(session_id)["files"],
    }
