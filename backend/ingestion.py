"""
GitHub repository ingestion - fast version.
Uses Git Trees API (1 call for all file paths) + concurrent downloads.
Adapted from CodeRAG's indexing approach.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx

from embeddings import generate_embeddings
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
    "coverage", ".pytest_cache", "eggs", ".eggs",
}

ALLOWED_EXTENSIONS = set(LANGUAGE_MAP.keys())
MAX_FILE_SIZE = 80_000   # 80KB per file — skip huge generated files
MAX_FILES     = 80       # cap total files on free tier (memory limit)
MAX_CONCURRENT = 8       # parallel downloads — fast but won't hit rate limits


def detect_language(filepath: str) -> str:
    return LANGUAGE_MAP.get(Path(filepath).suffix.lower(), "unknown")


def should_index(path: str, size: int) -> bool:
    """Return True if this file is worth indexing."""
    p = Path(path)
    # Skip ignored dirs anywhere in path
    if any(part in IGNORE_DIRS for part in p.parts):
        return False
    # Must be a known code/text extension
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    # Skip huge files (minified JS, lock files, etc.)
    if size > MAX_FILE_SIZE:
        return False
    # Skip obvious non-code files by name
    name = p.name.lower()
    if name in {"package-lock.json", "yarn.lock", "poetry.lock", "pipfile.lock", "composer.lock"}:
        return False
    return True


async def index_github_repo(
    session_id: str,
    repo_url: str,
    github_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fast GitHub repo ingestion:
    1. One API call to get the full file tree
    2. Concurrent downloads of all code files
    3. Batch embedding generation
    """
    clear_session(session_id)

    # Parse owner/repo
    parts = repo_url.rstrip("/").split("/")
    if "github.com" not in repo_url:
        return {"error": "Invalid GitHub URL. Use: https://github.com/owner/repo"}
    idx = parts.index("github.com")
    owner = parts[idx + 1]
    repo_name = parts[idx + 2].replace(".git", "")

    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    indexed = 0
    skipped = 0
    errors = []

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:

        # ── Step 1: Get default branch ────────────────────────────────
        repo_resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo_name}"
        )
        if repo_resp.status_code == 404:
            return {"error": f"Repo not found: {owner}/{repo_name}. Check the URL and make sure it's public."}
        if repo_resp.status_code == 403:
            return {"error": "GitHub rate limit hit. Add a GitHub token in the token field to get 5000 requests/hour."}
        if repo_resp.status_code != 200:
            return {"error": f"GitHub API error: {repo_resp.status_code}"}

        default_branch = repo_resp.json().get("default_branch", "main")

        # ── Step 2: Get full file tree in ONE API call ─────────────────
        tree_resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{default_branch}",
            params={"recursive": "1"},
        )
        if tree_resp.status_code != 200:
            return {"error": f"Could not fetch file tree: {tree_resp.status_code}"}

        tree_data = tree_resp.json()
        if tree_data.get("truncated"):
            logger.warning("Tree truncated — repo is very large, only partial indexing")

        # Filter to indexable files
        all_files = [
            item for item in tree_data.get("tree", [])
            if item["type"] == "blob" and should_index(item["path"], item.get("size", 0))
        ]

        # Cap to MAX_FILES to stay within Render free memory
        if len(all_files) > MAX_FILES:
            logger.info(f"Capping at {MAX_FILES} files (found {len(all_files)})")
            all_files = all_files[:MAX_FILES]

        if not all_files:
            return {
                "repo": f"{owner}/{repo_name}",
                "indexed_files": 0,
                "skipped_files": 0,
                "errors": ["No indexable code files found in this repo."],
                "files": [],
            }

        # ── Step 3: Download files concurrently ───────────────────────
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def fetch_file(item: dict) -> Optional[Dict]:
            async with semaphore:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/{default_branch}/{item['path']}"
                try:
                    r = await client.get(raw_url, timeout=15)
                    if r.status_code == 200:
                        return {
                            "path": item["path"],
                            "name": Path(item["path"]).name,
                            "content": r.text,
                            "language": detect_language(item["path"]),
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch {item['path']}: {e}")
                return None

        # Run all downloads concurrently
        results = await asyncio.gather(*[fetch_file(f) for f in all_files])
        downloaded = [r for r in results if r is not None]

        # ── Step 4: Generate embeddings and index ─────────────────────
        for file_data in downloaded:
            try:
                embedding = generate_embeddings(file_data["content"])
                if embedding is not None:
                    add_to_index(
                        session_id=session_id,
                        embeddings=embedding,
                        content=file_data["content"],
                        filename=file_data["name"],
                        filepath=file_data["path"],
                        language=file_data["language"],
                    )
                    indexed += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Error indexing {file_data['path']}: {e}")
                skipped += 1

        skipped += len(all_files) - len(downloaded)

    stats = get_index_stats(session_id)
    return {
        "repo": f"{owner}/{repo_name}",
        "indexed_files": indexed,
        "skipped_files": skipped,
        "errors": errors[:5],
        "files": stats["files"],
    }


async def index_uploaded_files(
    session_id: str,
    files_content: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Index a list of uploaded files: [{"filename": ..., "content": ...}]"""
    indexed = 0
    skipped = 0

    for f in files_content:
        filename = f.get("filename", "unknown")
        content = f.get("content", "")
        if not content.strip():
            skipped += 1
            continue
        embedding = generate_embeddings(content)
        if embedding is not None:
            add_to_index(
                session_id=session_id,
                embeddings=embedding,
                content=content,
                filename=filename,
                filepath=filename,
                language=detect_language(filename),
            )
            indexed += 1
        else:
            skipped += 1

    return {
        "indexed_files": indexed,
        "skipped_files": skipped,
        "files": get_index_stats(session_id)["files"],
    }
