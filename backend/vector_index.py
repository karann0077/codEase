"""
FAISS vector index for code search - adapted from CodeRAG's index.py
Stores and retrieves code embeddings for semantic search.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = settings.embedding_dim  # 384 for all-MiniLM-L6-v2
INDEX_FILE = "/tmp/devpilot_index.faiss"
METADATA_FILE = "/tmp/devpilot_metadata.npy"

# In-memory index per session
_sessions: Dict[str, Dict] = {}


def _get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "index": faiss.IndexFlatIP(EMBEDDING_DIM),
            "metadata": [],
        }
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    """Clear all indexed files for a session."""
    _sessions[session_id] = {
        "index": faiss.IndexFlatIP(EMBEDDING_DIM),
        "metadata": [],
    }


def add_to_index(
    session_id: str,
    embeddings: np.ndarray,
    content: str,
    filename: str,
    filepath: str,
    language: str = "unknown",
) -> None:
    """Add embeddings and metadata to the session's FAISS index."""
    session = _get_session(session_id)
    idx = session["index"]
    meta = session["metadata"]

    if embeddings is None or embeddings.size == 0:
        return

    vecs = embeddings.astype("float32", copy=True)
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    meta.append({
        "content": content[:5000],
        "filename": filename,
        "filepath": filepath,
        "language": language,
    })
    logger.debug(f"[{session_id}] Indexed {filename} (total: {idx.ntotal})")


def search_index(session_id: str, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    """Search the FAISS index for the most similar code chunks."""
    session = _get_session(session_id)
    idx = session["index"]
    meta = session["metadata"]

    if idx.ntotal == 0:
        return []

    qvec = query_embedding.astype("float32", copy=True)
    faiss.normalize_L2(qvec)
    k = min(k, idx.ntotal)
    distances, indices = idx.search(qvec, k)

    results = []
    for i, doc_idx in enumerate(indices[0]):
        if 0 <= doc_idx < len(meta):
            results.append({
                **meta[doc_idx],
                "score": float(distances[0][i]),
            })
    return results


def get_index_stats(session_id: str) -> Dict[str, Any]:
    """Get stats about what's been indexed for a session."""
    session = _get_session(session_id)
    meta = session["metadata"]
    return {
        "total_files": len(meta),
        "files": [{"filename": m["filename"], "filepath": m["filepath"], "language": m["language"]} for m in meta],
    }
