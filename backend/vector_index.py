"""
FAISS vector index — faithful to CodeRAG's index.py.
Key differences from original:
- Per-session indexes (multi-user support)
- Stores chunk-level metadata (not just file-level)
- similarity score clamped to [0,1] like CodeRAG
"""
import logging
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = settings.embedding_dim   # 384

# Per-session store: { session_id: { "index": faiss.Index, "metadata": [...] } }
_sessions: Dict[str, Dict] = {}


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length — same as CodeRAG's _l2_normalize."""
    if mat is None or mat.size == 0:
        return mat
    faiss.normalize_L2(mat)
    return mat


def _get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "index":    faiss.IndexFlatIP(EMBEDDING_DIM),
            "metadata": [],
        }
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    _sessions[session_id] = {
        "index":    faiss.IndexFlatIP(EMBEDDING_DIM),
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
    """Add embeddings — faithful to CodeRAG's add_to_index."""
    session = _get_session(session_id)
    idx  = session["index"]
    meta = session["metadata"]

    if embeddings is None or embeddings.size == 0:
        logger.warning(f"Empty embeddings for {filename}")
        return

    vecs = embeddings.astype("float32", copy=True)
    vecs = _l2_normalize(vecs)   # cosine similarity via IndexFlatIP
    idx.add(vecs)

    # Store content snippet — CodeRAG uses [:3000]
    meta.append({
        "content":  content[:3000],
        "filename": filename,
        "filepath": filepath,
        "language": language,
    })
    logger.debug(f"[{session_id}] Indexed {filename} (total: {idx.ntotal})")


def search_index(
    session_id: str,
    query_embedding: np.ndarray,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search — faithful to CodeRAG's search_code.
    Returns results with 'distance' key (similarity score clamped to [0,1]).
    """
    session = _get_session(session_id)
    idx  = session["index"]
    meta = session["metadata"]

    if idx.ntotal == 0:
        return []

    qvec = query_embedding.astype("float32", copy=True)
    faiss.normalize_L2(qvec)   # normalize query too — same as CodeRAG

    k = min(k, idx.ntotal)
    distances, indices = idx.search(qvec, k)

    results = []
    for i, doc_idx in enumerate(indices[0]):
        if 0 <= doc_idx < len(meta):
            results.append({
                **meta[doc_idx],
                # Clamp to [0,1] — same as CodeRAG's prompt_flow.py
                "distance": float(max(0.0, min(1.0, distances[0][i]))),
            })
        else:
            logger.warning(f"Index {doc_idx} out of bounds (metadata len={len(meta)})")

    return results


def get_index_stats(session_id: str) -> Dict[str, Any]:
    session = _get_session(session_id)
    meta = session["metadata"]
    return {
        "total_files": len(meta),
        "files": [
            {
                "filename": m["filename"],
                "filepath": m["filepath"],
                "language": m["language"],
            }
            for m in meta
        ],
    }
