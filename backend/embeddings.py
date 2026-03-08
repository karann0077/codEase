"""
Embeddings using fastembed.
Self-contained: both sync and async versions here.
"""
import asyncio
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)
_model = None


def load_model_at_startup():
    global _model
    try:
        logger.info("Loading fastembed BAAI/bge-small-en-v1.5...")
        from fastembed import TextEmbedding
        _model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="/tmp/fastembed_cache",
        )
        list(_model.embed(["warmup"]))
        logger.info("Embedding model ready")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        _model = None


def get_model():
    global _model
    if _model is None:
        load_model_at_startup()
    return _model


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i: i + max_chars] for i in range(0, len(text), max_chars)]


def _embed_sync(text: str) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    try:
        model = get_model()
        if model is None:
            return None
        chunks = chunk_text(text, max_chars=1500)
        vecs = np.array(list(model.embed(chunks)), dtype="float32")
        return np.mean(vecs, axis=0).reshape(1, -1)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """Sync version — kept for compatibility."""
    return _embed_sync(text)


async def generate_embeddings_async(text: str) -> Optional[np.ndarray]:
    """Async version — runs in thread so event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_sync, text)
