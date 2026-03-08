"""
Embeddings using fastembed - model is pre-loaded at startup, not on first request.
"""
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)
_model = None


def load_model_at_startup():
    """Call this once when the server starts — downloads & caches the model."""
    global _model
    try:
        logger.info("Pre-loading fastembed model: BAAI/bge-small-en-v1.5 ...")
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # Warm up with a dummy embed so the first real request is instant
        list(_model.embed(["warmup"]))
        logger.info("Embedding model ready ✓")
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


def generate_embeddings(text: str) -> Optional[np.ndarray]:
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
