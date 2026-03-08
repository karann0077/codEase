"""
Embeddings module — uses fastembed (lightweight, no GPU needed, ~50MB).
Model: BAAI/bge-small-en-v1.5 → 384 dims, fast on CPU, great for code search.
Much lighter than sentence-transformers (~2GB) so it works on Render free tier.
Adapted from CodeRAG's embeddings approach.
"""
import logging
from typing import List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

_model = None


def get_model():
    """Lazy-load fastembed model (downloads ~50MB on first run, then cached)."""
    global _model
    if _model is None:
        logger.info("Loading fastembed model: BAAI/bge-small-en-v1.5")
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Embedding model ready")
    return _model


def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into chunks to fit model context."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    return [text[i: i + max_chars] for i in range(0, len(text), max_chars)]


def generate_embeddings(text: str) -> Optional[np.ndarray]:
    """
    Generate a (1, 384) float32 embedding vector for the given text.
    Averages across chunks for long files.
    """
    if not text or not text.strip():
        return None

    try:
        model = get_model()
        chunks = chunk_text(text, max_chars=2000)

        # fastembed returns a generator — collect into list
        vecs = np.array(list(model.embed(chunks)), dtype="float32")

        # Average all chunk vectors → single representative embedding
        avg = np.mean(vecs, axis=0).reshape(1, -1)
        return avg

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None
