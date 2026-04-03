"""Embedding-based search using ollama + numpy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from loguru import logger

from fdi_pln_2609_p4.corpus import Chunk

CACHE_DIR = Path(".cache")
EMBED_PATH = CACHE_DIR / "embeddings.npy"
EMBED_MODEL = "bge-m3"


def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a single text via ollama."""
    import ollama

    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return np.array(resp.embeddings[0], dtype=np.float32)


def build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """Compute embeddings for all chunks in batches and save to cache."""
    import ollama

    CACHE_DIR.mkdir(exist_ok=True)
    batch_size = 50
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        logger.info(
            f"Embedding batch {i // batch_size + 1}/{math.ceil(len(chunks) / batch_size)}"
        )
        resp = ollama.embed(model=EMBED_MODEL, input=texts)
        for emb in resp.embeddings:
            all_embeddings.append(np.array(emb, dtype=np.float32))

    matrix = np.stack(all_embeddings)
    np.save(EMBED_PATH, matrix)
    logger.info(f"Embeddings saved to {EMBED_PATH}, shape={matrix.shape}")
    return matrix


def load_or_build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """Load embeddings from cache or compute them."""
    if EMBED_PATH.exists():
        logger.info("Loading embeddings from cache...")
        matrix = np.load(EMBED_PATH)
        if matrix.shape[0] == len(chunks):
            return matrix
        logger.warning("Cache size mismatch, rebuilding embeddings...")
    return build_embeddings(chunks)


def search(
    query: str,
    chunks: list[Chunk],
    top_k: int = 10,
) -> list[tuple[Chunk, float]]:
    """Search chunks by cosine similarity of embeddings."""
    matrix = load_or_build_embeddings(chunks)

    vec = get_embedding(query)
    query_norm = np.linalg.norm(vec)
    if query_norm == 0:
        return []
    vec /= query_norm

    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    scores = matrix @ vec / norms

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]
