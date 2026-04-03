"""TF-IDF search using numpy only (no scikit-learn)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from fdi_pln_2609_p4.corpus import Chunk, preprocess

CACHE_DIR = Path(".cache")
TFIDF_PATH = CACHE_DIR / "tfidf.npz"
VOCAB_PATH = CACHE_DIR / "tfidf_vocab.json"


def build_index(chunks: list[Chunk]) -> tuple[np.ndarray, dict[str, int]]:
    """Build TF-IDF matrix from chunks and save to cache."""
    logger.info("Building TF-IDF index...")

    # Tokenize all chunks
    tokenized = [preprocess(c["text"]) for c in chunks]

    # Build vocabulary
    vocab: dict[str, int] = {}
    for tokens in tokenized:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)

    V = len(vocab)
    N = len(chunks)
    logger.info(f"Vocabulary size: {V}, chunks: {N}")

    # TF matrix (N x V)
    tf = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        for t in tokens:
            tf[i, vocab[t]] += 1
        tf[i] /= len(tokens)

    # IDF vector
    df = np.count_nonzero(tf, axis=0).astype(np.float32)
    idf = np.log((N + 1) / (df + 1)) + 1

    # TF-IDF
    matrix = tf * idf

    # L2-normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms

    # Save
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(TFIDF_PATH, matrix=matrix, idf=idf)
    VOCAB_PATH.write_text(json.dumps(vocab), encoding="utf-8")
    logger.info(f"TF-IDF index saved to {TFIDF_PATH}")

    return matrix, vocab


def load_or_build_index(chunks: list[Chunk]) -> tuple[np.ndarray, dict[str, int]]:
    """Load TF-IDF index from cache or build it."""
    if TFIDF_PATH.exists() and VOCAB_PATH.exists():
        logger.info("Loading TF-IDF index from cache...")
        data = np.load(TFIDF_PATH)
        matrix = data["matrix"]
        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        return matrix, vocab
    return build_index(chunks)


def search(
    query: str,
    chunks: list[Chunk],
    top_k: int = 10,
) -> list[tuple[Chunk, float]]:
    """Search chunks by TF-IDF cosine similarity."""
    matrix, vocab = load_or_build_index(chunks)

    tokens = preprocess(query)
    if not tokens:
        return []

    # Query vector
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1

    norm = np.linalg.norm(vec)
    if norm == 0:
        return []
    vec /= norm

    scores = matrix @ vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]
