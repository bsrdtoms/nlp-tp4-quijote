"""Embedding-based search using bge-m3.

Uses SSPCloud API when API_TOKEN is set in .env, otherwise falls back to local ollama.
Local ollama is slow (~2h to build cache on CPU) — SSPCloud is the recommended setup.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from fdi_pln_2609_p4.corpus import Chunk

load_dotenv()

CACHE_DIR = Path(".cache")
EMBED_PATH = CACHE_DIR / "embeddings.npy"
EMBED_MODEL = "bge-m3"

_API_TOKEN = os.getenv("API_TOKEN")
_SSP_URL = "https://llm.lab.sspcloud.fr/ollama/api/embed"


def _embed_ssp(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": f"{EMBED_MODEL}:latest", "input": texts}).encode()
    req = urllib.request.Request(
        _SSP_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embeddings"]


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    import ollama

    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return [list(e) for e in resp.embeddings]


def _embed(texts: list[str]) -> list[list[float]]:
    if _API_TOKEN:
        return _embed_ssp(texts)
    return _embed_ollama(texts)


def build_embeddings(chunks: list[Chunk]) -> np.ndarray:
    """Compute embeddings for all chunks in batches and save to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    backend = "SSPCloud" if _API_TOKEN else "ollama local"
    logger.info(f"Building embeddings via {backend}...")

    batch_size = 50
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        logger.info(f"Embedding batch {i // batch_size + 1}/{math.ceil(len(chunks) / batch_size)}")
        for emb in _embed(texts):
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

    vec = np.array(_embed([query])[0], dtype=np.float32)
    query_norm = np.linalg.norm(vec)
    if query_norm == 0:
        return []
    vec /= query_norm

    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1
    scores = matrix @ vec / norms

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]
