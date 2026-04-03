"""Text loading and chunking for Don Quijote."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TypedDict

import nltk
from loguru import logger

# Lazy spaCy load
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        logger.info("Loading spaCy model es_core_news_sm...")
        _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    return _nlp


def _get_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        return set(stopwords.words("spanish"))


def preprocess(text: str) -> list[str]:
    """Lemmatize text, remove stopwords and non-alpha tokens."""
    nlp = _get_nlp()
    stops = _get_stopwords()
    doc = nlp(text.lower())
    return [
        token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stops
    ]


def _load_text() -> str:
    """Load quijote_clean.txt from package data or parent directories."""
    # Try package data first
    try:
        ref = importlib.resources.files("fdi_pln_2609_p4.data").joinpath(
            "quijote_clean.txt"
        )
        return ref.read_text(encoding="utf-8")
    except Exception:
        pass

    # Fallback: walk up from this file
    here = Path(__file__).parent
    for parent in [here, here.parent, here.parent.parent]:
        candidate = parent / "quijote_clean.txt"
        if candidate.exists():
            logger.info(f"Loading text from {candidate}")
            return candidate.read_text(encoding="utf-8")

    raise FileNotFoundError("quijote_clean.txt not found")


class Chunk(TypedDict):
    text: str
    start: int
    end: int
    index: int


def _make_chunks(text: str, size: int = 150, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping word chunks."""
    words = text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(words):
        end = min(i + size, len(words))
        chunk_text = " ".join(words[i:end])
        chunks.append({"text": chunk_text, "start": i, "end": end, "index": idx})
        idx += 1
        if end == len(words):
            break
        i += step
    return chunks


_cache: list[Chunk] | None = None


def get_chunks() -> list[Chunk]:
    """Return cached list of text chunks."""
    global _cache
    if _cache is None:
        logger.info("Loading and chunking Don Quijote...")
        text = _load_text()
        _cache = _make_chunks(text)
        logger.info(f"{len(_cache)} chunks created")
    return _cache
