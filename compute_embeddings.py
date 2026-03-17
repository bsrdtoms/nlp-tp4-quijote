"""
Calcule les embeddings de tous les paragraphes via l'API SSPCloud
et les sauvegarde dans .cache/embeddings.joblib.

Usage : python compute_embeddings.py
"""

import os
import requests
import numpy as np
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("API_TOKEN")
URL = "https://llm.lab.sspcloud.fr/ollama/api/embed"
MODEL = "bge-m3:latest"
BATCH_SIZE = 500
CACHE = Path(".cache/embeddings.joblib")
CACHE.parent.mkdir(exist_ok=True)

TEXT = Path("quijote_clean.txt").read_text(encoding="utf-8")
PARAGRAPHES = [p.strip() for p in TEXT.split("\n\n") if len(p.strip()) > 50]


def get_embeddings_batch(texts):
    resp = requests.post(
        URL,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"model": MODEL, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def compute():
    all_embeddings = []
    total = len(PARAGRAPHES)
    for i in range(0, total, BATCH_SIZE):
        batch = PARAGRAPHES[i : i + BATCH_SIZE]
        embeddings = get_embeddings_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"{min(i + BATCH_SIZE, total)}/{total}")
    matrix = np.array(all_embeddings, dtype=np.float32)
    joblib.dump(matrix, CACHE)
    print(f"Sauvegardé : {CACHE} ({matrix.shape})")


if __name__ == "__main__":
    compute()
