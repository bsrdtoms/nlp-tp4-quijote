# FDI PLN 2609 – Práctica 4 : Buscador en Don Quijote

Moteur de recherche dans *Don Quijote de la Mancha* avec interface TUI Textual.

**Groupe G09** — Thomas Bossard

## Installation

```bash
uv sync
uv run python -m spacy download es_core_news_sm
```

## Lancer l'application

```bash
uv run fdi-pln-2609-p4
```

## Modes de recherche

| Mode | Description |
|------|-------------|
| Búsqueda clásica | TF-IDF numpy + lemmatisation spaCy + stopwords espagnols |
| Búsqueda semántica | Embeddings bge-m3 (Ollama local) + similarité cosinus |
| RAG | Recherche TF-IDF + génération llama3 avec références |

## Mise en place Ollama (pour les modes sémantique et RAG)

> **Note** : le mode sémantique (bge-m3) est conçu pour être utilisé avec l'API SSPCloud.
> En local sur CPU, le calcul du cache prend ~2h — préférer SSPCloud si disponible.

### Option 1 — SSPCloud (recommandé)

Créer un fichier `.env` à la racine :

```
API_TOKEN=votre_token_sspcloud
```

Le code utilisera automatiquement `llm.lab.sspcloud.fr` pour les embeddings.

### Option 2 — Ollama local

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull bge-m3
ollama pull llama3
ollama serve
```

Le cache d'embeddings est calculé automatiquement au premier lancement du mode sémantique.

| Opération | Durée estimée |
|-----------|--------------|
| Cache TF-IDF (premier lancement) | ~30s |
| Cache embeddings bge-m3 via SSPCloud | ~5 min |
| Cache embeddings bge-m3 en local (CPU) | ~2h |
| Requête TF-IDF | < 1s |
| Requête sémantique | ~2s |
| Requête RAG (génération llama3) | ~30s |

## Dépendances

Voir `pyproject.toml`. Géré par `uv`.
