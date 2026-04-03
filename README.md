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

### 1. Installer Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Télécharger les modèles

```bash
ollama pull bge-m3
ollama pull llama3
```

### 3. Lancer Ollama

```bash
ollama serve
```

Le cache d'embeddings est calculé automatiquement au premier lancement du mode sémantique.

| Opération | Durée estimée |
|-----------|--------------|
| Cache TF-IDF (premier lancement) | ~30s |
| Cache embeddings bge-m3 (premier lancement) | ~2h sur CPU |
| Requête TF-IDF | < 1s |
| Requête sémantique | ~2s |
| Requête RAG (génération llama3) | ~30s |

## Dépendances

Voir `pyproject.toml`. Géré par `uv`.
