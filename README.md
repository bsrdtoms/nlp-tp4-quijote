# NLP TP 4 – Recherche dans Don Quijote

Application de recherche dans *Don Quijote de la Mancha* avec plusieurs méthodes de recherche.

## Prérequis

- Python 3.12

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Lancer l'application

```bash
.venv/bin/python search.py
```

## Modes de recherche

| Mode | Description |
|------|-------------|
| Recherche exacte | Paragraphes contenant la requête telle quelle |
| TF-IDF | Similarité cosinus sur les fréquences de mots |
| TF-IDF + stemming | Idem avec racinisation (SnowballStemmer) |
| TF-IDF + lemmatisation (spaCy) | Idem avec lemmatisation (es_core_news_sm) |
| Embeddings (bge-m3) | Similarité sémantique via bge-m3 (Ollama), meilleure qualité |
| Embeddings (spaCy) | Similarité sémantique via word vectors spaCy (es_core_news_lg), 100% local |

## Embeddings (bge-m3) – mise en place (une seule fois)

Le mode Embeddings bge-m3 utilise le modèle `bge-m3` via **Ollama** (local, par défaut).

### 1. Installer Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Télécharger le modèle bge-m3

```bash
ollama pull bge-m3
```

### 3. Calculer et mettre en cache les embeddings

```bash
.venv/bin/python compute_embeddings.py
```

Cela génère `.cache/embeddings.joblib`. À ne refaire qu'une seule fois.

### 4. Lancer Ollama avant d'utiliser l'app

```bash
ollama serve
```

Ollama doit tourner en arrière-plan pour que le mode Embeddings fonctionne.

---

### Utiliser l'API SSPCloud à la place (optionnel)

Si tu as un token SSPCloud, crée un fichier `.env` :

```
API_TOKEN=ton_token
```

Le code utilisera automatiquement l'API SSPCloud au lieu d'Ollama local.

## Embeddings (spaCy) – mise en place

Ce mode utilise `es_core_news_lg` (déjà inclus dans `requirements.txt`). Le cache est calculé automatiquement au premier lancement du mode — pas d'étape supplémentaire.
