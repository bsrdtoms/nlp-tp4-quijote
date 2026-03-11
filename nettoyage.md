# Journal de nettoyage – Don Quijote de la Mancha

## Source

- Fichier brut : `el quijote.txt` (38 059 lignes, encodage UTF-8, Project Gutenberg)
- SHA/identité : non calculé ; fichier conservé intact

## Opérations effectuées

### 1. Extraction du texte réel (lignes 1069–37702)

**Pourquoi :** Le fichier brut contient un header Gutenberg (lignes 1–27), un front matter et une table des matières (lignes 28–1068), puis le footer Gutenberg (lignes 37703+). Ces sections sont du bruit pour l'indexation TF-IDF.

**Comment :** Extraction en Python par slicing sur les lignes (index 0-based 1068:37702) :

```python
lines = open("el quijote.txt", encoding="utf-8").readlines()
clean = lines[1068:37702]
open("quijote_clean.txt", "w", encoding="utf-8").writelines(clean)
```

**Résultat :** `quijote_clean.txt` — 36 634 lignes.

- Première ligne : `Capítulo primero. Que trata de la condición y ejercicio del famoso hidalgo`
- Dernière ligne : `Fin`

### 2. Vérification des artefacts

Vérification automatique après extraction :

| Critère | Résultat |
|---|---|
| Lignes contenant uniquement des espaces (≠ `\n`) | 0 |
| Lignes commençant par 2 espaces ou plus (indentation parasite) | 0 |

**Conclusion :** Aucun nettoyage supplémentaire nécessaire. Le fichier brut Gutenberg est propre une fois le header/footer supprimé.

## Ce qui a été supprimé

| Lignes brutes | Contenu supprimé |
|---|---|
| 1–1068 | Header Gutenberg (licence, avertissement, métadonnées) + Tasa, Dedicatoria, Prólogo, table des matières |
| 37703–38059 | Lignes vides finales + footer Gutenberg (End of Project Gutenberg…) |

## Ce qui n'a PAS été modifié

- Casse du texte (non normalisée — la recherche TF-IDF gère les minuscules via `analyzer="word"`)
- Tirets em `—` des dialogues (conservés pour la lisibilité)
- Sauts de ligne au sein des paragraphes (conservés ; le split se fait sur `\n{2,}`)
- Accents et caractères espagnols (UTF-8 conservé intégralement)
