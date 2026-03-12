from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, Select
from textual.containers import ScrollableContainer

TEXT = Path("quijote_clean.txt").read_text(encoding="utf-8")
PARAGRAPHES = [p.strip() for p in TEXT.split("\n\n") if len(p.strip()) > 50]

CACHE = Path(".cache/tfidf.joblib")
CACHE.parent.mkdir(exist_ok=True)

if CACHE.exists():
    VECTORIZER, MATRICE = joblib.load(CACHE)
else:
    VECTORIZER = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    MATRICE = VECTORIZER.fit_transform(PARAGRAPHES)
    joblib.dump((VECTORIZER, MATRICE), CACHE)


def recherche(requete):
    return [p for p in PARAGRAPHES if requete.lower() in p.lower()]


def recherche_tfidf(requete, top_k=10):
    vec = VECTORIZER.transform([requete])
    scores = cosine_similarity(vec, MATRICE).flatten()
    idx = np.argsort(scores)[::-1][:top_k]
    return [(PARAGRAPHES[i], round(float(scores[i]), 3)) for i in idx if scores[i] > 0]


class QuijoteApp(App):

    BINDINGS = [("ctrl+c", "quit", "Quitter")]

    METHODES = [("Recherche exacte", "exact"), ("TF-IDF", "tfidf"), ("Embeddings (bientôt)", "embeddings")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Select(self.METHODES, value="exact", id="methode")
        yield Input(placeholder="Entrez votre requête…")
        yield Static(id="status")
        yield ScrollableContainer(Static(id="results"))
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        methode = self.query_one("#methode", Select).value
        if methode == "exact":
            resultats = recherche(event.value)
            self.query_one("#status", Static).update(f"{len(resultats)} résultat(s)")
            self.query_one("#results", Static).update("\n\n---\n\n".join(resultats[:10]))
        elif methode == "tfidf":
            resultats = recherche_tfidf(event.value)
            self.query_one("#status", Static).update(f"{len(resultats)} résultat(s)")
            self.query_one("#results", Static).update(
                "\n\n---\n\n".join(f"[{score}] {texte}" for texte, score in resultats)
            )
        else:
            self.query_one("#status", Static).update("Méthode pas encore implémentée.")


QuijoteApp().run()
