from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, Select
from textual.containers import ScrollableContainer

TEXT = Path("quijote_clean.txt").read_text(encoding="utf-8")
PARAGRAPHES = [p.strip() for p in TEXT.split("\n\n") if len(p.strip()) > 50]


def recherche(requete):
    return [p for p in PARAGRAPHES if requete.lower() in p.lower()]


class QuijoteApp(App):

    BINDINGS = [("ctrl+c", "quit", "Quitter")]

    METHODES = [("Recherche exacte", "exact"), ("TF-IDF (bientôt)", "tfidf"), ("Embeddings (bientôt)", "embeddings")]

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
        else:
            self.query_one("#status", Static).update("Méthode pas encore implémentée.")


QuijoteApp().run()
