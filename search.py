from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.containers import ScrollableContainer

TEXT = Path("quijote_clean.txt").read_text(encoding="utf-8")
PARAGRAPHES = [p.strip() for p in TEXT.split("\n\n") if len(p.strip()) > 50]


def recherche(requete):
    return [p for p in PARAGRAPHES if requete.lower() in p.lower()]


class QuijoteApp(App):

    BINDINGS = [("ctrl+c", "quit", "Quitter")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Entrez votre requête…")
        yield Static(id="status")
        yield ScrollableContainer(Static(id="results"))
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        resultats = recherche(event.value)
        self.query_one("#status", Static).update(f"{len(resultats)} résultat(s)")
        self.query_one("#results", Static).update("\n\n---\n\n".join(resultats[:10]))


QuijoteApp().run()
