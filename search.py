from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static


class QuijoteApp(App):

    BINDINGS = [("ctrl+c", "quit", "Quitter")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Entrez votre requête…")
        yield Static(id="results")
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.query_one("#results", Static).update(f"Requête : {event.value}")


QuijoteApp().run()
