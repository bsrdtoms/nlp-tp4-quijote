"""Textual TUI for Don Quijote search engine."""

from __future__ import annotations

import time
from typing import ClassVar

import nltk
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, Input, Select, Static

from fdi_pln_2609_p4.corpus import get_chunks

MODES = [
    ("Búsqueda clásica", "classic"),
    ("Búsqueda semántica", "semantic"),
    ("RAG", "rag"),
]


def _score_markup(score: float) -> str:
    if score >= 0.5:
        color = "green"
    elif score >= 0.2:
        color = "yellow"
    else:
        color = "dim"
    return f"[{color}]Score: {score:.3f}[/{color}]"


def _render_results(results: list[tuple[dict, float]]) -> str:
    lines: list[str] = []
    for i, (chunk, score) in enumerate(results, 1):
        badge = f"[bold cyan]#{i}[/bold cyan]"
        score_str = _score_markup(score)
        preview = chunk["text"][:300]
        if len(chunk["text"]) > 300:
            preview += "..."
        lines.append(f"{badge} {score_str}\n{preview}\n")
    return "\n".join(lines)


class SearchApp(App):
    """Don Quijote search TUI."""

    CSS: ClassVar[str] = """
    #status {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #results {
        border: solid $primary;
        padding: 1;
    }
    Input {
        margin: 1 0;
    }
    Select {
        margin: 1 0;
    }
    """

    def __init__(self, chunks: list[dict]) -> None:
        super().__init__()
        self.chunks = chunks
        self._tfidf_ready = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Select(options=MODES, value="classic", id="mode")
        yield Input(placeholder="Escribe tu búsqueda...", id="query")
        yield Static("Listo.", id="status")
        yield ScrollableContainer(Static("", id="results_inner"), id="results")
        yield Footer()

    def on_mount(self) -> None:
        # Pre-build TF-IDF index (fast enough to do at startup)
        self._build_tfidf()

    def _build_tfidf(self) -> None:
        from fdi_pln_2609_p4 import classic

        try:
            classic.load_or_build_index(self.chunks)
            self._tfidf_ready = True
            self.query_one("#status", Static).update("Índice TF-IDF listo.")
        except Exception as e:
            self.query_one("#status", Static).update(f"[red]Error TF-IDF: {e}[/red]")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        mode_widget = self.query_one("#mode", Select)
        mode = str(mode_widget.value)

        status = self.query_one("#status", Static)
        results_widget = self.query_one("#results_inner", Static)

        status.update("Buscando...")

        t0 = time.monotonic()
        try:
            if mode == "classic":
                from fdi_pln_2609_p4 import classic

                results = classic.search(query, self.chunks)
                elapsed = time.monotonic() - t0
                status.update(f"{len(results)} resultados | {elapsed:.2f}s")
                results_widget.update(_render_results(results))

            elif mode == "semantic":
                status.update("Calculando embeddings...")
                try:
                    from fdi_pln_2609_p4 import semantic

                    results = semantic.search(query, self.chunks)
                    elapsed = time.monotonic() - t0
                    status.update(f"{len(results)} resultados | {elapsed:.2f}s")
                    results_widget.update(_render_results(results))
                except Exception as e:
                    status.update(f"[red]Error ollama: {e}[/red]")
                    results_widget.update(
                        "[red]Ollama no disponible. ¿Está corriendo el servidor?[/red]"
                    )

            elif mode == "rag":
                status.update("Consultando LLM...")
                try:
                    from fdi_pln_2609_p4 import classic, rag as rag_mod

                    results = classic.search(query, self.chunks, top_k=5)
                    answer = rag_mod.rag(query, results)
                    elapsed = time.monotonic() - t0
                    status.update(f"RAG completado | {elapsed:.2f}s")

                    refs = "\n\n[bold]Fragmentos de referencia:[/bold]\n"
                    for i, (chunk, score) in enumerate(results[:3], 1):
                        refs += f"\n[bold cyan][{i}][/bold cyan] {_score_markup(score)}\n{chunk['text'][:200]}...\n"

                    results_widget.update(answer + refs)
                except Exception as e:
                    status.update(f"[red]Error LLM: {e}[/red]")
                    results_widget.update(
                        "[red]Ollama no disponible. ¿Está corriendo el servidor?[/red]"
                    )

        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")
            results_widget.update(f"[red]{e}[/red]")


def main() -> None:
    """Entry point: download NLTK data, load chunks, run TUI."""
    nltk.download("stopwords", quiet=True)

    chunks = get_chunks()
    app = SearchApp(chunks)
    app.run()


if __name__ == "__main__":
    main()
