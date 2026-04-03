"""RAG (Retrieval-Augmented Generation) using ollama."""

from __future__ import annotations

from loguru import logger

from fdi_pln_2609_p4.corpus import Chunk

DEFAULT_MODEL = "llama3:latest"


def rag(
    query: str,
    results: list[tuple[Chunk, float]],
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate an answer grounded in retrieved chunks.

    Takes the top 3 results, builds a numbered context and asks the LLM
    to answer in Spanish with citations.
    """
    import ollama

    top = results[:3]
    if not top:
        return "No se encontraron fragmentos relevantes."

    context_parts = [f"[{i + 1}] {chunk['text']}" for i, (chunk, _) in enumerate(top)]
    context = "\n\n".join(context_parts)

    prompt = (
        f"Usa los siguientes fragmentos del Quijote para responder a la pregunta. "
        f"Cita las referencias usando [1], [2], [3] según corresponda.\n\n"
        f"Fragmentos:\n{context}\n\n"
        f"Pregunta: {query}\n\n"
        f"Respuesta:"
    )

    logger.info(f"Calling ollama model={model}")
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.message.content
