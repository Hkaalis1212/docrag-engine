"""
generation/prompts.py
-----------------------
System prompt and context-block formatting for grounded generation.
Separated from generate.py so the prompt text itself can be iterated on,
diffed, and referenced (e.g. by eval/run_eval.py in Phase 4) without
touching the calling logic around it.
"""

from __future__ import annotations

from retrieval.hybrid import RetrievedChunk

GENERATION_SYSTEM_PROMPT = """You are a documentation Q&A assistant. Answer \
the user's question using ONLY the numbered context passages provided. \
Follow these rules strictly:

1. Every factual claim must end with a bracketed citation to the passage(s) \
it came from, e.g. "DEF spec is 32.5% urea [2]." If a sentence draws on \
multiple passages, cite all of them: "... [1][3]."
2. Do not use any knowledge beyond the provided passages, even if you \
believe it to be true or recall it from elsewhere.
3. If the passages do not contain enough information to answer the \
question - fully or partially - say so explicitly, and state what IS \
covered versus what is missing. Do not guess or fill the gap.
4. Be concise and direct. This is technical documentation, not a dialogue."""


def build_context_blocks(chunks: list[RetrievedChunk]) -> str:
    """Numbers chunks 1..N in the exact order given - that numbering is the
    citation contract the system prompt relies on, so callers must pass
    chunks in the order they want them referenced as [1], [2], ..."""
    return "\n\n".join(
        f"[{i + 1}] (source: {c.metadata.get('doc_id', 'unknown')}) {c.text}"
        for i, c in enumerate(chunks)
    )
