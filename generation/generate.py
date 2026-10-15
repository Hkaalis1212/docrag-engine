"""
generation/generate.py
------------------------
Grounded answer generation: given a question and the retrieved/reranked
chunks, produces an answer that cites its sources inline ([1], [2], ...)
and explicitly says when the context doesn't support a full answer, rather
than filling the gap from the model's own knowledge.

A low-confidence retrieval short-circuits generation entirely (see
answer_question()) - no LLM call is made if the retrieved chunks don't look
relevant enough to answer from, which also means zero hallucination risk
from asking the model to work with context it shouldn't trust.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

from generation.prompts import GENERATION_SYSTEM_PROMPT, build_context_blocks
from retrieval.hybrid import RetrievedChunk

DEFAULT_GENERATION_MODEL = "claude-sonnet-5-5"
# Below this, we don't trust the retrieved context enough to generate from
# it. Calibrated against rerank_score's 0-10 scale (4.0 = "tangentially
# related, unlikely to help" per the rerank prompt's own scale definition
# in retrieval/rerank.py), not fused_score, which has no fixed scale and is
# only ever comparable within a single retrieval call.
DEFAULT_CONFIDENCE_FLOOR = 4.0

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Citation:
    marker: int
    chunk_id: str
    claim_text: str


@dataclass
class GeneratedAnswer:
    question: str
    answer_text: str
    citations: list[Citation]
    used_chunks: list[RetrievedChunk]
    insufficient_context: bool = False
    searched_summary: str = ""  # populated only when insufficient_context


def _split_claims(answer_text: str) -> list[str]:
    # Sentence-ish split that keeps a trailing citation marker attached to
    # its own sentence - good enough for citation-claim pairing, not meant
    # to be a general-purpose sentence tokenizer.
    pieces = _SENTENCE_SPLIT_RE.split(answer_text.strip())
    return [p.strip() for p in pieces if p.strip()]


def _extract_citations(answer_text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    citations = []
    for claim in _split_claims(answer_text):
        markers = sorted({int(m) for m in _CITATION_RE.findall(claim)})
        for marker in markers:
            if 1 <= marker <= len(chunks):
                citations.append(
                    Citation(marker=marker, chunk_id=chunks[marker - 1].chunk_id, claim_text=claim)
                )
            # Out-of-range markers (model cited a passage number that wasn't
            # provided) are silently dropped rather than raising - a
            # malformed citation shouldn't crash the whole answer when the
            # answer text itself may still be useful.
    return citations


def _default_llm_call(system: str, user: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model, max_tokens=1024, system=system, messages=[{"role": "user", "content": user}]
    )
    return response.content[0].text


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str = DEFAULT_GENERATION_MODEL,
    llm_call: Callable[[str, str, str], str] | None = None,
) -> GeneratedAnswer:
    """
    Always calls the LLM - the confidence floor / "insufficient context"
    short circuit lives in answer_question(), one level up, so this stays a
    pure "given context, generate a cited answer" primitive. That split
    matters for Phase 4: the eval harness needs to generate against the
    full retrieved set regardless of confidence, to measure faithfulness
    even on weak-context cases, which answer_question()'s gate would skip.
    """
    caller = llm_call or _default_llm_call
    context = build_context_blocks(chunks)
    user_prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"
    answer_text = caller(GENERATION_SYSTEM_PROMPT, user_prompt, model).strip()
    citations = _extract_citations(answer_text, chunks)
    return GeneratedAnswer(
        question=question, answer_text=answer_text, citations=citations, used_chunks=chunks
    )


def _top_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return float("-inf")
    top = chunks[0]
    if top.rerank_score is not None:
        return top.rerank_score
    # No rerank pass was run. fused_score has no fixed scale to compare
    # against a confidence floor meaningfully, so rather than guessing a
    # threshold that would silently misfire, retrieval without reranking
    # always passes through to generation and lets the model's own
    # "insufficient context" instruction (prompts.py, rule 3) do the
    # gatekeeping instead.
    return float("inf")


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    model: str = DEFAULT_GENERATION_MODEL,
    llm_call: Callable[[str, str, str], str] | None = None,
) -> GeneratedAnswer:
    """Top-level entry point: short-circuits to a structured 'not found'
    response when the retrieved context isn't confident enough to answer
    from, otherwise delegates to generate_answer()."""
    if _top_confidence(chunks) < confidence_floor:
        searched = ", ".join(
            sorted({c.metadata.get("doc_id", "unknown") for c in chunks})
        ) or "no documents"
        return GeneratedAnswer(
            question=question,
            answer_text=(
                "I couldn't find a confident answer to this in the indexed "
                f"documentation. I searched: {searched}. You may want to check "
                "these documents manually, or the answer may not be covered "
                "in the current corpus."
            ),
            citations=[],
            used_chunks=chunks,
            insufficient_context=True,
            searched_summary=searched,
        )
    return generate_answer(question, chunks, model=model, llm_call=llm_call)
