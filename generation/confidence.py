"""
generation/confidence.py
--------------------------
Composite answer confidence, scored on three independent axes:

  retrieval_confidence - how relevant was the top retrieved chunk, really?
  citation_coverage    - what fraction of citations verified as SUPPORTED?
  completeness          - did the answer address every part of the question?

Each axis is 0.0-1.0. retrieval_confidence and citation_coverage are pure
arithmetic over data the pipeline already has - no extra LLM call needed.
completeness is the one genuinely subjective axis and is scored by Claude,
with the scoring function injected for the same testability reason as
retrieval/rerank.py and generation/verify_citations.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from generation.verify_citations import CitationVerdict
from retrieval.hybrid import RetrievedChunk

DEFAULT_COMPLETENESS_MODEL = "claude-haiku-5-5"

# Weighting is a design choice, not a derived constant - citation coverage
# gets the most weight because an uncited or unsupported claim is the
# single most dangerous failure mode (a confident-sounding hallucination),
# worse than a merely incomplete but honest answer.
DEFAULT_WEIGHTS = {"retrieval": 0.3, "citation": 0.4, "completeness": 0.3}

COMPLETENESS_SYSTEM_PROMPT = """You judge whether an answer addresses every \
part of the question asked - not whether the answer is correct. Respond \
with ONLY a single number from 0.0 to 1.0: 1.0 means every part of the \
question was addressed, 0.0 means the answer is off-topic or addresses \
none of it. No other text."""


@dataclass
class ConfidenceScore:
    retrieval_confidence: float
    citation_coverage: float
    completeness: float
    composite: float


def _retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    top = chunks[0]
    if top.rerank_score is not None:
        return max(0.0, min(1.0, top.rerank_score / 10.0))
    # Without a rerank pass, fused_score has no fixed scale to normalize
    # against - a fixed neutral value is more honest than fabricating a
    # normalization that would silently misrepresent confidence for every
    # caller that skips reranking (see the identical reasoning in
    # generation/generate.py's _top_confidence).
    return 0.5


def _citation_coverage(verdicts: list[CitationVerdict]) -> float:
    if not verdicts:
        # No citations at all: in a well-formed pipeline this only happens
        # on the answer_question() "insufficient context" path, which the
        # composite score isn't meant to penalize a second time - treat as
        # neutral (fully covered, trivially), not zero.
        return 1.0
    supported = sum(1 for v in verdicts if v.verdict == "SUPPORTED")
    return supported / len(verdicts)


def claude_completeness_score(
    question: str,
    answer_text: str,
    model: str = DEFAULT_COMPLETENESS_MODEL,
    client=None,
) -> float:
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=16,
        system=COMPLETENESS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer: {answer_text}"}],
    )
    return max(0.0, min(1.0, float(response.content[0].text.strip())))


def score_confidence(
    question: str,
    answer_text: str,
    chunks: list[RetrievedChunk],
    citation_verdicts: list[CitationVerdict],
    weights: dict[str, float] | None = None,
    completeness_fn: Callable[[str, str], float] | None = None,
) -> ConfidenceScore:
    weights = weights or DEFAULT_WEIGHTS
    retrieval = _retrieval_confidence(chunks)
    citation = _citation_coverage(citation_verdicts)
    scorer = completeness_fn or claude_completeness_score
    completeness = scorer(question, answer_text)

    composite = (
        weights["retrieval"] * retrieval
        + weights["citation"] * citation
        + weights["completeness"] * completeness
    )
    return ConfidenceScore(
        retrieval_confidence=retrieval,
        citation_coverage=citation,
        completeness=completeness,
        composite=composite,
    )
