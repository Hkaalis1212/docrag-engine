"""
generation/verify_citations.py
--------------------------------
Citation verification: for every claim-citation pair the generator
produced, checks whether the cited chunk ACTUALLY supports that specific
claim. This is the quality layer most RAG systems skip entirely - a
citation marker only proves the model intended to cite something, not that
the cited text backs up what it said next to it.

Batches every claim-citation pair into one Claude call (the same
cost-aware pattern as retrieval/rerank.py) rather than one call per pair.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from generation.generate import Citation
from retrieval.hybrid import RetrievedChunk

DEFAULT_VERIFY_MODEL = "claude-haiku-5-5"

VERIFY_SYSTEM_PROMPT = """You are a citation-verification component. You \
will be given a numbered list of (claim, supporting passage) pairs. For \
each pair, judge whether the passage actually supports the claim:

SUPPORTED   - the passage directly backs up what the claim states
PARTIAL     - the passage is related but doesn't fully support the specific claim
UNSUPPORTED - the passage does not support the claim, or contradicts it

Respond with ONLY a JSON array of strings, one verdict per pair, in the \
same order given. Example for 3 pairs: \
["SUPPORTED", "UNSUPPORTED", "PARTIAL"]. No other text, no markdown code \
fence."""

_VALID_VERDICTS = {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}


@dataclass
class CitationVerdict:
    citation: Citation
    verdict: str  # SUPPORTED | PARTIAL | UNSUPPORTED
    chunk_text: str


def _build_user_prompt(pairs: list[tuple[Citation, str]]) -> str:
    blocks = [
        f"[{i}] Claim: {citation.claim_text}\nPassage: {chunk_text}"
        for i, (citation, chunk_text) in enumerate(pairs)
    ]
    return "\n\n".join(blocks)


def _parse_verdicts(raw: str, expected_count: int) -> list[str]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    verdicts = json.loads(cleaned)
    if len(verdicts) != expected_count:
        raise ValueError(f"Verifier returned {len(verdicts)} verdicts for {expected_count} pairs")
    for v in verdicts:
        if v not in _VALID_VERDICTS:
            raise ValueError(f"Unexpected verdict value: {v!r}")
    return verdicts


def claude_verify(
    citations: list[Citation],
    chunk_lookup: dict[str, str],
    model: str = DEFAULT_VERIFY_MODEL,
    client=None,
) -> list[str]:
    if not citations:
        return []
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    pairs = [(c, chunk_lookup.get(c.chunk_id, "")) for c in citations]
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=VERIFY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(pairs)}],
    )
    return _parse_verdicts(response.content[0].text, len(citations))


def verify_citations(
    citations: list[Citation],
    used_chunks: list[RetrievedChunk],
    verify_fn: Callable[[list[Citation], dict[str, str]], list[str]] | None = None,
) -> list[CitationVerdict]:
    """
    `verify_fn` is injected for the same reason as retrieval/rerank.py's
    `score_fn`: this function's pairing/assembly logic is unit tested
    against a fake verifier, independent of the real Claude-backed
    `claude_verify` (network + API key, tested separately against a stub
    client).
    """
    if not citations:
        return []
    chunk_lookup = {c.chunk_id: c.text for c in used_chunks}
    verifier = verify_fn or claude_verify
    verdicts = verifier(citations, chunk_lookup)
    return [
        CitationVerdict(citation=c, verdict=v, chunk_text=chunk_lookup.get(c.chunk_id, ""))
        for c, v in zip(citations, verdicts)
    ]
