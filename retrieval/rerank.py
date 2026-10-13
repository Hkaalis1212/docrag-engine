"""
retrieval/rerank.py
--------------------
Second-pass relevance scoring over the RRF-fused candidates. Fusion ranks by
rank position across two lexical/semantic signals — it has no idea what the
question is actually asking. Reranking adds that: it reads the literal
question text against each candidate and catches cases where a chunk looks
topically related (good dense + sparse rank) but doesn't actually answer
what was asked, demoting it below one that does.

One batched Claude call scores every candidate at once (a JSON array of
scores, same order as the input) rather than one call per candidate —
roughly 20x fewer API calls at the default 20-candidate window, for the
cost of a longer single prompt. This is the production-cost-aware choice
over the naive one-call-per-chunk design, and the kind of tradeoff worth
being able to explain in an interview.
"""

from __future__ import annotations

import json
import os
from typing import Callable

from retrieval.hybrid import RetrievedChunk

DEFAULT_RERANK_MODEL = "claude-haiku-5-5"
DEFAULT_CANDIDATE_WINDOW = 20
DEFAULT_TOP_N = 5

RERANK_SYSTEM_PROMPT = """You are a relevance-scoring component in a retrieval \
pipeline. You will be given a user question and a numbered list of candidate \
text passages. Score each passage's relevance to answering the question on a \
0-10 integer scale:

0-2  = irrelevant or off-topic
3-5  = tangentially related, unlikely to help answer the question
6-8  = relevant, contains some useful information
9-10 = directly and substantially answers the question

Respond with ONLY a JSON array of integers, one score per passage, in the \
same order as given. Example for 3 passages: [2, 8, 5]. No other text, no \
markdown code fence."""


def _build_user_prompt(question: str, candidates: list[RetrievedChunk]) -> str:
    passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
    return f"Question: {question}\n\nPassages:\n\n{passages}"


def _parse_scores(raw: str, expected_count: int) -> list[float]:
    # Claude occasionally wraps JSON in a code fence despite instructions —
    # strip it rather than failing the whole rerank pass over formatting.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    scores = json.loads(cleaned)
    if len(scores) != expected_count:
        raise ValueError(
            f"Reranker returned {len(scores)} scores for {expected_count} candidates"
        )
    return [float(s) for s in scores]


def claude_relevance_scores(
    question: str,
    candidates: list[RetrievedChunk],
    model: str = DEFAULT_RERANK_MODEL,
    client=None,
) -> list[float]:
    """
    Real scoring function — calls Claude once with every candidate batched
    into a single prompt. Requires ANTHROPIC_API_KEY unless `client` is
    injected (used by integration tests with a stub client, so the parsing
    and prompt-building logic can be exercised without a live API key).
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=RERANK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(question, candidates)}],
    )
    return _parse_scores(response.content[0].text, len(candidates))


def rerank(
    question: str,
    candidates: list[RetrievedChunk],
    top_n: int = DEFAULT_TOP_N,
    candidate_window: int = DEFAULT_CANDIDATE_WINDOW,
    score_fn: Callable[[str, list[RetrievedChunk]], list[float]] | None = None,
) -> list[RetrievedChunk]:
    """
    Takes the RRF-fused candidate list (already sorted best-first), rescopes
    to the top `candidate_window`, scores each for relevance, and returns the
    top `top_n` by that score.

    `score_fn` is injected specifically so the selection/sorting logic here
    can be unit tested with a fake scorer, independent of the real
    Claude-backed `claude_relevance_scores` (which needs a network call and
    an API key, and belongs in an integration test instead — see
    tests/test_rerank.py for both).
    """
    windowed = candidates[:candidate_window]
    if not windowed:
        return []

    scorer = score_fn or claude_relevance_scores
    scores = scorer(question, windowed)

    for chunk, score in zip(windowed, scores):
        chunk.rerank_score = score

    return sorted(windowed, key=lambda c: c.rerank_score, reverse=True)[:top_n]
