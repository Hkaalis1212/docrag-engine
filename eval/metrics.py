"""
eval/metrics.py
-----------------
The four per-case metrics the eval harness scores every golden question on:

  answer_correctness  - LLM-as-judge against the golden reference answer
  faithfulness         - what fraction of the answer's claims carry ANY citation
                          at all (catches bare, uncited assertions)
  citation_accuracy    - of the citations that exist, what fraction actually
                          verified as SUPPORTED (catches a citation attached
                          to the wrong/unsupporting passage)
  retrieval_relevance  - recall: what fraction of the golden case's expected
                          source documents actually showed up somewhere in
                          the retrieved set

faithfulness and citation_accuracy are deliberately split rather than
collapsed into one "citations are good" number - they catch two different
failure modes (nothing cited at all, vs. something cited that doesn't
support the claim), and a chunking/retrieval strategy could move one
without moving the other.
"""

from __future__ import annotations

import os
from typing import Callable

from generation.generate import GeneratedAnswer, _split_claims
from generation.verify_citations import CitationVerdict
from retrieval.hybrid import RetrievedChunk

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"

CORRECTNESS_SYSTEM_PROMPT = """You judge whether a generated answer is \
factually consistent with a golden reference answer for the same question. \
The generated answer does not need to match wording, length, or level of \
detail - only whether the facts it states agree with the reference and it \
doesn't contradict or fabricate anything the reference doesn't support. \
Respond with ONLY a single number from 0.0 to 1.0: 1.0 means fully \
consistent and substantively complete, 0.0 means contradicts the reference \
or is substantively wrong. No other text."""


def answer_correctness(
    question: str,
    golden_answer: str,
    generated_answer: str,
    judge_fn: Callable[[str, str, str], float] | None = None,
) -> float:
    judge = judge_fn or _claude_correctness_judge
    return judge(question, golden_answer, generated_answer)


def _claude_correctness_judge(
    question: str, golden_answer: str, generated_answer: str, model: str = DEFAULT_JUDGE_MODEL
) -> float:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {golden_answer}\n\n"
        f"Generated answer: {generated_answer}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=16,
        system=CORRECTNESS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return max(0.0, min(1.0, float(response.content[0].text.strip())))


def faithfulness(answer: GeneratedAnswer) -> float:
    """
    Fraction of the answer's sentences that carry at least one citation.
    Pure text analysis - no LLM call. An "insufficient context" answer (see
    generation/generate.py) is a single honest sentence with no citations by
    design; callers should treat that case as not applicable to this metric
    rather than reading it as a faithfulness failure (eval/run_eval.py does
    this via GeneratedAnswer.insufficient_context, not this function, so
    faithfulness() itself stays a pure, context-free text metric).
    """
    claims = _split_claims(answer.answer_text)
    if not claims:
        return 1.0
    cited_claim_texts = {c.claim_text for c in answer.citations}
    covered = sum(1 for claim in claims if claim in cited_claim_texts)
    return covered / len(claims)


def citation_accuracy(verdicts: list[CitationVerdict]) -> float:
    """Of the citations that exist, what fraction verified SUPPORTED.
    Neutral (1.0) when there are no citations to judge - same convention as
    generation/confidence.py's _citation_coverage, which this duplicates in
    spirit but keeps separate: confidence.py scores one answer for display,
    this scores one eval case for aggregate reporting, and the two call
    sites shouldn't have to share an import just to stay in sync on a
    three-line fraction."""
    if not verdicts:
        return 1.0
    supported = sum(1 for v in verdicts if v.verdict == "SUPPORTED")
    return supported / len(verdicts)


def retrieval_relevance(retrieved_chunks: list[RetrievedChunk], expected_source_docs: list[str]) -> float:
    """
    Recall over expected source documents: what fraction of
    `expected_source_docs` appear among the retrieved chunks' `source_file`
    metadata. An unanswerable golden case passes `expected_source_docs=[]`;
    by convention that's treated as perfect recall (1.0) - there's nothing
    to have missed, and the thing actually worth checking there is
    `GeneratedAnswer.insufficient_context`, not this metric.
    """
    if not expected_source_docs:
        return 1.0
    retrieved_docs = {c.metadata.get("source_file") for c in retrieved_chunks}
    found = sum(1 for doc in expected_source_docs if doc in retrieved_docs)
    return found / len(expected_source_docs)
