import pytest

from retrieval.hybrid import RetrievedChunk
from retrieval.rerank import claude_relevance_scores, rerank


def _chunk(chunk_id, text, fused_score=0.0):
    return RetrievedChunk(chunk_id=chunk_id, text=text, fused_score=fused_score)


# ── rerank() with an injected fake scorer ───────────────────────────────────

def test_rerank_reorders_by_score_not_fused_rank():
    candidates = [
        _chunk("a", "weakly related", fused_score=0.9),  # best fused rank
        _chunk("b", "directly answers the question", fused_score=0.1),  # worst fused rank
    ]

    def fake_scorer(question, chunks):
        return [2.0, 9.0]  # "b" actually more relevant despite worse fusion rank

    result = rerank("some question", candidates, score_fn=fake_scorer)

    assert result[0].chunk_id == "b"
    assert result[0].rerank_score == 9.0


def test_rerank_trims_to_top_n():
    candidates = [_chunk(str(i), f"text {i}") for i in range(10)]

    def fake_scorer(question, chunks):
        return list(range(len(chunks)))  # ascending: chunk "9" scores highest

    result = rerank("q", candidates, top_n=3, score_fn=fake_scorer)

    assert len(result) == 3
    assert result[0].chunk_id == "9"


def test_rerank_applies_candidate_window_before_scoring():
    candidates = [_chunk(str(i), f"text {i}") for i in range(30)]
    seen_counts = []

    def fake_scorer(question, chunks):
        seen_counts.append(len(chunks))
        return [0.0] * len(chunks)

    rerank("q", candidates, candidate_window=20, score_fn=fake_scorer)

    assert seen_counts == [20]


def test_rerank_on_empty_candidates_returns_empty():
    assert rerank("q", [], score_fn=lambda q, c: []) == []


# ── claude_relevance_scores() with a stub client (no network, no API key) ──

class _StubMessage:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class _StubMessages:
    def __init__(self, response_text):
        self._response_text = response_text

    def create(self, **kwargs):
        return _StubMessage(self._response_text)


class _StubClient:
    def __init__(self, response_text):
        self.messages = _StubMessages(response_text)


def test_claude_relevance_scores_parses_plain_json():
    client = _StubClient("[2, 8, 5]")
    candidates = [_chunk(str(i), f"text {i}") for i in range(3)]

    scores = claude_relevance_scores("q", candidates, client=client)

    assert scores == [2.0, 8.0, 5.0]


def test_claude_relevance_scores_strips_markdown_code_fence():
    client = _StubClient("```json\n[1, 2, 3]\n```")
    candidates = [_chunk(str(i), f"text {i}") for i in range(3)]

    scores = claude_relevance_scores("q", candidates, client=client)

    assert scores == [1.0, 2.0, 3.0]


def test_claude_relevance_scores_raises_on_count_mismatch():
    client = _StubClient("[1, 2]")  # only 2 scores for 3 candidates
    candidates = [_chunk(str(i), f"text {i}") for i in range(3)]

    with pytest.raises(ValueError):
        claude_relevance_scores("q", candidates, client=client)
