import pytest

from generation.generate import Citation
from generation.verify_citations import claude_verify, verify_citations
from retrieval.hybrid import RetrievedChunk


def _citation(chunk_id, claim_text, marker=1):
    return Citation(marker=marker, chunk_id=chunk_id, claim_text=claim_text)


def _chunk(chunk_id, text):
    return RetrievedChunk(chunk_id=chunk_id, text=text)


# ── verify_citations() with an injected fake verifier ───────────────────────

def test_verify_citations_pairs_verdicts_with_citations_in_order():
    citations = [_citation("a", "claim one"), _citation("b", "claim two")]
    chunks = [_chunk("a", "text a"), _chunk("b", "text b")]

    def fake_verify(cites, lookup):
        return ["SUPPORTED", "UNSUPPORTED"]

    results = verify_citations(citations, chunks, verify_fn=fake_verify)

    assert results[0].citation.chunk_id == "a"
    assert results[0].verdict == "SUPPORTED"
    assert results[1].citation.chunk_id == "b"
    assert results[1].verdict == "UNSUPPORTED"


def test_verify_citations_fills_in_chunk_text():
    citations = [_citation("a", "claim")]
    chunks = [_chunk("a", "the actual source text")]

    results = verify_citations(citations, chunks, verify_fn=lambda c, l: ["SUPPORTED"])

    assert results[0].chunk_text == "the actual source text"


def test_verify_citations_on_empty_list_returns_empty():
    assert verify_citations([], [_chunk("a", "text")]) == []


def test_verify_citations_passes_chunk_lookup_to_verifier():
    citations = [_citation("a", "claim")]
    chunks = [_chunk("a", "source text")]
    captured = {}

    def fake_verify(cites, lookup):
        captured["lookup"] = lookup
        return ["SUPPORTED"]

    verify_citations(citations, chunks, verify_fn=fake_verify)

    assert captured["lookup"] == {"a": "source text"}


# ── claude_verify() with a stub client (no network, no API key) ────────────

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


def test_claude_verify_parses_plain_json():
    client = _StubClient('["SUPPORTED", "PARTIAL"]')
    citations = [_citation("a", "claim one"), _citation("b", "claim two")]

    verdicts = claude_verify(citations, {"a": "text a", "b": "text b"}, client=client)

    assert verdicts == ["SUPPORTED", "PARTIAL"]


def test_claude_verify_strips_markdown_code_fence():
    client = _StubClient('```json\n["UNSUPPORTED"]\n```')
    citations = [_citation("a", "claim")]

    verdicts = claude_verify(citations, {"a": "text"}, client=client)

    assert verdicts == ["UNSUPPORTED"]


def test_claude_verify_rejects_invalid_verdict_value():
    client = _StubClient('["MAYBE_SUPPORTED"]')
    citations = [_citation("a", "claim")]

    with pytest.raises(ValueError):
        claude_verify(citations, {"a": "text"}, client=client)


def test_claude_verify_rejects_count_mismatch():
    client = _StubClient('["SUPPORTED"]')  # only 1 for 2 citations
    citations = [_citation("a", "claim one"), _citation("b", "claim two")]

    with pytest.raises(ValueError):
        claude_verify(citations, {"a": "text a", "b": "text b"}, client=client)


def test_claude_verify_on_empty_citations_skips_the_call():
    client = _StubClient("should never be read")
    assert claude_verify([], {}, client=client) == []
