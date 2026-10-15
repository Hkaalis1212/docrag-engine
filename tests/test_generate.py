import pytest

from generation.generate import answer_question, generate_answer
from retrieval.hybrid import RetrievedChunk


def _chunk(chunk_id, text, metadata=None, rerank_score=None, fused_score=0.0):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=metadata or {"doc_id": "doc1"},
        rerank_score=rerank_score,
        fused_score=fused_score,
    )


# ── generate_answer() + citation extraction ─────────────────────────────────

def test_generate_answer_extracts_single_citation():
    chunks = [_chunk("a", "DEF spec is 32.5 percent urea.")]

    def fake_llm(system, user, model):
        return "DEF concentration spec is 32.5% urea [1]."

    result = generate_answer("What's the DEF spec?", chunks, llm_call=fake_llm)

    assert len(result.citations) == 1
    assert result.citations[0].marker == 1
    assert result.citations[0].chunk_id == "a"
    assert "32.5%" in result.citations[0].claim_text


def test_generate_answer_extracts_multiple_citations_in_one_sentence():
    chunks = [_chunk("a", "text a"), _chunk("b", "text b")]

    def fake_llm(system, user, model):
        return "Both passages agree on this point [1][2]."

    result = generate_answer("q", chunks, llm_call=fake_llm)

    markers = sorted(c.marker for c in result.citations)
    assert markers == [1, 2]


def test_generate_answer_drops_out_of_range_citation():
    chunks = [_chunk("a", "text a")]

    def fake_llm(system, user, model):
        return "This cites a passage that doesn't exist [5]."

    result = generate_answer("q", chunks, llm_call=fake_llm)

    assert result.citations == []  # dropped, not raised


def test_generate_answer_passes_question_and_context_to_llm():
    chunks = [_chunk("a", "the actual passage text")]
    captured = {}

    def fake_llm(system, user, model):
        captured["user"] = user
        return "answer [1]."

    generate_answer("my specific question", chunks, llm_call=fake_llm)

    assert "my specific question" in captured["user"]
    assert "the actual passage text" in captured["user"]


def test_generate_answer_never_marked_insufficient():
    chunks = [_chunk("a", "text")]

    def fake_llm(system, user, model):
        return "no citations here"

    result = generate_answer("q", chunks, llm_call=fake_llm)
    assert result.insufficient_context is False


# ── answer_question() confidence gate ───────────────────────────────────────

def test_answer_question_short_circuits_below_confidence_floor():
    chunks = [_chunk("a", "weak match", rerank_score=1.5)]
    called = []

    def fake_llm(system, user, model):
        called.append(True)
        return "should never run"

    result = answer_question("q", chunks, confidence_floor=4.0, llm_call=fake_llm)

    assert result.insufficient_context is True
    assert called == []  # LLM never invoked
    assert result.citations == []


def test_answer_question_proceeds_above_confidence_floor():
    chunks = [_chunk("a", "strong match", rerank_score=8.0)]

    def fake_llm(system, user, model):
        return "a confident answer [1]."

    result = answer_question("q", chunks, confidence_floor=4.0, llm_call=fake_llm)

    assert result.insufficient_context is False
    assert len(result.citations) == 1


def test_answer_question_without_rerank_score_always_proceeds():
    # No rerank pass was run (rerank_score is None) - should never trip the
    # confidence floor, per _top_confidence's documented fallback.
    chunks = [_chunk("a", "text", rerank_score=None)]

    def fake_llm(system, user, model):
        return "answer [1]."

    result = answer_question("q", chunks, confidence_floor=4.0, llm_call=fake_llm)
    assert result.insufficient_context is False


def test_answer_question_on_empty_chunks_short_circuits():
    result = answer_question("q", [], llm_call=lambda s, u, m: "unused")
    assert result.insufficient_context is True
    assert result.searched_summary == "no documents"


def test_answer_question_searched_summary_lists_doc_ids():
    chunks = [
        _chunk("a", "text", metadata={"doc_id": "doc1"}, rerank_score=1.0),
        _chunk("b", "text", metadata={"doc_id": "doc2"}, rerank_score=1.0),
    ]
    result = answer_question("q", chunks, confidence_floor=4.0, llm_call=lambda s, u, m: "unused")
    assert "doc1" in result.searched_summary
    assert "doc2" in result.searched_summary
