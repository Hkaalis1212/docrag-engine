from generation.confidence import score_confidence
from generation.generate import Citation
from generation.verify_citations import CitationVerdict
from retrieval.hybrid import RetrievedChunk


def _chunk(rerank_score=None, fused_score=0.0):
    return RetrievedChunk(chunk_id="a", text="text", rerank_score=rerank_score, fused_score=fused_score)


def _verdict(verdict):
    citation = Citation(marker=1, chunk_id="a", claim_text="claim")
    return CitationVerdict(citation=citation, verdict=verdict, chunk_text="text")


def test_retrieval_confidence_normalizes_rerank_score_to_unit_range():
    result = score_confidence(
        "q", "answer", [_chunk(rerank_score=8.0)], [], completeness_fn=lambda q, a: 1.0
    )
    assert result.retrieval_confidence == 0.8


def test_retrieval_confidence_falls_back_to_neutral_without_rerank_score():
    result = score_confidence(
        "q", "answer", [_chunk(rerank_score=None)], [], completeness_fn=lambda q, a: 1.0
    )
    assert result.retrieval_confidence == 0.5


def test_retrieval_confidence_zero_on_no_chunks():
    result = score_confidence("q", "answer", [], [], completeness_fn=lambda q, a: 1.0)
    assert result.retrieval_confidence == 0.0


def test_citation_coverage_is_fraction_supported():
    verdicts = [_verdict("SUPPORTED"), _verdict("SUPPORTED"), _verdict("UNSUPPORTED"), _verdict("PARTIAL")]
    result = score_confidence(
        "q", "answer", [_chunk(rerank_score=10.0)], verdicts, completeness_fn=lambda q, a: 1.0
    )
    assert result.citation_coverage == 0.5


def test_citation_coverage_neutral_when_no_citations():
    result = score_confidence(
        "q", "answer", [_chunk(rerank_score=10.0)], [], completeness_fn=lambda q, a: 1.0
    )
    assert result.citation_coverage == 1.0


def test_completeness_uses_injected_scorer():
    captured = {}

    def fake_completeness(question, answer_text):
        captured["question"] = question
        captured["answer"] = answer_text
        return 0.7

    result = score_confidence(
        "my question", "my answer", [_chunk(rerank_score=10.0)], [], completeness_fn=fake_completeness
    )

    assert result.completeness == 0.7
    assert captured == {"question": "my question", "answer": "my answer"}


def test_composite_is_weighted_sum_of_the_three_axes():
    result = score_confidence(
        "q",
        "answer",
        [_chunk(rerank_score=10.0)],  # retrieval = 1.0
        [_verdict("SUPPORTED")],  # citation = 1.0
        completeness_fn=lambda q, a: 1.0,  # completeness = 1.0
    )
    assert result.composite == 1.0  # weights sum to 1.0, all axes maxed


def test_composite_respects_custom_weights():
    result = score_confidence(
        "q",
        "answer",
        [_chunk(rerank_score=10.0)],  # retrieval = 1.0
        [],  # citation = 1.0 (neutral, no citations)
        weights={"retrieval": 1.0, "citation": 0.0, "completeness": 0.0},
        completeness_fn=lambda q, a: 0.0,  # would drag composite down if weighted
    )
    assert result.composite == 1.0
