from eval.metrics import answer_correctness, citation_accuracy, faithfulness, retrieval_relevance
from generation.generate import Citation, GeneratedAnswer
from generation.verify_citations import CitationVerdict
from retrieval.hybrid import RetrievedChunk


def _chunk(chunk_id, source_file="doc1.md"):
    return RetrievedChunk(chunk_id=chunk_id, text="text", metadata={"source_file": source_file})


def _citation(chunk_id, claim_text):
    return Citation(marker=1, chunk_id=chunk_id, claim_text=claim_text)


def _verdict(verdict, chunk_id="a", claim_text="claim"):
    return CitationVerdict(citation=_citation(chunk_id, claim_text), verdict=verdict, chunk_text="text")


# ── answer_correctness() ────────────────────────────────────────────────────

def test_answer_correctness_uses_injected_judge():
    captured = {}

    def fake_judge(question, golden, generated):
        captured.update(question=question, golden=golden, generated=generated)
        return 0.75

    score = answer_correctness("q", "golden answer", "generated answer", judge_fn=fake_judge)

    assert score == 0.75
    assert captured == {"question": "q", "golden": "golden answer", "generated": "generated answer"}


# ── faithfulness() ───────────────────────────────────────────────────────────

def test_faithfulness_full_when_every_sentence_cited():
    answer = GeneratedAnswer(
        question="q",
        answer_text="DEF spec is 32.5% urea [1]. Below 20% is contamination [1].",
        citations=[
            _citation("a", "DEF spec is 32.5% urea [1]."),
            _citation("a", "Below 20% is contamination [1]."),
        ],
        used_chunks=[],
    )
    assert faithfulness(answer) == 1.0


def test_faithfulness_partial_when_some_sentences_uncited():
    answer = GeneratedAnswer(
        question="q",
        answer_text="Cited fact here [1]. Uncited fact here with no marker at all.",
        citations=[_citation("a", "Cited fact here [1].")],
        used_chunks=[],
    )
    assert faithfulness(answer) == 0.5


def test_faithfulness_on_empty_answer_text_is_neutral():
    answer = GeneratedAnswer(question="q", answer_text="", citations=[], used_chunks=[])
    assert faithfulness(answer) == 1.0


# ── citation_accuracy() ──────────────────────────────────────────────────────

def test_citation_accuracy_fraction_supported():
    verdicts = [_verdict("SUPPORTED"), _verdict("SUPPORTED"), _verdict("UNSUPPORTED"), _verdict("PARTIAL")]
    assert citation_accuracy(verdicts) == 0.5


def test_citation_accuracy_neutral_on_no_citations():
    assert citation_accuracy([]) == 1.0


def test_citation_accuracy_zero_when_all_unsupported():
    verdicts = [_verdict("UNSUPPORTED"), _verdict("UNSUPPORTED")]
    assert citation_accuracy(verdicts) == 0.0


# ── retrieval_relevance() ────────────────────────────────────────────────────

def test_retrieval_relevance_full_recall():
    chunks = [_chunk("a", "doc1.md"), _chunk("b", "doc2.md")]
    assert retrieval_relevance(chunks, ["doc1.md", "doc2.md"]) == 1.0


def test_retrieval_relevance_partial_recall():
    chunks = [_chunk("a", "doc1.md")]
    assert retrieval_relevance(chunks, ["doc1.md", "doc2.md"]) == 0.5


def test_retrieval_relevance_zero_when_nothing_expected_found():
    chunks = [_chunk("a", "doc3.md")]
    assert retrieval_relevance(chunks, ["doc1.md"]) == 0.0


def test_retrieval_relevance_neutral_when_nothing_expected():
    chunks = [_chunk("a", "doc1.md")]
    assert retrieval_relevance(chunks, []) == 1.0


def test_retrieval_relevance_does_not_double_count_duplicate_source_file():
    # Two chunks from the same expected doc shouldn't inflate the score
    # past 1.0 for that one expected doc.
    chunks = [_chunk("a", "doc1.md"), _chunk("b", "doc1.md")]
    assert retrieval_relevance(chunks, ["doc1.md"]) == 1.0
