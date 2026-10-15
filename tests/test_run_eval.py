import json

from eval.run_eval import (
    CaseResult,
    EvalCase,
    aggregate_by_category,
    aggregate_metrics,
    build_comparison_report,
    load_golden_dataset,
    run_case,
    run_strategy,
    StrategyReport,
)
from generation.generate import GeneratedAnswer
from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore
from retrieval.hybrid import HybridRetriever


# ── load_golden_dataset() ───────────────────────────────────────────────────

def test_load_golden_dataset_parses_jsonl(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({
            "id": "D1", "question": "q1", "golden_answer": "a1", "category": "direct",
            "source_docs": ["doc1.md"], "expect_insufficient": False,
        }) + "\n"
        + json.dumps({
            "id": "U1", "question": "q2", "golden_answer": "a2", "category": "unanswerable",
            "source_docs": [], "expect_insufficient": True,
        }) + "\n"
    )

    cases = load_golden_dataset(path)

    assert len(cases) == 2
    assert cases[0].id == "D1"
    assert cases[0].category == "direct"
    assert cases[1].expect_insufficient is True


def test_load_golden_dataset_skips_blank_lines(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n"
        + json.dumps({
            "id": "D1", "question": "q", "golden_answer": "a", "category": "direct",
            "source_docs": ["doc1.md"], "expect_insufficient": False,
        })
        + "\n\n"
    )
    cases = load_golden_dataset(path)
    assert len(cases) == 1


def test_load_golden_dataset_rejects_unknown_category(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({
            "id": "X1", "question": "q", "golden_answer": "a", "category": "bogus",
            "source_docs": [], "expect_insufficient": False,
        }) + "\n"
    )
    try:
        load_golden_dataset(path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


# ── run_case() with a fully fake pipeline ───────────────────────────────────

def _retriever(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["DEF spec is 32.5 percent urea", "unrelated content about tires"],
        metadatas=[{"source_file": "doc1.md"}, {"source_file": "doc2.md"}],
    )
    sparse = SparseIndex()
    sparse.build(
        chunk_ids=["a", "b"],
        texts=["DEF spec is 32.5 percent urea", "unrelated content about tires"],
        metadatas=[{"source_file": "doc1.md"}, {"source_file": "doc2.md"}],
    )

    class FakeEmbedder:
        def embed_one(self, text):
            return [1.0, 0.0]  # always "closest" to chunk "a"

    return HybridRetriever(store, sparse, FakeEmbedder())


def test_run_case_wires_the_full_pipeline_together(tmp_path):
    case = EvalCase(
        id="D1", question="What's the DEF spec?", golden_answer="32.5% urea",
        category="direct", source_docs=["doc1.md"], expect_insufficient=False,
    )
    retriever = _retriever(tmp_path)

    result = run_case(
        case, "fixed", retriever,
        rerank_score_fn=lambda q, c: [10.0] * len(c),
        llm_call=lambda s, u, m: "DEF spec is 32.5% urea [1].",
        verify_fn=lambda cites, lookup: ["SUPPORTED"] * len(cites),
        judge_fn=lambda q, g, gen: 1.0,
    )

    assert isinstance(result, CaseResult)
    assert result.case.id == "D1"
    assert result.correctness == 1.0
    assert result.citation_accuracy == 1.0
    assert result.retrieval_relevance == 1.0  # doc1.md was retrieved
    assert result.insufficient_context_correct is True  # expected False, got False


def test_run_case_insufficient_context_path(tmp_path):
    case = EvalCase(
        id="U1", question="unrelated off-topic question", golden_answer="not covered",
        category="unanswerable", source_docs=[], expect_insufficient=True,
    )
    retriever = _retriever(tmp_path)

    result = run_case(
        case, "fixed", retriever,
        rerank_score_fn=lambda q, c: [0.0] * len(c),  # below confidence floor
        confidence_floor=4.0,
        judge_fn=lambda q, g, gen: 1.0,
    )

    assert result.generated_answer.insufficient_context is True
    assert result.insufficient_context_correct is True
    assert result.citation_accuracy == 1.0  # neutral, no citations to judge


def test_run_case_flags_wrong_insufficient_context_prediction(tmp_path):
    # Golden case expects a real answer, but confidence floor is set so high
    # that the pipeline bails out anyway - should be flagged as incorrect.
    case = EvalCase(
        id="D1", question="q", golden_answer="a", category="direct",
        source_docs=["doc1.md"], expect_insufficient=False,
    )
    retriever = _retriever(tmp_path)

    result = run_case(
        case, "fixed", retriever,
        rerank_score_fn=lambda q, c: [1.0] * len(c),
        confidence_floor=99.0,  # impossible to clear
        judge_fn=lambda q, g, gen: 0.0,
    )

    assert result.generated_answer.insufficient_context is True
    assert result.insufficient_context_correct is False  # expected False, got True


# ── aggregate_metrics() / aggregate_by_category() ───────────────────────────

def _fake_result(category="direct", correctness=1.0, faithfulness=1.0,
                  citation_accuracy=1.0, retrieval_relevance=1.0, insuff_correct=True):
    case = EvalCase(
        id="x", question="q", golden_answer="a", category=category,
        source_docs=["doc1.md"], expect_insufficient=False,
    )
    answer = GeneratedAnswer(question="q", answer_text="a", citations=[], used_chunks=[])
    return CaseResult(
        case=case, strategy="fixed", generated_answer=answer,
        correctness=correctness, faithfulness=faithfulness,
        citation_accuracy=citation_accuracy, retrieval_relevance=retrieval_relevance,
        insufficient_context_correct=insuff_correct,
    )


def test_aggregate_metrics_averages_correctly():
    results = [_fake_result(correctness=1.0), _fake_result(correctness=0.0)]
    agg = aggregate_metrics(results)
    assert agg["n"] == 2
    assert agg["correctness"] == 0.5


def test_aggregate_metrics_on_empty_list():
    agg = aggregate_metrics([])
    assert agg["n"] == 0
    assert agg["correctness"] == 0.0


def test_aggregate_by_category_splits_correctly():
    results = [
        _fake_result(category="direct", correctness=1.0),
        _fake_result(category="direct", correctness=0.5),
        _fake_result(category="unanswerable", correctness=0.0),
    ]
    by_cat = aggregate_by_category(results)
    assert by_cat["direct"]["n"] == 2
    assert by_cat["direct"]["correctness"] == 0.75
    assert by_cat["unanswerable"]["n"] == 1


# ── build_comparison_report() ───────────────────────────────────────────────

def test_build_comparison_report_includes_every_strategy():
    reports = [
        StrategyReport(strategy="fixed", results=[_fake_result()]),
        StrategyReport(strategy="structure", results=[_fake_result(correctness=0.5)]),
    ]
    report = build_comparison_report(reports)
    assert "fixed" in report
    assert "structure" in report
    assert "Chunking Strategy Bake-Off" in report


def test_build_comparison_report_handles_empty_results():
    reports = [StrategyReport(strategy="fixed", results=[])]
    report = build_comparison_report(reports)
    assert "fixed" in report  # doesn't crash on an empty strategy
