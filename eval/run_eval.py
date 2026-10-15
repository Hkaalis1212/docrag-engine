"""
eval/run_eval.py
------------------
The chunking-strategy bake-off: loads the golden Q&A dataset, builds a
fully separate index per chunking strategy, runs every question through
the full pipeline (retrieve -> rerank -> generate -> verify -> score) for
each strategy, and prints a comparison report.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --strategies structure fixed
    python -m eval.run_eval --golden eval/golden_dataset.jsonl --corpus corpus/throttleguard-docs

Needs ANTHROPIC_API_KEY (rerank, generation, citation verification, and
judging all call Claude) and a working local embedding model. Every call
that hits an LLM is injectable here too, the same pattern as the rest of
the pipeline, specifically so this module's orchestration/aggregation
logic is unit-testable with fakes - see tests/test_run_eval.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import answer_correctness, citation_accuracy, faithfulness, retrieval_relevance
from generation.generate import DEFAULT_CONFIDENCE_FLOOR, GeneratedAnswer, answer_question
from generation.verify_citations import verify_citations
from indexing.embeddings import get_embedder
from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore
from retrieval.hybrid import HybridRetriever, RetrievedChunk
from retrieval.rerank import rerank
from scripts.seed import seed

DEFAULT_GOLDEN_PATH = "eval/golden_dataset.jsonl"
DEFAULT_CORPUS_DIR = "corpus/throttleguard-docs"
DEFAULT_STRATEGIES = ["fixed", "structure", "semantic"]

VALID_CATEGORIES = {"direct", "multi_hop", "unanswerable", "ambiguous"}


@dataclass
class EvalCase:
    id: str
    question: str
    golden_answer: str
    category: str
    source_docs: list[str]
    expect_insufficient: bool


@dataclass
class CaseResult:
    case: EvalCase
    strategy: str
    generated_answer: GeneratedAnswer
    correctness: float
    faithfulness: float
    citation_accuracy: float
    retrieval_relevance: float
    insufficient_context_correct: bool


@dataclass
class StrategyReport:
    strategy: str
    results: list[CaseResult] = field(default_factory=list)


def load_golden_dataset(path: str | Path) -> list[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["category"] not in VALID_CATEGORIES:
                raise ValueError(
                    f"{path}:{line_no}: unknown category {row['category']!r}, "
                    f"expected one of {sorted(VALID_CATEGORIES)}"
                )
            cases.append(
                EvalCase(
                    id=row["id"],
                    question=row["question"],
                    golden_answer=row["golden_answer"],
                    category=row["category"],
                    source_docs=row["source_docs"],
                    expect_insufficient=row["expect_insufficient"],
                )
            )
    return cases


def run_case(
    case: EvalCase,
    strategy: str,
    retriever: HybridRetriever,
    dense_k: int = 10,
    sparse_k: int = 10,
    rerank_top_n: int = 5,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    rerank_score_fn: Callable | None = None,
    llm_call: Callable | None = None,
    verify_fn: Callable | None = None,
    judge_fn: Callable | None = None,
) -> CaseResult:
    fused = retriever.retrieve(case.question, dense_k=dense_k, sparse_k=sparse_k)
    reranked = rerank(case.question, fused, top_n=rerank_top_n, score_fn=rerank_score_fn)
    answer = answer_question(
        case.question, reranked, confidence_floor=confidence_floor, llm_call=llm_call
    )

    verdicts = (
        []
        if answer.insufficient_context
        else verify_citations(answer.citations, answer.used_chunks, verify_fn=verify_fn)
    )

    return CaseResult(
        case=case,
        strategy=strategy,
        generated_answer=answer,
        correctness=answer_correctness(
            case.question, case.golden_answer, answer.answer_text, judge_fn=judge_fn
        ),
        faithfulness=faithfulness(answer),
        citation_accuracy=citation_accuracy(verdicts),
        # Scored against what generation actually saw (the reranked top-N),
        # not the wider pre-rerank fused set - that's the set whose recall
        # actually bounds answer quality.
        retrieval_relevance=retrieval_relevance(reranked, case.source_docs),
        insufficient_context_correct=(answer.insufficient_context == case.expect_insufficient),
    )


def run_strategy(
    strategy: str,
    cases: list[EvalCase],
    retriever: HybridRetriever,
    **run_case_kwargs,
) -> StrategyReport:
    report = StrategyReport(strategy=strategy)
    for case in cases:
        report.results.append(run_case(case, strategy, retriever, **run_case_kwargs))
    return report


def aggregate_metrics(results: list[CaseResult]) -> dict:
    if not results:
        return {
            "n": 0, "correctness": 0.0, "faithfulness": 0.0,
            "citation_accuracy": 0.0, "retrieval_relevance": 0.0,
            "insufficient_context_accuracy": 0.0,
        }
    n = len(results)
    return {
        "n": n,
        "correctness": sum(r.correctness for r in results) / n,
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "citation_accuracy": sum(r.citation_accuracy for r in results) / n,
        "retrieval_relevance": sum(r.retrieval_relevance for r in results) / n,
        "insufficient_context_accuracy": sum(
            1 for r in results if r.insufficient_context_correct
        ) / n,
    }


def aggregate_by_category(results: list[CaseResult]) -> dict[str, dict]:
    by_cat: dict[str, list[CaseResult]] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r)
    return {cat: aggregate_metrics(rs) for cat, rs in by_cat.items()}


def build_comparison_report(reports: list[StrategyReport]) -> str:
    lines = ["# Chunking Strategy Bake-Off\n"]
    lines.append("| Strategy | N | Correctness | Faithfulness | Citation Acc. | Retrieval Rec. | Insuff.-Ctx Acc. |")
    lines.append("|---|---|---|---|---|---|---|")
    for report in reports:
        agg = aggregate_metrics(report.results)
        lines.append(
            f"| {report.strategy} | {agg['n']} | {agg['correctness']:.2f} | "
            f"{agg['faithfulness']:.2f} | {agg['citation_accuracy']:.2f} | "
            f"{agg['retrieval_relevance']:.2f} | {agg['insufficient_context_accuracy']:.2f} |"
        )

    lines.append("\n## By category\n")
    for report in reports:
        lines.append(f"### {report.strategy}\n")
        lines.append("| Category | N | Correctness | Faithfulness | Citation Acc. | Retrieval Rec. |")
        lines.append("|---|---|---|---|---|---|")
        for cat, agg in sorted(aggregate_by_category(report.results).items()):
            lines.append(
                f"| {cat} | {agg['n']} | {agg['correctness']:.2f} | "
                f"{agg['faithfulness']:.2f} | {agg['citation_accuracy']:.2f} | "
                f"{agg['retrieval_relevance']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _build_retriever(strategy: str, corpus_dir: str, work_dir: Path) -> HybridRetriever:
    persist_dir = work_dir / f"chroma_{strategy}"
    sparse_path = work_dir / f"sparse_{strategy}.pkl"
    seed(corpus_dir, strategy, persist_dir=str(persist_dir), sparse_index_path=str(sparse_path))
    embedder = get_embedder()
    return HybridRetriever(
        VectorStore(persist_dir=str(persist_dir)), SparseIndex.load(sparse_path), embedder
    )


def run_eval(
    golden_path: str = DEFAULT_GOLDEN_PATH,
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    strategies: list[str] = DEFAULT_STRATEGIES,
    work_dir: str = "eval_indexes",
) -> list[StrategyReport]:
    cases = load_golden_dataset(golden_path)
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True)

    reports = []
    for strategy in strategies:
        print(f"\n=== Building index: {strategy} ===")
        retriever = _build_retriever(strategy, corpus_dir, work_path)
        print(f"=== Running {len(cases)} eval cases against {strategy} ===")
        reports.append(run_strategy(strategy, cases, retriever))

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chunking-strategy bake-off.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--work-dir", default="eval_indexes")
    parser.add_argument("--out", default="eval_report.md")
    args = parser.parse_args()

    reports = run_eval(args.golden, args.corpus, args.strategies, args.work_dir)
    report_text = build_comparison_report(reports)
    print("\n" + report_text)
    Path(args.out).write_text(report_text)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
