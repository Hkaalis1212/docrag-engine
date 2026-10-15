# docrag-engine

A production-grade hybrid-retrieval RAG system: ingests internal
documentation, indexes it with both dense (vector) and sparse (BM25) search,
fuses and reranks results, and generates grounded answers with verified
inline citations — benchmarked against a 50+ question golden eval set across
three different chunking strategies.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design, stack
rationale, and phase-by-phase build plan.

## Status

| Phase | Status |
|---|---|
| 1 — Ingestion, chunking, indexing | ✅ built |
| 2 — Hybrid retrieval (dense + BM25 + RRF + rerank) | ✅ built |
| 3 — Grounded generation + citation verification | ✅ built |
| 4 — Evaluation framework (golden set + metrics) | ✅ built |
| 5 — API + dashboard | ⬜ not started |
| 6 — Portfolio polish (demo + case study) | ⬜ not started |

## Quickstart (Phase 1)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Index the sample corpus (ThrottleGuard's own engineering docs) with the
# structure-aware chunker. First run downloads the local embedding model
# (~130MB, one-time, no API key needed).
python -m scripts.seed --corpus corpus/throttleguard-docs --strategy structure

# Each strategy gets its OWN index (separate Chroma dir + BM25 file) - do
# not reuse the same --persist-dir/--sparse-index-path across strategies,
# or the second run's BM25 index silently overwrites the first's:
python -m scripts.seed --corpus corpus/throttleguard-docs --strategy fixed \
    --persist-dir chroma_db_fixed --sparse-index-path sparse_index_fixed.pkl
python -m scripts.seed --corpus corpus/throttleguard-docs --strategy semantic \
    --persist-dir chroma_db_semantic --sparse-index-path sparse_index_semantic.pkl

# Or just run the Phase 4 eval harness, which builds all three automatically:
python -m eval.run_eval
```

Run the test suite (no API key, no model download required — chunking and
loader tests use a fake embedder):

```bash
pytest tests/ -v
```

## Why ThrottleGuard's docs as the sample corpus

Real, owned, technical documentation — rule tables, threshold values, env
var references, file trees — with exactly the kind of structure that makes
chunking strategy actually matter for retrieval quality. Swappable: point
`--corpus` at any directory of markdown/text/HTML/PDF files and it reindexes
from scratch.

## Quickstart (Phase 2)

```python
from indexing.vector_store import VectorStore
from indexing.sparse_index import SparseIndex
from indexing.embeddings import get_embedder
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import rerank

retriever = HybridRetriever(
    VectorStore(persist_dir="chroma_db"),
    SparseIndex.load("sparse_index.pkl"),
    get_embedder(),
)

fused = retriever.retrieve("What's the DEF concentration spec?")   # RRF top-10
top5 = rerank("What's the DEF concentration spec?", fused)          # Claude rerank, needs ANTHROPIC_API_KEY

# Compare against dense-only (the Phase 5 dashboard toggle's second path):
dense_only = retriever.retrieve_dense_only("What's the DEF concentration spec?")
```

`rerank()` needs `ANTHROPIC_API_KEY`; `retrieve()`/`retrieve_dense_only()` don't — only the second-pass relevance scoring calls out to Claude.

## Quickstart (Phase 3)

```python
from generation.generate import answer_question
from generation.verify_citations import verify_citations
from generation.confidence import score_confidence

result = answer_question(question, top5)  # top5 from rerank(), above

if result.insufficient_context:
    print(result.answer_text)  # structured "couldn't find this" response
else:
    verdicts = verify_citations(result.citations, result.used_chunks)
    confidence = score_confidence(question, result.answer_text, top5, verdicts)
    print(result.answer_text)
    print(f"composite confidence: {confidence.composite:.2f}")
```

All three calls need `ANTHROPIC_API_KEY` (generation, citation verification, and the completeness axis of confidence scoring all call Claude). `answer_question()` never calls the model at all when the reranked top chunk scores below the confidence floor — the "I couldn't find this" response is returned with zero LLM calls.

## Quickstart (Phase 4)

```bash
# Builds a fully separate index per strategy (fixed/structure/semantic),
# runs all 67 golden questions through the full pipeline against each one,
# and writes a markdown comparison report.
python -m eval.run_eval --out eval_report.md
```

`eval/golden_dataset.jsonl` has 67 hand-written Q&A pairs against the sample corpus, across four categories:

| Category | Count | Tests |
|---|---|---|
| `direct` | 41 | single-document lookups |
| `multi_hop` | 12 | answer spans exactly two source documents |
| `unanswerable` | 8 | genuinely not covered by the corpus — `expect_insufficient: true` |
| `ambiguous` | 6 | deliberately underspecified questions with multiple valid readings |

Every case carries `source_docs` (which corpus file(s) the answer should come from, for the `retrieval_relevance` metric) and `expect_insufficient` (whether `answer_question()` should correctly refuse to answer). The run needs `ANTHROPIC_API_KEY` (rerank, generation, citation verification, and the correctness judge all call Claude) and a working local embedding model.

## Design decisions worth asking about in an interview

- **Local embeddings over a hosted API** — `indexing/embeddings.py` swaps
  OpenAI's `text-embedding-3-small` for a local `sentence-transformers`
  model, so ingestion and retrieval need zero API key and zero per-call
  cost. See `ARCHITECTURE.md`'s stack table for the full rationale and the
  documented swap point if a hosted provider is ever needed.
- **Three switchable chunking strategies, not one** — `ingestion/chunking.py`
  implements fixed-size, structure-aware (markdown headers), and semantic
  (embedding-breakpoint) chunking behind one interface, each tagging its
  output so Phase 4's eval harness can run a head-to-head comparison instead
  of asserting one is better.
- **Dedup before index, not after** — `ingestion/dedup.py` runs a
  cosine-similarity check against every chunk already accepted, so the same
  content appearing in two source docs doesn't silently eat two retrieval
  slots.
- **Vector and sparse indexes built from the same chunk list in one pass** —
  `scripts/seed.py` embeds and BM25-indexes the identical post-dedup chunk
  set in the same run, specifically so the two indexes can't drift apart.
- **Fusion by rank, not raw score** — `retrieval/hybrid.py`'s Reciprocal
  Rank Fusion combines dense cosine similarity and BM25 score by each
  candidate's RANK position in its own list, not the raw score — the two
  scores live on incomparable scales (bounded [0,1] vs. unbounded,
  corpus-dependent), so fusing by rank avoids having to calibrate them
  against each other.
- **One batched rerank call, not N** — `retrieval/rerank.py` scores all 20
  candidates in a single Claude call (a JSON array in, a JSON array of
  scores out) instead of one call per candidate — ~20x fewer API calls for
  one longer prompt, the cost-aware tradeoff over the naive design.
- **A real BM25 gotcha, documented where it bites** — `rank_bm25`'s IDF
  formula has no +1 smoothing, so a term appearing in more than half a
  (especially small) corpus gets a *negative* idf, which can silently drop
  an otherwise-relevant chunk below the `score > 0` floor. Documented in
  `indexing/sparse_index.py` and specifically worked around in the Phase 2
  integration test, rather than quietly working past it.
- **The confidence gate costs zero LLM calls when it fires** —
  `generation/generate.py`'s `answer_question()` checks the top reranked
  chunk's score *before* calling Claude at all; on a genuinely off-corpus
  question, the "I couldn't find this" response is returned with no
  generation call, no hallucination risk, and no API spend.
- **Citation verification is a second, independent LLM pass** — a citation
  marker only proves the model *intended* to cite something; `generate.py`
  doesn't know if the cited passage actually backs up the sentence it's
  attached to. `verify_citations.py` re-checks every claim-citation pair
  against the literal source text in a separate call, and
  `confidence.py`'s citation-coverage axis is weighted highest of the three
  (0.4) specifically because an unsupported citation — a confident-sounding
  hallucination with a fake source — is the most dangerous failure mode a
  RAG system can produce.
- **Faithfulness and citation accuracy are two different metrics on
  purpose** — `eval/metrics.py` keeps "did every claim get cited at all"
  (`faithfulness`, pure text analysis, zero LLM calls) separate from "are
  the citations that exist actually correct" (`citation_accuracy`, from
  the independent verification pass). A chunking strategy could move one
  without moving the other, and collapsing them into one number would
  hide that.
- **A real design gap found and fixed mid-build**: Phase 1's `seed.py`
  originally documented a single shared, strategy-tagged Chroma collection
  for the Phase 4 bake-off to filter by — but `SparseIndex` has no concept
  of tagging, so a second `seed()` run would have silently overwritten the
  first strategy's BM25 index. Fixed by having each strategy build a fully
  separate index (its own Chroma dir + BM25 file) instead of retrofitting
  metadata filtering into already-tested code — simpler, and avoids the
  cross-strategy contamination risk entirely.
