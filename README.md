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
| 2 — Hybrid retrieval (dense + BM25 + RRF + rerank) | ⬜ not started |
| 3 — Grounded generation + citation verification | ⬜ not started |
| 4 — Evaluation framework (golden set + metrics) | ⬜ not started |
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

# Try the other two strategies against the same corpus to compare later:
python -m scripts.seed --corpus corpus/throttleguard-docs --strategy fixed
python -m scripts.seed --corpus corpus/throttleguard-docs --strategy semantic
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
