# docrag-engine — Architecture

A production-grade hybrid-retrieval RAG system: ingest internal documentation,
index it with both dense (vector) and sparse (BM25) search, fuse and rerank
results, generate grounded answers with verified citations, and measure the
whole pipeline against a golden eval set.

This document is the build plan. It exists so design decisions are made once,
on paper, before code — and so each decision has a one-line "why," which is
the part that actually gets asked about in interviews.

## Why this project

RAG is the most commonly requested skill in AI engineering job postings, and
the most commonly faked — a single-PDF LangChain quickstart with no eval, no
hybrid retrieval, and no citation verification. This system is scoped to
demonstrate the production concerns that separate the two: chunking strategy
tradeoffs, retrieval fusion, a reranking pass, citation verification against
the source text, and a 50+ question eval harness that produces real numbers
instead of a vibe.

## Stack and the deviations from the reference guide

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem standard for this kind of pipeline |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`), local | **Deviation**: the reference guide specs OpenAI `text-embedding-3-small`. A local, free embedding model means a reviewer can clone the repo and run the full ingestion + retrieval pipeline with zero API key and zero cost — only generation needs a key. Quality is slightly behind OpenAI's model; documented as a tunable swap point (`embeddings/provider.py`), not a hard dependency. |
| Vector store | ChromaDB (embedded, file-based) | No server to stand up; matches "clone and run" goal |
| Sparse search | `rank_bm25` | Exact keyword/term matching — catches config keys, function names, error codes that dense search blurs over |
| Fusion | Reciprocal Rank Fusion (RRF), configurable weighting | Standard, explainable, no training required |
| Reranker | Claude Haiku as cross-encoder substitute (relevance scoring prompt) | **Deviation**: guide suggests a dedicated cross-encoder model or LLM-as-judge; using Claude directly avoids a second model dependency and keeps the stack to one LLM provider |
| Generation | Claude Sonnet (Anthropic API) | **Deviation**: guide specs GPT-4o. Matches the stack already used in ThrottleGuard/Mya — one provider, one set of prompting idioms, one bill |
| Chunking | Custom fixed-size, `langchain-text-splitters` (structure-aware), custom semantic (embedding-breakpoint) | Three real, switchable strategies — the comparison is the point, not any one implementation |
| API | FastAPI | Async-native, OpenAPI docs for free |
| Frontend | Streamlit | Fastest path to a usable query/comparison dashboard; consistent with other work |
| Containerization | Docker + docker-compose | One command to reproduce the whole stack |

## Demo corpus

Sample corpus is ThrottleGuard's own engineering documentation (`CLAUDE.md`
plus module docstrings) — real, owned, technical documentation with the kind
of structure (numbered rules, threshold tables, file trees, env var tables)
that makes chunking-strategy choice actually matter. Swappable: point
`ingest` at any directory of markdown/text/HTML/PDF and it reindexes.

## Pipeline

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  Ingestion  │ ──▶ │   Chunking    │ ──▶ │  Embedding    │
│ (md/txt/    │     │ (3 strategies,│     │  + indexing   │
│  html/pdf)  │     │  switchable)  │     │ (Chroma + BM25│
└─────────────┘     └───────────────┘     │  in lockstep) │
                                           └──────┬────────┘
                                                  │
  ┌───────────────────────────────────────────────┘
  ▼
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│   Question  │ ──▶ │ Hybrid        │ ──▶ │   RRF fusion │
│             │     │ retrieval     │     │  + rerank    │
│             │     │ (dense + BM25)│     │  (top 20→5)  │
└─────────────┘     └───────────────┘     └──────┬────────┘
                                                  │
  ┌───────────────────────────────────────────────┘
  ▼
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  Grounded   │ ──▶ │   Citation    │ ──▶ │  Confidence   │
│ generation  │     │ verification  │     │   scoring     │
│ (cited [n]) │     │ (claim↔chunk) │     │  (3 axes)     │
└─────────────┘     └───────────────┘     └──────────────┘
```

## Phase breakdown

### Phase 1 — Ingestion and chunking
- Multi-format loader (md/txt/html/pdf) → clean plaintext + metadata (source
  file, section heading, char offsets)
- Raw + processed documents both stored, so re-chunking never requires
  re-uploading
- Three chunking strategies, each tagged on its output chunks so eval can
  compare them later:
  1. **Fixed-size + overlap** — baseline, strategy-agnostic
  2. **Structure-aware** — splits on markdown headers first, falls back to
     recursive character splitting within oversized sections
  3. **Semantic** — splits on embedding-similarity breakpoints between
     adjacent sentences (topic-boundary detection)
- Embed every chunk, write to Chroma; build BM25 over the identical chunk
  set in the same pass so the two indexes can never drift apart
- Near-duplicate detection (cosine > 0.95) before insert — skip and log,
  don't silently waste retrieval slots on repeated content

### Phase 2 — Hybrid retrieval
- Dense: top-k (k=10) cosine similarity against Chroma
- Sparse: top-k (k=10) BM25 score against the same chunk corpus
- Fusion: Reciprocal Rank Fusion, default 0.7 dense / 0.3 sparse, both
  configurable per query
- Rerank: top-20 fused candidates scored for relevance by Claude Haiku
  against the literal question text, keep top 5

### Phase 3 — Generation and citation
- System prompt: answer only from provided context, cite every claim
  `[n]`, explicitly say when context is insufficient
- Citation verification: each `[n]` claim pair re-checked by a second
  Claude call — does chunk `n` actually support the sentence it's attached
  to? Unsupported citations are flagged in the response, not silently kept
- Confidence score, three axes: retrieval confidence (top-chunk
  relevance), citation coverage (% of claims with a verified citation),
  completeness (did the answer address every part of the question)
- Below-threshold retrieval confidence → structured "couldn't find this"
  response naming what was searched, not a hallucinated guess

### Phase 4 — Evaluation
- 50+ hand-written golden Q&A pairs against the demo corpus: direct
  lookups, multi-hop (spans two source docs), unanswerable-from-corpus,
  and deliberately ambiguous questions
- Per-case metrics: answer correctness (LLM-as-judge vs. golden answer),
  faithfulness (claims grounded in retrieved context), retrieval relevance,
  citation accuracy
- Chunking-strategy bake-off: full eval suite run three times (once per
  strategy), results tabulated — this produces the actual numbers for the
  case study, not a guess

### Phase 5 — API and dashboard
- `POST /v1/ask` — question in, answer + citations + confidence + source
  metadata out
- `GET /v1/documents` — list indexed documents
- `POST /v1/ingest` — add documents, triggers re-chunk/re-embed
- OpenAPI docs free from FastAPI
- Streamlit dashboard: ask a question, see the answer with clickable
  citations, the ranked retrieved chunks, the three-axis confidence
  breakdown, and a hybrid-vs-dense-only side-by-side toggle
- `docker-compose up` brings up API + frontend + a seed script that
  indexes the sample corpus automatically

### Phase 6 — Portfolio polish
- Short (<4 min) demo walkthrough: ingest → ask → citation-verification
  catching a hallucination → hybrid vs. dense-only comparison
- Case study written from the Phase 4 numbers: "hybrid retrieval scores
  X% faithfulness / Y% citation accuracy vs. Z% dense-only, on a 50-question
  eval suite over real technical documentation" — lead with the number,
  then the mechanism, per the usual format

## Repo layout

```
docrag-engine/
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── corpus/
│   └── throttleguard-docs/      ← sample corpus (swappable)
├── ingestion/
│   ├── loaders.py               ← md/txt/html/pdf → plaintext + metadata
│   ├── chunking.py               ← 3 strategies, switchable
│   └── dedup.py
├── indexing/
│   ├── embeddings.py             ← local embedding provider (swap point)
│   ├── vector_store.py           ← Chroma wrapper
│   └── sparse_index.py           ← BM25 wrapper
├── retrieval/
│   ├── hybrid.py                 ← dense + sparse + RRF fusion
│   └── rerank.py
├── generation/
│   ├── prompts.py
│   ├── generate.py               ← grounded answer + citations
│   ├── verify_citations.py
│   └── confidence.py
├── eval/
│   ├── golden_dataset.jsonl       ← 50+ hand-written Q&A pairs
│   ├── metrics.py
│   └── run_eval.py                ← produces the strategy bake-off report
├── api/
│   └── main.py                    ← FastAPI app
├── dashboard/
│   └── app.py                     ← Streamlit UI
├── scripts/
│   └── seed.py                    ← indexes corpus/ on first run
└── tests/
```

## Status

Phases 1-5 are built: ingestion/chunking/indexing, hybrid retrieval,
grounded generation + citation verification + confidence scoring, the
golden-set evaluation framework + chunking-strategy bake-off, and the
FastAPI service + Streamlit dashboard. See the status table in
`README.md` for the full phase checklist.
