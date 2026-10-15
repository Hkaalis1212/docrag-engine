"""
api/main.py
------------
FastAPI service exposing the RAG pipeline: ask a question, list indexed
documents, ingest a new one, or compare hybrid vs. dense-only retrieval.

Run:
    uvicorn api.main:app --reload

Configuration (env vars):
    DOCRAG_PERSIST_DIR        Chroma persist directory (default: chroma_db)
    DOCRAG_SPARSE_INDEX_PATH  BM25 pickle path (default: sparse_index.pkl)
    DOCRAG_STRATEGY           Chunking strategy used for new ingests (default: structure)
    DOCRAG_INGEST_DIR         Where raw ingested document text is written (default: ingested_docs)

The pipeline (vector store + sparse index + embedder + retriever) is built
lazily on first request via get_pipeline(), not in a startup/lifespan hook -
that keeps it a plain, overridable FastAPI dependency, so tests can swap in
a fully fake pipeline with `app.dependency_overrides[get_pipeline] = ...`
instead of needing a real embedding model or API key just to exercise the
route logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from generation.confidence import ConfidenceScore, score_confidence
from generation.generate import DEFAULT_CONFIDENCE_FLOOR, GeneratedAnswer, answer_question
from generation.verify_citations import verify_citations
from indexing.embeddings import get_embedder
from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore
from ingestion.chunking import chunk_document
from ingestion.dedup import DuplicateFilter
from ingestion.loaders import load_file
from retrieval.hybrid import HybridRetriever, RetrievedChunk
from retrieval.rerank import rerank

DOCRAG_PERSIST_DIR = os.getenv("DOCRAG_PERSIST_DIR", "chroma_db")
DOCRAG_SPARSE_INDEX_PATH = os.getenv("DOCRAG_SPARSE_INDEX_PATH", "sparse_index.pkl")
DOCRAG_STRATEGY = os.getenv("DOCRAG_STRATEGY", "structure")
DOCRAG_INGEST_DIR = os.getenv("DOCRAG_INGEST_DIR", "ingested_docs")


@dataclass
class Pipeline:
    vector_store: VectorStore
    sparse_index: SparseIndex
    embedder: object
    retriever: HybridRetriever


def build_pipeline(
    persist_dir: str = DOCRAG_PERSIST_DIR, sparse_index_path: str = DOCRAG_SPARSE_INDEX_PATH
) -> Pipeline:
    vector_store = VectorStore(persist_dir=persist_dir)
    sparse_path = Path(sparse_index_path)
    sparse_index = SparseIndex.load(sparse_path) if sparse_path.exists() else SparseIndex()
    embedder = get_embedder()
    retriever = HybridRetriever(vector_store, sparse_index, embedder)
    return Pipeline(vector_store, sparse_index, embedder, retriever)


_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


app = FastAPI(title="docrag-engine", description="Hybrid-retrieval RAG over internal docs")


# ── Request/response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    dense_k: int = 10
    sparse_k: int = 10
    rerank_top_n: int = 5
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    source_file: str
    claim_text: str
    verdict: Optional[str] = None


class ConfidenceOut(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    completeness: float
    composite: float


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    fused_score: float
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    insufficient_context: bool
    citations: list[CitationOut]
    confidence: Optional[ConfidenceOut]
    used_chunks: list[ChunkOut]


class DocumentSummary(BaseModel):
    source_file: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int


class IngestRequest(BaseModel):
    filename: str
    content: str


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks_in_index: int


class CompareRetrievalRequest(BaseModel):
    question: str
    top_k: int = 5


class CompareRetrievalResponse(BaseModel):
    question: str
    hybrid: list[ChunkOut]
    dense_only: list[ChunkOut]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_chunk_out(chunk: RetrievedChunk) -> ChunkOut:
    return ChunkOut(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source_file=chunk.metadata.get("source_file", "unknown"),
        fused_score=chunk.fused_score,
        rerank_score=chunk.rerank_score,
    )


def _source_file_for(chunk_id: str, used_chunks: list[RetrievedChunk]) -> str:
    for ch in used_chunks:
        if ch.chunk_id == chunk_id:
            return ch.metadata.get("source_file", "unknown")
    return "unknown"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health(pipeline: Pipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=pipeline.vector_store.count())


@app.get("/v1/documents", response_model=DocumentsResponse)
def list_documents(pipeline: Pipeline = Depends(get_pipeline)) -> DocumentsResponse:
    all_chunks = pipeline.vector_store.get_all()
    counts: dict[str, int] = {}
    for c in all_chunks:
        source = c["metadata"].get("source_file", "unknown")
        counts[source] = counts.get(source, 0) + 1
    documents = [DocumentSummary(source_file=k, chunk_count=v) for k, v in sorted(counts.items())]
    return DocumentsResponse(documents=documents, total_chunks=len(all_chunks))


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Pipeline = Depends(get_pipeline)) -> AskResponse:
    if pipeline.vector_store.count() == 0:
        raise HTTPException(status_code=503, detail="Index is empty - ingest documents first.")

    fused = pipeline.retriever.retrieve(req.question, dense_k=req.dense_k, sparse_k=req.sparse_k)
    reranked = rerank(req.question, fused, top_n=req.rerank_top_n)
    answer: GeneratedAnswer = answer_question(
        req.question, reranked, confidence_floor=req.confidence_floor
    )

    if answer.insufficient_context:
        return AskResponse(
            question=req.question,
            answer=answer.answer_text,
            insufficient_context=True,
            citations=[],
            confidence=None,
            used_chunks=[_to_chunk_out(c) for c in answer.used_chunks],
        )

    verdicts = verify_citations(answer.citations, answer.used_chunks)
    # Citation objects in `verdicts` are the SAME objects as in
    # `answer.citations` (verify_citations() doesn't copy them), so identity
    # is a safe join key here - and safer than (chunk_id, claim_text), which
    # could collide if a claim cited the same chunk twice.
    verdict_by_citation = {id(v.citation): v.verdict for v in verdicts}
    confidence: ConfidenceScore = score_confidence(req.question, answer.answer_text, reranked, verdicts)

    citations_out = [
        CitationOut(
            marker=c.marker,
            chunk_id=c.chunk_id,
            source_file=_source_file_for(c.chunk_id, answer.used_chunks),
            claim_text=c.claim_text,
            verdict=verdict_by_citation.get(id(c)),
        )
        for c in answer.citations
    ]

    return AskResponse(
        question=req.question,
        answer=answer.answer_text,
        insufficient_context=False,
        citations=citations_out,
        confidence=ConfidenceOut(
            retrieval_confidence=confidence.retrieval_confidence,
            citation_coverage=confidence.citation_coverage,
            completeness=confidence.completeness,
            composite=confidence.composite,
        ),
        used_chunks=[_to_chunk_out(c) for c in answer.used_chunks],
    )


@app.post("/v1/compare_retrieval", response_model=CompareRetrievalResponse)
def compare_retrieval(
    req: CompareRetrievalRequest, pipeline: Pipeline = Depends(get_pipeline)
) -> CompareRetrievalResponse:
    """No LLM call at all - pure retrieval comparison, same reasoning as
    retrieve()/retrieve_dense_only() themselves (see retrieval/hybrid.py).
    Powers the dashboard's hybrid-vs-dense-only toggle at zero API cost."""
    hybrid = pipeline.retriever.retrieve(req.question, dense_k=req.top_k, sparse_k=req.top_k)[: req.top_k]
    dense_only = pipeline.retriever.retrieve_dense_only(req.question, top_k=req.top_k)
    return CompareRetrievalResponse(
        question=req.question,
        hybrid=[_to_chunk_out(c) for c in hybrid],
        dense_only=[_to_chunk_out(c) for c in dense_only],
    )


@app.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, pipeline: Pipeline = Depends(get_pipeline)) -> IngestResponse:
    """
    Adds one new document to the index.

    The vector side is a true incremental add. The BM25 side is NOT -
    rank_bm25's IDF is a statistic over the whole corpus, so there is no
    correct way to "add one document" to an existing BM25Okapi index; the
    only correct move is rebuilding it from the full chunk set (existing +
    new), which is what this does. Fine at this corpus's scale (a rebuild
    over a few hundred chunks is milliseconds); a much larger, ingest-heavy
    corpus would need a different sparse-index strategy entirely.
    """
    ingest_dir = Path(DOCRAG_INGEST_DIR)
    ingest_dir.mkdir(exist_ok=True)
    dest = ingest_dir / req.filename
    dest.write_text(req.content, encoding="utf-8")

    try:
        doc = load_file(dest)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    kwargs = {"embed_fn": pipeline.embedder.embed} if DOCRAG_STRATEGY == "semantic" else {}
    chunks = chunk_document(doc, strategy=DOCRAG_STRATEGY, **kwargs)
    if not chunks:
        return IngestResponse(
            filename=req.filename, chunks_added=0,
            total_chunks_in_index=pipeline.vector_store.count(),
        )

    embeddings = pipeline.embedder.embed([c.text for c in chunks])

    dup_filter = DuplicateFilter()
    kept_chunks, kept_embeddings = [], []
    for chunk, emb in zip(chunks, embeddings):
        if dup_filter.is_duplicate(emb):
            continue
        kept_chunks.append(chunk)
        kept_embeddings.append(emb)

    metadatas = [
        {
            "doc_id": c.doc_id,
            "source_file": req.filename,
            "strategy": c.strategy,
            "chunk_index": c.chunk_index,
            "char_count": c.char_count,
            "section_heading": c.section_heading or "",
        }
        for c in kept_chunks
    ]
    pipeline.vector_store.add(
        ids=[c.chunk_id for c in kept_chunks],
        embeddings=kept_embeddings,
        documents=[c.text for c in kept_chunks],
        metadatas=metadatas,
    )

    all_chunks = pipeline.vector_store.get_all()
    pipeline.sparse_index.build(
        chunk_ids=[c["chunk_id"] for c in all_chunks],
        texts=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )
    pipeline.sparse_index.save(DOCRAG_SPARSE_INDEX_PATH)

    return IngestResponse(
        filename=req.filename,
        chunks_added=len(kept_chunks),
        total_chunks_in_index=pipeline.vector_store.count(),
    )
