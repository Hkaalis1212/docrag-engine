"""
retrieval/hybrid.py
--------------------
Dense (Chroma) + sparse (BM25) retrieval, combined with Reciprocal Rank
Fusion (RRF).

RRF scores each candidate by its RANK position in each ranked list, not the
raw similarity/BM25 score — those live on different, incomparable scales
(cosine similarity is bounded [0,1]; BM25 is an unbounded, corpus-dependent
magnitude). Fusing by rank means a chunk ranked #1 in both lists always
outranks one ranked #1 in only one list, with no need to calibrate the two
scales against each other first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore

DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3
# Standard RRF damping constant (from the original Cormack et al. paper) —
# de-emphasizes the gap between rank 1 and rank 2 so one list's top pick
# doesn't automatically steamroll a candidate both lists rank moderately.
DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 10


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None  # set by retrieval.rerank.rerank(), if run

    @property
    def in_both(self) -> bool:
        return self.dense_rank is not None and self.sparse_rank is not None


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """
    dense_hits: VectorStore.query() output — dicts with chunk_id/text/metadata/score,
                already ranked best-first.
    sparse_hits: SparseIndex.query() output — SparseHit(chunk_id, score, text, metadata),
                 already ranked best-first.
    """
    by_id: dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        by_id[hit["chunk_id"]] = RetrievedChunk(
            chunk_id=hit["chunk_id"],
            text=hit["text"],
            metadata=hit["metadata"],
            dense_rank=rank,
            dense_score=hit["score"],
        )

    for rank, hit in enumerate(sparse_hits, start=1):
        existing = by_id.get(hit.chunk_id)
        if existing is not None:
            existing.sparse_rank = rank
            existing.sparse_score = hit.score
        else:
            by_id[hit.chunk_id] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                metadata=hit.metadata,
                sparse_rank=rank,
                sparse_score=hit.score,
            )

    for chunk in by_id.values():
        score = 0.0
        if chunk.dense_rank is not None:
            score += dense_weight * (1.0 / (rrf_k + chunk.dense_rank))
        if chunk.sparse_rank is not None:
            score += sparse_weight * (1.0 / (rrf_k + chunk.sparse_rank))
        chunk.fused_score = score

    return sorted(by_id.values(), key=lambda c: c.fused_score, reverse=True)


class HybridRetriever:
    """
    Ties the vector store, sparse index, and embedder together into one
    `retrieve(question)` call. Kept thin on purpose — `reciprocal_rank_fusion`
    above is the part with actual logic, and is tested independently of any
    index/embedder so the fusion math never needs a model or a populated
    store to verify.
    """

    def __init__(self, vector_store: VectorStore, sparse_index: SparseIndex, embedder):
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        dense_k: int = DEFAULT_TOP_K,
        sparse_k: int = DEFAULT_TOP_K,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_one(question)
        dense_hits = self.vector_store.query(query_embedding, top_k=dense_k)
        sparse_hits = self.sparse_index.query(question, top_k=sparse_k)
        return reciprocal_rank_fusion(
            dense_hits, sparse_hits, dense_weight, sparse_weight, rrf_k
        )

    def retrieve_dense_only(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """Dense-only retrieval path — exists specifically so the Phase 5
        dashboard's hybrid-vs-dense-only comparison toggle has a real second
        code path to compare against, not a simulated one."""
        query_embedding = self.embedder.embed_one(question)
        dense_hits = self.vector_store.query(query_embedding, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=h["chunk_id"],
                text=h["text"],
                metadata=h["metadata"],
                dense_rank=rank,
                dense_score=h["score"],
                fused_score=h["score"],
            )
            for rank, h in enumerate(dense_hits, start=1)
        ]
