from indexing.sparse_index import SparseHit
from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore
from retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _dense_hit(chunk_id, score, text="dense text"):
    return {"chunk_id": chunk_id, "text": text, "metadata": {}, "score": score}


def _sparse_hit(chunk_id, score, text="sparse text"):
    return SparseHit(chunk_id=chunk_id, score=score, text=text, metadata={})


# ── reciprocal_rank_fusion ──────────────────────────────────────────────────

def test_chunk_in_both_lists_outranks_chunk_in_one():
    dense = [_dense_hit("a", 0.9), _dense_hit("b", 0.5)]
    sparse = [_sparse_hit("a", 10.0), _sparse_hit("c", 8.0)]

    fused = reciprocal_rank_fusion(dense, sparse)

    assert fused[0].chunk_id == "a"
    assert fused[0].in_both is True
    assert fused[0].fused_score > fused[1].fused_score
    assert fused[0].fused_score > fused[2].fused_score


def test_fusion_preserves_dense_only_and_sparse_only_hits():
    dense = [_dense_hit("a", 0.9)]
    sparse = [_sparse_hit("b", 5.0)]

    fused = reciprocal_rank_fusion(dense, sparse)
    ids = {c.chunk_id for c in fused}

    assert ids == {"a", "b"}


def test_dense_weight_zero_ignores_dense_rank():
    dense = [_dense_hit("a", 0.99)]  # top dense hit
    sparse = [_sparse_hit("b", 1.0)]  # weak sparse hit, but dense is ignored

    fused = reciprocal_rank_fusion(dense, sparse, dense_weight=0.0, sparse_weight=1.0)

    assert fused[0].chunk_id == "b"
    assert fused[1].fused_score == 0.0  # "a" only has a dense rank, now worth zero


def test_rrf_k_dampens_rank_one_advantage():
    dense = [_dense_hit("a", 1.0), _dense_hit("b", 0.9)]
    sparse: list = []

    low_k = reciprocal_rank_fusion(dense, sparse, rrf_k=1)
    high_k = reciprocal_rank_fusion(dense, sparse, rrf_k=1000)

    # With a tiny k, rank 1 vs rank 2 is a huge relative gap; with a huge k,
    # the two scores converge almost to equality.
    low_k_gap = low_k[0].fused_score - low_k[1].fused_score
    high_k_gap = high_k[0].fused_score - high_k[1].fused_score
    assert low_k_gap > high_k_gap


def test_empty_inputs_return_empty_list():
    assert reciprocal_rank_fusion([], []) == []


# ── HybridRetriever (real indexes, fake embedder) ───────────────────────────

class _FakeEmbedder:
    """Maps specific strings to orthogonal vectors so dense ranking is
    deterministic and checkable without a real model."""

    VECTORS = {
        "topic A question": [1.0, 0.0],
        "about topic A in depth": [1.0, 0.0],
        "about topic B instead": [0.0, 1.0],
    }

    def embed_one(self, text: str) -> list[float]:
        return self.VECTORS.get(text, [0.5, 0.5])


# Three documents, not two: rank_bm25's BM25Okapi computes IDF without +1
# smoothing, so a term appearing in >50% of the corpus gets a NEGATIVE idf —
# in a 2-doc corpus where both docs share a word, that word's contribution
# can push an otherwise-relevant doc's score below zero and out of the
# results entirely. A third, unrelated filler doc keeps "topic" below that
# 50% threshold so this test exercises real hybrid-combination behavior
# instead of that library edge case (which is covered on its own terms in
# tests/test_sparse_index.py).
_DOC_IDS = ["a", "b", "c"]
_DOC_TEXTS = ["about topic A in depth", "about topic B instead", "unrelated filler about neither subject"]
_DOC_EMBEDDINGS = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]


def test_hybrid_retriever_combines_both_indexes(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=_DOC_IDS,
        embeddings=_DOC_EMBEDDINGS,
        documents=_DOC_TEXTS,
        metadatas=[{}, {}, {}],
    )

    sparse = SparseIndex()
    sparse.build(chunk_ids=_DOC_IDS, texts=_DOC_TEXTS, metadatas=[{}, {}, {}])

    retriever = HybridRetriever(store, sparse, _FakeEmbedder())
    results = retriever.retrieve("topic A question")

    assert results[0].chunk_id == "a"
    assert results[0].in_both is True


def test_retrieve_dense_only_skips_sparse_signal(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=_DOC_IDS,
        embeddings=_DOC_EMBEDDINGS,
        documents=_DOC_TEXTS,
        metadatas=[{}, {}, {}],
    )
    sparse = SparseIndex()  # left empty — must not be consulted

    retriever = HybridRetriever(store, sparse, _FakeEmbedder())
    results = retriever.retrieve_dense_only("topic A question")

    assert results[0].chunk_id == "a"
    assert results[0].sparse_rank is None
