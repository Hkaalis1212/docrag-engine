from indexing.vector_store import VectorStore


def _store(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        documents=["about topic A", "about topic B", "also mostly topic A"],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    return store


def test_add_and_count(tmp_path):
    store = _store(tmp_path)
    assert store.count() == 3


def test_query_returns_nearest_neighbors_first(tmp_path):
    store = _store(tmp_path)
    results = store.query([1.0, 0.0], top_k=3)

    assert results[0]["chunk_id"] == "a"
    assert results[0]["score"] > results[1]["score"]
    assert {r["chunk_id"] for r in results} == {"a", "b", "c"}


def test_query_respects_top_k(tmp_path):
    store = _store(tmp_path)
    results = store.query([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_query_on_empty_store_returns_empty(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    assert store.query([1.0, 0.0], top_k=5) == []


def test_reset_clears_collection(tmp_path):
    store = _store(tmp_path)
    assert store.count() == 3
    store.reset()
    assert store.count() == 0
