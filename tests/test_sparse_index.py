from indexing.sparse_index import SparseIndex


def _build_index() -> SparseIndex:
    index = SparseIndex()
    index.build(
        chunk_ids=["a", "b", "c"],
        texts=[
            "DEF concentration spec is 32.5 percent urea",
            "turbo boost psi reading was low on this truck",
            "completely unrelated passage about tire pressure",
        ],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    return index


def test_query_returns_relevant_hit_first():
    index = _build_index()
    hits = index.query("DEF urea concentration spec", top_k=3)

    assert hits[0].chunk_id == "a"
    assert hits[0].text.startswith("DEF concentration")
    assert hits[0].metadata == {"doc_id": "d1"}


def test_query_excludes_zero_score_hits():
    index = _build_index()
    hits = index.query("zzz_nonexistent_token_zzz", top_k=3)
    assert hits == []


def test_query_respects_top_k():
    index = _build_index()
    hits = index.query("truck tire pressure boost", top_k=1)
    assert len(hits) == 1


def test_get_text_round_trips():
    index = _build_index()
    assert index.get_text("b") == "turbo boost psi reading was low on this truck"
    assert index.get_text("missing") is None


def test_save_and_load_round_trip(tmp_path):
    index = _build_index()
    path = tmp_path / "sparse.pkl"
    index.save(path)

    loaded = SparseIndex.load(path)
    hits = loaded.query("DEF urea concentration spec", top_k=1)
    assert hits[0].chunk_id == "a"


def test_empty_index_returns_no_hits():
    index = SparseIndex()
    assert index.query("anything", top_k=5) == []
