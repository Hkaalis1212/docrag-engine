from ingestion.dedup import DuplicateFilter, cosine_similarity


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_duplicate_filter_flags_near_identical_embeddings():
    f = DuplicateFilter(threshold=0.95)

    assert f.is_duplicate([1.0, 0.0, 0.0]) is False  # first one is always accepted
    assert f.is_duplicate([0.999, 0.001, 0.0]) is True  # near-identical
    assert f.is_duplicate([0.0, 1.0, 0.0]) is False  # orthogonal, not a dupe
    assert f.duplicate_count == 1


def test_duplicate_filter_respects_custom_threshold():
    loose = DuplicateFilter(threshold=0.5)
    loose.is_duplicate([1.0, 0.0])
    # 0.6 cosine similarity clears a 0.5 threshold -> counts as a duplicate
    assert loose.is_duplicate([0.6, 0.8]) is True
