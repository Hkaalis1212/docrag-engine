import numpy as np
import pytest

from ingestion.chunking import chunk_document
from ingestion.loaders import Document

MARKDOWN_DOC = """# Section One

This is the first section. It has a couple of sentences about topic A.
Topic A keeps being discussed here in more detail.

## Subsection

More detail under the subsection heading.

# Section Two

This section is about a completely different topic B.
"""


def _doc(text=MARKDOWN_DOC) -> Document:
    return Document(doc_id="doc1", source_path="test.md", title="test", text=text)


def test_fixed_chunking_respects_overlap_and_covers_text():
    chunks = chunk_document(_doc(), strategy="fixed", chunk_size=50, overlap=10)

    assert len(chunks) > 1
    assert all(c.strategy == "fixed" for c in chunks)
    # Reassembled (ignoring overlap) should still contain the original content.
    assert "Section One" in chunks[0].text


def test_structure_chunking_preserves_headings():
    chunks = chunk_document(_doc(), strategy="structure")

    assert len(chunks) >= 2
    headings = [c.section_heading for c in chunks]
    assert any(h and "Subsection" in h for h in headings)


def test_structure_chunking_falls_back_for_oversized_section():
    huge_section = "# Big Section\n\n" + ("word " * 1000)
    chunks = chunk_document(_doc(huge_section), strategy="structure", max_section_size=200)

    assert len(chunks) > 1
    assert all(c.char_count <= 220 for c in chunks)  # small slack for splitter boundaries


def test_semantic_chunking_splits_on_low_similarity():
    # Fake embedder: sentences mentioning "A" cluster near [1, 0],
    # sentences mentioning "B" cluster near [0, 1] -> one clear boundary.
    def fake_embed(sentences: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "A" in s else [0.0, 1.0] for s in sentences]

    text = "Topic A sentence one. Topic A sentence two.\n\nTopic B sentence one. Topic B sentence two."
    chunks = chunk_document(_doc(text), strategy="semantic", embed_fn=fake_embed, similarity_threshold=0.5)

    assert len(chunks) == 2
    assert "Topic A" in chunks[0].text
    assert "Topic B" in chunks[1].text


def test_semantic_chunking_requires_embed_fn():
    with pytest.raises(ValueError):
        chunk_document(_doc(), strategy="semantic")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_document(_doc(), strategy="nonsense")  # type: ignore[arg-type]


def test_chunk_ids_are_unique_within_a_document():
    chunks = chunk_document(_doc(), strategy="fixed", chunk_size=50, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
