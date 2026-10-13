"""
chunking.py
-----------
Three switchable chunking strategies over a loaded Document:

  1. fixed       - fixed-size character windows with overlap (baseline)
  2. structure   - splits on markdown headers first, falls back to recursive
                   character splitting within any section that's still too big
  3. semantic    - splits on embedding-similarity breakpoints between
                   adjacent sentences (topic-boundary detection)

Every chunk is tagged with the strategy that produced it so the eval
framework (Phase 4) can run the same question set across all three and
compare.

Usage:
    from ingestion.chunking import chunk_document

    chunks = chunk_document(doc, strategy="structure")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ingestion.loaders import Document

Strategy = Literal["fixed", "structure", "semantic"]

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: Strategy
    chunk_index: int
    char_count: int
    section_heading: str | None = None
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(doc_id: str, strategy: Strategy, index: int) -> str:
    return f"{doc_id}:{strategy}:{index}"


# ── Strategy 1: fixed-size + overlap ───────────────────────────────────────

def _chunk_fixed(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(doc.text)
    return [
        Chunk(
            chunk_id=_make_chunk_id(doc.doc_id, "fixed", i),
            doc_id=doc.doc_id,
            text=piece,
            strategy="fixed",
            chunk_index=i,
            char_count=len(piece),
        )
        for i, piece in enumerate(pieces)
    ]


# ── Strategy 2: structure-aware (markdown headers) ─────────────────────────

def _chunk_structure(
    doc: Document,
    max_section_size: int = DEFAULT_CHUNK_SIZE * 2,
) -> list[Chunk]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS, strip_headers=False
    )
    sections = header_splitter.split_text(doc.text)

    # Any section still too large (e.g. a long code block under one heading)
    # falls back to recursive character splitting WITHIN that section, so a
    # heading's identity is preserved on every resulting sub-chunk.
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_section_size, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    chunks: list[Chunk] = []
    index = 0
    for section in sections:
        heading = (
            section.metadata.get("h3")
            or section.metadata.get("h2")
            or section.metadata.get("h1")
        )
        pieces = (
            [section.page_content]
            if len(section.page_content) <= max_section_size
            else fallback_splitter.split_text(section.page_content)
        )
        for piece in pieces:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc.doc_id, "structure", index),
                    doc_id=doc.doc_id,
                    text=piece,
                    strategy="structure",
                    chunk_index=index,
                    char_count=len(piece),
                    section_heading=heading,
                )
            )
            index += 1
    return chunks


# ── Strategy 3: semantic (embedding-similarity breakpoints) ───────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(`])")


def _split_sentences(text: str) -> list[str]:
    # Paragraph boundaries are kept as hard breaks before sentence splitting,
    # so a breakpoint never merges two different sections of a doc.
    sentences = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip())
    return sentences


def _chunk_semantic(
    doc: Document,
    embed_fn: Callable[[list[str]], list[list[float]]],
    similarity_threshold: float = 0.6,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE * 2,
) -> list[Chunk]:
    """
    Groups adjacent sentences into a chunk until the cosine similarity between
    consecutive sentence embeddings drops below `similarity_threshold`
    (a topic boundary), or the chunk hits `max_chunk_size`.

    `embed_fn` is injected rather than imported directly so this module has
    no hard dependency on the embedding provider (see indexing/embeddings.py
    for the swap point).
    """
    import numpy as np

    sentences = _split_sentences(doc.text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [
            Chunk(
                chunk_id=_make_chunk_id(doc.doc_id, "semantic", 0),
                doc_id=doc.doc_id,
                text=sentences[0],
                strategy="semantic",
                chunk_index=0,
                char_count=len(sentences[0]),
            )
        ]

    embeddings = np.array(embed_fn(sentences))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-9, None)
    # Cosine similarity between each sentence and the next one.
    sims = (normed[:-1] * normed[1:]).sum(axis=1)

    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]
    index = 0

    def flush():
        nonlocal index
        text = " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(doc.doc_id, "semantic", index),
                doc_id=doc.doc_id,
                text=text,
                strategy="semantic",
                chunk_index=index,
                char_count=len(text),
            )
        )
        index += 1

    for i, sim in enumerate(sims):
        candidate_len = sum(len(s) for s in current) + len(sentences[i + 1])
        is_boundary = sim < similarity_threshold or candidate_len > max_chunk_size
        if is_boundary:
            flush()
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    flush()
    return chunks


_STRATEGIES: dict[Strategy, Callable] = {
    "fixed": _chunk_fixed,
    "structure": _chunk_structure,
    "semantic": _chunk_semantic,
}


def chunk_document(
    doc: Document,
    strategy: Strategy = "structure",
    **kwargs,
) -> list[Chunk]:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(_STRATEGIES)}")
    if strategy == "semantic" and "embed_fn" not in kwargs:
        raise ValueError("strategy='semantic' requires an embed_fn kwarg")
    return _STRATEGIES[strategy](doc, **kwargs)
