"""
scripts/seed.py
----------------
End-to-end Phase 1 pipeline: load the corpus, chunk it (one strategy per
run), embed every chunk, write to both the vector store and the BM25 index
in lockstep, with near-duplicate filtering in between.

Usage:
    python -m scripts.seed --corpus corpus/throttleguard-docs --strategy structure
    python -m scripts.seed --corpus corpus/throttleguard-docs --strategy fixed
    python -m scripts.seed --corpus corpus/throttleguard-docs --strategy semantic

Each run indexes into the SAME Chroma collection / BM25 index, tagged by
strategy in metadata — Phase 4's eval harness filters by strategy to compare
them head to head.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunking import Chunk, chunk_document
from ingestion.dedup import DuplicateFilter
from ingestion.loaders import load_directory
from indexing.embeddings import get_embedder
from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore

SPARSE_INDEX_PATH = "sparse_index.pkl"


def _chunk_to_metadata(chunk: Chunk) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "strategy": chunk.strategy,
        "chunk_index": chunk.chunk_index,
        "char_count": chunk.char_count,
        "section_heading": chunk.section_heading or "",
    }


def seed(corpus_dir: str, strategy: str, persist_dir: str = "chroma_db") -> None:
    t0 = time.time()
    docs = load_directory(corpus_dir)
    print(f"Loaded {len(docs)} source documents from {corpus_dir}")

    embedder = get_embedder()

    all_chunks: list[Chunk] = []
    for doc in docs:
        kwargs = {"embed_fn": embedder.embed} if strategy == "semantic" else {}
        chunks = chunk_document(doc, strategy=strategy, **kwargs)
        all_chunks.extend(chunks)
        print(f"  {doc.title!r}: {len(chunks)} chunks ({strategy})")

    print(f"Total chunks before dedup: {len(all_chunks)}")

    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts)

    dup_filter = DuplicateFilter()
    kept_chunks, kept_embeddings = [], []
    for chunk, emb in zip(all_chunks, embeddings):
        if dup_filter.is_duplicate(emb):
            continue
        kept_chunks.append(chunk)
        kept_embeddings.append(emb)

    print(
        f"Dropped {dup_filter.duplicate_count} near-duplicate chunks, "
        f"{len(kept_chunks)} remain"
    )

    store = VectorStore(persist_dir=persist_dir)
    store.add(
        ids=[c.chunk_id for c in kept_chunks],
        embeddings=kept_embeddings,
        documents=[c.text for c in kept_chunks],
        metadatas=[_chunk_to_metadata(c) for c in kept_chunks],
    )

    sparse = SparseIndex()
    sparse.build(
        chunk_ids=[c.chunk_id for c in kept_chunks],
        texts=[c.text for c in kept_chunks],
        metadatas=[_chunk_to_metadata(c) for c in kept_chunks],
    )
    sparse.save(SPARSE_INDEX_PATH)

    elapsed = time.time() - t0
    print(
        f"Indexed {len(kept_chunks)} chunks into Chroma "
        f"({store.count()} total in collection) and BM25 in {elapsed:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the hybrid index from a doc corpus.")
    parser.add_argument("--corpus", default="corpus/throttleguard-docs")
    parser.add_argument(
        "--strategy", choices=["fixed", "structure", "semantic"], default="structure"
    )
    parser.add_argument("--persist-dir", default="chroma_db")
    args = parser.parse_args()
    seed(args.corpus, args.strategy, args.persist_dir)


if __name__ == "__main__":
    main()
