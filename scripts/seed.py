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

Each run builds a FULLY SEPARATE index — its own Chroma persist directory
and its own BM25 pickle file, passed explicitly via `persist_dir` /
`sparse_index_path`. (An earlier version of this docstring described a
single shared, strategy-tagged collection instead; that design was never
actually implementable — SparseIndex has no concept of filtering by tag, so
a second seed() run would silently overwrite the first strategy's sparse
index rather than coexist with it. Separate indexes per strategy avoids the
problem entirely and is what Phase 4's eval harness, eval/run_eval.py,
actually does — see RUN_DIRS there.)
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


def _chunk_to_metadata(chunk: Chunk, source_file_lookup: dict[str, str]) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "source_file": source_file_lookup.get(chunk.doc_id, "unknown"),
        "strategy": chunk.strategy,
        "chunk_index": chunk.chunk_index,
        "char_count": chunk.char_count,
        "section_heading": chunk.section_heading or "",
    }


def seed(
    corpus_dir: str,
    strategy: str,
    persist_dir: str = "chroma_db",
    sparse_index_path: str = SPARSE_INDEX_PATH,
) -> None:
    t0 = time.time()
    docs = load_directory(corpus_dir)
    print(f"Loaded {len(docs)} source documents from {corpus_dir}")
    # doc_id (an opaque path hash) -> original filename, so downstream
    # metadata stays human-readable - eval/metrics.py's retrieval-relevance
    # check needs to compare against the filenames the golden dataset names,
    # not an opaque hash.
    source_file_lookup = {doc.doc_id: doc.metadata["filename"] for doc in docs}

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

    metadatas = [_chunk_to_metadata(c, source_file_lookup) for c in kept_chunks]

    store = VectorStore(persist_dir=persist_dir)
    store.add(
        ids=[c.chunk_id for c in kept_chunks],
        embeddings=kept_embeddings,
        documents=[c.text for c in kept_chunks],
        metadatas=metadatas,
    )

    sparse = SparseIndex()
    sparse.build(
        chunk_ids=[c.chunk_id for c in kept_chunks],
        texts=[c.text for c in kept_chunks],
        metadatas=metadatas,
    )
    sparse.save(sparse_index_path)

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
    parser.add_argument("--sparse-index-path", default=SPARSE_INDEX_PATH)
    args = parser.parse_args()
    seed(args.corpus, args.strategy, args.persist_dir, args.sparse_index_path)


if __name__ == "__main__":
    main()
