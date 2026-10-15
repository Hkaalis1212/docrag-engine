"""
vector_store.py
----------------
Thin wrapper around an embedded ChromaDB collection. File-based — no server
to stand up, which keeps the "clone and run" story intact.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

DEFAULT_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "docrag_chunks"


class VectorStore:
    def __init__(self, persist_dir: str | Path = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        # Chroma rejects a completely empty metadata dict outright (raises
        # ValueError), even though `None` or a populated dict are both fine.
        # A caller with genuinely no metadata for a chunk (an edge case in
        # tests, or a minimal ingestion path) shouldn't crash over it.
        safe_metadatas = [m if m else {"_empty": True} for m in metadatas]
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=safe_metadatas
        )

    def query(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """Returns a list of {chunk_id, text, metadata, distance} dicts."""
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        return [
            {
                "chunk_id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                # Chroma returns cosine *distance*; similarity = 1 - distance.
                "score": 1.0 - result["distances"][0][i],
            }
            for i in range(len(result["ids"][0]))
        ]

    def get_all(self) -> list[dict]:
        """Every chunk currently in the collection, as {chunk_id, text, metadata}.
        Used by GET /v1/documents (to list what's indexed) and POST
        /v1/ingest (to rebuild the BM25 index, which has no incremental-add
        API of its own - see indexing/sparse_index.py)."""
        if self.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            {"chunk_id": result["ids"][i], "text": result["documents"][i], "metadata": result["metadatas"][i]}
            for i in range(len(result["ids"]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
