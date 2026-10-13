"""
sparse_index.py
----------------
BM25 keyword index, built over the identical chunk set as the vector store
so the two indexes can never drift apart — both are populated in the same
ingestion pass from the same chunk list (see scripts/seed.py).

Catches exact-term matches (config keys, function names, error codes,
pricing figures) that dense/semantic search can blur past.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SparseHit:
    chunk_id: str
    score: float


class SparseIndex:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def build(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        self._chunk_ids = list(chunk_ids)
        self._texts = list(texts)
        self._metadatas = list(metadatas)
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def query(self, query_text: str, top_k: int = 10) -> list[SparseHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            SparseHit(chunk_id=self._chunk_ids[i], score=float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]

    def get_text(self, chunk_id: str) -> str | None:
        try:
            return self._texts[self._chunk_ids.index(chunk_id)]
        except ValueError:
            return None

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunk_ids": self._chunk_ids,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SparseIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index.build(data["chunk_ids"], data["texts"], data["metadatas"])
        return index
