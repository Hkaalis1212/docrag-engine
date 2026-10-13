"""
embeddings.py
-------------
Local embedding provider — the deliberate swap point vs. the reference
guide's OpenAI text-embedding-3-small. Using a local sentence-transformers
model means ingestion and retrieval need zero API key and zero per-call
cost; only the generation step (Claude) needs a key.

To swap in a hosted provider instead (OpenAI, Voyage, Cohere), implement the
same `embed(texts) -> list[list[float]]` interface and point
`get_embedder()` at it — nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from functools import lru_cache

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    """Process-wide singleton — the model is ~130MB; load it once."""
    return LocalEmbedder()
