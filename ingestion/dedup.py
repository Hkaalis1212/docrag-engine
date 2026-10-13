"""
dedup.py
--------
Near-duplicate detection before a chunk is inserted into the index.

Prevents the retriever from wasting top-k slots on redundant content when
the same information appears in multiple source documents (common in
internal docs — a config table copy-pasted into both a README and a wiki
page, for example).
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIMILARITY_THRESHOLD = 0.95


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DuplicateFilter:
    """
    Stateful filter: call `is_duplicate(embedding)` for each candidate chunk
    in insertion order. Returns True (and logs) if the chunk is a near-dupe
    of anything already accepted; otherwise records it and returns False.

    O(n) per check against everything accepted so far — fine at the scale of
    a single internal-docs corpus (thousands, not millions, of chunks). A
    production system at larger scale would use an ANN index for this check
    instead of a linear scan.
    """

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._accepted: list[np.ndarray] = []
        self.duplicate_count = 0

    def is_duplicate(self, embedding: list[float] | np.ndarray) -> bool:
        vec = np.asarray(embedding)
        for existing in self._accepted:
            if cosine_similarity(vec, existing) > self.threshold:
                self.duplicate_count += 1
                return True
        self._accepted.append(vec)
        return False
