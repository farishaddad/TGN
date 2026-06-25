"""
In-memory embedding cache (BRIGHT Lambda architecture, CIKM 2022).

Stores pre-computed entity embeddings with timestamps and staleness
tracking. Local implementation uses a dict; production replaces with
Redis (interface matches redis-py so swapping is a one-line change).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CacheEntry:
    """A cached embedding with metadata."""

    embedding: np.ndarray
    timestamp: float  # When this embedding was computed
    access_count: int = 0


class EmbeddingCache:
    """In-memory embedding cache with staleness tracking.

    Interface is compatible with redis-py patterns so production
    migration to Redis/ElastiCache is straightforward.

    Args:
        max_age_seconds: Maximum staleness before entry is considered stale
    """

    def __init__(self, max_age_seconds: float = 3600.0):
        self.max_age_seconds = max_age_seconds
        self._store: dict[int, CacheEntry] = {}

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._store)

    def get(self, node_id: int) -> Optional[np.ndarray]:
        """Retrieve a cached embedding.

        Args:
            node_id: Node to look up

        Returns:
            Embedding array or None if not cached
        """
        entry = self._store.get(node_id)
        if entry is None:
            return None
        entry.access_count += 1
        return entry.embedding

    def set(self, node_id: int, embedding: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Store an embedding in the cache.

        Args:
            node_id: Node ID to store
            embedding: Embedding vector
            timestamp: When embedding was computed (default: now)
        """
        if timestamp is None:
            timestamp = time.time()
        self._store[node_id] = CacheEntry(
            embedding=embedding.copy(),
            timestamp=timestamp,
        )

    def is_stale(self, node_id: int, reference_time: Optional[float] = None) -> bool:
        """Check if a cached embedding is stale.

        Args:
            node_id: Node to check
            reference_time: Current time (default: time.time())

        Returns:
            True if entry doesn't exist or is older than max_age_seconds
        """
        entry = self._store.get(node_id)
        if entry is None:
            return True
        if reference_time is None:
            reference_time = time.time()
        return (reference_time - entry.timestamp) > self.max_age_seconds

    def get_timestamp(self, node_id: int) -> Optional[float]:
        """Get the timestamp when an embedding was last computed."""
        entry = self._store.get(node_id)
        return entry.timestamp if entry else None

    def delete(self, node_id: int) -> bool:
        """Remove an entry from the cache.

        Returns:
            True if entry existed and was removed
        """
        if node_id in self._store:
            del self._store[node_id]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def get_stale_nodes(self, reference_time: Optional[float] = None) -> list[int]:
        """Get all node IDs with stale embeddings.

        Args:
            reference_time: Current time reference

        Returns:
            List of stale node IDs
        """
        if reference_time is None:
            reference_time = time.time()
        return [
            node_id for node_id, entry in self._store.items()
            if (reference_time - entry.timestamp) > self.max_age_seconds
        ]

    def get_all_node_ids(self) -> list[int]:
        """Get all cached node IDs."""
        return list(self._store.keys())

    def save(self, path: str | Path) -> None:
        """Persist cache to disk (numpy format).

        Args:
            path: Directory to save cache files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings as a single numpy file
        node_ids = sorted(self._store.keys())
        if not node_ids:
            return

        embeddings = np.stack([self._store[nid].embedding for nid in node_ids])
        timestamps = np.array([self._store[nid].timestamp for nid in node_ids])

        np.savez(
            path / "cache.npz",
            node_ids=np.array(node_ids),
            embeddings=embeddings,
            timestamps=timestamps,
        )

        # Save metadata
        meta = {
            "max_age_seconds": self.max_age_seconds,
            "size": len(node_ids),
        }
        (path / "meta.json").write_text(json.dumps(meta))

    def load(self, path: str | Path) -> "EmbeddingCache":
        """Load cache from disk.

        Args:
            path: Directory containing saved cache files

        Returns:
            self (for chaining)
        """
        path = Path(path)
        data = np.load(path / "cache.npz")

        node_ids = data["node_ids"]
        embeddings = data["embeddings"]
        timestamps = data["timestamps"]

        for nid, emb, ts in zip(node_ids, embeddings, timestamps):
            self._store[int(nid)] = CacheEntry(
                embedding=emb, timestamp=float(ts),
            )

        # Load metadata
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.max_age_seconds = meta.get("max_age_seconds", self.max_age_seconds)

        return self

    def stats(self) -> dict[str, float]:
        """Cache statistics for monitoring."""
        if not self._store:
            return {"size": 0, "avg_age": 0, "stale_pct": 0}

        now = time.time()
        ages = [now - entry.timestamp for entry in self._store.values()]
        stale_count = sum(1 for a in ages if a > self.max_age_seconds)

        return {
            "size": len(self._store),
            "avg_age": float(np.mean(ages)),
            "max_age": float(np.max(ages)),
            "stale_pct": stale_count / len(self._store),
            "total_accesses": sum(e.access_count for e in self._store.values()),
        }
