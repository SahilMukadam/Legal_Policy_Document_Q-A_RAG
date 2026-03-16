"""
Caching Module.

Provides in-memory caching with TTL (time-to-live) for:
    - RAG answers (avoid redundant LLM calls for same question)
    - Search results (avoid re-embedding identical queries)

Why caching matters:
    - LLM calls cost money and take 1-3 seconds each
    - Same question asked twice = same answer, no need to call again
    - Search embeddings are deterministic — same query = same vector

Usage:
    from src.utils.cache import ResponseCache

    cache = ResponseCache(ttl_seconds=300)  # 5 min TTL
    cache.set("key", {"answer": "..."})
    result = cache.get("key")  # Returns cached or None
"""

import time
import hashlib
from typing import Any


class ResponseCache:
    """Simple in-memory cache with TTL expiration."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 500):
        """
        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 min).
            max_size: Maximum number of entries before eviction.
        """
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, dict] = {}

    @staticmethod
    def _make_key(*args) -> str:
        """Create a deterministic cache key from arguments."""
        raw = "|".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """
        Get a cached value if it exists and hasn't expired.

        Returns:
            Cached value or None if miss/expired.
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() - entry["timestamp"] > self.ttl:
            # Expired — remove and return None
            del self._cache[key]
            return None

        entry["hits"] += 1
        return entry["value"]

    def set(self, key: str, value: Any):
        """Store a value in the cache."""
        # Evict oldest entries if at capacity
        if len(self._cache) >= self.max_size:
            self._evict()

        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "hits": 0,
        }

    def _evict(self):
        """Remove the oldest 20% of entries."""
        if not self._cache:
            return

        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k]["timestamp"],
        )

        num_to_remove = max(1, len(sorted_keys) // 5)
        for key in sorted_keys[:num_to_remove]:
            del self._cache[key]

    def invalidate(self, key: str):
        """Remove a specific entry."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        active = sum(1 for e in self._cache.values() if now - e["timestamp"] <= self.ttl)
        total_hits = sum(e["hits"] for e in self._cache.values())

        return {
            "total_entries": len(self._cache),
            "active_entries": active,
            "expired_entries": len(self._cache) - active,
            "total_hits": total_hits,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
        }
