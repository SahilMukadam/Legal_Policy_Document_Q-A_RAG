"""
Tests for the caching module.
Run with: pytest tests/test_cache.py -v
"""

import time
import pytest

from src.utils.cache import ResponseCache


class TestResponseCache:

    def test_set_and_get(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.set("key1", {"answer": "Hello"})

        result = cache.get("key1")
        assert result is not None
        assert result["answer"] == "Hello"

    def test_cache_miss(self):
        cache = ResponseCache(ttl_seconds=60)
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self):
        """Entries expire after TTL."""
        cache = ResponseCache(ttl_seconds=1)  # 1 second TTL
        cache.set("key1", {"data": "test"})

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.5)
        assert cache.get("key1") is None

    def test_make_key_deterministic(self):
        """Same args should produce same key."""
        key1 = ResponseCache._make_key("question", 5, "all")
        key2 = ResponseCache._make_key("question", 5, "all")
        assert key1 == key2

    def test_make_key_different_args(self):
        """Different args should produce different keys."""
        key1 = ResponseCache._make_key("question1", 5, "all")
        key2 = ResponseCache._make_key("question2", 5, "all")
        assert key1 != key2

    def test_invalidate(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.set("key1", "value1")
        cache.invalidate("key1")

        assert cache.get("key1") is None

    def test_clear(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_max_size_eviction(self):
        """Cache should evict old entries when at capacity."""
        cache = ResponseCache(ttl_seconds=60, max_size=5)

        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Should have at most max_size entries
        assert len(cache._cache) <= 5

    def test_hit_counting(self):
        cache = ResponseCache(ttl_seconds=60)
        cache.set("key1", "value1")

        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.stats()
        assert stats["total_hits"] == 3

    def test_stats(self):
        cache = ResponseCache(ttl_seconds=60, max_size=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 60
