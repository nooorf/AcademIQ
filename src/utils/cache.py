"""
src/utils/cache.py

Thread-safe LRU query cache with TTL expiry for the retrieval pipeline.

Caching is critical at scale — identical queries from multiple students
(e.g. "What is the GPA requirement?") would otherwise recompute
tokenisation, MinHash signatures, and cosine similarity on every hit.

Usage:
    from src.utils.cache import QueryCache
    cache = QueryCache(max_size=256, ttl_seconds=3600)
    
    key = cache.make_key(query, method, top_k)
    if (hit := cache.get(key)) is not None:
        return hit
    result = expensive_retrieval(...)
    cache.put(key, result)
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from src.utils.config_loader import cfg


class QueryCache:
    """
    Thread-safe LRU cache with time-to-live (TTL) expiry.

    Design decisions:
      - OrderedDict for O(1) LRU eviction (move_to_end + popitem).
      - SHA-256 key hashing prevents pathological key sizes from long queries.
      - TTL prevents stale results after index rebuild.
      - Thread lock guards concurrent API requests.
    """

    def __init__(self,
                 max_size:    int = None,
                 ttl_seconds: int = None):
        self.max_size    = max_size    or cfg("cache", "max_size",    default=256)
        self.ttl_seconds = ttl_seconds or cfg("cache", "ttl_seconds", default=3600)
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock  = threading.Lock()

        # Counters for hit-rate monitoring
        self._hits   = 0
        self._misses = 0

    # ── Key generation ────────────────────────────────────────────────────

    @staticmethod
    def make_key(query: str, method: str, top_k: int,
                 extra: str = "") -> str:
        """Deterministic cache key from query parameters."""
        raw = f"{query.strip().lower()}|{method}|{top_k}|{extra}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            ts, value = self._store[key]
            if time.time() - ts > self.ttl_seconds:
                # Expired — evict
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    # ── Insert ────────────────────────────────────────────────────────────

    def put(self, key: str, value: Any) -> None:
        """Insert or update. Evicts LRU entry if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (time.time(), value)
                return

            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)   # evict oldest

            self._store[key] = (time.time(), value)

    # ── Monitoring ────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size":       self.size,
            "max_size":   self.max_size,
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(self.hit_rate, 4),
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits   = 0
            self._misses = 0