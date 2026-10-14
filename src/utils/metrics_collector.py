"""
src/utils/metrics_collector.py

Centralised performance metrics collector for the pipeline.
Records per-query latency breakdown and aggregates P50/P99 statistics.

Usage:
    from src.utils.metrics_collector import metrics

    metrics.record(query="...", method="TF-IDF",
                   retrieval_ms=3.2, rerank_ms=1.1, generation_ms=450,
                   cache_hit=False, num_chunks=5)

    print(metrics.summary())
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryRecord:
    """Single query execution record."""
    query:          str
    method:         str
    retrieval_ms:   float
    rerank_ms:      float
    generation_ms:  float
    total_ms:       float
    cache_hit:      bool
    num_chunks:     int
    precision_k:    Optional[float] = None   # fraction of top-k chunks deemed relevant
    timestamp:      float = field(default_factory=time.time)


class MetricsCollector:
    """
    Thread-safe metrics aggregator.

    Stores the last N query records and provides percentile-based
    latency statistics for the web app's Health view.
    """

    def __init__(self, max_history: int = 500):
        self._records: list[QueryRecord] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def record(self, *,
               query: str,
               method: str,
               retrieval_ms: float  = 0.0,
               rerank_ms: float     = 0.0,
               generation_ms: float = 0.0,
               cache_hit: bool      = False,
               num_chunks: int      = 0,
               precision_k: Optional[float] = None) -> None:
        """Record a single query execution."""
        total = retrieval_ms + rerank_ms + generation_ms
        rec = QueryRecord(
            query=query, method=method,
            retrieval_ms=retrieval_ms, rerank_ms=rerank_ms,
            generation_ms=generation_ms, total_ms=total,
            cache_hit=cache_hit, num_chunks=num_chunks,
            precision_k=precision_k,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history:]

    def _latencies(self, field: str = "total_ms") -> list[float]:
        with self._lock:
            return [getattr(r, field) for r in self._records]

    def p50(self, field: str = "total_ms") -> float:
        vals = self._latencies(field)
        return round(statistics.median(vals), 2) if vals else 0.0

    def p99(self, field: str = "total_ms") -> float:
        vals = sorted(self._latencies(field))
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.99)
        return round(vals[min(idx, len(vals) - 1)], 2)

    def avg(self, field: str = "total_ms") -> float:
        vals = self._latencies(field)
        return round(statistics.mean(vals), 2) if vals else 0.0

    @property
    def total_queries(self) -> int:
        return len(self._records)

    @property
    def cache_hit_rate(self) -> float:
        with self._lock:
            if not self._records:
                return 0.0
            hits = sum(1 for r in self._records if r.cache_hit)
            return round(hits / len(self._records), 4)

    def method_breakdown(self) -> dict[str, int]:
        """Count queries by retrieval method."""
        with self._lock:
            counts: dict[str, int] = {}
            for r in self._records:
                counts[r.method] = counts.get(r.method, 0) + 1
            return counts

    def precision_k_stats(self) -> dict:
        """Return avg, p50, and p99 Precision@k over queries that have a score."""
        with self._lock:
            scores = [r.precision_k for r in self._records if r.precision_k is not None]
        if not scores:
            return {"avg": None, "p50": None, "p99": None, "n": 0}
        sorted_scores = sorted(scores)
        p50 = statistics.median(sorted_scores)
        idx = int(len(sorted_scores) * 0.99)
        p99 = sorted_scores[min(idx, len(sorted_scores) - 1)]
        return {
            "avg": round(statistics.mean(scores), 4),
            "p50": round(p50, 4),
            "p99": round(p99, 4),
            "n":   len(scores),
        }

    def summary(self) -> dict:
        return {
            "total_queries":    self.total_queries,
            "cache_hit_rate":   self.cache_hit_rate,
            "p50_total_ms":     self.p50("total_ms"),
            "p99_total_ms":     self.p99("total_ms"),
            "p50_retrieval":    self.p50("retrieval_ms"),
            "p99_retrieval":    self.p99("retrieval_ms"),
            "avg_chunks":       self.avg("num_chunks") if self._records else 0,
            "method_breakdown": self.method_breakdown(),
            "precision_k":      self.precision_k_stats(),
        }

    def recent(self, n: int = 10) -> list[dict]:
        """Return the last n query records as dicts."""
        with self._lock:
            recs = self._records[-n:]
        return [
            {
                "query":         r.query[:60],
                "method":        r.method,
                "retrieval_ms":  r.retrieval_ms,
                "rerank_ms":     r.rerank_ms,
                "generation_ms": r.generation_ms,
                "total_ms":      r.total_ms,
                "cache_hit":     r.cache_hit,
            }
            for r in reversed(recs)
        ]


# ── Singleton instance ────────────────────────────────────────────────────────
metrics = MetricsCollector()