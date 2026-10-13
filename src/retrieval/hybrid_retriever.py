"""
src/retrieval/hybrid_retriever.py

Centralised hybrid retrieval orchestrator.

Fuses results from all three retrieval methods (TF-IDF, MinHash+LSH, SimHash)
using Reciprocal Rank Fusion (RRF) with configurable per-method weights.

Also supports:
  - Adaptive routing based on query characteristics
  - Cascade mode (TF-IDF first → fallback to LSH if low confidence)
  - Per-method timing breakdown for the profiling dashboard

Usage:
    from src.retrieval.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(tfidf, minhash, simhash, reranker)
    results, timings = hybrid.retrieve(query, top_k=5, strategy="rrf_all")
"""

from __future__ import annotations

import time
from typing import Optional

from src.utils.config_loader import cfg


class HybridRetriever:
    """
    Orchestrates multi-method retrieval with RRF fusion.

    Strategies:
        rrf_all   : 3-way RRF (MinHash + SimHash + TF-IDF)
        rrf_lsh   : 2-way RRF (MinHash + SimHash only)
        cascade   : TF-IDF first → fallback to LSH if top-1 score < 0.10
        adaptive  : auto-route based on query length and characteristics
    """

    def __init__(self, tfidf=None, minhash=None, simhash=None, reranker=None):
        self.tfidf    = tfidf
        self.minhash  = minhash
        self.simhash  = simhash
        self.reranker = reranker

        self.rrf_k = cfg("hybrid", "rrf_k", default=60)
        self.weights = cfg("hybrid", "method_weights", default={
            "tfidf":  1.0,
            "minhash": 0.8,
            "simhash": 0.6,
        })

    # ── Core RRF fusion ───────────────────────────────────────────────────

    @staticmethod
    def rrf_fuse(results_lists: list[tuple[str, list[dict]]],
                 weights: dict[str, float] = None,
                 k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion with per-method weights.

        Each retriever's RRF score: weight * (1 / (k + rank + 1))
        Scores summed across methods. Correctly handles different scoring
        scales (Jaccard 0.03–0.15 vs cosine 0.05–0.30 vs Hamming 0.5–0.9).

        Args:
            results_lists: list of (method_name, results) tuples
            weights: per-method weight multiplier
            k: constant from RRF paper (default 60, Cormack 2009)

        Returns:
            Fused list of chunk dicts sorted by aggregate RRF score.
        """
        if weights is None:
            weights = {}

        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, dict]  = {}
        contributions: dict[str, dict[str, float]] = {}  # chunk_id → {method: score}

        for method_name, results in results_lists:
            w = weights.get(method_name, 1.0)
            for rank, chunk in enumerate(results):
                cid = chunk["id"]
                rrf_score = w * (1.0 / (k + rank + 1))
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

                if cid not in chunk_map:
                    chunk_map[cid] = chunk

                if cid not in contributions:
                    contributions[cid] = {}
                contributions[cid][method_name] = round(rrf_score, 6)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        fused: list[dict] = []
        for cid in sorted_ids:
            c = dict(chunk_map[cid])
            c["score"] = round(rrf_scores[cid], 6)
            c["rrf_contributions"] = contributions.get(cid, {})
            fused.append(c)

        return fused

    # ── Retrieval strategies ──────────────────────────────────────────────

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 strategy: str = "rrf_all",
                 hamming_threshold: int = None,
                 use_reranker: bool = True) -> tuple[list[dict], dict]:
        """
        Execute retrieval with the chosen strategy.

        Returns:
            (chunks, timings_dict)
        """
        candidates_k = top_k * 2 if use_reranker else top_k
        timings: dict[str, float] = {}

        if strategy == "rrf_all":
            chunks, timings = self._rrf_all(query, candidates_k, hamming_threshold)
        elif strategy == "rrf_lsh":
            chunks, timings = self._rrf_lsh(query, candidates_k, hamming_threshold)
        elif strategy == "cascade":
            chunks, timings = self._cascade(query, candidates_k, hamming_threshold)
        elif strategy == "adaptive":
            chunks, timings = self._adaptive(query, candidates_k, hamming_threshold)
        else:
            chunks, timings = self._rrf_all(query, candidates_k, hamming_threshold)

        # Re-rank if enabled
        if use_reranker and self.reranker and chunks:
            t0 = time.perf_counter()
            results = self.reranker.rerank(query, chunks, top_k=top_k)
            results = self.reranker.apply_diversity_penalty(results)
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return results, timings

        return chunks[:top_k], timings

    # ── Strategy: 3-way RRF ───────────────────────────────────────────────

    def _rrf_all(self, query: str, candidates_k: int,
                 hamming_threshold: int = None) -> tuple[list[dict], dict]:
        """Fuse all three methods via weighted RRF."""
        timings: dict[str, float] = {}
        results_lists: list[tuple[str, list[dict]]] = []

        if self.tfidf:
            t0 = time.perf_counter()
            tfidf_res = self.tfidf.query(query, top_k=candidates_k)
            timings["tfidf_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            results_lists.append(("tfidf", tfidf_res))

        if self.minhash:
            t0 = time.perf_counter()
            mh_res = self.minhash.query(query, top_k=candidates_k)
            timings["minhash_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            results_lists.append(("minhash", mh_res))

        if self.simhash:
            t0 = time.perf_counter()
            sh_kwargs = {"top_k": candidates_k}
            if hamming_threshold is not None:
                sh_kwargs["hamming_threshold"] = hamming_threshold
            sh_res = self.simhash.query(query, **sh_kwargs)
            timings["simhash_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            results_lists.append(("simhash", sh_res))

        fused = self.rrf_fuse(results_lists, weights=self.weights, k=self.rrf_k)
        return fused[:candidates_k], timings

    # ── Strategy: LSH-only RRF ────────────────────────────────────────────

    def _rrf_lsh(self, query: str, candidates_k: int,
                 hamming_threshold: int = None) -> tuple[list[dict], dict]:
        """Fuse MinHash + SimHash only (no TF-IDF)."""
        timings: dict[str, float] = {}
        results_lists: list[tuple[str, list[dict]]] = []

        if self.minhash:
            t0 = time.perf_counter()
            mh_res = self.minhash.query(query, top_k=candidates_k)
            timings["minhash_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            results_lists.append(("minhash", mh_res))

        if self.simhash:
            t0 = time.perf_counter()
            sh_kwargs = {"top_k": candidates_k}
            if hamming_threshold is not None:
                sh_kwargs["hamming_threshold"] = hamming_threshold
            sh_res = self.simhash.query(query, **sh_kwargs)
            timings["simhash_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            results_lists.append(("simhash", sh_res))

        fused = self.rrf_fuse(results_lists, weights=self.weights, k=self.rrf_k)
        return fused[:candidates_k], timings

    # ── Strategy: Cascade ─────────────────────────────────────────────────

    def _cascade(self, query: str, candidates_k: int,
                 hamming_threshold: int = None) -> tuple[list[dict], dict]:
        """TF-IDF first → fall back to LSH methods if low confidence."""
        timings: dict[str, float] = {}

        if self.tfidf:
            t0 = time.perf_counter()
            tfidf_res = self.tfidf.query(query, top_k=candidates_k)
            timings["tfidf_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            # Check confidence: if top-1 score is reasonable, use TF-IDF alone
            if tfidf_res and tfidf_res[0].get("score", 0) >= 0.10:
                return tfidf_res[:candidates_k], timings

        # Low confidence → trigger LSH methods and fuse
        return self._rrf_all(query, candidates_k, hamming_threshold)

    # ── Strategy: Adaptive ────────────────────────────────────────────────

    def _adaptive(self, query: str, candidates_k: int,
                  hamming_threshold: int = None) -> tuple[list[dict], dict]:
        """
        Auto-route based on query characteristics:
          - Short queries (≤5 words) → TF-IDF (keyword-heavy, discriminative)
          - Medium queries → 2-way RRF (MinHash + TF-IDF)
          - Long queries (≥15 words) → 3-way RRF (all methods)
        """
        word_count = len(query.split())

        if word_count <= 5:
            timings: dict[str, float] = {}
            if self.tfidf:
                t0 = time.perf_counter()
                res = self.tfidf.query(query, top_k=candidates_k)
                timings["tfidf_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                timings["strategy"] = "tfidf_only"
                return res[:candidates_k], timings

        if word_count <= 14:
            # Medium: MinHash + TF-IDF (2-way)
            timings = {}
            results_lists = []

            if self.tfidf:
                t0 = time.perf_counter()
                tf_res = self.tfidf.query(query, top_k=candidates_k)
                timings["tfidf_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                results_lists.append(("tfidf", tf_res))

            if self.minhash:
                t0 = time.perf_counter()
                mh_res = self.minhash.query(query, top_k=candidates_k)
                timings["minhash_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                results_lists.append(("minhash", mh_res))

            fused = self.rrf_fuse(results_lists, self.weights, self.rrf_k)
            return fused[:candidates_k], timings

        # Long query: all methods
        return self._rrf_all(query, candidates_k, hamming_threshold)