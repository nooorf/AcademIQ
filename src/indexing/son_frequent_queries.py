"""
src/indexing/son_frequent_queries.py

SON (Savasere, Omiecinski, Navathe) algorithm for mining frequent query
patterns from the query log.

Two-pass approach adapted for query-pattern mining:
  Pass 1 (Local):  Partition the query log. Each partition independently finds
                   candidate frequent itemsets using A-Priori with a scaled
                   support threshold (support / num_partitions).
  Pass 2 (Global): Verify candidates against the full query log.

Usage:
    from src.indexing.son_frequent_queries import SONFrequentQueries

    son = SONFrequentQueries(min_support=0.1)
    frequent = son.mine(query_log)
    # Returns: [frozenset({'gpa', 'cgpa'}), frozenset({'attendance', 'xf'}), ...]
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Optional

from src.utils.preprocessing import tokenize, STOP_WORDS


class SONFrequentQueries:
    """
    SON algorithm for frequent itemset mining on query keyword sets.

    Parameters:
        min_support   : Minimum fraction of queries containing the itemset.
        num_partitions: Number of partitions for the local pass.
        max_itemset_size: Maximum itemset cardinality to mine (2 or 3).
    """

    def __init__(self,
                 min_support: float = 0.1,
                 num_partitions: int = 4,
                 max_itemset_size: int = 3):
        self.min_support      = min_support
        self.num_partitions   = num_partitions
        self.max_itemset_size = max_itemset_size

    # ── Tokenise queries into keyword baskets ─────────────────────────────

    @staticmethod
    def _query_to_basket(query: str) -> frozenset[str]:
        """Convert a query string into a set of meaningful keywords."""
        tokens = tokenize(query, as_set=True)
        # Remove very short tokens that add noise to itemset mining
        return frozenset(t for t in tokens if len(t) >= 3)

    # ── A-Priori on a single partition ────────────────────────────────────

    def _apriori_local(self,
                       baskets: list[frozenset[str]],
                       local_support: int) -> set[frozenset[str]]:
        """
        Run A-Priori algorithm on a partition with scaled support threshold.

        Returns candidate frequent itemsets (size 2 up to max_itemset_size).
        """
        # Pass 1: Count individual items
        item_counts: Counter[str] = Counter()
        for basket in baskets:
            for item in basket:
                item_counts[item] += 1

        # Frequent singletons
        freq_items = {
            item for item, count in item_counts.items()
            if count >= local_support
        }

        candidates: set[frozenset[str]] = set()

        # Generate pairs from frequent singletons
        pair_counts: Counter[frozenset[str]] = Counter()
        for basket in baskets:
            freq_in_basket = basket & freq_items
            for pair in combinations(sorted(freq_in_basket), 2):
                pair_counts[frozenset(pair)] += 1

        freq_pairs = {
            pair for pair, count in pair_counts.items()
            if count >= local_support
        }
        candidates.update(freq_pairs)

        # Optionally mine triples
        if self.max_itemset_size >= 3 and freq_pairs:
            triple_counts: Counter[frozenset[str]] = Counter()
            # Candidate triples from frequent pairs (Apriori principle)
            pair_items = set()
            for pair in freq_pairs:
                pair_items.update(pair)

            for basket in baskets:
                freq_in_basket = sorted(basket & pair_items)
                for triple in combinations(freq_in_basket, 3):
                    fs = frozenset(triple)
                    # All sub-pairs must be frequent (Apriori pruning)
                    sub_pairs = [frozenset(c) for c in combinations(triple, 2)]
                    if all(sp in freq_pairs for sp in sub_pairs):
                        triple_counts[fs] += 1

            freq_triples = {
                t for t, count in triple_counts.items()
                if count >= local_support
            }
            candidates.update(freq_triples)

        return candidates

    # ── SON two-pass algorithm ────────────────────────────────────────────

    def mine(self, query_log: list[str]) -> list[dict]:
        """
        Run the full SON algorithm on a query log.

        Args:
            query_log: List of raw query strings.

        Returns:
            List of dicts:
              { "itemset": frozenset, "support": float, "size": int }
            sorted by support descending.
        """
        if not query_log:
            return []

        # Convert to baskets
        baskets = [self._query_to_basket(q) for q in query_log]
        baskets = [b for b in baskets if len(b) >= 2]  # need ≥2 items

        if not baskets:
            return []

        n = len(baskets)
        abs_threshold = max(1, int(n * self.min_support))

        # ── Partition ─────────────────────────────────────────────────
        part_size = math.ceil(n / self.num_partitions)
        partitions = [baskets[i:i + part_size]
                      for i in range(0, n, part_size)]

        # ── Pass 1 (Local): Find candidate itemsets per partition ──────
        all_candidates: set[frozenset[str]] = set()
        for partition in partitions:
            local_threshold = max(1, int(len(partition) * self.min_support))
            local_candidates = self._apriori_local(partition, local_threshold)
            all_candidates.update(local_candidates)

        if not all_candidates:
            return []

        # ── Pass 2 (Global): Verify candidates against full log ───────
        global_counts: Counter[frozenset[str]] = Counter()
        for basket in baskets:
            for candidate in all_candidates:
                if candidate.issubset(basket):
                    global_counts[candidate] += 1

        # Filter by global support
        frequent = [
            {
                "itemset": itemset,
                "support": round(count / n, 4),
                "count":   count,
                "size":    len(itemset),
            }
            for itemset, count in global_counts.items()
            if count >= abs_threshold
        ]

        frequent.sort(key=lambda x: x["support"], reverse=True)
        return frequent

    # ── Utility: pre-warm cache keys ──────────────────────────────────────

    def suggest_cache_warmup(self,
                             frequent_patterns: list[dict],
                             top_n: int = 10) -> list[str]:
        """
        Generate suggested queries for cache pre-warming based on
        frequent keyword patterns.

        Returns list of synthetic query strings constructed from
        the most common keyword co-occurrences.
        """
        suggestions = []
        for pattern in frequent_patterns[:top_n]:
            keywords = sorted(pattern["itemset"])
            query = "What is the policy on " + " and ".join(keywords) + "?"
            suggestions.append(query)
        return suggestions