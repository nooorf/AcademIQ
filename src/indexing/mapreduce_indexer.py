"""
src/indexing/mapreduce_indexer.py

Simulated MapReduce framework for distributed chunk indexing.

Demonstrates the MapReduce paradigm applied to building retrieval indexes:
  Map   : Each worker processes a partition of chunks → emits (chunk_id, signature)
  Shuffle: Group signatures by LSH band bucket
  Reduce : Merge signatures into the final index structure

Uses Python multiprocessing.Pool to parallelise across CPU cores, simulating
a distributed cluster showing that index construction scales horizontally with worker count.

Usage:
    python -m src.indexing.mapreduce_indexer \\
        --chunks data/processed/chunks.jsonl \\
        --method minhash \\
        --workers 4

    # Or from Python:
    from src.indexing.mapreduce_indexer import MapReduceIndexer
    indexer = MapReduceIndexer(n_workers=4)
    index = indexer.build_minhash_index(chunks)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from src.retrieval.minhash_lsh import MinHash, MinHashLSH
from src.utils.config_loader import cfg
from src.utils.preprocessing import tokenize


# ── Module-level map functions (must be picklable for multiprocessing) ────────

def _map_minhash(args: tuple) -> list[tuple[str, dict, MinHash]]:
    """
    Map phase: process a partition of chunks → (chunk_id, chunk_dict, minhash).
    
    Each worker independently computes MinHash signatures for its partition.
    This is embarrassingly parallel — no inter-worker communication needed.
    """
    partition, num_perm = args
    results = []
    for chunk in partition:
        tokens = tokenize(chunk["text"], as_set=True)
        if not tokens:
            continue
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(token.encode("utf-8"))
        results.append((chunk["id"], chunk, m))
    return results


def _map_simhash(args: tuple) -> list[tuple[str, dict, int]]:
    """
    Map phase: process a partition of chunks → (chunk_id, chunk_dict, fingerprint).
    """
    import mmh3
    partition, bits = args
    results = []
    for chunk in partition:
        tokens = tokenize(chunk["text"])
        if not tokens:
            results.append((chunk["id"], chunk, 0))
            continue
        votes = [0] * bits
        for token in tokens:
            h = mmh3.hash64(token, signed=False)[0]
            for i in range(bits):
                if h & (1 << i):
                    votes[i] += 1
                else:
                    votes[i] -= 1
        fp = 0
        for i in range(bits):
            if votes[i] > 0:
                fp |= (1 << i)
        results.append((chunk["id"], chunk, fp))
    return results


def _map_tfidf_partition(args: tuple) -> list[tuple[str, dict, list[str]]]:
    """
    Map phase: process a partition of chunks → (chunk_id, chunk_dict, tokens).
    Pre-tokenises text for the TF-IDF vectorizer.
    """
    partition, _ = args
    results = []
    for chunk in partition:
        tokens = tokenize(chunk["text"])
        results.append((chunk["id"], chunk, tokens))
    return results


# ── MapReduce Indexer ─────────────────────────────────────────────────────────

class MapReduceIndexer:
    """
    Simulated MapReduce framework for building retrieval indexes.

    Architecture:
        ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
        │  Partition   │────▶│  Map (×N)    │────▶│   Shuffle    │
        │  (split by   │     │  (parallel   │     │  (group by   │
        │   workers)   │     │   workers)   │     │   chunk_id)  │
        └─────────────┘     └──────────────┘     └──────────────┘
                                                        │
                                                        ▼
                                                 ┌──────────────┐
                                                 │   Reduce     │
                                                 │  (build LSH  │
                                                 │   index)     │
                                                 └──────────────┘

    Metrics collected at each phase for the scalability report.
    """

    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or cfg("mapreduce", "n_workers", default=4)
        self.n_workers = min(self.n_workers, cpu_count())
        self._timings: dict[str, float] = {}

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)

    # ── Partitioning ──────────────────────────────────────────────────────

    def _partition(self, chunks: list[dict]) -> list[list[dict]]:
        """Split chunks into roughly equal partitions for map workers."""
        n = len(chunks)
        size = math.ceil(n / self.n_workers)
        partitions = []
        for i in range(0, n, size):
            partitions.append(chunks[i:i + size])
        return partitions

    # ── MinHash Index ─────────────────────────────────────────────────────

    def build_minhash_index(self,
                            chunks: list[dict],
                            num_perm: int = None,
                            threshold: float = None) -> dict:
        """
        Build MinHash+LSH index using MapReduce parallelism.

        Returns:
            dict with keys: lsh, signatures, chunks, chunk_order, timings
        """
        num_perm  = num_perm  or cfg("minhash", "num_perm",  default=128)
        threshold = threshold or cfg("minhash", "threshold", default=0.3)

        print(f"MapReduce MinHash — {len(chunks)} chunks × {self.n_workers} workers")

        # ── PARTITION ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        partitions = self._partition(chunks)
        self._timings["partition_ms"] = (time.perf_counter() - t0) * 1000
        print(f"  Partition: {len(partitions)} partitions "
              f"({self._timings['partition_ms']:.1f}ms)")

        # ── MAP (parallel) ────────────────────────────────────────────
        t0 = time.perf_counter()
        map_args = [(p, num_perm) for p in partitions]

        with Pool(processes=self.n_workers) as pool:
            map_results = pool.map(_map_minhash, map_args)

        self._timings["map_ms"] = (time.perf_counter() - t0) * 1000
        total_mapped = sum(len(r) for r in map_results)
        print(f"  Map:       {total_mapped} signatures "
              f"({self._timings['map_ms']:.1f}ms)")

        # ── SHUFFLE (flatten + group) ─────────────────────────────────
        t0 = time.perf_counter()
        all_results: list[tuple[str, dict, MinHash]] = []
        for batch in map_results:
            all_results.extend(batch)

        signatures: dict[str, MinHash] = {}
        chunk_store: dict[str, dict]   = {}
        chunk_order: list[str]         = []
        for cid, chunk, sig in all_results:
            signatures[cid] = sig
            chunk_store[cid] = chunk
            chunk_order.append(cid)

        self._timings["shuffle_ms"] = (time.perf_counter() - t0) * 1000
        print(f"  Shuffle:   grouped {len(signatures)} entries "
              f"({self._timings['shuffle_ms']:.1f}ms)")

        # ── REDUCE (build LSH index) ──────────────────────────────────
        t0 = time.perf_counter()
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for cid, sig in signatures.items():
            lsh.insert(cid, sig)

        self._timings["reduce_ms"] = (time.perf_counter() - t0) * 1000
        print(f"  Reduce:    LSH index built "
              f"({self._timings['reduce_ms']:.1f}ms)")

        total = sum(self._timings.values())
        self._timings["total_ms"] = total
        print(f"  Total:     {total:.1f}ms  "
              f"(Bands={lsh.b}, Rows={lsh.r})")

        return {
            "lsh":         lsh,
            "signatures":  signatures,
            "chunks":      chunk_store,
            "chunk_order": chunk_order,
            "num_perm":    num_perm,
            "threshold":   threshold,
            "timings":     dict(self._timings),
            "n_workers":   self.n_workers,
        }

    # ── SimHash Index ─────────────────────────────────────────────────────

    def build_simhash_index(self,
                            chunks: list[dict],
                            bits: int = None,
                            hamming_threshold: int = None) -> dict:
        """Build SimHash index using MapReduce parallelism."""
        bits              = bits              or cfg("simhash", "bits", default=64)
        hamming_threshold = hamming_threshold or cfg("simhash", "hamming_threshold", default=10)

        print(f"MapReduce SimHash — {len(chunks)} chunks × {self.n_workers} workers")

        t0 = time.perf_counter()
        partitions = self._partition(chunks)
        self._timings["partition_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with Pool(processes=self.n_workers) as pool:
            map_results = pool.map(_map_simhash,
                                   [(p, bits) for p in partitions])
        self._timings["map_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fingerprints: dict[str, int]  = {}
        chunk_store:  dict[str, dict] = {}
        chunk_order:  list[str]       = []
        empty_chunks: set[str]        = set()

        for batch in map_results:
            for cid, chunk, fp in batch:
                fingerprints[cid] = fp
                chunk_store[cid]  = chunk
                chunk_order.append(cid)
                tokens = tokenize(chunk["text"])
                if not tokens:
                    empty_chunks.add(cid)

        self._timings["shuffle_ms"] = (time.perf_counter() - t0) * 1000

        total = sum(self._timings.values())
        self._timings["total_ms"] = total
        print(f"  Total: {total:.1f}ms — {len(fingerprints)} fingerprints")

        return {
            "fingerprints":      fingerprints,
            "chunks":            chunk_store,
            "chunk_order":       chunk_order,
            "empty_chunks":      empty_chunks,
            "bits":              bits,
            "hamming_threshold": hamming_threshold,
            "timings":           dict(self._timings),
            "n_workers":         self.n_workers,
        }

    # ── Save helpers ──────────────────────────────────────────────────────

    @staticmethod
    def save_index(index_data: dict, path: str) -> None:
        """Persist a MapReduce-built index to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(index_data, f)
        size_mb = p.stat().st_size / 1_048_576
        print(f"Index saved → {p}  ({size_mb:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build retrieval indexes using MapReduce parallelism")
    parser.add_argument("--chunks",  required=True, help="Path to chunks.jsonl")
    parser.add_argument("--method",  default="minhash",
                        choices=["minhash", "simhash"],
                        help="Index type to build")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--output",  default=None,
                        help="Output index path (auto-generated if omitted)")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    indexer = MapReduceIndexer(n_workers=args.workers)

    if args.method == "minhash":
        result = indexer.build_minhash_index(chunks)
        out = args.output or "data/processed/minhash_mr_index.pkl"
    else:
        result = indexer.build_simhash_index(chunks)
        out = args.output or "data/processed/simhash_mr_index.pkl"

    MapReduceIndexer.save_index(result, out)
    print(f"\nTimings breakdown: {result['timings']}")


if __name__ == "__main__":
    main()