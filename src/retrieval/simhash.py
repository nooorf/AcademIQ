"""
src/retrieval/simhash.py

Approximate retrieval using SimHash fingerprints + Hamming distance.
Implemented from scratch — no external hashing library except mmh3.


Usage:
    python -m src.retrieval.simhash \
        --chunks data/processed/chunks.jsonl \
        --index  data/processed/simhash_index.pkl
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import mmh3

from src.utils.preprocessing import tokenize, expand_query
from src.utils.config_loader import cfg


SIMHASH_BITS: int = cfg("simhash", "bits", default=64)


# ── Core SimHash functions ─────────────────────────────────────────────────────

def simhash(tokens: list[str], bits: int = SIMHASH_BITS) -> int:
    """
    Compute a SimHash fingerprint for a list of tokens.

    Algorithm:
        1. For each token, compute a b-bit hash using MurmurHash3.
        2. For each bit position i:
             vote[i] += 1  if bit i of hash is 1
             vote[i] -= 1  if bit i of hash is 0
        3. Final fingerprint: bit i = 1 if vote[i] > 0, else 0.

    Term frequency is captured naturally — repeated tokens vote more.
    """
    if not tokens:
        return 0

    votes = [0] * bits
    for token in tokens:
        h = mmh3.hash64(token, signed=False)[0]
        for i in range(bits):
            if h & (1 << i):
                votes[i] += 1
            else:
                votes[i] -= 1

    fingerprint = 0
    for i in range(bits):
        if votes[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits (Brian Kernighan's method)."""
    xor   = h1 ^ h2
    count = 0
    while xor:
        xor &= xor - 1
        count += 1
    return count


def hamming_similarity(h1: int, h2: int, bits: int = SIMHASH_BITS) -> float:
    """Convert Hamming distance to a [0, 1] similarity score."""
    return 1.0 - (hamming_distance(h1, h2) / bits)


# ── Retriever class ───────────────────────────────────────────────────────────

class SimHashRetriever:

    def __init__(self,
                 bits:              int = None,   # type: ignore[assignment]
                 hamming_threshold: int = None):  # type: ignore[assignment]
        self.bits              = bits              if bits              is not None \
                                 else cfg("simhash", "bits",              SIMHASH_BITS)
        self.hamming_threshold = hamming_threshold if hamming_threshold is not None \
                                 else cfg("simhash", "hamming_threshold", 10)
        self.fingerprints: dict[str, int]  = {}
        self.chunks:       dict[str, dict] = {}
        self.chunk_order:  list[str]       = []
        # Tracks chunks with genuinely empty token sets to avoid
        # false-positive matches against fingerprint 0
        self._empty_chunks: set[str]       = set()

    # ── Build ──────────────────────────────────────────────────────────────

    def fit(self, chunks: list[dict]) -> None:
        """
        Compute SimHash fingerprints for every chunk.

        Raises:
            ValueError: if chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot fit SimHashRetriever on an empty chunk list.")

        self.chunks      = {c["id"]: c for c in chunks}
        self.chunk_order = [c["id"] for c in chunks]

        print(f"Computing SimHash fingerprints for {len(chunks)} chunks "
              f"(bits={self.bits}, threshold={self.hamming_threshold})...")

        for chunk in chunks:
            tokens = tokenize(chunk["text"])   # list — TF weighting via repetition
            if not tokens:
                self.fingerprints[chunk["id"]] = 0
                self._empty_chunks.add(chunk["id"])
                continue
            self.fingerprints[chunk["id"]] = simhash(tokens, self.bits)

        print(f"Fingerprints computed — {len(self.fingerprints)} chunks indexed.")
        if self._empty_chunks:
            print(f"  Empty chunks (fingerprint=0): {len(self._empty_chunks)}")

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bits":              self.bits,
                "hamming_threshold": self.hamming_threshold,
                "fingerprints":      self.fingerprints,
                "chunks":            self.chunks,
                "chunk_order":       self.chunk_order,
                "empty_chunks":      self._empty_chunks,
            }, f)
        size_mb = path.stat().st_size / 1_048_576
        print(f"Index saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "SimHashRetriever":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"SimHash index not found: {path}\n"
                "Run: python -m src.retrieval.simhash "
                "--chunks data/processed/chunks.jsonl "
                "--index data/processed/simhash_index.pkl"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj                  = cls(bits=data["bits"],
                                   hamming_threshold=data["hamming_threshold"])
        obj.fingerprints     = data["fingerprints"]
        obj.chunks           = data["chunks"]
        obj.chunk_order      = data["chunk_order"]
        obj._empty_chunks    = data.get("empty_chunks", set())
        print(f"Index loaded — {len(obj.fingerprints)} chunks  "
              f"(bits={obj.bits}, threshold={obj.hamming_threshold})")
        return obj

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, q: str,
              top_k: int = 5,
              hamming_threshold: int = None) -> list[dict]:     # type: ignore[assignment]
        """
        Retrieve top_k most relevant chunks for query string q.

        Steps:
            1. Expand query with canonical synonyms.
            2. Tokenise with shared preprocessing (len >= 2 fix applied).
            3. Compute SimHash fingerprint for expanded query.
            4. Scan all fingerprints; filter by threshold.
            5. Fall back to full ranked list if < top_k pass the threshold.
            6. Skip chunks flagged as empty (fingerprint=0 ambiguity guard).

        Returns:
            List of chunk dicts with 'score', 'hamming_distance', 'latency_ms'.
        """
        if not self.fingerprints:
            raise RuntimeError("Index not built. Call fit() or load() first.")

        threshold = (hamming_threshold if hamming_threshold is not None
                     else self.hamming_threshold)

        t0 = time.perf_counter()

        q_expanded = expand_query(q)
        q_tokens   = tokenize(q_expanded)

        if not q_tokens:
            return []

        q_fp = simhash(q_tokens, self.bits)

        # Scan all fingerprints — O(n) but small corpus
        scored: list[tuple[str, int, float]] = []
        for cid, fp in self.fingerprints.items():
            if cid in self._empty_chunks:
                continue  # skip genuinely empty chunks
            dist = hamming_distance(q_fp, fp)
            sim  = hamming_similarity(q_fp, fp, self.bits)
            scored.append((cid, dist, sim))

        # Filter by threshold first
        candidates = [(cid, d, s) for cid, d, s in scored if d <= threshold]
        candidates.sort(key=lambda x: x[1])   # ascending distance = descending similarity

        # Fallback: threshold too strict → use full ranked list
        if len(candidates) < top_k:
            scored.sort(key=lambda x: x[1])
            candidates = scored

        top = candidates[:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000

        results: list[dict] = []
        for cid, dist, sim in top:
            chunk = dict(self.chunks[cid])
            chunk["score"]            = round(sim, 4)
            chunk["hamming_distance"] = dist
            chunk["latency_ms"]       = round(latency_ms, 2)
            results.append(chunk)

        return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main(chunks_path: str, index_path: str) -> None:
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    retriever = SimHashRetriever()
    retriever.fit(chunks)
    retriever.save(index_path)

    print("\n--- Smoke test (loaded from disk) ---")
    test_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What is the policy on plagiarism?",
    ]
    retriever_loaded = SimHashRetriever.load(index_path)
    for q in test_queries:
        results = retriever_loaded.query(q, top_k=3)
        print(f"\nQ: {q}")
        if not results:
            print("  No results returned.")
            continue
        for r in results:
            print(f"  score={r['score']:.4f}  "
                  f"hamming={r['hamming_distance']:2d}  "
                  f"page={r['page']:3d}  "
                  f"section={r['section'][:50]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SimHash retrieval index")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index",  required=True,
                        default=cfg("paths", "simhash_index",
                                    default="data/processed/simhash_index.pkl"))
    args = parser.parse_args()
    main(args.chunks, args.index)