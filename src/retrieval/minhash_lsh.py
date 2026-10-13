"""
src/retrieval/minhash_lsh.py

Approximate retrieval using MinHash signatures + Locality Sensitive Hashing.


Usage:
    python -m src.retrieval.minhash_lsh \
        --chunks data/processed/chunks.jsonl \
        --index  data/processed/minhash_index.pkl
"""

import argparse
import hashlib
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path

from src.utils.preprocessing import tokenize, expand_query
from src.utils.config_loader import cfg


MAX_HASH = (1 << 64) - 1


class MinHash:
    """Simple MinHash signature implementation over string tokens."""

    def __init__(self, num_perm: int = 128):
        if num_perm <= 0:
            raise ValueError("num_perm must be > 0")
        self.num_perm = num_perm
        self.hashvalues = [MAX_HASH] * num_perm

    @staticmethod
    def _perm_hash(token: bytes, perm_idx: int) -> int:
        seed = perm_idx.to_bytes(4, byteorder="little", signed=False)
        digest = hashlib.sha1(seed + token).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def update(self, token: bytes) -> None:
        for i in range(self.num_perm):
            hv = self._perm_hash(token, i)
            if hv < self.hashvalues[i]:
                self.hashvalues[i] = hv

    def jaccard(self, other: "MinHash") -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("MinHash signatures must have the same num_perm")
        equal = sum(1 for a, b in zip(self.hashvalues, other.hashvalues) if a == b)
        return equal / float(self.num_perm)


class MinHashLSH:
    """Banding-based LSH index for MinHash signatures."""

    def __init__(self, threshold: float = 0.3, num_perm: int = 128):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if num_perm <= 0:
            raise ValueError("num_perm must be > 0")
        self.threshold = threshold
        self.num_perm = num_perm
        self.b, self.r = self._choose_bands_and_rows(threshold, num_perm)
        self._buckets = [dict() for _ in range(self.b)]

    @staticmethod
    def _choose_bands_and_rows(threshold: float, num_perm: int) -> tuple[int, int]:
        best_b, best_r = 1, num_perm
        best_err = float("inf")
        for r in range(1, num_perm + 1):
            b = num_perm // r
            if b == 0:
                continue
            approx_t = (1.0 / b) ** (1.0 / r)
            err = abs(approx_t - threshold)
            if err < best_err:
                best_err = err
                best_b, best_r = b, r
        return best_b, best_r

    def _band_key(self, sig: MinHash, band_idx: int) -> tuple[int, ...]:
        start = band_idx * self.r
        end = min(start + self.r, self.num_perm)
        return tuple(sig.hashvalues[start:end])

    def insert(self, key: str, sig: MinHash) -> None:
        for band_idx in range(self.b):
            bkey = self._band_key(sig, band_idx)
            bucket = self._buckets[band_idx].setdefault(bkey, set())
            bucket.add(key)

    def query(self, sig: MinHash) -> list[str]:
        candidates: set[str] = set()
        for band_idx in range(self.b):
            bkey = self._band_key(sig, band_idx)
            matches = self._buckets[band_idx].get(bkey)
            if matches:
                candidates.update(matches)
        return list(candidates)


def build_minhash(token_set: set[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for token in token_set:
        m.update(token.encode("utf-8"))
    return m


def build_weighted_minhash(token_list: list[str],
                           idf_weights: dict[str, float],
                           num_perm: int = 128) -> MinHash:
    """
    Create a weighted MinHash signature using TF-IDF weights.

    Technique: Each token is hashed multiple times proportional to its
    TF-IDF weight (rounded to nearest integer). High-IDF tokens like
    "cgpa", "xf", "plagiarism" produce more hash values, dominating
    the signature. Low-IDF tokens like "policy" contribute less.

    This converts standard Jaccard (set similarity) into a weighted
    variant better suited for information retrieval.

    Args:
        token_list  : List of tokens (may contain duplicates for TF).
        idf_weights : dict mapping token → IDF weight.
        num_perm    : Number of hash permutations.
    """
    m = MinHash(num_perm=num_perm)

    # Compute TF for this document
    tf = Counter(token_list)
    max_tf = max(tf.values()) if tf else 1

    for token, freq in tf.items():
        # Normalized TF * IDF → weight
        idf = idf_weights.get(token, 1.0)
        weight = (0.5 + 0.5 * freq / max_tf) * idf
        # Hash the token N times proportional to weight (min 1)
        repeat = max(1, int(round(weight * 2)))
        for i in range(repeat):
            # Use indexed encoding to generate distinct hash values
            m.update(f"{token}_{i}".encode("utf-8"))

    return m


# ── Retriever class ───────────────────────────────────────────────────────────

class MinHashLSHRetriever:

    def __init__(self,
                 num_perm:  int   = None,   # type: ignore[assignment]
                 threshold: float = None,  # type: ignore[assignment]
                 use_weighted: bool = None):  # type: ignore[assignment]
        self.num_perm     = num_perm  if num_perm  is not None else cfg("minhash", "num_perm",  128)
        self.threshold    = threshold if threshold is not None else cfg("minhash", "threshold", 0.3)
        self.use_weighted = use_weighted if use_weighted is not None else cfg("minhash", "weighted", True)
        self.lsh:           MinHashLSH  = None   # type: ignore[assignment]
        self.signatures:    dict        = {}     # chunk_id -> MinHash
        self.chunks:        dict        = {}     # chunk_id -> chunk dict
        self.chunk_order:   list[str]   = []
        self.idf_weights:   dict[str, float] = {}  # token -> IDF weight
        self._token_counts: dict[str, int]   = {}  # chunk_id -> token count

    # ── Build ──────────────────────────────────────────────────────────────

    def _compute_idf(self, chunks: list[dict]) -> dict[str, float]:
        """
        Compute IDF weights across the corpus for weighted MinHash.

        IDF(t) = log(N / df(t)) where df(t) is the number of chunks
        containing token t. High-IDF tokens ("cgpa", "xf", "plagiarism")
        are more discriminative; low-IDF tokens ("policy", "required")
        appear everywhere and carry less signal.
        """
        n = len(chunks)
        df: Counter = Counter()
        for chunk in chunks:
            unique_tokens = tokenize(chunk["text"], as_set=True)
            for token in unique_tokens:
                df[token] += 1

        idf = {}
        for token, freq in df.items():
            idf[token] = math.log((n + 1) / (freq + 1)) + 1.0  # smoothed IDF
        return idf

    def fit(self, chunks: list[dict]) -> None:
        """
        Build MinHash signatures and LSH index for every chunk.

        When use_weighted=True, computes corpus-wide IDF weights and uses
        them to bias the MinHash signature generation. This improves
        retrieval relevance by making discriminative tokens ("cgpa", "xf")
        dominate the signature over common terms.

        Raises:
            ValueError: if chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot fit MinHashLSHRetriever on an empty chunk list.")

        self.chunks      = {c["id"]: c for c in chunks}
        self.chunk_order = [c["id"] for c in chunks]

        self.lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm =self.num_perm,
        )

        mode = "weighted" if self.use_weighted else "uniform"
        print(f"Building MinHash signatures for {len(chunks)} chunks "
              f"(num_perm={self.num_perm}, threshold={self.threshold}, mode={mode})...")

        # Compute IDF weights if using weighted mode
        if self.use_weighted:
            self.idf_weights = self._compute_idf(chunks)
            print(f"  IDF vocabulary: {len(self.idf_weights)} tokens")

        skipped = 0
        for chunk in chunks:
            if self.use_weighted:
                # Weighted MinHash: TF-IDF weighted token hashing
                token_list = tokenize(chunk["text"])  # list for TF counting
                if not token_list:
                    skipped += 1
                    continue
                sig = build_weighted_minhash(token_list, self.idf_weights, self.num_perm)
                self._token_counts[chunk["id"]] = len(set(token_list))
            else:
                # Standard MinHash: uniform set hashing
                tokens = tokenize(chunk["text"], as_set=True)
                if not tokens:
                    skipped += 1
                    continue
                sig = build_minhash(tokens, self.num_perm)
                self._token_counts[chunk["id"]] = len(tokens)

            self.signatures[chunk["id"]] = sig
            self.lsh.insert(chunk["id"], sig)

        if skipped:
            print(f"  Skipped {skipped} empty chunks (no tokens after preprocessing).")
        print(f"Index built — {len(self.signatures)} chunks indexed.")
        print(f"Bands: {self.lsh.b}   Rows per band: {self.lsh.r}")

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "num_perm":      self.num_perm,
                "threshold":     self.threshold,
                "use_weighted":  self.use_weighted,
                "signatures":    self.signatures,
                "chunks":        self.chunks,
                "chunk_order":   self.chunk_order,
                "idf_weights":   self.idf_weights,
                "token_counts":  self._token_counts,
                # LSH not pickled — rebuilt on load from signatures
            }, f)
        size_mb = path.stat().st_size / 1_048_576
        print(f"Index saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "MinHashLSHRetriever":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"MinHash index not found: {path}\n"
                "Run: python -m src.retrieval.minhash_lsh "
                "--chunks data/processed/chunks.jsonl "
                "--index data/processed/minhash_index.pkl"
            )
        with open(path, "rb") as f:
            try:
                data = pickle.load(f)
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "This MinHash index was built with an older dependency-based "
                    "format and is not compatible with the current in-house "
                    "MinHash implementation. Rebuild the index with:\n"
                    "python -m src.retrieval.minhash_lsh "
                    "--chunks data/processed/chunks.jsonl "
                    "--index data/processed/minhash_index.pkl"
                ) from e

        obj               = cls(num_perm=data["num_perm"],
                                threshold=data["threshold"],
                                use_weighted=data.get("use_weighted", False))
        obj.signatures    = data["signatures"]
        obj.chunks        = data["chunks"]
        obj.chunk_order   = data["chunk_order"]
        obj.idf_weights   = data.get("idf_weights", {})
        obj._token_counts = data.get("token_counts", {})

        # Rebuild LSH from stored signatures
        obj.lsh = MinHashLSH(threshold=obj.threshold, num_perm=obj.num_perm)
        for cid, sig in obj.signatures.items():
            obj.lsh.insert(cid, sig)

        mode = "weighted" if obj.use_weighted else "uniform"
        print(f"Index loaded — {len(obj.signatures)} chunks  "
              f"(num_perm={obj.num_perm}, threshold={obj.threshold}, mode={mode})")
        return obj

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, q: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top_k most relevant chunks for query string q.

        Steps:
            1. Expand query with canonical handbook synonyms (deduplicated).
            2. Tokenise with shared preprocessing (len >= 2 — "xf", "75" kept).
            3. Build MinHash signature for expanded query.
            4. LSH candidate retrieval.
            5. Rank candidates by estimated Jaccard similarity.
            6. Brute-force fallback if LSH returns fewer than top_k.

        Returns:
            List of chunk dicts with 'score' and 'latency_ms' added.
        """
        if self.lsh is None:
            raise RuntimeError("Index not built. Call fit() or load() first.")

        t0 = time.perf_counter()

        q_expanded = expand_query(q)
        q_tokens   = tokenize(q_expanded, as_set=True)

        if not q_tokens:
            return []

        if self.use_weighted:
            q_token_list = tokenize(q_expanded)
            q_sig = build_weighted_minhash(q_token_list, self.idf_weights, self.num_perm)
        else:
            q_sig = build_minhash(q_tokens, self.num_perm)

        # LSH candidate retrieval
        candidate_ids = self.lsh.query(q_sig)

        # Rank by estimated Jaccard
        scored: list[tuple[str, float]] = []
        for cid in candidate_ids:
            jaccard = q_sig.jaccard(self.signatures[cid])
            scored.append((cid, jaccard))

        # Budgeted fallback — scan at most 3×top_k (not entire index)
        if len(scored) < top_k:
            scored_ids = {cid for cid, _ in scored}
            budget = top_k * 3
            # Prioritise chunks with higher token overlap potential
            remaining = [
                cid for cid in self.chunk_order
                if cid not in scored_ids and cid in self.signatures
            ]
            fallback = [
                (cid, q_sig.jaccard(self.signatures[cid]))
                for cid in remaining[:budget]
            ]
            fallback.sort(key=lambda x: x[1], reverse=True)
            scored.extend(fallback[:top_k - len(scored)])

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        latency_ms = (time.perf_counter() - t0) * 1000

        results: list[dict] = []
        for cid, jaccard in top:
            chunk = dict(self.chunks[cid])
            chunk["score"]      = round(jaccard, 4)
            chunk["latency_ms"] = round(latency_ms, 2)
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

    retriever = MinHashLSHRetriever()
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
    retriever_loaded = MinHashLSHRetriever.load(index_path)
    for q in test_queries:
        results = retriever_loaded.query(q, top_k=3)
        print(f"\nQ: {q}")
        if not results:
            print("  No results returned.")
            continue
        for r in results:
            print(f"  score={r['score']:.4f}  page={r['page']:3d}  "
                  f"section={r['section'][:55]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MinHash + LSH retrieval index")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index",  required=True,
                        default=cfg("paths", "minhash_index",
                                    default="data/processed/minhash_index.pkl"))
    args = parser.parse_args()
    main(args.chunks, args.index)