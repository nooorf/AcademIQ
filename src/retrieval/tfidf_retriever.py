"""
src/retrieval/tfidf_retriever.py

Exact baseline retriever using TF-IDF + cosine similarity.
Ground truth against which MinHash and SimHash are benchmarked.

Usage:
    python -m src.retrieval.tfidf_retriever \
        --chunks data/processed/chunks.jsonl \
        --index  data/processed/tfidf_index.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.preprocessing import STOP_WORDS, expand_query
from src.utils.config_loader import cfg


# ── Combined stop words ───────────────────────────────────────────────────────
# Merge sklearn English stop words with NUST-specific ones from shared module.
# Add policy-structural terms that appear in every chunk and carry no signal.
_EXTRA_STOP_WORDS = STOP_WORDS | {
    "programme", "programmes",
    "course", "courses",
}
_ALL_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(_EXTRA_STOP_WORDS))

# ── Known noise section (page 45 grading table) ───────────────────────────────
# This chunk contains a dense grading table and fires on almost every query.
# Listed here for transparency — can be disabled by clearing the set.
_SECTION_BLOCKLIST: set[str] = set()

# ── Section domain mapping for partition-first search ─────────────────────────
# Maps query keywords to likely section headings. When a query contains
# these keywords, we search the matching partition first (faster).
_SECTION_DOMAIN_MAP: dict[str, list[str]] = {
    "gpa":         ["academic standards", "award of bachelor degree", "academic deficiencies"],
    "cgpa":        ["academic standards", "award of bachelor degree", "academic deficiencies"],
    "attendance":  ["offering and registration", "examinations"],
    "plagiarism":  ["unfair means", "code of conduct", "ill-discipline"],
    "cheating":    ["unfair means", "code of conduct"],
    "repeat":      ["repetition", "retest"],
    "fail":        ["repetition", "academic deficiencies", "warning"],
    "probation":   ["warning", "probation", "suspension"],
    "hostel":      ["living on campus", "hostels"],
    "withdraw":    ["withdrawal", "adding dropping"],
    "drop":        ["adding dropping", "withdrawal"],
    "degree":      ["award of bachelor degree", "academic standards"],
    "graduation":  ["award of bachelor degree", "completion"],
    "transfer":    ["inter-institutional", "change of programme"],
    "scholarship": ["financial assistance"],
    "fee":         ["financial", "tuition"],
    "discipline":  ["code of conduct", "ill-discipline"],
    "grade":       ["academic standards", "offering and registration"],
    "exam":        ["examinations", "end semester"],
}


class TFIDFRetriever:

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer = None   # type: ignore[assignment]
        self.matrix      = None
        self.chunks: list[dict] = []
        # Sparse matrix partitioning structures
        self._partitions:     dict[str, list[int]] = {}
        self._partition_built = False

    # ── Build ──────────────────────────────────────────────────────────────

    def fit(self, chunks: list[dict]) -> None:
        """
        Build TF-IDF index from a list of chunk dicts.

        Args:
            chunks: list of dicts, each with at least id, text, page, section, source.

        Raises:
            ValueError: if chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot fit TFIDFRetriever on an empty chunk list.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range  = tuple(cfg("tfidf", "ngram_range",   default=[1, 2])),
            max_features = cfg("tfidf", "max_features",        default=50_000),
            min_df       = cfg("tfidf", "min_df",              default=1),
            max_df       = cfg("tfidf", "max_df",              default=0.85),
            sublinear_tf = cfg("tfidf", "sublinear_tf",        default=True),
            strip_accents= "unicode",
            analyzer     = "word",
            stop_words   = _ALL_STOP_WORDS,
        )

        print(f"Fitting TF-IDF on {len(texts)} chunks...")
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"Vocabulary size : {len(self.vectorizer.vocabulary_)}")
        print(f"Matrix shape    : {self.matrix.shape}")

        # Build section-level partitions for partition-first search
        self._build_partitions()

    def _build_partitions(self) -> None:
        """
        Build section-domain index for sparse matrix partitioning.

        Groups chunk indices by their section heading, enabling targeted
        searches within a subset of the corpus when the query matches
        a known domain keyword.

        This reduces the effective search space from O(n) to O(n/p)
        where p is the number of partitions.
        """
        self._partitions = {}
        for idx, chunk in enumerate(self.chunks):
            section = chunk.get("section", "").lower()
            for domain, section_keywords in _SECTION_DOMAIN_MAP.items():
                for kw in section_keywords:
                    if kw in section:
                        if domain not in self._partitions:
                            self._partitions[domain] = []
                        self._partitions[domain].append(idx)
                        break  # one domain match per chunk-domain pair

        self._partition_built = True
        total_indexed = sum(len(v) for v in self._partitions.values())
        print(f"Partitions built: {len(self._partitions)} domains, "
              f"{total_indexed} partition entries")

    def _detect_domain(self, query: str) -> str | None:
        """
        Detect query domain from keywords for partition-first search.
        Returns the best-matching domain key or None.
        """
        q_lower = query.lower()
        for domain in _SECTION_DOMAIN_MAP:
            if domain in q_lower:
                return domain
        return None

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer":  self.vectorizer,
                "matrix":      self.matrix,
                "chunks":      self.chunks,
                "partitions":  self._partitions,
            }, f)
        size_mb = path.stat().st_size / 1_048_576
        print(f"Index saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "TFIDFRetriever":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"TF-IDF index not found: {path}\n"
                "Run: python -m src.retrieval.tfidf_retriever "
                "--chunks data/processed/chunks.jsonl "
                "--index data/processed/tfidf_index.pkl"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj             = cls()
        obj.vectorizer  = data["vectorizer"]
        obj.matrix      = data["matrix"]
        obj.chunks      = data["chunks"]
        obj._partitions = data.get("partitions", {})
        obj._partition_built = bool(obj._partitions)
        return obj

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, q: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top_k most relevant chunks for query string q.

        Steps:
            1. Expand query with handbook synonyms (deduplicated).
            2. Vectorise expanded query with fitted TF-IDF.
            3. Cosine similarity against full corpus matrix.
            4. Return top_k chunks (excluding blocklisted sections).

        Returns:
            List of chunk dicts with 'score' and 'latency_ms' added.
        """
        if self.vectorizer is None:
            raise RuntimeError("Index not built. Call fit() or load() first.")

        t0 = time.perf_counter()

        # Expand with deduplicated handbook synonyms
        q_expanded = expand_query(q)
        q_vec      = self.vectorizer.transform([q_expanded])

        # Partition-first search: if query matches a domain, search that
        # partition exclusively first for faster results.
        domain = self._detect_domain(q) if self._partition_built else None

        if domain and domain in self._partitions:
            # Search only the matching partition
            partition_indices = self._partitions[domain]
            partition_matrix  = self.matrix[partition_indices]
            scores = cosine_similarity(q_vec, partition_matrix).flatten()

            # Map back to global indices
            top_local = scores.argsort()[-min(top_k * 3, len(scores)):][::-1]
            top_indices = [partition_indices[i] for i in top_local]
            top_scores  = [scores[i] for i in top_local]

            # If partition results are weak, fall back to full corpus
            if not top_scores or top_scores[0] < 0.05:
                scores     = cosine_similarity(q_vec, self.matrix).flatten()
                top_indices = scores.argsort()[-min(top_k * 3, len(scores)):][::-1]
                top_scores  = [scores[i] for i in top_indices]
        else:
            # Full corpus search (no matching partition)
            scores = cosine_similarity(q_vec, self.matrix).flatten()
            top_indices = scores.argsort()[-min(top_k * 3, len(scores)):][::-1].tolist()
            top_scores  = [scores[i] for i in top_indices]

        latency_ms = (time.perf_counter() - t0) * 1000

        results: list[dict] = []
        for idx, score in zip(top_indices, top_scores):
            chunk = dict(self.chunks[idx])
            # Skip blocklisted sections
            if chunk.get("section", "") in _SECTION_BLOCKLIST:
                continue
            chunk["score"]      = float(score)
            chunk["latency_ms"] = round(latency_ms, 2)
            results.append(chunk)
            if len(results) == top_k:
                break

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

    retriever = TFIDFRetriever()
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
    retriever_loaded = TFIDFRetriever.load(index_path)
    for q in test_queries:
        results = retriever_loaded.query(q, top_k=3)
        print(f"\nQ: {q}")
        for r in results:
            print(f"  score={r['score']:.4f}  page={r['page']:3d}  "
                  f"section={r['section'][:55]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TF-IDF retrieval index")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index",  required=True,
                        default=cfg("paths", "tfidf_index",
                                    default="data/processed/tfidf_index.pkl"))
    args = parser.parse_args()
    main(args.chunks, args.index)