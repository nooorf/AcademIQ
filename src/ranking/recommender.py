"""
src/ranking/recommender.py

Recommendation-based chunk re-ranker (competitive edge extension).

Two-stage pipeline:
    Stage 1 — Coarse retrieval  : MinHash / SimHash / TF-IDF → top_k * 2 candidates
    Stage 2 — Fine re-ranking   : ChunkReranker              → top_k final chunks


Usage:
    python -m src.ranking.recommender \
        --chunks data/processed/chunks.jsonl \
        --index  data/processed/reranker_index.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.preprocessing import expand_query, STOP_WORDS
from src.utils.config_loader import cfg

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Reranker stop words — same approach as tfidf_retriever.py
_RERANKER_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(STOP_WORDS).union({
    "university", "programme", "programmes",
    "course", "courses", "academic",
}))


class ChunkReranker:

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer      = None   # type: ignore[assignment]
        self.matrix                           = None
        self.chunk_ids:  list[str]            = []
        self._id_to_idx: dict[str, int]       = {}     # O(1) lookup
        self.chunks:     dict[str, dict]      = {}

    # ── Build ──────────────────────────────────────────────────────────────

    def fit(self, chunks: list[dict]) -> None:
        """
        Build a secondary TF-IDF matrix over all chunk texts.

        This is separate from the retrieval TF-IDF index — its job is to score
        query-chunk relevance at re-ranking time using the expanded query.

        Args:
            chunks: list of chunk dicts (id, text, page, section, source).
        """
        if not chunks:
            raise ValueError("Cannot fit ChunkReranker on an empty chunk list.")

        self.chunks     = {c["id"]: c for c in chunks}
        self.chunk_ids  = [c["id"]   for c in chunks]
        # FIX: O(1) dict lookup replacing O(n) list.index()
        self._id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        texts           = [c["text"] for c in chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range =(1, 2),
            max_features=30_000,
            min_df      =1,
            max_df      =0.85,          # FIX: was 0.95 — too permissive
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words  =_RERANKER_STOP_WORDS,  # FIX: was missing entirely
        )

        print(f"Fitting re-ranker on {len(texts)} chunks...")
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"Re-ranker vocabulary : {len(self.vectorizer.vocabulary_)}")
        print(f"Re-ranker matrix     : {self.matrix.shape}")

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix":     self.matrix,
                "chunk_ids":  self.chunk_ids,
                "_id_to_idx": self._id_to_idx,
                "chunks":     self.chunks,
            }, f)
        size_mb = path.stat().st_size / 1_048_576
        print(f"Re-ranker index saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "ChunkReranker":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Re-ranker index not found: {path}\n"
                "Run: python -m src.ranking.recommender "
                "--chunks data/processed/chunks.jsonl "
                "--index data/processed/reranker_index.pkl"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj             = cls()
        obj.vectorizer  = data["vectorizer"]
        obj.matrix      = data["matrix"]
        obj.chunk_ids   = data["chunk_ids"]
        obj.chunks      = data["chunks"]
        # Rebuild dict if loading an index saved before this fix
        obj._id_to_idx  = data.get(
            "_id_to_idx",
            {cid: idx for idx, cid in enumerate(obj.chunk_ids)}
        )
        print(f"Re-ranker loaded — {len(obj.chunk_ids)} chunks in index")
        return obj

    # ── Re-rank ────────────────────────────────────────────────────────────

    def rerank(self, query: str,
               candidate_chunks: list[dict],
               top_k: int = 5) -> list[dict]:
        """
        Score each candidate against the query using cosine similarity on the
        re-ranker's TF-IDF matrix.

        FIX: Query is expanded with handbook synonyms BEFORE vectorisation.
        Previously the raw query was used, causing "minimum GPA" to match
        "minimum credit hours" (page 36) instead of "CGPA 2.00" (page 31).

        Returns a NEW list of chunk dicts (copies) sorted by relevance_score
        descending. Does NOT mutate candidate_chunks.
        """
        if self.vectorizer is None:
            raise RuntimeError("Re-ranker not built. Call fit() or load() first.")

        if not candidate_chunks:
            return []

        # FIX: expand query before transform (was using raw query)
        q_expanded = expand_query(query)
        q_vec      = self.vectorizer.transform([q_expanded])

        # FIX: O(1) dict lookup
        valid: list[tuple[str, int]] = []
        for c in candidate_chunks:
            cid = c["id"]
            if cid in self._id_to_idx:
                valid.append((cid, self._id_to_idx[cid]))

        if not valid:
            # Synthetic / out-of-index chunks — return copies with score 0
            result = [dict(c) for c in candidate_chunks]
            for c in result:
                c["relevance_score"] = 0.0
            return result[:top_k]

        row_indices  = [idx for _, idx in valid]
        cand_matrix  = self.matrix[row_indices]
        scores       = cosine_similarity(q_vec, cand_matrix).flatten()

        scored_chunks: list[dict] = []
        for (cid, _), score in zip(valid, scores):
            chunk = dict(self.chunks[cid])           # copy — never mutate index
            chunk["relevance_score"] = round(float(score), 4)
            scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_chunks[:top_k]

    def apply_diversity_penalty(self,
                                chunks: list[dict],
                                penalty: float = None) -> list[dict]:   # type: ignore[assignment]
        """
        Penalise consecutive chunks from the same section.

        FIX: Returns a NEW sorted list (deep-copies chunks) — previously
        mutated the caller's list in-place, breaking idempotency and
        corrupting stored session-state scores in the Streamlit app.

        Args:
            chunks  : list of chunk dicts already scored by rerank().
            penalty : score deduction per repeated section occurrence.

        Returns:
            NEW list with diversity-adjusted relevance_score, sorted descending.
        """
        if penalty is None:
            penalty = cfg("recommender", "diversity_penalty", default=0.15)

        # Deep-copy so caller's list is never touched
        result: list[dict] = [dict(c) for c in chunks]

        seen: dict[str, int] = {}
        for chunk in result:
            section = chunk.get("section", "")
            count   = seen.get(section, 0)
            if count > 0:
                chunk["relevance_score"] = round(
                    chunk["relevance_score"] - (penalty * count), 4
                )
            seen[section] = count + 1

        result.sort(key=lambda x: x["relevance_score"], reverse=True)
        return result


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

    reranker = ChunkReranker()
    reranker.fit(chunks)
    reranker.save(index_path)

    print("\n--- Smoke test: two-stage pipeline ---")
    from src.retrieval.tfidf_retriever import TFIDFRetriever
    tfidf_path = cfg("paths", "tfidf_index", "data/processed/tfidf_index.pkl")
    tfidf      = TFIDFRetriever.load(tfidf_path)
    reranker_loaded = ChunkReranker.load(index_path)

    test_queries = [
        "What is the minimum GPA requirement?",
        "What happens if a student fails a course?",
        "What is the attendance policy?",
        "How many times can a course be repeated?",
        "What is the policy on plagiarism?",
    ]
    for q in test_queries:
        candidates = tfidf.query(q, top_k=10)
        reranked   = reranker_loaded.rerank(q, candidates, top_k=5)
        reranked   = reranker_loaded.apply_diversity_penalty(reranked)
        print(f"\nQ: {q}")
        for r in reranked:
            print(f"  relevance={r['relevance_score']:.4f}  "
                  f"page={r['page']:3d}  "
                  f"section={r['section'][:55]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build recommendation re-ranker index")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index",  required=True,
                        default=cfg("paths", "reranker_index",
                                    default="data/processed/reranker_index.pkl"))
    args = parser.parse_args()
    main(args.chunks, args.index)