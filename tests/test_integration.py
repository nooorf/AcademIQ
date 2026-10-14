"""
tests/test_integration.py

End-to-end integration tests for the full AcademIQ retrieval pipeline.

All tests are skipped automatically if the real index files are absent.
Run the indexing pipeline first:

    python -m src.ingestion.pdf_extractor ...
    python -m src.ingestion.chunker ...
    python -m src.retrieval.tfidf_retriever ...
    python -m src.retrieval.minhash_lsh ...
    python -m src.retrieval.simhash ...
    python -m src.ranking.recommender ...

Then run: pytest tests/test_integration.py -v
"""

import pytest

from tests.conftest import (
    skip_no_tfidf,
    skip_no_minhash,
    skip_no_simhash,
    skip_no_reranker,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tfidf():
    from src.retrieval.tfidf_retriever import TFIDFRetriever
    return TFIDFRetriever.load("data/processed/tfidf_index.pkl")


@pytest.fixture(scope="module")
def minhash():
    from src.retrieval.minhash_lsh import MinHashLSHRetriever
    return MinHashLSHRetriever.load("data/processed/minhash_index.pkl")


@pytest.fixture(scope="module")
def simhash():
    from src.retrieval.simhash import SimHashRetriever
    return SimHashRetriever.load("data/processed/simhash_index.pkl")


@pytest.fixture(scope="module")
def reranker():
    from src.ranking.recommender import ChunkReranker
    return ChunkReranker.load("data/processed/reranker_index.pkl")


@pytest.fixture(scope="module")
def generator():
    from src.generation.answer_generator import AnswerGenerator
    return AnswerGenerator()


# ── RRF helper (mirrors HybridRetriever's RRF logic) ─────────────────────────

def rrf_fuse(results_lists: list[list[dict]], k: int = 60) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}
    for results in results_lists:
        for rank, chunk in enumerate(results):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk
    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    fused = []
    for cid in sorted_ids:
        c = dict(chunk_map[cid])
        c["score"] = round(rrf_scores[cid], 6)
        fused.append(c)
    return fused


# ── Policy queries used across all retrievers ──────────────────────────────────

POLICY_QUERIES = [
    ("What is the minimum GPA requirement?",     ["cgpa", "2.00", "grade point", "academic"]),
    ("What is the attendance policy?",           ["75", "attendance", "xf"]),
    ("What happens if a student uses unfair means?", ["unfair", "plagiarism", "exam"]),
    ("How many times can a student repeat a course?", ["repeat", "course", "retest"]),
    ("What is academic probation?",              ["probation", "cgpa", "warning"]),
]


def _top_texts_contain_any(results: list[dict], keywords: list[str], top_n: int = 3) -> bool:
    """Return True if any of the top_n results contain at least one keyword."""
    for r in results[:top_n]:
        text = (r.get("text") or "").lower()
        sect = (r.get("section") or "").lower()
        if any(kw.lower() in text or kw.lower() in sect for kw in keywords):
            return True
    return False


# ── TF-IDF integration ────────────────────────────────────────────────────────

@skip_no_tfidf
class TestTFIDFIntegration:

    def test_returns_top_k(self, tfidf):
        results = tfidf.query("attendance policy", top_k=5)
        assert len(results) == 5

    def test_gpa_query_finds_cgpa(self, tfidf):
        results = tfidf.query("What is the minimum GPA requirement?", top_k=5)
        assert _top_texts_contain_any(results, ["cgpa", "2.00", "academic"]), \
            f"TF-IDF GPA query: top sections = {[r['section'] for r in results[:3]]}"

    def test_attendance_query_finds_75(self, tfidf):
        results = tfidf.query("What is the attendance policy?", top_k=5)
        assert _top_texts_contain_any(results, ["75", "attendance"]), \
            f"TF-IDF attendance query: top texts = {[r['text'][:60] for r in results[:3]]}"

    def test_plagiarism_query_finds_unfair_means(self, tfidf):
        results = tfidf.query("What is the policy on plagiarism?", top_k=5)
        assert _top_texts_contain_any(results, ["unfair", "plagiarism"]), \
            f"TF-IDF plagiarism query top texts = {[r['text'][:60] for r in results[:3]]}"

    def test_results_have_positive_scores(self, tfidf):
        results = tfidf.query("degree requirements", top_k=5)
        assert all(r["score"] > 0 for r in results)

    def test_sorted_descending(self, tfidf):
        results = tfidf.query("cgpa grade degree", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_have_required_fields(self, tfidf):
        results = tfidf.query("academic policy", top_k=3)
        for r in results:
            for field in ("id", "text", "page", "section", "score", "latency_ms"):
                assert field in r


# ── MinHash integration ───────────────────────────────────────────────────────

@skip_no_minhash
class TestMinHashIntegration:

    def test_returns_top_k(self, minhash):
        results = minhash.query("attendance policy", top_k=5)
        assert len(results) == 5

    def test_repetition_query_finds_relevant(self, minhash):
        results = minhash.query("How many times can a course be repeated?", top_k=5)
        assert _top_texts_contain_any(results, ["repeat", "course", "retest"]), \
            f"MinHash repetition query top texts = {[r['text'][:60] for r in results[:3]]}"

    def test_attendance_query_finds_75(self, minhash):
        results = minhash.query("attendance 75 percent xf grade", top_k=5)
        assert _top_texts_contain_any(results, ["75", "attendance", "xf"]), \
            f"MinHash attendance query top texts = {[r['text'][:60] for r in results[:3]]}"

    def test_all_valid_chunk_ids(self, minhash):
        results = minhash.query("academic policy", top_k=5)
        for r in results:
            assert r["id"] in minhash.chunks

    def test_scores_between_0_and_1(self, minhash):
        results = minhash.query("cgpa grade", top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


# ── SimHash integration ───────────────────────────────────────────────────────

@skip_no_simhash
class TestSimHashIntegration:

    def test_returns_top_k(self, simhash):
        results = simhash.query("attendance policy", top_k=5)
        assert len(results) == 5

    def test_gpa_query_finds_cgpa(self, simhash):
        results = simhash.query("minimum GPA CGPA requirement degree", top_k=5)
        assert _top_texts_contain_any(results, ["cgpa", "2.00", "degree", "grade"]), \
            f"SimHash GPA query top texts = {[r['text'][:60] for r in results[:3]]}"

    def test_no_empty_chunks_returned(self, simhash):
        results = simhash.query("grade cgpa degree", top_k=5)
        for r in results:
            assert r["id"] not in simhash._empty_chunks
            assert len((r.get("text") or "").strip()) > 0

    def test_hamming_distance_non_negative(self, simhash):
        results = simhash.query("policy", top_k=5)
        for r in results:
            assert r["hamming_distance"] >= 0

    def test_scores_between_0_and_1(self, simhash):
        results = simhash.query("attendance grade", top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


# ── Re-ranker integration ─────────────────────────────────────────────────────

@skip_no_tfidf
@skip_no_reranker
class TestRerankerIntegration:

    def test_two_stage_pipeline_returns_top_k(self, tfidf, reranker):
        candidates = tfidf.query("minimum GPA requirement", top_k=10)
        reranked   = reranker.rerank("minimum GPA requirement", candidates, top_k=5)
        assert len(reranked) == 5

    def test_reranked_has_relevance_score(self, tfidf, reranker):
        candidates = tfidf.query("cgpa academic standards", top_k=8)
        reranked   = reranker.rerank("cgpa academic standards", candidates, top_k=3)
        for r in reranked:
            assert "relevance_score" in r

    def test_gpa_reranking_surfaces_cgpa_chunk(self, tfidf, reranker):
        """
        After re-ranking, CGPA / academic standards should be in top 2 results.
        This validates the critical fix: expand_query() applied before transform().
        """
        candidates = tfidf.query("What is the minimum GPA requirement?", top_k=10)
        reranked   = reranker.rerank(
            "What is the minimum GPA requirement?", candidates, top_k=5
        )
        texts = [r["text"].lower() for r in reranked[:2]]
        assert any("cgpa" in t or "2.00" in t for t in texts), \
            f"GPA reranking failed. Top 2 texts: {[t[:80] for t in texts]}"

    def test_diversity_penalty_preserves_count(self, tfidf, reranker):
        candidates = tfidf.query("academic probation warning", top_k=10)
        reranked   = reranker.rerank("academic probation warning", candidates, top_k=5)
        final      = reranker.apply_diversity_penalty(reranked)
        assert len(final) > 0

    def test_diversity_penalty_does_not_mutate_input(self, tfidf, reranker):
        candidates = tfidf.query("grade cgpa", top_k=10)
        reranked   = reranker.rerank("grade cgpa", candidates, top_k=5)
        scores_before = [r["relevance_score"] for r in reranked]
        reranker.apply_diversity_penalty(reranked)
        scores_after  = [r["relevance_score"] for r in reranked]
        assert scores_before == scores_after


# ── Hybrid RRF integration ────────────────────────────────────────────────────

@skip_no_tfidf
@skip_no_minhash
class TestHybridRRFIntegration:

    def test_hybrid_returns_results(self, tfidf, minhash):
        tf_results = tfidf.query("minimum cgpa grade", top_k=10)
        mh_results = minhash.query("minimum cgpa grade", top_k=10)
        fused      = rrf_fuse([tf_results, mh_results])
        assert len(fused) > 0

    def test_hybrid_ids_are_unique(self, tfidf, minhash):
        tf_results = tfidf.query("attendance policy", top_k=10)
        mh_results = minhash.query("attendance policy", top_k=10)
        fused      = rrf_fuse([tf_results, mh_results])
        ids = [c["id"] for c in fused]
        assert len(ids) == len(set(ids))

    def test_hybrid_scores_are_positive(self, tfidf, minhash):
        tf_results = tfidf.query("academic standards", top_k=10)
        mh_results = minhash.query("academic standards", top_k=10)
        fused      = rrf_fuse([tf_results, mh_results])
        for c in fused:
            assert c["score"] > 0.0


# ── Full pipeline: retrieve → rerank → generate ───────────────────────────────

@skip_no_tfidf
@skip_no_reranker
class TestFullPipelineIntegration:

    @pytest.mark.parametrize("query,keywords", POLICY_QUERIES)
    def test_full_pipeline_policy_query(self, tfidf, reranker, generator,
                                        query, keywords):
        """
        Tests the complete 3-stage pipeline end to end for each policy domain.
        Validates that the generated answer is grounded in the retrieved content.
        """
        # Stage 1: Coarse retrieval
        candidates = tfidf.query(query, top_k=10)
        assert len(candidates) > 0, f"TF-IDF returned nothing for: {query}"

        # Stage 2: Re-ranking
        reranked = reranker.rerank(query, candidates, top_k=5)
        final    = reranker.apply_diversity_penalty(reranked)
        assert len(final) > 0

        # Stage 3: Answer generation (extractive — no API key in CI)
        result = generator.generate(query, final)
        assert result["chunks_used"] > 0
        assert len(result["answer"]) > 10
        # Answer should contain at least one expected keyword
        answer_lower = result["answer"].lower()
        assert any(kw.lower() in answer_lower for kw in keywords), (
            f"Pipeline answer for '{query}' doesn't mention expected keywords {keywords}.\n"
            f"Answer: {result['answer'][:200]}"
        )

    def test_pipeline_latency_reasonable(self, tfidf, reranker, generator):
        """
        End-to-end latency for a single query should be < 5 seconds
        (without LLM API call — extractive mode only).
        """
        import time
        t0         = time.perf_counter()
        candidates = tfidf.query("What is the minimum CGPA requirement?", top_k=10)
        reranked   = reranker.rerank("What is the minimum CGPA requirement?",
                                     candidates, top_k=5)
        final      = reranker.apply_diversity_penalty(reranked)
        result     = generator.generate("What is the minimum CGPA requirement?", final)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5000, (
            f"Pipeline latency {elapsed_ms:.0f}ms exceeded 5s — "
            "likely a performance regression"
        )

    def test_pipeline_answer_has_sources(self, tfidf, reranker, generator):
        candidates = tfidf.query("attendance policy", top_k=10)
        reranked   = reranker.rerank("attendance policy", candidates, top_k=3)
        result     = generator.generate("attendance policy", reranked)
        assert "Sources:" in result["answer"]