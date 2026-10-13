"""
tests/test_recommender.py

Tests for src/ranking/recommender.py

Critical tests:
  - rerank() applies query expansion before vectorising (the critical fix)
  - apply_diversity_penalty() does NOT mutate the caller's list
  - O(1) dict lookup is used (not O(n) list.index)
  - GPA query correctly surfaces CGPA/2.00 chunk after expansion
"""

import pytest

from tests.conftest import skip_no_reranker, skip_no_tfidf
from src.ranking.recommender import ChunkReranker


class TestChunkRerankerFit:

    def test_fit_builds_matrix(self, sample_chunks):
        r = ChunkReranker()
        r.fit(sample_chunks)
        assert r.matrix is not None
        assert r.matrix.shape[0] == len(sample_chunks)

    def test_fit_empty_raises(self):
        r = ChunkReranker()
        with pytest.raises(ValueError, match="empty"):
            r.fit([])

    def test_id_to_idx_built(self, sample_chunks):
        r = ChunkReranker()
        r.fit(sample_chunks)
        assert len(r._id_to_idx) == len(sample_chunks)
        for c in sample_chunks:
            assert c["id"] in r._id_to_idx

    def test_o1_lookup_not_list_index(self, sample_chunks):
        """Verify dict is used, not list.index()."""
        r = ChunkReranker()
        r.fit(sample_chunks)
        # dict lookup should work and return correct index
        for i, c in enumerate(sample_chunks):
            assert r._id_to_idx[c["id"]] == i


class TestRerank:

    @pytest.fixture(autouse=True)
    def fitted(self, sample_chunks):
        self.r  = ChunkReranker()
        self.r.fit(sample_chunks)
        self.chunks = sample_chunks

    def test_rerank_returns_list(self):
        results = self.r.rerank("GPA requirement", self.chunks[:3])
        assert isinstance(results, list)

    def test_rerank_returns_top_k(self):
        results = self.r.rerank("policy", self.chunks, top_k=3)
        assert len(results) == 3

    def test_result_has_relevance_score(self):
        results = self.r.rerank("cgpa degree", self.chunks, top_k=2)
        for r in results:
            assert "relevance_score" in r

    def test_sorted_descending_by_relevance(self):
        results = self.r.rerank("grade cgpa minimum", self.chunks, top_k=4)
        scores  = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_does_not_mutate_candidates(self):
        """rerank() must return new dicts, not mutate the input list."""
        originals = [dict(c) for c in self.chunks[:3]]
        self.r.rerank("grade cgpa", self.chunks[:3], top_k=2)
        for orig, current in zip(originals, self.chunks[:3]):
            assert orig == current, "rerank() mutated the candidate list"

    def test_empty_candidates_returns_empty(self):
        assert self.r.rerank("any query", [], top_k=5) == []

    def test_out_of_index_chunks_get_score_zero(self):
        unknown = [{"id": "unknown_x", "text": "some text", "page": 1,
                    "section": "S", "source": "t.pdf", "word_count": 2}]
        results = self.r.rerank("query", unknown, top_k=1)
        assert len(results) == 1
        assert results[0]["relevance_score"] == 0.0

    def test_gpa_query_expansion_applied(self):
        """
        CRITICAL BUG FIX: rerank() must expand 'GPA' to 'cgpa', '2.00' etc.
        before vectorising. Without expansion, 'minimum' would match
        'minimum credit hours' over 'CGPA 2.00'.
        """
        results = self.r.rerank("What is the minimum GPA requirement?",
                                 self.chunks, top_k=3)
        top_section = results[0]["section"].lower()
        # After expansion, CGPA/degree/academic chunk should rank higher
        # than a generic "minimum" match
        assert any(
            "cgpa" in r["text"].lower() or "degree" in r["section"].lower()
            or "academic" in r["section"].lower()
            for r in results[:2]
        ), f"GPA query expansion not applied correctly. Top section: {top_section}"

    def test_query_before_fit_raises(self):
        r = ChunkReranker()
        with pytest.raises(RuntimeError, match="not built"):
            r.rerank("query", [])


class TestDiversityPenalty:

    @pytest.fixture(autouse=True)
    def fitted(self, sample_chunks):
        self.r      = ChunkReranker()
        self.r.fit(sample_chunks)
        self.chunks = sample_chunks

    def test_returns_new_list(self):
        reranked = self.r.rerank("grade cgpa", self.chunks, top_k=4)
        original = [dict(c) for c in reranked]
        result   = self.r.apply_diversity_penalty(reranked)
        # Result is a different list object
        assert result is not reranked

    def test_does_not_mutate_input(self):
        """
        CRITICAL BUG FIX: apply_diversity_penalty() must not mutate caller's list.
        Previously modified relevance_score in-place on the passed list.
        """
        reranked = self.r.rerank("grade cgpa", self.chunks, top_k=4)
        scores_before = [r["relevance_score"] for r in reranked]
        self.r.apply_diversity_penalty(reranked)
        scores_after  = [r["relevance_score"] for r in reranked]
        assert scores_before == scores_after, (
            "apply_diversity_penalty() mutated the input list's scores"
        )

    def test_idempotent_on_original(self):
        """Calling twice with the original list should give same result each time."""
        reranked = self.r.rerank("grade cgpa", self.chunks, top_k=4)
        result1  = self.r.apply_diversity_penalty(reranked)
        result2  = self.r.apply_diversity_penalty(reranked)
        assert [r["relevance_score"] for r in result1] == \
               [r["relevance_score"] for r in result2]

    def test_duplicate_sections_penalised(self, sample_chunks):
        """Chunks from the same section should get lower scores on second occurrence."""
        # Create chunks with duplicate sections
        dup_chunks = [
            {"id": "a", "text": "cgpa grade minimum requirement",
             "page": 1, "section": "GPA Rules", "source": "t.pdf", "word_count": 4,
             "relevance_score": 0.9},
            {"id": "b", "text": "cgpa grade point 2.00 cumulative",
             "page": 2, "section": "GPA Rules", "source": "t.pdf", "word_count": 5,
             "relevance_score": 0.8},
            {"id": "c", "text": "attendance 75 percent xf grade awarded",
             "page": 3, "section": "Attendance", "source": "t.pdf", "word_count": 5,
             "relevance_score": 0.7},
        ]
        result = self.r.apply_diversity_penalty(dup_chunks, penalty=0.15)
        # Second "GPA Rules" chunk (id="b") should be penalised
        b_score = next(r["relevance_score"] for r in result if r["id"] == "b")
        assert b_score < 0.8, f"Duplicate section not penalised: score={b_score}"

    def test_sorted_after_penalty(self, sample_chunks):
        reranked = self.r.rerank("grade cgpa", self.chunks, top_k=4)
        result   = self.r.apply_diversity_penalty(reranked)
        scores   = [r["relevance_score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestRerankerPersistence:

    def test_save_load_roundtrip(self, sample_chunks, tmp_path):
        r = ChunkReranker()
        r.fit(sample_chunks)
        path = str(tmp_path / "reranker_test.pkl")
        r.save(path)
        loaded = ChunkReranker.load(path)
        assert len(loaded.chunk_ids) == len(sample_chunks)
        # _id_to_idx rebuilt on load
        assert len(loaded._id_to_idx) == len(sample_chunks)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ChunkReranker.load(str(tmp_path / "missing.pkl"))


@skip_no_reranker
@skip_no_tfidf
class TestRerankerIntegration:

    @pytest.fixture(autouse=True)
    def load_real(self):
        from src.retrieval.tfidf_retriever import TFIDFRetriever
        self.tfidf    = TFIDFRetriever.load("data/processed/tfidf_index.pkl")
        self.reranker = ChunkReranker.load("data/processed/reranker_index.pkl")

    def test_two_stage_pipeline_returns_top_k(self):
        candidates = self.tfidf.query("minimum GPA requirement", top_k=10)
        reranked   = self.reranker.rerank("minimum GPA requirement",
                                           candidates, top_k=5)
        assert len(reranked) == 5

    def test_diversity_penalty_does_not_crash(self):
        candidates = self.tfidf.query("academic probation warning", top_k=10)
        reranked   = self.reranker.rerank("academic probation warning",
                                           candidates, top_k=5)
        final      = self.reranker.apply_diversity_penalty(reranked)
        assert len(final) > 0

    def test_gpa_reranked_finds_cgpa_chunk(self):
        candidates = self.tfidf.query("What is the minimum GPA requirement?", top_k=10)
        reranked   = self.reranker.rerank(
            "What is the minimum GPA requirement?", candidates, top_k=5
        )
        # After fix: CGPA/2.00/Academic Standards should appear in top 2
        texts = [r["text"].lower() for r in reranked[:2]]
        assert any("cgpa" in t or "2.00" in t for t in texts), \
            f"GPA reranking failed. Top 2 texts: {[t[:80] for t in texts]}"