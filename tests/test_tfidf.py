"""
tests/test_tfidf.py

Tests for src/retrieval/tfidf_retriever.py

Unit tests use synthetic chunks (no index file needed).
Integration tests load the real index and are skipped if it's absent.
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from tests.conftest import skip_no_tfidf
from src.retrieval.tfidf_retriever import TFIDFRetriever


# ── Unit tests — no index file required ───────────────────────────────────────

class TestTFIDFFit:

    def test_fit_builds_vocabulary(self, sample_chunks):
        r = TFIDFRetriever()
        r.fit(sample_chunks)
        assert r.vectorizer is not None
        assert len(r.vectorizer.vocabulary_) > 0

    def test_fit_empty_raises(self):
        r = TFIDFRetriever()
        with pytest.raises(ValueError, match="empty"):
            r.fit([])

    def test_matrix_shape_matches_chunks(self, sample_chunks):
        r = TFIDFRetriever()
        r.fit(sample_chunks)
        assert r.matrix.shape[0] == len(sample_chunks)

    def test_chunks_stored(self, sample_chunks):
        r = TFIDFRetriever()
        r.fit(sample_chunks)
        assert len(r.chunks) == len(sample_chunks)


class TestTFIDFQuery:

    @pytest.fixture(autouse=True)
    def fitted_retriever(self, sample_chunks):
        self.r = TFIDFRetriever()
        self.r.fit(sample_chunks)

    def test_query_returns_list(self):
        results = self.r.query("What is the GPA requirement?")
        assert isinstance(results, list)

    def test_query_returns_top_k(self):
        for k in [1, 2, 3]:
            results = self.r.query("attendance policy", top_k=k)
            assert len(results) == k

    def test_query_top_k_larger_than_corpus(self, sample_chunks):
        results = self.r.query("policy", top_k=len(sample_chunks) + 10)
        assert len(results) == len(sample_chunks)

    def test_result_has_required_fields(self):
        results = self.r.query("CGPA requirement", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "id"         in r
        assert "text"       in r
        assert "page"       in r
        assert "section"    in r
        assert "score"      in r
        assert "latency_ms" in r

    def test_scores_are_non_negative(self):
        results = self.r.query("attendance", top_k=3)
        for r in results:
            assert r["score"] >= 0.0

    def test_results_sorted_descending(self):
        results = self.r.query("CGPA grade point", top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_gpa_query_finds_cgpa_chunk(self):
        """Expansion: 'GPA' → 'cgpa', '2.00' — should find the CGPA chunk."""
        results = self.r.query("What is the minimum GPA requirement?", top_k=3)
        sections = [r["section"].lower() for r in results]
        assert any("cgpa" in s or "degree" in s or "academic" in s or "standard" in s
                   for s in sections), \
            "GPA query should surface CGPA/degree-related chunks"

    def test_attendance_query_finds_attendance_chunk(self):
        results = self.r.query("What is the attendance policy?", top_k=3)
        texts   = [r["text"].lower() for r in results]
        assert any("75" in t or "attendance" in t for t in texts), \
            "Attendance query should find the 75% attendance chunk"

    def test_plagiarism_query_finds_unfair_means(self):
        results = self.r.query("What is the plagiarism policy?", top_k=3)
        texts   = [r["text"].lower() for r in results]
        assert any("unfair" in t or "plagiarism" in t for t in texts), \
            "Plagiarism query should find the unfair means chunk"

    def test_query_before_fit_raises(self):
        r = TFIDFRetriever()
        with pytest.raises(RuntimeError, match="not built"):
            r.query("any query")


class TestTFIDFPersistence:

    def test_save_and_load_roundtrip(self, sample_chunks, tmp_path):
        r = TFIDFRetriever()
        r.fit(sample_chunks)
        path = str(tmp_path / "tfidf_test.pkl")
        r.save(path)

        loaded = TFIDFRetriever.load(path)
        assert len(loaded.chunks) == len(sample_chunks)
        assert len(loaded.vectorizer.vocabulary_) > 0

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TFIDFRetriever.load(str(tmp_path / "nonexistent.pkl"))

    def test_loaded_query_matches_original(self, sample_chunks, tmp_path):
        r = TFIDFRetriever()
        r.fit(sample_chunks)
        path = str(tmp_path / "tfidf_test.pkl")
        r.save(path)

        loaded  = TFIDFRetriever.load(path)
        results_orig   = r.query("CGPA requirement", top_k=3)
        results_loaded = loaded.query("CGPA requirement", top_k=3)
        orig_ids   = [x["id"] for x in results_orig]
        loaded_ids = [x["id"] for x in results_loaded]
        assert orig_ids == loaded_ids


# ── Integration tests — require real index ────────────────────────────────────

@skip_no_tfidf
class TestTFIDFIntegration:

    @pytest.fixture(autouse=True)
    def load_real_index(self):
        self.r = TFIDFRetriever.load("data/processed/tfidf_index.pkl")

    def test_index_has_chunks(self):
        assert len(self.r.chunks) > 0

    def test_gpa_top_result_relevant(self):
        results = self.r.query("What is the minimum GPA requirement?", top_k=5)
        assert len(results) > 0
        sections = [r["section"].lower() for r in results]
        assert any(
            "academic" in s or "degree" in s or "cgpa" in s.lower() or "standard" in s
            for s in sections
        ), f"Top results for GPA query: {sections}"

    def test_attendance_top_result_contains_75(self):
        results = self.r.query("What is the attendance policy?", top_k=5)
        assert any("75" in r["text"] for r in results), \
            "Attendance query should surface a chunk containing '75'"

    def test_plagiarism_finds_unfair_means(self):
        results = self.r.query("What is the policy on plagiarism?", top_k=5)
        assert any("unfair" in r["text"].lower() for r in results), \
            "Plagiarism query must find 'unfair means' section"

    def test_all_results_have_positive_score(self):
        results = self.r.query("degree requirements graduation", top_k=5)
        assert all(r["score"] > 0 for r in results)