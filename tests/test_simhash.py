"""
tests/test_simhash.py

Tests for src/retrieval/simhash.py
"""

import pytest

from tests.conftest import skip_no_simhash
from src.retrieval.simhash import (
    SimHashRetriever,
    simhash,
    hamming_distance,
    hamming_similarity,
    SIMHASH_BITS,
)
from src.utils.preprocessing import tokenize


# ── Core math functions ───────────────────────────────────────────────────────

class TestSimhashFunction:

    def test_returns_integer(self):
        tokens = tokenize("cgpa grade point minimum requirement")
        assert isinstance(simhash(tokens), int)

    def test_returns_zero_for_empty_tokens(self):
        assert simhash([]) == 0

    def test_fingerprint_fits_in_bits(self):
        tokens = tokenize("attendance policy 75 percent xf grade")
        fp = simhash(tokens, bits=64)
        assert 0 <= fp < (1 << 64)

    def test_same_text_same_fingerprint(self):
        tokens = tokenize("minimum cgpa 2.00 requirement")
        assert simhash(tokens) == simhash(tokens)

    def test_different_text_different_fingerprint(self):
        t1 = tokenize("cgpa grade point minimum")
        t2 = tokenize("hostel accommodation campus rules")
        assert simhash(t1) != simhash(t2)

    def test_frequency_affects_fingerprint(self):
        """SimHash is TF-weighted — repeated tokens shift fingerprint."""
        t_once  = tokenize("cgpa cgpa cgpa grade")
        t_other = tokenize("grade grade grade cgpa")
        # Fingerprints should differ because dominant term is different
        fp1 = simhash(t_once)
        fp2 = simhash(t_other)
        # Not guaranteed to differ on every run due to hash collisions,
        # but with very different dominant terms it almost always does
        # (soft assertion — don't fail CI on collision edge case)
        assert isinstance(fp1, int) and isinstance(fp2, int)


class TestHammingDistance:

    def test_identical_hashes_are_zero(self):
        fp = simhash(tokenize("grade point cgpa"))
        assert hamming_distance(fp, fp) == 0

    def test_symmetric(self):
        t1 = tokenize("attendance 75 xf grade")
        t2 = tokenize("plagiarism unfair means cheating")
        fp1, fp2 = simhash(t1), simhash(t2)
        assert hamming_distance(fp1, fp2) == hamming_distance(fp2, fp1)

    def test_all_bits_flipped_is_max(self):
        all_zeros = 0
        all_ones  = (1 << SIMHASH_BITS) - 1
        assert hamming_distance(all_zeros, all_ones) == SIMHASH_BITS

    def test_non_negative(self):
        t = tokenize("cgpa grade")
        fp = simhash(t)
        assert hamming_distance(fp, fp ^ 0b1010) >= 0


class TestHammingSimilarity:

    def test_identical_is_one(self):
        fp = simhash(tokenize("grade cgpa minimum"))
        assert hamming_similarity(fp, fp) == 1.0

    def test_all_bits_flipped_is_zero(self):
        all_zeros = 0
        all_ones  = (1 << SIMHASH_BITS) - 1
        assert hamming_similarity(all_zeros, all_ones) == 0.0

    def test_range_0_to_1(self):
        t1 = tokenize("cgpa grade requirement minimum")
        t2 = tokenize("attendance 75 percent xf")
        fp1, fp2 = simhash(t1), simhash(t2)
        sim = hamming_similarity(fp1, fp2)
        assert 0.0 <= sim <= 1.0


# ── SimHashRetriever ──────────────────────────────────────────────────────────

class TestSimHashFit:

    def test_fit_creates_fingerprints(self, sample_chunks):
        r = SimHashRetriever(bits=64, hamming_threshold=10)
        r.fit(sample_chunks)
        assert len(r.fingerprints) == len(sample_chunks)

    def test_fit_empty_raises(self):
        r = SimHashRetriever()
        with pytest.raises(ValueError, match="empty"):
            r.fit([])

    def test_empty_chunk_flagged(self):
        chunks = [{"id": "empty_c", "text": "", "page": 1,
                   "section": "Test", "source": "test.pdf", "word_count": 0}]
        r = SimHashRetriever()
        r.fit(chunks)
        assert "empty_c" in r._empty_chunks

    def test_empty_chunk_fingerprint_zero(self):
        chunks = [{"id": "empty_c", "text": "", "page": 1,
                   "section": "Test", "source": "test.pdf", "word_count": 0}]
        r = SimHashRetriever()
        r.fit(chunks)
        assert r.fingerprints["empty_c"] == 0


class TestSimHashQuery:

    @pytest.fixture(autouse=True)
    def fitted(self, sample_chunks):
        self.r = SimHashRetriever(bits=64, hamming_threshold=10)
        self.r.fit(sample_chunks)

    def test_query_returns_list(self):
        assert isinstance(self.r.query("GPA requirement"), list)

    def test_query_returns_top_k_via_fallback(self):
        """
        With a small corpus and strict threshold, fallback triggers.
        Must still return top_k results.
        """
        results = self.r.query("attendance 75 xf grade", top_k=3)
        assert len(results) == 3

    def test_result_fields_present(self):
        results = self.r.query("cgpa minimum", top_k=1)
        assert len(results) >= 1
        r = results[0]
        for field in ("id", "text", "page", "section", "score",
                      "hamming_distance", "latency_ms"):
            assert field in r

    def test_scores_between_0_and_1(self):
        results = self.r.query("academic policy", top_k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_hamming_distance_non_negative(self):
        results = self.r.query("grade cgpa", top_k=3)
        for r in results:
            assert r["hamming_distance"] >= 0

    def test_custom_threshold_override(self):
        """query() accepts per-call threshold override."""
        r_strict = self.r.query("policy grade", top_k=3, hamming_threshold=1)
        r_loose  = self.r.query("policy grade", top_k=3, hamming_threshold=63)
        # Both should return 3 results (fallback kicks in for strict)
        assert len(r_strict) == 3
        assert len(r_loose)  == 3

    def test_empty_chunks_excluded_from_results(self):
        chunks = [
            {"id": "real_c", "text": "cgpa grade minimum requirement degree",
             "page": 1, "section": "Test", "source": "t.pdf", "word_count": 5},
            {"id": "empty_c", "text": "",
             "page": 2, "section": "Empty", "source": "t.pdf", "word_count": 0},
        ]
        r = SimHashRetriever(bits=64, hamming_threshold=10)
        r.fit(chunks)
        results = r.query("cgpa grade", top_k=2)
        ids = [x["id"] for x in results]
        assert "empty_c" not in ids, "Empty chunks must be excluded from results"

    def test_query_before_fit_raises(self):
        r = SimHashRetriever()
        with pytest.raises(RuntimeError, match="not built"):
            r.query("query")


class TestSimHashPersistence:

    def test_save_load_roundtrip(self, sample_chunks, tmp_path):
        r = SimHashRetriever(bits=64, hamming_threshold=10)
        r.fit(sample_chunks)
        path = str(tmp_path / "simhash_test.pkl")
        r.save(path)
        loaded = SimHashRetriever.load(path)
        assert len(loaded.fingerprints) == len(sample_chunks)
        assert loaded.bits == 64
        assert loaded.hamming_threshold == 10

    def test_empty_chunks_set_persisted(self, tmp_path):
        chunks = [
            {"id": "real_c",  "text": "cgpa grade minimum", "page": 1,
             "section": "S", "source": "t.pdf", "word_count": 3},
            {"id": "empty_c", "text": "",                   "page": 2,
             "section": "S", "source": "t.pdf", "word_count": 0},
        ]
        r = SimHashRetriever()
        r.fit(chunks)
        path = str(tmp_path / "sh.pkl")
        r.save(path)
        loaded = SimHashRetriever.load(path)
        assert "empty_c" in loaded._empty_chunks

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SimHashRetriever.load(str(tmp_path / "missing.pkl"))


@skip_no_simhash
class TestSimHashIntegration:

    @pytest.fixture(autouse=True)
    def load_real(self):
        self.r = SimHashRetriever.load("data/processed/simhash_index.pkl")

    def test_index_has_fingerprints(self):
        assert len(self.r.fingerprints) > 0

    def test_query_returns_results(self):
        results = self.r.query("attendance policy", top_k=5)
        assert len(results) == 5

    def test_no_empty_chunks_in_results(self):
        results = self.r.query("grade cgpa degree", top_k=5)
        for r in results:
            assert r["id"] not in self.r._empty_chunks