"""
tests/test_minhash.py

Tests for src/retrieval/minhash_lsh.py
"""

import pytest

from tests.conftest import skip_no_minhash
from src.retrieval.minhash_lsh import MinHash, MinHashLSHRetriever, build_minhash
from src.utils.preprocessing   import tokenize, expand_query


class TestBuildMinHash:

    def test_returns_minhash_object(self):
        tokens = {"cgpa", "grade", "point"}
        sig    = build_minhash(tokens, num_perm=64)
        assert isinstance(sig, MinHash)

    def test_empty_token_set(self):
        sig = build_minhash(set(), num_perm=64)
        assert isinstance(sig, MinHash)

    def test_similar_sets_have_high_jaccard(self):
        tokens_a = {"cgpa", "grade", "point", "cumulative", "2.00"}
        tokens_b = {"cgpa", "grade", "point", "cumulative", "average"}
        sig_a = build_minhash(tokens_a, num_perm=256)
        sig_b = build_minhash(tokens_b, num_perm=256)
        # Jaccard ≥ 3/6 = 0.5 — MinHash estimate should be in ballpark
        assert sig_a.jaccard(sig_b) > 0.2

    def test_dissimilar_sets_have_low_jaccard(self):
        tokens_a = {"cgpa", "grade", "point"}
        tokens_b = {"hostel", "accommodation", "campus", "dormitory"}
        sig_a = build_minhash(tokens_a, num_perm=256)
        sig_b = build_minhash(tokens_b, num_perm=256)
        assert sig_a.jaccard(sig_b) < 0.3


class TestTokenizerFix:
    """Tests that verify the len >= 2 fix is applied in minhash preprocessing."""

    def test_xf_kept_in_tokenizer(self):
        tokens = tokenize("student awarded XF grade attendance", as_set=True)
        assert "xf" in tokens, "'xf' must be in tokens — was dropped by len > 2 bug"

    def test_75_kept_in_tokenizer(self):
        tokens = tokenize("attendance below 75 percent not allowed", as_set=True)
        assert "75" in tokens, "'75' must be in tokens — was dropped by len > 2 bug"

    def test_expansion_75_survives_tokenize(self):
        expanded = expand_query("What is the attendance policy?")
        tokens   = tokenize(expanded, as_set=True)
        assert "75" in tokens, (
            "'75' from attendance expansion must survive tokenize() — "
            "previously expansion added '75' but tokenize() dropped it"
        )


class TestMinHashFit:

    def test_fit_indexes_all_non_empty_chunks(self, sample_chunks):
        r = MinHashLSHRetriever(num_perm=64, threshold=0.3)
        r.fit(sample_chunks)
        assert len(r.signatures) == len(sample_chunks)

    def test_fit_empty_raises(self):
        r = MinHashLSHRetriever(num_perm=64, threshold=0.3)
        with pytest.raises(ValueError, match="empty"):
            r.fit([])

    def test_chunk_order_matches_input(self, sample_chunks):
        r = MinHashLSHRetriever(num_perm=64, threshold=0.3)
        r.fit(sample_chunks)
        assert r.chunk_order == [c["id"] for c in sample_chunks]


class TestMinHashQuery:

    @pytest.fixture(autouse=True)
    def fitted(self, sample_chunks):
        self.r = MinHashLSHRetriever(num_perm=128, threshold=0.3)
        self.r.fit(sample_chunks)

    def test_query_returns_list(self):
        assert isinstance(self.r.query("GPA requirement"), list)

    def test_query_returns_at_most_top_k(self):
        results = self.r.query("degree requirement", top_k=3)
        assert len(results) <= 3

    def test_result_fields_present(self):
        results = self.r.query("attendance policy", top_k=1)
        assert len(results) >= 1
        r = results[0]
        for field in ("id", "text", "page", "section", "score", "latency_ms"):
            assert field in r

    def test_scores_between_0_and_1(self):
        results = self.r.query("cgpa grade point", top_k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_fallback_always_returns_top_k(self):
        """LSH may find 0 candidates — fallback brute-force must fill top_k."""
        from tests.conftest import SAMPLE_CHUNKS
        # Use a very high threshold so LSH finds no candidates,
        # forcing the brute-force fallback path.
        # num_perm=256 with threshold=0.9 gives valid band config but
        # still prevents real matches on tiny synthetic chunks.
        r = MinHashLSHRetriever(num_perm=256, threshold=0.9)
        r.fit(list(SAMPLE_CHUNKS))
        results = r.query("academic policy grade", top_k=3)
        assert len(results) == 3, (
            "Fallback brute-force must guarantee top_k results "
            "even when LSH threshold finds no candidates"
        )

    def test_query_before_fit_raises(self):
        r = MinHashLSHRetriever()
        with pytest.raises(RuntimeError, match="not built"):
            r.query("query")


class TestMinHashPersistence:

    def test_save_load_roundtrip(self, sample_chunks, tmp_path):
        r = MinHashLSHRetriever(num_perm=64, threshold=0.3)
        r.fit(sample_chunks)
        path = str(tmp_path / "minhash_test.pkl")
        r.save(path)
        loaded = MinHashLSHRetriever.load(path)
        assert len(loaded.signatures) == len(sample_chunks)
        assert loaded.num_perm == 64
        assert loaded.threshold == 0.3

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MinHashLSHRetriever.load(str(tmp_path / "missing.pkl"))

    def test_loaded_results_consistent(self, sample_chunks, tmp_path):
        r = MinHashLSHRetriever(num_perm=128, threshold=0.3)
        r.fit(sample_chunks)
        path = str(tmp_path / "mh.pkl")
        r.save(path)
        loaded = MinHashLSHRetriever.load(path)
        ids_orig   = [x["id"] for x in r.query("grade policy", top_k=3)]
        ids_loaded = [x["id"] for x in loaded.query("grade policy", top_k=3)]
        assert ids_orig == ids_loaded


@skip_no_minhash
class TestMinHashIntegration:

    @pytest.fixture(autouse=True)
    def load_real(self):
        self.r = MinHashLSHRetriever.load("data/processed/minhash_index.pkl")

    def test_repetition_query_finds_relevant(self):
        results = self.r.query("How many times can a course be repeated?", top_k=5)
        texts   = [r["text"].lower() for r in results]
        assert any("repeat" in t for t in texts), \
            "Repetition query should find repetition-related chunks"

    def test_all_results_have_valid_ids(self):
        results = self.r.query("academic policy", top_k=5)
        for r in results:
            assert r["id"] in self.r.chunks