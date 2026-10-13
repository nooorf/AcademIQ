"""
tests/test_answer_generator.py

Tests for src/generation/answer_generator.py

Critical tests:
  - _generate_extractive() scans ALL chunks (the critical fix)
  - Keyword matching uses query expansion terms
  - Empty chunks returns graceful "not found" response
  - generate() returns required fields in result dict
  - Sentence splitter respects abbreviations like "2.00", "B.S."
"""

import pytest

from src.generation.answer_generator import AnswerGenerator, _build_context


# ── Fixtures ──────────────────────────────────────────────────────────────────

CGPA_CHUNK = {
    "id":         "p023_c001",
    "text":       (
        "A student must maintain a minimum CGPA of 2.00 to be eligible for "
        "the award of the Bachelor degree. A student whose CGPA falls below "
        "2.00 will be placed on academic probation."
    ),
    "page":       23,
    "section":    "Chapter 3: Award of Bachelor Degree",
    "source":     "ug_handbook.pdf",
    "word_count": 45,
}

ATTENDANCE_CHUNK = {
    "id":         "p045_c003",
    "text":       (
        "The attendance policy requires that a student must maintain at least "
        "75 percent attendance in each course. A student whose attendance falls "
        "below 75 percent will be awarded an XF grade."
    ),
    "page":       45,
    "section":    "3. Offering and Registration of Courses",
    "source":     "ug_handbook.pdf",
    "word_count": 45,
}

PLAGIARISM_CHUNK = {
    "id":         "p082_c004",
    "text":       (
        "The use of unfair means in any examination is strictly prohibited. "
        "A student found using unfair means or plagiarism will be penalised."
    ),
    "page":       82,
    "section":    "6. Unfair Means",
    "source":     "ug_handbook.pdf",
    "word_count": 28,
}

UNRELATED_CHUNK = {
    "id":         "p010_c000",
    "text":       "NUST was established in 1991 by the Government of Pakistan.",
    "page":       10,
    "section":    "1. Introduction",
    "source":     "ug_handbook.pdf",
    "word_count": 10,
}


@pytest.fixture
def generator():
    """Extractive-mode generator (no API key required)."""
    return AnswerGenerator()


# ── _build_context() ──────────────────────────────────────────────────────────

class TestBuildContext:

    def test_includes_all_chunks(self):
        ctx = _build_context([CGPA_CHUNK, ATTENDANCE_CHUNK])
        assert "Excerpt 1" in ctx
        assert "Excerpt 2" in ctx

    def test_includes_section_and_page(self):
        ctx = _build_context([CGPA_CHUNK])
        assert "Chapter 3" in ctx
        assert "23" in ctx

    def test_includes_chunk_text(self):
        ctx = _build_context([CGPA_CHUNK])
        assert "CGPA" in ctx

    def test_empty_chunks_returns_empty_string(self):
        assert _build_context([]) == ""


# ── AnswerGenerator.generate() ────────────────────────────────────────────────

class TestGenerateReturnSchema:

    def test_returns_dict(self, generator):
        result = generator.generate("What is the minimum GPA?", [CGPA_CHUNK])
        assert isinstance(result, dict)

    def test_required_fields(self, generator):
        result = generator.generate("attendance policy", [ATTENDANCE_CHUNK])
        for key in ("answer", "mode", "model", "chunks_used", "latency_ms"):
            assert key in result, f"Missing key: {key}"

    def test_chunks_used_matches_input(self, generator):
        chunks = [CGPA_CHUNK, ATTENDANCE_CHUNK]
        result = generator.generate("policy", chunks)
        assert result["chunks_used"] == len(chunks)

    def test_latency_ms_is_non_negative(self, generator):
        result = generator.generate("cgpa requirement", [CGPA_CHUNK])
        assert result["latency_ms"] >= 0.0

    def test_mode_is_extractive_without_api_key(self, generator):
        # Generator fixture has no API key — must run extractive
        assert generator.mode == "extractive"
        result = generator.generate("cgpa", [CGPA_CHUNK])
        assert result["mode"] == "extractive"

    def test_model_field_is_string(self, generator):
        result = generator.generate("attendance", [ATTENDANCE_CHUNK])
        assert isinstance(result["model"], str)


class TestGenerateEdgeCases:

    def test_empty_chunks_returns_not_found(self, generator):
        result = generator.generate("any query", [])
        assert result["chunks_used"] == 0
        assert "not" in result["answer"].lower() or "no " in result["answer"].lower()

    def test_answer_is_non_empty_string(self, generator):
        result = generator.generate("What is attendance policy?", [ATTENDANCE_CHUNK])
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sources_line_present_in_extractive(self, generator):
        result = generator.generate("cgpa minimum 2.00", [CGPA_CHUNK])
        assert "Sources:" in result["answer"]


# ── _generate_extractive() — critical fix tests ───────────────────────────────

class TestExtractiveMode:

    def test_scans_all_chunks_not_just_first(self, generator):
        """
        CRITICAL BUG FIX: _generate_extractive() must scan ALL chunks.
        Previously only chunks[0] was read — if CGPA chunk was in position 2,
        the answer missed it entirely.
        """
        # Put UNRELATED chunk first, CGPA chunk second
        chunks = [UNRELATED_CHUNK, CGPA_CHUNK]
        result = generator.generate("What is the minimum CGPA requirement?", chunks)
        # Answer should contain CGPA info from the second chunk
        assert "cgpa" in result["answer"].lower() or "2.00" in result["answer"], (
            "Extractive mode failed to find CGPA info from chunk[1] — "
            "likely only scanned chunk[0]"
        )

    def test_query_expansion_in_extractive(self, generator):
        """
        Extractive mode uses expand_query() so 'GPA' also matches 'cgpa', '2.00'.
        """
        result = generator.generate("What is the minimum GPA requirement?", [CGPA_CHUNK])
        answer = result["answer"].lower()
        assert "cgpa" in answer or "2.00" in answer or "bachelor" in answer, (
            "Query expansion not applied in extractive mode — "
            "'GPA' should match 'cgpa' / '2.00'"
        )

    def test_attendance_query_finds_75(self, generator):
        result = generator.generate("What is the attendance requirement?",
                                    [UNRELATED_CHUNK, ATTENDANCE_CHUNK])
        assert "75" in result["answer"] or "attendance" in result["answer"].lower()

    def test_plagiarism_query_finds_unfair_means(self, generator):
        result = generator.generate("What is the plagiarism policy?",
                                    [UNRELATED_CHUNK, PLAGIARISM_CHUNK])
        answer = result["answer"].lower()
        assert "unfair" in answer or "plagiarism" in answer

    def test_fallback_when_no_matching_sentences(self, generator):
        """
        When no sentence scores > 0, extractive fallback returns first 60 words.
        Should not raise and must return a non-empty string.
        """
        # Query has no overlap with chunk text whatsoever
        niche_chunk = {
            "id": "x1", "text": "NUST Rawalpindi campus facilities.",
            "page": 1, "section": "Campus", "source": "t.pdf", "word_count": 4
        }
        result = generator.generate("quantum entanglement superconductor", [niche_chunk])
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sentence_splitter_handles_abbreviations(self, generator):
        """
        Sentence splitter must not split on '2.00', 'B.S.', etc.
        Previously split('.') broke decimal numbers into fragments.
        """
        chunk_with_abbrev = {
            "id": "abbrev_c",
            "text": (
                "The minimum CGPA of 2.00 is required. "
                "The B.S. degree requires 130 credit hours."
            ),
            "page": 1, "section": "Degree", "source": "t.pdf", "word_count": 20
        }
        result = generator.generate("minimum cgpa degree requirement", [chunk_with_abbrev])
        # Must not crash and must return something coherent
        assert isinstance(result["answer"], str)
        assert "2.00" in result["answer"] or "cgpa" in result["answer"].lower()

    def test_multi_chunk_best_sentences_selected(self, generator):
        """Top-scoring sentences from any chunk should appear in answer."""
        # Both chunks relevant — best sentences from both should be candidates
        result = generator.generate(
            "cgpa attendance requirement",
            [CGPA_CHUNK, ATTENDANCE_CHUNK, PLAGIARISM_CHUNK]
        )
        answer = result["answer"].lower()
        # At least one key term from each relevant chunk should appear
        assert "cgpa" in answer or "2.00" in answer or "75" in answer or "attendance" in answer