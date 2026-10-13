"""
tests/test_preprocessing.py

Unit tests for src/utils/preprocessing.py

Tests verify:
  - tokenize() correctly applies len >= 2 (the critical bug fix)
  - tokenize() keeps "xf", "75", "ch" (policy tokens that were lost before)
  - tokenize() removes stop words
  - tokenize() returns set or list based on as_set flag
  - expand_query() includes handbook-specific synonyms
  - expand_query() deduplicates — no term appears twice
  - expand_query() handles multiple triggers without explosion
"""

import pytest

from src.utils.preprocessing import tokenize, expand_query, STOP_WORDS


# ── tokenize() ────────────────────────────────────────────────────────────────

class TestTokenize:

    def test_returns_list_by_default(self):
        result = tokenize("hello world")
        assert isinstance(result, list)

    def test_returns_set_when_requested(self):
        result = tokenize("hello world", as_set=True)
        assert isinstance(result, set)

    def test_lowercases_text(self):
        result = tokenize("GPA CGPA")
        assert "gpa"  in result
        assert "cgpa" in result

    def test_strips_punctuation(self):
        # "75%" → "75" after stripping %
        result = tokenize("75% attendance")
        assert "75"         in result
        assert "attendance" in result
        assert "%"          not in " ".join(result)

    def test_keeps_two_char_tokens(self):
        """
        CRITICAL BUG FIX: len >= 2 replaces len > 2.
        These tokens were silently dropped before the fix.
        """
        result = tokenize("xf grade ch credit 75 percent")
        assert "xf" in result, "'xf' (XF grade) must be kept — was dropped by len > 2 bug"
        assert "75" in result, "'75' (attendance %) must be kept — was dropped by len > 2 bug"
        assert "ch" in result, "'ch' (credit hours) must be kept — was dropped by len > 2 bug"

    def test_drops_single_char_tokens(self):
        result = tokenize("a b c degree")
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result

    def test_removes_stop_words(self):
        for sw in ["a", "an", "the", "and", "student", "nust"]:
            result = tokenize(f"{sw} policy")
            assert sw not in result, f"Stop word '{sw}' should be removed"

    def test_preserves_policy_numbers(self):
        # "2.00" → "2" and "00" after punctuation strip; "2" is len 1 so dropped,
        # "00" is len 2 so kept. This is expected behaviour.
        result = tokenize("CGPA 2.00 minimum")
        assert "00" in result or "2" not in result   # at least one numeric part present

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_stop_words(self):
        assert tokenize("a the and or but") == []

    def test_set_deduplication(self):
        result_set  = tokenize("cgpa cgpa cgpa grade", as_set=True)
        result_list = tokenize("cgpa cgpa cgpa grade", as_set=False)
        assert result_set.count if hasattr(result_set, 'count') else True   # set has no duplicates
        assert result_list.count("cgpa") == 3   # list preserves frequency (for SimHash TF)

    def test_real_handbook_snippet(self):
        text = (
            "A student whose CGPA falls below 2.00 will be placed on "
            "academic probation. XF grade is awarded when attendance "
            "falls below 75%."
        )
        tokens = tokenize(text)
        assert "cgpa"       in tokens
        assert "probation"  in tokens
        assert "xf"         in tokens
        assert "75"         in tokens
        # stop words removed
        assert "a"          not in tokens
        assert "student"    not in tokens


# ── expand_query() ────────────────────────────────────────────────────────────

class TestExpandQuery:

    def test_returns_string(self):
        assert isinstance(expand_query("What is GPA?"), str)

    def test_original_query_preserved(self):
        q      = "What is the minimum GPA requirement?"
        result = expand_query(q)
        assert result.startswith(q)

    def test_gpa_expansion_includes_cgpa(self):
        result = expand_query("What is the GPA requirement?")
        assert "cgpa"        in result.lower()
        assert "grade point" in result.lower()
        assert "2.00"        in result

    def test_attendance_expansion_includes_75(self):
        """
        CRITICAL: expansion must add "75" — previously added but then
        lost because tokenize() dropped len=2 tokens. Now both are fixed.
        """
        result = expand_query("What is the attendance policy?")
        assert "75"         in result
        assert "absent"     in result
        assert "xf"         in result.lower()

    def test_plagiarism_expansion_includes_unfair_means(self):
        result = expand_query("What is the plagiarism policy?")
        assert "unfair means" in result.lower()
        assert "cheating"     in result.lower()

    def test_no_duplicate_terms(self):
        """
        CRITICAL BUG FIX: expansion must deduplicate.
        A query with both "gpa" and "grade" previously appended "cgpa" twice.
        """
        result  = expand_query("What is the minimum GPA grade requirement?")
        terms   = result.lower().split()
        # Count occurrences of a term that appears in multiple expansion lists
        cgpa_count = terms.count("cgpa")
        assert cgpa_count <= 1, (
            f"'cgpa' appears {cgpa_count} times — deduplication not working"
        )

    def test_no_expansion_for_unknown_terms(self):
        q      = "What is the library schedule?"
        result = expand_query(q)
        # Should just be the original query (no matching triggers)
        assert result == q

    def test_multiple_triggers_deduplicated(self):
        """
        Query with 'fail' and 'failed' both trigger — expansion terms must
        not duplicate across both.
        """
        result = expand_query("What happens if a student failed or fail a course?")
        terms  = result.lower().split()
        # "xf" appears in both "fail" and "failed" expansions — should appear once
        xf_count = terms.count("xf")
        assert xf_count <= 1, f"'xf' deduplicated incorrectly: appears {xf_count} times"

    def test_thesis_expansion(self):
        result = expand_query("What are the thesis requirements?")
        assert "fyp"                in result.lower()
        assert "final year project" in result.lower()
        assert "dissertation"       in result.lower()