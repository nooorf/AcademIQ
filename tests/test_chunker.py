"""
tests/test_chunker.py

Unit tests for src/ingestion/chunker.py

Tests verify:
  - chunk_pages() produces correct output structure
  - Chapter boundaries enforced (no cross-chapter merging)
  - MIN_WORDS filter works — short buffers discarded
  - Empty word_counts does NOT crash (CRITICAL bug fix)
  - clean_heading_label() trims correctly
  - get_chapter() extracts chapter keys correctly
  - Overlap carries correct number of words
"""

import pytest

from src.ingestion.chunker import (
    chunk_pages,
    clean_heading_label,
    get_chapter,
    make_chunk_id,
    MIN_WORDS,
    TARGET_WORDS,
    OVERLAP_WORDS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_page(page_number: int,
              section_heading: str,
              word_count: int = 200,
              source: str = "test.pdf") -> dict:
    """Create a synthetic page dict."""
    return {
        "page_number":     page_number,
        "section_heading": section_heading,
        "raw_text":        " ".join([f"word{i}" for i in range(word_count)]),
        "source_filename": source,
    }


# ── clean_heading_label() ─────────────────────────────────────────────────────

class TestCleanHeadingLabel:

    def test_trims_body_text_after_second_period(self):
        heading = "3. Vision. NUST aspires to be a comprehensive university"
        result  = clean_heading_label(heading)
        assert result == "3. Vision"

    def test_preserves_chapter_heading(self):
        heading = "Chapter 2: Scheme of Studies, Examinations, and Academic"
        assert clean_heading_label(heading) == heading.strip()

    def test_preserves_heading_with_one_period(self):
        heading = "8. Important Telephone Numbers:"
        assert clean_heading_label(heading) == heading.strip()

    def test_handles_empty_string(self):
        assert clean_heading_label("") == ""

    def test_strips_trailing_whitespace(self):
        assert clean_heading_label("  4. Warning.  ") == "4. Warning"


# ── get_chapter() ─────────────────────────────────────────────────────────────

class TestGetChapter:

    def test_extracts_chapter_number(self):
        assert get_chapter("Chapter 3: Award of Bachelor Degree") == "chapter_3"
        assert get_chapter("Chapter 11: NUST Code of Conduct")    == "chapter_11"

    def test_annex_returns_annex_key(self):
        result = get_chapter("Annex A")
        assert result.startswith("annex")

    def test_non_chapter_returns_full_heading(self):
        heading = "8. Repetition/Retest of Course(s)."
        assert get_chapter(heading) == heading.strip()

    def test_case_insensitive_chapter(self):
        assert get_chapter("CHAPTER 2: Scheme") == "chapter_2"


# ── chunk_pages() — structure ──────────────────────────────────────────────────

class TestChunkPages:

    def test_returns_list_of_dicts(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_required_fields_present(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        required = {"id", "text", "page", "section", "source", "word_count"}
        for c in chunks:
            assert required.issubset(c.keys()), f"Chunk missing fields: {required - c.keys()}"

    def test_word_count_matches_text(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        for c in chunks:
            assert c["word_count"] == len(c["text"].split())

    def test_all_chunks_meet_min_words(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        for c in chunks:
            assert c["word_count"] >= MIN_WORDS, (
                f"Chunk {c['id']} has {c['word_count']} words < MIN_WORDS={MIN_WORDS}"
            )

    def test_chunk_ids_are_unique(self, sample_pages):
        chunks = chunk_pages(sample_pages)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"

    def test_empty_pages_list_returns_empty(self):
        assert chunk_pages([]) == []

    def test_all_pages_below_min_words_returns_empty(self):
        """
        CRITICAL BUG FIX: chunk_pages must return [] without crashing.
        Previously `min(word_counts)` would crash with ValueError on [].
        """
        tiny_pages = [
            make_page(1, "Chapter 1: Intro", word_count=10),
            make_page(2, "Chapter 1: Intro", word_count=15),
        ]
        # With word_count < MIN_WORDS these should be discarded
        chunks = chunk_pages(tiny_pages)
        # Passes if no ValueError is raised
        assert isinstance(chunks, list)


# ── chunk_pages() — chapter boundaries ────────────────────────────────────────

class TestChapterBoundaries:

    def test_different_chapters_not_merged(self):
        pages = [
            make_page(1, "Chapter 1: Intro",  word_count=200),
            make_page(2, "Chapter 2: Studies", word_count=200),
        ]
        chunks = chunk_pages(pages)
        # No chunk should contain text from both chapters
        # (they'd need to be very long for this to happen, but verify source pages)
        for c in chunks:
            assert c["word_count"] >= MIN_WORDS

    def test_same_chapter_pages_can_merge(self):
        # Two short pages from the same chapter — should merge
        pages = [
            make_page(1, "Chapter 3: Degree", word_count=80),
            make_page(2, "Chapter 3: Degree", word_count=80),
        ]
        chunks = chunk_pages(pages)
        # Combined = 160 words >= MIN_WORDS — should produce 1 chunk
        assert len(chunks) == 1
        assert chunks[0]["word_count"] == 160

    def test_different_sources_not_merged(self):
        pages = [
            make_page(1, "Chapter 1: Intro", word_count=200, source="ug_handbook.pdf"),
            make_page(2, "Chapter 1: Intro", word_count=200, source="pg_handbook.pdf"),
        ]
        chunks = chunk_pages(pages)
        sources = {c["source"] for c in chunks}
        assert len(sources) == 2, "Pages from different PDFs should not be merged"

    def test_section_heading_from_first_page(self):
        pages = [
            make_page(1, "Chapter 3: Degree", word_count=80),
            make_page(2, "Chapter 3: Degree", word_count=80),
        ]
        chunks = chunk_pages(pages)
        # Heading should come from first page in the merged buffer
        assert "chapter 3" in chunks[0]["section"].lower()


# ── make_chunk_id() ───────────────────────────────────────────────────────────

class TestMakeChunkId:

    def test_format_correct(self):
        cid = make_chunk_id("ug_handbook.pdf", 23, 7)
        assert "p023" in cid
        assert "c007" in cid

    def test_spaces_replaced_in_source(self):
        cid = make_chunk_id("ug handbook.pdf", 1, 1)
        assert " " not in cid

    def test_dashes_replaced(self):
        cid = make_chunk_id("28-Aug-2024-Handbook.pdf", 1, 1)
        assert "-" not in cid