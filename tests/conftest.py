"""
tests/conftest.py

Shared pytest fixtures and helpers used across all test modules.

Fixtures that depend on real index files are skipped automatically
when those files don't exist, so the test suite can run in CI without
a fully ingested corpus.
"""

import json
from pathlib import Path

import pytest

# ── Minimal synthetic chunk corpus ─────────────────────────────────────────────
# Covers the key policy domains the retrieval pipeline is tested against.
# All tests that don't need the real index use this fixture.

SAMPLE_CHUNKS = [
    {
        "id":         "handbook_p023_c001",
        "text":       (
            "A student must maintain a minimum CGPA of 2.00 to be eligible for "
            "the award of the Bachelor degree. A student whose CGPA falls below "
            "2.00 will be placed on academic probation. The cumulative grade "
            "point average is calculated by dividing total quality points by "
            "total credit hours attempted."
        ),
        "page":       23,
        "section":    "Chapter 3: Award of Bachelor Degree and Academic Deficiencies",
        "source":     "ug_handbook.pdf",
        "word_count": 62,
    },
    {
        "id":         "handbook_p027_c002",
        "text":       (
            "Repetition or retest of courses is permitted under specific conditions. "
            "A student who fails a course with an F grade may repeat the course in "
            "a subsequent semester. A student who receives an XF grade due to "
            "shortage of attendance is not allowed to appear in the final "
            "examination and must repeat the course. The XF grade is recorded "
            "permanently on the transcript."
        ),
        "page":       27,
        "section":    "8. Repetition/Retest of Course(s).",
        "source":     "ug_handbook.pdf",
        "word_count": 73,
    },
    {
        "id":         "handbook_p045_c003",
        "text":       (
            "The attendance policy requires that a student must maintain at least "
            "75 percent attendance in each course. A student whose attendance falls "
            "below 75 percent will be awarded an XF grade and will not be allowed "
            "to appear in the final examination. Attendance is mandatory for all "
            "lectures, tutorials, and laboratory sessions."
        ),
        "page":       45,
        "section":    "3. Offering and Registration of Courses",
        "source":     "ug_handbook.pdf",
        "word_count": 64,
    },
    {
        "id":         "handbook_p082_c004",
        "text":       (
            "The use of unfair means in any examination is strictly prohibited. "
            "A student found using unfair means or plagiarism will be penalised. "
            "Plagiarism includes copying from another student, submitting work "
            "that is not one's own, and using unauthorised materials during "
            "examinations. Academic dishonesty is taken seriously and may result "
            "in expulsion."
        ),
        "page":       82,
        "section":    "6. Unfair Means",
        "source":     "ug_handbook.pdf",
        "word_count": 68,
    },
    {
        "id":         "handbook_p031_c005",
        "text":       (
            "Academic standards for the award of degree require that a student "
            "must complete all prescribed credit hours with a minimum CGPA of "
            "2.00. A student on probation who fails to improve the CGPA will "
            "face suspension from the university. The grading scale uses letter "
            "grades from A to F with corresponding grade points."
        ),
        "page":       31,
        "section":    "5. Academic Standards for Award of Degree",
        "source":     "ug_handbook.pdf",
        "word_count": 61,
    },
]


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Return synthetic chunks for unit tests that don't need real indexes."""
    return [dict(c) for c in SAMPLE_CHUNKS]


@pytest.fixture
def sample_pages() -> list[dict]:
    """Return synthetic raw pages for chunker tests."""
    return [
        {
            "page_number":     23,
            "section_heading": "Chapter 3: Award of Bachelor Degree",
            "raw_text":        (
                "Chapter 3: Award of Bachelor Degree\n"
                "A student must maintain a minimum CGPA of 2.00 to be eligible "
                "for the award of the Bachelor degree. Students on probation are "
                "required to improve their CGPA failing which they will be "
                "suspended. The cumulative grade point average reflects academic "
                "performance across all semesters. Credit hours are assigned to "
                "each course based on contact hours per week. A minimum of 130 "
                "credit hours are required for the Bachelor degree."
            ),
            "source_filename": "ug_handbook.pdf",
        },
        {
            "page_number":     24,
            "section_heading": "Chapter 3: Award of Bachelor Degree",
            "raw_text":        (
                "4. Warning.\n"
                "A student whose CGPA falls below 2.00 at the end of any semester "
                "will be issued a warning. The warning is recorded on the academic "
                "transcript. A student who receives two consecutive warnings will "
                "be placed on academic probation. Probation is a serious academic "
                "standing that requires improvement in the subsequent semester."
            ),
            "source_filename": "ug_handbook.pdf",
        },
        {
            "page_number":     27,
            "section_heading": "Chapter 3: Award of Bachelor Degree",
            "raw_text":        (
                "8. Repetition/Retest of Course(s).\n"
                "A student who fails a course may repeat it in a subsequent "
                "semester. The grade obtained on repetition replaces the original "
                "grade for CGPA calculation. A student may repeat a course at "
                "most twice. Courses awarded an XF grade due to shortage of "
                "attendance must be repeated in full."
            ),
            "source_filename": "ug_handbook.pdf",
        },
    ]


# ── Real index availability checks ───────────────────────────────────────────

def index_available(name: str) -> bool:
    paths = {
        "tfidf":   "data/processed/tfidf_index.pkl",
        "minhash": "data/processed/minhash_index.pkl",
        "simhash": "data/processed/simhash_index.pkl",
        "reranker":"data/processed/reranker_index.pkl",
        "chunks":  "data/processed/chunks.jsonl",
    }
    return Path(paths.get(name, "")).exists()


skip_no_tfidf   = pytest.mark.skipif(not index_available("tfidf"),
                                      reason="TF-IDF index not built")
skip_no_minhash = pytest.mark.skipif(not index_available("minhash"),
                                      reason="MinHash index not built")
skip_no_simhash = pytest.mark.skipif(not index_available("simhash"),
                                      reason="SimHash index not built")
skip_no_reranker= pytest.mark.skipif(not index_available("reranker"),
                                      reason="Re-ranker index not built")
skip_no_chunks  = pytest.mark.skipif(not index_available("chunks"),
                                      reason="chunks.jsonl not present")