"""
src/utils/preprocessing.py

All modules now import from here.
"""

import re
from typing import Union


# ── Stop words ────────────────────────────────────────────────────────────────
# English function words + NUST-specific high-frequency, non-discriminative terms.
# Kept as frozenset for O(1) membership testing.
STOP_WORDS: frozenset[str] = frozenset({
    # English function words
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "as", "if",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "every", "any", "all", "some", "such", "than", "then",
    "can", "up", "out", "into", "about", "more", "when", "who",
    "which", "their", "them", "he", "she", "they", "we", "you",
    "our", "your", "my", "his", "her", "me", "us", "only", "own",
    "other", "same", "very", "just", "one", "two", "under", "upon",
    "after", "before", "during", "between", "through", "above",
    # NUST-specific noise (appears in >85% of chunks — no discriminative value)
    "student", "students", "nust", "per", "also", "however",
    "therefore", "given", "following", "respective", "concerned",
    "thereof", "therein", "institution", "institutions",
    "semester", "case", "university", "shall", "according",
    "hereafter", "hereinafter", "whereas", "provided",
})


# ── Query expansion ───────────────────────────────────────────────────────────
# Canonical map — maps natural-language query words to handbook vocabulary.
# NUST handbook uses "CGPA 2.00", "75% attendance", "XF grade" etc. while
# students phrase queries differently. One map used by ALL retrievers.
QUERY_EXPANSION: dict[str, list[str]] = {
    "gpa": [
        "cgpa", "gpa", "grade point", "grade point average",
        "2.00", "cumulative", "academic standards",
    ],
    "cgpa": [
        "cgpa", "gpa", "grade point", "grade point average",
        "cumulative", "2.00", "academic standards",
    ],
    "fail": [
        "fail", "failed", "xf", "f grade", "academic deficien",
        "not pass", "repetition",
    ],
    "failed": [
        "fail", "failed", "xf", "f grade", "academic deficien",
        "repetition", "retest",
    ],
    "attendance": [
        "attendance", "75", "75%", "absent", "present",
        "xf", "xf grade", "not allowed appear", "shortage",
        "75 percent",
    ],
    "absent": [
        "absent", "attendance", "75", "xf grade", "shortage",
    ],
    "repeat": [
        "repeat", "repeated", "repetition", "retest",
        "w grade", "xf", "retake", "clearance",
    ],
    "repeated": [
        "repeat", "repeated", "repetition", "retest", "retake",
    ],
    "retake": [
        "repeat", "repetition", "retest", "w grade", "retake",
    ],
    "suspend": [
        "suspend", "suspension", "withdrawal", "probation",
        "expelled", "removed",
    ],
    "suspension": [
        "suspend", "suspension", "probation", "warning",
        "expelled", "removed",
    ],
    "probation": [
        "probation", "warning", "academic deficien", "cgpa",
        "below", "suspension", "deficient",
    ],
    "warning": [
        "warning", "probation", "academic deficien", "cgpa",
        "deficient",
    ],
    "withdraw": [
        "withdraw", "withdrawal", "drop", "w grade", "dropping",
    ],
    "drop": [
        "drop", "withdraw", "withdrawal", "w grade",
        "adding dropping",
    ],
    "minimum": [
        "minimum", "required", "least", "not less than", "must",
    ],
    "requirement": [
        "requirement", "required", "must", "condition",
        "criteria", "eligibility",
    ],
    "grade": [
        "grade", "grading", "cgpa", "gpa", "marks", "result",
        "assessment",
    ],
    "grading": [
        "grading", "grade", "cgpa", "marking scheme",
        "assessment", "percentage",
    ],
    "exam": [
        "examination", "exam", "end semester", "mid semester",
        "test", "invigilation",
    ],
    "examination": [
        "examination", "exam", "end semester", "mid semester",
        "assessment", "invigilation",
    ],
    "degree": [
        "degree", "award", "bachelor", "graduation", "cgpa",
        "credit hours", "convocation",
    ],
    "graduation": [
        "graduation", "degree", "award", "bachelor",
        "credit hours", "cgpa", "convocation",
    ],
    "credit": [
        "credit hours", "credit", "ch", "contact hour",
    ],
    "transfer": [
        "transfer", "inter-institutional", "change of programme",
        "migration",
    ],
    "scholarship": [
        "scholarship", "financial assistance", "merit", "award",
        "medal", "gold medal",
    ],
    "fee": [
        "fee", "tuition", "dues", "payment", "fine", "deposit",
    ],
    "hostel": [
        "hostel", "hostelite", "accommodation", "living",
        "campus", "residence",
    ],
    "discipline": [
        "discipline", "misconduct", "code of conduct", "prohibited",
        "ill-discipline", "punishment",
    ],
    "plagiarism": [
        "plagiarism", "unfair means", "cheating", "misconduct",
        "academic dishonesty", "copying", "unauthorized",
    ],
    "unfair": [
        "unfair means", "plagiarism", "cheating", "misconduct",
        "academic dishonesty", "copying",
    ],
    "cheating": [
        "cheating", "unfair means", "plagiarism", "misconduct",
        "academic dishonesty",
    ],
    "counselling": [
        "counselling", "counseling", "advisor", "c3a", "support",
    ],
    "counseling": [
        "counselling", "counseling", "advisor", "c3a", "support",
    ],
    "internship": [
        "internship", "industrial", "training", "contact hour",
    ],
    "thesis": [
        "thesis", "fyp", "final year project", "design project",
        "dissertation",
    ],
    "fyp": [
        "fyp", "final year project", "thesis", "design project",
    ],
    "club": [
        "club", "clubs", "society", "societies", "extracurricular",
    ],
    "society": [
        "society", "societies", "club", "clubs", "extracurricular",
    ],
    "medal": [
        "medal", "gold medal", "distinction", "dean list",
        "rector list",
    ],
    "freeze": [
        "freeze", "deferment", "deferred", "semester break",
    ],
    "deferment": [
        "deferment", "freeze", "deferred", "semester break",
    ],
    "conduct": [
        "conduct", "code of conduct", "discipline", "misconduct",
        "prohibited", "ill-discipline",
    ],
    "migration": [
        "migration", "transfer", "inter-institutional",
        "change of programme",
    ],
    "policy": [
        "policy", "regulation", "rules", "procedures",
    ],
}

# Pre-compiled punctuation stripper (shared, avoids re-compiling per call)
_PUNCT_RE = re.compile(r'[^a-z0-9\s]')


def tokenize(text: str, as_set: bool = False) -> Union[list[str], set[str]]:
    """
    Canonical tokenizer used by all retrieval modules (TF-IDF prep,
    MinHash, SimHash).

    Steps:
        1. Lowercase
        2. Strip all non-alphanumeric characters (preserves digits like 75, 2)
        3. Split on whitespace
        4. Remove stop words
        5. Minimum token length of 2 (was >2 — that silently dropped "xf",
           "75", "ch", "gp" which are critical policy tokens)

    Args:
        text   : Input text string.
        as_set : Return a set (order-independent, for Jaccard/MinHash) or
                 list (preserves frequency, for SimHash TF-weighting).

    Returns:
        list[str] by default, set[str] when as_set=True.
    """
    lowered = text.lower()
    stripped = _PUNCT_RE.sub(' ', lowered)
    tokens = [
        t for t in stripped.split()
        if len(t) >= 2 and t not in STOP_WORDS   # FIX: was > 2, now >= 2
    ]
    return set(tokens) if as_set else tokens


def expand_query(q: str) -> str:
    """
    Append handbook-specific synonyms for every trigger word found in q.

    Key properties:
        - Deduplication: each expansion term is added at most once, preventing
          duplicate terms from inflating TF weights in the vectorizer.
        - Canonical: same map and same dedup logic for all three retrievers,
          so cross-method comparisons are fair.

    Example:
        Input : "What is the minimum GPA requirement?"
        Output: "What is the minimum GPA requirement? cgpa gpa grade point
                 2.00 cumulative minimum required least not less than must
                 requirement required condition criteria"
        (deduplicated — "required" from both "minimum" and "requirement"
         appears only once)
    """
    q_lower = q.lower()
    seen: set[str] = set()
    parts: list[str] = [q]

    for trigger, expansions in QUERY_EXPANSION.items():
        if trigger in q_lower:
            for term in expansions:
                if term not in seen:
                    seen.add(term)
                    parts.append(term)

    return " ".join(parts)