"""
src/ingestion/chunker.py

Splits raw_pages.jsonl into overlapping text chunks suitable for retrieval.

Strategy:
  - Merge consecutive pages from the SAME chapter until target word count.
  - Pages from different chapters are NEVER merged — chapter boundary always
    forces a chunk boundary.
  - Sliding-window overlap (OVERLAP_WORDS) is applied both within a chapter
    and carries the tail of a flushed chunk into the next one.
  - The section_heading stored on each chunk is the heading of the FIRST page
    in that chunk (most specific label for that content).

Usage:
    python -m src.ingestion.chunker \
        --input  data/processed/raw_pages.jsonl \
        --output data/processed/chunks.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from src.utils.config_loader import cfg


# ── Configuration (read from config.yaml with safe defaults) ──────────────────
TARGET_WORDS  = cfg("ingestion", "chunk_size",    default=350)
MAX_WORDS     = int(TARGET_WORDS * 1.43)          # ~500 words hard ceiling
OVERLAP_WORDS = cfg("ingestion", "chunk_overlap", default=50)
MIN_WORDS     = cfg("ingestion", "min_chunk_size", default=100)


# ── Helpers ───────────────────────────────────────────────────────────────────

_CHAPTER_RE = re.compile(r'^Chapter\s+(\d+)\s*:', re.IGNORECASE)


def get_chapter(section_heading: str) -> str:
    """
    Extract 'Chapter N' boundary key from a section heading.
    Annex pages are treated as their own boundary group.
    Falls back to the full heading string for everything else.
    """
    m = _CHAPTER_RE.match(section_heading.strip())
    if m:
        return f"chapter_{m.group(1)}"
    if section_heading.strip().lower().startswith("annex"):
        return "_".join(section_heading.strip().split()[:2]).lower()
    return section_heading.strip()


def clean_heading_label(heading: str) -> str:
    """
    Trim headings that captured body text due to missing hard line breaks.

    '3. Vision. NUST aspires to be a comprehensive university…'
    → '3. Vision'

    'Chapter 2: Scheme of Studies...' → kept as-is (no second period)
    """
    heading = heading.strip()
    parts = heading.split('. ')
    if len(parts) >= 3:
        return '. '.join(parts[:2]).rstrip('.')
    return heading.rstrip('.')


def words(text: str) -> list[str]:
    return text.split()


def make_chunk_id(source: str, page: int, idx: int) -> str:
    stem = Path(source).stem.replace(' ', '_').replace('-', '_')
    return f"{stem}_p{page:03d}_c{idx:03d}"


# ── Core chunking logic ───────────────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Merge and split pages into TARGET_WORDS chunks with OVERLAP_WORDS overlap.

    Args:
        pages: list of dicts from raw_pages.jsonl

    Returns:
        list of chunk dicts (id, text, page, section, source, word_count)
    """
    chunks: list[dict] = []
    chunk_idx   = 0
    discarded   = 0

    buffer_words:   list[str] = []
    buffer_page:    int       = 0
    buffer_heading: str       = "Preamble"
    buffer_source:  str       = ""
    buffer_chapter: str       = ""

    def flush(buf_words: list[str],
              buf_page: int,
              buf_heading: str,
              buf_source: str) -> None:
        nonlocal chunk_idx, discarded
        text = ' '.join(buf_words).strip()
        if len(buf_words) < MIN_WORDS:
            discarded += 1
            return
        chunk_idx += 1
        chunks.append({
            "id":         make_chunk_id(buf_source, buf_page, chunk_idx),
            "text":       text,
            "page":       buf_page,
            "section":    clean_heading_label(buf_heading),
            "source":     buf_source,
            "word_count": len(buf_words),
        })

    for page in pages:
        page_words   = words(page["raw_text"])
        page_chapter = get_chapter(page["section_heading"])
        page_source  = page["source_filename"]

        if not page_words:
            continue

        # ── Chapter / source boundary: flush current buffer ────────────────
        if buffer_words and (page_chapter != buffer_chapter or
                             page_source  != buffer_source):
            flush(buffer_words, buffer_page, buffer_heading, buffer_source)
            buffer_words   = []
            buffer_page    = page["page_number"]
            buffer_heading = page["section_heading"]
            buffer_source  = page_source
            buffer_chapter = page_chapter

        # Initialise buffer on very first page
        if not buffer_words:
            buffer_page    = page["page_number"]
            buffer_heading = page["section_heading"]
            buffer_source  = page_source
            buffer_chapter = page_chapter

        buffer_words.extend(page_words)

        # ── Overflow split with overlap ────────────────────────────────────
        while len(buffer_words) >= MAX_WORDS:
            chunk_w = buffer_words[:TARGET_WORDS]
            flush(chunk_w, buffer_page, buffer_heading, buffer_source)
            # Keep last OVERLAP_WORDS for context continuity
            buffer_words = buffer_words[TARGET_WORDS - OVERLAP_WORDS:]

    # ── Final flush ────────────────────────────────────────────────────────
    if buffer_words:
        flush(buffer_words, buffer_page, buffer_heading, buffer_source)

    if discarded > 0:
        print(f"Warning: {discarded} buffer(s) discarded (below MIN_WORDS={MIN_WORDS})")

    return chunks


# ── Entry point ───────────────────────────────────────────────────────────────

def main(input_file: str, output_file: str) -> None:
    input_path  = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pages: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))

    print(f"Loaded {len(pages)} pages from {input_path}")

    chunks = chunk_pages(pages)

    # FIX: Guard against empty chunk list before calling min() / statistics
    if not chunks:
        print("Warning: no chunks were produced. "
              "Check that input pages have >= MIN_WORDS words.")
        return

    try:
        with open(output_path, 'w', encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    except OSError as e:
        raise OSError(f"Failed to write chunks to {output_path}: {e}") from e

    word_counts = [c["word_count"] for c in chunks]
    print(f"\nChunks produced : {len(chunks)}")
    print(f"Word count      : min={min(word_counts)}, "
          f"max={max(word_counts)}, "
          f"avg={sum(word_counts) // len(word_counts)}")
    print(f"Output          : {output_path}")

    print("\n--- Sample chunks ---")
    for c in chunks[:5]:
        preview = ' '.join(c['text'].split()[:20])
        print(f"  [{c['id']}] page={c['page']} section={c['section'][:50]!r}")
        print(f"    {preview}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk raw handbook pages into retrieval-ready segments")
    parser.add_argument("--input",  required=True, help="Path to raw_pages.jsonl")
    parser.add_argument("--output", required=True, help="Path to output chunks.jsonl")
    args = parser.parse_args()
    main(args.input, args.output)