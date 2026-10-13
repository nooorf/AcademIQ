"""
src/ingestion/pdf_extractor.py

Extract text page-by-page from NUST handbooks, preserving section headings.

"""

import argparse
import json
import re
from pathlib import Path

import pdfplumber


# ── Noise patterns (headers / footers / page numbers) ────────────────────────
_NOISE_PATTERNS = [
    re.compile(r'^NUST\s+Undergraduate\s+Student\s+Handbook\s*$',       re.IGNORECASE),
    re.compile(r'^NUST\s+Undergraduate\s+Student\s+Handbook\s+\d+\s*$', re.IGNORECASE),
    re.compile(r'^\d+\s+NUST\s+Undergraduate',                          re.IGNORECASE),
    re.compile(r'^\d+\s*$'),     # standalone page numbers
    re.compile(r'^\f\s*$'),      # form-feed characters
]

# ── Heading detection patterns ────────────────────────────────────────────────
_CHAPTER_RE          = re.compile(r'^Chapter\s+\d+\s*:',          re.IGNORECASE)
_NUMBERED_SECTION_RE = re.compile(r'^\d{1,3}\.\s+[A-Z][A-Za-z]')
_ANNEX_RE            = re.compile(r'^Annex\s+[A-Z]',              re.IGNORECASE)


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 4:
        return False
    # Never treat sub-bullets as headings
    if re.match(r'^[a-z]\.\s',    line): return False
    if re.match(r'^\([0-9]+\)\s', line): return False
    if re.match(r'^\([a-z]\)\s',  line): return False

    if _CHAPTER_RE.match(line):
        return True
    if _NUMBERED_SECTION_RE.match(line):
        label = re.split(r'[,.]', line, maxsplit=1)[0]
        if len(label.split()) <= 10:
            return True
    if _ANNEX_RE.match(line):
        return True
    return False


def is_noise(line: str) -> bool:
    stripped = line.strip()
    return any(pat.match(stripped) for pat in _NOISE_PATTERNS)


def clean_page_text(raw_text: str) -> str:
    """Remove per-page noise and collapse excessive blank lines."""
    lines   = raw_text.split('\n')
    cleaned = [ln for ln in lines if not is_noise(ln)]
    result  = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned))
    return result.strip()


def extract_pdfs(input_dir: str, output_file: str) -> None:
    input_path  = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    results: list[dict] = []
    pdf_errors:  int    = 0
    page_errors: int    = 0

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        current_heading = "Preamble"

        try:
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        raw_text = page.extract_text() or ""
                        if not raw_text.strip():
                            continue

                        cleaned_text = clean_page_text(raw_text)
                        if not cleaned_text:
                            continue

                        page_number = i + 1
                        lines       = cleaned_text.split('\n')
                        page_label  = None

                        for line in lines:
                            if is_heading(line):
                                heading = line.strip()
                                if page_label is None:
                                    page_label = heading
                                current_heading = heading

                        try:
                            record = {
                                "page_number":     page_number,
                                "section_heading": page_label if page_label else current_heading,
                                "raw_text":        cleaned_text,
                                "source_filename": pdf_file.name,
                            }
                            # Validate serializability before appending
                            json.dumps(record, ensure_ascii=False)
                            results.append(record)
                        except (TypeError, ValueError) as ser_err:
                            print(f"  WARN: page {page_number} skipped "
                                  f"(serialisation error): {ser_err}")
                            page_errors += 1

                    except Exception as page_err:
                        print(f"  WARN: page {i+1} failed: {page_err}")
                        page_errors += 1
                        continue

        except Exception as pdf_err:
            print(f"  ERROR processing {pdf_file.name}: {pdf_err}")
            pdf_errors += 1
            continue

    if not results:
        print("ERROR: No pages were successfully extracted. "
              "Check that the PDFs contain text layers.")
        return

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except OSError as e:
        raise OSError(f"Failed to write output to {output_path}: {e}") from e

    print(f"\nDone — {len(results)} pages extracted from {len(pdf_files)} file(s)")
    if page_errors:
        print(f"  Page-level errors : {page_errors}")
    if pdf_errors:
        print(f"  PDF-level errors  : {pdf_errors}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from NUST handbooks page by page")
    parser.add_argument("--input",  required=True, help="Directory containing PDF(s)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()
    extract_pdfs(args.input, args.output)