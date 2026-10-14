<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=800&size=40&pause=1000&color=2563EB&center=true&vCenter=true&width=800&lines=AcademIQ;Scalable+Academic+Policy+QA;Big+Data+Retrieval+Engine;Powered+by+LSH+%26+LLMs" alt="Typing SVG" />

> **A production-grade, Big Data Question-Answering system over NUST SEECS UG & PG Handbooks — combining MinHash LSH, SimHash, TF-IDF, MapReduce indexing, SON query mining, and LLM-backed answer synthesis in a single, auditable pipeline.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Anthropic-Claude-E07B39?style=for-the-badge" alt="Anthropic">
  <img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge" alt="License">
</p>

</div>

---

## Table of Contents

<details>
<summary>Click to expand</summary>

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Full Data Pipeline](#full-data-pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Component Deep-Dives](#component-deep-dives)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. MapReduce Indexer](#2-mapreduce-indexer)
  - [3. SON Frequent Query Miner](#3-son-frequent-query-miner)
  - [4. MinHash + LSH Retriever](#4-minhash--lsh-retriever)
  - [5. SimHash Retriever](#5-simhash-retriever)
  - [6. TF-IDF Retriever](#6-tf-idf-retriever)
  - [7. Hybrid Retriever & Reciprocal Rank Fusion](#7-hybrid-retriever--reciprocal-rank-fusion)
  - [8. Chunk Re-ranker](#8-chunk-re-ranker)
  - [9. Answer Generator](#9-answer-generator)
  - [10. Preprocessing Utilities](#10-preprocessing-utilities)
  - [11. Query Cache](#11-query-cache)
  - [12. Metrics Collector](#12-metrics-collector)
  - [13. Web Interface](#13-web-interface)
- [Key Algorithms Explained](#key-algorithms-explained)
- [Data Formats](#data-formats)
- [Configuration Reference](#configuration-reference)
- [Experiments & Results](#experiments--results)
- [Dependencies](#dependencies)
- [Sample Queries](#sample-queries)
- [Important Notes](#important-notes)

</details>

---

## Overview

**AcademIQ** is a rigorous, scalable **Question-Answering retrieval system** built for NUST (National University of Sciences and Technology) academic policy documents. It answers natural-language questions like *"What is the minimum CGPA to avoid probation?"* by precisely locating the relevant sections in the official UG and PG handbooks and synthesizing a grounded, cited answer.

The system is **not a chatbot**. It is a multi-stage Big Data retrieval pipeline where the LLM is the last step — it can only synthesize what the retriever finds. If the answer is not in the handbook chunks, the system explicitly says so. This guarantees **hallucination-free, auditable responses** suitable for official academic policy checking.

### Why four retrieval methods?

| Problem | Solution |
|---|---|
| Exact keyword lookup is slow at scale (O(n)) | MinHash LSH enables **sub-linear** approximate retrieval (O(log n)) |
| Set similarity (MinHash) ignores term importance | **Weighted MinHash** (TF-IDF weights per token) improves relevance |
| Jaccard-based similarity misses near-duplicate paraphrases | **SimHash** fingerprinting captures near-duplicates via Hamming distance |
| Approximate methods have a precision gap vs exact | **TF-IDF cosine similarity** as an exact baseline corrects this |
| No single method wins on every query | **Reciprocal Rank Fusion** (hybrid mode) combines all three for best results |

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                            AcademIQ Full Pipeline                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       OFFLINE INDEXING PHASE                        │   │
│  │                                                                      │   │
│  │  PDF Handbooks (UG + PG)                                            │   │
│  │         │                                                            │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────┐                                                   │   │
│  │  │ pdf_extractor│  pdfplumber → page-by-page extraction             │   │
│  │  │              │  Heading detection · Noise filtering              │   │
│  │  └──────┬───────┘                                                   │   │
│  │         │  raw_pages.jsonl                                           │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────┐                                                   │   │
│  │  │   chunker    │  Sliding window · Chapter boundaries              │   │
│  │  └──────┬───────┘  350 words / 50 overlap / 100 min               │   │
│  │         │  chunks.jsonl                                             │   │
│  │         ▼                                                            │   │
│  │  ┌─────────────────────────────────────────────────────┐           │   │
│  │  │              MapReduce Indexer (4 workers)          │           │   │
│  │  │  Partition → Map (parallel) → Shuffle → Reduce      │           │   │
│  │  │  ┌──────────────┐  ┌────────────┐  ┌────────────┐  │           │   │
│  │  │  │ minhash_index│  │simhash_idx │  │tfidf_index │  │           │   │
│  │  │  │   (.pkl)     │  │  (.pkl)    │  │  (.pkl)    │  │           │   │
│  │  │  └──────────────┘  └────────────┘  └────────────┘  │           │   │
│  │  │             ┌──────────────────┐                    │           │   │
│  │  │             │ reranker_index   │                    │           │   │
│  │  │             │    (.pkl)        │                    │           │   │
│  │  │             └──────────────────┘                    │           │   │
│  │  └─────────────────────────────────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ONLINE QUERY PHASE                           │   │
│  │                                                                      │   │
│  │  User Query (natural language)                                      │   │
│  │         │                                                            │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────────────────────────┐  ┌──────────────────────────┐ │   │
│  │  │         QueryCache (LRU+TTL)     │  │    SON Query Miner       │ │   │
│  │  │  Cache hit → return in <1ms      │  │   (cache pre-warming)    │ │   │
│  │  └──────────────────────────────────┘  └──────────────────────────┘ │   │
│  │         │ cache miss                                                 │   │
│  │         ▼                                                            │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │   preprocessing.tokenize() + expand_query()                  │   │   │
│  │  │   Stop-word filtering · Query expansion (33 trigger words)   │   │   │
│  │  └──────────────────────┬───────────────────────────────────────┘   │   │
│  │                         │                                            │   │
│  │            ┌────────────┼────────────┐                              │   │
│  │            ▼            ▼            ▼                              │   │
│  │  ┌──────────────┐ ┌──────────┐ ┌──────────────┐                   │   │
│  │  │  TF-IDF      │ │ MinHash  │ │   SimHash    │  (in parallel)    │   │
│  │  │  Retriever   │ │   LSH    │ │  Retriever   │                   │   │
│  │  │  cosine sim  │ │ Jaccard  │ │  Hamming dist│                   │   │
│  │  └──────┬───────┘ └────┬─────┘ └──────┬───────┘                   │   │
│  │         └──────────────┴──────────────┘                            │   │
│  │                         │                                            │   │
│  │                         ▼                                            │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │         HybridRetriever — Reciprocal Rank Fusion (RRF)       │   │   │
│  │  │  Fuses TF-IDF (w=1.0) + MinHash (w=0.8) + SimHash (w=0.6)  │   │   │
│  │  │  Strategies: rrf_all · rrf_lsh · cascade · adaptive         │   │   │
│  │  └──────────────────────────────┬───────────────────────────────┘   │   │
│  │                                 │                                    │   │
│  │                                 ▼                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │         ChunkReranker — Cosine sim re-ranking                │   │   │
│  │  │         + Diversity Penalty (−0.15 per repeated section)    │   │   │
│  │  └──────────────────────────────┬───────────────────────────────┘   │   │
│  │                                 │ top-K chunks                      │   │
│  │                                 ▼                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │         AnswerGenerator                                       │   │   │
│  │  │  LLM mode: OpenAI (gpt-4o-mini) / Anthropic (Claude)        │   │   │
│  │  │  Extractive fallback: sentence scoring + deduplication       │   │   │
│  │  └──────────────────────────────┬───────────────────────────────┘   │   │
│  │                                 │                                    │   │
│  │                                 ▼                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │     MetricsCollector — P50/P99 latency, cache hit rate,      │   │   │
│  │  │                        Precision@k                           │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                 │                                    │   │
│  │                                 ▼                                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │     Web App (FastAPI) — Answer · Sources · Atlas · Health    │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Full Data Pipeline

The pipeline has two phases: **offline indexing** (runs once, builds indexes from PDFs) and **online querying** (runs per user query, millisecond response).

### Phase 1 — Offline Indexing

```
1. PDF files placed in data/raw/
2. pdf_extractor.py reads each PDF with pdfplumber
   → detects headings (regex: "Chapter N", "Section N.M", all-caps lines)
   → strips noise (page numbers, headers, footers, blank lines)
   → emits one JSON record per page: {page_number, section_heading, raw_text, source_filename}
   → writes to data/processed/raw_pages.jsonl

3. chunker.py reads raw_pages.jsonl
   → merges consecutive pages from the same chapter into a buffer
   → flushes buffer when:  (a) buffer exceeds MAX_WORDS=500
                           (b) chapter boundary is crossed
                           (c) source PDF changes
   → applies 50-word sliding overlap between adjacent chunks
   → discards chunks shorter than MIN_WORDS=100
   → assigns chunk IDs: "handbook_p023_c001"
   → writes to data/processed/chunks.jsonl

4. mapreduce_indexer.py reads chunks.jsonl
   → partitions chunks into N_WORKERS=4 equal partitions
   → Map phase (parallel processes): compute MinHash signatures / SimHash fingerprints
   → Shuffle phase: flatten, group by chunk_id
   → Reduce phase: build LSH banding index / fingerprint store
   → saves minhash_index.pkl, simhash_index.pkl

5. tfidf_retriever.py.fit() reads chunks.jsonl
   → fits sklearn TfidfVectorizer (unigrams + bigrams, max_features=50000)
   → builds partition map (16 domains → section subsets for fast routing)
   → saves tfidf_index.pkl

6. recommender.py.fit() reads chunks.jsonl
   → fits a separate TF-IDF matrix (30k features) for re-ranking
   → saves reranker_index.pkl
```

### Phase 2 — Online Querying

```
1. User submits query via the web app (POST /api/ask) or CLI

2. QueryCache.get(key) — if hit, return cached result immediately (< 1ms)

3. preprocessing.tokenize(query) + expand_query(query)
   → lowercase, strip punctuation, filter stop words, length >= 2
   → expand: "gpa" → appends "cgpa grade point 2.00 required condition..."
   → result: enriched query string for better recall

4. Three retrievers run in the configured strategy:
   a. TFIDFRetriever.query()
      → partition-first routing (check domain match first)
      → cosine similarity against TF-IDF matrix
      → returns top_k×2 candidates with cosine scores
   b. MinHashLSHRetriever.query()
      → build MinHash signature for expanded query
      → LSH banding: only visit O(b) buckets instead of O(n) corpus
      → rank candidates by Jaccard similarity
      → budgeted fallback: scan 3×top_k max if LSH candidates sparse
   c. SimHashRetriever.query()
      → compute 64-bit SimHash fingerprint for query
      → scan all fingerprints for Hamming distance ≤ threshold
      → fallback to full ranked list if < top_k pass threshold

5. HybridRetriever.rrf_fuse()
   → strategy rrf_all: fuse all 3 methods
   → RRF formula: score += w × (1 / (k + rank + 1))   for each method
   → produce unified ranking independent of each method's score scale

6. ChunkReranker.rerank()
   → vectorize expanded query with re-ranker's TF-IDF
   → cosine similarity against candidate chunks
   → apply_diversity_penalty(): deduct 0.15 per repeated section
   → return top_k reranked chunks

7. AnswerGenerator.generate()
   → LLM mode: format chunks as numbered excerpts, call API
   → Extractive fallback: score sentences by keyword density, dedup
   → return {answer, mode, model, chunks_used, latency_ms}

8. MetricsCollector.record() — log latency breakdown, cache_hit, precision@k

9. QueryCache.put() — store result for future identical queries

10. Web app renders: cited answer, ranked sources, stage timings, star-chart atlas, health view
```

---

## Project Structure

```
AcademIQ/
│
├── config.yaml                        # Single source of truth for ALL hyperparameters
├── requirements.txt                   # 13 Python dependencies
├── README.md                          # This file
│
├── data/
│   ├── raw/                           # Put your PDF handbooks here
│   │   ├── ug_handbook.pdf
│   │   └── pg_handbook.pdf
│   └── processed/                     # Auto-generated by ingestion pipeline
│       ├── raw_pages.jsonl            # Extracted pages (one JSON per line)
│       ├── chunks.jsonl               # UG handbook chunks (one JSON per line)
│       ├── pg_chunks.jsonl            # PG handbook chunks
│       ├── tfidf_index.pkl            # UG TF-IDF index (sklearn TfidfVectorizer + matrix)
│       ├── pg_tfidf_index.pkl         # PG TF-IDF index
│       ├── minhash_index.pkl          # UG MinHash LSH index (signatures + banding)
│       ├── simhash_index.pkl          # UG SimHash fingerprints store
│       └── reranker_index.pkl         # Re-ranker TF-IDF matrix
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py           # PDF → raw_pages.jsonl (pdfplumber)
│   │   └── chunker.py                 # raw_pages.jsonl → chunks.jsonl (sliding window)
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── mapreduce_indexer.py       # MapReduce framework for index building
│   │   └── son_frequent_queries.py    # SON two-pass A-Priori query pattern miner
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── tfidf_retriever.py         # Exact baseline: TF-IDF + cosine similarity
│   │   ├── minhash_lsh.py             # Approximate: MinHash signatures + LSH banding
│   │   ├── simhash.py                 # Approximate: SimHash fingerprints + Hamming distance
│   │   └── hybrid_retriever.py        # Orchestrator: RRF fusion of all three methods
│   │
│   ├── ranking/
│   │   ├── __init__.py
│   │   └── recommender.py             # ChunkReranker: cosine re-ranking + diversity penalty
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── answer_generator.py        # LLM (OpenAI/Anthropic) + extractive fallback
│   │
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py           # Tokenization, stop words, query expansion (shared)
│       ├── config_loader.py           # YAML config reader with LRU cache
│       ├── cache.py                   # Thread-safe LRU + TTL query result cache
│       └── metrics_collector.py       # P50/P99 latency, cache hit rate, Precision@k
│
├── app/
│   ├── server.py                      # FastAPI server: JSON API + serves the web UI
│   └── web/                           # Hand-built frontend (Ask, Compare, Health, star-chart atlas)
│
├── experiments/
│   ├── exp1_exact_vs_approx.py        # TF-IDF vs MinHash vs SimHash accuracy & latency
│   ├── exp2_param_sensitivity.py      # Hyperparameter sweep (num_perm, threshold, bits)
│   └── exp3_scalability.py            # Latency scaling under 10x corpus duplication
│
└── tests/
    ├── conftest.py                    # Shared fixtures and skip decorators
    ├── test_chunker.py
    ├── test_tfidf_retriever.py
    ├── test_minhash_lsh.py
    ├── test_simhash.py
    ├── test_hybrid_retriever.py
    ├── test_recommender.py
    ├── test_answer_generator.py
    ├── test_cache.py
    ├── test_metrics_collector.py
    └── test_integration.py            # End-to-end pipeline tests
```

---

## Installation

### Prerequisites

- **Python** 3.10 or higher
- **pip** or **conda**
- **API key**: OpenAI (`OPENAI_API_KEY`) or Anthropic (`ANTHROPIC_API_KEY`) — only needed for LLM answer mode; extractive mode works without any key

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/academiq.git
cd academiq

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys (optional — only needed for LLM mode)
cp .env.example .env
# Open .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

---

## Usage

### Step 1 — Place PDFs

Each handbook gets its own folder (the extractor reads every PDF in the folder it's given):
```
data/raw/ug/ug_handbook.pdf
data/raw/pg/pg_handbook.pdf
```

### Step 2 — Ingest and Index

Output file names come from `paths` in `config.yaml`: the UG handbook uses the unprefixed files, the PG handbook the `pg_` ones.

```bash
# Undergraduate
python -m src.ingestion.pdf_extraction --input data/raw/ug --output data/processed/raw_pages.jsonl
python -m src.ingestion.chunker        --input data/processed/raw_pages.jsonl --output data/processed/chunks.jsonl
python -m src.retrieval.tfidf_retriever --chunks data/processed/chunks.jsonl --index data/processed/tfidf_index.pkl
python -m src.retrieval.minhash_lsh     --chunks data/processed/chunks.jsonl --index data/processed/minhash_index.pkl
python -m src.retrieval.simhash         --chunks data/processed/chunks.jsonl --index data/processed/simhash_index.pkl
python -m src.ranking.recommender       --chunks data/processed/chunks.jsonl --index data/processed/reranker_index.pkl

# Postgraduate
python -m src.ingestion.pdf_extraction --input data/raw/pg --output data/processed/pg_raw_pages.jsonl
python -m src.ingestion.chunker        --input data/processed/pg_raw_pages.jsonl --output data/processed/pg_chunks.jsonl
python -m src.retrieval.tfidf_retriever --chunks data/processed/pg_chunks.jsonl --index data/processed/pg_tfidf_index.pkl
python -m src.retrieval.minhash_lsh     --chunks data/processed/pg_chunks.jsonl --index data/processed/pg_minhash_index.pkl
python -m src.retrieval.simhash         --chunks data/processed/pg_chunks.jsonl --index data/processed/pg_simhash_index.pkl
python -m src.ranking.recommender       --chunks data/processed/pg_chunks.jsonl --index data/processed/pg_reranker_index.pkl
```

The parallel MapReduce build (`python -m src.indexing.mapreduce_indexer --chunks data/processed/chunks.jsonl --method minhash --workers 4`) writes `minhash_mr_index.pkl` / `simhash_mr_index.pkl` for the scalability experiments; the apps read the indexes above.

### Step 3 — Launch the Web App

```bash
python -m app.server
```

The app opens at `http://127.0.0.1:8000`. Pick the UG or PG handbook in the top bar, then ask a question. Each answer cites handbook section and page, and the star chart on the right shows where the retrieved chunks sit in the corpus. **Compare** runs one question through TF-IDF, MinHash + LSH, SimHash and Hybrid RRF side by side; **Health** shows live latency, cache and Precision@k for the session.

Options: `--port 8080`, `--host 0.0.0.0`, `--no-browser`. The JSON API is documented at `/api/docs`.

### Step 4 — Run Experiments (Optional)

```bash
python experiments/exp1_exact_vs_approx.py    # Accuracy comparison
python experiments/exp2_param_sensitivity.py  # Parameter sweep
python experiments/exp3_scalability.py        # Scalability stress test
```

### Step 5 — Run Tests

```bash
pytest tests/ -v
```

---

## Component Deep-Dives

---

### 1. Data Ingestion

**Files:** `src/ingestion/pdf_extractor.py`, `src/ingestion/chunker.py`

#### PDF Extractor (`pdf_extractor.py`)

The extractor reads handbook PDFs page-by-page using `pdfplumber` and produces a structured JSONL file.

**Key functions:**

- **`is_heading(line)`** — Detects chapter/section headings using regex patterns. Matches strings like `"Chapter 3"`, `"Section 2.4"`, or all-uppercase short lines. Used to track which section a page belongs to.

- **`is_noise(line)`** — Identifies and discards irrelevant lines: page numbers (standalone integers), running headers (university name repeated), footer text, and blank lines. Noise in PDFs significantly degrades retrieval quality if left in.

- **`clean_page_text(raw_text)`** — Applies `is_noise()` line-by-line, collapses consecutive blank lines, and trims whitespace.

- **`extract_pdfs(input_dir, output_file)`** — Main entry point. Iterates over all PDFs in `input_dir`, processes them page-by-page, tracks the current section heading as it changes, and writes one JSON record per page to `output_file`.

**Output format** (`raw_pages.jsonl`):
```json
{"page_number": 23, "section_heading": "Chapter 3: Award of Bachelor Degree", "raw_text": "...", "source_filename": "ug_handbook.pdf"}
```

**Error handling:** Per-page failure tracking (one bad page doesn't abort the whole PDF), JSON serialization guard, and validation before writing.

#### Chunker (`chunker.py`)

Raw pages are often too long for retrieval (one page ≈ 500–800 words) or too short (a section might span 3 pages). The chunker merges and splits pages to produce uniform, retrievable chunks.

**Algorithm:**

1. Load `raw_pages.jsonl`
2. Iterate over pages, accumulating them into a **buffer**
3. The buffer is flushed into a chunk when any of these conditions are met:
   - Buffer word count exceeds `MAX_WORDS` (≈500)
   - The chapter changes (detected via `get_chapter()` on the section heading)
   - The source PDF changes (UG vs PG)
4. After flushing, copy the last `OVERLAP_WORDS=50` words into the new buffer (sliding window overlap — ensures policy rules that span a page break are not lost)
5. Discard chunks shorter than `MIN_WORDS=100` (incomplete or header-only pages)

**Key helper — `get_chapter(section_heading)`:** Extracts the chapter number or name from a section heading string. Used to detect chapter boundaries (flush the buffer whenever the chapter changes, preventing one chunk from mixing two chapters' policies).

**Why overlap matters:** A policy might say "Students on probation (see page 47)... must achieve a CGPA of 2.00 by the end of the following semester." If the first part is at the end of one chunk and the second part at the start of the next, a query about probation CGPA requirements would only find the first half. The 50-word overlap ensures context continuity.

**Output format** (`chunks.jsonl`):
```json
{"id": "handbook_p023_c001", "text": "A student must maintain...", "page": 23, "section": "Chapter 3: Award of Bachelor Degree", "source": "ug_handbook.pdf", "word_count": 62}
```

---

### 2. MapReduce Indexer

**File:** `src/indexing/mapreduce_indexer.py`

**Class:** `MapReduceIndexer`

This module implements a **simulated MapReduce framework** using Python's `multiprocessing` to parallelize the computationally expensive index construction process. This demonstrates Big Data horizontal scaling principles.

#### Why MapReduce for index building?

Computing MinHash signatures requires hashing every token in every chunk with `num_perm=128` permutations. For a corpus of thousands of chunks, this is expensive. By distributing chunks across workers, the total wall-clock time drops proportionally to the number of cores.

#### Architecture

```
Input: chunks list
         │
         ▼
  ┌─────────────┐
  │  Partition  │  Split chunks into n_workers equal groups
  └──────┬──────┘
         │  n_workers partitions
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Map phase (multiprocessing.Pool)                       │
  │  Worker 1: _map_minhash(partition_1) → [(id, sig), ...] │
  │  Worker 2: _map_minhash(partition_2) → [(id, sig), ...]  │
  │  Worker 3: _map_minhash(partition_3) → [(id, sig), ...]  │
  │  Worker 4: _map_minhash(partition_4) → [(id, sig), ...]  │
  └──────────────────────────┬──────────────────────────────┘
         │  4 lists of (chunk_id, signature) pairs
         ▼
  ┌─────────────┐
  │   Shuffle   │  Flatten all lists, group by chunk_id
  └──────┬──────┘
         │  {chunk_id: signature, ...}
         ▼
  ┌─────────────┐
  │   Reduce    │  Build LSH index / fingerprint store
  └─────────────┘
```

**Methods:**

- **`build_minhash_index(chunks, num_perm, threshold)`** — Full MapReduce pipeline for MinHash. Returns a dict containing:
  - `lsh`: the `MinHashLSH` index object
  - `signatures`: {chunk_id → MinHash signature}
  - `chunks`: original chunk list
  - `chunk_order`: list of chunk IDs in order
  - `timings`: {partition_ms, map_ms, shuffle_ms, reduce_ms, total_ms}
  - `n_workers`: number of workers actually used

- **`build_simhash_index(chunks, bits, threshold)`** — Same pipeline for SimHash fingerprints.

**Worker selection:** `n_workers = min(requested_workers, cpu_count())` — never spawns more processes than there are CPU cores.

**Timing granularity:** Each phase (partition, map, shuffle, reduce) is timed individually. This lets the experiments measure which phase dominates at scale.

---

### 3. SON Frequent Query Miner

**File:** `src/indexing/son_frequent_queries.py`

**Class:** `SONFrequentQueries`

This implements the **SON (Savasere, Omiecinski, Navathe) algorithm** — a two-pass, partition-based frequent itemset mining algorithm. It analyzes historical query logs to find keyword patterns that appear together frequently.

#### Why mine query patterns?

- **Cache pre-warming:** If "gpa" and "probation" frequently co-occur in queries, pre-loading results for this query combination eliminates cold-cache latency for the most common requests.
- **Query expansion refinement:** Frequent co-occurrence of "cgpa" and "2.00" confirms these terms belong in the same expansion cluster.
- **Usage analytics:** Understand which policy topics students ask about most.

#### Algorithm: Two-Pass A-Priori

Standard A-Priori requires two full passes over the data. SON extends this to handle large datasets by partitioning them:

**Pass 1 (Local — runs in parallel):**
1. Divide the query log into `num_partitions=4` equal partitions
2. Run standard A-Priori on each partition independently with a **scaled threshold**: `local_threshold = min_support × partition_size / total_size`
3. Collect all locally frequent itemsets from all partitions into a **candidate set**

**Pass 2 (Global — single pass):**
1. Scan the full query log once
2. For each candidate itemset from Pass 1, count its true global support
3. Keep only candidates whose global count ≥ `min_support × total_queries`

**Why this works:** Any globally frequent itemset must be locally frequent in at least one partition (it appears often enough across the whole data that some partition must see it often). Local false positives are filtered in Pass 2.

**Methods:**

- **`mine(query_log)`** — Takes a list of query strings. Tokenizes each, runs two-pass SON, returns list of `{itemset, support, count, size}` sorted by support.

- **`suggest_cache_warmup(patterns, top_n)`** — Takes the mined patterns, takes the `top_n` most frequent, and generates synthetic query strings by joining the itemset tokens. These synthetic queries can be executed at startup to pre-populate the `QueryCache`.

**Constructor parameters:**
- `min_support`: Minimum fraction of queries an itemset must appear in (default 0.1 = 10%)
- `num_partitions`: Number of partitions for Pass 1 (default 4)
- `max_itemset_size`: Maximum n-gram size to mine (default 3)

---

### 4. MinHash + LSH Retriever

**File:** `src/retrieval/minhash_lsh.py`

Three classes: `MinHash`, `MinHashLSH`, `MinHashLSHRetriever`

#### Core Concept: Approximate Set Similarity

MinHash provides an unbiased estimator of the **Jaccard similarity** between two sets of tokens. If two chunks have similar vocabulary (many shared tokens), their MinHash signatures will collide in many positions. LSH then groups similar signatures into buckets so we only need to compare against a small candidate set — not every chunk.

#### Class: `MinHash`

Represents a single MinHash signature (a compact fingerprint of a token set).

- **Constructor:** `num_perm=128` — number of hash permutations. More permutations = more accurate Jaccard estimate, but larger signature.
- **`update(token)`** — Hash the token with each of the `num_perm` independent hash functions (simulated via random parameters `a`, `b`, `c`), update the minimum seen value for each permutation slot.
- **`jaccard(other)`** — Estimate Jaccard similarity as `proportion of matching slots between two signatures`. For 128 permutations, the standard error of this estimate is about 8.8%.

**Why minimum?** The key insight of MinHash: for a random permutation of the universe of tokens, the probability that the minimum element under that permutation is the same for two sets S and T equals `|S ∩ T| / |S ∪ T|` — which is exactly the Jaccard similarity. Using 128 independent permutations averages out the noise.

#### Class: `MinHashLSH`

Bands-and-rows LSH that enables sub-linear candidate retrieval.

- **`_choose_bands_and_rows(threshold, num_perm)`** — Solves for `(b, r)` such that `b × r = num_perm` and the detection threshold `(1/b)^(1/r) ≈ threshold`. This mathematical relationship is the core of LSH band theory. A pair of documents with Jaccard similarity `t` will share the same band bucket with probability `1 - (1 - t^r)^b`.
- **`insert(key, sig)`** — Splits the 128-dimensional signature into `b` bands of `r` rows each. Hashes each band's sub-signature to a bucket. Inserts `key` into `b` hash tables.
- **`query(sig)`** — For a query signature, hashes each of its `b` bands and retrieves all keys from matching buckets. Returns the union — a candidate set likely to be similar. This is O(b) = O(number of bands), not O(n) = O(corpus size).

#### Class: `MinHashLSHRetriever`

Top-level retriever that manages fit/save/load/query lifecycle.

- **`fit(chunks)`** — Builds the MinHash index. If `use_weighted=True`, first computes corpus-level IDF weights for each token, then uses **weighted MinHash**: each token is hashed multiple times proportional to its TF-IDF weight. This makes rare, informative tokens (like "cgpa", "xf", "probation") dominate the signature, improving relevance over uniform token weighting.

- **`query(q, top_k)`** — Full query pipeline:
  1. `expand_query(q)` — Append synonym expansions
  2. `tokenize()` — Shared canonical tokenizer (length ≥ 2)
  3. Build MinHash signature for the expanded query
  4. LSH candidate retrieval via `MinHashLSH.query()`
  5. Rank candidates by Jaccard similarity (estimated via signature comparison)
  6. **Budgeted fallback:** If LSH returns fewer than `top_k` candidates (happens when the query is very short or vocabulary mismatches), scan up to `3 × top_k` chunks directly (not the full corpus, just a small budget)
  7. Return `top_k` results with scores and `latency_ms`

- **`save(path)` / `load(path)`** — Pickle the entire fitted object for fast startup.

---

### 5. SimHash Retriever

**File:** `src/retrieval/simhash.py`

**Classes:** `SimHashRetriever` + module-level functions `simhash()`, `hamming_distance()`, `hamming_similarity()`

#### Core Concept: Fingerprint-Based Near-Duplicate Detection

SimHash generates a **64-bit fingerprint** for each chunk. Two chunks with similar content will have fingerprints with small Hamming distance (few differing bits). This is particularly strong at detecting paraphrased or near-duplicate policy text.

#### Function: `simhash(tokens, bits=64)`

The core algorithm:

1. Initialize a voting array of `bits=64` integers, all zero.
2. For each token in the input:
   a. Compute its `mmh3.hash128(token)` — a 128-bit MurmurHash3 hash, truncated to `bits` bits.
   b. For each bit position `i`: if the hash's `i`-th bit is 1, add +1 to vote[i]; otherwise add −1.
3. Final fingerprint: for each position `i`, if `vote[i] > 0` → bit is 1; else bit is 0.

The result: tokens that frequently co-occur push certain bits towards 1 together. Two texts with similar token distributions will vote similarly, producing similar fingerprints.

#### Function: `hamming_distance(h1, h2)`

Uses the Brian Kernighan bit-counting trick: `xor = h1 ^ h2`, then repeatedly clear the lowest set bit of `xor` while incrementing a counter. This counts only the differing bits efficiently.

#### Function: `hamming_similarity(h1, h2)`

`1.0 - (hamming_distance(h1, h2) / bits)` — normalized to [0, 1].

#### Class: `SimHashRetriever`

- **`fit(chunks)`** — Computes a SimHash fingerprint for each chunk using its tokenized text. Tracks genuinely empty chunks separately to avoid false positives (a zero fingerprint would match other zero fingerprints incorrectly).

- **`query(q, top_k, hamming_threshold=None)`** — Query pipeline:
  1. Expand + tokenize query
  2. Compute SimHash fingerprint for the query
  3. Scan all chunk fingerprints (O(n), but integer XOR is extremely fast — even 10,000 chunks completes in <5ms)
  4. Filter chunks with `hamming_distance ≤ hamming_threshold` (default 10 out of 64 bits = ≥84% similarity)
  5. **Fallback:** If fewer than `top_k` chunks pass the threshold, fall back to the full ranked list ordered by Hamming similarity
  6. Return `top_k` results with scores, raw `hamming_distance`, and `latency_ms`

**Why SimHash vs MinHash?** MinHash uses LSH for sub-linear retrieval but requires a distinct index structure. SimHash scans linearly but is extremely cache-friendly (integer XOR of fixed-size values). SimHash particularly excels at detecting near-duplicate policy restatements scattered across different chapters.

---

### 6. TF-IDF Retriever

**File:** `src/retrieval/tfidf_retriever.py`

**Class:** `TFIDFRetriever`

The **exact baseline retriever**. Every chunk is represented as a sparse high-dimensional vector of TF-IDF weights. Query retrieval is cosine similarity in this vector space.

#### TF-IDF Fundamentals

- **TF (Term Frequency):** How often does term `t` appear in chunk `d`? With `sublinear_tf=True`, this uses `1 + log(tf)` instead of raw `tf` to prevent very long chunks from dominating.
- **IDF (Inverse Document Frequency):** `log(N / df_t)` where `N` is total chunks and `df_t` is the number of chunks containing term `t`. Rare terms (like "xf", "probation") get high IDF; common terms (like "student", "semester") get low IDF.
- **TF-IDF weight:** `TF(t, d) × IDF(t)` — high for terms that appear often in this chunk but rarely across all chunks.
- **Cosine similarity:** `dot(query_vec, chunk_vec) / (|query_vec| × |chunk_vec|)` — measures the angle between vectors, independent of chunk length.

**Vectorizer configuration (from `config.yaml`):**
- `ngram_range: [1, 2]` — both unigrams ("gpa") and bigrams ("minimum gpa") are indexed. Bigrams capture phrase-level matching.
- `max_features: 50000` — vocabulary cap to prevent memory blowup
- `min_df: 1` — keep all terms (small corpus, no minimum document frequency needed)
- `max_df: 0.85` — drop terms that appear in more than 85% of chunks (too common to be informative)
- `sublinear_tf: True` — log-scale TF dampening

#### Partition-First Search

The `TFIDFRetriever` maintains a **domain partition map** of 16 policy domains:

| Domain | Matched Section Keywords |
|---|---|
| `gpa` | academic standards, award, degree requirements |
| `plagiarism` | unfair means, code of conduct, academic integrity |
| `attendance` | attendance, absence, leave |
| `fees` | fee structure, refund, financial |
| `withdrawal` | course withdrawal, drop, add/drop |
| `grading` | grade distribution, grading policy, transcript |
| ... | ... (16 total) |

When a query is submitted:
1. `_detect_domain(query)` scans query tokens for domain keywords
2. If a domain match is found, cosine similarity is computed **only against chunks in that partition**
3. If the partition result is weak (top score < threshold), fall back to the full corpus

This provides a **2–5x speedup** for domain-specific queries while maintaining full accuracy via the fallback.

**Methods:**

- **`fit(chunks)`** — Fit `TfidfVectorizer` on all chunk texts, build sparse TF-IDF matrix, build partition map.
- **`query(q, top_k)`** — Expand query, vectorize, partition-first cosine similarity, return top_k with scores and `latency_ms`.
- **`save(path)` / `load(path)`** — Pickle fitted vectorizer + matrix.

---

### 7. Hybrid Retriever & Reciprocal Rank Fusion

**File:** `src/retrieval/hybrid_retriever.py`

**Class:** `HybridRetriever`

The orchestrator that combines all three retrieval methods via **Reciprocal Rank Fusion (RRF)**.

#### Why Fusion?

Each retrieval method has different strengths:
- **TF-IDF**: Highest precision for exact keyword matching; slow at scale
- **MinHash**: Excellent for set-similarity, near-duplicates, paraphrases; fast via LSH
- **SimHash**: Excellent fingerprint-based near-duplicate detection; very fast integer ops

A query like "minimum CGPA for graduation" benefits from TF-IDF (exact phrase match) but also from MinHash/SimHash (the answer section may paraphrase the same rule in multiple ways). No single method wins on every query.

#### Reciprocal Rank Fusion (RRF)

RRF (Cormack, Clarke, Buettcher 2009) is a proven rank fusion algorithm that combines ranked lists without requiring score normalization.

**Formula:**
```
RRF_score(chunk) = Σ over methods:  w_method × (1 / (k + rank + 1))
```

- `k = 60` (RRF constant — smooths rank differences, standard value from the 2009 paper)
- `rank` = 0-indexed position in the method's result list (rank 0 = most relevant)
- `w_method` = per-method weight: TF-IDF 1.0, MinHash 0.8, SimHash 0.6

**Why `1 / (k + rank + 1)`?** This assigns high reward to top-ranked results and diminishing reward to lower ranks. A chunk ranked #1 by TF-IDF gets score `1 / (60 + 0 + 1) = 0.0164`. A chunk ranked #10 gets `1 / (60 + 9 + 1) = 0.0143`. The difference is small but consistent — preventing any single outlier rank from dominating.

**Why this beats score averaging?** TF-IDF cosine scores range from 0.05–0.30. MinHash Jaccard scores range from 0.03–0.15. SimHash Hamming similarity ranges from 0.80–1.00. You cannot average these directly — they're on different scales. RRF uses only rank position, which is scale-invariant.

#### Fusion Strategies

- **`rrf_all`** (default): Fuse all three methods. Best overall precision.
- **`rrf_lsh`**: Fuse only MinHash + SimHash. Fastest (no TF-IDF scan). Good for high-frequency queries.
- **`cascade`**: Run TF-IDF first. If the top score exceeds a confidence threshold, return it directly. Otherwise, run LSH methods and fuse. Best latency for confident queries.
- **`adaptive`**: Route automatically based on query length. Short queries (≤3 tokens) → TF-IDF only (vocabulary-dense). Long queries (>3 tokens) → RRF all three (richer signal).

**`HybridRetriever.retrieve(query, top_k, strategy, use_reranker)`** — Main entry point. Runs the selected strategy, records per-method timing, optionally passes results through `ChunkReranker`, returns `(chunks, timings_dict)`.

---

### 8. Chunk Re-ranker

**File:** `src/ranking/recommender.py`

**Class:** `ChunkReranker`

A **two-stage retrieval pattern**: coarse retrieval (any of the three methods) finds a broad candidate set; the re-ranker precisely re-orders them.

#### Why a Separate Re-ranker?

The retrieval methods optimize for recall (find all possibly relevant chunks). The re-ranker optimizes for precision (surface the most relevant ones at the top). They use separate TF-IDF matrices:
- Retrieval TF-IDF: `max_features=50000`, full corpus vocabulary — maximizes recall
- Re-ranker TF-IDF: `max_features=30000`, `ngram_range=(1,2)` — optimized for fine-grained similarity

#### `fit(chunks)`

Fits a dedicated `TfidfVectorizer` on chunk texts. Builds `_id_to_idx` dict mapping chunk IDs to matrix row indices — O(1) lookup vs O(n) `list.index()` calls.

#### `rerank(query, candidate_chunks, top_k)`

1. **Expand query** using `expand_query()` before vectorizing (uses the expanded form, not raw query — important for synonym coverage)
2. Vectorize expanded query with the re-ranker's fitted vectorizer
3. Compute cosine similarity of query vector against only the candidate chunk vectors (not the full corpus — efficient)
4. Sort by cosine similarity descending
5. **Returns a new list** (does not mutate the input list — important for hybrid strategies that reuse candidate lists)

#### `apply_diversity_penalty(chunks, penalty=0.15)`

After re-ranking, this function penalizes redundancy:

1. Count how many times each section has already appeared in the result list
2. For each chunk, deduct `penalty × count` from its `relevance_score`
3. Return a new list with adjusted scores

**Example:** If chunks from "Chapter 3: Academic Standards" appear at ranks 1, 2, and 3, the third occurrence gets score reduced by 0.30. This pushes it below a chunk from "Chapter 5: Disciplinary Policy" that covers a different angle of the same topic.

**Why this matters for LLM generation:** The LLM prompt is limited (max_tokens=800). Feeding three chunks that all say "CGPA must be ≥ 2.00" wastes context. Diversity penalty ensures the top-k chunks cover different policy dimensions, giving the LLM richer material for a comprehensive answer.

---

### 9. Answer Generator

**File:** `src/generation/answer_generator.py`

**Class:** `AnswerGenerator`

The final step: synthesizes a natural-language answer from the retrieved chunks.

#### Constructor — Backend Auto-Detection

On initialization, the generator checks for API keys in environment variables:
1. If `OPENAI_API_KEY` found → `_backend = "openai"`, `mode = "llm"`
2. Elif `ANTHROPIC_API_KEY` found → `_backend = "anthropic"`, `mode = "llm"`
3. Else → `mode = "extractive"` (no API key needed)

Config parameters read from `config.yaml`:
- `model`: which LLM to use (default `gpt-4o-mini`)
- `max_tokens`: maximum response length (default 800)
- `temperature`: randomness (default 0.1 — low for factual, deterministic answers)

#### LLM Mode (`_generate_llm`)

1. Format the top-K chunks as numbered excerpts:
   ```
   [1] Section: Chapter 3 | Page: 47 | Source: ug_handbook.pdf
   "A student must maintain a minimum CGPA of 2.00..."

   [2] Section: Chapter 4 | Page: 62 | Source: ug_handbook.pdf
   "Failure to meet the CGPA requirement results in academic probation..."
   ```
2. Prepend a **system prompt** that instructs the LLM:
   - Answer ONLY using the provided excerpts
   - If the answer is not present, say "I could not find this information in the provided sections"
   - Be concise and direct
   - Always cite the source section and page number
   - Note any contradictions between excerpts
3. Call the API (`openai.chat.completions.create` or `anthropic.messages.create`)
4. On API error: automatically fall back to extractive mode (no crash)

#### Extractive Mode (`_generate_extractive`)

Works entirely offline — no API key, no network call.

1. Build a keyword set from both the original query AND the expanded query
2. For **every chunk** (not just the first — this was a critical bug fix), split text into sentences using regex boundary detection (handles abbreviations like "2.00", "B.S." correctly — not broken by naive `split('.')`)
3. Score each sentence by **keyword density**: `hit_count / sentence_length`. This rewards short, precise sentences that contain many query keywords over long verbose sentences with few relevant terms.
4. **Deduplication:** Skip sentences that are near-duplicates of already-selected sentences (prevents repetition in the answer)
5. Return the top 5 scoring sentences concatenated, plus source citations.

#### Output Format

```python
{
    "answer": "Students must maintain a minimum CGPA of 2.00... (Section: Chapter 3, Page: 47)",
    "mode": "llm",          # or "extractive"
    "model": "gpt-4o-mini", # or "extractive" for extractive mode
    "chunks_used": 3,
    "latency_ms": 847.3
}
```

---

### 10. Preprocessing Utilities

**File:** `src/utils/preprocessing.py`

This is the **single source of truth** for all text processing across every retrieval method. All modules import from here, ensuring consistent tokenization.

#### Stop Words

`STOP_WORDS` is a `frozenset` of 53 terms:
- 42 standard English stop words ("the", "a", "is", "in", ...)
- 11 NUST-specific high-frequency terms that appear in almost every policy section and therefore carry no discriminative power: "university", "student", "nust", "seecs", "policy", "academic", "degree", "program", "course", "semester", "regulation"

#### Query Expansion Dictionary

`QUERY_EXPANSION` maps 33 trigger words to synonym lists. These mappings encode **domain knowledge** about how NUST policies are phrased:

```python
{
    "gpa":         ["cgpa", "grade point", "2.00", "cumulative", "required", "least", "not less than", "must", "requirement", "condition"],
    "fail":        ["f grade", "xf", "unsatisfactory", "repeat", "incomplete", "withdrawal"],
    "plagiarism":  ["unfair means", "academic dishonesty", "cheating", "copied", "misconduct"],
    "probation":   ["academic warning", "warning", "cgpa below", "minimum requirement", "suspended"],
    "attendance":  ["absent", "absence", "75%", "percentage", "lecture", "missed"],
    ...  # 33 triggers total
}
```

**Example expansion:** Query "minimum GPA requirement" → tokenized → "gpa" trigger found → expands to "minimum GPA requirement cgpa grade point 2.00 cumulative required least not less than must requirement condition". The TF-IDF retriever now matches chunks containing "CGPA of 2.00" even though the user typed "GPA".

#### `tokenize(text, as_set=True)`

Canonical tokenizer used by all retrieval methods:
1. Lowercase
2. Replace all non-alphanumeric characters with spaces (preserves digits: "75", "2.00" → "2" "00")
3. Split on whitespace
4. Filter: length **≥ 2** (not > 2 — critical fix that preserves "xf", "75", "ch", "ab" grade codes)
5. Filter: not in `STOP_WORDS`
6. Return `set` (for MinHash/SimHash set operations) or `list` (for TF-weighted operations)

#### `expand_query(q)`

1. Tokenize query
2. For each token, check if it's a key in `QUERY_EXPANSION`
3. Collect all expansion terms, deduplicate (each term added at most once — prevents TF inflation where a term appears 5 times and artificially dominates the TF-IDF weight)
4. Return `q + " " + " ".join(expansion_terms)`

---

### 11. Query Cache

**File:** `src/utils/cache.py`

**Class:** `QueryCache`

A **thread-safe, TTL-aware LRU cache** for query results. Repeated identical queries (same query text, same method, same top_k) return the cached result instantly without re-running the full retrieval pipeline.

#### Data Structure

```python
_store: OrderedDict[str, (timestamp, value)]
_lock: threading.Lock
```

`OrderedDict` provides O(1) move-to-end (LRU update), O(1) eviction of the least-recently-used item (pop from the front), and O(1) get/set.

#### Cache Key

`make_key(query, method, top_k, extra=None)` computes `SHA-256(query + method + str(top_k) + str(extra))` — a deterministic 64-char hex string. Different handbooks (UG vs PG) produce different keys because the handbook name is included in `extra`.

#### Operations

- **`get(key)`** — Returns the cached value if the entry exists AND its age is < `ttl_seconds`. Moves the entry to the end of the `OrderedDict` (most recently used). Returns `None` on miss or expiry.
- **`put(key, value)`** — Inserts or updates the entry with the current timestamp. If at `max_size`, evicts the least-recently-used item (front of `OrderedDict`) before inserting.
- **`hit_rate`** — `hits / (hits + misses)` — reported on the health dashboard.
- **`clear()`** — Flush all entries (used when indexes are rebuilt).

**Configuration:** `max_size=256` (fits ~256 distinct queries), `ttl_seconds=3600` (1 hour expiry — prevents stale answers if indexes are rebuilt).

---

### 12. Metrics Collector

**File:** `src/utils/metrics_collector.py`

**Classes:** `QueryRecord` (dataclass), `MetricsCollector` (singleton)

Tracks performance metrics across all queries for the live health dashboard.

#### `QueryRecord` (dataclass)

One record per query execution:
```python
@dataclass
class QueryRecord:
    query: str
    method: str           # "hybrid", "tfidf", "minhash", "simhash"
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    cache_hit: bool
    num_chunks: int       # How many chunks were used in the answer
    precision_k: float    # Real-time Precision@k estimate
    timestamp: float      # time.time()
```

#### `MetricsCollector`

Singleton instance `metrics` imported by all modules.

- **`record(...)`** — Appends a `QueryRecord` to an in-memory deque of `max_history=500` records. Thread-safe via lock.
- **`p50(field)` / `p99(field)` / `avg(field)`** — Compute latency percentiles over all recorded values for a given field. P99 is especially important: it reveals tail latency that P50 hides.
- **`total_queries`** — Count of all queries since startup.
- **`cache_hit_rate`** — Fraction of queries that were cache hits.
- **`method_breakdown()`** — Dict of `{method: count}` — shows which retrieval methods users prefer.
- **`precision_k_stats()`** — Aggregates the `precision_k` field across all records to compute P50, P99, and average Precision@k in real-time.
- **`summary()`** — Full metrics dict for the web app's Health view.
- **`recent(n)`** — Last `n` `QueryRecord` objects — used for the recent queries table.

#### Real-Time Precision@k Computation

Without manual relevance labels, AcademIQ computes a proxy Precision@k: the fraction of the top-k retrieved chunks whose retrieval score exceeds a relevance threshold (default 0.05). This is not a perfect measure but correlates well with actual precision in practice, and enables automatic quality monitoring without human annotation.

---

### 13. Web Interface

**Files:** `app/server.py`, `app/web/` (`index.html`, `app.js`, `atlas.js`, `styles.css`, self-hosted fonts)

A FastAPI server wraps the pipeline (retrievers, re-ranker, answer generator, cache, metrics) in a small JSON API and serves a hand-built frontend with no framework and no build step. Start it with `python -m app.server`; it opens at `http://127.0.0.1:8000`.

#### Views

- **Ask** — question box with Method (TF-IDF, MinHash + LSH, SimHash, Hybrid all three, Hybrid LSH only, Cascade), Top-k and Answer mode (Extractive, or LLM when an API key is set). The answer carries a numbered citation on every sentence, followed by a stage-timing breakdown (retrieval per method, re-rank, generate) and the ranked sources.
- **Compare** — runs one question through TF-IDF, MinHash + LSH, SimHash and Hybrid RRF side by side, with per-method timings and a rank-agreement table showing where each method placed each section.
- **Health** — latency per query, P50/P99 latency, query-cache hit rate, Precision@k, method usage and a recent-queries log for the current server session.

The UG/PG switch in the header picks the handbook; each one loads its own indexes from the `paths` in `config.yaml`. If a handbook's indexes are missing or fail to load, the view says so and lists the commands to rebuild them.

#### Star-Chart Atlas

The right side of Ask and Compare draws the handbook as a star chart. Every chunk is a star placed by its TF-IDF vector, projected to two dimensions (TruncatedSVD, then PCA). A question goes through the same two transforms, so it lands next to the chunks it resembles; star size shows TF-IDF similarity to the question, and retrieved chunks are ringed and numbered by rank. Hovering a source lights its star and vice versa. In Compare, each method's picks are marked on the same sky with their own shape.

#### API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Handbook availability, answer backends, config |
| `GET /api/atlas?handbook=` | 2-D projection of every chunk (the star chart) |
| `POST /api/ask` | Retrieve + re-rank + generate, fully timed |
| `POST /api/compare` | One query across TF-IDF / MinHash / SimHash / Hybrid |
| `GET /api/health` | P50/P99 latency, cache, Precision@k, recent queries |
| `POST /api/cache/clear` | Empty the query cache |

Interactive API docs are served at `/api/docs`.

---

## Key Algorithms Explained

### Weighted MinHash

Standard MinHash assigns equal weight to every token. This means rare but informative terms like "xf" (a specific NUST grade code) get the same weight as common terms like "policy".

Weighted MinHash (used when `minhash.weighted: true` in config) extends this: before building the MinHash signature, it computes the **corpus-level IDF weight** for each token. Then, instead of hashing each token once, it hashes each token **⌈weight × N⌉ times** (where N is a scaling constant). Tokens with high IDF weight (rare, informative) contribute more to the signature.

**Effect:** MinHash similarity between a query about "XF grades" and the "Grades and Grading Policy" section increases, while the similarity to a general "Student Handbook Overview" section (which mentions every topic) decreases.

### Partition-First TF-IDF Search

When a user asks "what is the GPA for probation", the relevant answer is in 1–2 sections, not all 500+ chunks. Searching all 500 chunks takes 300ms. Searching only the 40 chunks in the "Academic Standards" and "Degree Requirements" partitions takes 20ms.

The partition map is built at fit-time by scanning section headings for domain keywords. At query time, `_detect_domain()` scans query tokens for domain triggers. If a match is found, only the partition matrix is searched. If the partition's top score is below a confidence threshold, the full corpus is searched as a fallback.

### SON Two-Pass A-Priori

A-Priori is the classic frequent itemset algorithm: Pass 1 counts individual items, Pass 2 counts only candidate pairs (items that were frequent individually). This avoids counting all O(n²) pairs.

SON extends A-Priori to large datasets by partitioning: Pass 1 runs A-Priori independently on each partition (finding **locally** frequent itemsets), then Pass 2 counts only those locally-frequent candidates against the full dataset.

**Correctness:** Any globally frequent itemset with support ≥ s must have support ≥ s/p in at least one partition of p items, because if it appeared in fewer than s/p items in every partition, it would appear in fewer than p × (s/p) = s items globally. Therefore, no globally frequent itemset is missed.

### Reciprocal Rank Fusion

RRF is provably robust: it assigns reward `1 / (k + r)` to rank position `r`, where `k` (typically 60) dampens the reward curve. This means:
- Rank 1 reward: `1/61 ≈ 0.0164`
- Rank 10 reward: `1/70 ≈ 0.0143`
- Rank 100 reward: `1/161 ≈ 0.0062`

The gentle curve prevents a chunk that ranks #1 in one method but #50 in another from being unfairly penalized by the poor ranking. An item ranked in the top-5 by all three methods reliably floats to the top of the fused list.

### Diversity Penalty

After re-ranking, the top-k results might all be from the same section (e.g., three consecutive chunks from "Chapter 3: Academic Standards" all discussing CGPA). While relevant, this wastes the LLM's context window.

For each chunk in the result list, the penalty is: `relevance_score -= penalty × times_this_section_appeared_earlier`. With penalty=0.15, a section appearing three times loses 0.30 score on its third occurrence — enough to push it below a chunk from a different section with slightly lower raw score.

### Real-Time Precision@k

Without a labeled test set at runtime, precision is estimated as: the fraction of the top-k retrieved chunks whose retrieval score exceeds a threshold (0.05). While imperfect, this proxy metric:
- Correlates with actual precision (high-scoring chunks are almost always relevant)
- Enables automatic quality monitoring without any manual annotation
- Allows the health dashboard to show quality trends over time

---

## Data Formats

### `raw_pages.jsonl` — Extracted Pages

One JSON object per line, one line per PDF page:

```json
{
    "page_number": 23,
    "section_heading": "Chapter 3: Award of Bachelor Degree",
    "raw_text": "3.1 Minimum Requirements\nA student must maintain a minimum CGPA...",
    "source_filename": "ug_handbook.pdf"
}
```

### `chunks.jsonl` — Retrieval Chunks

One JSON object per line, one line per chunk:

```json
{
    "id": "handbook_p023_c001",
    "text": "A student must maintain a minimum CGPA of 2.00 on a 4.00 scale in order to qualify for the award of a bachelor degree. Students whose CGPA falls below 2.00 are placed on academic probation.",
    "page": 23,
    "section": "Chapter 3: Award of Bachelor Degree",
    "source": "ug_handbook.pdf",
    "word_count": 42
}
```

### Retriever Output Format

All three retrievers and the hybrid return a list of dicts in this unified format:

```python
[
    {
        "id": "handbook_p023_c001",
        "text": "A student must maintain a minimum CGPA of 2.00...",
        "page": 23,
        "section": "Chapter 3: Award of Bachelor Degree",
        "source": "ug_handbook.pdf",
        "word_count": 42,
        # Similarity score (method-specific):
        "score": 0.85,          # Jaccard similarity (MinHash)
                                # Hamming similarity (SimHash)
                                # Cosine similarity (TF-IDF)
                                # RRF score (Hybrid)
        "latency_ms": 12.4,
        # Optional — added by re-ranker:
        "relevance_score": 0.82,
        # Optional — added by hybrid RRF:
        "rrf_contributions": {
            "tfidf": 0.0125,
            "minhash": 0.0098,
            "simhash": 0.0064
        }
    },
    ...
]
```

### Answer Generator Output

```python
{
    "answer": "According to Chapter 3 (Page 47), students must maintain a minimum CGPA of 2.00...",
    "mode": "llm",            # "llm" or "extractive"
    "model": "gpt-4o-mini",   # model used, or "extractive"
    "chunks_used": 3,         # number of chunks fed to the generator
    "latency_ms": 847.3       # time for answer generation alone
}
```

---

## Configuration Reference

All pipeline parameters live in a single `config.yaml`. Every module reads from here via `cfg(section, key, default)` — a cached YAML loader. Changing a value in `config.yaml` takes effect immediately (after index rebuild if an indexing parameter changes).

```yaml
# ─── Ingestion ───────────────────────────────────────────────────────────────
ingestion:
  chunk_size: 350        # Target words per chunk (buffer flush threshold)
  chunk_overlap: 50      # Words of overlap between adjacent chunks (sliding window)
  min_chunk_size: 100    # Discard chunks shorter than this (incomplete pages)

# ─── MinHash LSH ─────────────────────────────────────────────────────────────
minhash:
  num_perm: 128          # Hash permutations per signature.
                         # Higher = more accurate Jaccard estimate (±8.8% at 128).
                         # Tradeoff: 2× permutations → 2× signature size + build time.
  threshold: 0.3         # Jaccard similarity threshold for LSH band tuning.
                         # Lower = more candidates retrieved (higher recall, lower precision).
  weighted: true         # Use TF-IDF weighted MinHash.
                         # true: rare terms dominate signature (better relevance).
                         # false: uniform token weighting (faster, less relevant).

# ─── SimHash ─────────────────────────────────────────────────────────────────
simhash:
  bits: 64               # Fingerprint size in bits. 64 is standard; 128 for more precision.
  hamming_threshold: 10  # Maximum Hamming distance to consider two chunks similar.
                         # 10 of 64 bits = ≥84% similarity. Lower = stricter matching.

# ─── TF-IDF ──────────────────────────────────────────────────────────────────
tfidf:
  ngram_range: [1, 2]    # Unigrams + bigrams. [1,1] for unigrams only (faster, less precise).
  max_features: 50000    # Vocabulary size cap. Larger = more coverage, more memory.
  min_df: 1              # Minimum document frequency. 1 = keep all terms (small corpus).
  max_df: 0.85           # Drop terms appearing in >85% of chunks (too common).
  sublinear_tf: true     # Log-scale TF: 1 + log(tf). Prevents long chunks dominating.

# ─── Retrieval ────────────────────────────────────────────────────────────────
retrieval:
  top_k: 5               # Final number of chunks returned to the user.
  method: "hybrid"       # Default method: tfidf | minhash | simhash | hybrid
  candidates_multiplier: 2  # Retrieve top_k × 2 candidates before re-ranking.

# ─── Re-ranker ───────────────────────────────────────────────────────────────
recommender:
  diversity_penalty: 0.15  # Score deduction per repeated section occurrence.
                            # 0 = no diversity enforcement. 0.5 = aggressive diversity.

# ─── LLM Generation ──────────────────────────────────────────────────────────
generation:
  model: "gpt-4o-mini"   # LLM model. "claude-3-haiku-20240307" for Anthropic.
  max_tokens: 800        # Maximum tokens in the generated answer.
  temperature: 0.1       # 0.0 = fully deterministic. Higher = more creative (less factual).

# ─── MapReduce ───────────────────────────────────────────────────────────────
mapreduce:
  n_workers: 4           # Parallel workers. Capped at cpu_count() automatically.
  chunk_partition_size: 50  # Chunks per map worker partition.

# ─── Hybrid Strategy ─────────────────────────────────────────────────────────
hybrid:
  strategy: "rrf_all"    # rrf_all | rrf_lsh | cascade | adaptive
  rrf_k: 60              # RRF smoothing constant (Cormack 2009 recommendation).
  method_weights:        # Per-method contribution weight in RRF fusion.
    tfidf: 1.0           # TF-IDF has highest weight (highest precision baseline).
    minhash: 0.8
    simhash: 0.6

# ─── Query Cache ─────────────────────────────────────────────────────────────
cache:
  enabled: true
  max_size: 256          # Maximum cached query results. LRU eviction.
  ttl_seconds: 3600      # Cache entry lifetime (1 hour). Prevents stale answers.

# ─── File Paths ──────────────────────────────────────────────────────────────
paths:
  # UG handbook indexes
  ug_chunks: "data/processed/chunks.jsonl"
  ug_tfidf_index: "data/processed/tfidf_index.pkl"
  minhash_index: "data/processed/minhash_index.pkl"
  simhash_index: "data/processed/simhash_index.pkl"
  reranker_index: "data/processed/reranker_index.pkl"

  # PG handbook indexes
  pg_chunks: "data/processed/pg_chunks.jsonl"
  pg_tfidf_index: "data/processed/pg_tfidf_index.pkl"
```

---

## Experiments & Results

Three experiments rigorously evaluate the system.

### Experiment 1 — Exact vs Approximate Retrieval (`exp1_exact_vs_approx.py`)

**Setup:** 15 evaluation queries with manually labeled relevant sections. Each method is evaluated on Precision@5, Recall@5, NDCG@5, F1@5, latency (mean ± std dev), peak memory, and index size.

**Evaluation queries cover:** GPA requirements, fail policies, attendance rules, course repetition limits, plagiarism, academic probation, CGPA calculation, scholarships, course withdrawal, grade distribution, transfer policies, credit hours, final exams, probation rules, suspension policy.

| Method | Precision@5 | Avg Latency | Std Dev | Index Size | Memory |
|---|:---:|:---:|:---:|:---:|:---:|
| TF-IDF (baseline) | **0.87** | 312ms | ±48ms | 48.0 MB | 95 MB |
| MinHash + LSH | 0.79 | 28ms | ±6ms | 6.2 MB | 18 MB |
| SimHash | 0.74 | 19ms | ±4ms | 1.1 MB | 8 MB |
| **Hybrid + Re-ranker** | **0.85** | **41ms** | ±8ms | 7.3 MB | 26 MB |

**Key findings:**
- MinHash LSH achieves an **11× speedup** over TF-IDF with only 8 percentage point precision loss
- SimHash is the fastest (19ms) and most memory-efficient (1.1 MB index) — ideal for high-frequency queries
- The Hybrid + Re-ranker combination recovers most of the precision gap (+6pp over MinHash) while remaining 7× faster than TF-IDF
- TF-IDF index is 7× larger than LSH-based indexes (48 MB vs 6–7 MB)

### Experiment 2 — Hyperparameter Sensitivity (`exp2_param_sensitivity.py`)

Sweeps the key hyperparameters to find optimal settings:

| Parameter | Values Tested | Optimal |
|---|---|---|
| `num_perm` (MinHash) | 64, 128, 256, 512 | **128** — best precision/latency balance |
| `threshold` (LSH) | 0.2, 0.3, 0.4, 0.5 | **0.3** — recall stable, precision peaks |
| `simhash_bits` | 32, 64, 128 | **64** — standard, minimal gain from 128 |
| `hamming_threshold` | 5, 10, 15, 20 | **10** — catches paraphrases without noise |

**Key finding for `num_perm`:** Going from 64 to 128 permutations raises Precision@5 from 0.71 to 0.79 (+8pp). Going from 128 to 256 raises it to 0.81 (+2pp) at 2× build time. Diminishing returns justify stopping at 128 for production.

### Experiment 3 — Scalability Stress Test (`exp3_scalability.py`)

The corpus is duplicated 1×, 2×, 5×, and 10× to simulate scaling from ~500 to ~5,000 chunks.

| Corpus Scale | TF-IDF Latency | LSH Latency | SimHash Latency |
|---|:---:|:---:|:---:|
| 1× (~500 chunks) | 312ms | 28ms | 19ms |
| 2× (~1,000 chunks) | 624ms | 31ms | 22ms |
| 5× (~2,500 chunks) | 1,560ms | 38ms | 35ms |
| 10× (~5,000 chunks) | 3,890ms | 54ms | 51ms |

**Key finding:** TF-IDF latency scales linearly (O(n) dot product). LSH latency scales sub-linearly (O(log n) — only visited buckets grow, not the full corpus). At 10× scale, LSH is **72× faster** than TF-IDF while maintaining acceptable precision. 

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pdfplumber` | 0.11.4 | PDF text extraction with layout awareness |
| `scikit-learn` | 1.5.2 | TF-IDF vectorization, cosine similarity |
| `numpy` | 1.26.4 | Numerical operations, array math |
| `fastapi` | 0.115.6 | JSON API and static serving for the web interface |
| `uvicorn` | 0.32.1 | ASGI server for FastAPI |
| `openai` | 1.51.0 | OpenAI API client (GPT-4o-mini) |
| `anthropic` | 0.25.8 | Anthropic API client (Claude models) |
| `pyyaml` | 6.0.2 | YAML config file parsing |
| `python-dotenv` | 1.0.1 | `.env` file loading for API keys |
| `mmh3` | 4.1.0 | MurmurHash3 — fast non-cryptographic hash for SimHash |
| `pandas` | 2.2.3 | DataFrame operations in experiments |
| `matplotlib` | 3.9.2 | Experiment result plots |
| `tqdm` | 4.66.5 | Progress bars for ingestion/indexing |
| `pytest` | 8.3.2 | Test suite runner |

---

## Sample Queries

These queries exercise different parts of the policy document and different retrieval methods:

**Academic Standing:**
- *What is the minimum CGPA required to avoid academic probation?*
- *How many semesters can a student stay on probation before suspension?*
- *What happens if a student's CGPA falls below 2.00?*

**Grading & Assessment:**
- *What is the grading scale and what does an XF grade mean?*
- *How many times can a student repeat a course?*
- *Can a student retake a passed course to improve their grade?*

**Attendance:**
- *What is the minimum attendance requirement per course?*
- *What are the consequences of exceeding the absence limit?*

**Fees & Withdrawal:**
- *What is the fee refund policy for course withdrawal?*
- *By when must a student drop a course to receive a full refund?*

**Academic Integrity:**
- *What is the policy on plagiarism and academic dishonesty?*
- *What are the consequences of using unfair means in an exam?*

**Graduation:**
- *What is the minimum CGPA required for graduation?*
- *How many credit hours are required for the bachelor's degree?*

---

## Important Notes

> **Retrieval Engine, Not a Chatbot**
>
> AcademIQ is fundamentally a rigorous retrieval engine. The LLM simply synthesizes the structured chunks provided to it. It has no external memory, no training on NUST policies, and no ability to infer information not present in the handbook chunks. If the exact policy is not in the retrieved chunks, the answer will explicitly say "I could not find this information in the provided sections." This guarantees **auditable, hallucination-free responses** suitable for official academic policy reference.

> **Dual-Handbook Architecture**
>
> UG and PG handbooks are indexed completely separately. Their chunks, TF-IDF matrices, MinHash signatures, and SimHash fingerprints are stored in separate files. The cache also uses handbook-specific keys. There is no cross-contamination — a UG query will never surface PG policy text.

> **Extractive Mode Requires No API Key**
>
> The system is fully functional without any LLM API key. Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` to empty strings and the `AnswerGenerator` automatically uses extractive mode — pulling the most relevant sentences directly from the retrieved chunks without any external API call.

> **Index Rebuild on Config Changes**
>
> Changing ingestion, minhash, simhash, or tfidf parameters in `config.yaml` requires re-running the ingestion and indexing pipeline. Changing generation, retrieval, recommender, cache, or hybrid parameters takes effect immediately without rebuilding.
