"""
experiments/exp3_scalability.py

Experiment 3: Scalability Test

Extended scale factors: 1×, 5×, 10×, 20×, 50×
Includes MapReduce index build comparison (1 vs 2 vs 4 workers).

Measures:
  - Query latency at scale (ms)
  - Index size growth (MB)
  - Build time: sequential vs MapReduce
  - Throughput: queries/second
  - Asymptotic complexity annotations

Run:
    python experiments/exp3_scalability.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.minhash_lsh     import MinHashLSHRetriever
from src.retrieval.simhash         import SimHashRetriever
from src.indexing.mapreduce_indexer import MapReduceIndexer
from src.utils.config_loader       import cfg


SCALE_FACTORS = [1, 5, 10, 20, 50]
TEST_QUERIES  = [
    "What is the minimum GPA requirement?",
    "What is the attendance policy?",
    "What happens if a student fails a course?",
    "How many times can a course be repeated?",
    "What is the policy on plagiarism?",
]
N_RUNS = 5


def load_base_chunks() -> list[dict]:
    chunks_path = Path(cfg("paths", "chunks", "data/processed/chunks.jsonl"))
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise ValueError("chunks.jsonl is empty.")
    return chunks


def scale_corpus(base_chunks: list[dict], factor: int) -> list[dict]:
    if factor == 1:
        return base_chunks
    scaled = []
    for i in range(factor):
        for chunk in base_chunks:
            c = dict(chunk)
            c["id"] = f"{chunk['id']}_s{i}"
            scaled.append(c)
    return scaled


def measure_latency(retriever, queries: list[str],
                    top_k: int = 5, n_runs: int = N_RUNS) -> float:
    times = []
    for q in queries:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            retriever.query(q, top_k=top_k)
            times.append((time.perf_counter() - t0) * 1000)
    return round(sum(times) / len(times), 2)


def measure_throughput(retriever, queries: list[str],
                       top_k: int = 5, duration_s: float = 2.0) -> float:
    """Measure queries/second over a fixed duration."""
    count = 0
    t_end = time.perf_counter() + duration_s
    qi = 0
    while time.perf_counter() < t_end:
        retriever.query(queries[qi % len(queries)], top_k=top_k)
        count += 1
        qi += 1
    return round(count / duration_s, 1)


def temp_index_size(retriever, tmp_path: str) -> float:
    p = Path(tmp_path)
    try:
        retriever.save(tmp_path)
        return round(p.stat().st_size / 1_048_576, 2)
    finally:
        if p.exists():
            p.unlink()


def run() -> None:
    results_dir = Path(cfg("paths", "results_dir", "experiments/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    base_chunks = load_base_chunks()
    rows = []
    mr_rows = []  # MapReduce comparison

    for factor in SCALE_FACTORS:
        corpus = scale_corpus(base_chunks, factor)
        n = len(corpus)
        print(f"\n{'='*60}")
        print(f"Scale {factor}× — {n} chunks")
        print(f"{'='*60}")

        # ── TF-IDF ────────────────────────────────────────────────────
        print("  Fitting TF-IDF...")
        t0 = time.perf_counter()
        tfidf = TFIDFRetriever()
        tfidf.fit(corpus)
        build = round((time.perf_counter() - t0) * 1000, 1)
        size = temp_index_size(tfidf, str(results_dir / "_tmp_tfidf.pkl"))
        lat = measure_latency(tfidf, TEST_QUERIES)
        thru = measure_throughput(tfidf, TEST_QUERIES)
        print(f"    build={build}ms  size={size}MB  lat={lat}ms  thru={thru}q/s")
        rows.append({"scale": factor, "chunks": n, "method": "TF-IDF",
                     "build_ms": build, "size_mb": size, "latency_ms": lat,
                     "throughput_qps": thru})

        # ── MinHash+LSH (sequential) ──────────────────────────────────
        print("  Fitting MinHash+LSH (sequential)...")
        t0 = time.perf_counter()
        minhash = MinHashLSHRetriever(
            num_perm=cfg("minhash", "num_perm", 128),
            threshold=cfg("minhash", "threshold", 0.3),
        )
        minhash.fit(corpus)
        build = round((time.perf_counter() - t0) * 1000, 1)
        size = temp_index_size(minhash, str(results_dir / "_tmp_mh.pkl"))
        lat = measure_latency(minhash, TEST_QUERIES)
        thru = measure_throughput(minhash, TEST_QUERIES)
        print(f"    build={build}ms  size={size}MB  lat={lat}ms  thru={thru}q/s")
        rows.append({"scale": factor, "chunks": n, "method": "MinHash+LSH",
                     "build_ms": build, "size_mb": size, "latency_ms": lat,
                     "throughput_qps": thru})

        # ── SimHash ───────────────────────────────────────────────────
        print("  Fitting SimHash...")
        t0 = time.perf_counter()
        simhash = SimHashRetriever(
            bits=cfg("simhash", "bits", 64),
            hamming_threshold=cfg("simhash", "hamming_threshold", 10),
        )
        simhash.fit(corpus)
        build = round((time.perf_counter() - t0) * 1000, 1)
        size = temp_index_size(simhash, str(results_dir / "_tmp_sh.pkl"))
        lat = measure_latency(simhash, TEST_QUERIES)
        thru = measure_throughput(simhash, TEST_QUERIES)
        print(f"    build={build}ms  size={size}MB  lat={lat}ms  thru={thru}q/s")
        rows.append({"scale": factor, "chunks": n, "method": "SimHash",
                     "build_ms": build, "size_mb": size, "latency_ms": lat,
                     "throughput_qps": thru})

        # ── MapReduce MinHash (1, 2, 4 workers) ──────────────────────
        for n_workers in [1, 2, 4]:
            print(f"  MapReduce MinHash ({n_workers} workers)...")
            indexer = MapReduceIndexer(n_workers=n_workers)
            t0 = time.perf_counter()
            mr_result = indexer.build_minhash_index(corpus)
            mr_build = round((time.perf_counter() - t0) * 1000, 1)
            mr_rows.append({
                "scale": factor, "chunks": n,
                "workers": n_workers,
                "build_ms": mr_build,
                "map_ms": mr_result["timings"].get("map_ms", 0),
                "shuffle_ms": mr_result["timings"].get("shuffle_ms", 0),
                "reduce_ms": mr_result["timings"].get("reduce_ms", 0),
            })
            print(f"    build={mr_build}ms  "
                  f"(map={mr_result['timings'].get('map_ms', 0):.0f}ms  "
                  f"shuffle={mr_result['timings'].get('shuffle_ms', 0):.0f}ms  "
                  f"reduce={mr_result['timings'].get('reduce_ms', 0):.0f}ms)")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "exp3_scalability.csv", index=False)
    print(f"\nResults saved → {results_dir}/exp3_scalability.csv")
    print(df.to_string(index=False))

    df_mr = pd.DataFrame(mr_rows)
    df_mr.to_csv(results_dir / "exp3_mapreduce.csv", index=False)
    print(f"\nMapReduce results → {results_dir}/exp3_mapreduce.csv")
    print(df_mr.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 3 — Scalability Test\n"
                 f"(Scale: {SCALE_FACTORS})", fontsize=14, fontweight="bold")

    colors  = {"TF-IDF": "#4C72B0", "MinHash+LSH": "#DD8452", "SimHash": "#55A868"}
    markers = {"TF-IDF": "o",       "MinHash+LSH": "s",       "SimHash": "^"}

    for method in ["TF-IDF", "MinHash+LSH", "SimHash"]:
        sub = df[df["method"] == method].sort_values("chunks")
        axes[0][0].plot(sub["chunks"], sub["latency_ms"],
                        marker=markers[method], color=colors[method],
                        label=method, linewidth=2)
        axes[0][1].plot(sub["chunks"], sub["size_mb"],
                        marker=markers[method], color=colors[method],
                        label=method, linewidth=2)
        axes[1][0].plot(sub["chunks"], sub["build_ms"],
                        marker=markers[method], color=colors[method],
                        label=method, linewidth=2)
        axes[1][1].plot(sub["chunks"], sub["throughput_qps"],
                        marker=markers[method], color=colors[method],
                        label=method, linewidth=2)

    for ax, title, ylabel in zip(
        [axes[0][0], axes[0][1], axes[1][0], axes[1][1]],
        ["Query Latency vs Corpus Size",
         "Index Size vs Corpus Size",
         "Build Time vs Corpus Size",
         "Throughput vs Corpus Size"],
        ["Avg Latency (ms)", "Index Size (MB)",
         "Build Time (ms)", "Queries/Second"]
    ):
        ax.set_title(title); ax.set_xlabel("Number of chunks")
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "exp3_scalability.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── MapReduce comparison plot ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MapReduce Index Build — Sequential vs Parallel",
                 fontsize=14, fontweight="bold")

    mr_colors = {1: "#4C72B0", 2: "#DD8452", 4: "#55A868"}
    for w in [1, 2, 4]:
        sub = df_mr[df_mr["workers"] == w].sort_values("chunks")
        axes[0].plot(sub["chunks"], sub["build_ms"],
                     marker="o", label=f"{w} worker(s)",
                     color=mr_colors[w], linewidth=2)

    # Add sequential MinHash build for comparison
    seq = df[df["method"] == "MinHash+LSH"].sort_values("chunks")
    axes[0].plot(seq["chunks"], seq["build_ms"],
                 marker="x", label="Sequential", color="#C44E52",
                 linewidth=2, linestyle="--")

    axes[0].set_title("Build Time: MapReduce vs Sequential")
    axes[0].set_xlabel("Number of chunks"); axes[0].set_ylabel("Build Time (ms)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Speedup plot
    for w in [2, 4]:
        sub_w = df_mr[df_mr["workers"] == w].sort_values("chunks")
        sub_1 = df_mr[df_mr["workers"] == 1].sort_values("chunks")
        if len(sub_w) == len(sub_1):
            speedup = sub_1["build_ms"].values / np.maximum(sub_w["build_ms"].values, 1)
            axes[1].plot(sub_w["chunks"], speedup,
                         marker="o", label=f"{w}÷1 workers",
                         color=mr_colors[w], linewidth=2)

    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Speedup vs Single Worker")
    axes[1].set_xlabel("Number of chunks"); axes[1].set_ylabel("Speedup (×)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "exp3_mapreduce.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved → {results_dir}/")


if __name__ == "__main__":
    run()