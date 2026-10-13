"""
experiments/exp1_exact_vs_approx.py

Experiment 1: Exact (TF-IDF) vs Approximate (MinHash, SimHash) vs Hybrid Retrieval

Run:
    python experiments/exp1_exact_vs_approx.py
"""

import json
import math
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.minhash_lsh     import MinHashLSHRetriever
from src.retrieval.simhash         import SimHashRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.config_loader       import cfg


# ── 15 evaluation queries with manually labelled relevant sections ─────────────

EVAL_QUERIES = [
    {
        "query": "What is the minimum GPA requirement?",
        "relevant_sections": [
            "Chapter 3: Award of Bachelor Degree and Academic Deficiencies",
            "5. Academic Standards for Award of Degree",
            "3. Offering and Registration of Courses",
        ],
    },
    {
        "query": "What happens if a student fails a course?",
        "relevant_sections": [
            "8. Repetition/Retest of Course(s).",
            "30. Repetition of Courses",
            "14. Repetition of Courses",
            "4. Warning.",
        ],
    },
    {
        "query": "What is the attendance policy?",
        "relevant_sections": [
            "6. All End Semester Examinations are managed by the constituent institutions on dates",
            "26. Adding / Dropping of Course(s)",
            "5. Examinations.",
        ],
    },
    {
        "query": "How many times can a course be repeated?",
        "relevant_sections": [
            "8. Repetition/Retest of Course(s).",
            "30. Repetition of Courses",
            "14. Repetition of Courses",
        ],
    },
    {
        "query": "What is the policy on plagiarism?",
        "relevant_sections": [
            "6. Unfair Means",
            "Chapter 11: NUST Code of Conduct",
            "12. Acts of Ill-discipline/Prohibited Conduct",
        ],
    },
    {
        "query": "What are the rules for academic probation?",
        "relevant_sections": [
            "4. Warning.",
            "10. Probation.",
            "27. Suspension.",
            "6. Suspension.",
        ],
    },
    {
        "query": "How is the CGPA calculated?",
        "relevant_sections": [
            "5. Academic Standards for Award of Degree",
            "3. Offering and Registration of Courses",
            "8. Parameters for determining a student's grade in a course",
        ],
    },
    {
        "query": "What is the grading scale?",
        "relevant_sections": [
            "5. Academic Standards for Award of Degree",
            "3. Offering and Registration of Courses",
            "8. Parameters for determining a student's grade in a course",
            "9. Assessment Modes and Weighting.",
        ],
    },
    {
        "query": "What happens if a student is on probation?",
        "relevant_sections": [
            "4. Warning.",
            "10. Probation.",
            "6. Suspension.",
            "27. Suspension.",
        ],
    },
    {
        "query": "Can a student drop a course?",
        "relevant_sections": [
            "26. Adding / Dropping of Course(s)",
            "3. Offering and Registration of Courses",
            "16. Adding/Dropping of Course(s).",
        ],
    },
    {
        "query": "What are the degree requirements for graduation?",
        "relevant_sections": [
            "Chapter 3: Award of Bachelor Degree and Academic Deficiencies",
            "5. Academic Standards for Award of Degree",
            "16. Completion of the Degree.",
        ],
    },
    {
        "query": "What is the maximum number of credit hours per semester?",
        "relevant_sections": [
            "26. Adding / Dropping of Course(s)",
            "3. Offering and Registration of Courses",
            "2. Duration",
        ],
    },
    {
        "query": "What are the hostel rules?",
        "relevant_sections": [
            "Chapter 12: Living on Campus",
            "12. Conduct",
            "13. Dos and Don'ts.",
            "3. Duration of Student Stay in Hostels",
        ],
    },
    {
        "query": "How can a student apply for financial assistance?",
        "relevant_sections": [
            "9. Financial Assistance.",
            "Chapter 6: Academic Provisions & Flexibilities",
        ],
    },
    {
        "query": "What is the policy on course withdrawal?",
        "relevant_sections": [
            "7. Withdrawal.",
            "29. Withdrawal.",
            "26. Adding / Dropping of Course(s)",
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def index_size_mb(path: str) -> float:
    p = Path(path)
    return round(p.stat().st_size / 1_048_576, 2) if p.exists() else 0.0


def _section_match(chunk_section: str, relevant: str) -> bool:
    """Multi-pass section matching with proper normalisation."""
    s = chunk_section.lower().strip()
    r = relevant.lower().strip()
    return (s == r
            or s in r
            or r in s
            or s[:50] in r
            or r[:50] in s)


def precision_at_k(results: list[dict],
                   relevant_sections: list[str],
                   k: int = 5) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    top_k = results[:k]
    hits = 0
    for chunk in top_k:
        section = chunk.get("section", "")
        if any(_section_match(section, r) for r in relevant_sections):
            hits += 1
    return hits / k if k > 0 else 0.0


def recall_at_k(results: list[dict],
                relevant_sections: list[str],
                k: int = 5) -> float:
    """Recall@k: fraction of ALL relevant sections found in top-k."""
    if not relevant_sections:
        return 0.0
    top_k = results[:k]
    found = set()
    for chunk in top_k:
        section = chunk.get("section", "")
        for i, rel in enumerate(relevant_sections):
            if _section_match(section, rel):
                found.add(i)
    return len(found) / len(relevant_sections)


def ndcg_at_k(results: list[dict],
              relevant_sections: list[str],
              k: int = 5) -> float:
    """
    NDCG@k: position-aware metric rewarding relevant chunks ranked higher.
    Binary relevance (1 if match, 0 otherwise).
    """
    top_k = results[:k]
    dcg = 0.0
    for i, chunk in enumerate(top_k):
        section = chunk.get("section", "")
        rel = 1 if any(_section_match(section, r) for r in relevant_sections) else 0
        dcg += rel / math.log2(i + 2)

    ideal_hits = min(len(relevant_sections), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def f1_at_k(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def measure_latency(retriever, query: str, top_k: int = 5,
                    n_runs: int = 10) -> tuple[float, float]:
    """Returns (avg_ms, std_ms) over n_runs warm-cache runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        retriever.query(query, top_k=top_k)
        times.append((time.perf_counter() - t0) * 1000)
    avg = round(sum(times) / len(times), 2)
    std = round(statistics.stdev(times), 2) if len(times) > 1 else 0.0
    return avg, std


def measure_memory_mb(retriever, query: str, top_k: int = 5) -> float:
    """Peak memory during a single query (MB) via tracemalloc."""
    tracemalloc.start()
    retriever.query(query, top_k=top_k)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / 1_048_576, 3)


# ── Main experiment ───────────────────────────────────────────────────────────

def run() -> None:
    results_dir = Path(cfg("paths", "results_dir", "experiments/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    tfidf_path   = cfg("paths", "tfidf_index",   "data/processed/tfidf_index.pkl")
    minhash_path = cfg("paths", "minhash_index", "data/processed/minhash_index.pkl")
    simhash_path = cfg("paths", "simhash_index", "data/processed/simhash_index.pkl")

    print("Loading indexes...")
    try:
        tfidf   = TFIDFRetriever.load(tfidf_path)
        minhash = MinHashLSHRetriever.load(minhash_path)
        simhash = SimHashRetriever.load(simhash_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Build all indexes first, then re-run this experiment.")
        return

    hybrid = HybridRetriever(tfidf=tfidf, minhash=minhash, simhash=simhash)

    retrievers = {
        "TF-IDF":      tfidf,
        "MinHash+LSH": minhash,
        "SimHash":     simhash,
    }

    rows = []
    print(f"\nRunning {len(EVAL_QUERIES)} queries × {len(retrievers)+1} methods...\n")

    for item in EVAL_QUERIES:
        q   = item["query"]
        rel = item["relevant_sections"]
        print(f"Q: {q[:60]}")

        for method_name, ret in retrievers.items():
            results  = ret.query(q, top_k=5)
            prec     = precision_at_k(results, rel)
            rec      = recall_at_k(results, rel)
            ndcg     = ndcg_at_k(results, rel)
            f1       = f1_at_k(prec, rec)
            lat, std = measure_latency(ret, q)
            mem      = measure_memory_mb(ret, q)

            rows.append({
                "query": q, "method": method_name,
                "precision@5": round(prec, 3), "recall@5": round(rec, 3),
                "ndcg@5": round(ndcg, 3), "f1@5": round(f1, 3),
                "latency_ms": lat, "latency_std": std, "memory_mb": mem,
            })
            print(f"  {method_name:<15} P@5={prec:.2f}  R@5={rec:.2f}  "
                  f"NDCG={ndcg:.2f}  F1={f1:.2f}  "
                  f"lat={lat:.1f}±{std:.1f}ms  mem={mem:.3f}MB")

        # Hybrid (3-way RRF)
        h_res, _ = hybrid.retrieve(q, top_k=5, strategy="rrf_all",
                                   use_reranker=False)
        prec = precision_at_k(h_res, rel)
        rec  = recall_at_k(h_res, rel)
        ndcg = ndcg_at_k(h_res, rel)
        f1   = f1_at_k(prec, rec)
        htimes = []
        for _ in range(10):
            t0 = time.perf_counter()
            hybrid.retrieve(q, top_k=5, strategy="rrf_all", use_reranker=False)
            htimes.append((time.perf_counter() - t0) * 1000)
        hlat = round(sum(htimes) / len(htimes), 2)
        hstd = round(statistics.stdev(htimes), 2) if len(htimes) > 1 else 0.0

        rows.append({
            "query": q, "method": "Hybrid(All)",
            "precision@5": round(prec, 3), "recall@5": round(rec, 3),
            "ndcg@5": round(ndcg, 3), "f1@5": round(f1, 3),
            "latency_ms": hlat, "latency_std": hstd, "memory_mb": 0.0,
        })
        print(f"  {'Hybrid(All)':<15} P@5={prec:.2f}  R@5={rec:.2f}  "
              f"NDCG={ndcg:.2f}  F1={f1:.2f}  lat={hlat:.1f}±{hstd:.1f}ms")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "exp1_results.csv", index=False)
    print(f"Results saved → {results_dir}/exp1_results.csv")

    summary = df.groupby("method").agg(
        avg_precision=("precision@5", "mean"),
        avg_recall=("recall@5", "mean"),
        avg_ndcg=("ndcg@5", "mean"),
        avg_f1=("f1@5", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_memory_mb=("memory_mb", "mean"),
    ).round(3)

    index_sizes = {
        "TF-IDF":      index_size_mb(tfidf_path),
        "MinHash+LSH": index_size_mb(minhash_path),
        "SimHash":     index_size_mb(simhash_path),
        "Hybrid(All)": round(sum(index_size_mb(p)
                                 for p in [tfidf_path, minhash_path, simhash_path]), 2),
    }
    summary["index_size_mb"] = summary.index.map(lambda m: index_sizes.get(m, 0.0))

    print("\n=== Summary ===")
    print(summary.to_string())
    summary.to_csv(results_dir / "exp1_summary.csv")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Experiment 1 — Exact vs Approximate Retrieval\n"
                 "(15 queries × 4 methods)", fontsize=14, fontweight="bold")

    methods = summary.index.tolist()
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(methods)]

    for ax, metric, title in zip(
        axes[0],
        ["avg_precision", "avg_recall", "avg_ndcg"],
        ["Average Precision@5", "Average Recall@5", "Average NDCG@5"]
    ):
        bars = ax.bar(methods, summary[metric], color=colors)
        ax.set_title(title); ax.set_ylim(0, 1.0)
        for i, v in enumerate(summary[metric]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis='x', rotation=15)

    axes[1][0].bar(methods, summary["avg_latency_ms"], color=colors)
    axes[1][0].set_title("Average Query Latency (ms)")
    for i, v in enumerate(summary["avg_latency_ms"]):
        axes[1][0].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    axes[1][0].tick_params(axis='x', rotation=15)

    axes[1][1].bar(methods, summary["index_size_mb"], color=colors)
    axes[1][1].set_title("Index Size (MB)")
    for i, v in enumerate(summary["index_size_mb"]):
        axes[1][1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axes[1][1].tick_params(axis='x', rotation=15)

    axes[1][2].bar(methods, summary["avg_f1"], color=colors)
    axes[1][2].set_title("Average F1@5")
    axes[1][2].set_ylim(0, 1.0)
    for i, v in enumerate(summary["avg_f1"]):
        axes[1][2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[1][2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(results_dir / "exp1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmap
    pivot = df.pivot(index="query", columns="method", values="precision@5")
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(pivot.values, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [q[:45] + "..." if len(q) > 45 else q for q in pivot.index], fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Precision@5")
    ax.set_title("Per-Query Precision@5 by Method", fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved → {results_dir}/")


if __name__ == "__main__":
    run()