"""
experiments/exp2_param_sensitivity.py

Experiment 2: Parameter Sensitivity Analysis

Tests:
  A. MinHash num_perm in [32, 64, 128, 256]
  B. LSH threshold in [0.1, 0.2, 0.3, 0.5]
  C. SimHash hamming_threshold in [5, 10, 15, 20]

Run:
    python experiments/exp2_param_sensitivity.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.minhash_lsh import MinHashLSHRetriever, build_minhash
from src.retrieval.simhash     import (SimHashRetriever, simhash as sh_fn,
                                        hamming_distance)
from src.utils.preprocessing   import tokenize, expand_query
from src.utils.config_loader   import cfg
from experiments.exp1_exact_vs_approx import EVAL_QUERIES, precision_at_k, recall_at_k, ndcg_at_k, f1_at_k

FIXED_QUERIES   = [q["query"]             for q in EVAL_QUERIES[:5]]
FIXED_RELEVANT  = {q["query"]: q["relevant_sections"] for q in EVAL_QUERIES[:5]}


def load_chunks() -> list[dict]:
    chunks_path = Path(cfg("paths", "chunks", "data/processed/chunks.jsonl"))
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            "Run the ingestion pipeline first."
        )
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise ValueError("chunks.jsonl is empty. Re-run the ingestion pipeline.")
    return chunks


def run() -> None:
    results_dir = Path(cfg("paths", "results_dir", "experiments/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()

    # ── A: MinHash num_perm sensitivity ──────────────────────────────────
    print("=== A: MinHash num_perm sensitivity ===")
    perm_rows = []
    for num_perm in [32, 64, 128, 256]:
        print(f"  num_perm={num_perm}...")
        t_build = time.perf_counter()
        r = MinHashLSHRetriever(num_perm=num_perm, threshold=0.3)
        r.fit(chunks)
        build_time = round((time.perf_counter() - t_build) * 1000, 1)

        precisions, latencies = [], []
        for q_item in EVAL_QUERIES[:5]:
            q   = q_item["query"]
            rel = q_item["relevant_sections"]
            t0  = time.perf_counter()
            res = r.query(q, top_k=5)
            lat = (time.perf_counter() - t0) * 1000
            precisions.append(precision_at_k(res, rel))
            latencies.append(lat)

        perm_rows.append({
            "num_perm":        num_perm,
            "avg_precision@5": round(sum(precisions) / len(precisions), 3),
            "avg_latency_ms":  round(sum(latencies)  / len(latencies),  1),
            "build_time_ms":   build_time,
            "bands":           r.lsh.b,
            "rows_per_band":   r.lsh.r,
        })
        print(f"    P@5={perm_rows[-1]['avg_precision@5']:.3f}  "
              f"latency={perm_rows[-1]['avg_latency_ms']:.1f}ms  "
              f"build={build_time}ms  bands={r.lsh.b} rows={r.lsh.r}")

    pd.DataFrame(perm_rows).to_csv(results_dir / "exp2a_num_perm.csv", index=False)

    # ── B: LSH threshold sensitivity ─────────────────────────────────────
    print("\n=== B: LSH threshold sensitivity ===")
    thresh_rows = []
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        print(f"  threshold={threshold}...")
        r = MinHashLSHRetriever(num_perm=128, threshold=threshold)
        r.fit(chunks)

        candidate_counts, precisions = [], []
        for q_item in EVAL_QUERIES[:5]:
            q   = q_item["query"]
            rel = q_item["relevant_sections"]

            q_tokens       = tokenize(expand_query(q), as_set=True)
            q_sig          = build_minhash(q_tokens, 128)
            raw_candidates = r.lsh.query(q_sig)
            candidate_counts.append(len(raw_candidates))

            res = r.query(q, top_k=5)
            precisions.append(precision_at_k(res, rel))

        thresh_rows.append({
            "threshold":       threshold,
            "avg_candidates":  round(sum(candidate_counts) / len(candidate_counts), 1),
            "avg_precision@5": round(sum(precisions) / len(precisions), 3),
            "bands":           r.lsh.b,
            "rows_per_band":   r.lsh.r,
        })
        print(f"    P@5={thresh_rows[-1]['avg_precision@5']:.3f}  "
              f"avg_candidates={thresh_rows[-1]['avg_candidates']:.1f}  "
              f"bands={r.lsh.b}")

    pd.DataFrame(thresh_rows).to_csv(results_dir / "exp2b_threshold.csv", index=False)

    # ── C: SimHash hamming_threshold sensitivity ──────────────────────────
    print("\n=== C: SimHash hamming_threshold sensitivity ===")
    simhash_r = SimHashRetriever(bits=64, hamming_threshold=10)
    simhash_r.fit(chunks)

    hamming_rows = []
    for h_thresh in [5, 10, 15, 20]:
        print(f"  hamming_threshold={h_thresh}...")
        precisions, candidates_per_query = [], []

        for q_item in EVAL_QUERIES[:5]:
            q   = q_item["query"]
            rel = q_item["relevant_sections"]
            res = simhash_r.query(q, top_k=5, hamming_threshold=h_thresh)
            precisions.append(precision_at_k(res, rel))

            q_fp    = sh_fn(tokenize(expand_query(q)), 64)
            passing = sum(
                1 for cid, fp in simhash_r.fingerprints.items()
                if cid not in simhash_r._empty_chunks
                and hamming_distance(q_fp, fp) <= h_thresh
            )
            candidates_per_query.append(passing)

        hamming_rows.append({
            "hamming_threshold": h_thresh,
            "avg_candidates":    round(sum(candidates_per_query) / len(candidates_per_query), 1),
            "avg_precision@5":   round(sum(precisions) / len(precisions), 3),
        })
        print(f"    P@5={hamming_rows[-1]['avg_precision@5']:.3f}  "
              f"avg_candidates={hamming_rows[-1]['avg_candidates']:.1f}")

    pd.DataFrame(hamming_rows).to_csv(results_dir / "exp2c_hamming.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    df_perm    = pd.DataFrame(perm_rows)
    df_thresh  = pd.DataFrame(thresh_rows)
    df_hamming = pd.DataFrame(hamming_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment 2 — Parameter Sensitivity", fontsize=13)

    ax, ax2 = axes[0], axes[0].twinx()
    ax.plot(df_perm["num_perm"], df_perm["avg_precision@5"],
            "o-", color="#4C72B0", label="Precision@5")
    ax2.bar(df_perm["num_perm"], df_perm["build_time_ms"],
            alpha=0.3, color="#DD8452", width=30, label="Build time (ms)")
    ax.set_xlabel("num_perm"); ax.set_ylabel("Precision@5", color="#4C72B0")
    ax2.set_ylabel("Build time (ms)", color="#DD8452")
    ax.set_title("A: MinHash — num_perm"); ax.set_ylim(0, 1)

    ax, ax2 = axes[1], axes[1].twinx()
    ax.plot(df_thresh["threshold"], df_thresh["avg_precision@5"],
            "o-", color="#4C72B0")
    ax2.plot(df_thresh["threshold"], df_thresh["avg_candidates"],
             "s--", color="#DD8452")
    ax.set_xlabel("LSH threshold"); ax.set_ylabel("Precision@5", color="#4C72B0")
    ax2.set_ylabel("Avg candidates", color="#DD8452")
    ax.set_title("B: LSH — threshold"); ax.set_ylim(0, 1)

    ax, ax2 = axes[2], axes[2].twinx()
    ax.plot(df_hamming["hamming_threshold"], df_hamming["avg_precision@5"],
            "o-", color="#4C72B0")
    ax2.plot(df_hamming["hamming_threshold"], df_hamming["avg_candidates"],
             "s--", color="#DD8452")
    ax.set_xlabel("Hamming threshold"); ax.set_ylabel("Precision@5", color="#4C72B0")
    ax2.set_ylabel("Avg candidates", color="#DD8452")
    ax.set_title("C: SimHash — Hamming threshold"); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(results_dir / "exp2_param_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nAll results saved → {results_dir}/")


if __name__ == "__main__":
    run()