"""
app/server.py

AcademIQ — FastAPI server for the web interface.

Wraps the existing src/ pipeline (retrievers, re-ranker, answer generator,
cache, metrics) in a small JSON API and serves the hand-built frontend
in app/web/.

Run from the project root:
    python -m app.server            # http://127.0.0.1:8000
    python -m app.server --port 8080 --no-browser

Endpoints:
    GET  /api/status            handbook availability, answer backends, config
    GET  /api/atlas?handbook=   2-D projection of every chunk (the star chart)
    POST /api/ask               retrieve + re-rank + generate, fully timed
    POST /api/compare           one query across TF-IDF / MinHash / SimHash / Hybrid
    GET  /api/health            P50/P99 latency, cache, Precision@k, recent queries
    POST /api/cache/clear       empty the query cache
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA, TruncatedSVD

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.tfidf_retriever   import TFIDFRetriever
from src.retrieval.minhash_lsh       import MinHashLSHRetriever
from src.retrieval.simhash           import SimHashRetriever
from src.retrieval.hybrid_retriever  import HybridRetriever
from src.ranking.recommender         import ChunkReranker
from src.generation.answer_generator import AnswerGenerator, _SENT_SPLIT_RE
from src.utils.config_loader         import cfg
from src.utils.cache                 import QueryCache
from src.utils.metrics_collector     import metrics
from src.utils.preprocessing         import expand_query


WEB_DIR = Path(__file__).parent / "web"

HANDBOOKS = {
    "ug": "Undergraduate",
    "pg": "Postgraduate",
}

# UI label → hybrid strategy (None = single retriever)
METHODS = {
    "TF-IDF":               None,
    "MinHash+LSH":          None,
    "SimHash":              None,
    "Hybrid (All Methods)": "rrf_all",
    "Hybrid (LSH Only)":    "rrf_lsh",
    "Cascade":              "cascade",
}
COMPARE_METHODS = ["TF-IDF", "MinHash+LSH", "SimHash", "Hybrid (All Methods)"]

# Precision@k proxy: share of top-k chunks whose
# retriever score clears this floor. No manual relevance labels needed.
RELEVANCE_THRESHOLD = 0.05

NOT_FOUND_PHRASES = (
    "could not find this information",
    "no relevant handbook sections",
)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

class Handbook:
    """All indexes for one handbook plus its star-chart projection."""

    def __init__(self, prefix: str):
        self.prefix   = prefix
        self.tfidf    = TFIDFRetriever.load(cfg("paths", f"{prefix}_tfidf_index",
                                                f"data/processed/{prefix}_tfidf_index.pkl"))
        self.minhash  = MinHashLSHRetriever.load(cfg("paths", f"{prefix}_minhash_index",
                                                     f"data/processed/{prefix}_minhash_index.pkl"))
        self.simhash  = SimHashRetriever.load(cfg("paths", f"{prefix}_simhash_index",
                                                  f"data/processed/{prefix}_simhash_index.pkl"))
        self.reranker = ChunkReranker.load(cfg("paths", f"{prefix}_reranker_index",
                                               f"data/processed/{prefix}_reranker_index.pkl"))
        self.hybrid   = HybridRetriever(tfidf=self.tfidf, minhash=self.minhash,
                                        simhash=self.simhash, reranker=self.reranker)
        self.chunks   = self.tfidf.chunks
        self.index_of = {c.get("id"): i for i, c in enumerate(self.chunks)}
        self._build_projection()

    # ── Star chart ────────────────────────────────────────────────────────

    def _build_projection(self) -> None:
        """
        Project the TF-IDF matrix to 2-D: TruncatedSVD to a few dozen
        components, then PCA to two. Queries go through the same two
        transforms, so a question lands in the same coordinate space
        as the chunks it resembles.
        """
        X = self.tfidf.matrix
        n = X.shape[0]
        n_comp = max(2, min(40, n - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
        Z = self._svd.fit_transform(X)
        self._pca = PCA(n_components=2, random_state=0)
        P = self._pca.fit_transform(Z)

        # Robust bounds so one outlier chunk doesn't squash the chart
        self._lo = np.percentile(P, 2, axis=0)
        self._hi = np.percentile(P, 98, axis=0)
        self.points = [self._normalise(p) for p in P]

    def _normalise(self, p) -> tuple[float, float]:
        span = np.where(self._hi - self._lo == 0, 1, self._hi - self._lo)
        x, y = (np.asarray(p) - self._lo) / span
        # 0..1 with a margin; clamp outliers to the chart edge
        x = float(np.clip(0.06 + 0.88 * x, 0.02, 0.98))
        y = float(np.clip(0.06 + 0.88 * (1 - y), 0.02, 0.98))
        return round(x, 4), round(y, 4)

    def project_query(self, query: str) -> tuple[tuple[float, float], list[float]]:
        """Return the query's chart position and its similarity to every chunk."""
        q_vec = self.tfidf.vectorizer.transform([expand_query(query)])
        point = self._normalise(self._pca.transform(self._svd.transform(q_vec))[0])
        sims  = (self.tfidf.matrix @ q_vec.T).toarray().ravel()
        return point, [round(float(s), 4) for s in sims]

    def atlas(self) -> dict:
        stars = []
        for i, (c, (x, y)) in enumerate(zip(self.chunks, self.points)):
            stars.append({
                "i":       i,
                "id":      c.get("id", f"chunk_{i}"),
                "x":       x,
                "y":       y,
                "section": c.get("section", "Unknown"),
                "page":    c.get("page"),
                "words":   c.get("word_count") or len((c.get("text") or "").split()),
            })
        return {
            "handbook": self.prefix,
            "label":    HANDBOOKS[self.prefix],
            "n_chunks": len(stars),
            "n_sections": len({s["section"] for s in stars}),
            "stars":    stars,
            # Component values at the chart's normalised bounds, for axis ticks
            "axes": {"x": [round(float(self._lo[0]), 4), round(float(self._hi[0]), 4)],
                     "y": [round(float(self._lo[1]), 4), round(float(self._hi[1]), 4)]},
        }


_handbooks: dict[str, Handbook] = {}
_handbook_errors: dict[str, str] = {}
_load_lock = threading.Lock()


def get_handbook(prefix: str) -> Handbook:
    if prefix not in HANDBOOKS:
        raise HTTPException(404, f"Unknown handbook {prefix!r}. Expected 'ug' or 'pg'.")
    with _load_lock:
        if prefix not in _handbooks and prefix not in _handbook_errors:
            try:
                _handbooks[prefix] = Handbook(prefix)
            except FileNotFoundError as e:
                _handbook_errors[prefix] = str(e).splitlines()[0]
    if prefix in _handbook_errors:
        raise HTTPException(409, {"code": "index_missing", "handbook": prefix,
                                  "detail": _handbook_errors[prefix]})
    return _handbooks[prefix]


cache     = QueryCache()
generator = AnswerGenerator()
_gen_lock = threading.Lock()

# Session log — sequential query numbers shared by answers, history and health
_log: list[dict] = []
_log_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _score(chunk: dict) -> float:
    return float(chunk.get("score", chunk.get("relevance_score", 0)) or 0)


def execute_query(hb: Handbook, query: str, method: str, top_k: int) -> tuple[list[dict], dict]:
    """Retrieve + re-rank with caching and per-stage timing."""
    key = cache.make_key(query, method, top_k, extra=hb.prefix)
    cached = cache.get(key)
    if cached is not None:
        return cached, {"cache_hit": True, "cache_key": key,
                        "retrieval_ms": 0.0, "rerank_ms": 0.0}

    timings: dict = {}
    strategy = METHODS.get(method)
    if strategy:
        results, sub = hb.hybrid.retrieve(query, top_k=top_k, strategy=strategy,
                                          use_reranker=True)
        for k, v in sub.items():
            if isinstance(v, (int, float)):
                timings[k] = v
        timings["retrieval_ms"] = round(sum(v for k, v in timings.items()
                                            if k != "rerank_ms"), 2)
    else:
        retriever = {"TF-IDF": hb.tfidf, "MinHash+LSH": hb.minhash,
                     "SimHash": hb.simhash}[method]
        t0 = time.perf_counter()
        results = retriever.query(query, top_k=top_k * 2)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        if hb.reranker and results:
            results = hb.reranker.rerank(query, results, top_k=top_k)
            results = hb.reranker.apply_diversity_penalty(results)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    timings.setdefault("rerank_ms", 0.0)
    timings["cache_hit"] = False
    timings["cache_key"] = key
    cache.put(key, results)
    return results, timings


def _chunk_view(hb: Handbook, chunk: dict, rank: int) -> dict:
    return {
        "rank":    rank,
        "i":       hb.index_of.get(chunk.get("id")),
        "id":      chunk.get("id"),
        "section": chunk.get("section", "Unknown"),
        "page":    chunk.get("page"),
        "score":   round(_score(chunk), 4),
        "text":    chunk.get("text", ""),
    }


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _structure_answer(answer: str, chunks: list[dict]) -> dict:
    """
    Split the generator's answer into sentences and attach the rank of every
    retrieved chunk that supports each one. Extractive sentences are matched
    verbatim; an LLM answer's trailing "Sources:" line is matched by page.
    """
    body, _, sources_line = answer.partition("\n\nSources:")
    if not sources_line and "Sources:" in answer:
        body, _, sources_line = answer.rpartition("Sources:")
    body = body.strip()

    texts = [_norm(c.get("text", "")) for c in chunks]
    sentences = []
    for sent in _SENT_SPLIT_RE.split(body):
        sent = sent.strip()
        if not sent:
            continue
        probe = _norm(sent.rstrip("."))[:160]
        refs = [r for r, t in enumerate(texts, 1) if probe and probe in t]
        sentences.append({"text": sent, "refs": refs})

    cited = sorted({r for s in sentences for r in s["refs"]})
    if sources_line and not cited:
        pages = {int(p) for p in re.findall(r"Page\s+(\d+)", sources_line)}
        cited = [r for r, c in enumerate(chunks, 1) if c.get("page") in pages]
        if cited and sentences:
            sentences[-1]["refs"] = cited

    not_found = any(p in body.lower() for p in NOT_FOUND_PHRASES)
    return {"sentences": sentences, "cited": cited, "not_found": not_found}


# ══════════════════════════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="AcademIQ", docs_url="/api/docs", redoc_url=None)


class AskRequest(BaseModel):
    handbook: str = "ug"
    query:    str = Field(min_length=1, max_length=500)
    method:   str = "Hybrid (All Methods)"
    top_k:    int = Field(5, ge=3, le=15)
    use_llm:  bool = False


class CompareRequest(BaseModel):
    handbook: str = "ug"
    query:    str = Field(min_length=1, max_length=500)
    top_k:    int = Field(5, ge=3, le=15)


@app.get("/api/status")
def status() -> dict:
    books = {}
    for prefix, label in HANDBOOKS.items():
        try:
            hb = get_handbook(prefix)
            books[prefix] = {"label": label, "ready": True, "n_chunks": len(hb.chunks)}
        except HTTPException as e:
            books[prefix] = {"label": label, "ready": False,
                             "detail": e.detail.get("detail") if isinstance(e.detail, dict) else e.detail}
    return {
        "handbooks": books,
        "methods":   list(METHODS),
        "default_method": "Hybrid (All Methods)",
        "top_k":     {"min": 3, "max": 15, "default": cfg("retrieval", "top_k", 5)},
        "llm": {
            "available": generator._backend is not None,
            "backend":   generator._backend,
            "model":     generator.model if generator._backend else None,
        },
        "config": {
            "chunk_size":     cfg("ingestion", "chunk_size", 350),
            "chunk_overlap":  cfg("ingestion", "chunk_overlap", 50),
            "num_perm":       cfg("minhash", "num_perm", 128),
            "lsh_threshold":  cfg("minhash", "threshold", 0.3),
            "simhash_bits":   cfg("simhash", "bits", 64),
            "hamming":        cfg("simhash", "hamming_threshold", 10),
            "rrf_k":          cfg("hybrid", "rrf_k", 60),
            "weights":        cfg("hybrid", "method_weights", {}),
            "diversity":      cfg("recommender", "diversity_penalty", 0.15),
            "workers":        cfg("mapreduce", "n_workers", 4),
            "cache_max":      cache.max_size,
            "cache_ttl":      cache.ttl_seconds,
        },
    }


@app.get("/api/atlas")
def atlas(handbook: str = "ug") -> dict:
    return get_handbook(handbook).atlas()


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    hb = get_handbook(req.handbook)
    if req.method not in METHODS:
        raise HTTPException(422, f"Unknown method {req.method!r}.")
    query = req.query.strip()

    results, timings = execute_query(hb, query, req.method, req.top_k)

    gen_ms = 0.0
    if results:
        with _gen_lock:
            original = generator.mode
            generator.mode = "llm" if req.use_llm and generator._backend else "extractive"
            t0 = time.perf_counter()
            gen = generator.generate(query, results)
            gen_ms = round((time.perf_counter() - t0) * 1000, 2)
            generator.mode = original
        answer_text = gen.get("answer", "")
        answer_mode = gen.get("mode", "extractive")
        model = gen.get("model")
    else:
        answer_text = "No relevant handbook sections were retrieved for this query."
        answer_mode, model = "extractive", "extractive"

    precision_k = None
    if results and not timings["cache_hit"]:
        relevant = sum(1 for c in results if _score(c) >= RELEVANCE_THRESHOLD)
        precision_k = round(relevant / len(results), 4)

    metrics.record(query=query, method=req.method,
                   retrieval_ms=timings["retrieval_ms"], rerank_ms=timings["rerank_ms"],
                   generation_ms=gen_ms, cache_hit=timings["cache_hit"],
                   num_chunks=len(results), precision_k=precision_k)

    point, field = hb.project_query(query)
    stage_ms = {k: timings[k] for k in ("tfidf_ms", "minhash_ms", "simhash_ms") if k in timings}
    total_ms = round(timings["retrieval_ms"] + timings["rerank_ms"] + gen_ms, 2)

    with _log_lock:
        qno = len(_log) + 1
        _log.append({
            "qno": qno, "query": query, "handbook": req.handbook, "method": req.method,
            "retrieval_ms": timings["retrieval_ms"], "rerank_ms": timings["rerank_ms"],
            "generation_ms": gen_ms, "total_ms": total_ms,
            "cache_hit": timings["cache_hit"], "n": len(results),
            "precision_k": precision_k, "answer_mode": answer_mode,
            "t": time.time(),
        })

    return {
        "qno":       qno,
        "query":     query,
        "handbook":  req.handbook,
        "method":    req.method,
        "top_k":     req.top_k,
        "answer":    _structure_answer(answer_text, results),
        "answer_mode": answer_mode,
        "model":     model,
        "llm_requested": req.use_llm,
        "chunks":    [_chunk_view(hb, c, r) for r, c in enumerate(results, 1)],
        "timings": {
            "retrieval_ms":  timings["retrieval_ms"],
            "rerank_ms":     timings["rerank_ms"],
            "generation_ms": gen_ms,
            "total_ms":      total_ms,
            "stages":        stage_ms,
        },
        "cache_hit": timings["cache_hit"],
        "cache_key": timings["cache_key"],
        "precision_k": precision_k,
        "query_point": point,
        "field":     field,
    }


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    hb = get_handbook(req.handbook)
    query = req.query.strip()
    columns = []
    for method in COMPARE_METHODS:
        t0 = time.perf_counter()
        results, timings = execute_query(hb, query, method, req.top_k)
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)
        columns.append({
            "method":    method,
            "cache_hit": timings["cache_hit"],
            "retrieval_ms": timings["retrieval_ms"],
            "rerank_ms": timings["rerank_ms"],
            "total_ms":  round(timings["retrieval_ms"] + timings["rerank_ms"], 2)
                         if not timings["cache_hit"] else wall_ms,
            "chunks":    [_chunk_view(hb, c, r) for r, c in enumerate(results, 1)],
        })
    point, field = hb.project_query(query)
    return {"query": query, "handbook": req.handbook, "top_k": req.top_k,
            "columns": columns, "query_point": point, "field": field}


@app.get("/api/health")
def health() -> dict:
    summary = metrics.summary()
    with metrics._lock:
        method_pk: dict[str, list[float]] = {}
        for r in metrics._records:
            if r.precision_k is not None:
                method_pk.setdefault(r.method, []).append(r.precision_k)
    with _log_lock:
        log = list(_log)
    return {
        "summary": summary,
        "cache":   cache.stats(),
        "precision_by_method": {m: {"avg": round(sum(v) / len(v), 4), "n": len(v)}
                                for m, v in sorted(method_pk.items())},
        "history": log[-60:],
        "recent":  list(reversed(log[-12:])),
    }


@app.post("/api/cache/clear")
def clear_cache() -> dict:
    cache.clear()
    return cache.stats()


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the AcademIQ web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
