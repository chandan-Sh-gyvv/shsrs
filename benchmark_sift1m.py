"""
benchmark_sift1m.py
===================
Benchmarks SHSRSEngine on SIFT1M (1M x 128D vectors).

Handles the L2 vs cosine difference:
  SIFT1M ground truth uses L2 distance.
  SHSRS uses cosine (inner product on L2-normalised vectors).
  After L2-normalisation, cosine and L2 rankings are equivalent
  (||x-y||^2 = 2 - 2*cos(x,y) when ||x||=||y||=1), so the official
  GT file is used directly — no recomputation needed.

Uses search_batch for benchmarking — groups queries by cluster and
issues one batched FAISS call per cluster (~2.5x faster than sequential).

Recommended config for 1M x 128D:
  n_clusters = 1000
  M = 16
  ef_construction = 200

WARNING: Build time is long (~2-4 hours on CPU).
  The kNN graph alone takes ~30-60 min at 1M vectors using faiss flat.
  Run overnight or on a machine you can leave unattended.
  All intermediate results are cached — safe to interrupt and resume.

Usage:
    python benchmark_sift1m.py                    # full build + benchmark
    python benchmark_sift1m.py --benchmark_only   # skip build, load existing
    python benchmark_sift1m.py --n_queries 1000   # number of queries
    python benchmark_sift1m.py --quick            # 500 queries, fast check
"""

import sys
import time
import argparse
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shsrs import SHSRSEngine

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = Path("sift1m_data")
DEFAULT_INDEX_DIR = Path("shsrs_sift1m_index")
N_CLUSTERS        = 1000    # sqrt(1M) rule
M                 = 16      # more links for higher recall at 128D
EF_CONST          = 200
GT_K              = 10
N_QUERIES         = 1000    # queries for benchmarking
SEED              = 42

# Gap policy — will be recalibrated after build
GAP_POLICY_DEFAULT = [
    (0.20, 6),
    (0.12, 10),
    (0.06, 16),
    (0.02, 24),
    (0.00, 32),
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def cosine_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def recall_at_k(preds, gt, k=GT_K) -> float:
    """Recall@k — checks top-k predictions against all 100 GT neighbours."""
    hits = sum(len(set(p[:k]) & set(g[:100])) for p, g in zip(preds, gt))
    return hits / (len(gt) * k)


def run_benchmark(engine, queries_norm, ground_truth,
                  k=GT_K, probe=None, n_warmup=20, label=""):
    """
    Benchmark using search_batch — one batched FAISS call per cluster.
    Returns per-query latency stats by timing each query individually
    within the batch loop.
    """
    Q = len(queries_norm)

    # warmup
    engine.search_batch(queries_norm[:n_warmup], k=k, probe=probe)

    # time each query individually for accurate latency percentiles
    latencies = []
    all_preds = []

    for qi in range(Q):
        q_batch = queries_norm[qi:qi+1]  # [1, D] — still uses batch path
        t_q     = time.perf_counter()
        res     = engine.search_batch(q_batch, k=k, probe=probe)
        latencies.append((time.perf_counter() - t_q) * 1000)
        all_preds.append([r[0] for r in res[0]])

    latencies = np.array(latencies)
    return {
        "label":    label,
        "recall":   recall_at_k(all_preds, ground_truth, k),
        "lat_mean": latencies.mean(),
        "lat_p50":  np.percentile(latencies, 50),
        "lat_p95":  np.percentile(latencies, 95),
        "lat_p99":  np.percentile(latencies, 99),
        "qps":      Q / (latencies.sum() / 1000),
    }


def run_benchmark_batch(engine, queries_norm, ground_truth,
                        k=GT_K, probe=None, n_warmup=20, label=""):
    """
    Benchmark full batch throughput — all queries in one search_batch call.
    QPS reflects real-world batch throughput. Latency is avg (not per-query).
    """
    Q = len(queries_norm)

    # warmup
    engine.search_batch(queries_norm[:n_warmup], k=k, probe=probe)

    # full batch timing
    t0      = time.perf_counter()
    results = engine.search_batch(queries_norm, k=k, probe=probe)
    total   = time.perf_counter() - t0

    preds    = [[r[0] for r in res] for res in results]
    avg_lat  = total / Q * 1000

    return {
        "label":    label,
        "recall":   recall_at_k(preds, ground_truth, k),
        "lat_mean": avg_lat,
        "lat_p50":  avg_lat,
        "lat_p95":  avg_lat,
        "lat_p99":  avg_lat,
        "qps":      Q / total,
    }


def print_result(r, ref_recall=None, ref_lat=None):
    recall_str = f"{r['recall']*100:>5.1f}%"
    if ref_recall is not None:
        delta = (r['recall'] - ref_recall) * 100
        recall_str += f" ({delta:>+5.1f}pp)"
    lat_str = f"{r['lat_mean']:>6.2f}ms"
    if ref_lat is not None:
        speedup = ref_lat / r['lat_mean']
        lat_str += f" ({speedup:.1f}x)"
    print(f"  {r['label']:<32} Recall={recall_str}  "
          f"Mean={lat_str}  p95={r['lat_p95']:>6.2f}ms  "
          f"QPS={r['qps']:>6.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_only", action="store_true",
                        help="Skip build, load existing index")
    parser.add_argument("--n_queries", type=int, default=N_QUERIES)
    parser.add_argument("--quick",     action="store_true",
                        help="500 queries, reduced probe sweep")
    parser.add_argument("--data_dir",  default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--index_dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    if args.quick:
        args.n_queries = 500

    data_dir  = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    print("=" * 72)
    print("SHSRS — SIFT1M Benchmark  (1M x 128D, cosine, batched search)")
    print("=" * 72)

    # ── check data files ──────────────────────────────────────────────────────
    base_path = data_dir / "sift1m_base.npy"
    qry_path  = data_dir / "sift1m_queries.npy"
    gt_path   = data_dir / "sift1m_groundtruth.npy"

    for p in [base_path, qry_path, gt_path]:
        if not p.exists():
            print(f"\nERROR: {p} not found.")
            sys.exit(1)

    # ── load vectors ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading SIFT1M vectors...")
    t0   = time.time()
    base = np.load(base_path).astype(np.float32)
    N, D = base.shape
    print(f"  Base: {N:,} x {D}D  ({base.nbytes/1024/1024:.0f} MB)")

    print(f"  L2-normalising...")
    data_norm = cosine_normalize(base)
    del base; gc.collect()

    queries_raw  = np.load(qry_path).astype(np.float32)
    queries_norm = cosine_normalize(queries_raw)
    del queries_raw; gc.collect()

    # sample query subset — keep indices for GT alignment
    rng           = np.random.default_rng(SEED)
    query_indices = rng.choice(len(queries_norm), size=args.n_queries, replace=False)
    queries_norm  = queries_norm[query_indices]
    print(f"  {args.n_queries} queries sampled  (seed={SEED})")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # ── build or load index ───────────────────────────────────────────────────
    if args.benchmark_only and index_dir.exists():
        print(f"\n[2] Loading existing index from {index_dir} ...")
        t0     = time.time()
        engine = SHSRSEngine.load(index_dir)
        print(f"  {engine}")
        print(f"  Load time: {time.time()-t0:.2f}s")
        print(f"  Est RAM  : {engine.ram_estimate_mb():.0f} MB")
    else:
        print(f"\n[2] Building SHSRS index")
        print(f"  Config: n_clusters={N_CLUSTERS}, M={M}, ef_construction={EF_CONST}")
        print(f"  WARNING: This will take 2-4 hours on CPU.")
        print(f"  All steps are cached — safe to interrupt and resume.")

        t0     = time.time()
        engine = SHSRSEngine.build(
            vectors=data_norm,
            index_dir=index_dir,
            n_clusters=N_CLUSTERS,
            M=M,
            ef_construction=EF_CONST,
            gap_policy=GAP_POLICY_DEFAULT,
        )
        build_time = time.time() - t0
        print(f"\n  Build complete in {build_time/60:.1f} minutes")
        print(f"  {engine}")
        print(f"  Est RAM: {engine.ram_estimate_mb():.0f} MB")

    # ── recalibrate gap policy ────────────────────────────────────────────────
    print(f"\n[3] Recalibrating gap policy on {args.n_queries} queries...")
    new_policy = engine.calibrate_gap_policy(queries_norm)
    print(f"  Policy: {[(f'{t:.3f}', p) for t, p in new_policy]}")

    # ── ground truth ──────────────────────────────────────────────────────────
    print(f"\n[4] Loading official SIFT1M ground truth...")
    gt_raw       = np.load(gt_path)
    ground_truth = gt_raw[query_indices]   # align with sampled queries
    print(f"  GT shape: {ground_truth.shape}  (aligned to sampled queries)")

    # ── single-query latency benchmark ───────────────────────────────────────
    print(f"\n[5] Single-query latency  ({args.n_queries} queries, k={GT_K})")
    print(f"{'─'*72}")

    probe_list = [6, 10, 16, 24, 32] if args.quick else \
                 [4, 6, 8, 10, 12, 16, 20, 24, 32]

    sq_results = []

    r = run_benchmark(engine, queries_norm, ground_truth,
                      probe=None, label="adaptive probe")
    print_result(r)
    sq_results.append(r)

    for probe in probe_list:
        r = run_benchmark(engine, queries_norm, ground_truth,
                          probe=probe, label=f"fixed probe={probe}")
        print_result(r)
        sq_results.append(r)

    # ── full batch throughput benchmark ──────────────────────────────────────
    print(f"\n[6] Full batch throughput  ({args.n_queries} queries in one call)")
    print(f"{'─'*72}")

    bt_results = []

    r = run_benchmark_batch(engine, queries_norm, ground_truth,
                            probe=None, label="adaptive probe")
    print_result(r)
    bt_results.append(r)

    for probe in probe_list:
        r = run_benchmark_batch(engine, queries_norm, ground_truth,
                                probe=probe, label=f"fixed probe={probe}")
        print_result(r)
        bt_results.append(r)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Dataset : SIFT1M  {N:,} x {D}D  (cosine, L2-normalised)")
    print(f"  Index   : nc={N_CLUSTERS}, M={M}, ef_construction={EF_CONST}")
    print(f"  Search  : search_batch (batched FAISS, ~2.5x faster than sequential)")
    print()

    # single query summary
    best_recall   = max(sq_results, key=lambda r: r["recall"])
    best_speed    = min(sq_results, key=lambda r: r["lat_mean"])
    best_balanced = min(
        [r for r in sq_results if r["recall"] >= 0.90],
        key=lambda r: r["lat_mean"],
        default=best_recall
    )
    print(f"  [Single-query latency]")
    print(f"  Best recall  : {best_recall['label']:<28} "
          f"Recall={best_recall['recall']*100:.1f}%  "
          f"Lat={best_recall['lat_mean']:.2f}ms")
    print(f"  Best speed   : {best_speed['label']:<28} "
          f"Recall={best_speed['recall']*100:.1f}%  "
          f"Lat={best_speed['lat_mean']:.2f}ms")
    print(f"  Best balanced: {best_balanced['label']:<28} "
          f"Recall={best_balanced['recall']*100:.1f}%  "
          f"Lat={best_balanced['lat_mean']:.2f}ms  "
          f"QPS={best_balanced['qps']:.0f}")

    print()

    # batch throughput summary
    bt_best_recall   = max(bt_results, key=lambda r: r["recall"])
    bt_best_balanced = min(
        [r for r in bt_results if r["recall"] >= 0.90],
        key=lambda r: r["lat_mean"],
        default=bt_best_recall
    )
    print(f"  [Batch throughput]")
    print(f"  Best recall  : {bt_best_recall['label']:<28} "
          f"Recall={bt_best_recall['recall']*100:.1f}%  "
          f"QPS={bt_best_recall['qps']:.0f}")
    print(f"  Best balanced: {bt_best_balanced['label']:<28} "
          f"Recall={bt_best_balanced['recall']*100:.1f}%  "
          f"QPS={bt_best_balanced['qps']:.0f}")

    print(f"\n  RAM estimate : {engine.ram_estimate_mb():.0f} MB")
    print(f"{'='*72}")
    print("Done.")


if __name__ == "__main__":
    main()
