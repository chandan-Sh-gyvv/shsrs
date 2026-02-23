"""
benchmark_sift1m.py
===================
Benchmarks SHSRSEngine on SIFT1M (1M x 128D vectors).

Usage:
    python benchmark_sift1m.py                    # full build + benchmark
    python benchmark_sift1m.py --benchmark_only   # skip build, load existing
    python benchmark_sift1m.py --n_queries 1000   # more queries for GT
    python benchmark_sift1m.py --quick            # 500 queries, fast check
"""

import sys
import time
import argparse
import gc
import os
import numpy as np
import faiss
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shsrs import SHSRSEngine

DATA_DIR   = Path("sift1m_data")
INDEX_DIR  = Path("shsrs_sift1m_index")
N_CLUSTERS = 1000
M          = 16
EF_CONST   = 200
GT_K       = 10
N_QUERIES  = 1000

GAP_POLICY_DEFAULT = [
    (0.20, 8), (0.12, 12), (0.06, 16), (0.02, 24), (0.00, 32),
]
PROBE_TIERS_1M = [(90, 8), (75, 12), (50, 16), (25, 24), (0, 32)]


def cosine_normalize(x):
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def compute_gt_cosine(data_norm, queries_norm, k, batch_size=500):
    Q  = len(queries_norm)
    gt = np.zeros((Q, k), dtype=np.int64)
    print(f"  Computing cosine GT ({Q} queries, k={k}, batched)...")
    t0 = time.time()
    for start in range(0, Q, batch_size):
        end    = min(start + batch_size, Q)
        scores = queries_norm[start:end] @ data_norm.T
        gt[start:end] = np.argsort(-scores, axis=1)[:, :k]
        if (start // batch_size) % 5 == 0:
            print(f"    {end}/{Q} ({end/Q*100:.0f}%)")
    print(f"  GT done in {time.time()-t0:.1f}s")
    return gt


def recall_at_k(preds, gt):
    hits = sum(len(set(p) & set(g)) for p, g in zip(preds, gt))
    return hits / (len(gt) * len(gt[0]))


def run_benchmark(engine, data_norm, queries_norm, ground_truth,
                  k=GT_K, probe=None, n_warmup=20, label=""):
    N = len(data_norm)
    for q in queries_norm[:n_warmup]:
        engine.search(q, k=k, probe=probe)

    latencies, preds, cand_sizes = [], [], []
    for q in queries_norm:
        t0  = time.perf_counter()
        res = engine.search(q, k=k, probe=probe)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append([r[0] for r in res])
        cand_sizes.append(len(res))

    latencies = np.array(latencies)
    return {
        "label":    label,
        "recall":   recall_at_k(preds, ground_truth),
        "lat_mean": latencies.mean(),
        "lat_p95":  np.percentile(latencies, 95),
        "lat_p99":  np.percentile(latencies, 99),
        "cand_pct": np.mean(cand_sizes) / N * 100,
        "qps":      len(queries_norm) / (latencies.sum() / 1000),
    }


def print_result(r):
    print(f"  {r['label']:<32} Recall={r['recall']*100:>5.1f}%  "
          f"Mean={r['lat_mean']:>6.2f}ms  p95={r['lat_p95']:>6.2f}ms  "
          f"QPS={r['qps']:>5.0f}  Cands={r['cand_pct']:.3f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument("--n_queries", type=int, default=N_QUERIES)
    parser.add_argument("--quick",     action="store_true")
    parser.add_argument("--data_dir",  default=str(DATA_DIR))
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    if args.quick:
        args.n_queries = 500

    data_dir  = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    print("=" * 72)
    print("SHSRS - SIFT1M Benchmark  (1M x 128D, cosine)")
    print("=" * 72)

    # load vectors
    base_path = data_dir / "sift1m_base.npy"
    qry_path  = data_dir / "sift1m_queries.npy"
    if not base_path.exists():
        print(f"\nERROR: {base_path} not found. Run: python download_sift1m.py")
        sys.exit(1)

    print(f"\n[1] Loading SIFT1M vectors...")
    t0        = time.time()
    base      = np.load(base_path).astype(np.float32)
    N, D      = base.shape
    data_norm = cosine_normalize(base)
    del base; gc.collect()

    queries_norm = cosine_normalize(np.load(qry_path).astype(np.float32))
    rng          = np.random.default_rng(42)
    queries_norm = queries_norm[rng.choice(len(queries_norm),
                                           size=args.n_queries, replace=False)]
    print(f"  {N:,} x {D}D loaded  |  {args.n_queries} queries  |  "
          f"{time.time()-t0:.1f}s")

    # build or load
    if args.benchmark_only and index_dir.exists():
        print(f"\n[2] Loading index from {index_dir}...")
        t0     = time.time()
        engine = SHSRSEngine.load(index_dir)
        print(f"  {engine}  |  Load: {time.time()-t0:.2f}s  |  "
              f"RAM: {engine.ram_estimate_mb():.0f} MB")
    else:
        print(f"\n[2] Building SHSRS index (nc={N_CLUSTERS}, M={M})...")
        t0     = time.time()
        engine = SHSRSEngine.build(
            vectors=data_norm, index_dir=index_dir,
            n_clusters=N_CLUSTERS, M=M, ef_construction=EF_CONST,
            gap_policy=GAP_POLICY_DEFAULT,
        )
        print(f"  Done in {(time.time()-t0)/60:.1f} min  |  {engine}")

    # recalibrate
    print(f"\n[3] Recalibrating gap policy...")
    engine.calibrate_gap_policy(queries_norm, probe_tiers=PROBE_TIERS_1M)

    # ground truth
    gt_cache = index_dir / f"gt_cosine_{args.n_queries}q_k{GT_K}.npy"
    if gt_cache.exists():
        print(f"\n[4] [CACHE] Loading ground truth...")
        ground_truth = np.load(gt_cache)
    else:
        print(f"\n[4] Computing ground truth...")
        ground_truth = compute_gt_cosine(data_norm, queries_norm, GT_K)
        np.save(gt_cache, ground_truth)

    # benchmark
    print(f"\n[5] Benchmark  ({args.n_queries} queries, k={GT_K})")
    print(f"{'─'*72}")

    probe_list = [6, 10, 16, 24, 32] if args.quick else \
                 [4, 6, 8, 10, 12, 16, 20, 24, 32]
    results = []

    r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                      probe=None, label="adaptive probe")
    print_result(r); results.append(r)

    for probe in probe_list:
        r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                          probe=probe, label=f"fixed probe={probe}")
        print_result(r); results.append(r)

    # multithreaded benchmark
    print(f"\n[6] Multithreaded search  (probe=32, 1M scale)")
    print(f"    {'Threads':<14} {'Recall':>8} {'Mean':>8} {'p95':>8} "
          f"{'QPS':>7} {'Speedup':>9}")
    print(f"    {'─'*58}")

    baseline_lat = None
    max_threads  = os.cpu_count() or 4
    for n_threads in sorted(set([1, 2, 4, min(8, max_threads)])):
        engine.set_threads(n_threads)
        r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                          probe=32, label=f"{n_threads}T")
        if baseline_lat is None:
            baseline_lat = r['lat_mean']
        speedup = baseline_lat / r['lat_mean']
        print(f"    {n_threads} thread(s)        "
              f"Recall={r['recall']*100:>5.1f}%  "
              f"Mean={r['lat_mean']:>6.2f}ms  "
              f"p95={r['lat_p95']:>6.2f}ms  "
              f"QPS={r['qps']:>6.0f}  "
              f"{speedup:>7.2f}x")

    engine.set_threads(os.cpu_count() or 4)

    # summary
    best_recall   = max(results, key=lambda r: r["recall"])
    best_speed    = min(results, key=lambda r: r["lat_mean"])
    best_balanced = min(
        [r for r in results if r["recall"] >= 0.90],
        key=lambda r: r["lat_mean"], default=best_recall)

    print(f"\n{'='*72}")
    print(f"SUMMARY  |  SIFT1M {N:,} x {D}D  |  "
          f"nc={N_CLUSTERS} M={M}  |  RAM={engine.ram_estimate_mb():.0f}MB")
    print(f"{'='*72}")
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
    print(f"{'='*72}")
    print("Done.")


if __name__ == "__main__":
    main()
