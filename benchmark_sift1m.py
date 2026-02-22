"""
benchmark_sift1m.py
===================
Benchmarks SHSRSEngine on SIFT1M (1M × 128D vectors).

Handles the L2 vs cosine difference:
  SIFT1M ground truth uses L2 distance.
  SHSRS uses cosine (inner product on L2-normalised vectors).
  This script L2-normalises all vectors and recomputes ground truth
  on a 10K query subset using brute-force cosine search.

Recommended config for 1M × 128D:
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
    python benchmark_sift1m.py --n_queries 1000   # more queries for GT
    python benchmark_sift1m.py --quick            # 500 queries, fast check
"""

import sys
import time
import argparse
import gc
import numpy as np
import faiss
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shsrs import SHSRSEngine

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("sift1m_data")
INDEX_DIR     = Path("shsrs_sift1m_index")
N_CLUSTERS    = 1000    # sqrt(1M) rule
M             = 16      # more links for higher recall at 128D
EF_CONST      = 200
EF_SEARCH     = 50
GT_K          = 10
N_QUERIES     = 1000    # queries for benchmarking (subset of 10K)

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


def compute_gt_cosine(data_norm, queries_norm, k, batch_size=500):
    """Brute-force cosine GT in batches to avoid OOM at 1M scale."""
    Q = len(queries_norm)
    gt = np.zeros((Q, k), dtype=np.int64)
    print(f"  Computing cosine GT ({Q} queries, k={k}, batched)...")
    t0 = time.time()
    for start in range(0, Q, batch_size):
        end   = min(start + batch_size, Q)
        batch = queries_norm[start:end]
        scores = batch @ data_norm.T      # [batch, N]
        gt[start:end] = np.argsort(-scores, axis=1)[:, :k]
        if (start // batch_size) % 5 == 0:
            pct = end / Q * 100
            print(f"    {end}/{Q} ({pct:.0f}%)")
    print(f"  GT done in {time.time()-t0:.1f}s")
    return gt


def recall_at_k(preds, gt) -> float:
    hits = sum(len(set(p) & set(g)) for p, g in zip(preds, gt))
    return hits / (len(gt) * len(gt[0]))


def run_benchmark(engine, data_norm, queries_norm, ground_truth,
                  k=GT_K, probe=None, n_warmup=20, label=""):
    N = len(data_norm)
    for q in queries_norm[:n_warmup]:
        engine.search(q, k=k, probe=probe)

    latencies, preds, cand_sizes = [], [], []
    t0 = time.perf_counter()
    for q in queries_norm:
        t_q   = time.perf_counter()
        res   = engine.search(q, k=k, probe=probe)
        latencies.append((time.perf_counter() - t_q) * 1000)
        preds.append([r[0] for r in res])
        cand_sizes.append(len(res))

    latencies = np.array(latencies)
    return {
        "label":    label,
        "recall":   recall_at_k(preds, ground_truth),
        "lat_mean": latencies.mean(),
        "lat_p50":  np.percentile(latencies, 50),
        "lat_p95":  np.percentile(latencies, 95),
        "lat_p99":  np.percentile(latencies, 99),
        "cand_pct": np.mean(cand_sizes) / N * 100,
        "qps":      len(queries_norm) / (latencies.sum() / 1000),
    }


def print_result(r, ref_recall=None, ref_lat=None):
    recall_str = f"{r['recall']*100:>5.1f}%"
    if ref_recall:
        delta = (r['recall'] - ref_recall) * 100
        recall_str += f" ({delta:>+5.1f}pp)"
    lat_str = f"{r['lat_mean']:>6.2f}ms"
    if ref_lat:
        speedup = ref_lat / r['lat_mean']
        lat_str += f" ({speedup:.1f}x faster)"
    print(f"  {r['label']:<32} Recall={recall_str}  "
          f"Mean={lat_str}  p95={r['lat_p95']:>6.2f}ms  "
          f"QPS={r['qps']:>5.0f}  Cands={r['cand_pct']:.3f}%")


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
    parser.add_argument("--data_dir",  default=str(DATA_DIR))
    parser.add_argument("--index_dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    if args.quick:
        args.n_queries = 500

    data_dir  = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    print("=" * 72)
    print("SHSRS — SIFT1M Benchmark  (1M × 128D, cosine)")
    print("=" * 72)

    # ── check data files exist ────────────────────────────────────────────────
    base_path = data_dir / "sift1m_base.npy"
    qry_path  = data_dir / "sift1m_queries.npy"
    if not base_path.exists():
        print(f"\nERROR: {base_path} not found.")
        print("Run: python download_sift1m.py")
        sys.exit(1)

    # ── load vectors ──────────────────────────────────────────────────────────
    print(f"\n[1] Loading SIFT1M vectors...")
    t0   = time.time()
    base = np.load(base_path).astype(np.float32)
    N, D = base.shape
    print(f"  Base: {N:,} × {D}D  ({base.nbytes/1024/1024:.0f} MB)")

    print(f"  L2-normalising...")
    data_norm = cosine_normalize(base)
    del base; gc.collect()

    queries_raw  = np.load(qry_path).astype(np.float32)
    queries_norm = cosine_normalize(queries_raw)
    del queries_raw; gc.collect()

    # sample query subset
    rng           = np.random.default_rng(42)
    query_indices = rng.choice(len(queries_norm), size=args.n_queries, replace=False)
    queries_norm  = queries_norm[query_indices]
    print(f"  {args.n_queries} queries sampled for benchmarking")
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
        print(f"  Config: n_clusters={N_CLUSTERS}, M={M}, "
              f"ef_construction={EF_CONST}")
        print(f"  WARNING: This will take 2-4 hours on CPU.")
        print(f"  All steps are cached — safe to interrupt and resume.")
        print(f"  Starting build...")

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

    # ── ground truth (cosine) ─────────────────────────────────────────────────
    gt_cache = index_dir / f"gt_cosine_{args.n_queries}q_k{GT_K}.npy"
    if gt_cache.exists():
        print(f"\n[4] [CACHE] Loading cosine ground truth...")
        ground_truth = np.load(gt_cache)
    else:
        print(f"\n[4] Computing cosine ground truth...")
        ground_truth = compute_gt_cosine(data_norm, queries_norm, GT_K)
        np.save(gt_cache, ground_truth)
        print(f"  Saved to {gt_cache}")

    # ── benchmark ─────────────────────────────────────────────────────────────
    print(f"\n[5] Benchmark  ({args.n_queries} queries, k={GT_K})")
    print(f"{'─'*72}")

    probe_list = [6, 10, 16, 24, 32] if args.quick else \
                 [4, 6, 8, 10, 12, 16, 20, 24, 32]

    results = []

    # adaptive first
    r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                      probe=None, label="adaptive probe")
    print_result(r)
    results.append(r)

    # fixed probes
    for probe in probe_list:
        r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                          probe=probe, label=f"fixed probe={probe}")
        print_result(r)
        results.append(r)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Dataset : SIFT1M  {N:,} × {D}D  (cosine, L2-normalised)")
    print(f"  Index   : nc={N_CLUSTERS}, M={M}, ef_construction={EF_CONST}")
    print()

    best_recall   = max(results, key=lambda r: r["recall"])
    best_speed    = min(results, key=lambda r: r["lat_mean"])
    best_balanced = min(
        [r for r in results if r["recall"] >= 0.90],
        key=lambda r: r["lat_mean"],
        default=best_recall
    )

    print(f"  Best recall  : {best_recall['label']:<28} "
          f"Recall={best_recall['recall']*100:.1f}%  "
          f"Lat={best_recall['lat_mean']:.2f}ms")
    print(f"  Best speed   : {best_speed['label']:<28} "
          f"Recall={best_speed['recall']*100:.1f}%  "
          f"Lat={best_speed['lat_mean']:.2f}ms")
    print(f"  Best balanced: {best_balanced['label']:<28} "
          f"Recall={best_balanced['recall']*100:.1f}%  "
          f"Lat={best_balanced['lat_mean']:.2f}ms  QPS={best_balanced['qps']:.0f}")

    print(f"\n  RAM estimate : {engine.ram_estimate_mb():.0f} MB")
    print(f"{'='*72}")
    print("Done.")


if __name__ == "__main__":
    main()
