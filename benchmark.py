"""
benchmark.py
============
Benchmarks the production SHSRSEngine against brute-force ground truth.

Measures:
  - Recall@10
  - Latency (mean, p50, p95, p99)
  - Candidate coverage %
  - Queries per second

Tests:
  1. Adaptive probe (production default)
  2. Fixed probe sweep (6, 8, 10, 12, 15, 20)
  3. Latency under load (sustained QPS test)

Usage:
    python benchmark.py
    python benchmark.py --index shsrs_production_index --n_queries 500
    python benchmark.py --probe 12          # fixed probe only
    python benchmark.py --quick             # 100 queries, fast check
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shsrs import SHSRSEngine


# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

def compute_ground_truth(data_norm: np.ndarray,
                         queries_norm: np.ndarray, k: int) -> np.ndarray:
    print(f"  Computing brute-force ground truth ({len(queries_norm)} queries)...")
    t0 = time.time()
    scores = queries_norm @ data_norm.T
    gt = np.argsort(-scores, axis=1)[:, :k]
    print(f"  Done in {time.time()-t0:.1f}s")
    return gt


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def recall_at_k(preds, gt) -> float:
    hits = sum(len(set(p) & set(g)) for p, g in zip(preds, gt))
    return hits / (len(gt) * len(gt[0]))


def run_benchmark(engine: SHSRSEngine,
                  data_norm: np.ndarray,
                  queries_norm: np.ndarray,
                  ground_truth: np.ndarray,
                  k: int = 10,
                  probe: int | None = None,
                  n_warmup: int = 20,
                  label: str = "") -> dict:
    N = len(data_norm)

    # warmup
    for q in queries_norm[:n_warmup]:
        engine.search(q, k=k, probe=probe)

    # timed run — record per-query latency
    latencies = []
    preds     = []
    cand_sizes = []

    for q in queries_norm:
        t0    = time.perf_counter()
        res   = engine.search(q, k=k, probe=probe)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed * 1000)
        preds.append([r[0] for r in res])
        cand_sizes.append(len(res))

    latencies = np.array(latencies)
    recall    = recall_at_k(preds, ground_truth)
    cand_pct  = np.mean(cand_sizes) / N * 100
    qps       = len(queries_norm) / (latencies.sum() / 1000)

    return {
        "label":    label,
        "probe":    probe,
        "recall":   recall,
        "lat_mean": latencies.mean(),
        "lat_p50":  np.percentile(latencies, 50),
        "lat_p95":  np.percentile(latencies, 95),
        "lat_p99":  np.percentile(latencies, 99),
        "cand_pct": cand_pct,
        "qps":      qps,
    }


def print_result(r: dict, baseline_recall: float = 0.797,
                 baseline_lat: float = 6.61):
    recall_delta = (r["recall"] - baseline_recall) * 100
    lat_delta    = (baseline_lat - r["lat_mean"]) / baseline_lat * 100
    print(f"  {r['label']:<30} "
          f"Recall={r['recall']*100:>5.1f}% ({recall_delta:>+5.1f}pp)  "
          f"Mean={r['lat_mean']:>6.2f}ms  "
          f"p95={r['lat_p95']:>6.2f}ms  "
          f"p99={r['lat_p99']:>6.2f}ms  "
          f"QPS={r['qps']:>6.0f}  "
          f"Cands={r['cand_pct']:.3f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SUSTAINED QPS TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_qps_test(engine: SHSRSEngine, queries_norm: np.ndarray,
                 probe: int | None, duration_sec: float = 5.0) -> dict:
    """Run queries continuously for duration_sec, measure sustained QPS."""
    n_queries = 0
    t_end     = time.perf_counter() + duration_sec
    idx       = 0

    while time.perf_counter() < t_end:
        q = queries_norm[idx % len(queries_norm)]
        engine.search(q, k=10, probe=probe)
        n_queries += 1
        idx += 1

    return {
        "n_queries":    n_queries,
        "duration_sec": duration_sec,
        "qps":          n_queries / duration_sec,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark SHSRSEngine")
    parser.add_argument("--index",     default="shsrs_production_index")
    parser.add_argument("--embeddings", default="shsrs_real_embeddings.npy")
    parser.add_argument("--n_queries", type=int, default=300)
    parser.add_argument("--k",         type=int, default=10)
    parser.add_argument("--probe",     type=int, default=None,
                        help="If set, only benchmark this fixed probe")
    parser.add_argument("--quick",     action="store_true",
                        help="Quick mode: 100 queries, skip QPS test")
    args = parser.parse_args()

    if args.quick:
        args.n_queries = 100

    print("=" * 80)
    print("SHSRS — Production Benchmark")
    print("=" * 80)

    # ── load engine ───────────────────────────────────────────────────────────
    print(f"\n[1] Loading engine from {args.index} ...")
    t0 = time.time()
    engine = SHSRSEngine.load(args.index)
    load_time = time.time() - t0
    print(f"    {engine}")
    print(f"    Load time: {load_time:.2f}s")
    print(f"    Est RAM  : {engine.ram_estimate_mb():.0f} MB")

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\n[2] Loading embeddings from {args.embeddings} ...")
    import faiss
    raw       = np.load(args.embeddings).astype(np.float32)
    N, D      = raw.shape
    norms     = np.linalg.norm(raw, axis=1, keepdims=True)
    data_norm = raw / np.where(norms == 0, 1.0, norms)
    print(f"    {N:,} × {D}D loaded")

    # ── sample queries ────────────────────────────────────────────────────────
    rng           = np.random.default_rng(42)
    query_indices = rng.choice(N, size=args.n_queries, replace=False)
    queries_norm  = data_norm[query_indices]
    print(f"    {args.n_queries} queries sampled (seed=42)")

    # ── ground truth ──────────────────────────────────────────────────────────
    print(f"\n[3] Ground truth (k={args.k})")
    ground_truth = compute_ground_truth(data_norm, queries_norm, args.k)

    # ── benchmark ─────────────────────────────────────────────────────────────
    print(f"\n[4] Benchmark results")
    print(f"    (Baseline: flat KMeans 500c probe=4 → Recall=79.7%  Lat=6.61ms)\n")
    header = (f"  {'Method':<30} {'Recall':>12}  {'Mean':>8}  "
              f"{'p95':>8}  {'p99':>8}  {'QPS':>7}  {'Cands%':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_results = []

    if args.probe is not None:
        # single fixed probe
        r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                          k=args.k, probe=args.probe,
                          label=f"fixed probe={args.probe}")
        print_result(r)
        all_results.append(r)
    else:
        # adaptive probe first
        r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                          k=args.k, probe=None,
                          label="adaptive probe")
        print_result(r)
        all_results.append(r)

        # fixed probe sweep
        for probe in [6, 8, 10, 12, 15, 20]:
            r = run_benchmark(engine, data_norm, queries_norm, ground_truth,
                              k=args.k, probe=probe,
                              label=f"fixed probe={probe}")
            print_result(r)
            all_results.append(r)

    # ── sustained QPS test ────────────────────────────────────────────────────
    if not args.quick:
        print(f"\n[5] Sustained QPS test (5 seconds each)")
        print(f"    {'Method':<25} {'QPS':>8}  {'ms/query':>10}")
        print(f"    {'-'*25} {'-'*8}  {'-'*10}")

        for probe in ([None] if args.probe is None else [args.probe]):
            label = "adaptive" if probe is None else f"probe={probe}"
            res   = run_qps_test(engine, queries_norm, probe=probe)
            print(f"    {label:<25} {res['qps']:>8.0f}  "
                  f"{1000/res['qps']:>9.2f}ms")

        if args.probe is None:
            for probe in [8, 12]:
                res   = run_qps_test(engine, queries_norm, probe=probe)
                label = f"probe={probe}"
                print(f"    {label:<25} {res['qps']:>8.0f}  "
                      f"{1000/res['qps']:>9.2f}ms")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    best_recall = max(all_results, key=lambda r: r["recall"])
    best_speed  = min(all_results, key=lambda r: r["lat_mean"])
    best_balanced = min(
        [r for r in all_results if r["recall"] >= 0.90],
        key=lambda r: r["lat_mean"],
        default=best_recall
    )

    print(f"  Best recall  : {best_recall['label']:<25} "
          f"Recall={best_recall['recall']*100:.1f}%  "
          f"Lat={best_recall['lat_mean']:.2f}ms")
    print(f"  Best speed   : {best_speed['label']:<25} "
          f"Recall={best_speed['recall']*100:.1f}%  "
          f"Lat={best_speed['lat_mean']:.2f}ms")
    print(f"  Best balanced: {best_balanced['label']:<25} "
          f"Recall={best_balanced['recall']*100:.1f}%  "
          f"Lat={best_balanced['lat_mean']:.2f}ms")
    print(f"\n  Original baseline: Recall=79.7%  Lat=6.61ms  QPS~151")
    print(f"{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
