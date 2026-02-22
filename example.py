"""
example.py
==========
Production usage examples for SHSRSEngine.

Three scenarios:
  A) Build fresh from vectors
  B) Migrate from experiment cache (most common for you right now)
  C) Load and search in production
"""

import logging
import numpy as np
from shsrs import SHSRSEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# A) BUILD FRESH FROM VECTORS
# ══════════════════════════════════════════════════════════════════════════════

def example_build():
    """Build index from scratch. Takes ~20-30 min for 150K vectors."""
    vectors = np.load("shsrs_real_embeddings.npy").astype(np.float32)

    engine = SHSRSEngine.build(
        vectors=vectors,
        index_dir="shsrs_production_index",
        n_clusters=100,
        M=8,
        ef_construction=200,
        igar_iters=30,
    )
    print(engine)
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# B) MIGRATE FROM EXPERIMENT CACHE  ← run this first
# ══════════════════════════════════════════════════════════════════════════════

def example_migrate():
    """
    Convert existing experiment cache to production index.
    Run migrate.py instead of this for a cleaner CLI experience:
        python migrate.py
    """
    import subprocess
    subprocess.run(["python", "migrate.py"], check=True)


# ══════════════════════════════════════════════════════════════════════════════
# C) LOAD AND SEARCH  ← typical production path
# ══════════════════════════════════════════════════════════════════════════════

def example_search():
    """Load index and run searches."""

    # Load (fast — ~0.2s)
    engine = SHSRSEngine.load("shsrs_production_index")
    print(engine)

    # Load some vectors to use as queries
    data = np.load("shsrs_real_embeddings.npy").astype(np.float32)

    # ── single search (adaptive probe) ────────────────────────────────────────
    query = data[42]
    results = engine.search(query, k=10)
    print(f"\nAdaptive probe search (query idx=42):")
    for rank, (gid, score) in enumerate(results, 1):
        print(f"  {rank:>2}. id={gid:<8} score={score:.4f}")

    # ── single search (fixed probe) ───────────────────────────────────────────
    results_fixed = engine.search(query, k=10, probe=12)
    print(f"\nFixed probe=12 search:")
    for rank, (gid, score) in enumerate(results_fixed, 1):
        print(f"  {rank:>2}. id={gid:<8} score={score:.4f}")

    # ── batch search ──────────────────────────────────────────────────────────
    queries = data[:5]
    batch_results = engine.search_batch(queries, k=5)
    print(f"\nBatch search (5 queries, k=5):")
    for i, results in enumerate(batch_results):
        top = results[0]
        print(f"  query {i}: top result id={top[0]}, score={top[1]:.4f}")

    # ── recalibrate for new data/model ────────────────────────────────────────
    # Call this if you switch embedding models or datasets
    sample = data[np.random.choice(len(data), 1000, replace=False)]
    new_policy = engine.calibrate_gap_policy(sample)
    print(f"\nRecalibrated gap policy: {new_policy}")

    return engine


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"

    if mode == "build":
        example_build()
    elif mode == "migrate":
        example_migrate()
    else:
        example_search()
