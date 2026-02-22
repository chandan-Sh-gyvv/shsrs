"""
migrate.py
==========
One-time migration: converts the experiment cache (shsrs_hnsw_cache/)
built by shsrs_hnsw_hybrid.py into a clean production index
readable by SHSRSEngine.load().

Run once, then use SHSRSEngine.load("shsrs_production_index") going forward.

Usage:
    python migrate.py
    python migrate.py --src shsrs_hnsw_cache --dst shsrs_production_index
"""

import argparse
import os
import json
import shutil
import time
from pathlib import Path

import numpy as np
import faiss


def migrate(src: Path, dst: Path, n_clusters: int = 100, M: int = 8,
            ef_construction: int = 200, ef_search: int = 50):

    print(f"Migrating experiment cache → production index")
    print(f"  Source : {src}")
    print(f"  Dest   : {dst}")

    if not src.exists():
        raise FileNotFoundError(f"Source cache not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    # ── copy data_norm ────────────────────────────────────────────────────────
    # The experiment scripts don't cache data_norm directly,
    # so we need to reload the raw embeddings and normalise.
    embeddings_path = Path("shsrs_real_embeddings.npy")
    if not embeddings_path.exists():
        raise FileNotFoundError(
            "shsrs_real_embeddings.npy not found. "
            "Run from the same directory as your embeddings file."
        )

    print("\n[1] Loading and normalising embeddings...")
    data = np.load(embeddings_path).astype(np.float32)
    N, D = data.shape
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    data_norm = data / norms
    np.save(dst / "data_norm.npy", data_norm)
    print(f"    {N:,} × {D}D → data_norm.npy saved")

    # ── copy centers + labels ─────────────────────────────────────────────────
    print("\n[2] Copying cluster centers and labels...")
    center_src = src / f"centers_{n_clusters}.npy"
    label_src  = src / f"labels_{n_clusters}.npy"

    if not center_src.exists():
        raise FileNotFoundError(
            f"centers_{n_clusters}.npy not found in {src}. "
            f"Run shsrs_hnsw_hybrid.py with n_clusters={n_clusters} first."
        )

    shutil.copy(center_src, dst / "centers.npy")
    shutil.copy(label_src,  dst / "labels.npy")
    print(f"    centers_{n_clusters}.npy → centers.npy")
    print(f"    labels_{n_clusters}.npy  → labels.npy")

    # ── copy HNSW indexes ─────────────────────────────────────────────────────
    print(f"\n[3] Copying {n_clusters} HNSW indexes (M={M})...")
    hnsw_src = src / f"nc{n_clusters}"
    if not hnsw_src.exists():
        raise FileNotFoundError(
            f"HNSW index directory not found: {hnsw_src}. "
            f"Run shsrs_hnsw_hybrid.py with n_clusters={n_clusters} first."
        )

    n_copied = 0
    for c in range(n_clusters):
        idx_src = hnsw_src / f"hnsw_c{c}_M{M}.faiss"
        map_src = hnsw_src / f"hnsw_c{c}_M{M}_map.npy"

        if not idx_src.exists():
            print(f"    WARNING: cluster {c} index missing — skipping")
            continue

        shutil.copy(idx_src, dst / f"hnsw_c{c}.faiss")
        shutil.copy(map_src, dst / f"map_c{c}.npy")
        n_copied += 1

    print(f"    Copied {n_copied}/{n_clusters} cluster indexes")

    # ── write meta.json ───────────────────────────────────────────────────────
    print("\n[4] Writing meta.json...")
    gap_policy = [
        [0.185, 6],
        [0.124, 8],
        [0.066, 12],
        [0.027, 16],
        [0.000, 20],
    ]
    meta = {
        "n_clusters":      n_clusters,
        "dim":             D,
        "n_vectors":       N,
        "M":               M,
        "ef_construction": ef_construction,
        "ef_search":       ef_search,
        "gap_policy":      gap_policy,
        "n_indexes":       n_copied,
    }
    with open(dst / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    meta.json written")

    # ── verify ────────────────────────────────────────────────────────────────
    print("\n[5] Verifying production index...")
    import sys
    # ensure the shsrs package directory is on the path
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    # also try cwd in case script is run from a different location
    cwd = Path(os.getcwd())
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    from shsrs import SHSRSEngine

    engine = SHSRSEngine.load(dst)
    sample_query = data_norm[0]
    results = engine.search(sample_query, k=5)
    assert len(results) > 0, "Search returned no results!"
    print(f"    Test search OK — top result: id={results[0][0]}, "
          f"score={results[0][1]:.4f}")
    print(f"    Engine: {engine}")

    print(f"\n{'='*60}")
    print(f"Migration complete → {dst}")
    print(f"  Load with: engine = SHSRSEngine.load('{dst}')")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="shsrs_hnsw_cache",
                        help="Experiment cache directory")
    parser.add_argument("--dst", default="shsrs_production_index",
                        help="Production index output directory")
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--M", type=int, default=8)
    args = parser.parse_args()

    migrate(Path(args.src), Path(args.dst),
            n_clusters=args.n_clusters, M=args.M)
