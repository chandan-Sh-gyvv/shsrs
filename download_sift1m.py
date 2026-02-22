"""
download_sift1m.py
==================
Downloads SIFT1M (1 million 128D float32 vectors) and converts it to
the .npy format expected by SHSRSEngine.

SIFT1M is the standard ANN benchmark dataset used by FAISS, HNSW, ScaNN.
Results on SIFT1M are directly comparable to published literature.

Dataset stats:
  - 1,000,000 base vectors (128D float32)
  - 10,000 query vectors
  - 100 ground truth neighbours per query
  - Source: http://corpus-texmex.irisa.fr/

Output files:
  sift1m_base.npy        - 1M × 128D base vectors (raw, not normalised)
  sift1m_queries.npy     - 10K × 128D query vectors
  sift1m_groundtruth.npy - 10K × 100 ground truth indices

Usage:
    python download_sift1m.py
    python download_sift1m.py --out_dir sift1m_data
    python download_sift1m.py --skip_download  # if already downloaded
"""

import os
import sys
import struct
import argparse
import urllib.request
import tarfile
import numpy as np
from pathlib import Path


# ── URLs ──────────────────────────────────────────────────────────────────────
SIFT_URL  = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_SIZE = 168_083_265   # ~160 MB compressed


# ══════════════════════════════════════════════════════════════════════════════
# FVECS / IVECS READERS
# ══════════════════════════════════════════════════════════════════════════════

def read_fvecs(path: Path) -> np.ndarray:
    """Read a .fvecs file → float32 array [N, D]."""
    with open(path, "rb") as f:
        data = f.read()
    # each vector: [int32 dim][float32 × dim]
    dim = struct.unpack_from("i", data, 0)[0]
    record_size = 4 + dim * 4   # 4 bytes dim + dim floats
    n = len(data) // record_size
    print(f"  Reading {path.name}: {n:,} vectors × {dim}D")
    vecs = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        offset = i * record_size + 4   # skip dim header
        vecs[i] = struct.unpack_from(f"{dim}f", data, offset)
    return vecs


def read_fvecs_fast(path: Path) -> np.ndarray:
    """Faster fvecs reader using numpy stride tricks."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    dim = data[0]
    # each record is (1 + dim) int32s where first is dim header
    data = data.reshape(-1, 1 + dim)
    # columns 1: are the float data
    return data[:, 1:].view(np.float32).copy()


def read_ivecs(path: Path) -> np.ndarray:
    """Read a .ivecs file → int32 array [N, K]."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    k = data[0]
    data = data.reshape(-1, 1 + k)
    return data[:, 1:].copy()


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 / 1024
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:>5.1f}%  {mb:.0f}MB", end="", flush=True)


def download_sift(out_dir: Path):
    tar_path = out_dir / "sift.tar.gz"

    if tar_path.exists():
        print(f"  [CACHE] {tar_path} already exists, skipping download")
    else:
        print(f"  Downloading SIFT1M from {SIFT_URL}")
        print(f"  Expected size: ~160 MB")
        try:
            urllib.request.urlretrieve(SIFT_URL, tar_path, show_progress)
            print()  # newline after progress bar
        except Exception as e:
            print(f"\n  Download failed: {e}")
            print(f"\n  Manual download instructions:")
            print(f"  1. Go to: http://corpus-texmex.irisa.fr/")
            print(f"  2. Download: sift.tar.gz")
            print(f"  3. Place it in: {out_dir}/sift.tar.gz")
            print(f"  4. Re-run this script with --skip_download")
            sys.exit(1)

    # extract
    sift_dir = out_dir / "sift"
    if sift_dir.exists():
        print(f"  [CACHE] Already extracted to {sift_dir}")
    else:
        print(f"  Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(out_dir)
        print(f"  Extracted to {sift_dir}")

    return sift_dir


# ══════════════════════════════════════════════════════════════════════════════
# CONVERT
# ══════════════════════════════════════════════════════════════════════════════

def convert(sift_dir: Path, out_dir: Path):
    base_fvecs  = sift_dir / "sift_base.fvecs"
    query_fvecs = sift_dir / "sift_query.fvecs"
    gt_ivecs    = sift_dir / "sift_groundtruth.ivecs"

    for f in [base_fvecs, query_fvecs, gt_ivecs]:
        if not f.exists():
            raise FileNotFoundError(f"Expected file not found: {f}")

    # ── base vectors ──────────────────────────────────────────────────────────
    print("\n  Converting base vectors (1M × 128D)...")
    base = read_fvecs_fast(base_fvecs)
    print(f"  Shape: {base.shape}  dtype: {base.dtype}")
    out_base = out_dir / "sift1m_base.npy"
    np.save(out_base, base)
    size_mb = out_base.stat().st_size / 1024 / 1024
    print(f"  Saved → {out_base}  ({size_mb:.0f} MB)")

    # ── query vectors ─────────────────────────────────────────────────────────
    print("\n  Converting query vectors (10K × 128D)...")
    queries = read_fvecs_fast(query_fvecs)
    print(f"  Shape: {queries.shape}  dtype: {queries.dtype}")
    out_queries = out_dir / "sift1m_queries.npy"
    np.save(out_queries, queries)
    print(f"  Saved → {out_queries}")

    # ── ground truth ──────────────────────────────────────────────────────────
    print("\n  Converting ground truth (10K × 100 neighbours)...")
    gt = read_ivecs(gt_ivecs)
    print(f"  Shape: {gt.shape}  dtype: {gt.dtype}")
    out_gt = out_dir / "sift1m_groundtruth.npy"
    np.save(out_gt, gt)
    print(f"  Saved → {out_gt}")

    return base, queries, gt


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def validate(base, queries, gt):
    print("\n  Validating...")

    # basic shape checks
    assert base.shape    == (1_000_000, 128), f"Unexpected base shape: {base.shape}"
    assert queries.shape == (10_000, 128),    f"Unexpected query shape: {queries.shape}"
    assert gt.shape      == (10_000, 100),    f"Unexpected GT shape: {gt.shape}"

    # spot check: brute force top-1 for first 5 queries
    # should match ground truth index 0
    q   = queries[:5].astype(np.float32)
    b   = base.astype(np.float32)
    # L2 distance (SIFT uses L2, not cosine)
    # for spot check just verify GT[0] is plausible
    print(f"  GT[0] top-5 neighbours: {gt[0, :5]}")
    print(f"  Base vector norms (first 5): "
          f"{[f'{n:.1f}' for n in np.linalg.norm(base[:5], axis=1)]}")

    print("  All shape checks passed.")


# ══════════════════════════════════════════════════════════════════════════════
# SHSRS COMPATIBILITY NOTE
# ══════════════════════════════════════════════════════════════════════════════

def print_usage_note(out_dir: Path):
    print(f"""
{'='*65}
SIFT1M READY — SHSRS Usage Notes
{'='*65}

Files saved to: {out_dir}
  sift1m_base.npy        1,000,000 × 128D  float32  (~512 MB)
  sift1m_queries.npy        10,000 × 128D  float32
  sift1m_groundtruth.npy    10,000 × 100   int32

IMPORTANT: SIFT1M uses L2 distance, not cosine similarity.
SHSRS uses cosine (inner product on L2-normalised vectors).

Two options to handle this:

  Option A — Use SIFT as-is with L2 (change FAISS metric):
    Requires modifying engine.py to use METRIC_L2 instead of
    METRIC_INNER_PRODUCT. Ground truth is valid as-is.

  Option B — L2-normalise and use cosine (recommended for SHSRS):
    Normalise the vectors before building the index.
    Ground truth will differ slightly from provided GT,
    so you'll need to recompute GT after normalisation.
    The benchmark script handles this automatically.

The benchmark script (benchmark_sift1m.py) handles Option B
automatically — just point it at the files above.

Recommended SHSRS config for 1M × 128D:
  n_clusters    = 1000   (sqrt(1M) rule)
  M             = 16     (higher dim benefits from more links)
  ef_construction = 200
  RAM estimate  : ~1.5 GB index + ~512 MB vectors = ~2 GB total
  Build time    : ~2-4 hours (kNN graph is the bottleneck)

{'='*65}
""")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download and convert SIFT1M")
    parser.add_argument("--out_dir", default="sift1m_data",
                        help="Output directory (default: sift1m_data)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, use existing sift.tar.gz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SIFT1M Downloader & Converter for SHSRS")
    print("=" * 65)

    # check if already converted
    already_done = all(
        (out_dir / f).exists()
        for f in ["sift1m_base.npy", "sift1m_queries.npy",
                  "sift1m_groundtruth.npy"]
    )
    if already_done:
        print("\n[CACHE] All .npy files already exist.")
        base    = np.load(out_dir / "sift1m_base.npy",        mmap_mode="r")
        queries = np.load(out_dir / "sift1m_queries.npy",     mmap_mode="r")
        gt      = np.load(out_dir / "sift1m_groundtruth.npy", mmap_mode="r")
        print(f"  Base:    {base.shape}")
        print(f"  Queries: {queries.shape}")
        print(f"  GT:      {gt.shape}")
        print_usage_note(out_dir)
        return

    # download
    print("\n[1] Download")
    if args.skip_download:
        sift_dir = out_dir / "sift"
        print(f"  Skipping download, using existing files in {sift_dir}")
    else:
        sift_dir = download_sift(out_dir)

    # convert
    print("\n[2] Convert .fvecs/.ivecs → .npy")
    base, queries, gt = convert(sift_dir, out_dir)

    # validate
    print("\n[3] Validate")
    validate(base, queries, gt)

    print_usage_note(out_dir)


if __name__ == "__main__":
    main()
