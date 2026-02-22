# SHSRS — Semantic Hierarchical Search with Refined Subspace

A CPU-friendly approximate nearest neighbour (ANN) search engine for dense vector embeddings.  
Core contribution: **IGAR** (Iterative Graph-Aligned Reassignment) — a cluster refinement algorithm that improves partition quality by maximising kNN neighbour retention rather than centroid proximity.

## Results

| Dataset | Vectors | Recall@10 | Latency | QPS | Index RAM |
|---|---|---|---|---|---|
| Wikipedia (384D) | 150K | 95.4% | 1.91ms | 525 | 238 MB |
| SIFT1M (128D) | 1M | 93.4% | 1.88ms | 533 | 610 MB |

vs flat KMeans baseline: **+15.7pp recall, 3.5x faster, 9x fewer candidates** simultaneously.  
vs brute force: **261x speedup** at 95.4% recall (150K).

---

## Installation

```bash
pip install faiss-cpu numpy scikit-learn
```

Clone the repo:
```bash
git clone https://github.com/yourusername/shsrs.git
cd shsrs
pip install -r requirements.txt
```

---

## Quick Start

### Build an index from your vectors

```python
import numpy as np
from shsrs import SHSRSEngine

vectors = np.load("your_embeddings.npy").astype(np.float32)

engine = SHSRSEngine.build(
    vectors=vectors,
    index_dir="my_index",
    n_clusters=100,      # use ~sqrt(N) as a rule of thumb
    M=8,                 # HNSW graph degree (8=RAM-efficient, 16=more recall)
    ef_construction=200,
)
```

### Load and search

```python
from shsrs import SHSRSEngine

engine = SHSRSEngine.load("my_index")

# Single search — adaptive probe (recommended)
results = engine.search(query_vector, k=10)
# → [(global_id, cosine_score), ...]

# Fixed probe — deterministic latency
results = engine.search(query_vector, k=10, probe=12)

# Batch search
results = engine.search_batch(query_matrix, k=10)
```

### Migrating from the experiment cache

If you ran the experiment scripts first and want to convert to a production index:

```bash
python migrate.py
# → creates shsrs_production_index/
```

---

## Choosing Your Config

| Goal | n_clusters | M | probe | Expected recall |
|---|---|---|---|---|
| ≤500K vectors, RAM-tight | 100 | 8 | adaptive | ~95% |
| ≤500K vectors, high recall | 100 | 16 | 20 | ~97% |
| ~1M vectors, balanced | 1000 | 16 | adaptive | ~93% |
| ~1M vectors, high recall | 1000 | 16 | 32 | ~97% |

**Adaptive probe** (default) uses the centroid score gap to assign fewer probes to easy queries and more to boundary queries. Recalibrate when switching datasets:

```python
engine.calibrate_gap_policy(sample_queries)  # ~1000 representative queries
```

---

## Benchmarking

```bash
# Benchmark your production index
python benchmark.py

# Quick check (100 queries)
python benchmark.py --quick

# Fixed probe only
python benchmark.py --probe 12
```

### SIFT1M Benchmark

```bash
# Download SIFT1M (~160 MB)
python download_sift1m.py

# Build index and benchmark (first run ~55 min, subsequent runs use cache)
python benchmark_sift1m.py --quick
```

---

## Folder Structure

```
shsrs/
├── shsrs/
│   ├── __init__.py          # package entry point
│   └── engine.py            # SHSRSEngine — build, load, search
├── benchmark.py             # production benchmark harness
├── benchmark_sift1m.py      # SIFT1M 1M-scale benchmark
├── download_sift1m.py       # SIFT1M downloader + converter
├── example.py               # usage examples
├── migrate.py               # convert experiment cache → production index
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How It Works

**1. KMeans initialisation** — corpus partitioned into K macro clusters.

**2. IGAR refinement** — iteratively reassigns boundary vectors to the cluster
that the plurality of their kNN neighbours occupy, maximising *neighbour retention*:

```
R(L) = |{(i,j) ∈ kNN_graph : L(i) = L(j)}| / |kNN_edges|
```

On real sentence embeddings: retention improves from 54.3% → 61.5% (+7.25pp).  
On synthetic data: +0.01pp — IGAR's value is data-geometry dependent.

**3. Per-cluster HNSW** — one `faiss.IndexHNSWFlat` built per cluster using IGAR-refined labels.

**4. Gap-adaptive routing** — at query time, the centroid score gap
(top1_score − top2_score) determines probe count:
large gap = query deep inside one cluster = few probes needed;
small gap = boundary query = more probes needed.

---

## Paper

> **SHSRS: Semantic Hierarchical Search with Refined Subspace**  
> Chandan S H — Preprint, February 2026

See `SHSRS_paper.pdf` for full experimental results and analysis.

---

## Requirements

- Python 3.10+
- faiss-cpu >= 1.7.4
- numpy >= 1.24.0
- scikit-learn >= 1.3.0

GPU support: swap `faiss-cpu` for `faiss-gpu` — no code changes required.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
