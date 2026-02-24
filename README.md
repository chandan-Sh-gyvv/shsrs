# SHSRS — Semantic Hierarchical Search with Refined Subspace [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18733029.svg)](https://doi.org/10.5281/zenodo.18733029)

**Topology-aware hybrid ANN search for scalable, low-latency vector retrieval.**

SHSRS combines macro-cluster routing, graph-aligned cluster refinement (IGAR),
and per-cluster HNSW search to decouple query cost from dataset size.

Instead of searching the full corpus or traversing a global graph,
SHSRS routes queries into small local graphs — achieving high recall
with microscopic candidate coverage (~0.001% at 1M scale).

## Why SHSRS?

Traditional ANN systems face one of two trade-offs:

| Approach | Strength | Weakness |
|----------|----------|----------|
| Brute Force | Exact recall | O(N) scaling |
| IVF (flat clustering) | Fast | Boundary recall loss |
| Global HNSW | High recall | High RAM, global traversal cost |

SHSRS solves this by:

- Refining cluster boundaries using **kNN graph alignment (IGAR)**
- Localising HNSW search to small per-cluster graphs
- Using **adaptive probe routing** to control compute per query

Search cost becomes **probe-bounded**, not dataset-bounded.

## Architecture Overview

Query → Centroid Routing → Probe Selection → Local HNSW Search → Merge Results

Build-time pipeline:

1. KMeans partition
2. IGAR refinement (maximise neighbour retention)
3. Per-cluster HNSW index construction
4. Gap calibration for adaptive routing


## Results

### 150K Wikipedia (384D) — with cross-boundary links
| Config | Recall@10 | Latency | QPS |
|---|---|---|---|
| probe=6  | 91.0% | 1.03ms | 975 |
| probe=12 | 95.1% | 1.75ms | 573 |
| probe=20 | 97.3% | 2.76ms | 362 |
| adaptive | 95.7% | 2.20ms | 454 |

### 1M SIFT (128D) — with cross-boundary links
| Config | Recall@10 | Latency | QPS |
|---|---|---|---|
| probe=6  | 93.5% | 0.48ms | 2063 |
| probe=8  | 96.3% | 0.61ms | 1640 |
| probe=16 | 99.1% | 1.09ms | 918  |
| probe=32 | 99.9% | 1.97ms | 509  |

vs flat IVF baseline: +19.0pp recall, 6.4x lower latency
vs brute force: 261x speedup at 97.3% recall (150K)

## Scaling Behaviour

At 1M vectors (SIFT1M):

- Candidate coverage: ~0.001%
- 97.8% Recall@10 @ 3.79ms (CPU)
- Index RAM: 610 MB
- Build time: ~55 min CPU / ~2 min GPU
- Search latency scales with **probe count**, not dataset size.
---
## Paper
📄 **[SHSRS_paper.pdf](SHSRS_paper.pdf)**

> SHSRS: Semantic Hierarchical Search with Refined Subspace  
> Chandan S H, Karnataka Bengaluru — Preprint, 22 February 2026
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

## Requirements

- Python 3.10+
- faiss-cpu >= 1.7.4
- numpy >= 1.24.0
- scikit-learn >= 1.3.0

GPU support: swap `faiss-cpu` for `faiss-gpu` — no code changes required.

---
## When to Use SHSRS

- CPU-first deployments
- Large embedding corpora (100K – 10M scale)
- Retrieval-augmented generation (RAG)
- Semantic content search
- Cost-sensitive vector search infrastructure

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---
## Citation

If you use SHSRS in your research, please cite:

Chandan S H. (2026). SHSRS: Semantic Hierarchical Search with Refined Subspace. 
Zenodo. https://doi.org/10.5281/zenodo.18733029
