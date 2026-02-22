"""
shsrs/engine.py
===============
SHSRS — Semantic Hierarchical Search with Refined Subspace
Production search engine. No experiment code, no benchmarking.

Architecture:
    Query → centroid router (gap-adaptive probe) → local HNSW per cluster → merge → rerank

Usage:
    from shsrs.engine import SHSRSEngine

    # Build once
    engine = SHSRSEngine.build(
        vectors=my_vectors,          # np.ndarray [N, D] float32
        index_dir="shsrs_index",     # where to save
        n_clusters=100,
        M=8,
        ef_construction=200,
    )

    # Or load from disk
    engine = SHSRSEngine.load("shsrs_index")

    # Search
    results = engine.search(query_vector, k=10)
    # → list of (global_id, score) sorted by score desc

    # Batch search
    results = engine.search_batch(query_matrix, k=10)
    # → list of lists of (global_id, score)
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# ── defaults (match validated experiment settings) ────────────────────────────
_DEFAULT_N_CLUSTERS     = 100
_DEFAULT_M              = 8
_DEFAULT_EF_CONSTRUCTION = 200
_DEFAULT_EF_SEARCH      = 50
_DEFAULT_IGAR_ITERS     = 30
_DEFAULT_IGAR_SAMPLE    = 7500   # overridden by build() to max(7500, 0.20*N)
_DEFAULT_KNN_K          = 15

# Gap-adaptive probe policy
# Calibrated on 150K Wikipedia sentences (all-MiniLM-L6-v2, 384D).
# Re-run SHSRSEngine.calibrate_gap_policy() if switching datasets/models.
_DEFAULT_GAP_POLICY = [
    (0.185, 6),   # gap > p90 → very confident  → probe 6  (~88% recall)
    (0.124, 8),   # gap > p75 → confident        → probe 8  (~90% recall)
    (0.066, 12),  # gap > p50 → moderate         → probe 12 (~94% recall)
    (0.027, 16),  # gap > p25 → harder           → probe 16 (~96% recall)
    (0.000, 20),  # gap ≤ p25 → boundary query   → probe 20 (~97% recall)
]


# ══════════════════════════════════════════════════════════════════════════════
class SHSRSEngine:
    """
    Production semantic search engine.

    Build once, search many times.
    Thread-safe for concurrent reads (faiss IndexHNSWFlat is read-safe).
    """

    def __init__(self):
        self._data_norm:   Optional[np.ndarray]       = None
        self._centers:     Optional[np.ndarray]       = None
        self._indexes:     dict[int, faiss.Index]     = {}
        self._id_maps:     dict[int, np.ndarray]      = {}
        self._gap_policy:  list[tuple[float, int]]    = _DEFAULT_GAP_POLICY
        self._ef_search:   int                        = _DEFAULT_EF_SEARCH
        self._n_clusters:  int                        = _DEFAULT_N_CLUSTERS
        self._dim:         int                        = 0
        self._n_vectors:   int                        = 0

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        index_dir: str | Path,
        n_clusters: int  = _DEFAULT_N_CLUSTERS,
        M: int           = _DEFAULT_M,
        ef_construction: int = _DEFAULT_EF_CONSTRUCTION,
        igar_iters: int  = _DEFAULT_IGAR_ITERS,
        igar_sample: int = _DEFAULT_IGAR_SAMPLE,
        knn_k: int       = _DEFAULT_KNN_K,
        gap_policy: Optional[list] = None,
    ) -> "SHSRSEngine":
        """
        Build the index from raw vectors and save to disk.

        Parameters
        ----------
        vectors       : float32 array [N, D]. Will be L2-normalised internally.
        index_dir     : directory to write index files (created if absent).
        n_clusters    : number of macro clusters (100 recommended for ≤500K vecs).
        M             : HNSW graph degree. M=8 is RAM-efficient; M=16 for more recall.
        ef_construction: HNSW build quality. Higher = better recall at build time.
        igar_iters    : IGAR refinement iterations (30 is a good default).
        igar_sample   : vectors sampled per IGAR iteration.
        knn_k         : neighbours in kNN graph used by IGAR.
        gap_policy    : custom [(gap_threshold, probe_count), ...] list.
                        If None, uses the default calibrated policy.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        engine = cls()
        engine._n_clusters = n_clusters
        engine._gap_policy = gap_policy or _DEFAULT_GAP_POLICY

        N, D = vectors.shape
        engine._dim      = D
        engine._n_vectors = N
        logger.info(f"Building SHSRS index: {N:,} × {D}D → {n_clusters} clusters")

        # 1. Normalise
        logger.info("Normalising vectors...")
        data_norm = _cosine_normalize(vectors)
        engine._data_norm = data_norm

        # 2. KMeans
        logger.info("Running KMeans...")
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096,
                             n_init=5, random_state=42)
        km.fit(data_norm)
        labels  = km.labels_
        centers = _cosine_normalize(km.cluster_centers_)
        logger.info(f"KMeans done in {time.time()-t0:.1f}s")

        # 3. kNN graph for IGAR
        logger.info(f"Building kNN graph (k={knn_k})...")
        t0 = time.time()
        knn_graph = _build_knn_graph_faiss(data_norm, knn_k)
        logger.info(f"kNN graph built in {time.time()-t0:.1f}s")

        # 4. IGAR
        # Scale sample size to corpus: 20% of N, minimum 7500
        effective_sample = max(igar_sample, int(0.20 * N))
        if effective_sample != igar_sample:
            logger.info(f"IGAR sample scaled: {igar_sample} → {effective_sample:,} "
                        f"(20% of {N:,})")
        logger.info(f"Running IGAR ({igar_iters} iterations, "
                    f"sample={effective_sample:,})...")
        labels = _run_igar(labels, knn_graph, n_clusters, igar_iters, effective_sample)

        engine._centers = centers

        # 5. Per-cluster HNSW indexes
        logger.info(f"Building {n_clusters} local HNSW indexes (M={M})...")
        t0 = time.time()
        for c in range(n_clusters):
            mask = (labels == c)
            gids = np.where(mask)[0].astype(np.int64)
            vecs = data_norm[gids]
            if len(gids) == 0:
                continue

            idx = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = ef_construction
            idx.add(vecs)

            engine._indexes[c] = idx
            engine._id_maps[c] = gids

        logger.info(f"HNSW indexes built in {time.time()-t0:.1f}s")

        # 6. Save
        engine._save(index_dir, labels, M, ef_construction)
        logger.info(f"Index saved to {index_dir}")

        return engine

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ══════════════════════════════════════════════════════════════════════════

    def _save(self, index_dir: Path, labels: np.ndarray,
              M: int, ef_construction: int):
        np.save(index_dir / "data_norm.npy",  self._data_norm)
        np.save(index_dir / "centers.npy",    self._centers)
        np.save(index_dir / "labels.npy",     labels)

        for c, idx in self._indexes.items():
            faiss.write_index(idx, str(index_dir / f"hnsw_c{c}.faiss"))
            np.save(index_dir / f"map_c{c}.npy", self._id_maps[c])

        meta = {
            "n_clusters":      self._n_clusters,
            "dim":             self._dim,
            "n_vectors":       self._n_vectors,
            "M":               M,
            "ef_construction": ef_construction,
            "ef_search":       self._ef_search,
            "gap_policy":      self._gap_policy,
            "n_indexes":       len(self._indexes),
        }
        with open(index_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path) -> "SHSRSEngine":
        """Load a previously built index from disk."""
        index_dir = Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")

        with open(index_dir / "meta.json") as f:
            meta = json.load(f)

        engine = cls()
        engine._n_clusters  = meta["n_clusters"]
        engine._dim         = meta["dim"]
        engine._n_vectors   = meta["n_vectors"]
        engine._ef_search   = meta["ef_search"]
        engine._gap_policy  = [tuple(p) for p in meta["gap_policy"]]

        logger.info(f"Loading SHSRS index from {index_dir} "
                    f"({engine._n_vectors:,} vectors, "
                    f"{engine._n_clusters} clusters)...")

        engine._data_norm = np.load(index_dir / "data_norm.npy")
        engine._centers   = np.load(index_dir / "centers.npy")

        for c in range(engine._n_clusters):
            idx_path = index_dir / f"hnsw_c{c}.faiss"
            map_path = index_dir / f"map_c{c}.npy"
            if idx_path.exists():
                engine._indexes[c] = faiss.read_index(str(idx_path))
                engine._id_maps[c] = np.load(map_path)

        logger.info("Index loaded.")
        return engine

    # ══════════════════════════════════════════════════════════════════════════
    # SEARCH
    # ══════════════════════════════════════════════════════════════════════════

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        probe: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Search for the k nearest neighbours of a single query vector.

        Parameters
        ----------
        query : 1-D float array of length D. Raw or pre-normalised — both work.
        k     : number of results to return.
        probe : fixed probe count. If None, gap-adaptive probe is used.

        Returns
        -------
        List of (global_id, cosine_score) sorted by score descending.
        """
        q = _cosine_normalize(query.reshape(1, -1).astype(np.float32))[0]
        return self._search_normalised(q, k=k, probe=probe)

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        probe: Optional[int] = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Search for k nearest neighbours for a batch of query vectors.

        Parameters
        ----------
        queries : 2-D float array [Q, D].
        k       : number of results per query.
        probe   : fixed probe count. If None, gap-adaptive probe is used.

        Returns
        -------
        List of Q result lists, each a list of (global_id, cosine_score).
        """
        qs = _cosine_normalize(queries.astype(np.float32))
        return [self._search_normalised(q, k=k, probe=probe) for q in qs]

    def _search_normalised(
        self,
        q: np.ndarray,
        k: int,
        probe: Optional[int],
    ) -> list[tuple[int, float]]:
        scores       = self._centers @ q
        n_probe      = probe if probe is not None else self._probe_from_gap(scores)
        top_clusters = np.argpartition(-scores, min(n_probe, len(scores)-1))[:n_probe]

        candidate_ids = []
        for c in top_clusters:
            idx  = self._indexes.get(int(c))
            gids = self._id_maps.get(int(c))
            if idx is None:
                continue
            idx.hnsw.efSearch = max(self._ef_search, k)
            actual_k = min(k, idx.ntotal)
            _, lids = idx.search(q.reshape(1, -1), actual_k)
            valid = lids[0][lids[0] >= 0]
            candidate_ids.append(gids[valid])

        if not candidate_ids:
            return []

        cands = np.concatenate(candidate_ids)
        if len(cands) == 0:
            return []

        # rerank by exact cosine
        cand_scores = self._data_norm[cands] @ q
        if len(cands) > k:
            top_idx = np.argpartition(-cand_scores, k)[:k]
        else:
            top_idx = np.arange(len(cands))

        top_idx = top_idx[np.argsort(-cand_scores[top_idx])]
        return [(int(cands[i]), float(cand_scores[i])) for i in top_idx]

    # ══════════════════════════════════════════════════════════════════════════
    # ADAPTIVE PROBE
    # ══════════════════════════════════════════════════════════════════════════

    def _probe_from_gap(self, scores: np.ndarray) -> int:
        top2 = np.partition(-scores, 1)[:2]
        gap  = float(-top2[0] + top2[1])
        for threshold, probe in self._gap_policy:
            if gap > threshold:
                return probe
        return self._gap_policy[-1][1]

    def calibrate_gap_policy(
        self,
        sample_queries: np.ndarray,
        probe_tiers: list[tuple[float, int]] | None = None,
    ) -> list[tuple[float, int]]:
        """
        Recalibrate gap thresholds from a sample of real queries.
        Call this when switching to a new dataset or embedding model.

        Parameters
        ----------
        sample_queries : representative query vectors [Q, D].
        probe_tiers    : list of (percentile, probe_count) pairs.
                         Default: [(90,6),(75,8),(50,12),(25,16),(0,20)]

        Returns
        -------
        New gap_policy list (also updates self._gap_policy in place).
        """
        if probe_tiers is None:
            probe_tiers = [(90, 6), (75, 8), (50, 12), (25, 16), (0, 20)]

        qs = _cosine_normalize(sample_queries.astype(np.float32))
        gaps = []
        for q in qs:
            scores = self._centers @ q
            top2   = np.partition(-scores, 1)[:2]
            gaps.append(float(-top2[0] + top2[1]))
        gaps = np.array(gaps)

        policy = []
        for pct, probe in sorted(probe_tiers, reverse=True):
            threshold = float(np.percentile(gaps, pct)) if pct > 0 else 0.0
            policy.append((threshold, probe))

        self._gap_policy = policy
        logger.info(f"Gap policy recalibrated: {policy}")
        return policy

    # ══════════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    def ram_estimate_mb(self) -> float:
        """Rough RAM estimate for loaded index (vectors + graph links)."""
        vec_bytes   = self._n_vectors * self._dim * 4
        graph_bytes = sum(len(g) for g in self._id_maps.values()) * 16 * 2 * 4
        return (vec_bytes + graph_bytes) / 1024 / 1024

    def __repr__(self) -> str:
        return (f"SHSRSEngine(n_vectors={self._n_vectors:,}, "
                f"dim={self._dim}, n_clusters={self._n_clusters}, "
                f"n_indexes={len(self._indexes)})")


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _build_knn_graph_faiss(data: np.ndarray, k: int) -> np.ndarray:
    N, D  = data.shape
    index = faiss.IndexFlatIP(D)
    index.add(data)
    _, indices = index.search(data, k + 1)
    return indices[:, 1:].astype(np.int32)


def _run_igar(labels: np.ndarray, knn_graph: np.ndarray,
              n_clusters: int, n_iters: int, sample_size: int) -> np.ndarray:
    labels = labels.copy()
    N      = len(labels)

    def retention():
        return (labels[knn_graph] == labels[:, None]).sum() / (N * knn_graph.shape[1])

    logger.info(f"IGAR iter 0: retention={retention():.4f}")
    for it in range(1, n_iters + 1):
        idx        = np.random.choice(N, size=min(sample_size, N), replace=False)
        new_labels = labels.copy()
        for i in idx:
            nbrs = knn_graph[i]
            if len(nbrs):
                new_labels[i] = np.bincount(labels[nbrs], minlength=n_clusters).argmax()
        labels = new_labels
        if it % 10 == 0:
            logger.info(f"IGAR iter {it}: retention={retention():.4f}")

    logger.info(f"IGAR done: final retention={retention():.4f}")
    return labels
