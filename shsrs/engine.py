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

    # Batch search (optimised — groups queries by cluster, one FAISS call per cluster)
    results = engine.search_batch(query_matrix, k=10)
    # → list of lists of (global_id, score)
"""

from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self._data_norm:       Optional[np.ndarray]       = None
        self._centers:         Optional[np.ndarray]       = None
        self._indexes:         dict[int, faiss.Index]     = {}
        self._id_maps:         dict[int, np.ndarray]      = {}
        self._boundary_links:  dict[int, np.ndarray]      = {}
        self._gap_policy:      list[tuple[float, int]]    = _DEFAULT_GAP_POLICY
        self._ef_search:       int                        = _DEFAULT_EF_SEARCH
        self._n_clusters:      int                        = _DEFAULT_N_CLUSTERS
        self._dim:             int                        = 0
        self._n_vectors:       int                        = 0
        self._pq_m:            int                        = 0
        self._n_threads:       int                        = os.cpu_count() or 4
        self._executor:        Optional[ThreadPoolExecutor] = None

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
        pq_m: int = 0,
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
        engine._pq_m       = pq_m

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
        use_pq = pq_m > 0
        index_type = f"HNSW+PQ(m={pq_m})" if use_pq else "HNSWFlat"
        logger.info(f"Building {n_clusters} local {index_type} indexes (M={M})...")
        t0 = time.time()
        for c in range(n_clusters):
            mask = (labels == c)
            gids = np.where(mask)[0].astype(np.int64)
            vecs = data_norm[gids]
            if len(gids) == 0:
                continue

            if use_pq and len(gids) >= 256:
                # IndexHNSWPQ: HNSW graph navigation + PQ compressed storage
                # pq_m subquantisers, 8 bits each → D*4 / pq_m compression ratio
                idx = faiss.IndexHNSWPQ(D, M, pq_m, 8)
                idx.hnsw.efConstruction = ef_construction
                idx.train(vecs)   # PQ codebook training
                idx.add(vecs)
            else:
                # Fall back to flat for small clusters or pq_m=0
                idx = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = ef_construction
                idx.add(vecs)

            engine._indexes[c] = idx
            engine._id_maps[c] = gids

        logger.info(f"HNSW indexes built in {time.time()-t0:.1f}s")

        # 6. Cross-boundary links
        # Identify vectors whose nearest neighbour is in a different cluster
        # and store direct links to their cross-cluster neighbours.
        # At search time these links expand candidates without extra probes.
        logger.info("Building cross-boundary links...")
        t0 = time.time()
        engine._boundary_links = _build_boundary_links(
            labels, knn_graph, M)
        n_boundary = len(engine._boundary_links)
        pct        = n_boundary / N * 100
        logger.info(f"Cross-boundary links: {n_boundary:,} boundary vectors "
                    f"({pct:.1f}% of corpus) in {time.time()-t0:.1f}s")

        # 7. Save
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

        # save boundary links as two arrays: keys and a ragged value store
        if self._boundary_links:
            bl_keys = np.array(list(self._boundary_links.keys()), dtype=np.int64)
            bl_vals = np.array(list(self._boundary_links.values()),
                               dtype=object)
            np.save(index_dir / "boundary_keys.npy", bl_keys)
            np.save(index_dir / "boundary_vals.npy", bl_vals,
                    allow_pickle=True)

        meta = {
            "n_clusters":        self._n_clusters,
            "dim":               self._dim,
            "n_vectors":         self._n_vectors,
            "M":                 M,
            "ef_construction":   ef_construction,
            "ef_search":         self._ef_search,
            "gap_policy":        self._gap_policy,
            "n_indexes":         len(self._indexes),
            "pq_m":              self._pq_m,
            "has_boundary_links": len(self._boundary_links) > 0,
            "n_boundary_vectors": len(self._boundary_links),
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
        engine._pq_m        = meta.get("pq_m", 0)

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

        # load boundary links if present
        bl_keys_path = index_dir / "boundary_keys.npy"
        bl_vals_path = index_dir / "boundary_vals.npy"
        if bl_keys_path.exists():
            bl_keys = np.load(bl_keys_path)
            bl_vals = np.load(bl_vals_path, allow_pickle=True)
            engine._boundary_links = {
                int(k): v for k, v in zip(bl_keys, bl_vals)
            }
            logger.info(f"Loaded {len(engine._boundary_links):,} boundary links")

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
        n_threads: Optional[int] = None,
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
        if n_threads is not None:
            self._n_threads = n_threads
        q = _cosine_normalize(query.reshape(1, -1).astype(np.float32))[0]
        return self._search_normalised(q, k=k, probe=probe)

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        probe: Optional[int] = None,
        n_threads: Optional[int] = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Search for k nearest neighbours for a batch of query vectors.

        Optimised implementation: groups queries by cluster and issues one
        batched FAISS call per cluster instead of one call per (query, cluster)
        pair. This reduces FAISS call overhead from Q*probe → n_clusters and
        allows FAISS to use SIMD/AVX2 efficiently on batched inputs.

        Benchmarked speedup: ~2.7x over sequential at probe=16 on SIFT1M,
        with identical recall.

        Parameters
        ----------
        queries   : 2-D float array [Q, D]. Raw or pre-normalised — both work.
        k         : number of results per query.
        probe     : fixed probe count. If None, gap-adaptive probe is used
                    (computed per query from centroid score gap).

        Returns
        -------
        List of Q result lists, each a list of (global_id, cosine_score)
        sorted by score descending.
        """
        if n_threads is not None:
            self._n_threads = n_threads

        qs = _cosine_normalize(queries.astype(np.float32))
        Q  = len(qs)

        # Step 1: route all queries to clusters in one matrix multiply [Q, C]
        all_scores = qs @ self._centers.T

        # Step 2: determine probe per query — fixed or gap-adaptive
        if probe is not None:
            n_probes = [probe] * Q
        else:
            n_probes = [self._probe_from_gap(all_scores[qi]) for qi in range(Q)]

        # Step 3: get top-probe clusters per query and group by cluster
        cluster_to_queries: dict[int, list[int]] = {}
        query_top_clusters = []
        for qi in range(Q):
            n_probe = n_probes[qi]
            top_c   = np.argpartition(-all_scores[qi],
                                      min(n_probe, len(all_scores[qi]) - 1))[:n_probe]
            query_top_clusters.append(top_c)
            for c in top_c.tolist():
                if c not in cluster_to_queries:
                    cluster_to_queries[c] = []
                cluster_to_queries[c].append(qi)

        # Step 4: one batched FAISS call per cluster
        all_candidates: list[list[np.ndarray]] = [[] for _ in range(Q)]

        for c, query_ids in cluster_to_queries.items():
            idx  = self._indexes.get(c)
            gids = self._id_maps.get(c)
            if idx is None:
                continue

            batch    = qs[query_ids]                    # [batch_size, D]
            idx.hnsw.efSearch = max(self._ef_search, k)
            actual_k = min(k, idx.ntotal)
            _, lids  = idx.search(batch, actual_k)      # single batched FAISS call

            for j, qi in enumerate(query_ids):
                valid = lids[j][lids[j] >= 0]
                if len(valid):
                    all_candidates[qi].append(gids[valid])

        # Step 5: rerank per query with exact cosine
        results = []
        for qi in range(Q):
            if not all_candidates[qi]:
                results.append([])
                continue

            cands  = np.concatenate(all_candidates[qi])
            scores = self._data_norm[cands] @ qs[qi]

            # boundary link expansion (≤200K datasets)
            if self._boundary_links and len(cands) > 0 and self._n_vectors <= 200_000:
                n_expand   = min(k, len(cands))
                top_expand = np.argpartition(-scores, n_expand)[:n_expand]
                extra_ids  = []
                for idx in top_expand:
                    links = self._boundary_links.get(int(cands[idx]))
                    if links is not None:
                        extra_ids.append(links)
                if extra_ids:
                    new_cands = np.concatenate(extra_ids)
                    new_cands = new_cands[~np.isin(new_cands, cands)]
                    if len(new_cands) > 0:
                        new_scores = self._data_norm[new_cands] @ qs[qi]
                        cands      = np.concatenate([cands, new_cands])
                        scores     = np.concatenate([scores, new_scores])

            if len(cands) > k:
                top_idx = np.argpartition(-scores, k)[:k]
            else:
                top_idx = np.arange(len(cands))

            top_idx = top_idx[np.argsort(-scores[top_idx])]
            results.append([(int(cands[i]), float(scores[i])) for i in top_idx])

        return results

    def set_threads(self, n: int) -> None:
        """Set number of threads for parallel cluster search."""
        self._n_threads = max(1, n)
        # Recreate persistent executor with new thread count
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        self._executor = ThreadPoolExecutor(max_workers=self._n_threads) \
            if self._n_threads > 1 else None
        logger.info(f'Thread count set to {self._n_threads}')

    def _search_one_cluster(
        self,
        c: int,
        q: np.ndarray,
        k: int,
    ) -> Optional[np.ndarray]:
        """Search a single cluster index. Thread-safe — FAISS HNSW read is safe."""
        idx  = self._indexes.get(c)
        gids = self._id_maps.get(c)
        if idx is None:
            return None
        idx.hnsw.efSearch = max(self._ef_search, k)
        actual_k = min(k, idx.ntotal)
        _, lids = idx.search(q.reshape(1, -1), actual_k)
        valid = lids[0][lids[0] >= 0]
        return gids[valid] if len(valid) else None

    def _search_normalised(
        self,
        q: np.ndarray,
        k: int,
        probe: Optional[int],
    ) -> list[tuple[int, float]]:
        scores       = self._centers @ q
        n_probe      = probe if probe is not None else self._probe_from_gap(scores)
        top_clusters = np.argpartition(-scores, min(n_probe, len(scores)-1))[:n_probe]
        top_clusters = [int(c) for c in top_clusters]

        candidate_ids = []

        if self._executor is not None and len(top_clusters) > 1:
            # Parallel cluster search using persistent thread pool
            futures = {
                self._executor.submit(self._search_one_cluster, c, q, k): c
                for c in top_clusters
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None and len(result):
                    candidate_ids.append(result)
        else:
            # Single-threaded fallback
            for c in top_clusters:
                result = self._search_one_cluster(c, q, k)
                if result is not None and len(result):
                    candidate_ids.append(result)

        if not candidate_ids:
            return []

        cands = np.concatenate(candidate_ids)
        if len(cands) == 0:
            return []

        # compute scores once — reuse for both expansion gate and reranking
        cand_scores = self._data_norm[cands] @ q

        # expand with cross-boundary links — always expand top-k candidates
        # k is scale-invariant: we always need exactly k true neighbours
        # expanding top-k scorers gives the best candidates their boundary links
        if self._boundary_links and len(cands) > 0 and self._n_vectors <= 200_000:
            n_expand   = min(k, len(cands))
            top_expand = np.argpartition(-cand_scores, n_expand)[:n_expand]
            extra_ids  = []
            for idx in top_expand:
                links = self._boundary_links.get(int(cands[idx]))
                if links is not None:
                    extra_ids.append(links)
            if extra_ids:
                new_cands = np.concatenate(extra_ids)
                new_cands = new_cands[~np.isin(new_cands, cands)]
                if len(new_cands) > 0:
                    new_scores  = self._data_norm[new_cands] @ q
                    cands       = np.concatenate([cands, new_cands])
                    cand_scores = np.concatenate([cand_scores, new_scores])

        # rerank by exact cosine (scores already computed above)
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
        Automatically selects probe counts appropriate for n_clusters.
        Call this when switching to a new dataset or embedding model.

        Parameters
        ----------
        sample_queries : representative query vectors [Q, D].
        probe_tiers    : list of (percentile, probe_count) pairs.
                         If None, auto-selected based on n_clusters.

        Returns
        -------
        New gap_policy list (also updates self._gap_policy in place).
        """
        if probe_tiers is None:
            # Auto-scale probe counts based on n_clusters
            # Rule: probe counts should range from ~2% to ~5% of n_clusters
            # to keep candidate coverage reasonable
            K = self._n_clusters
            if K <= 100:
                # 150K scale: probe 6-20 out of 100 clusters (6%-20%)
                probe_tiers = [(90, 6), (75, 8), (50, 12), (25, 16), (0, 20)]
            elif K <= 500:
                # mid scale
                probe_tiers = [(90, 8), (75, 12), (50, 20), (25, 32), (0, 48)]
            else:
                # 1M scale: probe 8-32 out of 1000 clusters (0.8%-3.2%)
                probe_tiers = [(90, 8), (75, 12), (50, 16), (25, 24), (0, 32)]

            logger.info(f"Auto-selected probe tiers for K={K}: {probe_tiers}")

        qs = _cosine_normalize(sample_queries.astype(np.float32))
        gaps = []
        for q in qs:
            scores = self._centers @ q
            top2   = np.partition(-scores, 1)[:2]
            gaps.append(float(-top2[0] + top2[1]))
        gaps = np.array(gaps)

        logger.info(f"Gap distribution: p10={np.percentile(gaps,10):.4f} "
                    f"p50={np.percentile(gaps,50):.4f} "
                    f"p90={np.percentile(gaps,90):.4f} "
                    f"mean={gaps.mean():.4f}")

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
                f"n_indexes={len(self._indexes)}, "
                f"boundary_vectors={len(self._boundary_links):,})")


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


def _build_boundary_links(
    labels: np.ndarray,
    knn_graph: np.ndarray,
    M: int,
) -> dict[int, np.ndarray]:
    """
    Identify boundary vectors and store cross-cluster links.

    A vector is a boundary vector only if its NEAREST neighbour
    (knn_graph[:,0]) lives in a different cluster. This is a tight
    definition that keeps boundary set small and high-quality.

    At search time these links expand the candidate pool for
    boundary queries without requiring extra cluster probes.

    Parameters
    ----------
    labels    : cluster label per vector [N]
    knn_graph : kNN neighbour indices [N, k], sorted by distance
    M         : max cross-boundary links per boundary vector

    Returns
    -------
    dict mapping global_id → np.ndarray of foreign neighbour global_ids
    """
    # Only vectors whose NEAREST neighbour is in a different cluster
    nearest_neighbour_labels = labels[knn_graph[:, 0]]
    is_boundary = nearest_neighbour_labels != labels

    boundary_ids = np.where(is_boundary)[0]
    links        = {}

    for gid in boundary_ids:
        nbrs    = knn_graph[gid]
        own_c   = labels[gid]
        foreign = nbrs[labels[nbrs] != own_c][:M]
        if len(foreign) > 0:
            links[int(gid)] = foreign.astype(np.int64)

    return links


def _run_igar(labels: np.ndarray, knn_graph: np.ndarray,
              n_clusters: int, n_iters: int, sample_size: int) -> np.ndarray:
    """
    IGAR — Iterative Graph-Aligned Reassignment.

    For each sampled vector, reassigns it to the plurality cluster
    among its kNN neighbours. Uses numpy bincount internally which
    is C-compiled and already optimal — vectorised approaches are
    slower due to large intermediate array memory costs.
    """
    labels = labels.astype(np.int32).copy()
    N, k   = knn_graph.shape

    def retention():
        return (labels[knn_graph] == labels[:, None]).sum() / (N * k)

    logger.info(f"IGAR iter 0: retention={retention():.4f}")

    for it in range(1, n_iters + 1):
        idx        = np.random.choice(N, size=min(sample_size, N), replace=False)
        # Pre-gather all neighbour labels in one numpy op (fast)
        nbr_labels = labels[knn_graph[idx]]   # [S, k]
        # bincount per row — C-compiled, already optimal
        for j, i in enumerate(idx):
            labels[i] = np.bincount(
                nbr_labels[j], minlength=n_clusters).argmax()
        if it % 10 == 0:
            logger.info(f"IGAR iter {it}: retention={retention():.4f}")

    logger.info(f"IGAR done: final retention={retention():.4f}")
    return labels
