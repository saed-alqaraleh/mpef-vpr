"""Deterministic synthetic evidence for software tests and profiling only."""

from __future__ import annotations

import numpy as np

from .types import CandidateBatch


def make_synthetic_batch(queries: int = 200, candidates: int = 100,
                         max_inliers: int = 16, seed: int = 20260824) -> CandidateBatch:
    rng = np.random.default_rng(seed)
    q, k, p = int(queries), int(candidates), int(max_inliers)
    labels = np.zeros((q, k), dtype=np.int8)
    positive_index = rng.integers(0, min(12, k), size=q)
    labels[np.arange(q), positive_index] = 1

    latent = rng.normal(0.0, 0.7, (q, k)) + labels * 2.2
    similarity = latent + rng.normal(0.0, 0.5, (q, k))
    order = np.argsort(-similarity, axis=1, kind="stable")
    global_rank = np.empty((q, k), dtype=np.int32)
    for i in range(q):
        global_rank[i, order[i]] = np.arange(1, k + 1)

    match_count = rng.integers(24, 96, (q, k), dtype=np.int32)
    inlier_probability = sigmoid_for_synthetic(-1.2 + 1.6 * labels + 0.35 * latent)
    inlier_count = rng.binomial(match_count, np.clip(inlier_probability, 0.01, 0.9)).astype(np.int32)
    confidence = np.clip(rng.normal(0.35 + 0.35 * labels + 0.08 * latent, 0.12), 0.0, 1.0)
    ransac_score = np.clip(rng.normal(0.25 + 0.5 * labels + 0.05 * latent, 0.15), 0.0, 1.0)
    degenerate = (rng.random((q, k)) < (0.12 - 0.09 * labels)).astype(np.int8)

    query_xy = np.zeros((q, k, p, 2), dtype=np.float32)
    candidate_xy = np.zeros_like(query_xy)
    valid = np.zeros((q, k, p), dtype=bool)
    residual = np.full((q, k, p), np.nan, dtype=np.float32)
    for i in range(q):
        for j in range(k):
            n = min(int(inlier_count[i, j]), p)
            if n == 0:
                continue
            valid[i, j, :n] = True
            if labels[i, j]:
                qpts = rng.uniform([20, 20], [620, 460], size=(n, 2))
                cpts = qpts + rng.normal(0.0, 5.0, size=(n, 2))
                r = np.abs(rng.normal(1.0, 0.5, n))
            else:
                base = rng.uniform([100, 100], [540, 380])
                direction = rng.normal(size=2)
                direction /= max(np.linalg.norm(direction), 1e-12)
                t = rng.normal(0.0, 35.0, size=(n, 1))
                qpts = base + t * direction + rng.normal(0.0, 2.0, size=(n, 2))
                cpts = rng.uniform([60, 60], [580, 420], size=(n, 2))
                r = np.abs(rng.normal(7.0, 3.0, n))
            query_xy[i, j, :n] = np.clip(qpts, [0, 0], [639, 479])
            candidate_xy[i, j, :n] = np.clip(cpts, [0, 0], [639, 479])
            residual[i, j, :n] = r

    candidate_ids = np.array([[f"db_{j:05d}" for j in range(k)] for _ in range(q)])
    batch = CandidateBatch(
        query_ids=np.array([f"q_{i:05d}" for i in range(q)]),
        candidate_ids=candidate_ids,
        global_similarity=similarity.astype(np.float32),
        global_rank=global_rank,
        inlier_count=inlier_count,
        match_count=match_count,
        mean_match_confidence=confidence.astype(np.float32),
        ransac_score=ransac_score.astype(np.float32),
        query_inlier_xy=query_xy,
        candidate_inlier_xy=candidate_xy,
        inlier_valid=valid,
        reprojection_error=residual,
        query_width=np.full(q, 640, dtype=np.int32),
        query_height=np.full(q, 480, dtype=np.int32),
        candidate_width=np.full((q, k), 640, dtype=np.int32),
        candidate_height=np.full((q, k), 480, dtype=np.int32),
        degenerate_geometry=degenerate,
        labels=labels,
    )
    batch.validate(expected_k=k)
    return batch


def sigmoid_for_synthetic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))

