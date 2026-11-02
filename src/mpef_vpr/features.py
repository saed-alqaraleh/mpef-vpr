"""Directional evidence features for matching-aware visual place recognition.

Features combine global retrieval, local matching, spatial geometry, and
agreement over the fixed candidate shortlist.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import kendalltau

from .math_utils import normalized_entropy_from_scores, stable_percentile
from .types import CandidateBatch, FeatureSpec


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("global_similarity_percentile", "global", "increasing",
                "Empirical percentile of retriever similarity within the fixed shortlist."),
    FeatureSpec("global_rank_percentile", "global", "increasing",
                "One-based global rank converted so one is best and zero is worst."),
    FeatureSpec("global_query_margin", "global", "increasing",
                "Top-1 versus top-2 global-similarity gap, robustly normalized per query."),
    FeatureSpec("global_query_entropy", "global", "decreasing",
                "Normalized entropy of the robustly scaled global-score distribution."),
    FeatureSpec("log1p_inliers", "local", "increasing",
                "Natural log of one plus the robust-estimator inlier count."),
    FeatureSpec("inlier_ratio", "local", "increasing",
                "Inlier count divided by putative match count."),
    FeatureSpec("mean_match_confidence", "local", "increasing",
                "Mean confidence of putative local correspondences."),
    FeatureSpec("ransac_score", "local", "increasing",
                "Backend-native robust-estimation score, oriented so larger is better."),
    FeatureSpec("local_rank_percentile", "local", "increasing",
                "Percentile of a coefficient-free lexicographic local-evidence order."),
    FeatureSpec("query_hull_coverage", "spatial", "increasing",
                "Convex-hull area of query inliers divided by query-image area."),
    FeatureSpec("candidate_hull_coverage", "spatial", "increasing",
                "Convex-hull area of candidate inliers divided by candidate-image area."),
    FeatureSpec("joint_hull_coverage", "spatial", "increasing",
                "Geometric mean of query and candidate convex-hull coverage."),
    FeatureSpec("query_spatial_entropy", "spatial", "increasing",
                "Normalized Shannon entropy of query inliers over a fixed analysis grid."),
    FeatureSpec("candidate_spatial_entropy", "spatial", "increasing",
                "Normalized Shannon entropy of candidate inliers over a fixed analysis grid."),
    FeatureSpec("median_reprojection_error", "spatial", "decreasing",
                "Median valid-inlier reprojection residual in pixels."),
    FeatureSpec("p90_reprojection_error", "spatial", "decreasing",
                "90th percentile valid-inlier reprojection residual in pixels."),
    FeatureSpec("geometry_collinearity", "spatial", "decreasing",
                "Worst query/candidate inlier collinearity from normalized covariance eigenvalues."),
    FeatureSpec("degenerate_geometry", "spatial", "decreasing",
                "One when the robust estimator reports a degenerate or invalid model."),
    FeatureSpec("joint_rank_support", "agreement", "increasing",
                "Minimum of global and local rank percentiles for the candidate."),
    FeatureSpec("percentile_disagreement", "agreement", "decreasing",
                "Absolute global-versus-local percentile difference."),
    FeatureSpec("list_kendall_tau", "agreement", "increasing",
                "Query-level Kendall rank concordance between global and local orders."),
)


def _hull_coverage(xy: np.ndarray, width: float, height: float) -> float:
    points = np.asarray(xy, dtype=float)
    if len(points) < 3 or width <= 0 or height <= 0:
        return 0.0
    normalized = points / np.array([width, height], dtype=float)
    normalized = np.clip(normalized, 0.0, 1.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            area = float(ConvexHull(normalized).volume)
    except (QhullError, ValueError):
        return 0.0
    return float(np.clip(area, 0.0, 1.0))


def _spatial_entropy(xy: np.ndarray, width: float, height: float,
                     grid: tuple[int, int]) -> float:
    points = np.asarray(xy, dtype=float)
    gx, gy = grid
    if len(points) == 0 or width <= 0 or height <= 0:
        return 0.0
    x = np.clip((points[:, 0] / width * gx).astype(int), 0, gx - 1)
    y = np.clip((points[:, 1] / height * gy).astype(int), 0, gy - 1)
    counts = np.bincount(y * gx + x, minlength=gx * gy).astype(float)
    p = counts[counts > 0] / counts.sum()
    if len(p) <= 1:
        return 0.0
    entropy = -(p * np.log(p)).sum()
    return float(entropy / np.log(min(len(points), gx * gy)))


def _collinearity(xy: np.ndarray, width: float, height: float) -> float:
    points = np.asarray(xy, dtype=float)
    if len(points) < 3 or width <= 0 or height <= 0:
        return 1.0
    normalized = points / np.array([width, height], dtype=float)
    cov = np.cov(normalized, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    if not np.isfinite(eig).all() or eig[-1] <= 1e-15:
        return 1.0
    isotropy = float(np.clip(eig[0] / eig[-1], 0.0, 1.0))
    return 1.0 - isotropy


def _local_rank_percentile(inliers: np.ndarray, ratio: np.ndarray,
                           confidence: np.ndarray, global_rank: np.ndarray) -> np.ndarray:
    q, k = inliers.shape
    out = np.empty((q, k), dtype=float)
    for i in range(q):
        order = np.lexsort((
            np.arange(k),
            global_rank[i],
            -confidence[i],
            -ratio[i],
            -inliers[i],
        ))
        rank = np.empty(k, dtype=int)
        rank[order] = np.arange(k)
        out[i] = 1.0 if k == 1 else 1.0 - rank / (k - 1.0)
    return out


def _query_margin(similarity: np.ndarray) -> np.ndarray:
    q, k = similarity.shape
    if k < 2:
        return np.ones(q, dtype=float)
    sorted_scores = np.sort(similarity, axis=1)[:, ::-1]
    gap = sorted_scores[:, 0] - sorted_scores[:, 1]
    span = sorted_scores[:, 0] - sorted_scores[:, -1]
    return np.divide(gap, np.maximum(span, 1e-12), out=np.zeros(q), where=span > 1e-12)


def build_feature_matrix(batch: CandidateBatch, expected_k: int | None = 100,
                         entropy_grid: tuple[int, int] = (4, 4)) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert raw fixed-K evidence into a deterministic ``(Q*K,F)`` matrix."""

    q, k, _ = batch.validate(expected_k=expected_k)
    similarity = np.asarray(batch.global_similarity, dtype=float)
    global_rank = np.asarray(batch.global_rank, dtype=float)
    inliers = np.asarray(batch.inlier_count, dtype=float)
    matches = np.asarray(batch.match_count, dtype=float)
    confidence = np.clip(np.asarray(batch.mean_match_confidence, dtype=float), 0.0, 1.0)
    ransac_score = np.asarray(batch.ransac_score, dtype=float)
    ratio = np.divide(inliers, np.maximum(matches, 1.0))

    global_pct = stable_percentile(similarity, higher_is_better=True)
    global_rank_pct = 1.0 - (global_rank - 1.0) / max(k - 1.0, 1.0)
    global_rank_pct = np.clip(global_rank_pct, 0.0, 1.0)
    local_pct = _local_rank_percentile(inliers, ratio, confidence, global_rank)

    query_margin = np.repeat(_query_margin(similarity)[:, None], k, axis=1)
    query_entropy = np.repeat(normalized_entropy_from_scores(similarity)[:, None], k, axis=1)
    list_tau = np.empty(q, dtype=float)
    for i in range(q):
        tau = kendalltau(global_rank_pct[i], local_pct[i], nan_policy="omit").statistic
        list_tau[i] = 0.0 if not np.isfinite(tau) else float(tau)
    list_tau = np.repeat(list_tau[:, None], k, axis=1)

    q_hull = np.zeros((q, k), dtype=float)
    c_hull = np.zeros((q, k), dtype=float)
    q_entropy = np.zeros((q, k), dtype=float)
    c_entropy = np.zeros((q, k), dtype=float)
    median_residual = np.zeros((q, k), dtype=float)
    p90_residual = np.zeros((q, k), dtype=float)
    collinearity = np.ones((q, k), dtype=float)

    qwidth = np.asarray(batch.query_width, dtype=float).reshape(q, -1)[:, 0]
    qheight = np.asarray(batch.query_height, dtype=float).reshape(q, -1)[:, 0]
    cwidth = np.asarray(batch.candidate_width, dtype=float)
    cheight = np.asarray(batch.candidate_height, dtype=float)
    valid = np.asarray(batch.inlier_valid, dtype=bool)
    residuals = np.asarray(batch.reprojection_error, dtype=float)
    qxy_all = np.asarray(batch.query_inlier_xy, dtype=float)
    cxy_all = np.asarray(batch.candidate_inlier_xy, dtype=float)

    for i in range(q):
        for j in range(k):
            mask = valid[i, j]
            qxy = qxy_all[i, j, mask]
            cxy = cxy_all[i, j, mask]
            q_hull[i, j] = _hull_coverage(qxy, qwidth[i], qheight[i])
            c_hull[i, j] = _hull_coverage(cxy, cwidth[i, j], cheight[i, j])
            q_entropy[i, j] = _spatial_entropy(qxy, qwidth[i], qheight[i], entropy_grid)
            c_entropy[i, j] = _spatial_entropy(cxy, cwidth[i, j], cheight[i, j], entropy_grid)
            collinearity[i, j] = max(
                _collinearity(qxy, qwidth[i], qheight[i]),
                _collinearity(cxy, cwidth[i, j], cheight[i, j]),
            )
            r = residuals[i, j, mask]
            r = r[np.isfinite(r)]
            if len(r):
                median_residual[i, j] = float(np.median(r))
                p90_residual[i, j] = float(np.percentile(r, 90))
            else:
                median_residual[i, j] = 1e6
                p90_residual[i, j] = 1e6

    features = {
        "global_similarity_percentile": global_pct,
        "global_rank_percentile": global_rank_pct,
        "global_query_margin": query_margin,
        "global_query_entropy": query_entropy,
        "log1p_inliers": np.log1p(inliers),
        "inlier_ratio": ratio,
        "mean_match_confidence": confidence,
        "ransac_score": ransac_score,
        "local_rank_percentile": local_pct,
        "query_hull_coverage": q_hull,
        "candidate_hull_coverage": c_hull,
        "joint_hull_coverage": np.sqrt(q_hull * c_hull),
        "query_spatial_entropy": q_entropy,
        "candidate_spatial_entropy": c_entropy,
        "median_reprojection_error": median_residual,
        "p90_reprojection_error": p90_residual,
        "geometry_collinearity": collinearity,
        "degenerate_geometry": np.asarray(batch.degenerate_geometry, dtype=float),
        "joint_rank_support": np.minimum(global_rank_pct, local_pct),
        "percentile_disagreement": np.abs(global_rank_pct - local_pct),
        "list_kendall_tau": list_tau,
    }
    names = tuple(spec.name for spec in FEATURE_SPECS)
    x = np.column_stack([features[name].reshape(-1) for name in names])
    if not np.isfinite(x).all():
        raise ValueError("derived feature matrix contains non-finite values")
    return x.astype(np.float64), names

