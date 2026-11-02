"""Typed contracts for fixed-candidate VPR evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Direction = Literal["increasing", "decreasing"]
FeatureBlock = Literal["global", "local", "spatial", "agreement"]


@dataclass(frozen=True)
class FeatureSpec:
    """Scientific definition of one monotone model input."""

    name: str
    block: FeatureBlock
    direction: Direction
    description: str


@dataclass
class CandidateBatch:
    """Raw evidence for a dense fixed-K candidate batch.

    Arrays with shape ``(Q, K)`` describe candidate-level measurements. Arrays
    with shape ``(Q, K, P, 2)`` store padded inlier coordinates; ``inlier_valid``
    marks valid points. Reprojection residuals are padded to ``(Q, K, P)``.

    ``global_similarity`` must be oriented so larger is better. Distances must be
    negated by the retriever adapter before this contract is constructed.
    """

    query_ids: np.ndarray
    candidate_ids: np.ndarray
    global_similarity: np.ndarray
    global_rank: np.ndarray
    inlier_count: np.ndarray
    match_count: np.ndarray
    mean_match_confidence: np.ndarray
    ransac_score: np.ndarray
    query_inlier_xy: np.ndarray
    candidate_inlier_xy: np.ndarray
    inlier_valid: np.ndarray
    reprojection_error: np.ndarray
    query_width: np.ndarray
    query_height: np.ndarray
    candidate_width: np.ndarray
    candidate_height: np.ndarray
    degenerate_geometry: np.ndarray
    labels: np.ndarray | None = None

    def validate(self, expected_k: int | None = None) -> tuple[int, int, int]:
        q, k = np.asarray(self.global_similarity).shape
        if expected_k is not None and k != expected_k:
            raise ValueError(f"fixed-candidate contract requires K={expected_k}, got {k}")
        if np.asarray(self.query_ids).shape != (q,):
            raise ValueError("query_ids must have shape (Q,)")
        if np.asarray(self.candidate_ids).shape != (q, k):
            raise ValueError("candidate_ids must have shape (Q,K)")

        matrices = {
            "global_rank": self.global_rank,
            "inlier_count": self.inlier_count,
            "match_count": self.match_count,
            "mean_match_confidence": self.mean_match_confidence,
            "ransac_score": self.ransac_score,
            "candidate_width": self.candidate_width,
            "candidate_height": self.candidate_height,
            "degenerate_geometry": self.degenerate_geometry,
        }
        for name, value in matrices.items():
            if np.asarray(value).shape != (q, k):
                raise ValueError(f"{name} must have shape (Q,K)")

        if np.asarray(self.query_width).shape not in {(q,), (q, 1)}:
            raise ValueError("query_width must have shape (Q,) or (Q,1)")
        if np.asarray(self.query_height).shape not in {(q,), (q, 1)}:
            raise ValueError("query_height must have shape (Q,) or (Q,1)")

        qxy = np.asarray(self.query_inlier_xy)
        cxy = np.asarray(self.candidate_inlier_xy)
        valid = np.asarray(self.inlier_valid)
        residual = np.asarray(self.reprojection_error)
        if qxy.ndim != 4 or qxy.shape[:2] != (q, k) or qxy.shape[-1] != 2:
            raise ValueError("query_inlier_xy must have shape (Q,K,P,2)")
        if cxy.shape != qxy.shape:
            raise ValueError("candidate_inlier_xy must match query_inlier_xy")
        p = qxy.shape[2]
        if valid.shape != (q, k, p):
            raise ValueError("inlier_valid must have shape (Q,K,P)")
        if residual.shape != (q, k, p):
            raise ValueError("reprojection_error must have shape (Q,K,P)")
        if self.labels is not None and np.asarray(self.labels).shape != (q, k):
            raise ValueError("labels must have shape (Q,K)")

        finite = [
            self.global_similarity,
            self.inlier_count,
            self.match_count,
            self.mean_match_confidence,
            self.ransac_score,
            self.query_width,
            self.query_height,
            self.candidate_width,
            self.candidate_height,
        ]
        if not all(np.isfinite(np.asarray(x, dtype=float)).all() for x in finite):
            raise ValueError("candidate evidence contains non-finite required values")
        if np.any(np.asarray(self.inlier_count) < 0) or np.any(np.asarray(self.match_count) < 0):
            raise ValueError("match and inlier counts must be non-negative")
        if np.any(np.asarray(self.inlier_count) > np.asarray(self.match_count)):
            raise ValueError("inlier_count cannot exceed match_count")
        if np.any(np.asarray(self.global_rank) < 1):
            raise ValueError("global_rank is one-based and must be positive")
        if np.any(np.asarray(self.query_width) <= 0) or np.any(np.asarray(self.query_height) <= 0):
            raise ValueError("query dimensions must be positive")
        if np.any(np.asarray(self.candidate_width) <= 0) or np.any(np.asarray(self.candidate_height) <= 0):
            raise ValueError("candidate dimensions must be positive")
        return q, k, p

    @property
    def query_groups(self) -> np.ndarray:
        q, k = np.asarray(self.global_similarity).shape
        return np.repeat(np.asarray(self.query_ids), k)

    @property
    def flat_labels(self) -> np.ndarray:
        if self.labels is None:
            raise ValueError("labels are required for fitting or evaluation")
        return np.asarray(self.labels, dtype=np.int8).reshape(-1)

