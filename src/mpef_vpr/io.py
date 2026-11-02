"""Evidence-file loading with an explicit, versioned contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import CandidateBatch


REQUIRED_ARRAYS = (
    "query_ids",
    "candidate_ids",
    "global_similarity",
    "global_rank",
    "inlier_count",
    "match_count",
    "mean_match_confidence",
    "ransac_score",
    "query_inlier_xy",
    "candidate_inlier_xy",
    "inlier_valid",
    "reprojection_error",
    "query_width",
    "query_height",
    "candidate_width",
    "candidate_height",
    "degenerate_geometry",
)


def load_candidate_batch(path: str | Path, expected_k: int | None = 100,
                         require_labels: bool = False) -> CandidateBatch:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive]
        if missing:
            raise ValueError(f"{source} is missing required arrays: {missing}")
        labels = archive["labels"] if "labels" in archive else None
        if require_labels and labels is None:
            raise ValueError(f"{source} has no labels array")
        batch = CandidateBatch(
            **{name: archive[name] for name in REQUIRED_ARRAYS},
            labels=labels,
        )
    batch.validate(expected_k=expected_k)
    return batch


def subset_queries(batch: CandidateBatch, indices: np.ndarray) -> CandidateBatch:
    idx = np.asarray(indices, dtype=int)
    fields = {
        name: np.asarray(getattr(batch, name))[idx]
        for name in batch.__dataclass_fields__
        if name != "labels"
    }
    labels = None if batch.labels is None else np.asarray(batch.labels)[idx]
    return CandidateBatch(**fields, labels=labels)

