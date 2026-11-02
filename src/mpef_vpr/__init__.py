"""Monotone Probabilistic Evidence Fusion for visual place recognition."""

from .features import FEATURE_SPECS, build_feature_matrix
from .model import MPEFModel
from .ranking import rank_candidates
from .types import CandidateBatch, FeatureSpec

__all__ = [
    "CandidateBatch",
    "FEATURE_SPECS",
    "FeatureSpec",
    "MPEFModel",
    "build_feature_matrix",
    "rank_candidates",
]

__version__ = "0.1.0"

