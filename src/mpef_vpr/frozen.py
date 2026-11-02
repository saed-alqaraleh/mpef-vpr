"""Load the frozen, validation-selected backend and separately frozen calibrator.

Ranking uses the *uncalibrated* MPEF decision score and stable tie rules.
Calibrated candidate relevance is a separate output. Do not rank by calibrated
probability, since an isotonic map may create new ties.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .model import IsotonicMap, MPEFModel
from .features import FEATURE_SPECS
from .ranking import rank_candidates

MODEL_SHA256 = {
    'disk': 'cbd869f2a7fb2d296faab99ed202593e564446fa674fdc3df15ef924939ca161',
    'aliked': '43352cdf2d98b4e0c3b0f3babdd0901ba5011bb89a629a0fd56d6039a3b8a983',
}
CALIBRATOR_SHA256 = {
    'disk': 'efd681a2e5cfdb0bdb08aa3b7d0e32f1c8fd586117cd2ef6c90d2a5c4645217c',
    'aliked': '8fab2bc754ae92a6c27b1ac21fb75bfecd7f2af74979d6f8eba67aed4e8a4b21',
}

def _verified_json(path: Path, expected_sha256: str) -> dict:
    blob = path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'Frozen artifact hash mismatch for {path.name}: {actual}')
    return json.loads(blob)

def load_frozen_model(model_dir: str | Path, backend: str) -> MPEFModel:
    """Load only the canonical frozen backend artifacts, failing on altered bytes."""
    if backend not in MODEL_SHA256:
        raise ValueError(f'Unknown backend: {backend}')
    folder = Path(model_dir)
    record = _verified_json(folder / f'{backend}_model.json', MODEL_SHA256[backend])
    model = MPEFModel.from_dict(record)
    if [(s.name, s.direction) for s in model.feature_specs] != [(s.name, s.direction) for s in FEATURE_SPECS]:
        raise ValueError('Frozen model feature contract differs from the documented 21 features')
    cal = _verified_json(folder / f'{backend}_calibrator.json', CALIBRATOR_SHA256[backend])
    if cal['source_model_sha256'] != MODEL_SHA256[backend] or cal['method'] != 'mpef':
        raise ValueError('Calibrator is not linked to this frozen model')
    map_ = IsotonicMap(np.asarray(cal['x_thresholds'], dtype=float), np.asarray(cal['y_thresholds'], dtype=float))
    if len(map_.x) != cal['knot_count'] or len(map_.x) != len(map_.y):
        raise ValueError('Invalid frozen isotonic knot count')
    if np.any(np.diff(map_.x) < 0) or np.any(np.diff(map_.y) < -1e-12):
        raise ValueError('Calibration thresholds not nondecreasing')
    model.calibrator = map_
    return model

def rank_frozen_candidates(model: MPEFModel, features: np.ndarray,
                           global_ranks: np.ndarray, candidate_ids: np.ndarray):
    """Return Q x K ranked indices and Q x K calibrated probabilities."""
    rank = np.asarray(global_ranks)
    if rank.ndim != 2 or features.shape[0] != rank.size:
        raise ValueError('features must be flattened Q*K and global ranks Q*K')
    score = model.decision_function(features).reshape(rank.shape)
    order = rank_candidates(score, rank, candidate_ids)
    prob = model.predict_proba(features).reshape(rank.shape)
    return order, prob
