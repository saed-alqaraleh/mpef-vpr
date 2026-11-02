"""Sparse shape-constrained additive probability model for MPEF."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from .features import FEATURE_SPECS
from .math_utils import safe_logit, sigmoid
from .types import FeatureSpec


@dataclass
class IsotonicMap:
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "IsotonicMap":
        iso = IsotonicRegression(
            increasing=True,
            y_min=1e-6,
            y_max=1.0 - 1e-6,
            out_of_bounds="clip",
        )
        iso.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                sample_weight=np.asarray(sample_weight, dtype=float))
        return cls(np.asarray(iso.X_thresholds_, dtype=float),
                   np.asarray(iso.y_thresholds_, dtype=float))

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if len(self.x) == 1:
            return np.full_like(values, self.y[0], dtype=float)
        return np.interp(values, self.x, self.y, left=self.y[0], right=self.y[-1])

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> "IsotonicMap":
        return cls(np.asarray(value["x"], dtype=float), np.asarray(value["y"], dtype=float))


def query_equal_weights(groups: np.ndarray) -> np.ndarray:
    """Give every physical query equal total loss weight."""

    group = np.asarray(groups)
    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w / w.sum()


def weighted_log_loss(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray) -> float:
    label = np.asarray(y, dtype=float)
    prob = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    w = np.asarray(sample_weight, dtype=float)
    return float(np.sum(w * (-(label * np.log(prob) + (1.0 - label) * np.log1p(-prob)))) / w.sum())


class MPEFModel:
    """Monotone additive evidence model with a held-out calibration map.

    Each raw feature is oriented so larger values support relevance, fitted with
    an isotonic likelihood map on development data, and converted to centered
    log-odds evidence. Non-negative additive coefficients preserve every declared
    monotonic direction. L1 and L2 penalties control sparsity and stability.
    """

    def __init__(self, feature_specs: Sequence[FeatureSpec] = FEATURE_SPECS,
                 l1: float = 1e-3, l2: float = 1e-2):
        self.feature_specs = tuple(feature_specs)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.maps: list[IsotonicMap] = []
        self.centers: np.ndarray | None = None
        self.intercept: float | None = None
        self.coefficients: np.ndarray | None = None
        self.calibrator: IsotonicMap | None = None
        self.optimization: dict | None = None

    @property
    def fitted(self) -> bool:
        return self.intercept is not None and self.coefficients is not None and bool(self.maps)

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_specs):
            raise ValueError(f"X must have shape (N,{len(self.feature_specs)})")
        if not np.isfinite(arr).all():
            raise ValueError("X contains non-finite values")
        return arr

    def _orient(self, x: np.ndarray, index: int) -> np.ndarray:
        return x[:, index] if self.feature_specs[index].direction == "increasing" else -x[:, index]

    def _fit_maps(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        z = np.empty_like(x, dtype=float)
        self.maps = []
        centers = []
        for j in range(x.shape[1]):
            mapping = IsotonicMap.fit(self._orient(x, j), y, w)
            evidence = safe_logit(mapping.predict(self._orient(x, j)))
            center = float(np.average(evidence, weights=w))
            self.maps.append(mapping)
            centers.append(center)
            z[:, j] = evidence - center
        self.centers = np.asarray(centers, dtype=float)
        return z

    def _transform(self, x: np.ndarray) -> np.ndarray:
        arr = self._check_x(x)
        if not self.fitted or self.centers is None:
            raise RuntimeError("fit the model before prediction")
        z = np.empty_like(arr, dtype=float)
        for j, mapping in enumerate(self.maps):
            evidence = safe_logit(mapping.predict(self._orient(arr, j)))
            z[:, j] = evidence - self.centers[j]
        return z

    def fit(self, x: np.ndarray, y: np.ndarray, query_groups: np.ndarray) -> "MPEFModel":
        arr = self._check_x(x)
        label = np.asarray(y, dtype=float).reshape(-1)
        groups = np.asarray(query_groups).reshape(-1)
        if len(arr) != len(label) or len(arr) != len(groups):
            raise ValueError("X, y, and query_groups must have the same row count")
        if not set(np.unique(label)).issubset({0.0, 1.0}):
            raise ValueError("labels must be binary")
        if label.sum() == 0 or label.sum() == len(label):
            raise ValueError("development data must contain both classes")

        w = query_equal_weights(groups)
        z = self._fit_maps(arr, label, w)
        prevalence = float(np.average(label, weights=w))
        initial = np.r_[safe_logit(np.array([prevalence]))[0], np.zeros(z.shape[1])]

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = theta[0]
            coef = theta[1:]
            eta = intercept + z @ coef
            prob = sigmoid(eta)
            loss = weighted_log_loss(label, prob, w)
            penalty = self.l1 * coef.sum() + 0.5 * self.l2 * np.dot(coef, coef)
            error = prob - label
            grad_intercept = float(np.sum(w * error) / w.sum())
            grad_coef = (z.T @ (w * error)) / w.sum() + self.l1 + self.l2 * coef
            return loss + penalty, np.r_[grad_intercept, grad_coef]

        result = minimize(
            fun=lambda t: objective(t)[0],
            x0=initial,
            jac=lambda t: objective(t)[1],
            method="L-BFGS-B",
            bounds=[(None, None)] + [(0.0, None)] * z.shape[1],
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"MPEF optimization failed: {result.message}")
        self.intercept = float(result.x[0])
        self.coefficients = np.asarray(result.x[1:], dtype=float)
        self.calibrator = None
        self.optimization = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "development_rows": int(len(arr)),
            "development_queries": int(len(np.unique(groups))),
        }
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        z = self._transform(x)
        assert self.intercept is not None and self.coefficients is not None
        return self.intercept + z @ self.coefficients

    def predict_uncalibrated(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(x))

    def fit_calibrator(self, x: np.ndarray, y: np.ndarray,
                       query_groups: np.ndarray) -> "MPEFModel":
        if not self.fitted:
            raise RuntimeError("fit development model before calibration")
        label = np.asarray(y, dtype=float).reshape(-1)
        groups = np.asarray(query_groups).reshape(-1)
        raw = self.predict_uncalibrated(x)
        if len(raw) != len(label) or len(raw) != len(groups):
            raise ValueError("calibration X, y, and query_groups must align")
        self.calibrator = IsotonicMap.fit(raw, label, query_equal_weights(groups))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = self.predict_uncalibrated(x)
        return raw if self.calibrator is None else self.calibrator.predict(raw)

    def feature_contributions(self, x: np.ndarray) -> np.ndarray:
        z = self._transform(x)
        assert self.coefficients is not None
        return z * self.coefficients

    def evaluate_log_loss(self, x: np.ndarray, y: np.ndarray,
                          query_groups: np.ndarray, calibrated: bool = True) -> float:
        prob = self.predict_proba(x) if calibrated else self.predict_uncalibrated(x)
        return weighted_log_loss(y, prob, query_equal_weights(query_groups))

    @classmethod
    def select_on_validation(
        cls,
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        groups_dev: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        groups_val: np.ndarray,
        feature_specs: Sequence[FeatureSpec] = FEATURE_SPECS,
        l1_grid: Iterable[float] = (0.0, 1e-4, 1e-3),
        l2_grid: Iterable[float] = (1e-3, 1e-2, 1e-1),
    ) -> tuple["MPEFModel", list[dict]]:
        """Select regularization on validation data without refitting on it."""

        records: list[dict] = []
        best_model: MPEFModel | None = None
        best_key: tuple[float, float, float] | None = None
        weights_val = query_equal_weights(groups_val)
        for l1 in l1_grid:
            for l2 in l2_grid:
                model = cls(feature_specs=feature_specs, l1=float(l1), l2=float(l2))
                model.fit(x_dev, y_dev, groups_dev)
                probability = model.predict_uncalibrated(x_val)
                loss = weighted_log_loss(y_val, probability, weights_val)
                active = int(np.sum(model.coefficients > 1e-10))
                record = {"l1": float(l1), "l2": float(l2),
                          "validation_log_loss": loss, "active_features": active}
                records.append(record)
                key = (loss, float(l1), float(l2))
                if best_key is None or key < best_key:
                    best_key, best_model = key, model
        assert best_model is not None
        return best_model, records

    def to_dict(self) -> dict:
        if not self.fitted or self.centers is None:
            raise RuntimeError("cannot serialize an unfitted model")
        assert self.coefficients is not None and self.intercept is not None
        return {
            "format": "mpef_model_v1",
            "feature_specs": [asdict(s) for s in self.feature_specs],
            "l1": self.l1,
            "l2": self.l2,
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "centers": self.centers.tolist(),
            "maps": [m.to_dict() for m in self.maps],
            "calibrator": None if self.calibrator is None else self.calibrator.to_dict(),
            "optimization": self.optimization,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "MPEFModel":
        if value.get("format") != "mpef_model_v1":
            raise ValueError("unsupported MPEF model format")
        specs = tuple(FeatureSpec(**item) for item in value["feature_specs"])
        model = cls(specs, l1=value["l1"], l2=value["l2"])
        model.intercept = float(value["intercept"])
        model.coefficients = np.asarray(value["coefficients"], dtype=float)
        model.centers = np.asarray(value["centers"], dtype=float)
        model.maps = [IsotonicMap.from_dict(item) for item in value["maps"]]
        model.calibrator = None if value.get("calibrator") is None else IsotonicMap.from_dict(value["calibrator"])
        model.optimization = value.get("optimization")
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "MPEFModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

