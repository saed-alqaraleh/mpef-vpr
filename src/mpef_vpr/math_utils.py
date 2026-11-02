"""Small deterministic numerical utilities."""

from __future__ import annotations

import numpy as np


def stable_percentile(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Candidate percentile in ``[0,1]`` with stable index tie breaking.

    The best candidate receives 1 and the worst receives 0. Ties preserve the
    incoming candidate order, which must be the global retrieval order.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must have shape (Q,K)")
    q, k = x.shape
    if k == 1:
        return np.ones_like(x)
    out = np.empty_like(x)
    index = np.arange(k)
    for i in range(q):
        key = -x[i] if higher_is_better else x[i]
        order = np.lexsort((index, key))
        ranks = np.empty(k, dtype=int)
        ranks[order] = np.arange(k)
        out[i] = 1.0 - ranks / (k - 1.0)
    return out


def normalized_entropy_from_scores(scores: np.ndarray) -> np.ndarray:
    """Normalized softmax entropy per query using an adaptive robust scale."""

    x = np.asarray(scores, dtype=float)
    if x.ndim != 2:
        raise ValueError("scores must have shape (Q,K)")
    q, k = x.shape
    if k == 1:
        return np.zeros(q, dtype=float)
    centered = x - np.median(x, axis=1, keepdims=True)
    scale = np.median(np.abs(centered), axis=1, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = centered / scale
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(np.clip(z, -60.0, 0.0))
    p /= p.sum(axis=1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-15, 1.0))).sum(axis=1) / np.log(k)


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

