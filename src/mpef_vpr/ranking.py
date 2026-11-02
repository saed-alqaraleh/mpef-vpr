"""Deterministic candidate ranking."""

from __future__ import annotations

import numpy as np


def rank_candidates(probability: np.ndarray, global_rank: np.ndarray,
                    candidate_ids: np.ndarray) -> np.ndarray:
    """Return candidate indices ordered by probability, then frozen tie rules.

    Exact ties retain lower (better) global rank, then lexicographically smaller
    database ID. The rule is independent of the underlying sort implementation.
    """

    p = np.asarray(probability, dtype=float)
    gr = np.asarray(global_rank)
    ids = np.asarray(candidate_ids).astype(str)
    if p.ndim == 1:
        p, gr, ids = p[None, :], gr[None, :], ids[None, :]
        squeeze = True
    else:
        squeeze = False
    if p.shape != gr.shape or p.shape != ids.shape:
        raise ValueError("probability, global_rank, and candidate_ids must have equal shape")
    out = np.empty(p.shape, dtype=int)
    for i in range(len(p)):
        out[i] = np.lexsort((ids[i], gr[i], -p[i]))
    return out[0] if squeeze else out

