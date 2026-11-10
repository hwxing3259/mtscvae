"""Effective rank — one implementation, pure numpy."""
from __future__ import annotations

import numpy as np


def eff_rank(x: np.ndarray) -> float:
    """Participation ratio of an embedding's centred spectrum.

    ``(sum lambda)^2 / sum lambda^2`` over the eigenvalues of the column-mean-
    centred covariance, obtained from the squared singular values.

    This matches ``stage_score.py`` and ``score_screen.py``, which square the
    singular values first — **not** ``analyse_geometry.py``, which does not and
    therefore reports a systematically larger number. Every ``eff_rank`` and
    ``teacher_eff_rank`` in every shipped run record was produced by this
    formula, in float64, on the centred array.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Embedding. Cast to float64 before centring; the cast is load-bearing,
        as float32 accumulation moves the value in the 7th digit.

    Returns
    -------
    float
        The participation ratio, or ``nan`` for a degenerate (zero-variance)
        embedding.
    """
    z = np.asarray(x, np.float64)
    z = z - z.mean(0)
    v = np.linalg.svd(z, compute_uv=False) ** 2
    denom = float((v ** 2).sum())
    return float(v.sum() ** 2 / denom) if denom > 0 else float("nan")


def eff_rank_joint(a: np.ndarray, b: np.ndarray) -> float:
    """Effective rank of two embeddings stacked along the cell axis.

    Parameters
    ----------
    a, b : numpy.ndarray
        Embeddings with the same feature dimension.

    Returns
    -------
    float
    """
    return eff_rank(np.concatenate([np.asarray(a), np.asarray(b)], axis=0))
