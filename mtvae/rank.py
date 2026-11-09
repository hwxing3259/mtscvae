"""Effective rank of an embedding (participation ratio).

The module is pure numpy so that scoring code can import it without torch.
"""
from __future__ import annotations

import numpy as np


def eff_rank(x: np.ndarray) -> float:
    """Participation ratio of an embedding's centred spectrum.

    ``(sum lambda)^2 / sum lambda^2`` over the eigenvalues ``lambda`` of the
    covariance of the column-centred embedding, taken as the squared singular
    values of the centred array. (Using the singular values themselves would
    give a systematically larger number.)

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Embedding. Cast to float64 before centring, since float32 accumulation
        changes the value in the 7th significant digit.

    Returns
    -------
    float
        The participation ratio, or ``nan`` for a degenerate (zero-variance)
        embedding, so that the degenerate case is visible downstream rather
        than raising.
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
