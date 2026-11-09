"""Effective rank — one implementation, pure numpy.

Replaces eight byte-equivalent copies that had drifted apart in name only:
``eff_rank`` in ``data/stage_score.py``, ``pfc/bench/metrics.py``,
``pfc/build_teachers.py``, ``pfc/run_pfc_vae.py`` and ``validation/audit_runs.py``;
``_eff_rank`` in ``vae/run_vae_merged.py`` and ``data/snr_prepare_ours.py``;
``participation_ratio`` in ``vae/losses.py``. Verified bit-equal across all eight
on the shipped latents and the Harmony teacher before consolidating.

**Pure numpy, deliberately.** This module must be importable by the torch-free
scorers -- see ``suite/__init__``. It is also why the consolidation points this
way round: the guarded implementation lived in ``vae/losses.py``, which imports
torch, so moving the *callers* there was not an option.

The guard is the one thing that was not universal: only ``vae/losses.py`` checked
``denom > 0``. It is kept here, so a degenerate all-zero embedding returns
``nan`` rather than raising -- ``0/0`` in numpy is a warning, not an exception,
and a silent ``nan`` propagating into a metrics table is worse than an explicit
one.

NOT the same as ``dynamics/trainer.participation_ratio``, which takes a
``torch.Tensor`` inside the training loop and is deliberately left alone.
"""
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
