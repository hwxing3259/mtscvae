"""Entropic optimal-transport costs and Sinkhorn divergences, plus Stage-B
default constants.

:func:`sinkhorn_div` and :func:`sinkhorn_div_scaled` compute the debiased
Sinkhorn divergence ``OT(x, y) - 0.5 * (OT(x, x) + OT(y, y))`` with log-domain
Sinkhorn iterations and a squared-Euclidean cost. The scaled variant can divide
each cost matrix by its own detached mean, so one ``eps`` works across spaces
of different scale; see :mod:`mtvae.stage_b.losses` for a variant that shares
one scale across the three terms.

The module-level constants are Stage-B defaults and PFC stage bookkeeping
(``STAGE_TO_IDX``, ``IDX_TO_STAGE``, ``N_STAGES``, ``D_LATENT``), derived from
:mod:`mtvae._constants`.
"""
from __future__ import annotations

import numpy as np
import torch

from mtvae import _constants as config

# ---------------------------------------------------------------------------
# Stage-B defaults
# ---------------------------------------------------------------------------

TRAIN_ITERS = 3000
WARMUP_ITERS = 200
LR = 5e-4
BATCH = 256
SINKHORN_EPS = 0.01
SINKHORN_ITERS = 30

LAMBDA_CHAINED = 1.0
LAMBDA_PER_PAIR = 2.0   # one stage transition, vs several summed in the chained term
LOG_EVERY = 50

STAGE_TO_IDX = {s: i for i, s in enumerate(config.STAGES_ORDERED)}
IDX_TO_STAGE = {i: s for s, i in STAGE_TO_IDX.items()}
D_LATENT = config.LATENT_DIM
N_STAGES = len(config.STAGES_ORDERED)


# ---------------------------------------------------------------------------
# Sinkhorn optimal-transport costs
# ---------------------------------------------------------------------------


def log_sinkhorn_cost(x: torch.Tensor, y: torch.Tensor, eps: float,
                      n_iters: int) -> torch.Tensor:
    """Entropic OT cost between two point clouds, in log space.

    Parameters
    ----------
    x : torch.Tensor, shape (n, d)
        Source point cloud.
    y : torch.Tensor, shape (m, d)
        Target point cloud.
    eps : float
        Entropic regularisation strength.
    n_iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    torch.Tensor
        Scalar transport cost ``(P * C).sum()``.
    """
    n, m = x.shape[0], y.shape[0]
    C = torch.cdist(x, y).pow(2)
    log_a = torch.full((n,), -np.log(n), device=x.device, dtype=x.dtype)
    log_b = torch.full((m,), -np.log(m), device=y.device, dtype=y.dtype)
    u = torch.zeros(n, device=x.device, dtype=x.dtype)
    v = torch.zeros(m, device=y.device, dtype=y.dtype)
    for _ in range(n_iters):
        u = eps * (log_a - torch.logsumexp((-C + v[None, :]) / eps, dim=1))
        v = eps * (log_b - torch.logsumexp((-C + u[:, None]) / eps, dim=0))
    log_P = (u[:, None] + v[None, :] - C) / eps
    P = torch.exp(log_P)
    return (P * C).sum()


def sinkhorn_div(x: torch.Tensor, y: torch.Tensor, eps: float = SINKHORN_EPS,
                 n_iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Debiased Sinkhorn divergence ``OT(x,y) - 0.5*(OT(x,x) + OT(y,y))``.

    Parameters
    ----------
    x : torch.Tensor, shape (n, d)
        Source point cloud.
    y : torch.Tensor, shape (m, d)
        Target point cloud.
    eps : float, default SINKHORN_EPS
        Entropic regularisation strength.
    n_iters : int, default SINKHORN_ITERS
        Number of Sinkhorn iterations.

    Returns
    -------
    torch.Tensor
        Scalar divergence.
    """
    c_xy = log_sinkhorn_cost(x, y, eps, n_iters)
    c_xx = log_sinkhorn_cost(x, x, eps, n_iters)
    c_yy = log_sinkhorn_cost(y, y, eps, n_iters)
    return c_xy - 0.5 * (c_xx + c_yy)


def _log_sinkhorn_cost_scaled(x: torch.Tensor, y: torch.Tensor, eps: float,
                              n_iters: int,
                              scale_cost: bool = True) -> torch.Tensor:
    """Entropic OT cost with optional mean-normalisation of the cost matrix.

    Parameters
    ----------
    x : torch.Tensor, shape (n, d)
        Source point cloud.
    y : torch.Tensor, shape (m, d)
        Target point cloud.
    eps : float
        Entropic regularisation strength.
    n_iters : int
        Number of Sinkhorn iterations.
    scale_cost : bool, default True
        Divide the cost matrix by its (detached) mean before solving, so a
        single ``eps`` works across latent and feature spaces.

    Returns
    -------
    torch.Tensor
        Scalar transport cost.
    """
    n, m = x.shape[0], y.shape[0]
    C = torch.cdist(x, y).pow(2)
    if scale_cost:
        C = C / (C.mean().detach() + 1e-9)
    log_a = torch.full((n,), -np.log(n), device=x.device, dtype=x.dtype)
    log_b = torch.full((m,), -np.log(m), device=y.device, dtype=y.dtype)
    u = torch.zeros(n, device=x.device, dtype=x.dtype)
    v = torch.zeros(m, device=y.device, dtype=y.dtype)
    for _ in range(n_iters):
        u = eps * (log_a - torch.logsumexp((-C + v[None, :]) / eps, dim=1))
        v = eps * (log_b - torch.logsumexp((-C + u[:, None]) / eps, dim=0))
    log_P = (u[:, None] + v[None, :] - C) / eps
    P = torch.exp(log_P)
    return (P * C).sum()


def sinkhorn_div_scaled(x: torch.Tensor, y: torch.Tensor, eps: float,
                        n_iters: int,
                        scale_cost: bool = True) -> torch.Tensor:
    """Debiased Sinkhorn divergence with optional cost rescaling.

    With ``scale_cost=True`` each of the three costs is normalised by its own
    mean. In float32 this roughly halves the divergence of an exactly
    collapsed ``x`` (all rows equal), so collapse can score better than the
    observed data; use
    :func:`mtvae.stage_b.losses.sinkhorn_div_shared_scale` where a prediction
    can collapse.

    Parameters
    ----------
    x : torch.Tensor, shape (n, d)
        Source point cloud.
    y : torch.Tensor, shape (m, d)
        Target point cloud.
    eps : float
        Entropic regularisation strength.
    n_iters : int
        Number of Sinkhorn iterations.
    scale_cost : bool, default True
        Passed through to :func:`_log_sinkhorn_cost_scaled`.

    Returns
    -------
    torch.Tensor
        Scalar divergence.
    """
    c_xy = _log_sinkhorn_cost_scaled(x, y, eps, n_iters, scale_cost=scale_cost)
    c_xx = _log_sinkhorn_cost_scaled(x, x, eps, n_iters, scale_cost=scale_cost)
    c_yy = _log_sinkhorn_cost_scaled(y, y, eps, n_iters, scale_cost=scale_cost)
    return c_xy - 0.5 * (c_xx + c_yy)
