"""A Sinkhorn divergence that does not reward collapse.
:func:`sinkhorn_div_shared_scale` scales all three cost matrices by the cross term's."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _sinkhorn_cost(x: torch.Tensor, y: torch.Tensor, eps: float, n_iters: int,
                   scale: torch.Tensor | None) -> torch.Tensor:
    """Entropic OT cost with an externally supplied normalisation.

    Parameters
    ----------
    x : torch.Tensor, shape (n, d)
        Source cloud.
    y : torch.Tensor, shape (m, d)
        Target cloud.
    eps : float
        Entropic regularisation strength.
    n_iters : int
        Sinkhorn iterations.
    scale : torch.Tensor or None
        Detached positive scalar to divide the cost matrix by. ``None`` leaves
        the cost unnormalised.

    Returns
    -------
    torch.Tensor
        Scalar transport cost ``(P * C).sum()``.
    """
    n, m = x.shape[0], y.shape[0]
    C = torch.cdist(x, y).pow(2)
    if scale is not None:
        C = C / scale
    log_a = torch.full((n,), -np.log(n), device=x.device, dtype=x.dtype)
    log_b = torch.full((m,), -np.log(m), device=y.device, dtype=y.dtype)
    u = torch.zeros(n, device=x.device, dtype=x.dtype)
    v = torch.zeros(m, device=y.device, dtype=y.dtype)
    for _ in range(n_iters):
        u = eps * (log_a - torch.logsumexp((-C + v[None, :]) / eps, dim=1))
        v = eps * (log_b - torch.logsumexp((-C + u[:, None]) / eps, dim=0))
    P = torch.exp((u[:, None] + v[None, :] - C) / eps)
    return (P * C).sum()


def sinkhorn_div_shared_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    n_iters: int = 20,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence with one normalisation shared by all terms.

    Replacement for
    ``mtvae.stage_b.sinkhorn.sinkhorn_div_scaled(..., scale_cost=True)`` that
    ranks a collapsed prediction worst rather than best. The scale is the
    detached mean squared distance between ``x`` and ``y``.

    Parameters
    ----------
    x : torch.Tensor, shape (n, d)
        Predicted cloud.
    y : torch.Tensor, shape (m, d)
        Observed cloud.
    eps : float, default 0.05
        Entropic regularisation strength.
    n_iters : int, default 20
        Sinkhorn iterations.

    Returns
    -------
    torch.Tensor
        Scalar divergence, differentiable in ``x``.
    """
    scale = torch.cdist(x, y).pow(2).mean().detach() + 1e-9
    c_xy = _sinkhorn_cost(x, y, eps, n_iters, scale)
    c_xx = _sinkhorn_cost(x, x, eps, n_iters, scale)
    c_yy = _sinkhorn_cost(y, y, eps, n_iters, scale)
    return c_xy - 0.5 * (c_xx + c_yy)


def collapse_sanity(y: torch.Tensor, eps: float = 0.05,
                    n_iters: int = 20) -> Tuple[float, float, float]:
    """Score three reference answers with :func:`sinkhorn_div_shared_scale`.

    ``y`` is split into halves ``a`` and ``b``. A divergence suitable for
    training a movable decoder must score the independent sample below the
    collapsed prediction.

    Parameters
    ----------
    y : torch.Tensor, shape (n, d)
        An observed cloud.
    eps : float, default 0.05
        Entropic regularisation strength.
    n_iters : int, default 20
        Sinkhorn iterations.

    Returns
    -------
    tuple of float
        ``(self_score, independent_score, collapse_score)``: ``S(a, a)``,
        ``S(a, b)``, and ``S(mean(y), b)`` with the mean repeated for every
        row.

    Raises
    ------
    ValueError
        If ``y`` has fewer than four rows.
    """
    if y.shape[0] < 4:
        raise ValueError(f"need >= 4 rows to split, got {y.shape[0]}")
    half = y.shape[0] // 2
    a, b = y[:half], y[half:2 * half]
    const = y.mean(dim=0, keepdim=True).expand(half, y.shape[1]).contiguous()
    with torch.no_grad():
        return (float(sinkhorn_div_shared_scale(a, a, eps, n_iters)),
                float(sinkhorn_div_shared_scale(a, b, eps, n_iters)),
                float(sinkhorn_div_shared_scale(const, b, eps, n_iters)))
