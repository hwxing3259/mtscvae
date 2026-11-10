"""Sliced Wasserstein distance between two point clouds.
Stage A uses it to align the two modalities' latent distributions."""
from __future__ import annotations

import torch


def sliced_wasserstein_2(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_projections: int = 64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Monte Carlo estimate of the squared sliced 2-Wasserstein distance.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d)
        First point cloud.
    Y : torch.Tensor, shape (n, d)
        Second point cloud; must have the same number of rows as ``X``.
    n_projections : int, default 64
        Number of random projection directions.
    generator : torch.Generator, optional
        Generator for the projection directions; the global torch RNG is used
        when omitted.

    Returns
    -------
    torch.Tensor
        Scalar: the squared 1-D 2-Wasserstein distance, averaged over the
        projections.

    Raises
    ------
    AssertionError
        If the feature dimensions or the numbers of rows differ.

    Notes
    -----
    Directions are drawn uniformly on the unit sphere. For equal-size samples,
    the squared 1-D 2-Wasserstein distance along a direction ``v`` is the mean
    squared difference between ``sort(X @ v)`` and ``sort(Y @ v)``.
    """
    device = X.device
    dtype = X.dtype
    assert X.shape[1] == Y.shape[1], "feature dimensions must match"
    assert X.shape[0] == Y.shape[0], "SW here requires equal batch sizes; caller enforces this"
    d = X.shape[1]
    n = X.shape[0]

    # Sample projection directions: Gaussian → normalise.
    if generator is None:
        V = torch.randn(n_projections, d, device=device, dtype=dtype)
    else:
        V = torch.randn(n_projections, d, device=device, dtype=dtype, generator=generator)
    V = V / (V.norm(dim=1, keepdim=True) + 1e-8)

    # Project: (P, n) for X and (P, n) for Y
    X_proj = X @ V.T    # (n, P)
    Y_proj = Y @ V.T    # (n, P)

    # Sort along batch axis for each projection
    X_sorted, _ = torch.sort(X_proj, dim=0)
    Y_sorted, _ = torch.sort(Y_proj, dim=0)

    # 1-D squared W2 per projection = mean squared difference of sorted values
    sq = (X_sorted - Y_sorted).pow(2).mean(dim=0)   # (P,)
    sw2 = sq.mean()
    return sw2
