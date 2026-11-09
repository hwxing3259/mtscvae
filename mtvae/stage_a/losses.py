"""Stage-A reconstruction and KL terms, and latent-geometry diagnostics.

The KL to ``N(0, I)`` (:func:`gaussian_kl`) has four controls:

``beta_kl``
    Overall weight (applied by the trainer).
``kl_scale``
    ``"mind"`` divides the Gaussian KL by ``n_cells * n_stochastic_dims``, the
    normalisation of the MIND reference implementation, while the
    reconstruction is a per-element mean; this is an effective ``beta`` of
    ``1/D`` relative to the standard ELBO. ``"elbo"`` is the per-sample sum
    over dimensions, ``D`` times larger.
``kl_free_bits``
    Per-dimension floor in nats. Below the floor a dimension's KL is not
    penalised, which removes the incentive to prune it and so opposes
    posterior collapse.
``kl_dims``
    Factorised posterior: only the first ``k`` dimensions are stochastic and
    penalised; the rest are deterministic and cannot be pruned by the prior.

Diagnostics
-----------
:func:`participation_ratio` (the effective rank of :mod:`mtvae.rank`) should
be reported on the posterior **mean** and on a **sample** side by side: the
covariance of a sample is ``Cov(mu) + sigma^2 I``, so every collapsed
dimension adds ``sigma^2`` to the spectrum and inflates the sampled
participation ratio while carrying no information. :func:`active_units` is the
companion information measure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: Default reconstruction loss of each modality slot when the dataset declares
#: none: slot 0 (RNA) is an L2 mean; slot 1 (ATAC) is a Huber loss at
#: delta=1.0, which tolerates the clipped tails of the ATAC features.
DEFAULT_SLOT_LOSSES: tuple[str, str] = ("mse", "huber")

#: Huber delta.
_HUBER_DELTA = 1.0


def reconstruction_loss(kind: str, pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Reconstruction term for one modality.

    Lets a modality declare its noise model (``Modality.loss``) instead of
    inheriting the default of its slot.

    Parameters
    ----------
    kind : {"mse", "huber"}
        Which loss: mean squared error, or Huber at ``delta=1.0``. Unknown
        values raise rather than falling back to a default.
    pred, target : torch.Tensor
        Reconstruction and its target, same shape.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    ValueError
        If ``kind`` is not recognised.
    """
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=_HUBER_DELTA)
    raise ValueError(
        f"unknown reconstruction loss {kind!r}; expected one of "
        f"{sorted(('mse', 'huber'))}")


def gaussian_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    stoch_mask: torch.Tensor,
    free_bits: float = 0.0,
    scale: str = "mind",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal-Gaussian KL to ``N(0, I)``, factorised and floored.

    Parameters
    ----------
    mu, logvar : torch.Tensor, shape (n, d)
        Posterior parameters.
    stoch_mask : torch.Tensor, shape (d,)
        1.0 on stochastic dimensions, 0.0 on deterministic ones.  Masked
        dimensions contribute no KL.
    free_bits : float, default 0.0
        Per-dimension floor in nats, applied to the batch-mean KL of each
        dimension. ``0.0`` disables it.
    scale : {"mind", "elbo"}, default "mind"
        ``"elbo"`` returns nats per sample (sum over dimensions). ``"mind"``
        divides that by the number of stochastic dimensions (per-element
        normalisation, as in the MIND reference implementation).

    Returns
    -------
    kl : torch.Tensor
        Scalar, ready to be multiplied by ``beta_kl``.
    kl_per_dim : torch.Tensor, shape (d,)
        Batch-mean KL of each dimension, **before** the free-bits floor, for
        diagnostics.  Deterministic dimensions are exactly zero.

    Raises
    ------
    ValueError
        If ``scale`` is not a known option.
    """
    if scale not in ("mind", "elbo"):
        raise ValueError(f"Unknown kl_scale: {scale!r}")

    mask = stoch_mask.to(mu.dtype)
    # 0.5 * (mu^2 + sigma^2 - 1 - log sigma^2), per element.
    kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar) * mask
    kl_per_dim = kl_elem.mean(dim=0)                       # (d,)

    floored = (torch.clamp(kl_per_dim, min=float(free_bits)) * mask
               if free_bits > 0.0 else kl_per_dim)
    kl = floored.sum()                                     # nats per sample

    if scale == "mind":
        n_stoch = torch.clamp(mask.sum(), min=1.0)
        kl = kl / n_stoch

    return kl, kl_per_dim.detach()


def active_units(kl_per_dim: torch.Tensor, threshold: float = 1e-2) -> int:
    """Count dimensions carrying more than ``threshold`` nats.

    Parameters
    ----------
    kl_per_dim : torch.Tensor, shape (d,)
        Batch-mean KL of each dimension.
    threshold : float, default 0.01
        Nats above which a dimension counts as used (the conventional value).

    Returns
    -------
    int
        Number of active dimensions.
    """
    return int((kl_per_dim > float(threshold)).sum().item())


# The effective rank, exposed here under the name ``participation_ratio``.
from mtvae.rank import (  # noqa: E402
    eff_rank as participation_ratio,
)

def joint_participation_ratio(rna: np.ndarray, atac: np.ndarray) -> float:
    """Participation ratio of the row-stacked RNA+ATAC codes.

    Same as :func:`mtvae.rank.eff_rank_joint`: the two clouds are
    concatenated along the cell axis before centring, so a between-modality
    centroid offset contributes its own direction to the spectrum.

    Parameters
    ----------
    rna, atac : numpy.ndarray
        Per-modality latents, ``(n_m, d)``.

    Returns
    -------
    float
        Joint participation ratio.
    """
    return participation_ratio(np.concatenate([rna, atac], axis=0))


def geometry_report(mu_rna: np.ndarray, mu_atac: np.ndarray,
                    z_rna: np.ndarray, z_atac: np.ndarray) -> dict[str, float]:
    """Rank diagnostics on the mean and on a sample, side by side.

    Parameters
    ----------
    mu_rna, mu_atac : numpy.ndarray
        Posterior means.
    z_rna, z_atac : numpy.ndarray
        One reparameterised sample of the same cells.

    Returns
    -------
    dict of str to float
        ``eff_rank_{joint,rna,atac}_mu`` and the ``_sample`` counterparts.  A
        large gap between the two is the signature of noise-filled dimensions.
    """
    return {
        "eff_rank_joint_mu": joint_participation_ratio(mu_rna, mu_atac),
        "eff_rank_rna_mu": participation_ratio(mu_rna),
        "eff_rank_atac_mu": participation_ratio(mu_atac),
        "eff_rank_joint_sample": joint_participation_ratio(z_rna, z_atac),
        "eff_rank_rna_sample": participation_ratio(z_rna),
        "eff_rank_atac_sample": participation_ratio(z_atac),
    }
