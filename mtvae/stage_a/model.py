"""Stage-A model: per-modality trunks, a stochastic fusion head, and decoders.

The map for modality ``m`` is ``D_m . F . E_m``: the modality's encoder
``E_m``, a fusion layer ``F`` that emits a diagonal-Gaussian posterior
``q(z | x) = N(mu, diag(sigma^2))`` over the shared latent, and the modality's
decoder ``D_m``.

The posterior sits at the fusion output, not on the encoders, so the
per-modality trunks are plain :class:`~mtvae.nn.autoencoders.ModalityAutoencoder`
instances and pre-trained trunk checkpoints load unchanged (``staged`` mode).

The mean branch is a :func:`mtvae.nn.fusion_arch.make_fusion` block whose
output layer is zero-initialised, so ``mu = h`` exactly at initialisation. With
``sigma_mode="fixed"`` no log-variance parameters are created, so under
``deterministic=True`` the model is a deterministic fused autoencoder with the
same parameters and initialisation order.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


from mtvae.nn import fusion_arch  # noqa: E402
from mtvae.nn.autoencoders import (  # noqa: E402
    ModalityAutoencoder,
)

Modality = Literal["rna", "atac"]


def _zero_init_mlp(latent_dim: int, hidden: int, depth: int,
                   layer_norm: bool) -> nn.Sequential:
    """Build an MLP whose output layer is zeroed.

    Same layout as ``mtvae.nn.fusion_arch._identity_mlp``, so the log-variance
    head has the same shape and conditioning as the mean branch.

    Parameters
    ----------
    latent_dim : int
        Input and output width.
    hidden : int
        Hidden width.
    depth : int
        Number of hidden layers (>= 1).
    layer_norm : bool
        Whether to pre-normalise the input.

    Returns
    -------
    torch.nn.Sequential
        The head, contributing exactly zero at initialisation.
    """
    layers: list[nn.Module] = []
    if layer_norm:
        layers.append(nn.LayerNorm(latent_dim))
    layers.append(nn.Linear(latent_dim, hidden))
    layers.append(nn.SiLU())
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.SiLU())
    out = nn.Linear(hidden, latent_dim)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class StochasticFusion(nn.Module):
    """Fusion block emitting a diagonal-Gaussian posterior over the shared code.

    Parameters
    ----------
    latent_dim : int
        Shared-latent width.
    hidden_dim : int, default 1024
        Hidden width of both the mean branch and the log-variance head.
    depth : int, default 1
        Number of hidden layers in each branch.
    residual_scale : float, default 0.5
        Multiplier on the mean branch's residual refinement.
    layer_norm : bool, default False
        Pre-normalise the residual branches.
    arch : str, default "shared"
        Mean-branch architecture, passed to
        :func:`mtvae.nn.fusion_arch.make_fusion`.
    sigma_mode : {"learned", "fixed"}, default "learned"
        ``"learned"`` adds an amortised log-variance head; ``"fixed"`` creates no
        extra parameters and uses a constant ``sigma_init`` for every cell and
        dimension.
    sigma_init : float, default 0.1
        Posterior standard deviation at initialisation (and throughout when
        ``sigma_mode="fixed"``).
    stochastic_dims : int, default 0
        ``0`` makes every dimension stochastic. ``k > 0`` makes only dimensions
        ``[0, k)`` stochastic; the rest are passed through deterministically
        and carry no KL. This parallels the trainer's ``topo_identity_dims``,
        so the tilt and the prior can act on the same or different subspaces.

    Raises
    ------
    ValueError
        If ``sigma_init <= 0``, ``sigma_mode`` is unknown, or
        ``stochastic_dims`` is outside ``[0, latent_dim]``.

    Notes
    -----
    ``forward`` returns the posterior **mean**, so any code that calls the
    module as a plain fusion layer gets the mean, never a sample.
    """

    modality_aware: bool = True

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 1024,
        depth: int = 1,
        residual_scale: float = 0.5,
        layer_norm: bool = False,
        arch: str = "shared",
        sigma_mode: str = "learned",
        sigma_init: float = 0.1,
        stochastic_dims: int = 0,
    ) -> None:
        super().__init__()
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        if sigma_mode not in ("learned", "fixed"):
            raise ValueError(f"Unknown sigma_mode: {sigma_mode!r}")
        if not 0 <= stochastic_dims <= latent_dim:
            raise ValueError(
                f"stochastic_dims must lie in [0, {latent_dim}], got {stochastic_dims}"
            )

        self.latent_dim = int(latent_dim)
        self.sigma_mode = sigma_mode
        self.stochastic_dims = int(stochastic_dims)

        # Mean branch.
        self.mu_net = fusion_arch.make_fusion(
            arch, latent_dim, hidden_dim=hidden_dim, depth=depth,
            residual_scale=residual_scale, layer_norm=layer_norm,
        )
        self._mu_modality_aware = bool(getattr(self.mu_net, "modality_aware", False))

        # Log-variance head. Created only when learned, so the "fixed" path adds
        # no parameters and does not perturb initialisation order.
        self.logvar_net: nn.Module | None = (
            _zero_init_mlp(latent_dim, hidden_dim, depth, layer_norm)
            if sigma_mode == "learned" else None
        )
        self.register_buffer(
            "logvar_base",
            torch.full((latent_dim,), 2.0 * math.log(sigma_init)),
        )

        # Which dimensions carry noise and KL.
        mask = torch.ones(latent_dim)
        if self.stochastic_dims > 0:
            mask[self.stochastic_dims:] = 0.0
        self.register_buffer("stoch_mask", mask)

    def _mu(self, h: torch.Tensor, modality: Modality) -> torch.Tensor:
        """Posterior mean for one modality's encoder output."""
        if self._mu_modality_aware:
            return self.mu_net(h, modality)
        return self.mu_net(h)

    def forward(self, h: torch.Tensor, modality: Modality = "rna") -> torch.Tensor:
        """Return the posterior mean (same signature as a plain fusion layer)."""
        return self._mu(h, modality)

    def posterior(self, h: torch.Tensor,
                  modality: Modality) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` for one modality's encoder output.

        Parameters
        ----------
        h : torch.Tensor, shape (n, latent_dim)
            Encoder output.
        modality : {"rna", "atac"}
            Which modality ``h`` came from.

        Returns
        -------
        mu, logvar : torch.Tensor
            Both ``(n, latent_dim)``. On deterministic dimensions the noise is
            zeroed by ``stoch_mask`` in :meth:`rsample`, and the KL is masked
            out by the loss.
        """
        mu = self._mu(h, modality)
        base = self.logvar_base.to(mu.dtype).expand_as(mu)
        logvar = base if self.logvar_net is None else base + self.logvar_net(h)
        return mu, logvar

    def rsample(self, mu: torch.Tensor, logvar: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """Reparameterised draw ``mu + mask * sigma * eps``.

        Parameters
        ----------
        mu, logvar : torch.Tensor, shape (n, latent_dim)
            Posterior parameters.
        generator : torch.Generator, optional
            Seeded generator for the noise draw.

        Returns
        -------
        torch.Tensor
            One sample, shape ``(n, latent_dim)``.  Dimensions outside the
            stochastic block are returned exactly equal to ``mu``.
        """
        eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device,
                          generator=generator)
        return mu + self.stoch_mask.to(mu.dtype) * torch.exp(0.5 * logvar) * eps


class MergedVAE(nn.Module):
    """Two modality autoencoders sharing one stochastic bottleneck block.

    Parameters
    ----------
    rna_ae, atac_ae : torch.nn.Module
        The per-modality trunks (``ModalityAutoencoder`` or any module with
        ``encode``/``decode``). Constructed fresh in ``joint`` mode, loaded
        from pre-trained checkpoints in ``staged`` mode.
    fusion : StochasticFusion
        The shared stochastic bottleneck head.
    """

    def __init__(self, rna_ae: ModalityAutoencoder, atac_ae: ModalityAutoencoder,
                 fusion: StochasticFusion) -> None:
        super().__init__()
        self.rna_ae = rna_ae
        self.atac_ae = atac_ae
        self.fusion = fusion

    def _ae(self, modality: Modality) -> ModalityAutoencoder:
        """Return the trunk of ``modality``; raise ``ValueError`` if unknown."""
        if modality == "rna":
            return self.rna_ae
        if modality == "atac":
            return self.atac_ae
        raise ValueError(f"Unknown modality: {modality!r}")

    def posterior(self, x: torch.Tensor,
                  modality: Modality) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode features to the shared posterior ``(mu, logvar)``."""
        h = self._ae(modality).encode(x)
        return self.fusion.posterior(h, modality)

    def decode(self, z: torch.Tensor, modality: Modality) -> torch.Tensor:
        """Decode a shared code back to one modality's feature space."""
        return self._ae(modality).decode(z)

    def encode_mu(self, x: torch.Tensor, modality: Modality) -> torch.Tensor:
        """Return the posterior mean only — the latent that gets scored."""
        return self.posterior(x, modality)[0]

    def forward(self, x: torch.Tensor, modality: Modality,
                generator: torch.Generator | None = None,
                deterministic: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pass.

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)
            Feature batch for one modality.
        modality : {"rna", "atac"}
            Which modality ``x`` is.
        generator : torch.Generator, optional
            Seeded generator for the reparameterisation noise.
        deterministic : bool, default False
            If True, ``z = mu`` and no noise is drawn.

        Returns
        -------
        x_rec, z, mu, logvar : torch.Tensor
            Reconstruction and the three latent tensors.
        """
        mu, logvar = self.posterior(x, modality)
        z = mu if deterministic else self.fusion.rsample(mu, logvar, generator)
        return self.decode(z, modality), z, mu, logvar
