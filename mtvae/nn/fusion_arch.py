"""Fusion layers that map each modality's encoder output into the shared latent.
Each takes ``(z, modality)`` and starts as the identity; :func:`make_fusion` builds one by name."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

Modality = Literal["rna", "atac"]
#: Marks this module's fusions as taking ``(z, modality)``.
_MODALITY_AWARE = True
#: Modality keys of the per-modality branches.
MODALITIES: tuple[str, str] = ("rna", "atac")


def _identity_mlp(latent_dim: int, hidden: int, depth: int,
                  layer_norm: bool) -> nn.Sequential:
    """Build an MLP whose final layer is zeroed, so the block starts at zero.

    Parameters
    ----------
    latent_dim : int
        Input and output width.
    hidden : int
        Hidden width.
    depth : int
        Number of hidden layers (>= 1).
    layer_norm : bool
        Whether to pre-normalise the input.  Pre-norm conditions the residual
        branch without touching the identity path.

    Returns
    -------
    torch.nn.Sequential
        The refinement branch, with its output layer zero-initialised.
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
    # Zero the output layer: the residual branch contributes nothing at step 0,
    # so F is exactly the identity for every architecture, width and depth.
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class _ModalityAware(nn.Module):
    """Base for fusions whose ``forward`` takes ``(z, modality)``.

    Callers check :attr:`modality_aware` to decide whether to pass the
    modality, so a single-argument fusion module can be used unwrapped.
    """

    #: Marks the two-argument ``forward`` signature.
    modality_aware: bool = True


class _ResidualMLP(nn.Module):
    """One residual refinement branch: ``z -> z + scale * net(z)``."""

    def __init__(self, latent_dim: int, hidden: int, depth: int,
                 residual_scale: float, layer_norm: bool) -> None:
        super().__init__()
        self.net = _identity_mlp(latent_dim, hidden, depth, layer_norm)
        self.residual_scale = float(residual_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Refine ``z`` residually."""
        return z + self.net(z) * self.residual_scale


class SharedFusion(_ModalityAware):
    """One refinement map applied to both modalities (the default design).

    Parameters
    ----------
    latent_dim : int
        Shared-latent width.
    hidden_dim : int
        Hidden width of the refinement branch.
    depth : int, default 1
        Number of hidden layers.
    residual_scale : float, default 0.5
        Multiplier on the refinement.
    layer_norm : bool, default False
        Pre-normalise the residual branch.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, depth: int = 1,
                 residual_scale: float = 0.5, layer_norm: bool = False) -> None:
        super().__init__()
        self.block = _ResidualMLP(latent_dim, hidden_dim, depth,
                                  residual_scale, layer_norm)

    def forward(self, z: torch.Tensor, modality: Modality = "rna") -> torch.Tensor:
        """Apply the shared map; ``modality`` is accepted and ignored."""
        return self.block(z)


class PerModalityFusion(_ModalityAware):
    """Independent refinement map per modality.

    Has twice the parameters of :class:`SharedFusion`.

    Parameters
    ----------
    latent_dim : int
        Shared-latent width.
    hidden_dim : int
        Hidden width of each refinement branch.
    depth : int, default 1
        Number of hidden layers per branch.
    residual_scale : float, default 0.5
        Multiplier on each refinement.
    layer_norm : bool, default False
        Pre-normalise each residual branch.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, depth: int = 1,
                 residual_scale: float = 0.5, layer_norm: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict({
            m: _ResidualMLP(latent_dim, hidden_dim, depth, residual_scale, layer_norm)
            for m in MODALITIES
        })

    def forward(self, z: torch.Tensor, modality: Modality) -> torch.Tensor:
        """Apply the branch belonging to ``modality``."""
        return self.blocks[modality](z)


class SharedPrivateFusion(_ModalityAware):
    """Shared trunk plus a small per-modality correction.

    Between :class:`SharedFusion` and :class:`PerModalityFusion`: a common
    refinement carries what is shared, and a narrow one-hidden-layer private
    branch per modality supplies a modality-specific correction (e.g. a
    rotation) that a single tied map cannot represent.

    Parameters
    ----------
    latent_dim : int
        Shared-latent width.
    hidden_dim : int
        Hidden width of the shared trunk.
    private_dim : int, default 64
        Hidden width of each private branch.
    depth : int, default 1
        Number of hidden layers in the shared trunk.
    residual_scale : float, default 0.5
        Multiplier on both the shared and private refinements.
    layer_norm : bool, default False
        Pre-normalise the residual branches.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, private_dim: int = 64,
                 depth: int = 1, residual_scale: float = 0.5,
                 layer_norm: bool = False) -> None:
        super().__init__()
        self.shared = _ResidualMLP(latent_dim, hidden_dim, depth,
                                   residual_scale, layer_norm)
        self.private = nn.ModuleDict({
            m: _ResidualMLP(latent_dim, private_dim, 1, residual_scale, layer_norm)
            for m in MODALITIES
        })

    def forward(self, z: torch.Tensor, modality: Modality) -> torch.Tensor:
        """Apply the shared trunk, then the modality's private correction."""
        return self.private[modality](self.shared(z))


class PerModalityLinearFusion(_ModalityAware):
    """Per-modality affine map, identity-initialised.

    A modality-specific linear transform (e.g. a rotation or shear) with no
    nonlinearity, ``latent_dim * (latent_dim + 1)`` parameters per modality.

    Parameters
    ----------
    latent_dim : int
        Shared-latent width.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.maps = nn.ModuleDict({m: nn.Linear(latent_dim, latent_dim)
                                   for m in MODALITIES})
        for m in MODALITIES:
            nn.init.eye_(self.maps[m].weight)
            nn.init.zeros_(self.maps[m].bias)

    def forward(self, z: torch.Tensor, modality: Modality) -> torch.Tensor:
        """Apply the modality's affine map."""
        return self.maps[modality](z)


#: Architectures :func:`make_fusion` can build, by ``fusion_arch`` name.
ARCHITECTURES: tuple[str, ...] = (
    "shared", "per_mod", "shared_private", "per_mod_linear",
)


def make_fusion(arch: str, latent_dim: int, hidden_dim: int = 1024,
                depth: int = 1, residual_scale: float = 0.5,
                layer_norm: bool = False, private_dim: int = 64) -> nn.Module:
    """Build a fusion module by name.

    Parameters
    ----------
    arch : str
        One of :data:`ARCHITECTURES`.
    latent_dim : int
        Shared-latent width.
    hidden_dim : int, default 1024
        Hidden width of the (shared) refinement branch.
    depth : int, default 1
        Number of hidden layers.
    residual_scale : float, default 0.5
        Multiplier on the refinement.
    layer_norm : bool, default False
        Pre-normalise residual branches.
    private_dim : int, default 64
        Hidden width of the private branches in ``shared_private``.

    Returns
    -------
    torch.nn.Module
        A module whose ``forward(z, modality)`` returns the fused latent.

    Raises
    ------
    ValueError
        If ``arch`` is not a known architecture.
    """
    if arch == "shared":
        return SharedFusion(latent_dim, hidden_dim, depth, residual_scale, layer_norm)
    if arch == "per_mod":
        return PerModalityFusion(latent_dim, hidden_dim, depth, residual_scale, layer_norm)
    if arch == "shared_private":
        return SharedPrivateFusion(latent_dim, hidden_dim, private_dim, depth,
                                   residual_scale, layer_norm)
    if arch == "per_mod_linear":
        return PerModalityLinearFusion(latent_dim)
    raise ValueError(f"Unknown fusion_arch {arch!r}; expected one of {ARCHITECTURES}")
