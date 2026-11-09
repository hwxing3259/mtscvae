"""Declarative choice of a modality's encoder/decoder trunk.

A :class:`TrunkSpec` travels with its :class:`mtvae.data.Modality` and states
which trunk the modality gets: the plain MLP autoencoder, the
chromosome-aware Conv1d trunk, or the same Conv1d trunk without chromosome
boundaries. :func:`build` constructs the trunk.

The spec, rather than a ready-built ``nn.Module``, is what the caller supplies
because construction order matters for reproducibility: Stage-A training calls
``torch.manual_seed`` and only then constructs the trunks, so a module built by
the caller would be initialised from a different RNG position and every weight
would change. :func:`build` is called inside the trainer at the position of
the plain constructor.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Union

import numpy as np
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class TrunkSpec:
    """Which encoder/decoder trunk a modality gets.

    The spec is attached to its modality, so it cannot be applied to the
    wrong one even when both modalities have the same feature count.

    Parameters
    ----------
    kind : {"mlp", "spatial", "conv"}, default "mlp"
        ``"mlp"`` is :class:`mtvae.nn.autoencoders.ModalityAutoencoder`.
        ``"spatial"`` is the chromosome-aware Conv1d trunk, meaningful only
        where features are ordered along a genomic coordinate. ``"conv"`` is
        the same architecture with an all-ones boundary mask (no chromosome
        awareness); it has the same parameter shapes as ``"spatial"`` and so
        consumes the RNG identically.
    chr_ids : numpy.ndarray, optional
        Per-feature chromosome index. Required when ``kind="spatial"``, and its
        length must equal the modality's feature count.
    kernel_sizes : int or sequence of int, default 3
        Conv kernel widths.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or ``kind="spatial"`` without ``chr_ids``.
    """
    kind: str = "mlp"
    chr_ids: Optional[np.ndarray] = None
    kernel_sizes: Union[int, Sequence[int]] = 3

    def __post_init__(self) -> None:
        if self.kind not in ("mlp", "spatial", "conv"):
            raise ValueError(f"unknown trunk kind {self.kind!r}")
        if self.kind == "spatial" and self.chr_ids is None:
            raise ValueError("kind='spatial' requires chr_ids; use kind='conv' "
                             "for the same architecture without them")


def build(spec: TrunkSpec, *, input_dim: int, latent_dim: int) -> nn.Module:
    """Construct one modality trunk from its spec.

    Called by :func:`mtvae.stage_a.trainer.train_merged` after
    ``torch.manual_seed``, at the position of the plain
    ``ModalityAutoencoder(...)`` construction, so the trunk's initial weights
    depend only on the seed.

    Parameters
    ----------
    spec : TrunkSpec
        Which trunk to build.
    input_dim, latent_dim : int
        Feature count of the modality and latent dimension.

    Returns
    -------
    torch.nn.Module
        A trunk with ``encode``/``decode`` methods.

    Raises
    ------
    ValueError
        If a spatial spec's ``chr_ids`` length does not equal ``input_dim``.
    """
    from mtvae.nn.autoencoders import ModalityAutoencoder

    if spec.kind == "mlp":
        return ModalityAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    if spec.kind == "conv":
        # The spatial trunk with chr_ids=None (an all-ones mask): same class,
        # parameter shapes and RNG consumption, so the mask is the only
        # difference from "spatial".
        from mtvae.nn.atac_spatial_encoder import (
            SpatialModalityAutoencoder,
        )
        return SpatialModalityAutoencoder(
            input_dim=input_dim, latent_dim=latent_dim, chr_ids=None,
            spatial_kernel_sizes=spec.kernel_sizes)

    from mtvae.nn.atac_spatial_encoder import (
        SpatialModalityAutoencoder,
    )
    if len(spec.chr_ids) != input_dim:
        raise ValueError(
            f"chr_ids has {len(spec.chr_ids)} entries but this modality has "
            f"{input_dim} features; the peak list does not match the matrix")
    return SpatialModalityAutoencoder(
        input_dim=input_dim, latent_dim=latent_dim, chr_ids=spec.chr_ids,
        spatial_kernel_sizes=spec.kernel_sizes)
