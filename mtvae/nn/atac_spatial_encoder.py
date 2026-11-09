"""Chromosome-aware Conv1d autoencoder for peak-ordered ATAC features.

:class:`SpatialModalityAutoencoder` has the interface of
:class:`mtvae.nn.autoencoders.ModalityAutoencoder` and adds a residual
convolutional pre-encoder that smooths each peak with its genomic neighbours:

- encoder: :class:`SpatialResidualBlock` (one or more parallel 1-channel
  convolutions, summed, added to the input) followed by the MLP
  ``input_dim -> 512 -> 256 -> 128 -> latent_dim``;
- decoder: the mirrored MLP, with no convolution, so reconstruction stays
  per-peak.

The features must be peaks sorted by chromosome and then by start position.
A kernel that spans a chromosome boundary would mix unrelated peaks, so
:class:`ChromosomeAwareConv1d` zeroes every output position whose window
crosses a boundary. Passing ``chr_ids=None`` gives the same architecture with
an all-ones mask (the ``"conv"`` trunk kind of :mod:`mtvae.nn.trunks`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def load_chromosome_ids(peak_list_path: Path | str) -> np.ndarray:
    """Read a peak list and return the chromosome of every peak.

    Parameters
    ----------
    peak_list_path : pathlib.Path or str
        Text file with one peak per line, formatted ``chrom:start-end``.

    Returns
    -------
    numpy.ndarray of str, shape (n_peaks,)
        The ``chrom`` field of every line, in file order.
    """
    return np.array([line.strip().split(":", 1)[0]
                     for line in open(peak_list_path)])


# The boundary mask affects few positions: on the PFC peak list (5,000 peaks,
# 22 chromosomes) kernel 3 masks 42 positions, kernel 5 masks 84 and kernel 9
# masks 168. ``chr_ids=None`` (the "conv" trunk) has identical parameter shapes
# and RNG consumption, so the two variants are directly comparable at a fixed
# seed.


def build_valid_kernel_mask(chr_ids: np.ndarray, kernel_size: int) -> np.ndarray:
    """Mark the positions whose convolution window stays on one chromosome.

    A kernel of size ``k`` centred at position ``i`` covers
    ``[i - k // 2, i + k // 2]`` (clipped to the array).

    Parameters
    ----------
    chr_ids : numpy.ndarray, shape (n_peaks,)
        Chromosome of every peak, in feature order.
    kernel_size : int
        Convolution kernel width.

    Returns
    -------
    numpy.ndarray of bool, shape (n_peaks,)
        ``True`` where the window does not cross a chromosome boundary.
    """
    n = len(chr_ids)
    half = kernel_size // 2
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if not all(chr_ids[j] == chr_ids[i] for j in range(lo, hi)):
            mask[i] = False
    return mask


class ChromosomeAwareConv1d(nn.Module):
    """1-D convolution with hard masking at chromosome boundaries.

    A ``nn.Conv1d`` with ``padding=kernel_size // 2`` (same-length output)
    whose output is multiplied by a fixed mask that zeroes every position
    where the kernel crosses a chromosome boundary.

    Parameters
    ----------
    in_channels, out_channels, kernel_size : int
        As for ``torch.nn.Conv1d``.
    chr_ids : numpy.ndarray or None
        Chromosome of every feature. ``None`` gives an all-ones mask (a plain
        convolution).
    n_features : int, optional
        Number of features; required when ``chr_ids`` is ``None``.

    Raises
    ------
    ValueError
        If ``chr_ids`` and ``n_features`` are both ``None``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 chr_ids: np.ndarray | None, n_features: int | None = None):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        # chr_ids=None is the plain conv: an all-ones mask. It shares this code
        # path and buffer shape, so the two variants differ only in the mask.
        if chr_ids is None:
            if n_features is None:
                raise ValueError("plain conv needs n_features for the mask shape")
            mask = np.ones(int(n_features), dtype=bool)
        else:
            mask = build_valid_kernel_mask(chr_ids, kernel_size)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("valid_mask",
                             torch.tensor(mask, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``(B, C_in, L)`` to ``(B, C_out, L)`` and apply the mask."""
        h = self.conv(x)                 # (B, C_out, L)
        return h * self.valid_mask       # broadcast (L,) → (B, C_out, L)


class SpatialResidualBlock(nn.Module):
    """Chromosome-aware residual smoothing block, 1 channel in and out.

    Computes ``x + residual_scale * sum_k conv_k(x)`` over one or more
    parallel :class:`ChromosomeAwareConv1d` layers of different kernel sizes.
    Every convolution is zero-initialised, so the block starts as the
    identity.

    Parameters
    ----------
    chr_ids : numpy.ndarray of shape (n_peaks,) or None
        Chromosome of every peak; ``None`` for an all-ones mask.
    kernel_sizes : int or sequence of int, default 5
        One kernel size, or several for parallel multi-scale convolutions.
    residual_scale : float, default 1.0
        Multiplier on the convolution output before it is added to the input.
    n_features : int, optional
        Number of features; required when ``chr_ids`` is ``None``.
    """

    def __init__(self, chr_ids: np.ndarray | None, kernel_sizes=5,
                 residual_scale: float = 1.0, n_features: int | None = None):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = list(kernel_sizes)
        self.residual_scale = residual_scale
        self.convs = nn.ModuleList([
            ChromosomeAwareConv1d(1, 1, k, chr_ids, n_features)
            for k in self.kernel_sizes
        ])
        # Zero-init every conv so residual block starts as identity.
        for c in self.convs:
            nn.init.zeros_(c.conv.weight)
            nn.init.zeros_(c.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth ``(B, n_peaks)`` features; the output has the same shape."""
        x1 = x.unsqueeze(1)
        h = sum(c(x1) for c in self.convs).squeeze(1)
        return x + self.residual_scale * h


class _MLP(nn.Module):
    """SiLU MLP with a linear output; same layout as the one in
    :mod:`mtvae.nn.autoencoders`."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialModalityAutoencoder(nn.Module):
    """ATAC autoencoder with a chromosome-aware convolutional pre-encoder.

    Same interface as :class:`mtvae.nn.autoencoders.ModalityAutoencoder`:
    ``encode(x)`` returns ``(B, latent_dim)``, ``decode(z)`` returns
    ``(B, input_dim)``, ``forward(x)`` returns ``(x_rec, z)``, and the
    ``encoder`` and ``decoder`` attributes hold the MLPs. The spatial block
    (``spatial``) runs before the encoder MLP; the decoder has no spatial
    component.

    Parameters
    ----------
    input_dim : int
        Number of peaks.
    latent_dim : int
        Code dimension.
    chr_ids : numpy.ndarray or None
        Chromosome of every peak, length ``input_dim``; ``None`` for a plain
        convolution (all-ones mask).
    encoder_hidden : sequence of int, default (512, 256, 128)
        Encoder MLP hidden widths.
    decoder_hidden : sequence of int, default (128, 256, 512)
        Decoder MLP hidden widths.
    spatial_kernel_sizes : int or sequence of int, default 5
        Kernel size(s) of the spatial block.

    Raises
    ------
    ValueError
        If ``chr_ids`` is given and its length differs from ``input_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        chr_ids: np.ndarray | None,
        encoder_hidden: Sequence[int] = (512, 256, 128),
        decoder_hidden: Sequence[int] = (128, 256, 512),
        spatial_kernel_sizes=5,
    ):
        super().__init__()
        # chr_ids=None is the plain conv (all-ones mask).
        if chr_ids is not None and input_dim != len(chr_ids):
            raise ValueError(
                f"input_dim {input_dim} != len(chr_ids) {len(chr_ids)}; the peak "
                f"list does not match the matrix")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.spatial = SpatialResidualBlock(
            chr_ids=chr_ids, kernel_sizes=spatial_kernel_sizes,
            n_features=input_dim,
        )
        self.encoder = _MLP(
            in_dim=input_dim, hidden_dims=list(encoder_hidden),
            out_dim=latent_dim,
        )
        self.decoder = _MLP(
            in_dim=latent_dim, hidden_dims=list(decoder_hidden),
            out_dim=input_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial block then encoder MLP: ``(B, input_dim) -> (B, latent_dim)``."""
        return self.encoder(self.spatial(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder MLP: ``(B, latent_dim) -> (B, input_dim)``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_rec, z)``."""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
