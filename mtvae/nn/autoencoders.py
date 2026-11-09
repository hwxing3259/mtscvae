"""Per-modality MLP autoencoder.

:class:`ModalityAutoencoder` is the default encoder/decoder trunk for one
modality:

- encoder ``input_dim -> 512 -> 256 -> 128 -> latent_dim`` with SiLU
  activations and no batch norm;
- decoder ``latent_dim -> 128 -> 256 -> 512 -> input_dim`` with a linear
  output.

It is deterministic (no KL term of its own); the Stage-A VAE adds a stochastic
fusion layer on top of the encoder output.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class _MLP(nn.Module):
    """Simple SiLU-activated MLP with linear output.

    Parameters
    ----------
    in_dim : int
        Input dimensionality.
    hidden_dims : sequence of int
        Hidden-layer widths.
    out_dim : int
        Output dimensionality.
    final_activation : torch.nn.Module, optional
        Applied to the output layer; ``None`` for a linear output.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        final_activation: nn.Module | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityAutoencoder(nn.Module):
    """MLP autoencoder for one modality.

    Parameters
    ----------
    input_dim : int
        Number of features of the modality.
    latent_dim : int
        Dimension of the encoder output (the shared latent dimension,
        ``mtvae._constants.LATENT_DIM`` = 50 in both examples).
    encoder_hidden : sequence of int, default (512, 256, 128)
        Encoder hidden widths.
    decoder_hidden : sequence of int, default (128, 256, 512)
        Decoder hidden widths.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int] = (512, 256, 128),
        decoder_hidden: Sequence[int] = (128, 256, 512),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = _MLP(
            in_dim=input_dim, hidden_dims=list(encoder_hidden), out_dim=latent_dim,
        )
        self.decoder = _MLP(
            in_dim=latent_dim, hidden_dims=list(decoder_hidden), out_dim=input_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map features ``(n, input_dim)`` to codes ``(n, latent_dim)``."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map codes ``(n, latent_dim)`` to reconstructions ``(n, input_dim)``."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode.

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)

        Returns
        -------
        x_rec : torch.Tensor, shape (n, input_dim)
            Reconstruction.
        z : torch.Tensor, shape (n, latent_dim)
            Code.
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
