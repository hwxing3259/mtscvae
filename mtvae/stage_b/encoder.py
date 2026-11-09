"""The trainable Stage-A encoder path.

:class:`StageAEncoder` wraps the two per-modality trunks and the fusion head
and computes the Stage-A latent

    z = fusion.posterior(ae.encode(x), modality)[0]

(the posterior mean), so Stage B can fine-tune the encoder.
:func:`verify_reproduces_stage_a` checks, before Stage B starts, that a rebuilt
encoder reproduces the Stage-A latent it is anchored to: Stage B moves the
encoder, and the latent anchor is only meaningful against the right map.
Checkpoints are read by :mod:`mtvae.stage_b.merged_loader`.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn


from mtvae.stage_a.model import StochasticFusion  # noqa: E402

log = logging.getLogger("mtvae.stage_b.encoder")


class StageAEncoder(nn.Module):
    """The Stage-A map from features to the fused posterior mean, made trainable.

    Parameters
    ----------
    rna_ae, atac_ae : torch.nn.Module
        Per-modality trunks, weights already loaded.
    fusion : StochasticFusion
        Shared bottleneck, mean branch loaded.

    Notes
    -----
    Mirrors :class:`mtvae.stage_a.model.MergedVAE` without the decoders. Stage
    B's decoders are separate instances, so a co-adapting encoder cannot share
    weights with them.
    """

    def __init__(self, rna_ae: nn.Module, atac_ae: nn.Module,
                 fusion: StochasticFusion) -> None:
        super().__init__()
        self.rna_ae = rna_ae
        self.atac_ae = atac_ae
        self.fusion = fusion

    def trunk(self, modality: str) -> nn.Module:
        """Return the trunk for one modality.

        Raises
        ------
        ValueError
            If ``modality`` is not ``"rna"`` or ``"atac"``.
        """
        if modality == "rna":
            return self.rna_ae
        if modality == "atac":
            return self.atac_ae
        raise ValueError(f"unknown modality {modality!r}")

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Encode a feature batch to the fused posterior **mean**."""
        h = self.trunk(modality).encode(x)
        return self.fusion.posterior(h, modality)[0]

    @torch.no_grad()
    def encode_all(self, mat: np.ndarray, modality: str, *, device: str = "cpu",
                   chunk: int = 4096) -> np.ndarray:
        """Encode a full matrix in chunks of ``chunk`` rows.

        Runs in eval mode (restoring the previous mode afterwards), as
        :func:`mtvae.stage_a.trainer.extract_latents` does.

        Parameters
        ----------
        mat : numpy.ndarray, shape (n, n_features)
            Feature matrix.
        modality : {"rna", "atac"}
            Which trunk to use.
        device : str, default "cpu"
            Device on which to run the chunks.
        chunk : int, default 4096
            Rows per forward pass.

        Returns
        -------
        numpy.ndarray, shape (n, latent_dim)
            Posterior means.
        """
        was_training = self.training
        self.eval()
        out = []
        for i in range(0, len(mat), chunk):
            xb = torch.from_numpy(
                np.asarray(mat[i:i + chunk], dtype=np.float32)).to(device)
            out.append(self(xb, modality).cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(out, axis=0)


def verify_reproduces_stage_a(enc: StageAEncoder, x: np.ndarray,
                              z_ref: np.ndarray, modality: str, *,
                              device: str = "cpu", n: int = 4096,
                              atol: float = 1e-4) -> tuple[bool, dict]:
    """Check that a rebuilt encoder reproduces the Stage-A latent.

    Run this before Stage B: a wrongly loaded encoder still runs (see the
    module docstring), and Stage B anchors the encoder to ``z_ref``.

    Parameters
    ----------
    enc : StageAEncoder
        Freshly loaded, before any training.
    x : numpy.ndarray
        Feature matrix for ``modality``, at the clip Stage A encoded.
    z_ref : numpy.ndarray
        The Stage-A latent (posterior means) of the same cells.
    modality : {"rna", "atac"}
        Which trunk to check.
    device : str, default "cpu"
        Device on which to encode.
    n : int, default 4096
        Cells to check. The map is per-cell, so a prefix is sufficient.
    atol : float, default 1e-4
        Tolerance on the maximum absolute difference. Not bit-exact, because
        the reference may have been computed on another device. A correct
        load agrees to about 1e-6; a wrong fusion scale misses by order 1.

    Returns
    -------
    ok : bool
        Whether the maximum absolute difference is within ``atol``.
    detail : dict
        ``max_abs``, ``mean_abs``, ``corr`` and ``n_checked``, or ``error`` on
        a shape mismatch.
    """
    got = enc.encode_all(np.asarray(x[:n]), modality, device=device)
    ref = np.asarray(z_ref[:n], dtype=np.float32)
    if got.shape != ref.shape:
        return False, {"error": f"shape {got.shape} vs {ref.shape}"}
    d = np.abs(got - ref)
    corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
    detail = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
              "corr": corr, "n_checked": int(len(ref))}
    return bool(d.max() <= atol), detail
