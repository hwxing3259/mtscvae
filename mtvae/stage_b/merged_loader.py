"""Rebuild the Stage-A encoder and decoders from a merged checkpoint.

The merged checkpoint format stores one ``model_state_dict`` for the whole
:class:`~mtvae.stage_a.model.MergedVAE`, together with ``config`` (the
resolved Stage-A settings) and optional ``dataset``, ``arm``, ``seed`` and
``stages`` entries. :func:`load_stage_a_encoder_merged` reads it.

It differs from the PFC ``phase2_fusion`` format read by
:mod:`mtvae.stage_b.encoder` in one important way: here the ``fusion.*`` slice
of the state dict is the whole
:class:`~mtvae.stage_a.model.StochasticFusion` (``logvar_net`` included) and
is loaded into ``fusion``, not into ``fusion.mu_net``. All loads are strict,
since a non-strict load could leave the mean branch at its random
initialisation and still produce a plausible latent. As in the PFC format,
``residual_scale`` is a plain attribute rather than a parameter, so every
fusion field is read from ``config`` and never defaulted.

The decoders are loaded into separate module instances from the same tensors.
:class:`~mtvae.stage_b.encoder.StageAEncoder` holds the full autoencoders, so
borrowing ``enc.rna_ae.decoder`` would make the decoders share parameters with
the encoder being trained; :func:`decoders_are_independent` checks this.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


from mtvae.stage_b.encoder import (  # noqa: E402
    StageAEncoder,
)
from mtvae.stage_a.model import StochasticFusion  # noqa: E402

log = logging.getLogger("mtvae.stage_b.merged_loader")

#: Fusion fields that must come from the checkpoint. A wrong ``residual_scale``
#: fails silently; defaulting any of them builds a different model.
_FUSION_KEYS = ("fusion_arch", "fusion_hidden_dim", "fusion_depth",
                "fusion_residual_scale", "fusion_layer_norm",
                "sigma_mode", "sigma_init")


#: ATAC clip of the four-dataset example under its earlier convention
#: (``mtvae.adapters.LEGACY_FOURDS_ATAC_CLIP``); RNA is used as prepared
#: (already within +/-10). Stage B must feed the encoder features clipped
#: exactly as Stage A saw them: a different clip moves the encoder off its
#: training distribution for the extreme cells.
ATAC_ENCODE_CLIP = 5.0


def load_features(staged: Path) -> dict[str, np.ndarray]:
    """Load four-dataset feature matrices with the :data:`ATAC_ENCODE_CLIP` clip.

    Parameters
    ----------
    staged : pathlib.Path
        Directory holding ``rna_expr.npy`` and ``atac_expr.npy``.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{"rna": ..., "atac": ...}``, float32, ATAC clipped to
        +/-:data:`ATAC_ENCODE_CLIP`.
    """
    rna = np.asarray(np.load(staged / "rna_expr.npy", mmap_mode="r"),
                     dtype=np.float32)
    atac = np.clip(
        np.asarray(np.load(staged / "atac_expr.npy", mmap_mode="r"),
                   dtype=np.float32),
        -ATAC_ENCODE_CLIP, ATAC_ENCODE_CLIP)
    return {"rna": rna, "atac": atac}


def _input_dims_from(sd: dict[str, torch.Tensor]) -> dict[str, int]:
    """Per-modality feature count, read off the checkpoint's own tensors.

    The first encoder Linear has weight shape ``(hidden, n_features)``, so the
    feature count is read from the weight itself.

    Parameters
    ----------
    sd : dict of str to torch.Tensor
        A ``MergedVAE`` state dict.

    Returns
    -------
    dict of str to int
        ``{"rna": ..., "atac": ...}``.

    Raises
    ------
    KeyError
        If a trunk's first encoder weight is absent, which means this is not the
        checkpoint shape this loader handles.
    """
    out = {}
    for mod in ("rna", "atac"):
        key = f"{mod}_ae.encoder.net.0.weight"
        if key not in sd:
            raise KeyError(
                f"{key} missing; cannot read the {mod} feature count from this "
                f"checkpoint. Keys start: {sorted(sd)[:4]}")
        out[mod] = int(sd[key].shape[1])
    return out


def _split(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Take the ``prefix.*`` slice of a state dict, with the prefix removed.

    Parameters
    ----------
    sd : dict of str to torch.Tensor
        State dict.
    prefix : str
        Key prefix without the trailing dot, e.g. ``"rna_ae"``.

    Returns
    -------
    dict of str to torch.Tensor
    """
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def trunk_spec(meta: dict[str, Any] | None) -> Any:
    """The :class:`~mtvae.nn.trunks.TrunkSpec` a checkpoint's trunk entry describes.

    Parameters
    ----------
    meta : dict or None
        ``{"kind", "kernel_sizes", "chr_ids"}`` as written by
        :func:`trunk_meta`; ``None`` (checkpoints without the entry) is an MLP.

    Returns
    -------
    TrunkSpec
    """
    from mtvae.nn.trunks import TrunkSpec
    if not meta:
        return TrunkSpec("mlp")
    chr_ids = meta.get("chr_ids")
    return TrunkSpec(meta["kind"],
                     chr_ids=None if chr_ids is None else np.asarray(chr_ids),
                     kernel_sizes=meta.get("kernel_sizes", 3))


def trunk_meta(spec: Any) -> dict[str, Any]:
    """The checkpoint entry describing one trunk (inverse of :func:`trunk_spec`).

    Parameters
    ----------
    spec : TrunkSpec

    Returns
    -------
    dict
        ``{"kind", "kernel_sizes", "chr_ids"}``; ``chr_ids`` is an int array or
        None, so a checkpoint rebuilds a chromosome-aware trunk on its own.
    """
    return {"kind": spec.kind, "kernel_sizes": spec.kernel_sizes,
            "chr_ids": None if spec.chr_ids is None else np.asarray(spec.chr_ids)}


def build_encoder(c: dict[str, Any], trunks: dict[str, Any], dims: dict[str, int],
                  latent_dim: int) -> StageAEncoder:
    """Construct an untrained Stage-A encoder: RNA trunk, ATAC trunk, fusion.

    The construction order is that of the Stage-B RNG burn
    (:func:`mtvae.stage_b.handoff._burn_staged_prefix`), so loading a
    checkpoint and training Stage A in the same run leave the random stream at
    the same position before the SDE is built.

    Parameters
    ----------
    c : dict
        Stage-A config (the fusion fields).
    trunks : dict
        Per-slot trunk entries (see :func:`trunk_meta`); missing slots are MLPs.
    dims : dict
        Feature count per slot.
    latent_dim : int

    Returns
    -------
    StageAEncoder
    """
    from mtvae.nn.trunks import build
    rna_ae = build(trunk_spec(trunks.get("rna")), input_dim=dims["rna"],
                   latent_dim=latent_dim)
    atac_ae = build(trunk_spec(trunks.get("atac")), input_dim=dims["atac"],
                    latent_dim=latent_dim)
    fusion = StochasticFusion(
        latent_dim=latent_dim,
        hidden_dim=int(c["fusion_hidden_dim"]),
        depth=int(c["fusion_depth"]),
        residual_scale=float(c["fusion_residual_scale"]),
        layer_norm=bool(c["fusion_layer_norm"]),
        arch=str(c["fusion_arch"]),
        sigma_mode=str(c["sigma_mode"]),
        sigma_init=float(c["sigma_init"]),
        stochastic_dims=int(c.get("stochastic_dims", 0)),
    )
    return StageAEncoder(rna_ae, atac_ae, fusion)


def build_decoders(parts: dict[str, dict[str, torch.Tensor]], dims: dict[str, int],
                   latent_dim: int, *, device: str = "cpu",
                   frozen: bool = True) -> dict[str, nn.Module]:
    """Decoders as separate instances, loaded from per-slot autoencoder weights.

    Each is the decoder of a fresh MLP autoencoder; encoder-only ``spatial.*``
    weights of a chromosome-aware trunk are dropped first (the decoder is the
    same MLP for every trunk kind).

    Parameters
    ----------
    parts : dict
        Slot -> autoencoder state dict (``encoder.*``, ``decoder.*``, and
        ``spatial.*`` for a spatial trunk).
    dims : dict
        Feature count per slot.
    latent_dim : int
    device : str, optional
    frozen : bool, optional
        Put each decoder in eval mode with gradients off.

    Returns
    -------
    dict
        Slot -> decoder.
    """
    from mtvae.nn.autoencoders import ModalityAutoencoder
    out: dict[str, nn.Module] = {}
    for mod in ("rna", "atac"):
        ae = ModalityAutoencoder(input_dim=dims[mod], latent_dim=latent_dim)
        ae.load_state_dict({k: v for k, v in parts[mod].items()
                            if not k.startswith("spatial.")}, strict=True)
        dec = ae.decoder.to(device)
        if frozen:
            dec.eval()
            for p in dec.parameters():
                p.requires_grad = False
        out[mod] = dec
    return out


def load_stage_a_encoder_merged(
    ckpt_path: Path, *, device: str = "cpu",
) -> tuple[StageAEncoder, dict[str, nn.Module], dict[str, Any]]:
    """Rebuild the encoder and frozen decoders from a merged checkpoint.

    Feature and latent dimensions are read from the checkpoint (tensor shapes
    and ``config["latent_dim"]``), so datasets of different sizes can be
    loaded in one process.

    Parameters
    ----------
    ckpt_path : pathlib.Path
        A merged checkpoint.
    device : str, default "cpu"
        Where to place the modules.

    Returns
    -------
    encoder : StageAEncoder
        Trainable Stage-A path, weights loaded.
    decoders : dict of str to torch.nn.Module
        ``{"rna": ..., "atac": ...}``, separate instances, eval mode, grads off.
    meta : dict
        ``{"config", "dataset", "arm", "seed", "stages", "latent_dim"}``.

    Raises
    ------
    KeyError
        If the checkpoint lacks ``model_state_dict``, ``config``, any fusion
        field or ``latent_dim``; none of them is defaulted.
    """

    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    if "model_state_dict" not in ck:
        raise KeyError(
            f"{ckpt_path} has no 'model_state_dict'; keys = {sorted(ck)}. "
            f"A PFC phase2_fusion checkpoint is read by "
            f"mtvae.stage_b.encoder.load_stage_a_encoder instead.")
    if "config" not in ck:
        raise KeyError(f"{ckpt_path} has no 'config' block; fusion geometry "
                       f"cannot be defaulted (see module docstring).")

    c: dict[str, Any] = ck["config"]
    missing = [k for k in _FUSION_KEYS if k not in c]
    if missing:
        raise KeyError(
            f"{ckpt_path} config is missing {missing}. Defaulting fusion fields "
            f"builds a different model that loads without error.")

    sd = ck["model_state_dict"]
    # The latent width comes from the checkpoint, never from a package default.
    if "latent_dim" not in c:
        raise KeyError(
            f"{ckpt_path} config has no 'latent_dim'. It is not defaultable: the "
            f"latent width decides the fusion shape and, downstream, where the "
            f"SDE initialises.")
    latent_dim = int(c["latent_dim"])

    # Feature dimensions come from the checkpoint's own tensors, so the loader
    # holds no per-dataset state; a truncated or mismatched checkpoint still
    # fails the strict loads below.
    dims = _input_dims_from(sd)
    trunks = ck.get("trunks") or {}
    enc = build_encoder(c, trunks, dims, latent_dim)
    # Strict: a partial load would fail silently.
    enc.rna_ae.load_state_dict(_split(sd, "rna_ae"), strict=True)
    enc.atac_ae.load_state_dict(_split(sd, "atac_ae"), strict=True)
    # Into ``fusion``, not ``fusion.mu_net``: this slice includes logvar_net.
    enc.fusion.load_state_dict(_split(sd, "fusion"), strict=True)
    enc = enc.to(device)
    decoders = build_decoders({m: _split(sd, f"{m}_ae") for m in ("rna", "atac")},
                              dims, latent_dim, device=device)

    meta = {"config": c, "trunks": trunks, "dataset": ck.get("dataset"),
            "arm": ck.get("arm"), "seed": ck.get("seed"),
            "stages": ck.get("stages"), "latent_dim": latent_dim}
    log.info("Stage-A encoder rebuilt from %s (dataset=%s arm=%s seed=%s, "
             "fusion arch=%s residual_scale=%s, latent_dim=%d, %d stages)",
             ckpt_path.name, meta["dataset"], meta["arm"], meta["seed"],
             c["fusion_arch"], c["fusion_residual_scale"], latent_dim,
             len(meta["stages"] or []))
    return enc, decoders, meta


def decoders_are_independent(enc: StageAEncoder,
                             decoders: dict[str, nn.Module]) -> tuple[bool, str]:
    """Check that the decoders do not share parameters with the encoder.

    Compares the decoders against the decoders inside the encoder's trunks.
    Sharing would let the encoder's updates move the decoders, with no error
    and a plausible loss curve.

    Parameters
    ----------
    enc : StageAEncoder
        The encoder path.
    decoders : dict of str to torch.nn.Module
        ``{"rna": ..., "atac": ...}``.

    Returns
    -------
    tuple of (bool, str)
        Verdict and a human-readable detail line.
    """
    bad = []
    for mod in ("rna", "atac"):
        own = {id(p) for p in enc.trunk(mod).decoder.parameters()}
        shared = [n for n, p in decoders[mod].named_parameters() if id(p) in own]
        if shared:
            bad.append(f"{mod}: {len(shared)} shared tensor(s), e.g. {shared[0]}")
    if bad:
        return False, "; ".join(bad)
    return True, "decoders are separate instances from the encoder trunks"
