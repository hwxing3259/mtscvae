"""In-memory handoff from Stage A to Stage B.

:func:`build_handoff` takes a trained :class:`~mtvae.stage_a.model.MergedVAE`
and returns what Stage B needs: a trainable
:class:`~mtvae.stage_b.encoder.StageAEncoder` that wraps the live trunks and
fusion (so co-adaptation updates the model in place), two decoders, and
metadata for the run record. It is the in-memory equivalent of saving Stage
A and reloading it with :mod:`mtvae.stage_b.merged_loader`, and takes care of
three things that the round trip does implicitly.

1. Decoders are re-instantiated, never borrowed.
   ``MergedVAE.rna_ae`` holds the encoder and the decoder of that modality,
   so ``model.rna_ae.decoder`` shares tensors with the encoder Stage B
   trains. Borrowed decoders would make ``torch.optim.Adam`` reject
   parameters that appear in two groups, or, if that were bypassed, let the
   encoder's updates move the decoders. The decoders are therefore loaded
   into fresh modules from the model's state dict, and
   :func:`~mtvae.stage_b.merged_loader.decoders_are_independent` is asserted.

2. Dimensions are checked explicitly.
   With no ``load_state_dict(strict=True)``, a wrong latent width would not
   raise. Feature counts are read off the live trunks, the latent width is
   checked against the fusion output, and both are returned in ``meta``.

3. The RNG stream position is a declared choice.
   In the original two-script pipeline the Stage-B script seeds torch,
   rebuilds the Stage-A encoder from the checkpoint (RNA trunk, ATAC trunk,
   fusion), builds two autoencoders for the decoders, and only then
   constructs the :class:`~mtvae.nn.sde.SDESampler`. The SDE's initial
   weights, and every random draw of the training loop after them, depend on
   that sequence. The in-memory handoff needs only the decoder autoencoders,
   so ``rng_protocol`` (no default) states which stream to use:

   ``"reseed"``
       Re-seed torch and numpy from the run seed, then build only what is
       needed. Not bit-equal to the two-script pipeline: a different draw
       from the same seed.
   ``"staged_equivalent"``
       Re-seed, then construct and discard the modules the two-script
       pipeline builds before its decoders, so ``SDESampler`` (constructed by
       the caller after the handoff) initialises at the same stream position
       and the fit is bit-equal to the two-script pipeline on the same
       machine.

   ``stage_b`` names which Stage-B construction sequence to replay; see
   :data:`STAGE_B_KINDS`.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn


from mtvae.stage_b.encoder import (  # noqa: E402
    StageAEncoder,
)
from mtvae.stage_b.merged_loader import (  # noqa: E402
    _split, decoders_are_independent,
)
from mtvae.nn.autoencoders import (  # noqa: E402
    ModalityAutoencoder,
)

log = logging.getLogger("mtvae.stage_b.handoff")

#: The protocols ``build_handoff`` accepts. There is no default: a caller must
#: state which stream they intend (see the module docstring).
RNG_PROTOCOLS = ("reseed", "staged_equivalent")

#: Which Stage-B construction sequence the ``staged_equivalent`` burn replays.
#:
#:   "coadapt"  The co-adapt Stage B (:mod:`mtvae.stage_b.trainer`). Its
#:              loader builds two encoder trunks and the fusion, then two
#:              autoencoders for the decoders. The handoff builds the last two
#:              itself, so it burns three modules.
#:   "decoder"  The frozen-decoder variant of the original pipeline, whose
#:              loader builds only the two decoder autoencoders: no burn.
#:
#: Applying the wrong burn gives a different fit from the same seed.
STAGE_B_KINDS = ("coadapt", "decoder")

#: How many modules ``staged_equivalent`` constructs and discards, per kind.
_BURN_MODULES = {"coadapt": 3, "decoder": 0}


def reseed(seed: int) -> None:
    """Re-seed the global torch and numpy streams from ``seed``.

    The second seeding of a fit, which makes Stage B independent of what
    Stage A drew (Stage A seeds both streams itself in the same way).

    Parameters
    ----------
    seed : int
        The run seed; both stages use the same one.

    Notes
    -----
    The numpy global seed matters: openTSNE's ``PerplexityBasedNN`` takes no
    ``random_state`` and draws from numpy's global stream, and both examples
    use ``topo_affinity="full"``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))


def _burn_staged_prefix(input_dims: dict[str, int], latent_dim: int,
                        fusion_cfg: dict[str, Any], *,
                        atac_trunk: str = "mlp",
                        chr_ids: np.ndarray | None = None,
                        spatial_kernels: int = 3) -> int:
    """Construct and discard what the checkpoint loader would have built.

    :func:`~mtvae.stage_b.merged_loader.load_stage_a_encoder_merged` and
    :func:`~mtvae.stage_b.encoder.load_stage_a_encoder` build an RNA trunk, an
    ATAC trunk and the fusion before the two autoencoders whose decoders they
    keep. The handoff uses the live modules for the first three, so those
    constructions, and the torch RNG they consume, would not happen. This
    replays them, which makes ``staged_equivalent`` bit-exact.

    Parameters
    ----------
    input_dims : dict of str to int
        ``{"rna": ..., "atac": ...}`` feature counts.
    latent_dim : int
        Latent width.
    fusion_cfg : dict
        The fusion fields of the resolved Stage-A config (``fusion_arch``,
        ``fusion_hidden_dim``, ``fusion_depth``, ``fusion_residual_scale``,
        ``fusion_layer_norm``, ``sigma_mode``, ``sigma_init``, and optionally
        ``kl_dims``).
    atac_trunk : {"mlp", "spatial", "conv"}, default "mlp"
        ``"mlp"`` for the four-dataset example, ``"spatial"`` for PFC. The
        Conv1d trunks have different parameter shapes from the MLP and consume
        a different amount of RNG, so a wrong value gives a different fit.
    chr_ids : numpy.ndarray, optional
        Per-feature chromosome index. Required when ``atac_trunk="spatial"``;
        its length must equal ``input_dims["atac"]``.
    spatial_kernels : int, default 3
        Conv kernel width for the spatial trunk. Must match Stage A.

    Returns
    -------
    int
        Number of modules constructed and discarded, for the run record.

    Raises
    ------
    ValueError
        If a spatial burn is requested without usable ``chr_ids``.

    Notes
    -----
    The order mirrors the loaders and determines the stream position: RNA
    trunk, ATAC trunk, fusion. Reordering modules of the same size would not
    change the number of draws but would change every weight.
    """
    from mtvae.nn.trunks import TrunkSpec, build
    from mtvae.stage_a.model import StochasticFusion

    ModalityAutoencoder(input_dim=input_dims["rna"], latent_dim=latent_dim)
    # Dispatch on the actual trunk kind. "conv" and "spatial" have identical
    # parameter shapes, so they burn identically and stay comparable at a fixed
    # seed; only "mlp" differs.
    if atac_trunk in ("spatial", "conv"):
        if atac_trunk == "spatial" and (chr_ids is None
                                        or len(chr_ids) != input_dims["atac"]):
            raise ValueError(
                f"spatial burn needs chr_ids of length {input_dims['atac']}, got "
                f"{None if chr_ids is None else len(chr_ids)}. Without the right "
                f"trunk shape the burn lands on the wrong stream position and "
                f"staged_equivalent silently stops being equivalent.")
        spec = (TrunkSpec("spatial", chr_ids=chr_ids, kernel_sizes=spatial_kernels)
                if atac_trunk == "spatial"
                else TrunkSpec("conv", kernel_sizes=spatial_kernels))
        build(spec, input_dim=input_dims["atac"], latent_dim=latent_dim)
    else:
        ModalityAutoencoder(input_dim=input_dims["atac"], latent_dim=latent_dim)
    StochasticFusion(
        latent_dim=latent_dim,
        hidden_dim=int(fusion_cfg["fusion_hidden_dim"]),
        depth=int(fusion_cfg["fusion_depth"]),
        residual_scale=float(fusion_cfg["fusion_residual_scale"]),
        layer_norm=bool(fusion_cfg["fusion_layer_norm"]),
        arch=str(fusion_cfg["fusion_arch"]),
        sigma_mode=str(fusion_cfg["sigma_mode"]),
        sigma_init=float(fusion_cfg["sigma_init"]),
        stochastic_dims=int(fusion_cfg.get("kl_dims", 0)),
    )
    return 3


def build_handoff(model: nn.Module, cfg: Any, *, seed: int,
                  rng_protocol: str, stage_b: str = "coadapt",
                  device: str = "cpu",
                  atac_trunk: str = "mlp",
                  chr_ids: np.ndarray | None = None,
                  spatial_kernels: int = 3,
                  ) -> tuple[StageAEncoder, dict[str, nn.Module], dict[str, Any]]:
    """Hand a trained Stage-A model to Stage B, in memory.

    Parameters
    ----------
    model : torch.nn.Module
        The live ``MergedVAE`` returned by
        :func:`mtvae.stage_a.trainer.train_merged`. Its trunks and fusion
        become Stage B's trainable encoder, so co-adaptation updates this
        object in place. Latents already extracted are numpy copies and are
        unaffected.
    cfg : MergedVAEConfig
        The resolved Stage-A config; supplies the fusion settings and the
        latent width, which is checked against the live fusion.
    seed : int
        The run seed, used for the re-seed.
    rng_protocol : {"reseed", "staged_equivalent"}
        See the module docstring. No default.
    stage_b : {"coadapt", "decoder"}, default "coadapt"
        Which Stage-B construction sequence the burn replays; see
        :data:`STAGE_B_KINDS`.
    device : str, default "cpu"
        Where the encoder and decoders are placed.
    atac_trunk : {"mlp", "spatial", "conv"}, default "mlp"
        Kind of the trained ATAC trunk, for the burn.
    chr_ids : numpy.ndarray, optional
        Chromosome ids of a spatial ATAC trunk, for the burn.
    spatial_kernels : int, default 3
        Kernel width of a spatial or conv ATAC trunk, for the burn.

    Returns
    -------
    encoder : StageAEncoder
        Trainable Stage-A path wrapping the live trunks and fusion.
    decoders : dict of str to torch.nn.Module
        ``{"rna": ..., "atac": ...}``, freshly instantiated, in eval mode,
        gradients off, sharing no storage with ``encoder``.
    meta : dict
        ``{"latent_dim", "input_dims", "rng_protocol", "stage_b",
        "modules_burned", "decoder_independence"}``, for the run record.

    Raises
    ------
    ValueError
        If ``rng_protocol`` or ``stage_b`` is not recognised, if the latent
        width of the live fusion disagrees with ``cfg.latent_dim``, or if a
        spatial burn lacks usable ``chr_ids``, or a trunk carries decoder
        weights under a ``spatial.`` prefix.
    SystemExit
        If the decoders share storage with the encoder (impossible by
        construction; checked because the failure would otherwise be silent).
    """
    if rng_protocol not in RNG_PROTOCOLS:
        raise ValueError(
            f"rng_protocol must be one of {RNG_PROTOCOLS}, got {rng_protocol!r}. "
            f"It has no default on purpose: the two protocols place SDESampler at "
            f"different positions in the torch stream and therefore give "
            f"different fits from the same seed.")
    if stage_b not in STAGE_B_KINDS:
        raise ValueError(
            f"stage_b must be one of {STAGE_B_KINDS}, got {stage_b!r}. The two "
            f"Stage-B trainers have different staged construction prefixes, so "
            f"the burn is not one constant -- see STAGE_B_KINDS.")

    # --- dimensions, read off the live modules, then checked -----------------
    input_dims = {"rna": int(model.rna_ae.encoder.net[0].in_features),
                  "atac": int(model.atac_ae.encoder.net[0].in_features)}
    latent_dim = int(cfg.latent_dim)
    # No strict state-dict load validates the width here, so check it
    # explicitly; a wrong width would otherwise silently change the fit.
    fused_dim = int(model.fusion.mu_net.out_features) \
        if hasattr(model.fusion.mu_net, "out_features") else latent_dim
    if fused_dim != latent_dim:
        raise ValueError(
            f"cfg.latent_dim={latent_dim} but the live fusion emits {fused_dim}. "
            f"Nothing downstream would catch this mismatch.")

    fusion_cfg = {k: getattr(cfg, k) for k in
                  ("fusion_arch", "fusion_hidden_dim", "fusion_depth",
                   "fusion_residual_scale", "fusion_layer_norm",
                   "sigma_mode", "sigma_init")}
    fusion_cfg["kl_dims"] = getattr(cfg, "kl_dims", 0)

    # --- the second seeding, then the declared construction prefix ----------
    reseed(seed)
    burned = 0
    if rng_protocol == "staged_equivalent" and _BURN_MODULES[stage_b]:
        burned = _burn_staged_prefix(input_dims, latent_dim, fusion_cfg,
                                     atac_trunk=atac_trunk, chr_ids=chr_ids,
                                     spatial_kernels=spatial_kernels)

    # --- encoder: the LIVE modules, so co-adaptation actually co-adapts ------
    encoder = StageAEncoder(model.rna_ae, model.atac_ae, model.fusion).to(device)

    # --- decoders: fresh instances from the same tensors ---------------------
    # Not ``model.rna_ae.decoder`` (item 1 of the module docstring). The
    # checkpoint loaders also build these two autoencoders, so this part of the
    # construction sequence is shared by both protocols.
    sd = model.state_dict()
    decoders: dict[str, nn.Module] = {}
    for mod in ("rna", "atac"):
        ae = ModalityAutoencoder(input_dim=input_dims[mod], latent_dim=latent_dim)
        part = _split(sd, f"{mod}_ae")
        # A spatial/conv trunk has encoder-only ``spatial.*`` keys that the
        # plain autoencoder lacks; drop them rather than building a spatial
        # autoencoder. The decoder is the same symmetric MLP for every trunk,
        # and building the plain autoencoder keeps the RNG consumption that
        # _burn_staged_prefix and the checkpoint loaders assume.
        extra = sorted(k for k in part if k.startswith("spatial."))
        if extra:
            # Never drop a decoder weight: discarding decoder parameters under
            # ``spatial.`` would leave the decoder partly random.
            bad = [k for k in extra if "decoder" in k]
            if bad:
                raise ValueError(
                    f"{mod}: refusing to drop {bad} -- these are DECODER "
                    f"weights under a `spatial.` prefix. The frozen "
                    f"reconstruction teacher would be partly random.")
            part = {k: v for k, v in part.items() if not k.startswith("spatial.")}
            log.info("handoff: %s trunk is spatial/conv; dropped %d "
                     "encoder-only `spatial.*` keys for the decoder copy",
                     mod, len(extra))
        ae.load_state_dict(part, strict=True)
        dec = ae.decoder.to(device).eval()
        for p in dec.parameters():
            p.requires_grad = False
        decoders[mod] = dec

    ok, detail = decoders_are_independent(encoder, decoders)
    if not ok:
        raise SystemExit(
            f"ABORT: frozen decoders share weights with the encoder ({detail}). "
            f"The reconstruction target would drift with the student it is meant "
            f"to hold fixed. This is the failure the in-memory handoff exists to "
            f"avoid -- something replaced the re-instantiation with a borrow.")

    log.info("handoff: stage_b=%s rng_protocol=%s (%d modules burned) | "
             "latent_dim=%d | input_dims rna=%d atac=%d | %s",
             stage_b, rng_protocol, burned, latent_dim, input_dims["rna"],
             input_dims["atac"], detail)

    return encoder, decoders, {
        "latent_dim": latent_dim,
        "input_dims": input_dims,
        "rng_protocol": rng_protocol,
        "stage_b": stage_b,
        "modules_burned": burned,
        "decoder_independence": detail,
    }
