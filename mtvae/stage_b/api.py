"""Stage B as a library call.
Build the SDE after :meth:`StageA.from_vae`, whose handoff sets the RNG stream position."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from mtvae.stage_b.handoff import (
    RNG_PROTOCOLS as _HANDOFF_PROTOCOLS,
)
from mtvae.data import Dataset

#: Re-exported from :mod:`mtvae.stage_b.handoff`, where the burn is
#: implemented.
RNG_PROTOCOLS = _HANDOFF_PROTOCOLS
#: The model's two slot names, as in the checkpoint keys (``rna_ae.*``,
#: ``atac_ae.*``) and the trainers. A Dataset's own ``names`` are mapped onto
#: these by position, never compared to them.
SLOTS = ("rna", "atac")
#: Number of nn.Linear layers in one ModalityAutoencoder.
_LINEARS_PER_AE = 8


@dataclasses.dataclass(frozen=True)
class StageA:
    """What Stage B consumes from Stage A.

    ``decoders`` are always separate instances from ``encoder``'s own trunks
    (:func:`~mtvae.stage_b.handoff.build_handoff` asserts it); borrowed
    decoders would move with the encoder.

    Attributes
    ----------
    latents : mapping of str to numpy.ndarray
        Stage-A latents (posterior means), keyed by modality name; the
        anchor target of Stage B.
    decoders : mapping of str to torch.nn.Module
        Decoders keyed by modality name.
    latent_dim : int
        Latent dimension.
    input_dims : mapping of str to int
        Feature count of each modality.
    encoder : torch.nn.Module, optional
        The trainable Stage-A encoder path
        (:class:`~mtvae.stage_b.encoder.StageAEncoder`).
    provenance : mapping of str to str
        Source, RNG protocol, Stage-B kind and decoder-independence detail.
    rng_burn_applied : bool
        Whether the ``staged_equivalent`` burn ran.
    """
    latents: Mapping[str, np.ndarray]
    decoders: Mapping[str, torch.nn.Module]
    latent_dim: int
    input_dims: Mapping[str, int]
    encoder: Optional[torch.nn.Module] = None
    provenance: Mapping[str, str] = dataclasses.field(default_factory=dict)
    rng_burn_applied: bool = False

    @classmethod
    def from_vae(cls, fit: Any, latents: Any, *, rng_protocol: str,
                 stage_b: str = "coadapt", device: str = "cpu",
                 atac_spec: Any = None) -> "StageA":
        """Build from a fitted ``TrainedVAE``, via
        :func:`mtvae.stage_b.handoff.build_handoff`.

        Parameters
        ----------
        fit : TrainedVAE
            From :func:`mtvae.stage_a.api.train`.
        latents : Latents
            From ``fit.encode(data)``. Only the posterior means are kept:
            Stage B anchors the encoder to the map it computes, which is the
            mean.
        rng_protocol : {"reseed", "staged_equivalent"}
            See :data:`mtvae.stage_b.handoff.RNG_PROTOCOLS`.
        stage_b : {"coadapt", "decoder"}, default "coadapt"
            Which Stage-B construction sequence the burn replays; see
            :data:`mtvae.stage_b.handoff.STAGE_B_KINDS`.
        device : str, default "cpu"
            Where the encoder and decoders are placed.
        atac_spec : TrunkSpec, optional
            The slot-1 trunk of the dataset Stage A was fitted on. A spatial
            trunk's ``chr_ids`` and kernel size are needed to replay its
            construction in the ``staged_equivalent`` burn.

        Returns
        -------
        StageA
        """
        from mtvae.stage_b.handoff import build_handoff

        a, b = fit.order
        spatial = {}
        if atac_spec is not None and atac_spec.kind == "spatial":
            spatial = dict(chr_ids=atac_spec.chr_ids,
                           spatial_kernels=atac_spec.kernel_sizes)
        enc, decs, meta = build_handoff(
            fit.model, fit.hp, seed=int(fit.hp.seed), rng_protocol=rng_protocol,
            stage_b=stage_b, device=device,
            atac_trunk=fit.trunks.get(b, "mlp"), **spatial)
        # Back to the dataset's modality names; build_handoff uses slot names.
        slot_decs = {a: decs["rna"], b: decs["atac"]}
        return cls(latents={k: np.asarray(v) for k, v in latents.mu.items()},
                   decoders=slot_decs, latent_dim=meta["latent_dim"],
                   input_dims={a: meta["input_dims"]["rna"],
                               b: meta["input_dims"]["atac"]},
                   encoder=enc,
                   provenance={"source": "live TrainedVAE",
                               "rng_protocol": rng_protocol,
                               "stage_b": stage_b,
                               "decoder_independence":
                                   meta["decoder_independence"]},
                   rng_burn_applied=rng_protocol == "staged_equivalent")


@dataclasses.dataclass(frozen=True)
class TrainedDynamics:
    """A fitted latent SDE and what it takes to describe the fit.

    Attributes
    ----------
    sde : SDESampler
        The trained SDE.
    decoders : mapping of str to torch.nn.Module
        The (co-adapted) decoders, keyed by modality name.
    stage_times : numpy.ndarray
        Normalised stage times the SDE was trained on.
    history : dict of str to list
        Training history from :func:`mtvae.stage_b.trainer.train_sde_coadapt`.
    hp : mapping
        The Stage-B hyperparameters.
    rng_protocol : str
        The RNG protocol used.
    """
    sde: Any
    decoders: Mapping[str, torch.nn.Module]
    stage_times: np.ndarray
    history: Dict[str, list]
    hp: Any
    rng_protocol: str


def train_coadapt(vae: StageA, data: Dataset, hyperparameters: Mapping[str, Any],
                  *, sde: Optional[Any] = None,
                  rng_protocol: Optional[str] = None) -> TrainedDynamics:
    """Fit the latent SDE while co-adapting the encoder and decoders.

    The decoders in ``vae`` are trained in place (when ``decoder_lr > 0``),
    as is the encoder (when ``encoder_lr > 0``).

    Parameters
    ----------
    vae : StageA
        Must carry an ``encoder``.
    data : Dataset
        Needs ``stage_times``: Stage B integrates between stages. Each
        modality's ``train`` mask restricts the cells Stage B draws.
    hyperparameters : mapping
        Keyword arguments of
        :func:`mtvae.stage_b.trainer.train_sde_coadapt` (including ``seed``
        and ``device``). ``encoder_lr = 0`` keeps the encoder frozen, the
        control for any co-adaptation effect.
    sde : SDESampler
        Required, and must be built after ``StageA.from_vae`` so the re-seed
        (and the burn) have already positioned the stream.
    rng_protocol : {"reseed", "staged_equivalent"}, optional
        Defaults to what ``vae`` recorded.

    Returns
    -------
    TrainedDynamics

    Raises
    ------
    ValueError
        If the encoder is missing, ``stage_times`` is absent, ``sde`` is not
        supplied, or the declared protocol disagrees with what ``vae`` did.
    """
    from mtvae.stage_b.trainer import (
        train_sde_coadapt,
    )

    if vae.encoder is None:
        raise ValueError(
            "StageA carries no encoder, so there is nothing to co-adapt. Build it "
            "with StageA.from_vae(..., stage_b='coadapt').")
    if data.stage_times is None:
        raise ValueError(
            f"{data.name} has no stage_times; Stage B integrates between them. "
            f"Derive them with mtvae.data.stage_times_from.")
    proto = rng_protocol or vae.provenance.get("rng_protocol")
    if proto not in RNG_PROTOCOLS:
        raise ValueError(f"rng_protocol is required; one of {RNG_PROTOCOLS}")
    if proto == "staged_equivalent" and not vae.rng_burn_applied:
        raise ValueError(
            "rng_protocol='staged_equivalent' but this StageA was built without "
            "the burn; the SDE will not initialise where the released fits' "
            "SDE did.")
    if sde is None:
        raise ValueError(
            "sde is required. Its initial weights are the whole point of "
            "rng_protocol: build it AFTER StageA.from_vae.")

    # Modality name -> model slot: the trainer dispatches on the slot names.
    order = data.names
    slot_of = dict(zip(order, SLOTS))
    hist = train_sde_coadapt(
        sde, {slot_of[m]: vae.decoders[m] for m in order}, vae.encoder,
        {slot_of[m]: torch.as_tensor(vae.latents[m]) for m in order},
        {slot_of[m]: torch.as_tensor(data[m].x) for m in order},
        {slot_of[m]: data[m].stage for m in order},
        {slot_of[m]: (data[m].train if data[m].train is not None
                      else np.ones(data[m].n_cells, bool)) for m in order},
        np.asarray(data.stage_times, np.float32),
        **dict(hyperparameters))
    return TrainedDynamics(sde=sde, decoders=dict(vae.decoders),
                           stage_times=np.asarray(data.stage_times),
                           history=hist, hp=hyperparameters, rng_protocol=proto)
