"""Stage A as a library call.
:func:`train` fits the VAE, maybe on a subset; ``fit.encode(data)`` then encodes all cells."""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from mtvae.data import Dataset
from mtvae.rank import eff_rank, eff_rank_joint
from mtvae.stage_a.trainer import (
    MergedVAEConfig, extract_latents, train_merged,
)

#: The Stage-A hyperparameter type: an alias of the trainer's own dataclass,
#: so records and presets share one schema.
VAEHyperparameters = MergedVAEConfig


@dataclasses.dataclass
class Latents:
    """Posterior means and one reparameterised draw, per modality.

    Attributes
    ----------
    mu : dict of str to numpy.ndarray
        Posterior means, keyed by modality name.
    z : dict of str to numpy.ndarray
        One sample, keyed by modality name.
    """
    mu: Dict[str, np.ndarray]
    z: Dict[str, np.ndarray]


@dataclasses.dataclass
class TrainedVAE:
    """A fitted Stage-A model, plus what it takes to describe the fit.

    Attributes
    ----------
    model : MergedVAE
        The trained model.
    hp : MergedVAEConfig
        The hyperparameters.
    history : dict of str to list
        Training history from :func:`~mtvae.stage_a.trainer.train_merged`.
    order : tuple of str
        Modality names in slot order.
    trunks : dict of str to str
        Trunk kind of each modality.
    seconds : float
        Training wall-clock time.
    """
    model: Any
    hp: MergedVAEConfig
    history: Dict[str, list]
    order: tuple
    trunks: Dict[str, str]
    seconds: float

    def encode(self, data: Dataset, chunk: int = 4096) -> Latents:
        """Posterior means and one sample for every cell of ``data``.

        Pass the full dataset here even when the fit used a training mask.

        Parameters
        ----------
        data : Dataset
            Cells to encode (both modalities).
        chunk : int, default 4096
            Rows per forward pass.

        Returns
        -------
        Latents
        """
        a, b = data.modalities
        lat = extract_latents(self.model, a.x, b.x, self.hp, chunk=chunk)
        return Latents(mu={a.name: lat["mu_rna"], b.name: lat["mu_atac"]},
                       z={a.name: lat["z_rna"], b.name: lat["z_atac"]})

    def geometry(self, lat: Latents) -> Dict[str, float]:
        """Effective rank of the means and of the sample, joint and per modality.

        Parameters
        ----------
        lat : Latents
            Output of :meth:`encode`.

        Returns
        -------
        dict of str to float
            ``eff_rank_{joint,rna,atac}_mu`` and the ``_sample`` counterparts
            (``rna``/``atac`` denote the two slots).
        """
        a, b = self.order
        return {
            "eff_rank_joint_mu": eff_rank_joint(lat.mu[a], lat.mu[b]),
            "eff_rank_rna_mu": eff_rank(lat.mu[a]),
            "eff_rank_atac_mu": eff_rank(lat.mu[b]),
            "eff_rank_joint_sample": eff_rank_joint(lat.z[a], lat.z[b]),
            "eff_rank_rna_sample": eff_rank(lat.z[a]),
            "eff_rank_atac_sample": eff_rank(lat.z[b]),
        }

    def decoders(self) -> Dict[str, torch.nn.Module]:
        """The two decoder modules of the trained model, keyed by modality name.

        These are the model's own modules, not copies; Stage B uses
        re-instantiated decoders instead (:mod:`mtvae.stage_b.handoff`).
        """
        a, b = self.order
        return {a: self.model.rna_ae.decoder, b: self.model.atac_ae.decoder}

    @property
    def latent_dim(self) -> int:
        """Latent dimension."""
        return int(self.hp.latent_dim)

    @property
    def input_dims(self) -> Dict[str, int]:
        """Per-modality feature count, read off the trunks.

        Stage B needs these to rebuild the decoder autoencoders, whose
        construction also sets where the SDE initialises in the RNG stream
        (:mod:`mtvae.stage_b.handoff`).
        """
        a, b = self.order
        return {a: int(self.model.rna_ae.encoder.net[0].in_features),
                b: int(self.model.atac_ae.encoder.net[0].in_features)}


def train(data: Dataset, hyperparameters: MergedVAEConfig, *,
          sampler: Optional[Any] = None) -> TrainedVAE:
    """Fit the Stage-A VAE on ``data.training_view()``.

    Parameters
    ----------
    data : Dataset
        Two modalities, already scaled and clipped.
    hyperparameters : MergedVAEConfig
        Normally ``mtvae.presets.for_run(PRESET, seed=..., device=...)``.
    sampler : optional
        Reserved for a stratified batch sampler; not implemented.

    Returns
    -------
    TrainedVAE

    Raises
    ------
    NotImplementedError
        If ``sampler`` is given.

    Notes
    -----
    The trainer seeds ``torch``, then numpy's global stream, then builds its
    own ``Generator``. The global numpy seed matters: openTSNE's
    ``PerplexityBasedNN`` takes no ``random_state`` and draws from it, and
    both examples use ``topo_affinity="full"``.
    """
    if sampler is not None:
        raise NotImplementedError(
            "sampler injection is not implemented")
    fit_on = data.training_view()
    a, b = fit_on.modalities
    t0 = time.time()
    model, history = train_merged(
        a.x, b.x, a.stage, b.stage,
        n_stages=fit_on.n_stages, cfg=hyperparameters,
        rna_teacher=a.teacher, atac_teacher=b.teacher,
        trunks={"rna": a.trunk, "atac": b.trunk},
        # Declared by the modalities, or the slot defaults. The call is
        # positional: ``a`` and ``b`` fill slots 0 and 1 whatever they are
        # named, and the "rna"/"atac" keys are the model's slot names.
        losses=data.losses,
    )
    return TrainedVAE(model=model, hp=hyperparameters, history=history,
                      order=fit_on.names,
                      trunks={m.name: m.trunk.kind for m in fit_on.modalities},
                      seconds=round(time.time() - t0, 1))
