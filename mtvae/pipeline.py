"""Configure and fit both stages: :class:`PipelineConfig`, :func:`build_dataset` and :func:`train`.
Stage A is handed to Stage B in memory, and must reproduce the Stage-A latent exactly."""
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


from mtvae import _constants as _config  # noqa: E402
from mtvae.stage_b import api as dyn_api  # noqa: E402
from mtvae.stage_b.encoder import (  # noqa: E402
    verify_reproduces_stage_a,
)
from mtvae.nn.sde import SDESampler  # noqa: E402
from mtvae.presets import (  # noqa: E402
    FOURDATA_VAE, PFC_VAE, for_run,
)
from mtvae.data import Dataset  # noqa: E402
from mtvae.stage_a import api as vae_api  # noqa: E402
from mtvae.stage_a.trainer import MergedVAEConfig  # noqa: E402

log = logging.getLogger("mtvae.pipeline")

#: Trunk per slot. "rna"/"atac" are the model's slot names (as in the
#: checkpoint keys), not the dataset's modality names, which are free.
DEFAULT_ARCH: Dict[str, Any] = dict(rna_encoder="mlp", atac_encoder="mlp",
                                    spatial_kernels=3)

#: Data handling, the ``data:`` section of a config. ``clip`` bounds each
#: modality's input symmetrically (``None`` = unclipped); ``train_mask`` fits on
#: the example's training split (PFC: ``wholedonor_train_mask.npz``) and encodes
#: every cell; ``n_rows`` fits on the first ``n_rows`` rows of each modality's
#: order in ``bench_tiers.npz`` (the PFC benchmark tiers).
DEFAULT_DATA: Dict[str, Any] = dict(clip={"rna": 10.0, "atac": 10.0},
                                    train_mask=False, n_rows=None)

#: :class:`~mtvae.nn.sde.SDESampler` construction arguments. Kept in one dict
#: so the constructed module and the recorded config cannot differ.
SDE_GEOMETRY = dict(prior_mode="zero", diffusion_mode="scalar",
                    drift_hidden=(128, 128), diffusion_hidden=(32,),
                    time_embed_dim=4, time_hidden=10, diffusion_init=0.02,
                    dt_step=0.01)

#: Stage-B settings shared by both examples: keyword arguments of
#: :func:`mtvae.stage_b.trainer.train_sde_coadapt`. Every one is named here so
#: that every one reaches the run record.
DYNAMICS_DEFAULTS: Dict[str, Any] = dict(
    alpha_align=1.0, lambda_chained=1.0, lambda_per_pair=2.0,
    lambda_rec=1.0, lambda_dist=0.01, lambda_prox=1.0,
    lambda_latent_anchor=1.0,
    #: The SDE path-KL weight and its linear ramp (start, end iterations); see
    #: mtvae.stage_b.trainer. 0 removes the term.
    beta_kl=1e-4, kl_warmup=(200, 400),
    batch_align=256, batch_rec=256, batch_dist=256, batch_anchor=256,
    train_iters=3000, warmup_iters=200, lr=5e-4,
    decoder_lr=1e-5, encoder_lr=1e-5,
    scaled_z_sinkhorn=True, sinkhorn_eps_x=0.05, sinkhorn_iters_x=20,
    shipped_divergence=False, log_every=500,
    #: None: both modalities are clipped once by the dataset, so the encoder
    #: input and the reconstruction target are the same array.
    target_clip=None,
)


#: Everything that differs between the two examples. ``PipelineConfig.for_example``,
#: ``build_dataset``, ``train`` and the CLI read this table and have no
#: per-example branch.
#:
#: ``vae_preset`` is the Stage-A preset (:mod:`mtvae.presets`) and
#: ``vae_overrides`` is applied on top of it. ``dynamics`` is applied on top of
#: :data:`DYNAMICS_DEFAULTS`; the one Stage-B difference is ``lambda_dist``
#: (0.01 for the four datasets, 0.1 for PFC). ``adapter`` names the reader in
#: :mod:`mtvae.adapters`, ``stages`` the stage vocabulary (``None``: read from
#: the metadata) and ``arch`` the trunk of each slot.
EXAMPLES: Dict[str, Dict[str, Any]] = {
    "fourds": {
        "vae_preset": FOURDATA_VAE,
        "vae_overrides": {"beta_sw": 1.0},          # instead of the preset's 2.0
        "dynamics": {"lambda_dist": 0.01},
        "adapter": "fourds",
        "default_dataset": None,                    # HC/HO/DR/MN, must be given
        "stages": None,                             # read from the metadata
        "arch": {"rna_encoder": "mlp", "atac_encoder": "mlp",
                 "spatial_kernels": 3},
        "data": {"clip": {"rna": None, "atac": 5.0}},
        "dir": ("data", "fourds"),
    },
    "pfc": {
        "vae_preset": PFC_VAE,
        "vae_overrides": {},
        # RNA enters unclipped (data.clip) while its reconstruction target is
        # clipped at +/-10, as in the released fits.
        "dynamics": {"lambda_dist": 0.1,
                     "target_clip": {"rna": 10.0, "atac": 10.0}},
        "adapter": "pfc",
        "default_dataset": "PFC",
        # PFC's ATAC trunk is the chromosome-aware Conv1d. Its parameter shapes
        # differ from the MLP's, so it consumes a different amount of RNG and
        # also determines the Stage-B RNG burn.
        "arch": {"rna_encoder": "mlp", "atac_encoder": "spatial",
                 "spatial_kernels": 3},
        # Explicit, not sorted: lexical order of the stage names is not
        # developmental order.
        "stages": tuple(_config.STAGES_ORDERED),
        "spatial": True,
        "data": {"clip": {"rna": None, "atac": 10.0}, "train_mask": True},
        "dir": ("data", "pfc"),
    },
}


def _resolve_data(example: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge :data:`DEFAULT_DATA`, an example's ``data`` entry and overrides.

    ``clip`` merges per modality, so an override may name one modality only.

    Parameters
    ----------
    example, override : dict
        Partial ``data`` sections, applied in that order.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If a key is not one of ``clip``, ``train_mask``, ``n_rows``.
    """
    bad = (set(example) | set(override)) - set(DEFAULT_DATA)
    if bad:
        raise ValueError(f"unknown data settings {sorted(bad)}; known: {sorted(DEFAULT_DATA)}")
    out = {**DEFAULT_DATA, **example, **override}
    out["clip"] = {**DEFAULT_DATA["clip"], **example.get("clip", {}),
                   **override.get("clip", {})}
    out["clip"] = {m: (None if v is None else float(v)) for m, v in out["clip"].items()}
    out["train_mask"] = bool(out["train_mask"])
    out["n_rows"] = None if out["n_rows"] is None else int(out["n_rows"])
    return out


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Both stages' hyperparameters, as one object.

    Build one with :meth:`for_example`; :meth:`to_blob` and :meth:`from_blob`
    convert to and from the flat mapping stored in a run record.

    Parameters
    ----------
    vae : MergedVAEConfig
        Stage A (:class:`mtvae.stage_a.trainer.MergedVAEConfig`); also holds
        the run seed.
    dynamics : dict
        Stage B, passed to :func:`mtvae.stage_b.trainer.train_sde_coadapt`.
    rng_protocol : {"staged_equivalent", "reseed"}, default "staged_equivalent"
        ``staged_equivalent`` makes the fit bit-equal to running the two stages
        as separate scripts; ``reseed`` re-seeds before Stage B without
        replaying those constructions, giving a different (equally valid)
        Stage-B draw. See :mod:`mtvae.stage_b.handoff`.
    device : str, default "cuda"
        Compute device.
    example : str, default "fourds"
        Key into :data:`EXAMPLES` the configuration was started from.
    sde_geometry : dict, optional
        :class:`~mtvae.nn.sde.SDESampler` construction arguments; defaults to
        :data:`SDE_GEOMETRY`.
    arch : dict, optional
        ``{"rna_encoder", "atac_encoder", "spatial_kernels"}``: the trunk of
        each slot; defaults to :data:`DEFAULT_ARCH`.
    data : dict, optional
        ``{"clip", "train_mask", "n_rows"}``; defaults to :data:`DEFAULT_DATA`
        updated by the example's own entry.
    """
    vae: Any
    dynamics: Dict[str, Any]
    rng_protocol: str = "staged_equivalent"
    device: str = "cuda"
    example: str = "fourds"
    #: SDESampler construction arguments, part of the hyperparameters so that
    #: what is constructed is what is recorded.
    sde_geometry: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(SDE_GEOMETRY))
    #: Which trunk each slot gets. A spatial trunk also needs ``chr_ids``, which
    #: are per-feature data: the kind is named here, the peak list comes from
    #: the adapter.
    arch: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_ARCH))
    #: Input clipping, training split and row subset; see :data:`DEFAULT_DATA`.
    data: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_DATA))

    @property
    def seed(self) -> int:
        """The run seed. Stage A seeds from it; Stage B re-seeds from it."""
        return int(self.vae.seed)

    # ---- construction ------------------------------------------------------

    @classmethod
    def for_example(cls, example: str, *, seed: int, device: str = "cuda",
                    vae: Optional[Dict[str, Any]] = None,
                    dynamics: Optional[Dict[str, Any]] = None,
                    sde_geometry: Optional[Dict[str, Any]] = None,
                    arch: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    rng_protocol: str = "staged_equivalent") -> "PipelineConfig":
        """The configuration of ``example``, with overrides applied.

        Parameters
        ----------
        example : {"fourds", "pfc"}
            Key into :data:`EXAMPLES`.
        seed : int
            One seed for both stages.
        device : str, default "cuda"
            Compute device.
        vae : dict, optional
            ``MergedVAEConfig`` field overrides, applied on top of the
            example's preset and overrides.
        dynamics : dict, optional
            Stage-B overrides, applied on top of :data:`DYNAMICS_DEFAULTS` and
            the example's ``dynamics``.
        sde_geometry : dict, optional
            Overrides of :data:`SDE_GEOMETRY`.
        arch : dict, optional
            Overrides of the example's ``arch``.
        data : dict, optional
            Overrides of the example's ``data`` (``clip`` merges per modality).
        rng_protocol : {"staged_equivalent", "reseed"}, default "staged_equivalent"
            See :class:`PipelineConfig`.

        Returns
        -------
        PipelineConfig

        Raises
        ------
        KeyError
            If ``example`` is unknown.
        TypeError
            If a ``vae`` override names something that is not a
            ``MergedVAEConfig`` field.
        """
        if example not in EXAMPLES:
            raise KeyError(f"unknown example {example!r}; have {sorted(EXAMPLES)}")
        spec = EXAMPLES[example]
        dyn = {**DYNAMICS_DEFAULTS, **spec["dynamics"], **(dynamics or {})}
        # YAML and JSON return a tuple as a list, and PyYAML reads ``1e-4``
        # (no dot) as a string. Normalised here, since the CLI, YAML and the
        # library API all go through this call, so the value recorded and the
        # value trained on are the same.
        dyn["beta_kl"] = float(dyn["beta_kl"])
        dyn["kl_warmup"] = tuple(int(v) for v in dyn["kl_warmup"])
        return cls(
            vae=for_run(spec["vae_preset"], seed=seed, device=device,
                        **{**spec["vae_overrides"], **(vae or {})}),
            dynamics=dyn,
            rng_protocol=rng_protocol, device=device, example=example,
            sde_geometry={**SDE_GEOMETRY, **(sde_geometry or {})},
            arch={**DEFAULT_ARCH, **spec.get("arch", {}), **(arch or {})},
            data=_resolve_data(spec.get("data", {}), data or {}))

    # ---- flat representation for run records -------------------------------

    def to_blob(self) -> Dict[str, Any]:
        """Flatten to a single JSON-serialisable mapping covering both stages.

        Keys are prefixed by section: ``vae.`` (Stage A), ``sde.`` (Stage-B
        objective), ``geom.`` (SDE construction) and ``arch.`` (trunks), plus
        the top-level ``example``, ``rng_protocol`` and ``device``. The
        prefixes keep names that exist in both stages apart (``lr``,
        ``encoder_lr`` and ``log_every`` mean different things in each).

        Returns
        -------
        dict
            Round-trips through :meth:`from_blob`.
        """
        blob = {"example": self.example, "rng_protocol": self.rng_protocol,
                "device": self.device}
        blob.update({f"vae.{k}": v for k, v in vars(self.vae).items()})
        blob.update({f"sde.{k}": v for k, v in self.dynamics.items()})
        # ``geom.``, not ``sde.``: these are SDESampler construction arguments,
        # not objective terms.
        blob.update({f"geom.{k}": (list(v) if isinstance(v, tuple) else v)
                     for k, v in self.sde_geometry.items()})
        blob.update({f"arch.{k}": v for k, v in self.arch.items()})
        blob.update({f"data.{k}": v for k, v in self.data.items()})
        return blob

    @classmethod
    def from_blob(cls, blob: Dict[str, Any]) -> "PipelineConfig":
        """Rebuild from :meth:`to_blob`; an exact round trip, with no defaulting.

        Parameters
        ----------
        blob : dict
            Output of :meth:`to_blob` (e.g. a record's ``hyperparameters``).

        Returns
        -------
        PipelineConfig

        Raises
        ------
        KeyError
            If ``example``, ``rng_protocol`` or ``device`` is absent.
        """
        for k in ("example", "rng_protocol", "device"):
            if k not in blob:
                raise KeyError(f"blob has no {k!r}; it is not defaultable")
        vae = {k[4:]: v for k, v in blob.items() if k.startswith("vae.")}
        dyn = {k[4:]: v for k, v in blob.items() if k.startswith("sde.")}
        geom = {k[5:]: v for k, v in blob.items() if k.startswith("geom.")}
        # Tuples come back from JSON as lists; SDESampler's hidden-size
        # arguments must be tuples.
        for k in ("drift_hidden", "diffusion_hidden"):
            if k in geom:
                geom[k] = tuple(geom[k])
        # Same for the path-KL ramp. A blob without ``sde.beta_kl`` rebuilds
        # with the trainer default 0 (no path-KL term).
        if "kl_warmup" in dyn:
            dyn["kl_warmup"] = tuple(dyn["kl_warmup"])
        arch = {k[5:]: v for k, v in blob.items() if k.startswith("arch.")}
        data = {k[5:]: v for k, v in blob.items() if k.startswith("data.")}
        return cls(vae=MergedVAEConfig(**vae), dynamics=dyn,
                   rng_protocol=blob["rng_protocol"], device=blob["device"],
                   example=blob["example"], sde_geometry=geom or dict(SDE_GEOMETRY),
                   arch=arch or dict(DEFAULT_ARCH),
                   data=_resolve_data(data, {}))


def build_dataset(example: str, data_root: Path, *, dataset: Optional[str] = None,
                  data: Optional[Dict[str, Any]] = None,
                  arch: Optional[Dict[str, Any]] = None) -> Dataset:
    """Load one of the examples from the figshare layout as a :class:`Dataset`.

    Parameters
    ----------
    example : {"fourds", "pfc"}
    data_root : pathlib.Path
        The figshare data root; the example is read from
        ``data_root/data/pfc`` or ``data_root/data/fourds/<dataset>``.
    dataset : {"HC", "HO", "DR", "MN"}, optional
        Required for ``"fourds"``.
    data : dict, optional
        ``data`` section of a config (``clip``, ``train_mask``, ``n_rows``);
        unset keys come from the example. Pass ``cfg.data``.
    arch : dict, optional
        ``arch`` section of a config; decides each trunk. Pass ``cfg.arch``.

    Returns
    -------
    Dataset

    Raises
    ------
    KeyError
        If ``example`` is unknown.
    ValueError
        If the four-dataset example is asked for without a ``dataset`` label,
        a trunk kind is unknown, or a ``"spatial"`` trunk is requested for the
        four datasets (which have no peak list; use ``"conv"``).
    """
    from mtvae import adapters
    if example not in EXAMPLES:
        raise KeyError(f"unknown example {example!r}; have {sorted(EXAMPLES)}")
    spec = EXAMPLES[example]
    name = dataset or spec["default_dataset"]
    if name is None:
        raise ValueError(
            f"example {example!r} needs an explicit dataset label (HC/HO/DR/MN)")
    d = _resolve_data(spec.get("data", {}), data or {})
    a = {**DEFAULT_ARCH, **spec.get("arch", {}), **(arch or {})}
    for k in ("rna_encoder", "atac_encoder"):
        if a[k] not in ("mlp", "spatial", "conv"):
            raise ValueError(
                f"arch.{k}={a[k]!r}; expected 'mlp', 'spatial' or 'conv'")
    data_dir = Path(data_root).joinpath(*spec["dir"])
    if spec["adapter"] == "fourds":
        if "spatial" in (a["atac_encoder"], a["rna_encoder"]):
            raise ValueError(
                "the four-dataset inputs carry no peak list, so a 'spatial' "
                "trunk has no chr_ids to build from. Use 'conv' for the same "
                "architecture without them.")
        data_dir = data_dir / name
        ds = adapters.fourds(data_dir, name, clip=d["clip"])
    else:
        ds = adapters.pfc(
            data_dir, stages=spec["stages"],
            train_mask=(data_dir / "wholedonor_train_mask.npz"
                        if d["train_mask"] else None),
            spatial_kernels=int(a["spatial_kernels"]),
            atac_trunk=a["atac_encoder"], clip=d["clip"])
    if d["n_rows"] is not None:
        # The first n_rows of each modality's fixed order, in that order. Stage
        # times stay those of the full dataset.
        perm = np.load(data_dir / "bench_tiers.npz")
        ds = dataclasses.replace(ds, modalities=tuple(
            m.select(perm[f"perm_{m.name}"][:d["n_rows"]]) for m in ds.modalities))
    return ds


@dataclasses.dataclass
class Fit:
    """A trained pipeline: Stage A, the handoff, and Stage B.

    Attributes
    ----------
    vae : mtvae.stage_a.api.TrainedVAE
        The Stage-A fit. Its model is the live object Stage B co-adapted.
    latents : mtvae.stage_a.api.Latents
        Stage-A latents of every cell (before Stage B).
    stage_a : mtvae.stage_b.api.StageA
        The handoff: co-adapted encoder and decoders.
    dynamics : mtvae.stage_b.api.TrainedDynamics
        The Stage-B fit.
    data : Dataset
        The full dataset the fit was built on.
    hp : PipelineConfig
        The hyperparameters.
    seconds : dict
        Wall-clock time of each stage.
    gate : dict
        Per-modality result of the Stage-A reproduction check.
    stage_a_state : dict
        State dict of the Stage-A model before Stage B (see below).
    """
    vae: Any
    latents: Any
    stage_a: Any
    dynamics: Any
    data: Dataset
    hp: PipelineConfig
    seconds: Dict[str, float]
    gate: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: The Stage-A weights before Stage B ran. Stage B co-adapts the encoder in
    #: place, so ``vae.model`` no longer reproduces the Stage-A latents once
    #: ``train`` returns; a Stage-A checkpoint must be written from this
    #: snapshot.
    stage_a_state: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def encode(self, data: Optional[Dataset] = None) -> Dict[str, np.ndarray]:
        """Encode through the co-adapted (Stage-B) encoder.

        Parameters
        ----------
        data : Dataset, optional
            Defaults to the full dataset the fit was built on (not the
            training view).

        Returns
        -------
        dict of str to numpy.ndarray
            Latent of each modality, keyed by modality name.
        """
        d = data if data is not None else self.data
        enc = self.stage_a.encoder
        return {m.name: enc.encode_all(m.x, slot, device=self.hp.device)
                for m, slot in zip(d.modalities, dyn_api.SLOTS)}


def train(data: Dataset, hyperparameters: PipelineConfig, *,
          verify_n: int = 4096, verify_atol: float = 0.0) -> Fit:
    """Fit Stage A then Stage B, in one process, from one seed.

    Parameters
    ----------
    data : Dataset
        Any dataset satisfying the :mod:`mtvae.data` contract. Fitting uses
        ``data.training_view()``; latents are encoded over all cells.
    hyperparameters : PipelineConfig
        Both stages' settings.
    verify_n : int, default 4096
        Cells per modality checked by the Stage-A reproduction gate.
    verify_atol : float, default 0.0
        Tolerance of that gate; the in-memory handoff should be exact.

    Returns
    -------
    Fit

    Raises
    ------
    ValueError
        If ``data`` carries no ``stage_times``, or its trunks differ from
        ``hyperparameters.arch``.
    RuntimeError
        If the handed-over encoder does not reproduce the Stage-A latent.

    Notes
    -----
    Stage A seeds torch and numpy itself and draws from explicit
    ``Generator``s, so it does not depend on what ran earlier in the process.
    Stage B re-seeds from the same seed, and the SDE then initialises where
    ``rng_protocol`` puts it.
    """
    if data.stage_times is None:
        raise ValueError(
            f"{data.name} has no stage_times; Stage B integrates between them. "
            f"Derive them with mtvae.data.stage_times_from.")

    # The config names the trunks; the Dataset carries them. They must agree:
    # the trunk kind decides the Stage-B RNG burn, so a mismatch would
    # silently change the fit and the record would describe another one.
    declared = [hyperparameters.arch[f"{s}_encoder"] for s in ("rna", "atac")]
    carried = [m.trunk.kind for m in data.modalities]
    if declared != carried:
        raise ValueError(
            f"arch declares trunks {declared} but the Dataset carries {carried} "
            f"(slot order). The trunk kind decides the Stage-B burn, so this "
            f"would silently change where SDESampler initialises. Build the "
            f"Dataset with build_dataset(..., arch=hp.arch), or fix the config.")

    hp = hyperparameters
    t0 = time.time()
    log.info("STAGE A: %s | %d stages | latent_dim=%d | %d iters",
             data.name, data.n_stages, hp.vae.latent_dim, hp.vae.n_iters)
    fit = vae_api.train(data, hp.vae)
    latents = fit.encode(data)
    t_a = time.time()

    # Builds the encoder, re-instantiates the decoders (never borrows them),
    # and positions the torch stream. The ATAC trunk spec comes from the
    # Dataset, so the RNG burn matches the trunk actually trained.
    stage_a = dyn_api.StageA.from_vae(fit, latents, rng_protocol=hp.rng_protocol,
                                      stage_b="coadapt", device=hp.device,
                                      atac_spec=data.modalities[1].trunk)

    # The reproduction gate must run here, after the handoff and before Stage
    # B: Stage B co-adapts the encoder, so afterwards it no longer reproduces
    # the Stage-A latent by design. Nothing was reloaded, but this is still a
    # second forward pass through the same chunked path, so it must be exact.
    gate = {}
    for m in data.names:
        ok, d = verify_reproduces_stage_a(
            stage_a.encoder, data[m].x, np.asarray(latents.mu[m]),
            dyn_api.SLOTS[data.names.index(m)], device=hp.device,
            n=verify_n, atol=verify_atol)
        gate[m] = {"ok": bool(ok), **d}
    if not all(g["ok"] for g in gate.values()):
        raise RuntimeError(
            f"Stage-A gate failed BEFORE Stage B ran: "
            f"{ {m: g.get('max_abs') for m, g in gate.items()} }. Nothing was "
            f"reloaded, so this is the handoff itself, not a clip mismatch.")

    # Snapshot before Stage B updates the encoder; detached CPU clones, so it
    # does not change with the original.
    stage_a_state = {k: v.detach().cpu().clone()
                     for k, v in fit.model.state_dict().items()}

    # Built after the handoff, so the RNG burn has positioned the stream.
    sde = SDESampler(latent_dim=stage_a.latent_dim,
                     **hp.sde_geometry).to(hp.device)
    log.info("STAGE B: rng_protocol=%s | encoder_lr=%g decoder_lr=%g "
             "lambda_dist=%g", hp.rng_protocol,
             hp.dynamics.get("encoder_lr"), hp.dynamics.get("decoder_lr"),
             hp.dynamics.get("lambda_dist"))
    dyn = dyn_api.train_coadapt(
        stage_a, data, {**hp.dynamics, "seed": hp.seed, "device": hp.device},
        sde=sde, rng_protocol=hp.rng_protocol)

    return Fit(vae=fit, latents=latents, stage_a=stage_a, dynamics=dyn,
               data=data, hp=hp,
               seconds={"stage_a": round(t_a - t0, 1),
                        "stage_b": round(time.time() - t_a, 1)},
               gate=gate, stage_a_state=stage_a_state)
