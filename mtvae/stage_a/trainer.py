"""Stage-A training: the two-modality VAE with a shared latent.

:func:`train_merged` fits a :class:`~mtvae.stage_a.model.MergedVAE` in one of
two modes:

``mode="staged"``
    The trunks are first fitted on reconstruction alone (or loaded from a
    cache), then the stochastic fusion head is inserted and everything is
    fine-tuned under the full objective. The head is identity-initialised, so
    training starts from exactly the pre-trained function.
``mode="joint"``
    One network, one optimiser, the full objective from step 0, no
    pre-training.

The objective is the same in both modes::

    L = recon_rna + recon_atac
      + beta_sw          * mean_over_stages SW2(z_rna|s, z_atac|s)
      + lambda_topo_rna  * tilt(z_rna)
      + lambda_topo_atac * tilt(z_atac)
      + beta_kl          * KL(q(z|x) || N(0, I))

By default the reconstruction, the sliced-Wasserstein term and the tilt are
evaluated on the reparameterised sample, as in the MIND reference
implementation. ``beta_kl = 0`` with ``deterministic = True`` gives the
deterministic fused autoencoder with otherwise identical settings.

Reconstruction batches are drawn uniformly per modality; the
sliced-Wasserstein term draws equal-size batches per stage from both
modalities; each tilt term draws its own batch.

:func:`extract_latents` encodes every cell, and :func:`run_record` assembles a
JSON-serialisable summary of a fit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import Adam


from mtvae.nn.autoencoders import (  # noqa: E402
    ModalityAutoencoder,
)
from mtvae.stage_a.sliced_wasserstein import (  # noqa: E402
    sliced_wasserstein_2,
)
from mtvae.stage_a.pretrain import (  # noqa: E402
    Phase1Config, train_modality_ae,
)
from mtvae.stage_a.topology_tilt import (  # noqa: E402
    full_teacher_affinity, subset_affinity, tilt_loss_from_P, topology_tilt_loss,
)
from mtvae.stage_a.losses import (  # noqa: E402
    DEFAULT_SLOT_LOSSES, active_units, gaussian_kl, geometry_report,
    reconstruction_loss,
)
from mtvae.stage_a.model import (  # noqa: E402
    MergedVAE, StochasticFusion,
)

log = logging.getLogger("mtvae.stage_a.trainer")


@dataclass
class MergedVAEConfig:
    """Every Stage-A setting.

    The dataclass defaults match neither example (for instance
    ``mode="staged"`` and ``beta_kl=1.0``); the complete configurations of
    the two examples are in :mod:`mtvae.presets`, and
    :func:`mtvae.presets.for_run` specialises one for a fit.
    """

    # ----- mode -----
    mode: str = "staged"                 # "staged" | "joint"

    # ----- trunk pre-training (staged mode only) -----
    phase1_dir: Optional[Path] = None    # cache directory for the pre-trained trunks
    p1_rna_iters: int = 2000
    p1_atac_iters: int = 4000
    p1_rna_lr: float = 1e-3
    p1_atac_lr: float = 5e-4
    p1_batch: int = 512

    # ----- main loop -----
    n_iters: int = 6000
    batch_size_per_modality: int = 256
    lr: float = 5e-4                     # fusion head
    encoder_lr: float = 1e-4             # trunks + decoders (staged mode)
    joint_lr: float = 5e-4               # single lr for every parameter (joint mode)
    joint_warmup_iters: int = 0          # recon-only warmup before the penalties bite
    grad_clip: float = 5.0

    # ----- cross-modal OT term -----
    beta_sw: float = 2.0
    sw_batch_per_stage: int = 64
    sw_n_projections: int = 64

    # ----- MIND tilt -----
    lambda_topo_rna: float = 1e-2
    lambda_topo_atac: float = 1e-2
    topo_batch: int = 512
    topo_perplexity: float = 30.0
    topo_affinity: str = "full"          # "full" | "batch"
    # Factorised tilt: 0 (default) tilts the full latent; k > 0 tilts only
    # dims [0, k) and leaves the rest to reconstruction and the OT term.
    topo_identity_dims: int = 0

    # ----- the variational part -----
    beta_kl: float = 1.0
    kl_scale: str = "mind"               # "mind" (/ n_dims) | "elbo"
    kl_free_bits: float = 0.0            # nats per dimension
    kl_dims: int = 0                     # 0 = all dims stochastic; k = first k only
    sigma_mode: str = "learned"          # "learned" | "fixed"
    sigma_init: float = 0.1
    deterministic: bool = False          # z = mu; with beta_kl=0, a plain autoencoder
    # Which latent the two regularisers see. Reconstruction always uses the
    # sample (it is the likelihood term). The tilt and the OT term are
    # penalties on the geometry, so they can instead use the mean, which
    # removes the noise floor the sample adds to every pairwise distance.
    # "sample" follows the reference implementation and is the default.
    tilt_on: str = "sample"              # "sample" | "mu"
    sw_on: str = "sample"                # "sample" | "mu"

    # ----- fusion capacity -----
    fusion_arch: str = "shared"
    fusion_hidden_dim: int = 1024
    fusion_depth: int = 1
    fusion_residual_scale: float = 0.5
    fusion_layer_norm: bool = False

    # ----- shared -----
    latent_dim: int = 50
    seed: int = 314159
    device: str = "cpu"
    log_every: int = 100

    def resolved(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of every field."""
        d = asdict(self)
        d["phase1_dir"] = None if self.phase1_dir is None else str(self.phase1_dir)
        return d


def _stage_pools(stages: np.ndarray, n_stages: int) -> list[np.ndarray]:
    """Cell indices per stage.

    Parameters
    ----------
    stages : numpy.ndarray, shape (n_cells,)
        Integer stage index per cell.
    n_stages : int
        Total number of stages.

    Returns
    -------
    list of numpy.ndarray
        ``out[s]`` holds the indices of the cells at stage ``s``.
    """
    return [np.flatnonzero(stages == s) for s in range(n_stages)]


def _save_phase1(rna_ae: ModalityAutoencoder, atac_ae: ModalityAutoencoder,
                 phase1_dir: Path) -> None:
    """Save pre-trained trunks as ``checkpoints/{rna,atac}_ae.pt``.

    Parameters
    ----------
    rna_ae, atac_ae : ModalityAutoencoder
        Fitted trunks.
    phase1_dir : pathlib.Path
        Destination; ``checkpoints/`` is created inside it.
    """
    ck = Path(phase1_dir) / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    for name, ae in (("rna", rna_ae), ("atac", atac_ae)):
        torch.save({"model_state_dict": ae.state_dict()}, ck / f"{name}_ae.pt")
    log.info("Phase-1 trunks cached to %s", phase1_dir)


def pretrain_phase1(
    rna_expr: np.ndarray,
    atac_expr: np.ndarray,
    cfg: MergedVAEConfig,
) -> tuple[ModalityAutoencoder, ModalityAutoencoder]:
    """Fit the two trunks on reconstruction alone (staged mode).

    Uses :func:`mtvae.stage_a.pretrain.train_modality_ae` with MSE for RNA and
    Huber (delta=1) for ATAC, at the iteration counts and learning rates in
    ``cfg``.

    Parameters
    ----------
    rna_expr, atac_expr : numpy.ndarray
        Feature matrices, ``(n_cells, n_features)``. ATAC is expected to be
        clipped by the caller.
    cfg : MergedVAEConfig
        Run configuration.

    Returns
    -------
    tuple of ModalityAutoencoder
        ``(rna_ae, atac_ae)``.
    """
    log.info("Phase-1 pre-training: RNA %d iters @ %.1e, ATAC %d iters @ %.1e",
             cfg.p1_rna_iters, cfg.p1_rna_lr, cfg.p1_atac_iters, cfg.p1_atac_lr)
    rna_ae, _ = train_modality_ae(
        rna_expr,
        Phase1Config(input_dim=rna_expr.shape[1], latent_dim=cfg.latent_dim,
                     n_iters=cfg.p1_rna_iters, batch_size=cfg.p1_batch,
                     lr=cfg.p1_rna_lr, seed=cfg.seed, loss="mse"),
        device=cfg.device,
    )
    atac_ae, _ = train_modality_ae(
        atac_expr,
        Phase1Config(input_dim=atac_expr.shape[1], latent_dim=cfg.latent_dim,
                     n_iters=cfg.p1_atac_iters, batch_size=cfg.p1_batch,
                     lr=cfg.p1_atac_lr, seed=cfg.seed, loss="huber",
                     huber_delta=1.0),
        device=cfg.device,
    )
    return rna_ae, atac_ae


def load_phase1(phase1_dir: Path, rna_dim: int, atac_dim: int,
                latent_dim: int) -> tuple[ModalityAutoencoder, ModalityAutoencoder]:
    """Load cached pre-trained trunks from a checkpoint directory.

    Parameters
    ----------
    phase1_dir : pathlib.Path
        Directory containing ``checkpoints/rna_ae.pt`` and ``atac_ae.pt``.
    rna_dim, atac_dim : int
        Feature dimensions.
    latent_dim : int
        Shared-latent width.

    Returns
    -------
    tuple of ModalityAutoencoder
        ``(rna_ae, atac_ae)`` with the checkpoint weights loaded.

    Raises
    ------
    FileNotFoundError
        If either checkpoint is missing.
    """
    ck = Path(phase1_dir) / "checkpoints"
    rna_path, atac_path = ck / "rna_ae.pt", ck / "atac_ae.pt"
    for p in (rna_path, atac_path):
        if not p.exists():
            raise FileNotFoundError(f"Phase-1 checkpoint missing: {p}")
    rna_ae = ModalityAutoencoder(input_dim=rna_dim, latent_dim=latent_dim)
    atac_ae = ModalityAutoencoder(input_dim=atac_dim, latent_dim=latent_dim)
    rna_ae.load_state_dict(torch.load(rna_path, map_location="cpu")["model_state_dict"])
    atac_ae.load_state_dict(torch.load(atac_path, map_location="cpu")["model_state_dict"])
    log.info("Phase-1 trunks loaded from %s", phase1_dir)
    return rna_ae, atac_ae


@torch.no_grad()
def extract_latents(model: MergedVAE, rna_expr: np.ndarray, atac_expr: np.ndarray,
                    cfg: MergedVAEConfig, chunk: int = 4096,
                    ) -> dict[str, np.ndarray]:
    """Encode every cell to the posterior mean, plus one sample for diagnostics.

    Parameters
    ----------
    model : MergedVAE
        Trained model.
    rna_expr, atac_expr : numpy.ndarray
        Full feature matrices.
    cfg : MergedVAEConfig
        Run configuration (for device, seed and ``deterministic``).
    chunk : int, default 4096
        Rows encoded per forward pass.

    Returns
    -------
    dict of str to numpy.ndarray
        ``mu_rna``, ``mu_atac``, ``z_rna``, ``z_atac``. The means are the
        latent to use downstream; the samples serve the mean-vs-sample rank
        diagnostic (:func:`mtvae.stage_a.losses.geometry_report`).
    """
    model.eval()
    device = cfg.device
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 9_000)
    out: dict[str, np.ndarray] = {}
    for name, mat in (("rna", rna_expr), ("atac", atac_expr)):
        mus, zs = [], []
        for i in range(0, len(mat), chunk):
            xb = torch.from_numpy(np.asarray(mat[i:i + chunk], dtype=np.float32)).to(device)
            mu, logvar = model.posterior(xb, name)
            z = mu if cfg.deterministic else model.fusion.rsample(mu, logvar, gen)
            mus.append(mu.cpu().numpy())
            zs.append(z.cpu().numpy())
        out[f"mu_{name}"] = np.concatenate(mus, axis=0)
        out[f"z_{name}"] = np.concatenate(zs, axis=0)
    return out


def train_merged(
    rna_expr: np.ndarray,
    atac_expr: np.ndarray,
    rna_stages: np.ndarray,
    atac_stages: np.ndarray,
    n_stages: int,
    cfg: MergedVAEConfig,
    rna_teacher: Optional[np.ndarray] = None,
    atac_teacher: Optional[np.ndarray] = None,
    trunks: Optional[dict] = None,
    losses: Optional[tuple] = None,
) -> tuple[MergedVAE, dict[str, list]]:
    """Train the Stage-A model in either mode.

    Parameters
    ----------
    rna_expr, atac_expr : numpy.ndarray
        Feature matrices, ``(n_cells, n_features)``. ATAC clipped by the caller.
    rna_stages, atac_stages : numpy.ndarray
        Integer stage index per cell.
    n_stages : int
        Number of developmental stages.
    cfg : MergedVAEConfig
        Run configuration.
    rna_teacher, atac_teacher : numpy.ndarray, optional
        Fixed teacher embeddings for the tilt.  Required when the corresponding
        ``lambda_topo_*`` is positive.
    trunks : dict of str to TrunkSpec, optional
        Per-modality trunk declarations, ``{"rna": ..., "atac": ...}``, used in
        joint mode. ``None`` builds two plain ``ModalityAutoencoder``s, which
        is bit-identical to passing two ``TrunkSpec(kind="mlp")``. Specs
        rather than modules, because construction must happen inside this
        function, after seeding (see :mod:`mtvae.nn.trunks`).
    losses : tuple of str, optional
        Reconstruction loss of each modality, ``(rna, atac)``, each ``"mse"``
        or ``"huber"``. ``None`` uses
        :data:`mtvae.stage_a.losses.DEFAULT_SLOT_LOSSES`.

    Returns
    -------
    model : MergedVAE
        The trained model.
    history : dict of str to list
        Per-log-step term magnitudes and collapse diagnostics.

    Raises
    ------
    ValueError
        If the mode is unknown, or a tilt is enabled without its teacher.
    """
    if cfg.mode not in ("staged", "joint"):
        raise ValueError(f"Unknown mode: {cfg.mode!r}")
    if cfg.lambda_topo_rna > 0 and rna_teacher is None:
        raise ValueError("lambda_topo_rna > 0 requires rna_teacher")
    if cfg.lambda_topo_atac > 0 and atac_teacher is None:
        raise ValueError("lambda_topo_atac > 0 requires atac_teacher")

    device = cfg.device
    _loss_rna, _loss_atac = tuple(losses or DEFAULT_SLOT_LOSSES)
    # Losses are resolved before seeding, so a malformed ``losses`` argument
    # fails before any RNG is consumed. The global numpy seed also fixes
    # openTSNE's affinity, which takes no random_state and draws from it.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed % (2 ** 32))
    rng = np.random.default_rng(cfg.seed)

    # ---- trunks -------------------------------------------------------------
    if cfg.mode == "staged":
        # phase1_dir is a cache: load the trunks if it holds checkpoints,
        # otherwise fit them and save them there. Staged fits on the same data
        # and seed then share identical pre-trained trunks (immune to GPU
        # nondeterminism) and differ only in the main objective.
        cache = cfg.phase1_dir
        if cache is not None and (Path(cache) / "checkpoints" / "rna_ae.pt").exists():
            rna_ae, atac_ae = load_phase1(cache, rna_expr.shape[1],
                                          atac_expr.shape[1], cfg.latent_dim)
        else:
            rna_ae, atac_ae = pretrain_phase1(rna_expr, atac_expr, cfg)
            if cache is not None:
                _save_phase1(rna_ae, atac_ae, cache)
    else:
        log.info("Joint mode: trunks initialised from scratch, no pre-training")
        # Trunks are built here, after torch.manual_seed, RNA then ATAC. Both
        # the position and the order fix the initial weights: a trunk built by
        # the caller would start from a different RNG position. Hence
        # ``trunks`` carries specs, not modules.
        if trunks is None:
            rna_ae = ModalityAutoencoder(input_dim=rna_expr.shape[1],
                                         latent_dim=cfg.latent_dim)
            atac_ae = ModalityAutoencoder(input_dim=atac_expr.shape[1],
                                          latent_dim=cfg.latent_dim)
        else:
            from mtvae.nn import trunks as _tk
            rna_ae = _tk.build(trunks["rna"], input_dim=rna_expr.shape[1],
                               latent_dim=cfg.latent_dim)
            atac_ae = _tk.build(trunks["atac"], input_dim=atac_expr.shape[1],
                                latent_dim=cfg.latent_dim)

    fusion = StochasticFusion(
        cfg.latent_dim, hidden_dim=cfg.fusion_hidden_dim, depth=cfg.fusion_depth,
        residual_scale=cfg.fusion_residual_scale, layer_norm=cfg.fusion_layer_norm,
        arch=cfg.fusion_arch, sigma_mode=cfg.sigma_mode, sigma_init=cfg.sigma_init,
        stochastic_dims=cfg.kl_dims,
    )
    model = MergedVAE(rna_ae, atac_ae, fusion).to(device)

    # ---- optimiser ----------------------------------------------------------
    if cfg.mode == "joint":
        groups = [{"params": list(model.parameters()), "lr": cfg.joint_lr}]
    else:
        groups = [
            {"params": list(model.fusion.parameters()), "lr": cfg.lr},
            {"params": list(model.rna_ae.parameters()), "lr": cfg.encoder_lr},
            {"params": list(model.atac_ae.parameters()), "lr": cfg.encoder_lr},
        ]
    opt = Adam(groups, betas=(0.95, 0.99))
    n_train = sum(p.numel() for g in groups for p in g["params"])
    log.info("mode=%s trainable=%d beta_kl=%.4g kl_scale=%s free_bits=%.3g "
             "kl_dims=%d sigma_mode=%s deterministic=%s",
             cfg.mode, n_train, cfg.beta_kl, cfg.kl_scale, cfg.kl_free_bits,
             cfg.kl_dims, cfg.sigma_mode, cfg.deterministic)

    # ---- fixed tensors ------------------------------------------------------
    X_rna = torch.from_numpy(np.asarray(rna_expr, dtype=np.float32)).to(device)
    X_atac = torch.from_numpy(np.asarray(atac_expr, dtype=np.float32)).to(device)
    rna_pools = _stage_pools(rna_stages, n_stages)
    atac_pools = _stage_pools(atac_stages, n_stages)
    rna_all = np.arange(len(rna_expr))
    atac_all = np.arange(len(atac_expr))

    # Teacher affinities, precomputed once over all cells (topo_affinity="full").
    P_rna = P_atac = None
    T_rna = T_atac = None
    if cfg.lambda_topo_rna > 0:
        if cfg.topo_affinity == "full":
            P_rna = full_teacher_affinity(rna_teacher, cfg.topo_perplexity)
            log.info("teacher affinity [rna]: nnz=%d", P_rna.nnz)
        T_rna = torch.from_numpy(np.asarray(rna_teacher, dtype=np.float32)).to(device)
    if cfg.lambda_topo_atac > 0:
        if cfg.topo_affinity == "full":
            P_atac = full_teacher_affinity(atac_teacher, cfg.topo_perplexity)
            log.info("teacher affinity [atac]: nnz=%d", P_atac.nnz)
        T_atac = torch.from_numpy(np.asarray(atac_teacher, dtype=np.float32)).to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 1_000)
    sw_gen = torch.Generator(device=device)
    sw_gen.manual_seed(cfg.seed + 2_000)

    history: dict[str, list] = {
        "iter": [], "L_total": [], "L_recon_rna": [], "L_recon_atac": [],
        "L_sw": [], "L_topo_rna": [], "L_topo_atac": [], "L_kl": [],
        "kl_nats_per_sample": [], "active_units_rna": [], "sigma_mean": [],
        "recon_share": [],
    }

    def _tilt(zb: torch.Tensor, P, teacher_t, idx) -> torch.Tensor:
        """Tilt loss for one batch, on the precomputed or per-batch affinity.

        Honours ``topo_identity_dims``: with ``k > 0`` the Student-t affinity is
        built on ``zb[:, :k]`` only, so the tilt gradient never reaches the
        other dimensions. ``0`` tilts the full latent.
        """
        k = cfg.topo_identity_dims
        z_for_tilt = zb if k <= 0 else zb[:, :k]
        if P is not None:
            return tilt_loss_from_P(
                z_for_tilt,
                subset_affinity(P, idx, device=device, dtype=z_for_tilt.dtype))
        return topology_tilt_loss(z_for_tilt, teacher_t[idx],
                                  perplexity=cfg.topo_perplexity)

    t0 = time.time()
    for step in range(1, cfg.n_iters + 1):
        model.train()
        warming = cfg.mode == "joint" and step <= cfg.joint_warmup_iters

        # ---- (1) reconstruction + KL, on uniform per-modality batches -------
        i_rna = rng.choice(rna_all, size=min(cfg.batch_size_per_modality,
                                             len(rna_all)), replace=False)
        i_atac = rng.choice(atac_all, size=min(cfg.batch_size_per_modality,
                                               len(atac_all)), replace=False)
        ir = torch.from_numpy(i_rna).to(device)
        ia = torch.from_numpy(i_atac).to(device)

        xr, xa = X_rna[ir], X_atac[ia]
        rec_r, _, mu_r, lv_r = model(xr, "rna", gen, cfg.deterministic)
        rec_a, _, mu_a, lv_a = model(xa, "atac", gen, cfg.deterministic)
        # Per-modality reconstruction losses, as declared (``losses``).
        L_recon_rna = reconstruction_loss(_loss_rna, rec_r, xr)
        L_recon_atac = reconstruction_loss(_loss_atac, rec_a, xa)

        # The KL is computed whenever the model is stochastic, even at
        # beta_kl = 0, so such a fit reports the KL it would carry; that value
        # can be used to calibrate beta_kl against the other terms.
        if not cfg.deterministic:
            kl_r, kld_r = gaussian_kl(mu_r, lv_r, model.fusion.stoch_mask,
                                      cfg.kl_free_bits, cfg.kl_scale)
            kl_a, _ = gaussian_kl(mu_a, lv_a, model.fusion.stoch_mask,
                                  cfg.kl_free_bits, cfg.kl_scale)
            L_kl = kl_r + kl_a
        else:
            L_kl = torch.zeros((), device=device)
            kld_r = torch.zeros(cfg.latent_dim, device=device)

        # ---- (2) per-stage sliced Wasserstein -------------------------------
        L_sw = torch.zeros((), device=device)
        if cfg.beta_sw > 0 and not warming:
            n_used = 0
            for s in range(n_stages):
                pr, pa = rna_pools[s], atac_pools[s]
                # sliced_wasserstein_2 asserts equal batch sizes, so a stage that
                # is short on either side is drawn down to the smaller count.
                k = min(cfg.sw_batch_per_stage, len(pr), len(pa))
                if k < 2:
                    continue
                jr = rng.choice(pr, size=k, replace=False)
                ja = rng.choice(pa, size=k, replace=False)
                _, zr_s, zr_m, _ = model(X_rna[torch.from_numpy(jr).to(device)],
                                         "rna", gen, cfg.deterministic)
                _, za_s, za_m, _ = model(X_atac[torch.from_numpy(ja).to(device)],
                                         "atac", gen, cfg.deterministic)
                zr, za = ((zr_m, za_m) if cfg.sw_on == "mu" else (zr_s, za_s))
                L_sw = L_sw + sliced_wasserstein_2(
                    zr, za, n_projections=cfg.sw_n_projections, generator=sw_gen)
                n_used += 1
            if n_used:
                L_sw = L_sw / n_used

        # ---- (3) MIND tilt, one independent batch per modality --------------
        L_topo_rna = torch.zeros((), device=device)
        L_topo_atac = torch.zeros((), device=device)
        _ti = 2 if cfg.tilt_on == "mu" else 1        # forward() -> (x_rec, z, mu, logvar)
        if cfg.lambda_topo_rna > 0 and not warming:
            k = rng.choice(rna_all, size=min(cfg.topo_batch, len(rna_all)),
                           replace=False)
            zt = model(X_rna[torch.from_numpy(k).to(device)], "rna",
                       gen, cfg.deterministic)[_ti]
            L_topo_rna = _tilt(zt, P_rna, T_rna, k)
        if cfg.lambda_topo_atac > 0 and not warming:
            k = rng.choice(atac_all, size=min(cfg.topo_batch, len(atac_all)),
                           replace=False)
            zt = model(X_atac[torch.from_numpy(k).to(device)], "atac",
                       gen, cfg.deterministic)[_ti]
            L_topo_atac = _tilt(zt, P_atac, T_atac, k)

        # ---- (4) assemble ---------------------------------------------------
        L = (L_recon_rna + L_recon_atac
             + cfg.beta_sw * L_sw
             + cfg.lambda_topo_rna * L_topo_rna
             + cfg.lambda_topo_atac * L_topo_atac
             + cfg.beta_kl * L_kl)

        opt.zero_grad(set_to_none=True)
        L.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], cfg.grad_clip)
        opt.step()

        # ---- (5) diagnostics -------------------------------------------------
        if step % cfg.log_every == 0 or step in (1, cfg.n_iters):
            recon = float(L_recon_rna + L_recon_atac)
            total = float(L)
            sigma_mean = float(torch.exp(0.5 * lv_r).mean()) if not cfg.deterministic else 0.0
            au = active_units(kld_r)
            history["iter"].append(step)
            history["L_total"].append(total)
            history["L_recon_rna"].append(float(L_recon_rna))
            history["L_recon_atac"].append(float(L_recon_atac))
            history["L_sw"].append(float(L_sw))
            history["L_topo_rna"].append(float(L_topo_rna))
            history["L_topo_atac"].append(float(L_topo_atac))
            history["L_kl"].append(float(L_kl))
            history["kl_nats_per_sample"].append(float(kld_r.sum()))
            history["active_units_rna"].append(au)
            history["sigma_mean"].append(sigma_mean)
            history["recon_share"].append(recon / total if total else float("nan"))
            log.info(
                "iter %5d L %.4f | rec %.4f (%.1f%%) sw %.4f topo %.3f/%.3f "
                "kl %.5f | nats %.3f AU %d sigma %.4f",
                step, total, recon, 100.0 * recon / total if total else float("nan"),
                float(L_sw), float(L_topo_rna), float(L_topo_atac), float(L_kl),
                float(kld_r.sum()), au, sigma_mean,
            )

    log.info("training done in %.1f s", time.time() - t0)
    return model, history


def run_record(cfg: MergedVAEConfig, history: dict[str, list],
               latents: dict[str, np.ndarray],
               extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Assemble a JSON-serialisable record of a Stage-A fit.

    Parameters
    ----------
    cfg : MergedVAEConfig
        Fully resolved configuration.
    history : dict
        Training history from :func:`train_merged`.
    latents : dict of str to numpy.ndarray
        Output of :func:`extract_latents`.
    extra : dict, optional
        Additional fields (dataset code, seed, timings, input fingerprints).

    Returns
    -------
    dict
        JSON-serialisable record including the mean-vs-sample rank diagnostics.
    """
    geom = geometry_report(latents["mu_rna"], latents["mu_atac"],
                           latents["z_rna"], latents["z_atac"])
    rec: dict[str, Any] = {
        "hyperparameters": cfg.resolved(),
        "geometry": geom,
        "final": {k: (v[-1] if v else None) for k, v in history.items()},
        "history": history,
    }
    if extra:
        rec.update(extra)
    return rec
