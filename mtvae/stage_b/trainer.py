"""Stage-B training: a latent SDE co-adapted with the encoder and decoders.
Loss: latent transport, reconstruction, feature Sinkhorn, decoder proximity, anchor, path KL."""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


from mtvae.stage_b.encoder import StageAEncoder  # noqa: E402
from mtvae.stage_b.losses import (  # noqa: E402
    sinkhorn_div_shared_scale,
)
from mtvae.stage_b.sinkhorn import (  # noqa: E402
    sinkhorn_div, sinkhorn_div_scaled,
)

log = logging.getLogger("mtvae.stage_b.trainer")

#: Fraction of the Stage-A effective rank below which :func:`collapse_verdict`
#: reports a collapse. Fixed in advance, because alignment metrics do not
#: penalise (and can reward) rank collapse.
COLLAPSE_FLOOR: float = 0.80


def participation_ratio(z: torch.Tensor) -> float:
    """Effective rank of a batch of codes; torch version of
    :func:`mtvae.rank.eff_rank`.

    ``(sum lambda)^2 / sum lambda^2`` over the eigenvalues of the covariance of
    the column-centred codes, obtained as squared singular values, in float64.

    Parameters
    ----------
    z : torch.Tensor, shape (n, d)
        Codes to measure.

    Returns
    -------
    float
        Participation ratio in ``[1, d]``, or ``nan`` for a degenerate input.
    """
    x = z.detach().to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x) ** 2
    denom = (sv ** 2).sum()
    return float((sv.sum() ** 2) / denom) if float(denom) > 0 else float("nan")


def collapse_verdict(eff_rank_stage_a: float, eff_rank_coadapt: float,
                     floor: float = COLLAPSE_FLOOR) -> tuple[str, str]:
    """Collapse check on the encoder latent after Stage B.

    Parameters
    ----------
    eff_rank_stage_a, eff_rank_coadapt : float
        Effective rank of the latent before and after co-adaptation.
    floor : float, default COLLAPSE_FLOOR
        Minimum retained fraction.

    Returns
    -------
    verdict : {"OK", "COLLAPSED"}
        ``"COLLAPSED"`` if less than ``floor`` of the Stage-A effective rank
        is retained, or the Stage-A value is not usable.
    detail : str
        Human-readable summary.
    """
    if not np.isfinite(eff_rank_stage_a) or eff_rank_stage_a <= 0:
        return "COLLAPSED", f"Stage-A eff_rank not usable ({eff_rank_stage_a})"
    frac = eff_rank_coadapt / eff_rank_stage_a
    if frac < floor:
        return "COLLAPSED", (
            f"eff_rank {eff_rank_stage_a:.3f} -> {eff_rank_coadapt:.3f} "
            f"({frac:.1%} retained, floor {floor:.0%}). The latent collapsed; "
            f"treat the fit as failed.")
    return "OK", (f"eff_rank {eff_rank_stage_a:.3f} -> {eff_rank_coadapt:.3f} "
                  f"({frac:.1%} retained)")


def beta_kl_schedule(step: int, beta_kl: float,
                     kl_warmup: Tuple[int, int]) -> float:
    """Path-KL weight at ``step``: zero, then a linear ramp, then ``beta_kl``.

    Parameters
    ----------
    step : int
        1-based training iteration.
    beta_kl : float
        Weight reached at the end of the ramp.
    kl_warmup : tuple of int
        ``(start, end)``: weight 0 for ``step < start``, ``beta_kl`` for
        ``step >= end``, linear in between.

    Returns
    -------
    float
        The weight to apply at ``step``.
    """
    start, end = kl_warmup
    if step < start:
        return 0.0
    if step >= end:
        return float(beta_kl)
    return float(beta_kl * (step - start) / max(1, end - start))


def train_sde_coadapt(
    sde,
    decoders: Dict[str, nn.Module],
    encoder: StageAEncoder,
    z_stage_a: Dict[str, torch.Tensor],
    x_by_mod: Dict[str, torch.Tensor],
    t_idx_by_mod: Dict[str, np.ndarray],
    train_mask_by_mod: Dict[str, np.ndarray],
    stage_norm: np.ndarray,
    *,
    alpha_align: float = 1.0,
    lambda_chained: float = 1.0,
    lambda_per_pair: float = 2.0,
    lambda_rec: float = 1.0,
    lambda_dist: float = 1.0,
    lambda_prox: float = 1.0,
    lambda_latent_anchor: float = 1.0,
    beta_kl: float = 0.0,
    kl_warmup: Tuple[int, int] = (200, 400),
    batch_align: int = 256,
    batch_rec: int = 256,
    batch_dist: int = 256,
    batch_anchor: int = 256,
    train_iters: int = 3000,
    warmup_iters: int = 200,
    lr: float = 5e-4,
    decoder_lr: float = 1e-5,
    encoder_lr: float = 0.0,
    scaled_z_sinkhorn: bool = True,
    sinkhorn_eps_x: float = 0.05,
    sinkhorn_iters_x: int = 20,
    shipped_divergence: bool = False,
    seed: int = 22,
    device: str = "cpu",
    log_every: int = 50,
    target_clip: Optional[Dict[str, float]] = None,
) -> Dict[str, list]:
    """Train the SDE while the encoder and decoders co-adapt.

    The model slots are ``"rna"`` and ``"atac"``; every per-modality argument
    is keyed by them. ``sde``, ``decoders`` and ``encoder`` are trained in
    place.

    Parameters
    ----------
    sde : SDESampler
        Latent dynamics.
    decoders : dict of str to torch.nn.Module
        Per-modality decoders, separate instances from the encoder's trunks.
    encoder : StageAEncoder
        The Stage-A encoder path. It must already have passed
        :func:`mtvae.stage_b.encoder.verify_reproduces_stage_a`: this function
        moves the encoder, so the check cannot run afterwards, and the anchor
        is meaningless if the encoder does not reproduce ``z_stage_a``.
    z_stage_a : dict of str to torch.Tensor
        The frozen Stage-A latents (posterior means); the anchor target, and
        the observed latents when the encoder is frozen.
    x_by_mod : dict of str to torch.Tensor
        Feature matrices, at the clip the encoder was trained on. Placing them
        on the GPU beforehand avoids per-batch host-to-device copies.
    t_idx_by_mod : dict of str to numpy.ndarray
        Stage index of every cell.
    train_mask_by_mod : dict of str to numpy.ndarray
        Boolean mask of the cells Stage B may draw.
    stage_norm : numpy.ndarray
        Normalised stage times, one per stage.
    alpha_align : float, default 1.0
        Weight on the two latent transport terms together.
    lambda_chained, lambda_per_pair : float, default 1.0, 2.0
        Weights on ``L_chain_z`` and ``L_pair_z``.
    lambda_rec : float, default 1.0
        Weight on ``L_rec``; 0 skips the term.
    lambda_dist : float, default 1.0
        Weight on the feature-space term ``L_dist``.
    lambda_prox : float, default 1.0
        Weight on ``L_prox`` (used only when the decoders train).
    lambda_latent_anchor : float, default 1.0
        Weight on ``||z_enc(x) - z_stage_a||^2``. With a trainable encoder, 0
        is the unidentifiable control, not a neutral default; the drift is
        still measured and logged.
    beta_kl : float, default 0.0
        Weight on the SDE path KL ``0.5 * int |(f_post - f_prior) / g|^2 dt``,
        summed over the chained and per-pair rollouts. 0 removes the term.
        Not the Stage-A ``beta_kl``.
    kl_warmup : tuple of int, default (200, 400)
        ``(start, end)`` of the linear ramp; see :func:`beta_kl_schedule`.
    batch_align, batch_rec, batch_dist, batch_anchor : int, default 256
        Cells per draw for the transport, reconstruction, feature-space and
        anchor terms.
    train_iters : int, default 3000
        Optimisation steps.
    warmup_iters : int, default 200
        Steps during which the chained rollout always starts at stage 0.
    lr : float, default 5e-4
        SDE learning rate.
    decoder_lr : float, default 1e-5
        Decoder learning rate; 0 freezes the decoders.
    encoder_lr : float, default 0.0
        Encoder learning rate; 0 freezes the encoder (the observed latents
        are then ``z_stage_a``).
    scaled_z_sinkhorn : bool, default True
        Use :func:`~mtvae.stage_b.sinkhorn.sinkhorn_div_scaled` (with
        ``sinkhorn_eps_x``, ``sinkhorn_iters_x`` and cost scaling) for the
        latent terms; otherwise
        :func:`~mtvae.stage_b.sinkhorn.sinkhorn_div` with its defaults.
    sinkhorn_eps_x : float, default 0.05
        Entropic regularisation of the scaled Sinkhorn divergences.
    sinkhorn_iters_x : int, default 20
        Sinkhorn iterations of the scaled Sinkhorn divergences.
    shipped_divergence : bool, default False
        Use :func:`~mtvae.stage_b.sinkhorn.sinkhorn_div_scaled` for
        ``L_dist`` instead of
        :func:`~mtvae.stage_b.losses.sinkhorn_div_shared_scale`. The former
        can reward a collapsed decoder (see :mod:`mtvae.stage_b.losses`).
    seed : int, default 22
        Seed of the batch draws (numpy) and the Brownian increments (torch).
    device : str, default "cpu"
        Compute device.
    log_every : int, default 50
        Logging interval in steps (step 1 and the last step are always
        logged).
    target_clip : dict of str to float, optional
        Per-modality clip applied to reconstruction targets only. Once the
        encoder trains, ``x`` is also its input, so ``x_by_mod`` must stay at
        the clip the encoder was trained on; a tighter target clip is applied
        per batch here (exact, since ``clip(clip(x, a), b) == clip(x, b)`` for
        ``b <= a``).

    Returns
    -------
    dict of str to list
        History at each logged step: the loss terms (``L_chain_z``,
        ``L_pair_z``, ``L_rec``, ``L_dist``, ``L_prox``, ``L_anchor``,
        ``L_total``), ``eff_rank_zt`` and ``eff_rank_z_obs`` (effective rank
        of a propagated and an observed batch), ``z_drift`` and
        ``anchor_share``, and the path-KL diagnostics: ``L_pathKL`` (logged at
        every ``beta_kl``, including 0), ``beta_kl_t``, ``L_task`` (the loss
        without the KL term), ``s_start`` (the chained rollout's start stage,
        which sets the horizon of that step's KL), ``sigma_t0/_t05/_t1`` (the
        diffusion at t = 0, 0.5, 1) and ``grad_norm`` (the global norm before
        clipping at 5).

    Raises
    ------
    ValueError
        If ``beta_kl`` is negative or ``kl_warmup`` is not an ordered pair.
    RuntimeError
        At the first step the path KL is live, if it carries no gradient or is
        not finite; at step 1, if a requested encoder or decoder learning rate
        produced no nonzero gradient; and, when ``beta_kl > 0``, at a logged
        step whose loss is not finite.
    """
    mods = ("rna", "atac")
    n_stages = len(stage_norm)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    train_encoder = encoder_lr > 0.0
    train_decoder = decoder_lr > 0.0

    beta_kl = float(beta_kl)
    kl_warmup = (int(kl_warmup[0]), int(kl_warmup[1]))
    if beta_kl < 0.0:
        raise ValueError(f"beta_kl must be >= 0, got {beta_kl}")
    if not 0 <= kl_warmup[0] <= kl_warmup[1]:
        raise ValueError(f"kl_warmup must be (start, end) with 0 <= start <= "
                         f"end, got {kl_warmup}")
    # Set at the first step the term is live; the gradient check runs once.
    kl_live = False
    # A beta_kl > 0 whose ramp never starts within the fit trains the beta = 0
    # objective. Warn (before and after the loop) rather than raise: a fit
    # deliberately stopped before the ramp is a legitimate comparison.
    kl_never_live = beta_kl > 0.0 and all(
        beta_kl_schedule(s, beta_kl, kl_warmup) == 0.0
        for s in range(1, train_iters + 1))
    if kl_never_live:
        log.warning("beta_kl=%g never goes live: ramp %s does not start within "
                    "train_iters=%d, so this fit IS the beta_kl=0 objective.",
                    beta_kl, kl_warmup, train_iters)
    elif beta_kl > 0.0 and kl_warmup[1] > train_iters:
        log.warning("path-KL ramp %s ends after train_iters=%d: the weight never "
                    "reaches beta_kl=%g.", kl_warmup, train_iters, beta_kl)

    stage_idx = {m: {s: np.where((t_idx_by_mod[m] == s) & train_mask_by_mod[m])[0]
                     for s in range(n_stages)} for m in mods}
    stage_norm_t = torch.as_tensor(stage_norm, dtype=torch.float32, device=device)

    # Initial decoder weights, the reference of L_prox.
    ref_decoders = {m: [p.detach().clone() for p in decoders[m].parameters()]
                    for m in mods}

    groups = [{"params": list(sde.parameters()), "lr": lr}]
    if train_decoder:
        for m in mods:
            # requires_grad is set here: the loaders and the handoff return the
            # decoders frozen, and a frozen parameter in an Adam group never
            # receives a gradient, so decoder_lr would silently do nothing.
            # .train() is numerically inert (Linear + SiLU) but keeps the
            # module state consistent.
            for p in decoders[m].parameters():
                p.requires_grad_(True)
            decoders[m].train()
            groups.append({"params": list(decoders[m].parameters()), "lr": decoder_lr})
    else:
        for m in mods:
            for p in decoders[m].parameters():
                p.requires_grad_(False)
            decoders[m].eval()
    if train_encoder:
        groups.append({"params": list(encoder.parameters()), "lr": encoder_lr})
    else:
        for p in encoder.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    # Where the feature matrices live; see _rows() below.
    _x_on_gpu = {m: x_by_mod[m].is_cuda for m in mods}
    log.info("co-adapt: encoder_lr=%.3g (train=%s) decoder_lr=%.3g (train=%s) "
             "sde_lr=%.3g | latent_anchor=%.3g | path-KL beta=%.3g ramp %s%s "
             "| x on gpu: %s",
             encoder_lr, train_encoder, decoder_lr, train_decoder, lr,
             lambda_latent_anchor, beta_kl, kl_warmup,
             (" (off)" if beta_kl == 0.0 else
              " (NEVER LIVE in this fit)" if kl_never_live else ""),
             _x_on_gpu)
    if train_encoder and lambda_latent_anchor == 0.0:
        log.warning("encoder is training with NO latent anchor — this is the "
                    "unidentifiable control; expect and check for collapse.")

    def _draw(mod: str, s: int, n: int) -> np.ndarray:
        """Draw ``min(n, pool size)`` training cells of stage ``s``.

        Sampling is with replacement only when the pool is smaller than ``n``.
        """
        pool = stage_idx[mod][s]
        return rng.choice(pool, size=min(n, len(pool)), replace=len(pool) < n)

    # Where the feature matrices live decides where the per-draw gather
    # happens, which dominates the loop's cost once the encoder trains (about
    # a dozen feature batches per step). With ``x`` on the GPU the gather is a
    # device-side copy and the host-to-device traffic disappears. The result is
    # identical either way: a gather is a pure copy, and the indices come from
    # the same numpy RNG.
    def _rows(mod: str, idx: np.ndarray) -> torch.Tensor:
        """Gather feature rows, on whichever device the matrix lives."""
        x = x_by_mod[mod]
        if x.is_cuda:
            return x[torch.as_tensor(idx, device=x.device, dtype=torch.long)]
        return x[idx].to(device)

    def _tgt(mod: str, idx: np.ndarray) -> torch.Tensor:
        """Reconstruction target for a batch, at the target clip."""
        x = _rows(mod, idx)
        c = (target_clip or {}).get(mod)
        return x if c is None else x.clamp(-c, c)

    def _z(mod: str, idx: np.ndarray) -> torch.Tensor:
        """Observed latent for a batch: through the encoder if it is training."""
        if train_encoder:
            return encoder(_rows(mod, idx), mod)
        return z_stage_a[mod][idx].to(device)

    def _z_sink(a, b):
        """Sinkhorn divergence for the latent transport terms."""
        if scaled_z_sinkhorn:
            return sinkhorn_div_scaled(a, b, eps=sinkhorn_eps_x,
                                       n_iters=sinkhorn_iters_x, scale_cost=True)
        return sinkhorn_div(a, b)

    keys = ("iter", "L_chain_z", "L_pair_z", "L_rec", "L_dist", "L_prox",
            "L_anchor", "L_total", "eff_rank_zt", "eff_rank_z_obs", "z_drift",
            "anchor_share",
            # Path-KL diagnostics: RNG-free and read under no_grad, so they
            # are logged at beta_kl = 0 too.
            "L_task", "L_pathKL", "beta_kl_t", "s_start", "sigma_t0",
            "sigma_t05", "sigma_t1", "grad_norm")
    history: Dict[str, list] = {k: [] for k in keys}

    for step in range(1, train_iters + 1):
        sde.train()

        # ---- transport, chained from s_start, matched in latent space ----
        mod = mods[0] if rng.random() < 0.5 else mods[1]
        s_start = 0 if step <= warmup_iters else int(rng.integers(0, n_stages - 1))
        z_src = _z(mod, _draw(mod, s_start, batch_align))
        ts = stage_norm_t[s_start:]
        # The integrator accumulates the KL whether or not it is used, so
        # keeping it costs no RNG and no extra rollout. Never compute it with a
        # second forward_chain: that draws Brownian noise and shifts every
        # later batch and increment.
        traj, kl_chain = sde.forward_chain(z_src, ts, generator=gen)
        L_chain_z = torch.zeros((), device=device)
        for k in range(1, len(ts)):
            z_obs = _z(mod, _draw(mod, s_start + k, batch_align))
            L_chain_z = L_chain_z + _z_sink(traj[k], z_obs)

        s2 = int(rng.integers(0, n_stages - 1))
        z_src2 = _z(mod, _draw(mod, s2, batch_align))
        ts2 = torch.stack([stage_norm_t[s2], stage_norm_t[s2 + 1]])
        traj2, kl_pair = sde.forward_chain(z_src2, ts2, generator=gen)
        z_tgt2 = _z(mod, _draw(mod, s2 + 1, batch_align))
        L_pair_z = _z_sink(traj2[-1], z_tgt2)

        # ---- reconstruction. With a trainable encoder this term reaches the
        # encoder too, making it an autoencoder term that, with the anchor,
        # guards against latent collapse.
        L_rec = torch.zeros((), device=device)
        if lambda_rec > 0.0:
            for mr in mods:
                i_r = _draw(mr, int(rng.integers(0, n_stages)), batch_rec)
                L_rec = L_rec + ((decoders[mr](_z(mr, i_r))
                                  - _tgt(mr, i_r)) ** 2).mean()

        # ---- decoded distributional match at a propagated stage ----
        mod_d = mods[0] if rng.random() < 0.5 else mods[1]
        s_d = int(rng.integers(1, n_stages))
        z0_d = _z(mod_d, _draw(mod_d, 0, batch_dist))
        x_obs_d = _tgt(mod_d, _draw(mod_d, s_d, batch_dist))
        ts_d = torch.stack([stage_norm_t[0], stage_norm_t[s_d]])
        traj_d, _ = sde.forward_chain(z0_d, ts_d, generator=gen)
        z_t_d = traj_d[-1]
        x_pred_d = decoders[mod_d](z_t_d)
        if shipped_divergence:
            L_dist = sinkhorn_div_scaled(x_pred_d, x_obs_d, eps=sinkhorn_eps_x,
                                         n_iters=sinkhorn_iters_x, scale_cost=True)
        else:
            L_dist = sinkhorn_div_shared_scale(x_pred_d, x_obs_d,
                                               eps=sinkhorn_eps_x,
                                               n_iters=sinkhorn_iters_x)

        # ---- decoder weight anchor ----
        L_prox = torch.zeros((), device=device)
        if train_decoder and lambda_prox > 0.0:
            for mp in mods:
                for p, p_ref in zip(decoders[mp].parameters(), ref_decoders[mp]):
                    L_prox = L_prox + ((p - p_ref) ** 2).sum()

        # ---- LATENT anchor: the identifiability term ----
        # Measured whenever the encoder trains, applied only when lambda > 0.
        # In the lambda = 0 control it reports how far the unrestrained encoder
        # drifts, which sets the scale for choosing lambda.
        L_anchor = torch.zeros((), device=device)
        drift = float("nan")
        if train_encoder:
            grad_on = lambda_latent_anchor > 0.0
            with torch.set_grad_enabled(grad_on):
                acc = torch.zeros((), device=device)
                for ma in mods:
                    i_a = _draw(ma, int(rng.integers(0, n_stages)), batch_anchor)
                    z_now = encoder(_rows(ma, i_a), ma)
                    z_ref = z_stage_a[ma][i_a].to(device)
                    acc = acc + ((z_now - z_ref) ** 2).sum(1).mean()
            drift = float(acc.detach().cpu())
            if grad_on:
                L_anchor = acc

        L = (alpha_align * (lambda_chained * L_chain_z + lambda_per_pair * L_pair_z)
             + lambda_rec * L_rec
             + lambda_dist * L_dist
             + lambda_prox * L_prox
             + lambda_latent_anchor * L_anchor)
        L_task = L

        # ---- path KL ----
        # Added only while its weight is positive, never as 0.0 * KL, so at
        # beta_t = 0 the graph is exactly the one without the term. The KL of
        # the decoded rollout (traj_d) is not used.
        beta_t = beta_kl_schedule(step, beta_kl, kl_warmup)
        if beta_t > 0.0:
            L_pathKL = kl_chain[-1].mean() + kl_pair[-1].mean()
            if not kl_live:
                # A KL with no graph would leave the fit silently identical
                # to beta_kl = 0.
                if not L_pathKL.requires_grad:
                    raise RuntimeError(
                        f"beta_kl={beta_kl:g} is live at step {step} but the "
                        f"path-KL carries no gradient - the flag is inert.")
                if not bool(torch.isfinite(L_pathKL)):
                    raise RuntimeError(
                        f"path-KL is not finite at step {step} "
                        f"({float(L_pathKL)}); the diffusion may have collapsed.")
                kl_live = True
                log.info("path-KL live from step %d (beta_t %.3g, KL %.4g)",
                         step, beta_t, float(L_pathKL))
            L = L + beta_t * L_pathKL

        opt.zero_grad(set_to_none=True)
        L.backward()
        if step == 1:
            # Check that a requested encoder/decoder learning rate actually
            # trains something: the banner above reports train=True from the
            # lr alone, and an ignored lr would leave no other trace.
            for on, parts, name, lr_v in (
                    (train_decoder, [decoders[m] for m in mods], "decoder", decoder_lr),
                    (train_encoder, [encoder], "encoder", encoder_lr)):
                if not on:
                    continue
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                           for part in parts for p in part.parameters()):
                    raise RuntimeError(
                        f"{name}_lr={lr_v:g} was requested but not one {name} "
                        f"parameter received a nonzero gradient - the flag is "
                        f"inert and this run would silently train a frozen "
                        f"{name}.")
        # The return value is the pre-clip global norm. It is logged because
        # the clip is global over SDE + decoders + encoder: a large KL gradient
        # shrinks every other update with it.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], 5.0)
        opt.step()

        if step % log_every == 0 or step in (1, train_iters):
            with torch.no_grad():
                eff_zt = participation_ratio(z_t_d.detach())
                eff_obs = participation_ratio(z0_d.detach())
                # Read at every beta, trained on only when beta_t > 0. One
                # batch, whose horizon is set by s_start.
                kl_now = float(kl_chain[-1].mean() + kl_pair[-1].mean())
                # g depends on t alone; one row is enough.
                z_probe = z_t_d.detach()[:1]
                sig = [float(sde.g_net(z_probe, t)[0, 0]) for t in (0.0, 0.5, 1.0)]
            if beta_kl > 0.0 and not np.isfinite(float(L)):
                raise RuntimeError(
                    f"non-finite loss at step {step} with beta_kl={beta_kl:g} "
                    f"(task {float(L_task)}, KL {kl_now}, sigma {sig}); stopping "
                    f"before a NaN latent reaches the collapse gate, which "
                    f"would pass it.")
            for k, v in (("iter", step), ("L_chain_z", float(L_chain_z)),
                         ("L_pair_z", float(L_pair_z)), ("L_rec", float(L_rec)),
                         ("L_dist", float(L_dist)), ("L_prox", float(L_prox)),
                         ("L_anchor", float(L_anchor)), ("L_total", float(L)),
                         ("eff_rank_zt", eff_zt), ("eff_rank_z_obs", eff_obs),
                         ("z_drift", drift),
                         ("anchor_share", (lambda_latent_anchor * drift / float(L))
                          if (drift == drift and float(L) > 0) else float("nan")),
                         ("L_task", float(L_task)), ("L_pathKL", kl_now),
                         ("beta_kl_t", beta_t), ("s_start", s_start),
                         ("sigma_t0", sig[0]), ("sigma_t05", sig[1]),
                         ("sigma_t1", sig[2]), ("grad_norm", float(grad_norm))):
                history[k].append(v)
            share = (lambda_latent_anchor * drift / float(L)
                     if (drift == drift and float(L) > 0) else float("nan"))
            log.info("iter %5d | L %.4f | chain %.4f pair %.4f rec %.4f dist %.4f "
                     "prox %.3g | anchor %.4f (share %.1f%%) | eff_rank z_obs "
                     "%.2f z_t %.2f | KL %.4g x beta %.3g | sigma(0,.5,1) "
                     "%.3g %.3g %.3g | gnorm %.3g", step, float(L),
                     float(L_chain_z), float(L_pair_z), float(L_rec),
                     float(L_dist), float(L_prox), drift, 100 * share, eff_obs,
                     eff_zt, kl_now, beta_t, sig[0], sig[1], sig[2],
                     float(grad_norm))
    if beta_kl > 0.0 and not kl_live:
        log.warning("beta_kl=%g was requested but the path-KL never entered the "
                    "loss (ramp %s, train_iters %d): this fit is the beta_kl=0 "
                    "objective.", beta_kl, kl_warmup, train_iters)
    return history
