"""Fit both stages of mtvae for one seed, from a YAML config.
Writes model.pt, latents/, the Stage-A snapshot and run_record.json to <out-root>/<arm>/."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


from mtvae import _constants as config  # noqa: E402
from mtvae.stage_b.handoff import RNG_PROTOCOLS  # noqa: E402
from mtvae.stage_b.merged_loader import trunk_meta  # noqa: E402
from mtvae.stage_b.trainer import collapse_verdict, participation_ratio  # noqa: E402
from mtvae import config as config_io  # noqa: E402
from mtvae import pipeline as suite_model  # noqa: E402
from mtvae.pipeline import (  # noqa: E402
    EXAMPLES, PipelineConfig, build_dataset,
)

log = logging.getLogger("mtvae.fit")

#: Stage-A flags: (cli-name, MergedVAEConfig field, type). An unset flag
#: inherits from the config file or the preset, never from an argparse default.
_VAE_FLAGS = (
    ("--vae-n-iters", "n_iters", int),
    ("--vae-batch-size", "batch_size_per_modality", int),
    ("--vae-beta-kl", "beta_kl", float),
    ("--vae-beta-sw", "beta_sw", float),
    ("--vae-lambda-topo-rna", "lambda_topo_rna", float),
    ("--vae-lambda-topo-atac", "lambda_topo_atac", float),
    ("--vae-lr", "lr", float),
    ("--vae-encoder-lr", "encoder_lr", float),
    ("--vae-joint-lr", "joint_lr", float),
    ("--vae-grad-clip", "grad_clip", float),
    ("--vae-topo-batch", "topo_batch", int),
    ("--vae-sigma-init", "sigma_init", float),
    ("--vae-log-every", "log_every", int),
    ("--latent-dim", "latent_dim", int),
)

def _int_pair(s: str) -> Tuple[int, int]:
    """Parse ``"200,400"`` into ``(200, 400)`` for ``--sde-kl-warmup``.

    Parameters
    ----------
    s : str
        Two comma-separated integers.

    Returns
    -------
    tuple of int
    """
    a, b = (int(v) for v in s.split(","))
    return (a, b)


#: Stage-B flags: (cli-name, train_sde_coadapt kwarg, type). An unset flag
#: inherits from the config file or the example recipe.
_SDE_FLAGS = (
    ("--sde-encoder-lr", "encoder_lr", float),
    ("--sde-decoder-lr", "decoder_lr", float),
    ("--sde-lr", "lr", float),
    ("--lambda-dist", "lambda_dist", float),
    ("--lambda-rec", "lambda_rec", float),
    ("--lambda-prox", "lambda_prox", float),
    ("--lambda-latent-anchor", "lambda_latent_anchor", float),
    # Prefixed with --sde-: the Stage-A prior KL (--vae-beta-kl) and the SDE
    # path-KL are different terms.
    ("--sde-beta-kl", "beta_kl", float),
    ("--sde-kl-warmup", "kl_warmup", _int_pair),
    ("--alpha-align", "alpha_align", float),
    ("--lambda-chained", "lambda_chained", float),
    ("--lambda-per-pair", "lambda_per_pair", float),
    ("--batch-align", "batch_align", int),
    ("--batch-rec", "batch_rec", int),
    ("--batch-dist", "batch_dist", int),
    ("--batch-anchor", "batch_anchor", int),
    ("--sde-train-iters", "train_iters", int),
    ("--sde-warmup-iters", "warmup_iters", int),
    ("--sinkhorn-eps-x", "sinkhorn_eps_x", float),
    ("--sinkhorn-iters-x", "sinkhorn_iters_x", int),
    ("--sde-log-every", "log_every", int),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (both examples, both stages)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("what to run")
    g.add_argument("--config", type=Path, default=None,
                   help="YAML hyperparameter file covering both stages; flags "
                        "given alongside it take precedence. Write a commented "
                        "reference with `python -m mtvae.config --example <e> "
                        "--out run.yaml`.")
    g.add_argument("--example", default=None, choices=sorted(EXAMPLES),
                   help="the recipe and data reader to use (mtvae.pipeline."
                        "EXAMPLES); may come from --config instead")
    g.add_argument("--arm", required=True, help="output name for this fit")
    g.add_argument("--dataset", default=None,
                   help="HC/HO/DR/MN for --example fourds; defaults to the "
                        "example's own label otherwise")
    g.add_argument("--seed", type=int, default=None,
                   help="one seed for both stages (Stage B re-seeds from it); "
                        "defaults to the --config file's seed")
    g.add_argument("--data-root", type=Path,
                   default=os.environ.get("MTVAE_DATA"),
                   help="figshare data root (default $MTVAE_DATA); the example "
                        "is read from <data-root>/data/...")
    g.add_argument("--out-root", type=Path, required=True,
                   help="the fit is written to <out-root>/<arm>/")
    g.add_argument("--device", default=None)
    g.add_argument("--cudnn-deterministic", action=argparse.BooleanOptionalAction,
                   default=True)

    g = p.add_argument_group("Stage A")
    g.add_argument("--save-stage-a", action=argparse.BooleanOptionalAction,
                   default=True, help="write stage_a.pt and stage_a_latents.npz, the "
                                      "Stage-A snapshot taken before Stage B")
    for flag, field, typ in _VAE_FLAGS:
        g.add_argument(flag, type=typ, default=None, dest=f"vae_{field}",
                       help=f"MergedVAEConfig.{field}; unset inherits the preset")

    g = p.add_argument_group("the handoff")
    g.add_argument("--rng-protocol", choices=RNG_PROTOCOLS, default=None,
                   help="default: the --config file's value, else "
                        "staged_equivalent")
    g.add_argument("--verify-n", type=int, default=4096)
    g.add_argument("--verify-atol", type=float, default=0.0,
                   help="tolerance of the check that the Stage-B encoder "
                        "reproduces the Stage-A latents before training; 0 = "
                        "exact")

    g = p.add_argument_group("Stage B")
    for flag, field, typ in _SDE_FLAGS:
        g.add_argument(flag, type=typ, default=None, dest=f"sde_{field}",
                       help=f"train_sde_coadapt.{field}; unset inherits the "
                            f"example recipe")
    g.add_argument("--scaled-z-sinkhorn", action=argparse.BooleanOptionalAction,
                   default=None)
    g.add_argument("--shipped-divergence", action=argparse.BooleanOptionalAction,
                   default=None)
    g.add_argument("--drift-hidden", default=None)
    g.add_argument("--diffusion-init", type=float, default=None)
    g.add_argument("--sde-dt", type=float, default=None)
    return p


def config_from_args(args: argparse.Namespace, device: str) -> PipelineConfig:
    """Resolve both stages' hyperparameters before any data is loaded.

    The ``--config`` file (if any) is the baseline and flags actually given
    override it; everything else comes from the example's recipe. Nothing
    inherits an argparse default.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.
    device : str
        Resolved compute device.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    SystemExit
        If neither the flags nor the config file name an example or a seed.
    """
    # A --config file is the baseline; explicit flags override it, so one file
    # can drive several fits that differ by a flag.
    base = config_io.load(args.config) if args.config else None
    example = args.example or (base.example if base else None)
    if example is None:
        raise SystemExit("need --example or a --config naming one")
    # Every field of the config file is honoured, including its seed.
    seed = args.seed if args.seed is not None else (base.seed if base else None)
    if seed is None:
        raise SystemExit("need --seed, or a --config that names one")
    proto = (args.rng_protocol
             or (base.rng_protocol if base else None) or "staged_equivalent")

    vae = {f: getattr(args, f"vae_{f}") for _, f, _ in _VAE_FLAGS
           if getattr(args, f"vae_{f}") is not None}
    dyn = {f: getattr(args, f"sde_{f}") for _, f, _ in _SDE_FLAGS
           if getattr(args, f"sde_{f}") is not None}
    for name in ("scaled_z_sinkhorn", "shipped_divergence"):
        if getattr(args, name) is not None:
            dyn[name] = getattr(args, name)
    geom = {}
    if args.drift_hidden:
        geom["drift_hidden"] = tuple(int(v) for v in args.drift_hidden.split(","))
    if args.diffusion_init is not None:
        geom["diffusion_init"] = args.diffusion_init
    if args.sde_dt is not None:
        geom["dt_step"] = args.sde_dt
    if base is not None:
        # Start from the file, then layer the flags on top. The vae half goes
        # through for_run, which still rejects unknown fields.
        vae = {**{k: v for k, v in vars(base.vae).items()
                  if k not in ("seed", "device")}, **vae}
        dyn = {**base.dynamics, **dyn}
        geom = {**base.sde_geometry, **geom}
    return PipelineConfig.for_example(
        example, seed=seed, device=device, vae=vae, dynamics=dyn,
        sde_geometry=geom, arch=dict(base.arch) if base else None,
        data=dict(base.data) if base else None, rng_protocol=proto)


def main() -> int:
    """Run one fit end to end and write its outputs; return the exit code."""
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    args = build_parser().parse_args()
    t0 = time.time()

    _pre = config_io.load(args.config) if args.config else None
    device = args.device or (_pre.device if _pre else None) or config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; falling back to cpu")
        device = "cpu"
    if args.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    out_dir = Path(args.out_root) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # Both stages are resolved before any data is read, so the record
    # describes the run as configured.
    cfg = config_from_args(args, device)
    #: The resolved seed, which may have come from the config file.
    seed = cfg.seed
    log.info("%s | arm=%s seed=%d | A: beta_sw=%s topo_atac=%s | "
             "B: lambda_dist=%s enc_lr=%s dec_lr=%s path-KL beta=%s ramp %s "
             "| rng=%s",
             cfg.example, args.arm, seed, cfg.vae.beta_sw,
             cfg.vae.lambda_topo_atac, cfg.dynamics["lambda_dist"],
             cfg.dynamics["encoder_lr"], cfg.dynamics["decoder_lr"],
             cfg.dynamics["beta_kl"], cfg.dynamics["kl_warmup"],
             cfg.rng_protocol)

    if args.data_root is None:
        raise SystemExit("need --data-root, or MTVAE_DATA set to the figshare data root")
    data = build_dataset(cfg.example, args.data_root, dataset=args.dataset,
                         data=cfg.data, arch=cfg.arch)
    log.info("data %s: %s cells | clip %s | train mask %s | n_rows %s", data.name,
             {m.name: m.n_cells for m in data.modalities}, cfg.data["clip"],
             cfg.data["train_mask"], cfg.data["n_rows"])
    feats = {m.name: m.x for m in data.modalities}
    slot = dict(zip(data.names, ("rna", "atac")))
    #: What rebuilds each trunk: written into stage_a.pt and model.pt.
    trunks = {slot[m.name]: trunk_meta(m.trunk) for m in data.modalities}

    # One run: Stage A, the handoff and Stage B (pipeline.train).
    stage_a_rec: dict = {"mode": "train"}
    fit = suite_model.train(data, cfg, verify_n=args.verify_n,
                            verify_atol=args.verify_atol)
    enc, decoders = fit.stage_a.encoder, fit.stage_a.decoders
    z_a = {m: np.ascontiguousarray(fit.latents.mu[m], dtype=np.float32)
           for m in data.names}
    hist, sde = fit.dynamics.history, fit.dynamics.sde
    stage_a_rec.update(seconds=fit.seconds["stage_a"],
                       history=fit.vae.history,
                       handoff=dict(fit.stage_a.provenance))
    # Already gated inside pipeline.train, between the handoff and Stage B:
    # the only point at which the encoder still reproduces the Stage-A
    # latent, since Stage B then co-adapts it.
    gate = fit.gate
    stage_a_cfg = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(cfg.vae).items()}
    for m, g in gate.items():
        log.info("Stage-A gate [%s]: %s max|d|=%.3e", m,
                 "OK" if g["ok"] else "MISMATCH", g.get("max_abs", float("nan")))

    # ---- export ------------------------------------------------------------
    lat_dir = out_dir / "latents"
    lat_dir.mkdir(exist_ok=True)
    eff_before, eff_after, drift = {}, {}, {}
    for m in data.names:
        eff_before[m] = participation_ratio(torch.from_numpy(z_a[m]))
        z_new = enc.encode_all(feats[m], slot[m], device=device)
        np.save(lat_dir / f"z_coadapt_{m}.npy", z_new)
        eff_after[m] = participation_ratio(torch.from_numpy(z_new))
        drift[m] = float(np.linalg.norm(z_new - z_a[m], axis=1).mean())
        log.info("%s: eff_rank %.3f -> %.3f | mean per-cell drift %.4f",
                 m, eff_before[m], eff_after[m], drift[m])

    verdicts = {m: collapse_verdict(eff_before[m], eff_after[m])
                for m in data.names}
    for m, (v, detail) in verdicts.items():
        (log.warning if v == "COLLAPSED" else log.info)(
            "collapse gate [%s]: %s - %s", m, v, detail)

    if args.save_stage_a:
        np.savez_compressed(out_dir / "stage_a_latents.npz",
                            **{slot[m]: z_a[m] for m in data.names})
        # The snapshot taken before Stage B, not fit.vae.model.state_dict():
        # the model has since co-adapted and would not reproduce the latents
        # saved beside it.
        torch.save({"model_state_dict": fit.stage_a_state,
                    "config": stage_a_cfg, "trunks": trunks,
                    "dataset": data.name, "arm": args.arm, "seed": seed,
                    "stages": list(data.stages)},
                   out_dir / "stage_a.pt")

    geom = cfg.sde_geometry
    torch.save({"sde_state_dict": sde.state_dict(),
                **{f"decoder_{slot[m]}_state_dict": decoders[m].state_dict()
                   for m in data.names},
                "encoder_state_dict": enc.state_dict(),
                "seed": seed, "arm": args.arm, "dataset": data.name,
                "config": {**geom, "sde_dt": geom["dt_step"],
                           "latent_dim": z_a[data.names[0]].shape[1]},
                "stages": list(data.stages),
                "stage_norm": np.asarray(data.stage_times, dtype=np.float32),
                # Architecture, so model.pt rebuilds without stage_a.pt.
                "stage_a_config": stage_a_cfg, "trunks": trunks},
               out_dir / "model.pt")

    record = {
        "pipeline": "merged", "example": cfg.example,
        "arm": args.arm, "dataset": data.name, "seed": seed,
        "elapsed_s": round(time.time() - t0, 1),
        # Both stages' hyperparameters as one flat mapping; round-trips
        # through PipelineConfig.from_blob.
        "hyperparameters": cfg.to_blob(),
        "sde_geometry": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in geom.items()},
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "modalities": list(data.names), "stages": list(data.stages),
        "stage_times": [float(x) for x in data.stage_times],
        "losses": list(data.losses),
        "trunks": {m.name: m.trunk.kind for m in data.modalities},
        "clips": {m.name: m.clip for m in data.modalities},
        "resolved": {
            "device": device,
            "cudnn_deterministic": bool(args.cudnn_deterministic),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "stage_a": stage_a_rec,
        "stage_b": {
            "stage_a_gate": gate,
            "eff_rank_stage_a": eff_before, "eff_rank_coadapt": eff_after,
            "mean_cell_drift": drift,
            "collapse_verdict": {m: v for m, (v, _) in verdicts.items()},
            "collapse_detail": {m: d for m, (_, d) in verdicts.items()},
            "history": hist,
        },
    }
    (out_dir / "run_record.json").write_text(
        json.dumps(record, indent=2, default=str))
    log.info("wrote %s (%.1f s total)", out_dir, time.time() - t0)
    if any(v == "COLLAPSED" for v, _ in verdicts.values()):
        log.warning("the latent collapsed during Stage B (collapse gate); "
                    "treat this fit as failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
