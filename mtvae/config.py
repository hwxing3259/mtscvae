"""Read and write the hyperparameters of both stages as YAML.
Write a commented reference: ``python -m mtvae.config --example fourds --out my_run.yaml``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


from mtvae.pipeline import (  # noqa: E402
    EXAMPLES, PipelineConfig,
)

#: Section -> {knob: one-line doc}. The docs are prose; the values always come
#: from ``PipelineConfig.for_example``. A key absent from this table is
#: rejected by :func:`from_dict`.
KNOB_DOCS: Dict[str, Dict[str, str]] = {
    "run": {
        "example": "which recipe to start from: "
                   + " | ".join(sorted(EXAMPLES))
                   + ". For new data, start from the closer one and override.",
        "seed": "one seed. Stage A seeds from it; Stage B re-seeds from it.",
        "device": '"cuda" | "cpu".',
        "rng_protocol": "staged_equivalent: the random-number sequence of the "
                        "released fits; it replays three module constructions "
                        "before the SDE is built. reseed: skips them; every "
                        "Stage-B number then differs.",
    },
    "vae": {
        "mode": '"joint" | "staged". The paper uses joint; staged pretrains '
                "each autoencoder first and uses the p1_* knobs.",
        "phase1_dir": "reuse cached pretrained autoencoders (staged mode only).",
        "p1_rna_iters": "RNA pretraining iterations (staged mode only).",
        "p1_atac_iters": "ATAC pretraining iterations (staged mode only).",
        "p1_rna_lr": "RNA pretraining learning rate (staged mode only).",
        "p1_atac_lr": "ATAC pretraining learning rate (staged mode only).",
        "p1_batch": "pretraining batch size (staged mode only).",
        "n_iters": "Stage-A iterations: 6000 four-dataset, 3000 PFC.",
        "batch_size_per_modality": "cells per modality per step.",
        "lr": "fusion-head learning rate (not sde.lr).",
        "encoder_lr": "trunk and decoder learning rate in staged mode; unused "
                      "in joint mode, where joint_lr covers every parameter. "
                      "Not sde.encoder_lr.",
        "joint_lr": "learning rate of every parameter, joint mode.",
        "joint_warmup_iters": "reconstruction-only iterations before the "
                              "other terms switch on.",
        "grad_clip": "global gradient-norm clip.",
        "beta_sw": "weight of the per-stage sliced-Wasserstein cross-modal "
                   "matching. Interacts with lambda_topo_atac: tune the two "
                   "together.",
        "sw_batch_per_stage": "cells per stage in the sliced-Wasserstein term.",
        "sw_n_projections": "random projections in the sliced-Wasserstein term.",
        "lambda_topo_rna": "weight of the RNA neighbourhood tilt toward its "
                           "teacher. Larger values sharpen RNA clusters at the "
                           "cost of cross-modal alignment.",
        "lambda_topo_atac": "weight of the ATAC neighbourhood tilt toward its "
                            "teacher; see beta_sw.",
        "topo_batch": "cells per tilt evaluation.",
        "topo_perplexity": "openTSNE perplexity of the teacher affinities.",
        "topo_affinity": '"full" | "batch". "full" computes the teacher '
                         "affinities once over all cells; openTSNE draws from "
                         "numpy's global random state there, so pin openTSNE "
                         "(1.0.4) for exact reproduction.",
        "topo_identity_dims": "latent dimensions excluded from the tilt.",
        "beta_kl": "weight of the Stage-A prior KL: 5e-3 four-dataset, 3e-2 "
                   "PFC. Not sde.beta_kl.",
        "kl_scale": '"mind" (divided by n_cells * latent_dim) | "elbo" '
                    "(per-sample sum).",
        "kl_free_bits": "per-dimension KL floor in nats; guards against "
                        "posterior collapse.",
        "kl_dims": "0 = all dimensions stochastic; k = only the first k.",
        "sigma_mode": '"learned" | "fixed" posterior sigma.',
        "sigma_init": "initial posterior sigma. Latents are exported as "
                      "posterior means.",
        "deterministic": "z = mu, i.e. no stochastic bottleneck (an "
                         "autoencoder rather than a VAE).",
        "tilt_on": '"sample" | "mu": which latent the tilt sees.',
        "sw_on": '"sample" | "mu": which latent the sliced-Wasserstein term sees.',
        "fusion_arch": "architecture of the shared fusion layer.",
        "fusion_hidden_dim": "fusion width (1024 in both examples).",
        "fusion_depth": "fusion depth.",
        "fusion_residual_scale": "residual scale of the fusion. A plain "
                                 "attribute, not saved in the state dict: a "
                                 "checkpoint must be rebuilt with the value it "
                                 "was trained with.",
        "fusion_layer_norm": "layer norm inside the fusion.",
        "latent_dim": "latent dimension (50 in both examples).",
        "seed": "mirrored from run.seed; do not set separately.",
        "device": "mirrored from run.device; do not set separately.",
        "log_every": "Stage-A logging interval.",
    },
    "sde": {
        "encoder_lr": "learning rate of the encoder while it co-adapts with the "
                      "SDE (1e-5). 0 freezes the encoder.",
        "decoder_lr": "learning rate of the decoders while they co-adapt (1e-5). "
                      "0 freezes them.",
        "lr": "learning rate of the SDE (not vae.lr).",
        "lambda_dist": "weight of the feature-level Sinkhorn match of decoded "
                       "and observed cohorts: 0.01 four-dataset, 0.1 PFC.",
        "lambda_rec": "weight of the reconstruction term.",
        "lambda_prox": "weight of the penalty holding the decoders near their "
                       "Stage-A weights.",
        "lambda_latent_anchor": "weight of the penalty holding the encoder's "
                                "latents near the Stage-A latents. 0 leaves the "
                                "latent space unanchored.",
        "beta_kl": "weight of the SDE path KL, 0.5 * int |f_post / g|^2 dt "
                   "against a zero prior drift, over chained and pairwise "
                   "rollouts (1e-4 in the paper; 0 switches it off). Not "
                   "vae.beta_kl. In YAML write 1.0e-4: PyYAML reads 1e-4 as a "
                   "string (it is converted, but be explicit).",
        "kl_warmup": "[start, end] iterations of the linear path-KL ramp: 0 "
                     "before start, beta_kl from end.",
        "alpha_align": "weight of the latent dynamics alignment.",
        "lambda_chained": "weight of the chained (one rollout through all "
                          "stages) alignment.",
        "lambda_per_pair": "weight of the pairwise (one stage to the next) "
                           "alignment.",
        "batch_align": "cells per alignment step.",
        "batch_rec": "cells per reconstruction step.",
        "batch_dist": "cells per feature-level matching step.",
        "batch_anchor": "cells per anchor evaluation.",
        "train_iters": "Stage-B iterations.",
        "warmup_iters": "Stage-B warmup iterations.",
        "scaled_z_sinkhorn": "rescale latents before the latent Sinkhorn "
                             "divergence.",
        "sinkhorn_eps_x": "entropic regularisation of the feature-level Sinkhorn.",
        "sinkhorn_iters_x": "iterations of the feature-level Sinkhorn.",
        "shipped_divergence": "use the original rescaled Sinkhorn divergence, "
                              "which can reward a collapsed cohort. False (the "
                              "default) uses the corrected divergence.",
        "log_every": "Stage-B logging interval (not vae.log_every).",
        "target_clip": "per-modality clip of the reconstruction targets, as "
                       "{modality: value}; null = none. The encoder sees the "
                       "unclipped input, so a target clip tighter than the "
                       "input clip changes what is learned.",
        "seed": "mirrored from run.seed; do not set separately.",
        "device": "mirrored from run.device; do not set separately.",
    },
    "arch": {
        "rna_encoder": '"mlp" | "spatial" | "conv": the encoder trunk of slot 0. '
                       "Slot names are the model's, not the dataset's.",
        "atac_encoder": '"mlp" | "spatial" | "conv": the encoder trunk of slot 1. '
                        '"spatial" is the chromosome-aware Conv1d (PFC); it '
                        "needs a peak list giving each feature's chromosome. "
                        '"conv" is the same trunk without the chromosome '
                        "boundaries and needs no peak list. The trunk also "
                        "decides where the Stage-B SDE initialises.",
        "spatial_kernels": "kernel width of a spatial or conv trunk; must match "
                           "Stage A.",
    },
    "data": {
        "clip": "symmetric input bound per modality, {rna: ..., atac: ...}; null "
                "= unclipped. Applied when the data are read.",
        "train_mask": "fit on the example's training split (PFC: the "
                      "whole-donor split in wholedonor_train_mask.npz) and "
                      "encode every cell.",
        "n_rows": "fit on the first n_rows rows of each modality's order in "
                  "bench_tiers.npz (the PFC benchmark tiers); null = all rows.",
    },
    "geom": {
        "prior_mode": "SDE prior.",
        "diffusion_mode": '"scalar" | other diffusion parameterisations.',
        "drift_hidden": "drift network hidden sizes, e.g. [128, 128].",
        "diffusion_hidden": "diffusion network hidden sizes.",
        "time_embed_dim": "Fourier time-embedding output width.",
        "time_hidden": "Fourier time-embedding hidden width.",
        "diffusion_init": "initial diffusion scale.",
        "dt_step": "SDE integration step.",
    },
}

#: Blob prefix -> YAML section.
_SECTIONS = (("vae.", "vae"), ("sde.", "sde"), ("geom.", "geom"),
             ("arch.", "arch"), ("data.", "data"))


def to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    """Convert a config to a nested, YAML-friendly mapping.

    Parameters
    ----------
    cfg : PipelineConfig
        The configuration.

    Returns
    -------
    dict
        Top-level ``example``, ``seed``, ``device`` and ``rng_protocol``, and
        the sections ``vae``, ``sde``, ``geom`` and ``arch``. Round-trips
        through :func:`from_dict`.
    """
    blob = cfg.to_blob()
    out: Dict[str, Any] = {
        "example": blob["example"], "seed": int(cfg.vae.seed),
        "device": blob["device"], "rng_protocol": blob["rng_protocol"],
    }
    for prefix, sec in _SECTIONS:
        out[sec] = {k[len(prefix):]: v for k, v in blob.items()
                    if k.startswith(prefix)}
    # Seed and device are run-level; carrying them per stage too would allow
    # the copies to disagree.
    for sec in ("vae", "sde"):
        out[sec].pop("seed", None)
        out[sec].pop("device", None)
    return out


def from_dict(d: Dict[str, Any]) -> PipelineConfig:
    """Build a config from a (possibly partial) mapping.

    Anything absent comes from ``example``'s recipe, so a three-line file is a
    valid config. ``seed`` defaults to 0, ``device`` to ``"cuda"`` and
    ``rng_protocol`` to ``"staged_equivalent"``.

    Parameters
    ----------
    d : dict
        Must name ``example``; everything else is optional.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    KeyError
        If ``example`` is missing or unknown.
    ValueError
        If any key is not a known setting, so that a typo is an error rather
        than silently ignored.
    """
    if "example" not in d:
        raise KeyError(
            f"config must name 'example' (one of {sorted(EXAMPLES)}): it "
            f"supplies every value the file does not override.")
    unknown_top = set(d) - {"example", "seed", "device", "rng_protocol",
                            "vae", "sde", "geom", "arch", "data"}
    if unknown_top:
        raise ValueError(f"unknown top-level keys: {sorted(unknown_top)}")
    for sec in ("vae", "sde", "geom", "arch", "data"):
        bad = set(d.get(sec) or {}) - set(KNOB_DOCS[sec])
        if bad:
            raise ValueError(
                f"unknown {sec} knobs: {sorted(bad)}. Known: "
                f"{sorted(KNOB_DOCS[sec])}")

    geom = dict(d.get("geom") or {})
    for k in ("drift_hidden", "diffusion_hidden"):
        if k in geom:
            geom[k] = tuple(geom[k])
    return PipelineConfig.for_example(
        d["example"], seed=int(d.get("seed", 0)),
        device=d.get("device", "cuda"),
        vae=dict(d.get("vae") or {}), dynamics=dict(d.get("sde") or {}),
        sde_geometry=geom, arch=dict(d.get("arch") or {}),
        data=dict(d.get("data") or {}),
        rng_protocol=d.get("rng_protocol", "staged_equivalent"))


def load(path: Path | str) -> PipelineConfig:
    """Read a YAML config file.

    Parameters
    ----------
    path : pathlib.Path or str
        YAML file; see :func:`from_dict` for its contents.

    Returns
    -------
    PipelineConfig
    """
    return from_dict(yaml.safe_load(Path(path).read_text()) or {})


def dump(cfg: PipelineConfig, path: Path | str) -> None:
    """Write the resolved config as YAML, without comments.

    Parameters
    ----------
    cfg : PipelineConfig
        The configuration.
    path : pathlib.Path or str
        Output file.
    """
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False,
                                         default_flow_style=False))


def reference_yaml(example: str, seed: int = 0, device: str = "cuda",
                   cfg: PipelineConfig | None = None, title: str | None = None) -> str:
    """Every setting at the example's value, with a one-line comment above each.

    Values come from ``PipelineConfig.for_example``, never from a literal here.

    Parameters
    ----------
    example : str
        Key into :data:`mtvae.pipeline.EXAMPLES`.
    seed : int, default 0
        Seed written to the file.
    device : str, default "cuda"
        Device written to the file.
    cfg : PipelineConfig, optional
        Write this configuration instead of the example's recipe.
    title : str, optional
        First-line title of the file.

    Returns
    -------
    str
        YAML text.
    """
    cfg = cfg or PipelineConfig.for_example(example, seed=seed, device=device)
    d = to_dict(cfg)
    L = [
        f"# {title or 'mtvae hyperparameters'} -- every knob, example={example!r}.",
        "#",
        "# Generated from PipelineConfig, so the values are what would run. The",
        "# example's recipe alone is written by:",
        f"#   python -m mtvae.config "
        f"--example {example}",
        "#",
        "# Partial files are fine: delete everything you are not changing, and",
        "# the rest comes from the example's recipe. Unknown keys raise an error.",
        "#",
        "# `lr`, `encoder_lr` and `log_every` exist under both `vae:` (Stage A)",
        "# and `sde:` (Stage B) and are different settings.",
        "",
    ]
    for k in ("example", "seed", "device", "rng_protocol"):
        for line in _wrap(KNOB_DOCS["run"][k]):
            L.append(f"# {line}")
        L.append(_line(k, d[k]))
        L.append("")

    titles = {"data": "DATA -- input clipping, training split, row subset",
              "arch": "ARCHITECTURE -- which trunk each slot gets",
              "vae": "STAGE A -- the multimodal VAE",
              "sde": "STAGE B -- the latent SDE objective",
              "geom": "STAGE B -- SDE network geometry"}
    for sec in ("data", "arch", "vae", "sde", "geom"):
        L += ["", "# " + "=" * 72, f"# {titles[sec]}", "# " + "=" * 72, "",
              f"{sec}:"]
        for k, v in d[sec].items():
            doc = KNOB_DOCS[sec].get(k, "")
            for line in _wrap(doc):
                L.append(f"  # {line}")
            L.append("  " + _line(k, v))
            L.append("")
    return "\n".join(L) + "\n"


def _line(key: str, value: Any) -> str:
    """Format one ``key: value`` line.

    Lists and dicts are written inline by hand (``drift_hidden: [128, 128]``);
    scalars go through ``safe_dump``. Dumping ``{key: value}`` with
    ``default_flow_style=None`` would flow-style the single-key wrapper too
    (``{phase1_dir: null}``), which the reference could not re-read.
    """
    if isinstance(value, (list, tuple)):
        return f"{key}: [{', '.join(str(x) for x in value)}]"
    if isinstance(value, dict):
        # Inline, like lists. Block style would put the children at the same
        # indent as the key inside the section body and silently make them
        # siblings: valid YAML with the wrong structure.
        return f"{key}: {json.dumps(value)}"
    return yaml.safe_dump({key: value}, sort_keys=False,
                          default_flow_style=False).rstrip()


def _wrap(text: str, width: int = 74) -> list:
    """Wrap a doc line for a YAML comment."""
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out or [""]


def main() -> int:
    """Write a reference config to ``--out`` or stdout; return the exit code."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example", default="fourds", choices=sorted(EXAMPLES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None,
                   help="write here; default is stdout")
    a = p.parse_args()
    text = reference_yaml(a.example, seed=a.seed, device=a.device)
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out} ({len(text.splitlines())} lines)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
