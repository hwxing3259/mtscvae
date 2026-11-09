"""CPU smoke test: a tiny two-stage fit on synthetic data.

Run from the repository root::

    pytest tests/
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict

import numpy as np
import torch

import mtvae
from mtvae.nn.trunks import TrunkSpec


def _dataset(spatial: bool = False) -> mtvae.Dataset:
    """Two modalities, 4 stages, 160 cells, with teachers."""
    g = np.random.default_rng(0)
    n, n_st = 160, 4
    st = np.sort(np.concatenate([np.arange(n_st), g.integers(0, n_st, n - n_st)]))
    x = lambda d: (g.normal(size=(n, d)) + 0.3 * st[:, None]).astype(np.float32)
    atac = (TrunkSpec("spatial", chr_ids=np.repeat(np.arange(4), 8), kernel_sizes=3)
            if spatial else TrunkSpec("mlp"))
    mods = (mtvae.Modality("rna", x(40), st, teacher=x(10), trunk=TrunkSpec("mlp")),
            mtvae.Modality("atac", x(32), st, teacher=x(10), trunk=atac))
    times = mtvae.stage_times_from([st, st], [st.astype(float)] * 2, n_st)
    return mtvae.Dataset(name="synthetic", stages=tuple(f"s{i}" for i in range(n_st)),
                         modalities=mods, stage_times=times)


def _config(example: str = "fourds") -> mtvae.PipelineConfig:
    """The paper's recipe, shrunk to a few iterations."""
    cfg = mtvae.PipelineConfig.for_example(
        example, seed=1, device="cpu",
        dynamics=dict(train_iters=12, warmup_iters=2, kl_warmup=(2, 6), batch_align=32,
                      batch_rec=32, batch_dist=32, batch_anchor=32, log_every=1000))
    return dataclasses.replace(cfg, vae=dataclasses.replace(
        cfg.vae, n_iters=4, batch_size_per_modality=32, topo_batch=32, log_every=1000))


def _state(fit) -> dict:
    return {**{f"sde.{k}": v for k, v in fit.dynamics.sde.state_dict().items()},
            **{f"enc.{k}": v for k, v in fit.stage_a.encoder.state_dict().items()}}


def test_fit_encode_rollout() -> None:
    """Both stages train; latents and decoded rollouts have the right shapes."""
    data = _dataset()
    fit = mtvae.train(data, _config())
    z = fit.encode()
    assert z["rna"].shape == (160, 50) and z["atac"].shape == (160, 50)
    assert all(np.isfinite(v).all() for v in z.values())
    from mtvae.predict import rollouts
    res = rollouts(fit.dynamics.sde, OrderedDict(a=z["rna"][:10]), data.stage_times,
                   n_rollouts=2, decode_with=fit.stage_a.decoders["rna"],
                   reduce=lambda v: v.mean(0))
    assert np.stack(res["a"].per_stage).shape == (4, 40)


def test_same_seed_same_fit() -> None:
    """Two fits from one seed are identical, tensor for tensor."""
    a = _state(mtvae.train(_dataset(), _config()))
    b = _state(mtvae.train(_dataset(), _config()))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_spatial_trunk() -> None:
    """The chromosome-aware ATAC trunk trains through the same path."""
    fit = mtvae.train(_dataset(spatial=True), _config("pfc"))
    assert fit.encode()["atac"].shape == (160, 50)


def test_config_roundtrip(tmp_path) -> None:
    """A YAML config file round-trips, and the release recipe carries the path-KL."""
    from mtvae import config
    cfg = mtvae.PipelineConfig.for_example("pfc", seed=44, device="cpu")
    assert cfg.dynamics["beta_kl"] == 1e-4
    p = tmp_path / "run.yaml"
    config.dump(cfg, p)
    assert config.load(p).to_blob() == cfg.to_blob()
