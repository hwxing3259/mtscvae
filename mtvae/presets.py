"""Complete Stage-A configurations for the two examples.
Every field is set explicitly; :func:`for_run` sets the seed and device."""
from __future__ import annotations

import dataclasses
from typing import Any

from mtvae.stage_a.trainer import MergedVAEConfig

#: The 35 fields shared by both examples (not the dataclass defaults).
_COMMON: dict[str, Any] = dict(
    mode="joint",
    phase1_dir=None,
    p1_rna_iters=2000, p1_atac_iters=4000,
    p1_rna_lr=1e-3, p1_atac_lr=5e-4, p1_batch=512,
    lr=5e-4, encoder_lr=1e-4, joint_lr=5e-4,
    joint_warmup_iters=0, grad_clip=5.0,
    sw_batch_per_stage=64, sw_n_projections=64,
    lambda_topo_rna=1e-2,
    topo_batch=512, topo_perplexity=30.0, topo_affinity="full",
    topo_identity_dims=0,
    kl_scale="mind", kl_free_bits=0.0, kl_dims=0,
    sigma_mode="learned", sigma_init=0.1, deterministic=False,
    tilt_on="sample", sw_on="sample",
    fusion_arch="shared", fusion_hidden_dim=1024, fusion_depth=1,
    fusion_residual_scale=0.5, fusion_layer_norm=False,
    latent_dim=50, log_every=100, device="cuda",
)

#: Four-dataset benchmark (HC / HO / DR / MN).
FOURDATA_VAE = MergedVAEConfig(
    **_COMMON,
    n_iters=6000, batch_size_per_modality=256,
    beta_kl=5e-3, beta_sw=2.0, lambda_topo_atac=1e-2,
    seed=0,
)

#: Herring et al. (2022) PFC, Stage A.
#:
#: ``beta_sw`` and ``lambda_topo_atac`` interact: at ``lambda_topo_atac = 0.1``
#: the astrocyte cross-modal alignment holds only at ``beta_sw = 0.5``. Tune
#: them together.
#:
#: The ATAC trunk, ATAC clipping, training mask and cuDNN determinism are not
#: ``MergedVAEConfig`` fields; they are data and runtime settings, set through
#: :func:`mtvae.pipeline.build_dataset` and the :mod:`mtvae.fit` options.
PFC_VAE = MergedVAEConfig(
    **_COMMON,
    n_iters=3000, batch_size_per_modality=512,
    beta_kl=3e-2, beta_sw=0.5, lambda_topo_atac=1e-1,
    seed=22,
)

#: :data:`PFC_VAE` with the four-dataset batch size and iteration count
#: (256 x 6000), i.e. the same number of samples per modality as
#: 512 x 3000.
PFC_VAE_N8_TWIN = dataclasses.replace(PFC_VAE, n_iters=6000,
                                      batch_size_per_modality=256)

VAE_PRESETS: dict[str, MergedVAEConfig] = {
    "fourdata": FOURDATA_VAE,
    "pfc": PFC_VAE,
    "pfc_n8_twin": PFC_VAE_N8_TWIN,
}


def for_run(preset: MergedVAEConfig, **overrides: Any) -> MergedVAEConfig:
    """Specialise a preset for one run.

    Everything not named in ``overrides`` comes from the preset, so a run
    stays tied to its recipe.

    Parameters
    ----------
    preset : MergedVAEConfig
        One of :data:`VAE_PRESETS`.
    **overrides
        Fields to change — normally just ``seed`` and ``device``.

    Returns
    -------
    MergedVAEConfig

    Raises
    ------
    TypeError
        If an override names something that is not a ``MergedVAEConfig``
        field, so a typo is an error rather than silently ignored.
    """
    known = {f.name for f in dataclasses.fields(MergedVAEConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(
            f"not MergedVAEConfig fields: {sorted(unknown)}. "
            f"Adapter/runtime settings (atac_encoder, atac_clip, train_mask, "
            f"cudnn_deterministic) are not part of the VAE recipe.")
    return dataclasses.replace(preset, **overrides)
