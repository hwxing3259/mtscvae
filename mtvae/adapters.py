"""Readers that turn each example's files into a :class:`mtvae.data.Dataset`.

:func:`fourds` reads one of the four scMultiNODE benchmark datasets (HC, HO,
DR, MN); :func:`pfc` reads the Herring et al. (2022) human prefrontal cortex
atlas. The adapter is the only layer that touches the filesystem; any function
returning a valid ``Dataset`` works equally well with
:func:`mtvae.pipeline.train`.

Clipping
--------
``clip`` maps each modality to a symmetric input bound, or ``None`` for no
clipping; it is the ``data.clip`` setting of a config. Clipping is idempotent,
so data that is already bounded is unaffected. Each modality records the bound
it got in ``Modality.clip``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


from mtvae.data import (  # noqa: E402
    UNIFORM_CLIP, Dataset, Modality, stage_times_from,
)
from mtvae.nn.trunks import TrunkSpec  # noqa: E402

log = logging.getLogger("mtvae.adapters")

#: Default ``clip`` of both readers: every modality at
#: :data:`~mtvae.data.UNIFORM_CLIP`.
DEFAULT_CLIP = {"rna": UNIFORM_CLIP, "atac": UNIFORM_CLIP}


def _apply(x, clip):
    """Clip ``x`` to ``[-clip, clip]``; return it unchanged if ``clip`` is None."""
    return x if clip is None else np.clip(x, -clip, clip)


def _stages_and_index(metas: list[pd.DataFrame],
                      order: Optional[tuple] = None) -> tuple[tuple, list]:
    """Stage vocabulary and a per-modality integer index into it.

    Parameters
    ----------
    metas : list of pandas.DataFrame
        Per-modality metadata carrying ``stage_id``.
    order : tuple of str, optional
        Explicit stage order. ``None`` sorts the observed values -- fine when
        stage ids sort correctly, which is why PFC passes its order explicitly.

    Returns
    -------
    stages : tuple of str
        The stage vocabulary.
    idx : list of numpy.ndarray
        Per-modality integer stage index, int64.

    Raises
    ------
    ValueError
        If a cell's ``stage_id`` is outside the vocabulary.
    """
    stages = tuple(order) if order is not None else tuple(
        sorted(set().union(*[set(m["stage_id"]) for m in metas])))
    s2i = {s: i for i, s in enumerate(stages)}
    idx = []
    for m in metas:
        unknown = set(m["stage_id"]) - set(s2i)
        if unknown:
            raise ValueError(
                f"stage_id values {sorted(unknown)} are not in the vocabulary "
                f"{stages}; they would collapse into stage 0")
        idx.append(m["stage_id"].map(s2i).values.astype(np.int64))
    return stages, idx


def fourds(staged: Path, name: str, *, teachers: bool = True,
           clip: Optional[dict] = None) -> Dataset:
    """One of the four benchmark datasets (HC / HO / DR / MN).

    Stages are the sorted ``stage_id`` values of the metadata.

    Parameters
    ----------
    staged : pathlib.Path
        Holds ``{rna,atac}_expr.npy`` and ``{rna,atac}_metadata.csv`` (with
        ``stage_id`` and ``log_age`` columns), optionally
        ``train_mask_all.npz``, and the two ``{rna,atac}_teacher_pca.npy``.
    name : str
        Dataset label for records and logging.
    teachers : bool, default True
        Load the teachers. Required whenever ``lambda_topo_* > 0``, as in the
        four-dataset preset.
    clip : dict, optional
        ``{"rna": bound, "atac": bound}``, ``None`` for an unclipped
        modality. Defaults to :data:`DEFAULT_CLIP`.

    Returns
    -------
    Dataset
        Two MLP-trunk modalities, ``stage_times`` derived from ``log_age``.
    """
    clip = {**DEFAULT_CLIP, **(clip or {})}
    rna_clip, atac_clip = clip["rna"], clip["atac"]
    rna = _apply(np.asarray(np.load(staged / "rna_expr.npy", mmap_mode="r"),
                            dtype=np.float32), rna_clip)
    atac = _apply(np.asarray(np.load(staged / "atac_expr.npy", mmap_mode="r"),
                             dtype=np.float32), atac_clip)

    metas = [pd.read_csv(staged / f"{m}_metadata.csv") for m in ("rna", "atac")]
    stages, (t_rna, t_atac) = _stages_and_index(metas)

    mask_p = staged / "train_mask_all.npz"
    if mask_p.exists():
        mk = np.load(mask_p)
        tr = {"rna": mk["train_mask_rna"], "atac": mk["train_mask_atac"]}
    else:
        tr = {"rna": None, "atac": None}

    tea = {"rna": None, "atac": None}
    if teachers:
        for m in ("rna", "atac"):
            tea[m] = np.load(staged / f"{m}_teacher_pca.npy")

    mods = (
        Modality("rna", rna, t_rna, teacher=tea["rna"], train=tr["rna"],
                 trunk=TrunkSpec("mlp"), clip=rna_clip),
        Modality("atac", atac, t_atac, teacher=tea["atac"], train=tr["atac"],
                 trunk=TrunkSpec("mlp"), clip=atac_clip),
    )
    st = stage_times_from([t_rna, t_atac],
                          [metas[0]["log_age"].values, metas[1]["log_age"].values],
                          len(stages))
    log.info("%s: rna %s atac %s | %d stages | stage_times %s",
             name, rna.shape, atac.shape, len(stages), np.round(st, 3).tolist())
    return Dataset(name=name, stages=stages, modalities=mods, stage_times=st)


def pfc(data_dir: Path, *, stages: tuple, atac_teacher: Optional[Path] = None,
        rna_teacher: Optional[Path] = None, train_mask: Optional[Path] = None,
        peak_list: Optional[Path] = None, spatial_kernels: int = 3,
        atac_trunk: str = "spatial",
        clip: Optional[dict] = None) -> Dataset:
    """The Herring et al. (2022) PFC atlas, by default with the
    chromosome-aware ATAC trunk.

    Parameters
    ----------
    data_dir : pathlib.Path
        Holds ``{rna,atac}_expr.npy``, ``{rna,atac}_metadata.csv`` (with
        ``stage_id`` and ``log_age`` columns), the default teachers and
        ``atac_peak_list.txt``.
    stages : tuple of str
        The stage order. Passed explicitly because lexical order of the stage
        names is not developmental order.
    atac_teacher : pathlib.Path, optional
        Defaults to ``atac_harmony_100d.npy`` (a Harmony-corrected embedding;
        a plain PCA teacher does not preserve the astrocyte result).
    rna_teacher : pathlib.Path, optional
        Defaults to ``rna_teacher_pca.npy``.
    train_mask : pathlib.Path, optional
        ``.npz`` with ``train_mask_rna`` and ``train_mask_atac`` (the
        whole-donor split). The model fits on the masked cells and encodes
        all of them (see :meth:`mtvae.data.Dataset.training_view`).
    peak_list : pathlib.Path, optional
        Peak list for the spatial trunk; defaults to
        ``data_dir / "atac_peak_list.txt"``.
    spatial_kernels : int, default 3
        Conv kernel width of the spatial or conv trunk.
    atac_trunk : {"spatial", "conv", "mlp"}, default "spatial"
        ``"spatial"`` is the chromosome-aware Conv1d trunk and reads the peak
        list. ``"conv"`` is the same architecture without the boundary mask
        and needs no peak list. ``"mlp"`` is the plain MLP trunk.
    clip : dict, optional
        ``{"rna": bound, "atac": bound}``, ``None`` for an unclipped
        modality. Defaults to :data:`DEFAULT_CLIP`.

    Returns
    -------
    Dataset

    Raises
    ------
    FileNotFoundError
        If a teacher file is missing.
    ValueError
        If a teacher's row count differs from its matrix.
    """
    from mtvae.nn.spatial_trunk import (
        load_chromosome_ids,
    )

    clip = {**DEFAULT_CLIP, **(clip or {})}
    rna_clip, atac_clip = clip["rna"], clip["atac"]
    rna = _apply(np.asarray(np.load(data_dir / "rna_expr.npy", mmap_mode="r"),
                            dtype=np.float32), rna_clip)
    atac = _apply(np.asarray(np.load(data_dir / "atac_expr.npy", mmap_mode="r"),
                             dtype=np.float32), atac_clip)

    metas = [pd.read_csv(data_dir / f"{m}_metadata.csv") for m in ("rna", "atac")]
    stg, (t_rna, t_atac) = _stages_and_index(metas, order=stages)

    tea = {}
    for m, given, default in (("rna", rna_teacher, "rna_teacher_pca.npy"),
                              ("atac", atac_teacher, "atac_harmony_100d.npy")):
        tp = Path(given) if given else (data_dir / default)
        if not tp.exists():
            raise FileNotFoundError(
                f"{tp} missing: the PFC example needs its teacher embeddings "
                f"(data/pfc/ in the figshare item).")
        T = np.load(tp)
        n = len(rna) if m == "rna" else len(atac)
        if len(T) != n:
            raise ValueError(f"{m}: teacher has {len(T)} rows, matrix has {n}")
        tea[m] = T

    tr = {"rna": None, "atac": None}
    if train_mask:
        mk = np.load(train_mask)
        tr = {"rna": mk["train_mask_rna"], "atac": mk["train_mask_atac"]}

    if atac_trunk == "spatial":
        pl = peak_list or (data_dir / "atac_peak_list.txt")
        spec = TrunkSpec("spatial", chr_ids=load_chromosome_ids(pl),
                         kernel_sizes=spatial_kernels)
    elif atac_trunk == "conv":
        # No peak list: same architecture, all-ones mask.
        spec = TrunkSpec("conv", kernel_sizes=spatial_kernels)
    else:
        spec = TrunkSpec("mlp")

    mods = (
        Modality("rna", rna, t_rna, teacher=tea["rna"], train=tr["rna"],
                 trunk=TrunkSpec("mlp"), clip=rna_clip),
        Modality("atac", atac, t_atac, teacher=tea["atac"], train=tr["atac"],
                 trunk=spec, clip=atac_clip),
    )
    st = stage_times_from([t_rna, t_atac],
                          [metas[0]["log_age"].values, metas[1]["log_age"].values],
                          len(stg))
    log.info("PFC: rna %s atac %s | %d stages | atac trunk %s | stage_times %s",
             rna.shape, atac.shape, len(stg), spec.kind,
             np.round(st, 3).tolist())
    return Dataset(name="PFC", stages=stg, modalities=mods, stage_times=st)
