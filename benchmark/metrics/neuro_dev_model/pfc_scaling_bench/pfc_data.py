"""PFC cells, tiers and labels for the scaling benchmark.

Three jobs, kept here so no runner re-implements them:

1. **The tiers.**  :func:`build_tiers` permutes each modality's whole-donor
   training pool once and writes the permutation; tier ``N`` is its first ``N``
   rows (:func:`tier_rows`).  Rows are always handed out IN PERMUTATION ORDER, so
   in every embedding any method writes, rows ``[:N_EVAL]`` are the evaluation
   draw ``E``.  That convention is what lets one scorer read every method.
2. **The matrices.**  :func:`load_x` returns the staged z-scored matrices
   clipped at the pipeline's ``UNIFORM_CLIP`` -- exactly what our model sees --
   for any row set, so every method is handed the same numbers.
3. **The labels.**  :func:`labels` gives the 13 coarse types (the PFC example's
   vocabulary), the stage index and the donor, and :func:`scorer_dataset` wraps
   ``E`` as the four-dataset scorer's ``DatasetData`` so the published scorer
   runs unmodified.

Run once, before anything else::

    python -m neuro_dev_model.pfc_scaling_bench.pfc_data
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_dev_model.pfc_scaling_bench import design as D

log = logging.getLogger("pfc_data")

MODALITIES: tuple[str, ...] = ("rna", "atac")
TIERS_FILE: str = "tiers.npz"


# ---------------------------------------------------------------------------
# metadata and labels
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def metadata(modality: str) -> pd.DataFrame:
    """Read one modality's cell metadata.

    Parameters
    ----------
    modality : {"rna", "atac"}
        Which table.

    Returns
    -------
    pandas.DataFrame
        One row per cell, in matrix row order.
    """
    return pd.read_csv(D.DATA_DIR / f"{modality}_metadata.csv")


def labels(modality: str) -> dict[str, np.ndarray]:
    """Per-cell labels for one modality, over ALL cells in matrix row order.

    Parameters
    ----------
    modality : {"rna", "atac"}
        Which modality.

    Returns
    -------
    dict
        ``coarse`` -- the 13 coarse types or ``"Other"`` (the PFC example's
        vocabulary, mapped by the shipped ``coarsen_*``);
        ``scorer_type`` -- the same lower-cased with ``Other`` -> ``"unknown"``,
        which the four-dataset scorer drops from every type-based endpoint;
        ``stage`` -- 0-based index into :data:`design.STAGES`;
        ``donor`` -- the library id.
    """
    from neuro_dev_model.mtvae_pipeline.pfc.bench.metrics import (
        coarsen_atac, coarsen_rna)

    meta = metadata(modality)
    if modality == "rna":
        coarse = meta["major_clust"].map(coarsen_rna).to_numpy()
    else:
        coarse = meta["predicted_group"].map(coarsen_atac).to_numpy()
    s2i = {s: i for i, s in enumerate(D.STAGES)}
    stage = meta["stage_id"].map(s2i)
    if stage.isna().any():
        bad = sorted(set(meta["stage_id"][stage.isna()]))
        raise ValueError(f"{modality}: stage labels outside STAGES: {bad}")
    scorer_type = np.array(["unknown" if c == "Other" else str(c).lower()
                            for c in coarse])
    return {"coarse": coarse.astype(str), "scorer_type": scorer_type,
            "stage": stage.to_numpy().astype(np.int64),
            "donor": meta["donor_id"].astype(str).to_numpy()}


# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------
def _md5(a: np.ndarray) -> str:
    """Return the md5 of an array's bytes.

    Parameters
    ----------
    a : numpy.ndarray
        Any array.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_tiers(seed: int = D.PERM_SEED) -> dict[str, np.ndarray]:
    """Permute each modality's training pool and write the permutation.

    Parameters
    ----------
    seed : int, optional
        Permutation seed.  Each modality draws from its own stream
        ``default_rng([seed, k])`` so adding cells to one cannot move the other.

    Returns
    -------
    dict
        ``perm_rna`` / ``perm_atac``: global row indices of the pool, permuted.
    """
    mask = np.load(D.TRAIN_MASK)
    out = {}
    for k, m in enumerate(MODALITIES):
        pool = np.flatnonzero(mask[f"train_mask_{m}"])
        if len(pool) != len(metadata(m)) and len(mask[f"train_mask_{m}"]) != len(metadata(m)):
            raise ValueError(f"{m}: mask length does not match the metadata")
        out[f"perm_{m}"] = np.random.default_rng([seed, k]).permutation(pool)
    return out


def write_tiers() -> Path:
    """Build the tiers and write them with a provenance record.

    Returns
    -------
    pathlib.Path
        The written ``tiers.npz``.

    Raises
    ------
    RuntimeError
        If a tiers file already exists and differs from a fresh build -- the
        tiers must never change under existing fits.
    """
    tiers = build_tiers()
    path = D.out_dir() / TIERS_FILE
    if path.exists():
        old = np.load(path)
        for k, v in tiers.items():
            if not np.array_equal(old[k], v):
                raise RuntimeError(f"{path} exists and differs on {k}; refusing")
        log.info("%s already present and identical", path)
        return path
    np.savez(path, **tiers)
    rec = {"perm_seed": D.PERM_SEED, "n_eval": D.N_EVAL,
           "tiers": [D.tier_name(n) for n in D.TIERS],
           "train_mask": str(D.TRAIN_MASK)}
    for m in MODALITIES:
        perm = tiers[f"perm_{m}"]
        lab = labels(m)
        e = perm[:D.N_EVAL]
        rec[m] = {
            "pool": int(len(perm)), "md5_perm": _md5(perm),
            "eval_type_counts": pd.Series(lab["coarse"][e]).value_counts().to_dict(),
            "eval_stage_counts": pd.Series(np.asarray(D.STAGES)[lab["stage"][e]]
                                           ).value_counts().to_dict(),
            "tier_sizes": {D.tier_name(n): int(min(n or len(perm), len(perm)))
                           for n in D.TIERS},
        }
    (path.with_suffix(".json")).write_text(json.dumps(rec, indent=2, default=int))
    log.info("wrote %s", path)
    return path


@lru_cache(maxsize=1)
def load_tiers() -> dict[str, np.ndarray]:
    """Read the written permutation.

    Returns
    -------
    dict
        ``perm_rna`` / ``perm_atac``.
    """
    path = D.OUT_ROOT / TIERS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m "
                                f"neuro_dev_model.pfc_scaling_bench.pfc_data` first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def tier_rows(modality: str, n: int | None) -> np.ndarray:
    """Global row indices of one tier, in permutation order.

    Parameters
    ----------
    modality : {"rna", "atac"}
        Which modality.
    n : int or None
        Cells per modality; ``None`` (or more than the pool) is the whole pool.

    Returns
    -------
    numpy.ndarray
        Row indices; the first :data:`design.N_EVAL` are the evaluation draw.
    """
    perm = load_tiers()[f"perm_{modality}"]
    return perm if n is None else perm[:min(n, len(perm))]


def eval_rows(modality: str) -> np.ndarray:
    """Global row indices of the evaluation draw ``E``.

    Parameters
    ----------
    modality : {"rna", "atac"}
        Which modality.

    Returns
    -------
    numpy.ndarray
        The first :data:`design.N_EVAL` rows of the permutation.
    """
    return tier_rows(modality, D.N_EVAL)


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------
def load_x(modality: str, rows: np.ndarray, clip: float | None = 10.0) -> np.ndarray:
    """Load the staged matrix for ``rows``, in the order given.

    The gather from the memory map is done in sorted order (sequential reads)
    and then permuted back.

    Parameters
    ----------
    modality : {"rna", "atac"}
        Which matrix.
    rows : numpy.ndarray
        Global row indices, any order.
    clip : float or None, optional
        Symmetric bound.  Defaults to the pipeline's ``UNIFORM_CLIP`` so that
        competitors receive exactly the array our encoder receives.

    Returns
    -------
    numpy.ndarray
        ``(len(rows), n_features)`` float32.
    """
    mm = np.load(D.DATA_DIR / f"{modality}_expr.npy", mmap_mode="r")
    order = np.argsort(rows, kind="stable")
    x = np.empty((len(rows), mm.shape[1]), dtype=np.float32)
    x[order] = np.asarray(mm[rows[order]], dtype=np.float32)
    if clip is not None:
        np.clip(x, -clip, clip, out=x)
    return x


# ---------------------------------------------------------------------------
# the four-dataset scorer, pointed at E
# ---------------------------------------------------------------------------
def _register_spec() -> None:
    """Add a PFC entry to the four-dataset scorer's registry, in memory only.

    ``score_latent`` looks its dataset up by code to decide whether the
    co-assay endpoints apply; PFC is unpaired, so they do not.
    """
    from neuro_dev_model.mtvae_4ds_coadapt.data import snr_config as cfg

    if D.DATASET_CODE in cfg.DATASETS:
        return
    cfg.DATASETS[D.DATASET_CODE] = cfg.DatasetSpec(
        code=D.DATASET_CODE, ref_name="herring_pfc", archive_dir="",
        label="Human prefrontal cortex (Herring 2022, unpaired)",
        rna_expr="rna_expr.npy", atac_expr="atac_expr.npy",
        rna_meta="rna_metadata.csv", atac_meta="atac_metadata.csv",
        n_rna=len(metadata("rna")), n_atac=len(metadata("atac")),
        n_timepoints=len(D.STAGES), coassay=False, tp_col="stage_id",
        rna_type_col="major_clust", atac_type_col="predicted_group",
        rna_batch_col="donor_id", atac_batch_col="donor_id",
        shared_type_vocabulary=True,
        notes=("13 coarse types after coarsen_rna/coarsen_atac; 'Other' scored "
               "as unknown. ATAC labels are Seurat transfers from the RNA "
               "reference (mean confidence 0.497)."))


def scorer_dataset():
    """Wrap the evaluation draw as the four-dataset scorer's ``DatasetData``.

    Returns
    -------
    snr_data.DatasetData
        Rows in ``E`` order.  ``X`` is a zero-width placeholder: the scorer reads
        only labels, timepoints and batches.  ``tp`` is 1-based, as the scorer's
        loader produces it.
    """
    from neuro_dev_model.mtvae_4ds_coadapt.data import snr_config as cfg
    from neuro_dev_model.mtvae_4ds_coadapt.data import snr_data

    _register_spec()
    mods = {}
    for m in MODALITIES:
        r = eval_rows(m)
        lab = labels(m)
        meta = metadata(m).iloc[r].reset_index(drop=True)
        mods[m] = snr_data.ModalityData(
            X=np.empty((len(r), 0), dtype=np.float32),
            barcodes=meta["barcode"].astype(str).to_numpy(),
            features=np.empty(0, dtype=str),
            tp=lab["stage"][r] + 1,
            cell_type=lab["scorer_type"][r],
            batch=lab["donor"][r],
            meta=meta.set_index("barcode"))
    return snr_data.DatasetData(spec=cfg.dataset(D.DATASET_CODE),
                                rna=mods["rna"], atac=mods["atac"],
                                mode="harmonised")


def main() -> None:
    """Write the tiers and log the evaluation draw's composition."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = write_tiers()
    rec = json.loads(path.with_suffix(".json").read_text())
    for m in MODALITIES:
        log.info("%s pool %d | tiers %s", m, rec[m]["pool"], rec[m]["tier_sizes"])
        log.info("%s E types %s", m, rec[m]["eval_type_counts"])


if __name__ == "__main__":
    main()
