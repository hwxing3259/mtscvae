"""Unified data loading for the scMultiNODE replication benchmark."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from neuro_dev_model.mtvae.data import snr_config as cfg

log = logging.getLogger("snr_data")

Mode = Literal["reference", "harmonised"]

#: Cell-type column the *reference* loader ends up using, per dataset and
#: modality.  These are not always the registry's preferred column: the
#: reference uses HC's ontology ``cell_type`` (13 levels) rather than
#: ``author_cell_type`` (15 levels), and DR reads its annotation from the large
#: ``{rna,atac}_meta.csv`` files rather than the released small ones.
REFERENCE_TYPE_COL: dict[str, dict[str, str]] = {
    "HC": {"rna": "cell_type", "atac": "cell_type"},
    "HO": {"rna": "cell_type", "atac": "cell_type"},
    "DR": {"rna": "cell_type", "atac": "cell_type"},   # == manual_annot / refined_annotation
    "MN": {"rna": "CellType", "atac": "CellType"},
}


@dataclass
class ModalityData:
    """One modality of one dataset, after loading.

    Attributes
    ----------
    X : numpy.ndarray, shape (n_cells, n_features)
        Expression / accessibility matrix.
    barcodes : numpy.ndarray of str
        Cell identifiers, in row order.
    features : numpy.ndarray of str
        Feature names, in column order.
    tp : numpy.ndarray of int
        Integer timepoint per cell, 1-indexed (matching the reference, which
        adds 1 to the released ``tp`` column).
    cell_type : numpy.ndarray of str
        Cell-type label per cell.
    batch : numpy.ndarray of str or None
        Donor / batch / replicate label per cell, when the dataset has one.
    meta : pandas.DataFrame
        Full metadata table, indexed by barcode.
    """

    X: np.ndarray
    barcodes: np.ndarray
    features: np.ndarray
    tp: np.ndarray
    cell_type: np.ndarray
    batch: np.ndarray | None
    meta: pd.DataFrame

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features."""
        return int(self.X.shape[1])


@dataclass
class DatasetData:
    """Both modalities of one dataset.

    Attributes
    ----------
    spec : cfg.DatasetSpec
        Registry entry.
    rna, atac : ModalityData
        The two modalities.
    mode : str
        Which loading contract was applied.
    """

    spec: cfg.DatasetSpec
    rna: ModalityData
    atac: ModalityData
    mode: Mode

    def modality(self, name: str) -> ModalityData:
        """Return the named modality (``"rna"`` or ``"atac"``)."""
        return self.rna if name == "rna" else self.atac


def _normalize_log1p(X: np.ndarray, counts_per_cell_after: float = 1e4) -> np.ndarray:
    """Total-count normalise each cell then log1p, as ``FileUtils.preprocess``.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_cells, n_features)
        Count matrix.
    counts_per_cell_after : float, optional
        Target total per cell.

    Returns
    -------
    numpy.ndarray
        Normalised, log1p-transformed matrix.
    """
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(X / totals * counts_per_cell_after)


def _load_matrix(path, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one expression CSV.

    Parameters
    ----------
    path : pathlib.Path
        CSV path, cells in rows and features in columns, with an index column.
    dtype : numpy.dtype, optional
        Output dtype.

    Returns
    -------
    tuple
        ``(X, barcodes, features)``.
    """
    df = pd.read_csv(path, header=0, index_col=0)
    return (df.to_numpy(dtype=dtype),
            df.index.to_numpy().astype(str),
            df.columns.to_numpy().astype(str))


def _load_meta(spec: cfg.DatasetSpec, modality: str,
               barcodes: np.ndarray) -> pd.DataFrame:
    """Read and align one modality's metadata to the matrix row order.

    Parameters
    ----------
    spec : cfg.DatasetSpec
        Registry entry.
    modality : {'rna', 'atac'}
        Modality.
    barcodes : numpy.ndarray
        Row order to align to.

    Returns
    -------
    pandas.DataFrame
        Metadata reindexed to ``barcodes``.

    Raises
    ------
    ValueError
        If any barcode is missing from the metadata.
    """
    meta = pd.read_csv(spec.meta_path(modality), header=0, index_col=0,
                       low_memory=False)
    # DR's small metadata files keep the barcode in the index already; the large
    # ones the reference reads use a "cell" column. Support both.
    if not set(barcodes).issubset(set(meta.index.astype(str))):
        if "cell" in meta.columns:
            meta = meta.set_index("cell")
    meta.index = meta.index.astype(str)
    missing = set(barcodes) - set(meta.index)
    if missing:
        raise ValueError(
            f"{spec.code}/{modality}: {len(missing)} barcodes absent from "
            f"{spec.meta_path(modality).name} (e.g. {sorted(missing)[:3]})")
    return meta.loc[barcodes]


def load_dataset(code: str, mode: Mode = "harmonised") -> DatasetData:
    """Load both modalities of one dataset under the requested contract.

    Parameters
    ----------
    code : str
        Dataset code (``HC``, ``HO``, ``DR``, ``MN``).
    mode : {'reference', 'harmonised'}, optional
        ``reference`` reproduces ``FileUtils.loadSCData`` including its
        per-dataset preprocessing asymmetry.  ``harmonised`` applies
        total-count normalisation + log1p to **both** modalities of **every**
        dataset, so that all four are on one scale.

    Returns
    -------
    DatasetData
        Both modalities with aligned labels.
    """
    spec = cfg.dataset(code)
    log.info("loading %s (%s) in %s mode", spec.code, spec.label, mode)

    out: dict[str, ModalityData] = {}
    for modality in ("rna", "atac"):
        X, barcodes, features = _load_matrix(spec.expr_path(modality))
        meta = _load_meta(spec, modality, barcodes)

        if mode == "reference":
            X = _reference_preprocess(spec, modality, X)
        else:
            X = _normalize_log1p(X.astype(np.float64)).astype(np.float32)

        type_col = (REFERENCE_TYPE_COL[spec.code][modality] if mode == "reference"
                    else spec.type_col(modality))
        if type_col not in meta.columns:
            raise KeyError(
                f"{spec.code}/{modality}: cell-type column {type_col!r} not in "
                f"metadata; available: {list(meta.columns)[:20]}")

        tp = meta[spec.tp_col].to_numpy()
        # The reference adds 1 to the released tp for HC/HO/MN and builds DR's
        # tp as natsorted(time) + 1. The released DR tp is already 1-indexed.
        tp = tp.astype(float)
        if spec.code != "DR":
            tp = tp + 1
        tp = tp.astype(int)

        bcol = spec.batch_col(modality)
        batch = (meta[bcol].astype(str).to_numpy()
                 if bcol and bcol in meta.columns else None)

        out[modality] = ModalityData(
            X=X, barcodes=barcodes, features=features, tp=tp,
            cell_type=meta[type_col].astype(str).to_numpy(),
            batch=batch, meta=meta,
        )
        log.info("  %s: X=%s tp=%d..%d types=%d", modality, X.shape,
                 tp.min(), tp.max(), len(np.unique(out[modality].cell_type)))

    data = DatasetData(spec=spec, rna=out["rna"], atac=out["atac"], mode=mode)
    if mode == "reference":
        data = apply_reference_ordering(data)
    return data


def apply_reference_ordering(data: DatasetData) -> DatasetData:
    """Reproduce the cell reordering and label lowercasing the reference applies before scoring.

    ``Compare_Modal_Alignment_Metric.py`` does three things between loading and
    scoring that determine the row order of every published latent, and that a
    naive reimplementation silently omits:

    1. Cell types are lowercased (``str(x).lower()``, line 206-207).  This is why
       drosophila's hard-coded lowercase five-type list matches its Title-Case
       ``refined_annotation`` vocabulary.
    2. Cells are regrouped into per-timepoint blocks -- for ``t`` in
       ``1..n_tps`` the cells with ``tp == t`` are concatenated in order (lines
       208-223).  The saved latents are in this order, not the CSV order.
    3. The timepoint vector is then rebuilt from ``enumerate`` over those blocks
       (lines 224-225), making it **0-indexed** rather than the 1-indexed value
       used to form the blocks.

    Omitting step 2 leaves labels misaligned with the latents, which corrupts
    every label-dependent metric while leaving latent-only metrics (FOSCTTM,
    SCC, neighborhood overlap) untouched -- a failure mode that is invisible
    unless the two families are compared.

    Parameters
    ----------
    data : DatasetData
        Dataset loaded in CSV order with 1-indexed timepoints.

    Returns
    -------
    DatasetData
        A new object with cells reordered into timepoint blocks, cell types
        lowercased, and ``tp`` renumbered from 0.
    """
    out: dict[str, ModalityData] = {}
    for name in ("rna", "atac"):
        m = data.modality(name)
        blocks = [np.where(m.tp == t)[0] for t in range(1, int(m.tp.max()) + 1)]
        blocks = [b for b in blocks if b.size > 0]
        order = np.concatenate(blocks)
        new_tp = np.concatenate([np.repeat(i, b.size) for i, b in enumerate(blocks)])
        out[name] = ModalityData(
            X=m.X[order],
            barcodes=m.barcodes[order],
            features=m.features,
            tp=new_tp,
            cell_type=np.asarray([str(x).lower() for x in m.cell_type[order]]),
            batch=None if m.batch is None else m.batch[order],
            meta=m.meta.iloc[order],
        )
        log.info("  %s: reference ordering applied, tp now %d..%d",
                 name, new_tp.min(), new_tp.max())
    return DatasetData(spec=data.spec, rna=out["rna"], atac=out["atac"],
                       mode=data.mode)


def _reference_preprocess(spec: cfg.DatasetSpec, modality: str,
                          X: np.ndarray) -> np.ndarray:
    """Apply the reference loader's per-dataset preprocessing branch.

    Parameters
    ----------
    spec : cfg.DatasetSpec
        Registry entry.
    modality : {'rna', 'atac'}
        Modality.
    X : numpy.ndarray
        Raw matrix as read from disk.

    Returns
    -------
    numpy.ndarray
        Matrix after the branch that ``loadSCData`` applies to this dataset.
    """
    if modality == "rna":
        # HC alone is left untouched: the loader comments "already normalized".
        if spec.code == "HC":
            return X.astype(np.float32)
        return _normalize_log1p(X.astype(np.float64)).astype(np.float32)
    # No ATAC branch normalises for these four datasets; the loader only casts.
    return X.astype(np.float32)


def subsample(data: DatasetData, n_max: int, seed: int = 0) -> DatasetData:
    """Subsample both modalities to at most ``n_max`` cells each, stratified by timepoint.

    Used for the Tier A row of ``tab:nd-benchmark``, where every method must
    retrain on identical cells.

    Parameters
    ----------
    data : DatasetData
        Dataset to subsample.
    n_max : int
        Maximum cells per modality.
    seed : int, optional
        RNG seed.

    Returns
    -------
    DatasetData
        A new object holding the subsampled cells.
    """
    rng = np.random.default_rng(seed)
    picked: dict[str, ModalityData] = {}
    for name in ("rna", "atac"):
        m = data.modality(name)
        if m.n_cells <= n_max:
            picked[name] = m
            continue
        frac = n_max / m.n_cells
        keep: list[np.ndarray] = []
        for t in np.unique(m.tp):
            idx = np.where(m.tp == t)[0]
            take = max(1, int(round(len(idx) * frac)))
            keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
        sel = np.sort(np.concatenate(keep))[:n_max]
        picked[name] = ModalityData(
            X=m.X[sel], barcodes=m.barcodes[sel], features=m.features,
            tp=m.tp[sel], cell_type=m.cell_type[sel],
            batch=None if m.batch is None else m.batch[sel],
            meta=m.meta.iloc[sel],
        )
    return DatasetData(spec=data.spec, rna=picked["rna"], atac=picked["atac"],
                       mode=data.mode)


def load_authors_latent(code: str, method: str,
                        latent_dim: int = cfg.LATENT_DIM) -> dict[str, np.ndarray]:
    """Load one of the authors' published integrated latents.

    Parameters
    ----------
    code : str
        Dataset code.
    method : str
        Method name as it appears in the published filenames (``scMultiNODE``,
        ``SCOTv1``, ``SCOTv2``, ``Pamona``, ``UnionCom``, ``uniPort``,
        ``Seurat``).
    latent_dim : int, optional
        Latent dimensionality, 50 in the paper.

    Returns
    -------
    dict
        ``{"rna": array, "atac": array}``.
    """
    spec = cfg.dataset(code)
    root = cfg.AUTHORS_ROOT / "modal_integration" / "res" / "model_latent"
    stem = f"{spec.ref_name}-reduce-all-{method}-{latent_dim}dim"
    if method == "Seurat":
        rna = pd.read_csv(root / f"{stem}-rna.csv", header=0, index_col=None).values
        atac = pd.read_csv(root / f"{stem}-atac.csv", header=0, index_col=None).values
        return {"rna": np.asarray(rna, dtype=np.float64),
                "atac": np.asarray(atac, dtype=np.float64)}
    res = np.load(root / f"{stem}.npy", allow_pickle=True).item()
    return {"rna": np.asarray(res["rna_integrated"], dtype=np.float64),
            "atac": np.asarray(res["atac_integrated"], dtype=np.float64)}


def load_authors_metrics(code: str) -> dict[str, dict[str, float]]:
    """Load the authors' published metric dictionary for one dataset.

    Parameters
    ----------
    code : str
        Dataset code.

    Returns
    -------
    dict
        Mapping of method name to its published metrics.
    """
    spec = cfg.dataset(code)
    path = (cfg.AUTHORS_ROOT / "modal_integration" / "res" / "comparison"
            / f"{spec.ref_name}-reduce-all-ae-{cfg.LATENT_DIM}dim-metric.npy")
    return np.load(path, allow_pickle=True).item()
