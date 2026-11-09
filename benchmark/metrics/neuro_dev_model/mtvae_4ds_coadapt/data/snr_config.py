"""Dataset registry and shared configuration for the scMultiNODE replication benchmark.

This module is the single source of truth for the four datasets evaluated in
``scmultinode.pdf`` §3 -- human cortex (HC), human organoid (HO), drosophila
embryogenesis (DR) and mouse neocortex (MN).  Every other module in
``scmultinode_replication`` resolves paths and column names through here so that
all four datasets travel through one code path.

The registry values below were verified directly against the released
``reduce_processed`` CSVs (figshare ``10.6084/m9.figshare.27420657.v2``, member
``data.zip``, file id 50130363), not copied from the paper.  Anything the data
contradicts is recorded in :data:`DATASETS` with a comment.

Notes
-----
The reference loader ``scMultiNODE_ref/utils/FileUtils.loadSCData`` branches per
dataset and does *not* feed every dataset through identical preprocessing.  The
whole point of this module plus :mod:`snr_data` is to replace those branches with
one contract, so that a method comparison is a comparison of methods and not of
loaders.

Environment
-----------
``SNR_DATA_ROOT``
    Root holding the extracted archive tree (``<root>/data/<archive_dir>/...``).
    Defaults to ``~/data/archive``.
``SNR_RESULTS_ROOT``
    Root for tables, figures and run artifacts.  Defaults to
    ``~/results``.
``SNR_RUNS_ROOT``
    Root for per-method latent outputs.  Defaults to ``~/runs``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
DATA_ROOT: Path = Path(os.environ.get("SNR_DATA_ROOT", "~/data/archive"))
RESULTS_ROOT: Path = Path(os.environ.get("SNR_RESULTS_ROOT", "~/results"))
RUNS_ROOT: Path = Path(os.environ.get("SNR_RUNS_ROOT", "~/runs"))
AUTHORS_ROOT: Path = Path(
    os.environ.get("SNR_AUTHORS_ROOT", "~/data/authors_metrics"))

TABLES_DIR: Path = RESULTS_ROOT / "tables"
FIGURES_DIR: Path = RESULTS_ROOT / "figures"
PREFLIGHT_DIR: Path = RESULTS_ROOT / "preflight"

# ---------------------------------------------------------------------------
# Benchmark-wide constants (scmultinode.pdf §3)
# ---------------------------------------------------------------------------
LATENT_DIM: int = 50
N_HVG: int = 2000
#: Seeds for stochastic methods.  The handoff's iron rule: never a single fit.
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: k for the label-transfer classifier, matching Compare_Modal_Alignment_Metric.py.
LTA_N_NEIGHBORS: int = 50
#: k for neighborhood overlap, matching Compare_Modal_Alignment_Metric.py.
NEIGH_OVERLAP_K: int = 50


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to load and interpret one benchmark dataset.

    Attributes
    ----------
    code : str
        Two-letter benchmark code (``HC``, ``HO``, ``DR``, ``MN``).
    ref_name : str
        Name used by ``scMultiNODE_ref`` (``modal_integration.DATA_DIR_DICT`` key
        and the prefix of every published artifact filename).
    archive_dir : str
        Directory name inside the figshare ``data.zip`` tree.
    label : str
        Human-readable name for figures and tables.
    rna_expr, atac_expr : str
        Expression CSV filenames inside ``reduce_processed``.  These differ
        between datasets -- HC/HO/MN use ``all-<MOD>-data-hvg.csv`` while DR uses
        ``all-<MOD>_count_data.csv``.  This is trap §6.5 (loader asymmetry).
    rna_meta, atac_meta : str
        Cell-metadata CSV filenames.  DR again differs
        (``all_<MOD>_meta_data.csv``).
    n_rna, n_atac : int
        Cell counts verified against the released CSVs.
    n_timepoints : int
        Number of distinct timepoints.
    coassay : bool
        Whether the two modalities are the same cells (enables FOSCTTM,
        neighborhood overlap and SCC).
    tp_col : str
        Integer timepoint column, present in both modalities' metadata.
    rna_type_col, atac_type_col : str
        Cell-type annotation column, per modality.  They are *not* always the
        same column name, and their label vocabularies are *not* always the same
        set -- see :attr:`shared_type_vocabulary`.
    rna_batch_col, atac_batch_col : str or None
        Donor / batch / replicate column used for the batch control (§5.1).
        ``None`` means the dataset cannot support a batch control on that side.
    type_is_circular_with_time : bool
        True when the cell-type column is a relabelling of the timepoint /
        developmental-stage column, making every type-resolved metric circular
        with time.  Set for HO, where
        ``HumanOrganoid_Processing.py`` assigns ``meta_df["cell_type"] =
        meta_df.stage``.
    alt_type_cols : tuple of str
        Non-circular alternatives available in the released metadata, reported
        alongside the circular column so the reader can see both.
    shared_type_vocabulary : bool
        Whether the RNA and ATAC cell-type label sets are the same.  When False,
        cross-modal LTA-type is bounded above by the label overlap and *must* be
        reported with its map ceiling (§5.2).
    notes : str
        Free text recorded into the pre-flight report.
    """

    code: str
    ref_name: str
    archive_dir: str
    label: str
    rna_expr: str
    atac_expr: str
    rna_meta: str
    atac_meta: str
    n_rna: int
    n_atac: int
    n_timepoints: int
    coassay: bool
    tp_col: str
    rna_type_col: str
    atac_type_col: str
    rna_batch_col: str | None
    atac_batch_col: str | None
    type_is_circular_with_time: bool = False
    alt_type_cols: tuple[str, ...] = ()
    shared_type_vocabulary: bool = True
    notes: str = ""

    @property
    def reduce_dir(self) -> Path:
        """Return the dataset's ``reduce_processed`` directory."""
        return DATA_ROOT / "data" / self.archive_dir / "reduce_processed"

    def expr_path(self, modality: str) -> Path:
        """Return the expression CSV path for ``modality`` (``"rna"``/``"atac"``)."""
        return self.reduce_dir / (self.rna_expr if modality == "rna" else self.atac_expr)

    def meta_path(self, modality: str) -> Path:
        """Return the cell-metadata CSV path for ``modality``."""
        return self.reduce_dir / (self.rna_meta if modality == "rna" else self.atac_meta)

    def type_col(self, modality: str) -> str:
        """Return the cell-type column name for ``modality``."""
        return self.rna_type_col if modality == "rna" else self.atac_type_col

    def batch_col(self, modality: str) -> str | None:
        """Return the batch/donor column name for ``modality``, or ``None``."""
        return self.rna_batch_col if modality == "rna" else self.atac_batch_col


# ---------------------------------------------------------------------------
# The four datasets
# ---------------------------------------------------------------------------
HC = DatasetSpec(
    code="HC",
    ref_name="coassay_cortex",
    archive_dir="human_prefrontal_cortex_multiomic",
    label="Human cortex (co-assay)",
    rna_expr="all-RNA-data-hvg.csv",
    atac_expr="all-ATAC-data-hvg.csv",
    rna_meta="all-RNA-cell_meta.csv",
    atac_meta="all-ATAC-cell_meta.csv",
    n_rna=2277,
    n_atac=2277,
    n_timepoints=10,
    coassay=True,
    tp_col="tp",
    rna_type_col="author_cell_type",
    atac_type_col="author_cell_type",
    rna_batch_col="donor_id",
    atac_batch_col="donor_id",
    alt_type_cols=("cell_type",),
    notes=(
        "NOT the Herring PFC atlas used in the paper's Example 2. This is the "
        "co-assayed 10x multiome cortex; Herring is unpaired with zero barcode "
        "overlap. 12 donors across 10 timepoints, so a within-timepoint donor "
        "replicate control is possible only where a timepoint holds >1 donor."
    ),
)

HO = DatasetSpec(
    code="HO",
    ref_name="human_organoid",
    archive_dir="human_organoid_Fleck2022",
    label="Human organoid (co-assay)",
    rna_expr="all-RNA-data-hvg.csv",
    atac_expr="all-ATAC-data-hvg.csv",
    rna_meta="all-RNA-cell_meta.csv",
    atac_meta="all-ATAC-cell_meta.csv",
    n_rna=10000,
    n_atac=10000,
    n_timepoints=11,
    coassay=True,
    tp_col="tp",
    rna_type_col="cell_type",
    atac_type_col="cell_type",
    rna_batch_col="line",
    atac_batch_col="line",
    type_is_circular_with_time=True,
    alt_type_cols=("lineage", "lineage_coarse", "nowakowski_prediction", "pred_class"),
    notes=(
        "CIRCULARITY: cell_type has exactly the 4 levels of `stage` "
        "(EB/nect/nepi/organoid) because HumanOrganoid_Processing.py assigns "
        "meta_df['cell_type'] = meta_df.stage. Every type-resolved metric on HO "
        "(NMI, LTA-type, silhouette) therefore scores a developmental-stage "
        "label, not a cell type, and is circular with the time axis. Report the "
        "`lineage` column alongside as the non-circular alternative."
    ),
)

DR = DatasetSpec(
    code="DR",
    ref_name="drosophila",
    archive_dir="drosophila_embryonic",
    label="Drosophila embryogenesis (unaligned)",
    rna_expr="all-RNA_count_data.csv",
    atac_expr="all-ATAC_count_data.csv",
    rna_meta="all_RNA_meta_data.csv",
    atac_meta="all_ATAC_meta_data.csv",
    n_rna=2738,
    n_atac=4246,
    n_timepoints=11,
    coassay=False,
    tp_col="tp",
    rna_type_col="cell_type",
    atac_type_col="cell_type",
    rna_batch_col="orig.ident",   # exp1..exp12, 7 levels
    atac_batch_col="sample",      # exp<k>_hrs<..>_b<k>, 16 levels
    shared_type_vocabulary=False,
    alt_type_cols=("germ_layer",),
    notes=(
        "Filenames differ from every other dataset (all-<MOD>_count_data.csv, "
        "all_<MOD>_meta_data.csv). RNA carries 51 cell_type levels and ATAC 38; "
        "the vocabularies are not identical, so cross-modal LTA-type is bounded "
        "by label overlap and needs its map ceiling. `germ_layer` (RNA only) is "
        "the coarse shared alternative. tp is 1-indexed here, unlike HC/HO/MN."
    ),
)

MN = DatasetSpec(
    code="MN",
    ref_name="mouse_neocortex",
    archive_dir="Yuan2022_MouseNeocortex",
    label="Mouse neocortex (unaligned)",
    rna_expr="all-RNA-data-hvg.csv",
    atac_expr="all-ATAC-data-hvg.csv",
    rna_meta="all-RNA-cell_meta.csv",
    atac_meta="all-ATAC-cell_meta.csv",
    n_rna=6098,
    n_atac=1914,
    n_timepoints=3,
    coassay=False,
    tp_col="tp",
    rna_type_col="CellType",
    atac_type_col="CellType",
    rna_batch_col="combId",       # 65 levels: <timePoint>_<rep>_<type>
    atac_batch_col="replicate",
    shared_type_vocabulary=False,
    notes=(
        "Only 3 timepoints (P1/P7/P21) -- include it because the paper does, but "
        "no dynamics claim may rest on it. RNA and ATAC cell-type vocabularies "
        "differ (RNA has Inh_Vip/Microglia/Pericytes/COP/SCPN; ATAC has "
        "Inh_Pval/Inh_Npy/MFOL/Unknown and spells it ScPN), so cross-modal "
        "LTA-type is capped well below 1 by label mismatch alone. ATAC carries a "
        "true `replicate` column; RNA's replicate is embedded in `combId`."
    ),
)

#: Registry keyed by benchmark code, in the paper's presentation order.
DATASETS: dict[str, DatasetSpec] = {d.code: d for d in (HC, HO, DR, MN)}

#: Co-assay datasets, which alone support FOSCTTM / neighborhood overlap / SCC.
COASSAY_CODES: tuple[str, ...] = tuple(c for c, d in DATASETS.items() if d.coassay)

#: Method panel.  ``scMultiNODE`` plus the paper's six baselines, plus the
#: unintegrated control and our own model.
BASELINE_METHODS: tuple[str, ...] = (
    "SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat",
)
#: Methods that are deterministic and must be reported as a single value, never
#: as a band with sd = 0.  SCOT has zero rand/shuffle/seed/permut hits.
DETERMINISTIC_METHODS: frozenset[str] = frozenset({"SCOTv1", "SCOTv2", "rawPCA"})


def dataset(code: str) -> DatasetSpec:
    """Look up a dataset by its two-letter code.

    Parameters
    ----------
    code : str
        One of ``"HC"``, ``"HO"``, ``"DR"``, ``"MN"`` (case-insensitive).

    Returns
    -------
    DatasetSpec
        The registry entry.

    Raises
    ------
    KeyError
        If ``code`` is not a registered dataset.
    """
    key = code.upper()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {code!r}; known: {sorted(DATASETS)}")
    return DATASETS[key]


def ensure_dirs() -> None:
    """Create the results directory tree if it does not already exist."""
    for path in (TABLES_DIR, FIGURES_DIR, PREFLIGHT_DIR, RUNS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
