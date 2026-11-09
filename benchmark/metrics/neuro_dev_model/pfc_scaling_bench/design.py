"""Fixed design of the PFC scaling benchmark.

Every number that decides what is run, on which cells, and when a method is
declared infeasible lives here, so that the scheduler, the runners and the
aggregator cannot disagree about it.  Paths come from the environment (with
defaults read from ``config/paths.yml``), never from literals elsewhere.

The design in one paragraph
---------------------------
The pool is the whole-donor TRAINING split (117,932 RNA / 64,011 ATAC nuclei),
the cells our adopted fits are trained on.  One random permutation per modality
(seed :data:`PERM_SEED`) orders the pool; tier ``N`` is the first ``N`` cells of
that permutation, so tiers are NESTED and every tier contains the evaluation
draw ``E`` = the first :data:`N_EVAL` cells of each modality.  Every method is
fitted on a tier and every fit is scored on the same ``E``, so a difference
between two scores is a difference between methods or training-set sizes, never
between evaluation cells.  A method climbs the tiers until the projected wall
time of the next tier exceeds :data:`CAP_FACTOR` times our own fit time, or
its projected memory exceeds the memory budget; that is its feasibility
frontier.  Tiers are climbed breadth-first: every method's small tiers before
anyone's large ones.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Project root: three levels above this file (repo/neuro_dev_model/pfc_scaling_bench).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _paths_yml_processed() -> Path:
    """Return the PFC processed-data directory named in ``config/paths.yml``.

    Returns
    -------
    pathlib.Path
        ``<project_root>/neuro_dev/data/processed``.
    """
    import yaml

    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "paths.yml").read_text())
    return PROJECT_ROOT / cfg["datasets"]["neuro_dev"]["processed"]


#: Directory holding ``{rna,atac}_expr.npy``, both metadata CSVs, the teachers
#: and ``atac_peak_list.txt``.
DATA_DIR: Path = Path(os.environ["PFCSB_DATA_DIR"]) if "PFCSB_DATA_DIR" in os.environ \
    else _paths_yml_processed()

#: Whole-donor training mask (library-level split; see the PFC memo).
TRAIN_MASK: Path = Path(os.environ.get(
    "PFCSB_TRAIN_MASK",
    PROJECT_ROOT / "neuro_dev_model" / "mtvae" / "assets" / "masks"
    / "wholedonor_train_mask.npz"))

#: Read-only scMultiNODE reference tree (vendored baselines).
REF_ROOT: Path = Path(os.environ.get("PFCSB_REF_ROOT", PROJECT_ROOT / "scMultiNODE_ref"))

#: Everything this benchmark writes: tiers, embeddings, records, scores.
OUT_ROOT: Path = Path(os.environ.get(
    "PFCSB_OUT", PROJECT_ROOT / "neuro_dev_model" / "pfc_scaling_bench" / "runs"))

#: Developmental order. Explicit, never sorted: lexical order is not time.
STAGES: tuple[str, ...] = ("Gestation", "Neonatal", "Infancy", "Childhood",
                           "Adolescence", "Adulthood")

#: Seed of the per-modality pool permutation that defines every tier.
PERM_SEED: int = 0

#: Cells per modality in the evaluation draw E (= the first tier).
N_EVAL: int = int(os.environ.get("PFCSB_N_EVAL", 4000))

#: Cells per modality per tier; ``None`` is the whole training split.  Doubling
#: from N_EVAL.  The 64,000 tier is all but 11 ATAC nuclei, so "full" differs
#: from it only on the RNA side.
TIERS: tuple[int | None, ...] = (4000, 8000, 16000, 32000, 64000, None)


def tier_name(n: int | None) -> str:
    """Return the file-safe label of a tier.

    Parameters
    ----------
    n : int or None
        Cells per modality, or ``None`` for the full training split.

    Returns
    -------
    str
        ``"n4000"`` ... or ``"full"``.
    """
    return "full" if n is None else f"n{n}"


#: The per-fit cap is RELATIVE to our own method (user decision 2026-10-27):
#: a tier is run now only if its projected wall time is within CAP_FACTOR times
#: our timing seed's largest measured fit (ours is ~flat in N, so that is a
#: stand-in for our full-atlas time until the full tier is measured).  A tier
#: projected beyond it is DEFERRED, not abandoned: write a larger factor to
#: ``<OUT>/CAP_FACTOR`` and the scheduler picks deferred tiers up again.
CAP_FACTOR: float = float(os.environ.get("PFCSB_CAP_FACTOR", 3.0))
CAP_REFERENCE_METHOD: str = "Ours"

#: Fallback cap, used only before our first fit has been measured.
CAP_SECONDS: float = float(os.environ.get("PFCSB_CAP_SECONDS", 4 * 3600))

#: Hard kill for a fit that runs past its projection.
HARD_TIMEOUT_SECONDS: float = float(os.environ.get("PFCSB_HARD_TIMEOUT", 6 * 3600))

#: Fraction of the memory limit a single job may be projected to use.
#: This decides a method's memory frontier (``projected_oom``).
MEM_FRACTION: float = 0.80

#: Fraction of the memory limit that the jobs running side by side may be
#: projected to use together.  Raised from MEM_FRACTION to 0.93 by user decision
#: (2026-10-27, "take the faster one") so SCOTv2 at 16k (~45 GB projected) can run
#: beside our ~3 GB GPU fits instead of alone.  Accepted risk: a job that exceeds
#: its projection can be OOM-killed; an ``oom`` record made while other jobs ran
#: beside it is a scheduling artefact, not a frontier, and is re-run alone.
ADMIT_FRACTION: float = 0.93

#: CPU threads per CPU-lane job, pinned with taskset (the GPU lane gets 8, as the
#: recorded PFC fits).  Fixed across every CPU method
#: and every tier so wall times are comparable.  Never derived from load.
THREADS: int = int(os.environ.get("PFCSB_THREADS", 5))

#: Threads per scoring job (distance correlation and Leiden dominate).
SCORE_THREADS: int = int(os.environ.get("PFCSB_SCORE_THREADS", 4))

#: The seven methods of the four-dataset benchmark, in its order.
METHODS: tuple[str, ...] = ("Ours", "scMultiNODE", "SCOTv1", "SCOTv2", "Pamona",
                            "UnionCom", "uniPort")

#: Controls, scored but not ranked:
#:   rawPCA          -- each modality PCA-50 on its own, no alignment
#:   Ours_localtea   -- ours with BOTH tilt teachers rebuilt on the tier's cells
#:                      only (bounds the advantage of full-atlas teachers)
#:   Ours_cpu        -- ours on CPU, to separate the method from the device
CONTROLS: tuple[str, ...] = ()
#: Dropped by user decision 2026-10-27 ("only the 7 methods"); rawPCA fits made
#: before that stay on disk and are left out of every table.
RETIRED_CONTROLS: tuple[str, ...] = ("rawPCA", "Ours_localtea", "Ours_cpu")

#: Seeds.  Ours uses the PFC example's seeds; the stochastic competitors use the
#: four-dataset benchmark's.  SCOT is deterministic; Pamona and UnionCom reseed
#: globally to the constant 666, so extra seeds would be identical copies.
SEEDS: dict[str, tuple[int, ...]] = {
    #: subsampled tiers only; the full tier is the adopted fits (ADOPTED_FULL)
    "Ours": (22, 44),
    "Ours_localtea": (22, 44),
    "Ours_cpu": (22,),
    #: one seed per competitor (user decision 2026-10-27); seeds 1/2 already fitted
    #: before the decision stay on disk and are left out of every table
    "scMultiNODE": (0,),
    "uniPort": (0,),
    "SCOTv1": (0,),
    "SCOTv2": (0,),
    "Pamona": (0,),
    "UnionCom": (0,),
    "rawPCA": (0,),
}

#: Device class per method.  "gpu" jobs share the one GPU lane, one at a time,
#: so no GPU timing is ever measured under contention.  CPU jobs are started
#: with CUDA_VISIBLE_DEVICES="" so none of them can touch the GPU.
DEVICE: dict[str, str] = {
    "Ours": "gpu", "Ours_localtea": "gpu", "UnionCom": "gpu",
    "Ours_cpu": "cpu", "scMultiNODE": "cpu", "uniPort": "cpu", "SCOTv1": "cpu",
    "SCOTv2": "cpu", "Pamona": "cpu", "rawPCA": "cpu",
}

#: Prior exponent of wall time in N, used only to project a tier when one
#: measured point exists.  From the four-dataset record (HC 2.3k -> HO 10k):
#: SCOTv1 2.95, SCOTv2 2.6, scMultiNODE 1.07; Pamona/UnionCom are O(N^2) per
#: iteration by construction; the minibatch methods are ~flat.
TIME_EXPONENT_PRIOR: dict[str, float] = {
    "SCOTv1": 3.0, "SCOTv2": 2.6, "Pamona": 2.0, "UnionCom": 2.0,
    "scMultiNODE": 1.5, "rawPCA": 1.0,
    #: fixed-iteration minibatch training: time is ~flat in N (measured:
    #: uniPort 16 s at 4k, 124 s at full; ours 1,516 s at 4k)
    "uniPort": 0.2, "Ours": 0.2, "Ours_localtea": 0.2, "Ours_cpu": 0.2,
}

#: Prior exponent of peak memory in N.  The O(N^2) methods hold dense N x N
#: distance / coupling matrices.
MEM_EXPONENT_PRIOR: dict[str, float] = {
    "SCOTv1": 2.0, "SCOTv2": 2.0, "Pamona": 2.0, "UnionCom": 2.0,
    "scMultiNODE": 2.0, "uniPort": 1.0, "rawPCA": 1.0,
    #: measured: 2.3 / 2.4 / 2.8 GB at 4k / 8k / 16k -- nearly flat
    "Ours": 0.3, "Ours_localtea": 0.3, "Ours_cpu": 0.3,
}

#: Which tiers each method is run at.  "chain" = climb every tier until the
#: frontier.  Competitors carry one seed (user decision 2026-10-27); ours carries
#: three at every tier.
#: Ours_localtea is a control at the matched tier, plus one seed at full so its
#: timing includes teacher construction at the largest N.  Ours_cpu is timed at
#: the two ends only.
EXTRA_SEED_TIERS: dict[str, str] = {
    "Ours": "all", "Ours_localtea": "first",
}
#: Tiers per method when not all of TIERS.  Ours is NOT refitted on the full
#: training split (user decision 2026-10-27): its full tier is the adopted
#: coad_tilt001 fits, imported by ``import_adopted.py``.
METHOD_TIERS: dict[str, tuple[int | None, ...]] = {"Ours": TIERS[:-1]}

#: The adopted PFC fits that stand in for our full tier: Stage B latents over the
#: whole atlas, and the two run records whose wall times sum to a whole fit.
#: Fitted on another machine (2026-08-25 / 08-29), so their wall time is NOT on this
#: run's clock and is drawn and tabled as such.
ADOPTED_FULL: dict[str, object] = {
    "seeds": (22, 44, 88),
    "latents": ("~/pfc_final/repo/neuro_dev_model/mtvae_coadapt/results/"
                "tiltiso_2026_08_29/coad_tilt001_s{seed}/latents/z_coadapt_{mod}.npy"),
    "stage_b_record": ("~/pfc_final/repo/neuro_dev_model/mtvae_coadapt/results/"
                       "tiltiso_2026_08_29/coad_tilt001_s{seed}/run_record.json"),
    "stage_a_record": ("~/mtvae_pfc_2026_08_25/deterministic/vae_runs/PFC/"
                       "x_w0.5_a1e-1_s{seed}_s{seed}.json"),
}

SPARSE_TIERS: dict[str, tuple[int | None, ...]] = {
    "Ours_cpu": (4000, None),
    "Ours_localtea": (4000, None),
}

#: Label used for the scorer's dataset registry entry.
DATASET_CODE: str = "PFC"


def out_dir(*parts: str) -> Path:
    """Return (and create) a directory under :data:`OUT_ROOT`.

    Parameters
    ----------
    *parts : str
        Path components below the output root.

    Returns
    -------
    pathlib.Path
        The created directory.
    """
    p = OUT_ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_tag(method: str, n: int | None, seed: int) -> str:
    """Return the canonical stem of one fit's files.

    Parameters
    ----------
    method : str
        Method or control name.
    n : int or None
        Tier size per modality, ``None`` for full.
    seed : int
        Fit seed.

    Returns
    -------
    str
        e.g. ``"SCOTv2_n16000_s0"``.
    """
    return f"{method}_{tier_name(n)}_s{seed}"
