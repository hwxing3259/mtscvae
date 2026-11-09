"""Rare-population silhouette -- a faithful port of the authors' v3 analysis.

Source
------
``rsinghlab/scMultiNODE`` commit ``c4e444d2`` (2026-08-20), file
``downstream_analysis/rare_population/Rare_Population_Identification.py``.  This
is the code behind ``scmultinode.pdf`` Fig. 3C/D and it was released *after* the
snapshot vendored in ``scMultiNODE_ref/`` (whose README still points at bioRxiv
v2 and which contains no silhouette code at all).

The rule, restated so a reader does not have to diff two repositories
------------------------------------------------------------------------
1. Cell-type labels are ``str(x).lower()``; on mouse neocortex the label
   ``"undetermined"`` is remapped to ``"unknown"`` (their ``_remapUnknown``).
2. Cells carrying any of :data:`EXCLUDE_LABELS` are dropped **from the cell set
   entirely** -- they are absent from the silhouette background, not merely
   excluded from the candidate list.
3. Per modality *independently*, the surviving cell types are ranked by count
   ascending and the :data:`N_RARE` least abundant are taken (3 everywhere
   except human organoid, where it is 1).
4. Each of those types is scored *separately* with a one-vs-rest silhouette
   (``silhouette_samples`` on ``ct`` vs the rest, averaged over ``ct``'s cells),
   and the reported value is the ``nanmean`` of those N scores -- an average of
   separate scores, **not** a pooled rare-group silhouette.
5. The ``(rare=x.x%)`` annotation is ``100 * sum(counts of the N rare types) /
   (total retained cells)``, computed per modality.

Why the fraction differs between the C and D panels on DR/MN but not on HC/HO:
selection happens per modality, and on the co-assay datasets the two modalities
are the same cells.

Contracts
---------
``reference``
    Row order and labels as :func:`snr_data.apply_reference_ordering` produces
    them -- per-timepoint blocks, lowercased, reference cell-type column.  This
    is the order the authors' published ``.npy`` latents are stored in.
``harmonised``
    Original CSV row order, which is what every latent produced by this project
    is stored in, but still read from the **reference** cell-type column, since
    that column is what defines the paper's rare classes.  Mixing the two is the
    one mistake that silently produces a plausible wrong number.

Run from the project root::

    SNR_DATA_ROOT=neuro_dev/benchmark_datasets \\
    python -m scripts.scmultinode_replication.snr_rare_population --help
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from neuro_dev_model.mtvae.data import snr_config as cfg
from neuro_dev_model.mtvae.data import snr_data

log = logging.getLogger("snr_rare_population")

Contract = Literal["reference", "harmonised"]

#: Labels dropped from the cell set before anything is measured.  Verbatim from
#: ``Rare_Population_Identification.py``.
EXCLUDE_LABELS: tuple[str, ...] = ("unknown", "nan", "none", "")

#: Number of least-abundant cell types averaged into one bar.
DEFAULT_N_RARE: int = 3
N_RARE: dict[str, int] = {"HO": 1}

#: The baseline the authors draw as "Best Integ.".  Hard-coded per dataset in
#: their script -- it is *not* an argmax over the six baselines, so whether it is
#: actually the best on this metric is an open question this port can answer.
BEST_BASELINE: dict[str, str] = {
    "HC": "SCOTv2", "HO": "UnionCom", "DR": "Pamona", "MN": "SCOTv2",
}


def n_rare_for(code: str) -> int:
    """Return the number of rare cell types averaged for one dataset.

    Parameters
    ----------
    code : str
        Dataset code (``HC``, ``HO``, ``DR``, ``MN``).

    Returns
    -------
    int
        ``N_RARE[code]`` if present, else :data:`DEFAULT_N_RARE`.
    """
    return N_RARE.get(code.upper(), DEFAULT_N_RARE)


def remap_unknown(labels: np.ndarray, code: str) -> np.ndarray:
    """Apply the authors' per-dataset label remap.

    Mouse neocortex carries an ``"undetermined"`` annotation that their
    ``_remapUnknown`` folds into ``"unknown"``, which :data:`EXCLUDE_LABELS`
    then drops.  Without this step the MN rare class is a 5-cell junk label.

    Parameters
    ----------
    labels : numpy.ndarray
        Lower-cased cell-type labels.
    code : str
        Dataset code.

    Returns
    -------
    numpy.ndarray
        Labels with the remap applied.
    """
    out = np.asarray(labels).astype(str).copy()
    if code.upper() == "MN":
        out[out == "undetermined"] = "unknown"
    return out


def group_silhouette(latent: np.ndarray, labels: np.ndarray, cell_type: str,
                     exclude: Sequence[str] = EXCLUDE_LABELS) -> float:
    """One-vs-rest silhouette for a single cell type.

    Port of their ``groupSilhouette``: excluded labels are removed from the
    whole cell set first, the remainder is partitioned into ``cell_type`` versus
    everything else, and the mean silhouette over ``cell_type``'s cells is
    returned.

    Parameters
    ----------
    latent : numpy.ndarray, shape (n, d)
        Embedding, row-aligned with ``labels``.
    labels : numpy.ndarray, shape (n,)
        Lower-cased, remapped cell-type labels.
    cell_type : str
        The type to score.
    exclude : sequence of str, optional
        Labels dropped from the cell set.

    Returns
    -------
    float
        Mean one-vs-rest silhouette over ``cell_type``'s cells, or NaN when the
        partition is degenerate (fewer than 3 retained cells, or ``cell_type``
        is empty / is everything).
    """
    latent = np.asarray(latent)
    labels = np.asarray(labels)
    keep = ~np.isin(labels, list(exclude))
    x_mat, lab = latent[keep], labels[keep]
    if x_mat.shape[0] < 3:
        return float("nan")
    binary = (lab == cell_type).astype(int)
    n_ct = int(binary.sum())
    if n_ct == 0 or n_ct == len(binary):
        return float("nan")
    sil = silhouette_samples(x_mat, binary)
    return float(np.mean(sil[binary == 1]))


def rare_entries(labels: np.ndarray, n_rare: int,
                 exclude: Sequence[str] = EXCLUDE_LABELS
                 ) -> list[tuple[str, int]]:
    """Return the ``n_rare`` least-abundant retained cell types with their counts.

    Parameters
    ----------
    labels : numpy.ndarray
        Lower-cased, remapped cell-type labels.
    n_rare : int
        How many least-abundant types to take.
    exclude : sequence of str, optional
        Labels dropped from the candidate list.

    Returns
    -------
    list of tuple
        ``(cell_type, count)`` ordered by ascending count.  Ties break on the
        alphabetical order ``numpy.unique`` returns, matching the authors'
        stable sort.
    """
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    valid = [(str(u), int(c)) for u, c in zip(uniq, counts) if u not in exclude]
    return sorted(valid, key=lambda e: e[1])[:n_rare]


def rare_population(latent: np.ndarray, labels: np.ndarray, code: str,
                    exclude: Sequence[str] = EXCLUDE_LABELS) -> dict[str, Any]:
    """Score one latent's rare populations under the authors' v3 rule.

    Parameters
    ----------
    latent : numpy.ndarray, shape (n, d)
        Embedding for one modality, row-aligned with ``labels``.
    labels : numpy.ndarray, shape (n,)
        Lower-cased, remapped cell-type labels for the same modality.
    code : str
        Dataset code, which fixes ``n_rare``.
    exclude : sequence of str, optional
        Labels dropped from the cell set and the candidate list.

    Returns
    -------
    dict
        ``silhouette`` (the plotted value: nanmean over the rare types),
        ``rare_types`` / ``rare_counts`` / ``per_type_silhouette`` (the
        components), ``n_rare_types``, ``n_rare_cells``, ``n_retained_cells``
        and ``rare_pct`` (the panel annotation).
    """
    labels = np.asarray(labels)
    n_rare = n_rare_for(code)
    entries = rare_entries(labels, n_rare, exclude)
    retained = int((~np.isin(labels, list(exclude))).sum())
    per_type = [group_silhouette(latent, labels, ct, exclude) for ct, _ in entries]
    n_rare_cells = int(sum(c for _, c in entries))
    with np.errstate(invalid="ignore"):
        value = (float(np.nanmean(per_type))
                 if per_type and not np.all(np.isnan(per_type)) else float("nan"))
    return {
        "silhouette": value,
        "rare_types": "|".join(ct for ct, _ in entries),
        "rare_counts": "|".join(str(c) for _, c in entries),
        "per_type_silhouette": "|".join(f"{v:.6f}" for v in per_type),
        "n_rare_types": len(entries),
        "n_rare_cells": n_rare_cells,
        "n_retained_cells": retained,
        "rare_pct": 100.0 * n_rare_cells / retained if retained else float("nan"),
    }


def load_labels(code: str, contract: Contract = "harmonised"
                ) -> dict[str, np.ndarray]:
    """Load per-modality cell-type labels under one contract.

    Both contracts read the **reference** cell-type column, because that column
    is what defines the paper's rare classes.  They differ only in row order:
    ``reference`` reorders cells into per-timepoint blocks to match the authors'
    published latents; ``harmonised`` keeps CSV order to match this project's.

    Parameters
    ----------
    code : str
        Dataset code.
    contract : {'harmonised', 'reference'}, optional
        Row-order contract.

    Returns
    -------
    dict
        ``{"rna": labels, "atac": labels}``, lower-cased and remapped.
    """
    if contract == "reference":
        data = snr_data.load_dataset(code, mode="reference")
        return {m: remap_unknown(data.modality(m).cell_type, code)
                for m in ("rna", "atac")}

    data = snr_data.load_dataset(code, mode="harmonised")
    out: dict[str, np.ndarray] = {}
    for modality in ("rna", "atac"):
        mod = data.modality(modality)
        col = snr_data.REFERENCE_TYPE_COL[code.upper()][modality]
        if col not in mod.meta.columns:
            raise KeyError(
                f"{code}/{modality}: reference cell-type column {col!r} absent; "
                f"available: {list(mod.meta.columns)[:20]}")
        raw = np.asarray([str(x).lower() for x in mod.meta[col].to_numpy()])
        out[modality] = remap_unknown(raw, code)
    return out


def _iter_latents(runs_dir: Path, code: str) -> Iterable[tuple[str, int, Path]]:
    """Yield ``(method, seed, path)`` for every ``.npz`` latent of one dataset.

    Two on-disk layouts are supported, because the benchmark and the fusion-lab
    replication package write different ones:

    ``<runs_dir>/<CODE>/<method>_s<seed>.npz``
        The benchmark's per-dataset folders.
    ``<runs_dir>/<CODE>__<arm>_s<seed>.npz``
        The replication package's flat, dataset-prefixed files.

    Parameters
    ----------
    runs_dir : Path
        Directory holding either layout.
    code : str
        Dataset code.

    Yields
    ------
    tuple
        Method name, seed, and path.
    """
    folder = runs_dir / code
    if folder.is_dir():
        candidates = [(p, p.stem) for p in sorted(folder.glob("*.npz"))]
    else:
        candidates = [(p, p.stem.split("__", 1)[1])
                      for p in sorted(runs_dir.glob(f"{code}__*.npz"))]
        if not candidates:
            log.warning("%s: no latents under %s in either layout", code, runs_dir)
            return
    for path, stem in candidates:
        method, _, seed_tag = stem.rpartition("_s")
        if not method or not seed_tag:
            log.warning("%s: cannot parse %s, skipping", code, path.name)
            continue

        seed_digits = "".join(ch for ch in seed_tag.split("_")[0] if ch.isdigit())
        if not seed_digits:
            log.warning("%s: cannot parse seed from %s, skipping", code, path.name)
            continue
        yield method, int(seed_digits), path


def score_runs(runs_dir: Path, codes: Sequence[str],
               contract: Contract = "harmonised") -> pd.DataFrame:
    """Score every local latent for the requested datasets.

    Parameters
    ----------
    runs_dir : Path
        Root holding ``<CODE>/<method>_s<seed>.npz``.
    codes : sequence of str
        Dataset codes to score.
    contract : {'harmonised', 'reference'}, optional
        Row-order contract the latents were produced under.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, method, seed, modality).
    """
    rows: list[dict[str, Any]] = []
    for code in codes:
        labels = load_labels(code, contract)
        sizes = {m: len(v) for m, v in labels.items()}
        log.info("[%s] labels loaded (%s): rna=%d atac=%d, n_rare=%d",
                 code, contract, sizes["rna"], sizes["atac"], n_rare_for(code))
        for method, seed, path in _iter_latents(runs_dir, code):
            with np.load(path) as handle:
                latents = {m: np.asarray(handle[m]) for m in ("rna", "atac")
                           if m in handle}
            for modality, latent in latents.items():
                if latent.shape[0] != sizes[modality]:
                    log.warning(
                        "[%s] %s s%d %s: %d latent rows vs %d labels -- skipped "
                        "(subsampled run with no stored index)",
                        code, method, seed, modality.upper(),
                        latent.shape[0], sizes[modality])
                    continue
                res = rare_population(latent, labels[modality], code)
                rows.append({"dataset": code, "method": method, "seed": seed,
                             "modality": modality, "contract": contract, **res})
                log.info("[%s] %-12s s%d %-4s  sil=%7.4f  rare=%.2f%% (%s)",
                         code, method, seed, modality.upper(), res["silhouette"],
                         res["rare_pct"], res["rare_types"])
    return pd.DataFrame(rows)


def score_authors(codes: Sequence[str]) -> pd.DataFrame:
    """Score the authors' published latents, for validation against Fig. 3C/D.

    Parameters
    ----------
    codes : sequence of str
        Dataset codes to score.

    Returns
    -------
    pandas.DataFrame
        One row per (dataset, method, modality), scored in ``reference``
        contract -- the contract under which the published latents were stored.
    """
    rows: list[dict[str, Any]] = []
    methods = ("scMultiNODE",) + cfg.BASELINE_METHODS
    for code in codes:
        labels = load_labels(code, "reference")
        sizes = {m: len(v) for m, v in labels.items()}
        for method in methods:
            try:
                latent = snr_data.load_authors_latent(code, method)
            except (FileNotFoundError, KeyError) as exc:
                log.warning("[%s] %s: no published latent (%s)", code, method, exc)
                continue
            for modality in ("rna", "atac"):
                arr = np.asarray(latent[modality])
                if arr.shape[0] != sizes[modality]:
                    log.warning("[%s] %s %s: %d rows vs %d labels -- skipped",
                                code, method, modality.upper(),
                                arr.shape[0], sizes[modality])
                    continue
                res = rare_population(arr, labels[modality], code)
                rows.append({"dataset": code, "method": method, "seed": -1,
                             "modality": modality, "contract": "reference",
                             "source": "authors", **res})
                log.info("[%s authors] %-12s %-4s  sil=%7.4f  rare=%.2f%% (%s)",
                         code, method, modality.upper(), res["silhouette"],
                         res["rare_pct"], res["rare_types"])
    return pd.DataFrame(rows)


def main() -> None:
    """Score rare populations and write the long-form table."""
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=None,
                        help="root holding <CODE>/<method>_s<seed>.npz")
    parser.add_argument("--datasets", nargs="*", default=list(cfg.DATASETS))
    parser.add_argument("--contract", default="harmonised",
                        choices=["harmonised", "reference"])
    parser.add_argument("--authors", action="store_true",
                        help="also score the authors' published latents")
    parser.add_argument("--out", default=None, help="output .csv path")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    if args.runs_dir:
        frame = score_runs(Path(args.runs_dir), args.datasets, args.contract)
        frame["source"] = "runs"
        frames.append(frame)
    if args.authors:
        frames.append(score_authors(args.datasets))
    if not frames:
        parser.error("nothing to do: pass --runs-dir and/or --authors")

    table = pd.concat(frames, ignore_index=True)
    out = Path(args.out) if args.out else cfg.TABLES_DIR / "rare_population_v3.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    log.info("wrote %d rows to %s", len(table), out)


if __name__ == "__main__":
    main()
