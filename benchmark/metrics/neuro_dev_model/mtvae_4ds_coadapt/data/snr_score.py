"""Score one saved latent with the paper's metric panel *and* the nulls it omits."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from neuro_dev_model.mtvae_4ds_coadapt.data import snr_config as cfg
from neuro_dev_model.mtvae_4ds_coadapt.data import snr_data
from neuro_dev_model.mtvae_4ds_coadapt.data import snr_metrics as M

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snr_score")


def _select_common_types(rna_types: np.ndarray, atac_types: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Restrict both modalities to their shared, non-``unknown`` cell types.

    This is what ``_evaluateUnalignedAlignment`` does for the unaligned datasets
    and it is required for a label transfer between disjoint vocabularies to be
    computable at all.

    Parameters
    ----------
    rna_types, atac_types : numpy.ndarray
        Cell-type labels per modality.

    Returns
    -------
    tuple of numpy.ndarray
        Indices kept on each side.
    """
    common = set(np.unique(rna_types)) & set(np.unique(atac_types)) - {"unknown"}
    i1 = np.where(np.isin(rna_types, list(common)))[0]
    i2 = np.where(np.isin(atac_types, list(common)))[0]
    return i1, i2


def score_latent(code: str, rna_lat: np.ndarray, atac_lat: np.ndarray,
                 data: snr_data.DatasetData, seed: int = 0,
                 skip_slow: bool = False) -> dict[str, Any]:
    """Compute the full metric panel plus every null for one latent.

    Parameters
    ----------
    code : str
        Dataset code.
    rna_lat, atac_lat : numpy.ndarray
        Integrated embeddings.
    data : snr_data.DatasetData
        Loaded dataset supplying labels, timepoints and batches.
    seed : int, optional
        Seed for stochastic metrics (batch entropy, permutation nulls, Louvain).
    skip_slow : bool, optional
        Skip the distance-correlation time metric, which costs ~33 GB and
        several minutes at n = 10,000.

    Returns
    -------
    dict
        Flat record of metrics, nulls and diagnostics.
    """
    spec = cfg.dataset(code)
    rna, atac = data.rna, data.atac
    out: dict[str, Any] = {"dataset": code, "n_rna": len(rna_lat),
                           "n_atac": len(atac_lat)}

    # ---- label transfer, with its ceiling, majority baseline and permutation null
    i1, i2 = _select_common_types(rna.cell_type, atac.cell_type)
    out["n_rna_type_selected"] = int(len(i1))
    out["n_atac_type_selected"] = int(len(i2))
    if len(i1) and len(i2):
        out["lta_type"] = M.transfer_accuracy(rna_lat[i1], atac_lat[i2],
                                              rna.cell_type[i1], atac.cell_type[i2],
                                              n=cfg.LTA_N_NEIGHBORS)
        base = M.lta_baselines(rna.cell_type[i1], atac.cell_type[i2], seed=seed)
    else:
        out["lta_type"] = float("nan")
        base = M.lta_baselines(rna.cell_type, atac.cell_type, seed=seed)
    out["lta_type_majority"] = base["majority"]
    out["lta_type_ceiling"] = base["map_ceiling"]
    out["lta_type_perm_mean"] = base["permutation_mean"]
    out["lta_type_perm_sd"] = base["permutation_sd"]
    out["lta_type_over_majority"] = out["lta_type"] - base["majority"]

    out["lta_time"] = M.transfer_accuracy(rna_lat, atac_lat, rna.tp, atac.tp,
                                          n=cfg.LTA_N_NEIGHBORS)
    tbase = M.lta_baselines(rna.tp.astype(str), atac.tp.astype(str), seed=seed)
    out["lta_time_majority"] = tbase["majority"]
    out["lta_time_perm_mean"] = tbase["permutation_mean"]
    out["lta_time_over_majority"] = out["lta_time"] - tbase["majority"]

    # ---- modality mixing, against the unfused control
    both = np.concatenate([rna_lat, atac_lat], axis=0)
    modality = np.concatenate([np.zeros(len(rna_lat)), np.ones(len(atac_lat))])
    out["batch_entropy"] = M.batch_entropy(both, modality, seed=seed)

    # ---- correspondence, co-assay only, with nulls
    if spec.coassay and len(rna_lat) == len(atac_lat):
        out["foscttm"] = M.foscttm(rna_lat, atac_lat)
        out["neigh_overlap"] = M.neighbor_overlap(rna_lat, atac_lat,
                                                  n_neighbors=cfg.NEIGH_OVERLAP_K)
        out["scc"] = M.feature_corr_per_sample(rna_lat, atac_lat)
        nulls = M.foscttm_nulls(rna_lat, atac_lat, groups=rna.tp, seed=seed)
        out["foscttm_shuffled_null"] = nulls["shuffled"]
        out["foscttm_blocked_null"] = nulls.get("blocked")
        out["foscttm_blocked_ratio"] = nulls.get("blocked_ratio")

    # ---- clustering against cell type AND against batch, with a shuffled null
    clu = M.nmi_louvain(rna_lat, atac_lat, rna.cell_type, atac.cell_type, seed=seed)
    out.update({k: v for k, v in clu.items()})
    for name, lat, mod in (("rna", rna_lat, rna), ("atac", atac_lat, atac)):
        if mod.batch is None:
            continue
        bt = M.batch_vs_type_scores(lat, mod.cell_type, mod.batch, seed=seed)
        for k, v in bt.items():
            out[f"{name}_{k}"] = v
        out[f"{name}_knn30_type"] = M.knn_label_preservation(lat, mod.cell_type, k=30)
        out[f"{name}_knn30_batch"] = M.knn_label_preservation(lat, mod.batch, k=30)
        rare = M.rarest_type_silhouette(lat, mod.cell_type)
        out[f"{name}_rarest_type"] = rare["cell_type"]
        out[f"{name}_rarest_fraction"] = rare["fraction"]
        out[f"{name}_rarest_silhouette"] = rare["silhouette"]

    # ---- dynamics, with a shuffled-time null
    if not skip_slow:
        out["time_corr_rna"] = M.label_corr(rna_lat, rna.tp)
        out["time_corr_atac"] = M.label_corr(atac_lat, atac.tp)
        out["time_corr"] = 0.5 * (out["time_corr_rna"] + out["time_corr_atac"])
        rng = np.random.default_rng(seed)
        out["time_corr_shuffled_null"] = M.label_corr(rna_lat, rng.permutation(rna.tp))
    return out


def main() -> None:
    """Score one saved run and write its metric record."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(cfg.DATASETS))
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="harmonised", choices=["harmonised", "reference"])
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--source", default="runs", choices=["runs", "authors"],
                    help="'runs' scores our own re-run latents; 'authors' scores the "
                         "latents the paper's authors published, which our metric "
                         "implementation reproduces to 0.00000 for 6 of 7 methods "
                         "on both co-assay datasets")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--subsample", type=int, default=None,
                    help="Tier B runs were fitted on a subsample; pass the same "
                         "value so the labels scored match the cells fitted")
    ap.add_argument("--skip-slow", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else (cfg.RUNS_ROOT / args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else (cfg.RESULTS_ROOT / "scores" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.method}_s{args.seed}"
    if args.subsample:
        tag += f"_n{args.subsample}"
    npz_path = runs_dir / f"{tag}.npz"
    out_path = out_dir / f"{tag}.json"
    if args.source == "runs" and not npz_path.exists():
        log.warning("no latent at %s -- nothing to score", npz_path)
        return
    if out_path.exists() and not args.force:
        log.info("%s exists -- skipping", out_path)
        return

    if args.source == "authors":
        lat = snr_data.load_authors_latent(args.dataset, args.method)
        z = {"rna": lat["rna"], "atac": lat["atac"]}
    else:
        z = np.load(npz_path)
    # The authors' latents are stored in their scoring script's cell order
    # (timepoint blocks, labels lowercased), so they must be scored against
    # labels loaded under the same contract.
    mode = "reference" if args.source == "authors" else args.mode
    data = snr_data.load_dataset(args.dataset, mode=mode)
    if args.subsample:
        # Must reproduce the exact cells the Tier B fit used, or labels and
        # latent rows disagree silently.
        data = snr_data.subsample(data, args.subsample, seed=args.seed)
    t0 = time.time()
    rec = score_latent(args.dataset, z["rna"].astype(np.float64),
                       z["atac"].astype(np.float64), data,
                       seed=args.seed, skip_slow=args.skip_slow)
    # Record the mode that was USED, not the one requested: line ~206 forces
    # "reference" for authors' latents (their cell order / label contract), so
    # recording args.mode stamped every authors-sourced record "harmonised" --
    # hiding the very fact that makes those numbers non-comparable with ours.
    rec.update(method=args.method, seed=args.seed, mode=mode,
               mode_requested=args.mode,
               source=args.source,
               subsample=args.subsample, tier="B" if args.subsample else "C",
               score_seconds=time.time() - t0)
    # Timing/device provenance belongs to OUR fit and exists only for our own
    # latents. The authors publish latents but no timings, so attaching this
    # metadata to a --source=authors record would silently report our wall-clock
    # as theirs -- which it did, filling their tab:nd-benchmark Time column with
    # our numbers until this guard was added.
    run_meta_path = runs_dir / f"{tag}.json"
    if args.source == "runs" and run_meta_path.exists():
        with open(run_meta_path) as fh:
            meta = json.load(fh)
        rec["run_seconds"] = meta.get("seconds")
        rec["device"] = meta.get("device")
        rec["determinism"] = meta.get("determinism")
    elif args.source == "authors":
        rec["run_seconds"] = None
        rec["device"] = "n/a (authors' published latent; no training run here)"
        rec["determinism"] = "n/a (not refitted)"

    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    log.info("wrote %s (%.1fs)", out_path, rec["score_seconds"])


if __name__ == "__main__":
    main()
