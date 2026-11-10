"""Score one PFC embedding on the evaluation draw ``E``."""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np

from neuro_dev_model.pfc_scaling_bench import design as D
from neuro_dev_model.pfc_scaling_bench import pfc_data as P
from neuro_dev_model.pfc_scaling_bench import records as R

log = logging.getLogger("score")

#: The seven endpoints of the four-dataset figure, in its order.
ENDPOINTS: tuple[str, ...] = ("time_corr", "nmi", "batch_entropy", "lta_time",
                              "lta_type", "rna_rare_sil", "atac_rare_sil")


def fourds_panel(rna: np.ndarray, atac: np.ndarray, seed: int) -> dict[str, Any]:
    """The four-dataset scorer and rare-type rule on ``E``.

    Parameters
    ----------
    rna, atac : numpy.ndarray
        ``E`` rows of the embedding.
    seed : int
        The fit's seed, used for the scorer's stochastic steps.

    Returns
    -------
    dict
        The scorer's full flat record (endpoints, nulls, diagnostics) plus the
        rare-type endpoints and their components.
    """
    from neuro_dev_model.mtvae.data import snr_rare_population as rare
    from neuro_dev_model.mtvae_4ds_coadapt.data import snr_score

    data = P.scorer_dataset()
    out = snr_score.score_latent(D.DATASET_CODE, rna, atac, data, seed=seed)
    for m, lat in (("rna", rna), ("atac", atac)):
        lab = rare.remap_unknown(np.asarray([str(s).lower() for s in
                                             data.modality(m).cell_type]),
                                 D.DATASET_CODE)
        rp = rare.rare_population(lat, lab, D.DATASET_CODE)
        out[f"{m}_rare_sil"] = rp["silhouette"]
        for k in ("rare_types", "rare_counts", "per_type_silhouette", "rare_pct"):
            out[f"{m}_rare_{k}"] = rp[k]
    return out


def pfc_panel(rna: np.ndarray, atac: np.ndarray) -> dict[str, float]:
    """The PFC example's integration panel on ``E``.

    Parameters
    ----------
    rna, atac : numpy.ndarray
        ``E`` rows of the embedding.

    Returns
    -------
    dict
        ``pfc_{rna,atac}_{ari,nmi_kmeans,knn30}``, ``pfc_a2r``, ``pfc_r2a`` and
        ``pfc_overlap`` (their mean, exact because ``E`` takes equal counts).
    """
    from neuro_dev_model.mtvae_pipeline.pfc.bench.metrics import (
        cluster_score, knn_purity, xmod)

    lab = {m: P.labels(m)["coarse"][P.eval_rows(m)] for m in P.MODALITIES}
    out: dict[str, float] = {}
    for m, z in (("rna", rna), ("atac", atac)):
        cs = cluster_score(z, lab[m], k=20, ninit=10, random_state=0)
        out[f"pfc_{m}_ari"] = cs["ari"]
        out[f"pfc_{m}_nmi_kmeans"] = cs["nmi"]
        out[f"pfc_{m}_knn30"] = knn_purity(z, lab[m], k=30)
    out["pfc_a2r"] = xmod(atac, lab["atac"], rna, lab["rna"], k=15)
    out["pfc_r2a"] = xmod(rna, lab["rna"], atac, lab["atac"], k=15)
    out["pfc_overlap"] = 0.5 * (out["pfc_a2r"] + out["pfc_r2a"])
    return out


def score(tag: str) -> dict[str, Any]:
    """Score one embedding and write its score record.

    Parameters
    ----------
    tag : str
        The fit's canonical stem.

    Returns
    -------
    dict
        The written record.

    Raises
    ------
    RuntimeError
        If the fit did not succeed or its embedding is too short to hold ``E``.
    """
    fit = R.read(tag)
    if not fit or fit.get("status") != "ok":
        raise RuntimeError(f"{tag}: no successful fit record")
    z = np.load(R.emb_path(tag))
    if len(z["rna"]) < D.N_EVAL or len(z["atac"]) < D.N_EVAL:
        raise RuntimeError(f"{tag}: embedding shorter than E")
    rna, atac = z["rna"][:D.N_EVAL], z["atac"][:D.N_EVAL]
    t0 = time.time()
    rec: dict[str, Any] = {"tag": tag, "method": fit["method"], "tier": fit["tier"],
                           "n_tier": fit["n_tier"], "seed": fit["seed"],
                           "n_eval": D.N_EVAL}
    rec.update(fourds_panel(rna, atac, seed=int(fit["seed"])))
    rec.update(pfc_panel(rna, atac))
    rec["score_seconds"] = time.time() - t0
    rec.update({k: v for k, v in R.runtime_facts().items()
                if k in ("host", "threads", "numpy", "code_version")})
    R.write(tag, rec, kind="score")
    log.info("%s: %s", tag, {k: round(float(rec[k]), 4) for k in ENDPOINTS
                             if rec.get(k) is not None})
    return rec


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if R.read(args.tag, "score") and not args.force:
        log.info("%s already scored", args.tag)
        return
    score(args.tag)


if __name__ == "__main__":
    main()
