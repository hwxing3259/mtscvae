"""PFC integration metrics, VENDORED verbatim from the published benchmark.

Provenance
----------
Copied from the published benchmark harness --
``score_bidirectional.py`` (metrics + Tier-C subsample), ``score_rep.py``
(identical copies), and ``herring_cmp/a6_heldout_xmodal_centroid.py`` (centroid
cosine).  Vendored rather than imported so this package is self-contained: the
benchmark harness lives only in that split-off project, is untracked in git, and
its embeddings are gitignored.

**Do not "clean up" the bodies below.**  They reproduce a published table; a
subtle rewrite silently produces numbers that are no longer comparable to it.
The recipes here were verified by reproducing the published Tier-C cells exactly
(ours ATAC ARI 0.4198 +/- 0.0234 vs published 0.420 +/- 0.021; teacher
0.4079 +/- 0.0236 vs published 0.408 +/- 0.021).

Five traps this file deliberately preserves
-------------------------------------------
1. **Four ARI recipes coexist upstream.**  The published Tier-C band uses
   ``ari_seed`` -- a *single* k-means at ``random_state=seed`` where the same
   seed also redraws the 20k subsample.  ``score_bidirectional``'s own
   ``random_state=0`` single draw gives different numbers (0.491 vs the
   published 0.504 on RNA), and other tables in the same paper use
   ``n_init=4``.  Use :func:`ari_seed` + :func:`sub20k` together for anything
   compared against the published band.
2. **The coarse maps are asymmetric between modalities.**  ``PV_SCUBE3`` is in
   both vocabularies: RNA sends it to ``Other`` (not a dict key), ATAC sends it
   to ``PV`` (matches the ``startswith("PV")`` rule).  Preserved as-is.
3. **Chance is ``sum(p^2)``** = 0.115 (RNA) / 0.114 (ATAC), not ``1/13`` = 0.077.
   Both numbers appear in the paper, for different tables.
4. **The published centroid cosine is RAW (uncentered)** and its own
   label-shuffled null scores 0.96-0.999 -- *above every real cell type*.  It is
   reproduced here for continuity, and :func:`centroid_cosine` also returns the
   centred variant, which is the one that carries signal.
5. **ATAC ``ID2`` has 30 cells atlas-wide**, 5-11 in a typical 20k draw.  Any
   change to the draw can drop it and silently change the reported means.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

__all__ = ["RNA_TYPE_MAP", "TYPE_ORDER", "coarsen_atac", "coarsen_rna",
           "sub20k", "balanced_subsample", "cluster_score", "ari_seed",
           "knn_purity", "xmod", "ceiling", "centroid_cosine", "chance_level",
           "eff_rank"]

STAGES = ("Gestation", "Neonatal", "Infancy", "Childhood", "Adolescence", "Adulthood")

# _lib.py:77-79 -- lineage display order; also the row order of the published tables.
TYPE_ORDER = ["PN_dev", "MGE_dev", "L2-3", "L4", "L5-6", "SST", "VIP", "PV",
              "ID2", "OPC", "Oligo", "Astro", "Micro", "Other"]

# _lib.py:69-75 -- exact dict lookup on RNA `major_clust`; anything absent -> Other.
RNA_TYPE_MAP = {
    "Oligo": "Oligo", "OPC": "OPC", "Astro": "Astro", "Micro": "Micro",
    "L2-3_CUX2": "L2-3", "L4_RORB": "L4", "L5-6_TLE4": "L5-6",
    "L5-6_THEMIS": "L5-6", "SST": "SST", "VIP": "VIP", "PV": "PV",
    "ID2": "ID2", "PN_dev": "PN_dev", "MGE_dev": "MGE_dev",
}

# _lib.py:234-247 -- ORDERED prefix rules; order is load-bearing (PV before SST,
# so "PV_SST" -> PV; L2/L3 before L4 etc).
_ATAC_RULES = [("Astro", "Astro"), ("Oligo", "Oligo"), ("OPC", "OPC"),
               ("Micro", "Micro"), ("MGE", "MGE_dev"), ("PN_dev", "PN_dev"),
               ("PN dev", "PN_dev"), ("L2", "L2-3"), ("L3", "L2-3"),
               ("L4", "L4"), ("L5", "L5-6"), ("L6", "L5-6"), ("SST", "SST"),
               ("VIP", "VIP"), ("PV", "PV"), ("ID2", "ID2")]


def coarsen_atac(x) -> str:
    """Map an ATAC ``predicted_group`` (77 fine labels) to a coarse type.

    Parameters
    ----------
    x : str
        Fine label, e.g. ``"Astro_SLC1A2"``.

    Returns
    -------
    str
        One of the 13 coarse types, or ``"Other"``.
    """
    x = str(x)
    for pre, c in _ATAC_RULES:
        if x.startswith(pre):
            return c
    return "Other"


def coarsen_rna(x) -> str:
    """Map an RNA ``major_clust`` (19 labels) to a coarse type.

    Parameters
    ----------
    x : str
        Cluster label.

    Returns
    -------
    str
        One of the 13 coarse types, or ``"Other"``.
    """
    return RNA_TYPE_MAP.get(str(x), "Other")


def sub20k(lab: np.ndarray, seed: int = 0, n: int = 20000) -> np.ndarray:
    """Tier-C evaluation draw: ``n`` non-``Other`` cells, uniform.

    Parameters
    ----------
    lab : numpy.ndarray
        Coarse labels, one per cell.
    seed : int, optional
        RNG seed.  In the published band the *same* seed also selects the
        k-means ``random_state``, so subsample and init noise are entangled.
    n : int, optional
        Target size.

    Returns
    -------
    numpy.ndarray
        Sorted indices into the full cell array.
    """
    keep = np.where(lab != "Other")[0]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(keep, size=min(n, len(keep)), replace=False))


def balanced_subsample(stage_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Tier-A/B draw: stage-balanced, ``n // 6`` per developmental stage.

    Balanced by **stage**, not cell type, and it draws from all cells including
    ``Other`` (dropped later inside each metric), so effective N is ~4 % smaller.

    Parameters
    ----------
    stage_idx : numpy.ndarray
        Integer stage index per cell, 0..5.
    n : int
        Target size.
    seed : int
        RNG seed.

    Returns
    -------
    numpy.ndarray
        Sorted indices.
    """
    rng = np.random.default_rng(seed)
    per = n // 6
    picks = []
    for s in range(6):
        pool = np.where(stage_idx == s)[0]
        picks.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return np.sort(np.concatenate(picks))


def cluster_score(e: np.ndarray, lab: np.ndarray, k: int = 20,
                  ninit: int = 10, random_state: int = 0) -> dict:
    """k-means agreement with the coarse labels, as **both** ARI and NMI.

    ``Other`` cells are dropped before clustering, matching the published code.

    Parameters
    ----------
    e : numpy.ndarray
        Embedding, ``(n_cells, d)``.
    lab : numpy.ndarray
        Coarse labels.
    k, ninit, random_state : int, optional
        k-means settings.  The published Tier-C recipe is ``k=20, ninit=10``
        with ``random_state`` equal to the subsample seed.

    Returns
    -------
    dict
        ``{"ari": float, "nmi": float}``.
    """
    keep = lab != "Other"
    p = KMeans(n_clusters=k, n_init=ninit,
               random_state=random_state).fit_predict(e[keep].astype(np.float32))
    return {"ari": float(adjusted_rand_score(lab[keep], p)),
            "nmi": float(normalized_mutual_info_score(lab[keep], p))}


def ari_seed(e: np.ndarray, lab: np.ndarray, seed: int, k: int = 20) -> float:
    """The published Tier-C ARI: one k-means at ``random_state=seed``.

    Parameters
    ----------
    e : numpy.ndarray
        Embedding.
    lab : numpy.ndarray
        Coarse labels.
    seed : int
        Both the k-means ``random_state`` and (upstream) the subsample seed.
    k : int, optional
        Cluster count.

    Returns
    -------
    float
        Adjusted Rand index.
    """
    return cluster_score(e, lab, k=k, ninit=10, random_state=seed)["ari"]


def knn_purity(e: np.ndarray, lab: np.ndarray, k: int = 30) -> float:
    """Pair-level fraction of the ``k`` nearest neighbours sharing the query label.

    Note this is a **pair-level mean**, not a per-cell purity averaged over
    cells, and not a majority vote.  Self is excluded by dropping column 0,
    which is safe here (verified: the query is always returned first, no
    duplicate rows).

    Parameters
    ----------
    e : numpy.ndarray
        Embedding.
    lab : numpy.ndarray
        Coarse labels.
    k : int, optional
        Neighbour count.

    Returns
    -------
    float
        Purity in [0, 1].
    """
    keep = lab != "Other"
    e2, l2 = e[keep].astype(np.float32), lab[keep]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(e2)
    _, ix = nn.kneighbors(e2)
    return float((l2[ix[:, 1:k + 1]] == l2[:, None]).mean())


def xmod(e_src: np.ndarray, l_src: np.ndarray, e_tgt: np.ndarray,
         l_tgt: np.ndarray, k: int = 15) -> float:
    """Cross-modal transfer: kNN fit on the source modality, read out on the target.

    Abundance-weighted per-type recall over the types present in both.

    Parameters
    ----------
    e_src, l_src : numpy.ndarray
        Source embedding and its own labels.
    e_tgt, l_tgt : numpy.ndarray
        Target embedding and its own labels.
    k : int, optional
        Neighbours.

    Returns
    -------
    float
        Weighted recall.
    """
    common = [t for t in np.unique(l_tgt) if t != "Other" and t in set(l_src)]
    sk, tk = np.isin(l_src, common), np.isin(l_tgt, common)
    clf = KNeighborsClassifier(n_neighbors=k).fit(e_src[sk].astype(np.float32), l_src[sk])
    pred = clf.predict(e_tgt[tk].astype(np.float32))
    truth = l_tgt[tk]
    rec, w = [], []
    for t in common:
        m = truth == t
        if m.sum():
            rec.append((pred[m] == t).mean())
            w.append(m.sum())
    return float(np.average(rec, weights=w))


def ceiling(e: np.ndarray, lab: np.ndarray, k: int = 15, seed: int = 0) -> float:
    """Within-modality ceiling for :func:`xmod`: 50/50 split, same estimator.

    Parameters
    ----------
    e : numpy.ndarray
        Embedding.
    lab : numpy.ndarray
        Coarse labels.
    k : int, optional
        Neighbours.
    seed : int, optional
        Split RNG.

    Returns
    -------
    float
        Weighted recall with no modality gap.
    """
    keep = lab != "Other"
    e2, l2 = e[keep].astype(np.float32), lab[keep]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(l2))
    h = len(l2) // 2
    tr, te = perm[:h], perm[h:]
    clf = KNeighborsClassifier(n_neighbors=k).fit(e2[tr], l2[tr])
    pred = clf.predict(e2[te])
    truth = l2[te]
    rec, w = [], []
    for t in np.unique(truth):
        m = truth == t
        if m.sum():
            rec.append((pred[m] == t).mean())
            w.append(m.sum())
    return float(np.average(rec, weights=w))


def chance_level(lab: np.ndarray) -> float:
    """Chance for the weighted-recall transfer metric: ``sum(p^2)``.

    0.115 (RNA) / 0.114 (ATAC) on the full atlas -- **not** ``1/13 = 0.077``,
    which the paper quotes for a different table.

    Parameters
    ----------
    lab : numpy.ndarray
        Coarse labels.

    Returns
    -------
    float
        Expected weighted recall under random assignment.
    """
    l = lab[lab != "Other"]
    _, c = np.unique(l, return_counts=True)
    p = c / c.sum()
    return float((p ** 2).sum())


def centroid_cosine(z_rna: np.ndarray, lab_rna: np.ndarray,
                    z_atac: np.ndarray, lab_atac: np.ndarray,
                    min_cells: int = 10) -> dict:
    """Per-type cosine between the RNA and ATAC centroids in the fused latent.

    Returns **both** the published raw (uncentered) statistic and a centred one.

    Parameters
    ----------
    z_rna, z_atac : numpy.ndarray
        Fused latents per modality.
    lab_rna, lab_atac : numpy.ndarray
        Coarse labels per modality.
    min_cells : int, optional
        Types with fewer cells on either side are returned as NaN, matching
        ``a6_heldout_xmodal_centroid.MIN_CELLS``.

    Returns
    -------
    dict
        ``{type: {"raw": float, "centered": float, "n_rna": int, "n_atac": int}}``
        in :data:`TYPE_ORDER`, plus a ``"__mean__"`` entry.

    Notes
    -----
    The **raw** variant is what the paper prints, and its own label-shuffled
    null scores 0.96-0.999 -- above every real cell type -- so it should not be
    read as evidence of alignment on its own.  The centred variant subtracts the
    global mean of each modality's latent before taking the cosine, which
    removes the shared offset that drives the null.
    """
    mu_r, mu_a = z_rna.mean(0), z_atac.mean(0)

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    out: dict = {}
    raws, cens = [], []
    for t in TYPE_ORDER:
        if t == "Other":
            continue
        mr, ma = lab_rna == t, lab_atac == t
        n_r, n_a = int(mr.sum()), int(ma.sum())
        if n_r < min_cells or n_a < min_cells:
            out[t] = {"raw": float("nan"), "centered": float("nan"),
                      "n_rna": n_r, "n_atac": n_a}
            continue
        cr, ca = z_rna[mr].mean(0), z_atac[ma].mean(0)
        raw = _cos(cr, ca)
        cen = _cos(cr - mu_r, ca - mu_a)
        out[t] = {"raw": raw, "centered": cen, "n_rna": n_r, "n_atac": n_a}
        raws.append(raw)
        cens.append(cen)
    out["__mean__"] = {"raw": float(np.mean(raws)) if raws else float("nan"),
                       "centered": float(np.mean(cens)) if cens else float("nan"),
                       "n_rna": -1, "n_atac": -1}
    return out


# `eff_rank` was one of eight byte-equivalent copies; consolidated into
# suite/rank.py, which is PURE NUMPY so this module stays torch-free.
#
# RE-EXPORTED, not merely moved: results/four_datasets/scripts/stage_and_gates.sh
# does `from ...pfc.bench.metrics import eff_rank  # canonical; never reimplement`,
# so this name is a live external contract and must keep resolving here.
from neuro_dev_model.mtvae_pipeline.suite.rank import eff_rank  # noqa: E402,F401
