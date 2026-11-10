"""Evaluation metrics for the scMultiNODE replication benchmark.
Throughout, ``x1``/``domain1`` is RNA and ``x2``/``domain2`` is ATAC."""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import scipy.sparse
import scipy.spatial.distance
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

log = logging.getLogger("snr_metrics")

#: Row-block size used when materialising cross-distance matrices, chosen so a
#: block stays well under 1 GB in float64 for n up to a few 10^4.
_BLOCK = 2048


# ===========================================================================
# Faithful ports of scMultiNODE_ref/optim/evaluation.py
# ===========================================================================
def transfer_accuracy(domain1: np.ndarray, domain2: np.ndarray,
                      type1: np.ndarray, type2: np.ndarray,
                      n: int = 50) -> float:
    """Label Transfer Accuracy (UnionCom's metric), as the reference computes it.

    The classifier is fitted on ``domain2`` with labels ``type2`` and used to
    predict labels for ``domain1``, which are scored against ``type1``.  With
    the reference's call convention this means **fitting on ATAC and scoring on
    RNA**.

    Parameters
    ----------
    domain1 : numpy.ndarray, shape (n1, d)
        Query embedding (RNA side at the reference call sites).
    domain2 : numpy.ndarray, shape (n2, d)
        Reference embedding the classifier is fitted on (ATAC side).
    type1 : numpy.ndarray, shape (n1,)
        Ground-truth labels for ``domain1``.
    type2 : numpy.ndarray, shape (n2,)
        Labels used to fit the classifier.
    n : int, optional
        Number of neighbours, 50 at every reference call site.

    Returns
    -------
    float
        Accuracy of the transferred labels.

    Raises
    ------
    ValueError
        If either domain is empty, which the reference code does not guard and
        which occurs for drosophila when the hard-coded cell-type list does not
        intersect the ATAC label vocabulary.
    """
    if len(domain1) == 0 or len(domain2) == 0:
        raise ValueError(
            f"empty domain passed to transfer_accuracy: "
            f"n1={len(domain1)}, n2={len(domain2)}; the reference code selects "
            f"cell types by exact string match and does not guard this case")
    knn = KNeighborsClassifier(n_neighbors=min(n, len(domain2)))
    knn.fit(domain2, type2)
    return float(accuracy_score(type1, knn.predict(domain1)))


def feature_corr_per_sample(domain1: np.ndarray, domain2: np.ndarray) -> float:
    """Mean per-cell Pearson correlation between paired embeddings (the paper's SCC).

    Parameters
    ----------
    domain1, domain2 : numpy.ndarray, shape (n, d)
        Row-aligned embeddings of the same cells in two modalities.

    Returns
    -------
    float
        Mean across cells of the correlation between the two embedding rows.
    """
    xm = domain1 - np.mean(domain1, axis=1, keepdims=True)
    ym = domain2 - np.mean(domain2, axis=1, keepdims=True)
    num = np.sum(xm * ym, axis=1)
    den = np.sqrt(np.sum(xm ** 2, axis=1)) * np.sqrt(np.sum(ym ** 2, axis=1))
    return float(np.mean(num / den))


def neighbor_overlap(source_feature: np.ndarray, target_feature: np.ndarray,
                     n_neighbors: int = 50, block: int = _BLOCK) -> float:
    """Fraction of cells whose true partner is among their nearest partners.

    Algebraically identical to the reference ``neighborOverlap``, which for each
    source cell builds a kd-tree over ``[source_i] + target`` and asks whether
    index ``i + 1`` is among the ``n_neighbors`` returned.  Because ``source_i``
    sits at index 0 with distance 0 it always occupies one slot, so the
    condition is exactly "``target_i`` is among the ``n_neighbors - 1`` nearest
    targets to ``source_i``".  That is computed here by ranking one block of the
    cross-distance matrix at a time, which is O(n^2) instead of the reference's
    O(n^2 log n) with n tree builds.  :func:`verify_neighbor_overlap` checks the
    two agree.

    Parameters
    ----------
    source_feature : numpy.ndarray, shape (n, d)
        Source embedding.
    target_feature : numpy.ndarray, shape (n, d)
        Target embedding, row-aligned with ``source_feature``.
    n_neighbors : int, optional
        Neighbourhood size, 50 at the reference call site.
    block : int, optional
        Number of source rows processed per block.

    Returns
    -------
    float
        Fraction of cells whose partner falls inside the neighbourhood.
    """
    n = source_feature.shape[0]
    k = n_neighbors - 1
    hits = np.empty(n, dtype=bool)
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = scipy.spatial.distance.cdist(source_feature[start:stop], target_feature)
        true_d = d[np.arange(stop - start), np.arange(start, stop)]
        # Rank of the true partner among all targets (number strictly closer).
        rank = (d < true_d[:, None]).sum(axis=1)
        hits[start:stop] = rank < k
    return float(hits.mean())


def verify_neighbor_overlap(source_feature: np.ndarray, target_feature: np.ndarray,
                            n_neighbors: int = 50, n_check: int = 200,
                            seed: int = 0) -> dict[str, float]:
    """Check :func:`neighbor_overlap` against the reference loop on a subsample.

    Parameters
    ----------
    source_feature, target_feature : numpy.ndarray
        Row-aligned embeddings.
    n_neighbors : int, optional
        Neighbourhood size.
    n_check : int, optional
        Number of cells to subsample for the comparison.
    seed : int, optional
        RNG seed for the subsample.

    Returns
    -------
    dict
        ``fast``, ``reference`` and ``max_abs_diff`` on the subsample.
    """
    rng = np.random.default_rng(seed)
    n = source_feature.shape[0]
    idx = rng.choice(n, size=min(n_check, n), replace=False)
    src, tgt = source_feature[idx], target_feature[idx]

    fast = neighbor_overlap(src, tgt, n_neighbors=n_neighbors)

    ref_hits = []
    for i in range(src.shape[0]):
        tmp = np.concatenate([src[i][np.newaxis, :], tgt], axis=0)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(tmp)
        nbr_idx = nbrs.kneighbors(src[i][np.newaxis, :], n_neighbors,
                                  return_distance=False).squeeze()
        ref_hits.append((i + 1) in nbr_idx)
    ref = float(np.mean(ref_hits))
    return {"fast": fast, "reference": ref, "max_abs_diff": abs(fast - ref)}


def foscttm(x1_mat: np.ndarray, x2_mat: np.ndarray, block: int = _BLOCK) -> float:
    """Fraction Of Samples Closer Than the True Match, as the reference computes it.

    The reference subtracts 2 from the closer-count and divides by ``n - 2``,
    which differs from SCOT's ``calc_domainAveraged_FOSCTTM``.  That convention
    is preserved here so the numbers are comparable with the published table;
    :func:`foscttm_scot` implements SCOT's version for cross-checking.

    Parameters
    ----------
    x1_mat, x2_mat : numpy.ndarray, shape (n, d)
        Row-aligned embeddings of the same cells.
    block : int, optional
        Row-block size for the cross-distance computation.

    Returns
    -------
    float
        Mean fraction of cells closer than the true match. Lower is better;
        0.5 is chance.
    """
    n = x1_mat.shape[0]
    total = 0.0
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = scipy.spatial.distance.cdist(x1_mat[start:stop], x2_mat)
        true_d = d[np.arange(stop - start), np.arange(start, stop)]
        closer = (d < true_d[:, None]).sum(axis=1).astype(float)
        closer = np.maximum(closer - 2, 0.0)
        total += closer.sum()
    return float(total / n / (n - 2))


def foscttm_scot(x1_mat: np.ndarray, x2_mat: np.ndarray,
                 block: int = _BLOCK) -> float:
    """FOSCTTM in SCOT's convention, averaged over both directions.

    Parameters
    ----------
    x1_mat, x2_mat : numpy.ndarray, shape (n, d)
        Row-aligned embeddings.
    block : int, optional
        Row-block size.

    Returns
    -------
    float
        Domain-averaged FOSCTTM without the reference's ``-2`` correction.
    """
    n = x1_mat.shape[0]
    def _one(a: np.ndarray, b: np.ndarray) -> float:
        total = 0.0
        for start in range(0, n, block):
            stop = min(start + block, n)
            d = scipy.spatial.distance.cdist(a[start:stop], b)
            true_d = d[np.arange(stop - start), np.arange(start, stop)]
            total += (d < true_d[:, None]).sum(axis=1).sum()
        return float(total / n / (n - 1))
    return 0.5 * (_one(x1_mat, x2_mat) + _one(x2_mat, x1_mat))


def batch_entropy(data: np.ndarray, batches: np.ndarray, n_neighbors: int = 100,
                  n_pools: int = 100, n_samples_per_pool: int = 100,
                  seed: int | None = 0) -> float:
    """Batch-entropy mixing score, normalised by ``log2(n_batches)``.

    Faithful to ``optim.evaluation.batchEntropy`` except that the reference
    leaves ``numpy.random.choice`` unseeded, making the metric stochastic across
    runs.  A seed is accepted here so the benchmark is reproducible; pass
    ``seed=None`` for the reference's exact (unseeded) behaviour.

    Parameters
    ----------
    data : numpy.ndarray, shape (n, d)
        Concatenated embedding of both modalities.
    batches : numpy.ndarray, shape (n,)
        Batch label per row.  At the reference call sites this is the
        **modality** indicator, not the donor.
    n_neighbors : int, optional
        Neighbourhood size.
    n_pools : int, optional
        Number of resampling iterations.
    n_samples_per_pool : int, optional
        Cells drawn per iteration.
    seed : int or None, optional
        RNG seed; ``None`` reproduces the reference's unseeded behaviour.

    Returns
    -------
    float
        Mixing score in ``[0, 1]``; higher means better mixed.

    Raises
    ------
    ValueError
        If fewer than two batches are present.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
    batches = np.asarray(batches)
    batches_ = np.unique(batches)
    n_batches = len(batches_)
    if n_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8).fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    prior = np.array([np.mean(batches == b) for b in batches_])

    def _entropy(sample: np.ndarray) -> float:
        p = np.array([np.mean(sample == b) for b in batches_])
        a = np.sum(p / prior)
        adapt = (p / prior) / a
        return float(-np.sum(adapt * np.log(adapt + 1e-8)))

    score = 0.0
    rows, cols = kmatrix.nonzero()
    for _ in range(n_pools):
        indices = rng.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        sub = kmatrix[indices]
        r, c = sub.nonzero()
        score += float(np.mean([_entropy(batches[c[r == i]])
                                for i in range(n_samples_per_pool)]))
    return score / float(n_pools) / float(np.log2(n_batches))


def label_corr(x_mat: np.ndarray, x_label: np.ndarray) -> float:
    """Distance correlation between an embedding and a label vector.

    This is the paper's "time correlation" when ``x_label`` is the timepoint.

    Parameters
    ----------
    x_mat : numpy.ndarray, shape (n, d)
        Embedding.
    x_label : numpy.ndarray, shape (n,)
        Numeric label per cell.

    Returns
    -------
    float
        Distance correlation in ``[0, 1]``.
    """
    import dcor
    return float(dcor.distance_correlation(x_mat, np.asarray(x_label, dtype=float)))


#: Community-detection backend.  The reference uses ``scanpy.tl.louvain``, but the
#: installed ``louvain-igraph 0.7.1`` segfaults inside
#: ``louvain.VertexPartition.__init__`` against ``python-igraph 1.0.0`` -- an ABI
#: mismatch that kills the process even single-threaded and single-process.
#: ``leidenalg 0.12`` matches the installed igraph and is what scanpy now
#: recommends (it deprecates ``louvain`` in favour of ``leiden``).  The switch is a
#: deviation from the reference and is recorded in every output record.
CLUSTERING_BACKEND: str = "leiden"


def cluster_cells(x: np.ndarray, seed: int = 0,
                  backend: str | None = None) -> np.ndarray:
    """Community-detect an embedding, as ``Compare_Modal_Alignment_Clustering`` does.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_cells, n_features)
        Embedding to cluster.
    seed : int, optional
        Passed as the algorithm's ``random_state``.
    backend : {'leiden', 'louvain'}, optional
        Community-detection algorithm.  Defaults to
        :data:`CLUSTERING_BACKEND`.  ``louvain`` reproduces the reference exactly
        but segfaults in this environment; see the note on that constant.

    Returns
    -------
    numpy.ndarray
        Integer cluster assignment per cell.
    """
    import scanpy

    backend = backend or CLUSTERING_BACKEND
    ann = scanpy.AnnData(X=np.asarray(x, dtype=np.float32))
    scanpy.pp.neighbors(ann)
    if backend == "louvain":
        scanpy.tl.louvain(ann, random_state=seed)
        key = "louvain"
    else:
        scanpy.tl.leiden(ann, random_state=seed, flavor="igraph",
                         n_iterations=2, directed=False)
        key = "leiden"
    return np.asarray(ann.obs[key].values, dtype=int)


#: Backwards-compatible alias; the reference's name for the same step.
louvain_clusters = cluster_cells


def nmi_louvain(x1_feature: np.ndarray, x2_feature: np.ndarray,
                x1_label: np.ndarray, x2_label: np.ndarray,
                seed: int = 0) -> dict[str, float]:
    """Cell-type NMI via Louvain, per modality and averaged.

    Mirrors ``_evaluateClustering``: cells labelled ``"unknown"`` are dropped,
    each modality's cells in the integrated latent are clustered separately, and
    the two NMIs are averaged.  ARI is reported alongside because the ZebraHub
    work used ARI and both are wanted for comparability.

    Parameters
    ----------
    x1_feature, x2_feature : numpy.ndarray
        Integrated embeddings of the RNA and ATAC cells.
    x1_label, x2_label : numpy.ndarray
        Ground-truth cell types per modality.
    seed : int, optional
        Louvain random state.

    Returns
    -------
    dict
        ``nmi`` (average), ``nmi_rna``, ``nmi_atac``, ``ari``, ``ari_rna``,
        ``ari_atac`` and the number of clusters found per modality.
    """
    out: dict[str, float] = {}
    for tag, feat, lab in (("rna", x1_feature, x1_label), ("atac", x2_feature, x2_label)):
        lab = np.asarray(lab).astype(str)
        keep = lab != "unknown"
        feat, lab = feat[keep], lab[keep]
        pred = louvain_clusters(feat, seed=seed)
        out[f"nmi_{tag}"] = float(normalized_mutual_info_score(lab, pred))
        out[f"ari_{tag}"] = float(adjusted_rand_score(lab, pred))
        out[f"n_clusters_{tag}"] = float(len(np.unique(pred)))
    out["nmi"] = 0.5 * (out["nmi_rna"] + out["nmi_atac"])
    out["ari"] = 0.5 * (out["ari_rna"] + out["ari_atac"])
    return out


# ===========================================================================
# Nulls and baselines -- the part the paper does not report
# ===========================================================================
def lta_baselines(type1: np.ndarray, type2: np.ndarray,
                  n_permutations: int = 20, seed: int = 0) -> dict[str, float]:
    """Baselines a label transfer must beat before it means anything.

    Parameters
    ----------
    type1 : numpy.ndarray
        Ground-truth labels of the query (RNA) cells.
    type2 : numpy.ndarray
        Labels available to the classifier (ATAC cells).
    n_permutations : int, optional
        Number of label permutations for the permutation null.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``majority`` -- accuracy of always predicting the most common
        ``type2`` label; ``map_ceiling`` -- fraction of query cells whose label
        exists at all in the reference vocabulary, an upper bound no method can
        exceed; ``permutation_mean``/``permutation_sd`` -- accuracy when the
        reference labels are shuffled; ``n_shared_labels``.
    """
    rng = np.random.default_rng(seed)
    t1 = np.asarray(type1).astype(str)
    t2 = np.asarray(type2).astype(str)

    vals, counts = np.unique(t2, return_counts=True)
    majority_label = vals[np.argmax(counts)]
    majority = float((t1 == majority_label).mean())

    shared = set(vals) & set(np.unique(t1))
    ceiling = float(np.isin(t1, list(shared)).mean())

    perms = []
    for _ in range(n_permutations):
        perms.append(float((t1 == rng.permutation(t2)[:len(t1)]).mean())
                     if len(t2) >= len(t1) else
                     float((t1 == rng.choice(t2, size=len(t1))).mean()))
    return {
        "majority": majority,
        "majority_label": majority_label,
        "map_ceiling": ceiling,
        "permutation_mean": float(np.mean(perms)),
        "permutation_sd": float(np.std(perms)),
        "n_shared_labels": float(len(shared)),
    }


def foscttm_nulls(x1_mat: np.ndarray, x2_mat: np.ndarray,
                  groups: np.ndarray | None = None,
                  seed: int = 0) -> dict[str, float]:
    """Nulls for FOSCTTM: a shuffled null and a composition-conditioned floor.

    Parameters
    ----------
    x1_mat, x2_mat : numpy.ndarray, shape (n, d)
        Row-aligned embeddings.
    groups : numpy.ndarray, optional
        Group label per cell (e.g. timepoint or cell type).  When given, the
        blocked null is the FOSCTTM obtained by a permutation that respects
        group composition, reported as a ratio.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``shuffled`` -- FOSCTTM under a random pairing, which must land on
        ~0.5; ``blocked`` and ``blocked_ratio`` when ``groups`` is supplied.
        A latent that only recovers group identity, not cell identity, scores
        ``blocked_ratio`` near 1.
    """
    rng = np.random.default_rng(seed)
    n = x1_mat.shape[0]
    out = {"shuffled": foscttm(x1_mat, x2_mat[rng.permutation(n)])}

    if groups is not None:
        groups = np.asarray(groups)
        perm = np.arange(n)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            perm[idx] = rng.permutation(idx)
        out["blocked"] = foscttm(x1_mat, x2_mat[perm])
        out["blocked_ratio"] = (foscttm(x1_mat, x2_mat) / out["blocked"]
                                if out["blocked"] > 0 else float("nan"))
    return out


def clustering_nulls(labels: np.ndarray, pred: np.ndarray,
                     n_permutations: int = 20, seed: int = 0) -> dict[str, float]:
    """Shuffled-label null for NMI and ARI.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground-truth labels.
    pred : numpy.ndarray
        Cluster assignment.
    n_permutations : int, optional
        Number of permutations.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        Mean and sd of NMI and ARI under label permutation.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    nmis, aris = [], []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        nmis.append(normalized_mutual_info_score(shuffled, pred))
        aris.append(adjusted_rand_score(shuffled, pred))
    return {
        "nmi_null_mean": float(np.mean(nmis)), "nmi_null_sd": float(np.std(nmis)),
        "ari_null_mean": float(np.mean(aris)), "ari_null_sd": float(np.std(aris)),
    }


def batch_vs_type_scores(x: np.ndarray, cell_type: np.ndarray,
                         batch: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Score a latent against donor/batch as well as against cell type.

    On ZebraHub every method's latent clustered more by batch than by cell type
    in at least one modality.  This is the cleanest display of that confound and
    needs no classifier.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Embedding of one modality's cells.
    cell_type : numpy.ndarray, shape (n,)
        Cell-type labels.
    batch : numpy.ndarray, shape (n,)
        Donor / batch labels.
    seed : int, optional
        Louvain random state.

    Returns
    -------
    dict
        NMI and ARI against cell type and against batch, plus the batch-to-type
        ratio for each.  A ratio above 1 means the latent encodes batch more
        strongly than biology.
    """
    pred = louvain_clusters(x, seed=seed)
    ct = np.asarray(cell_type).astype(str)
    bt = np.asarray(batch).astype(str)
    out = {
        "nmi_type": float(normalized_mutual_info_score(ct, pred)),
        "nmi_batch": float(normalized_mutual_info_score(bt, pred)),
        "ari_type": float(adjusted_rand_score(ct, pred)),
        "ari_batch": float(adjusted_rand_score(bt, pred)),
        "n_clusters": float(len(np.unique(pred))),
    }
    out["nmi_batch_over_type"] = (out["nmi_batch"] / out["nmi_type"]
                                  if out["nmi_type"] > 0 else float("inf"))
    out["ari_batch_over_type"] = (out["ari_batch"] / out["ari_type"]
                                  if out["ari_type"] > 0 else float("inf"))
    return out


def knn_label_preservation(x: np.ndarray, labels: np.ndarray,
                           k: int = 30) -> float:
    """Fraction of a cell's k nearest neighbours sharing its label.

    This is the ``kNN-30`` column of ``tab:nd-benchmark`` -- a within-modality
    structure measure that needs no cross-modal correspondence.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Embedding.
    labels : numpy.ndarray, shape (n,)
        Labels.
    k : int, optional
        Neighbourhood size, excluding the cell itself.

    Returns
    -------
    float
        Mean same-label fraction across cells.
    """
    labels = np.asarray(labels).astype(str)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x)), n_jobs=-1).fit(x)
    idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def rarest_type_silhouette(x: np.ndarray, labels: np.ndarray,
                           rarest: str | None = None) -> dict[str, Any]:
    """One-vs-rest silhouette of the rarest cell type (the paper's Fig. 3C/D).

    Parameters
    ----------
    x : numpy.ndarray, shape (n, d)
        Embedding.
    labels : numpy.ndarray, shape (n,)
        Cell-type labels.
    rarest : str, optional
        Cell type to score.  Defaults to the least frequent label.

    Returns
    -------
    dict
        The scored label, its cell count and fraction, and the mean one-vs-rest
        silhouette of its cells.
    """
    labels = np.asarray(labels).astype(str)
    vals, counts = np.unique(labels, return_counts=True)
    if rarest is None:
        rarest = str(vals[np.argmin(counts)])
    mask = labels == rarest
    if mask.sum() < 2 or (~mask).sum() < 2:
        return {"cell_type": rarest, "n": int(mask.sum()),
                "fraction": float(mask.mean()), "silhouette": float("nan")}
    binary = np.where(mask, 1, 0)
    sil = silhouette_samples(x, binary)
    return {
        "cell_type": rarest,
        "n": int(mask.sum()),
        "fraction": float(mask.mean()),
        "silhouette": float(sil[mask].mean()),
    }
