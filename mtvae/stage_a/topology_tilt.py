"""Neighbourhood-aware topology tilt: a structure-preserving regulariser.
MIND's per-modality term ``-sum_ij P_ij log Q_ij(Z)``, with ``P`` from a teacher embedding."""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.metrics import pairwise_distances

_EPS = 1e-12


def teacher_affinity(
    emb: torch.Tensor,
    perplexity: float = 30.0,
    impl: str = "sklearn",
) -> torch.Tensor:
    """Symmetric t-SNE teacher affinity ``P`` from a clean embedding batch.

    Per-row Gaussian conditional probabilities whose entropy matches the
    target perplexity, symmetrised and normalised to sum 1 (the t-SNE input
    affinity). The result is fixed: it is computed under ``no_grad`` in numpy.

    Parameters
    ----------
    emb : torch.Tensor, shape (B, d)
        Teacher embedding of the batch cells.
    perplexity : float, default 30.0
        Target perplexity (effective number of neighbours). Clamped to at
        most ``max(2, (B - 1) / 3)`` for numerical stability.
    impl : {"sklearn", "opentsne"}, default "sklearn"
        ``"sklearn"`` uses scikit-learn's dense, exact
        ``_joint_probabilities``; ``"opentsne"`` uses
        ``openTSNE.affinity.PerplexityBasedNN``, the sparse kNN estimator MIND
        uses. They target the same quantity, but openTSNE truncates each row
        to about ``3 * perplexity`` neighbours.

    Returns
    -------
    torch.Tensor
        ``(B, B)`` symmetric affinity matrix, diagonal zero, summing to 1.
    """
    with torch.no_grad():
        B = emb.shape[0]
        perp = float(min(perplexity, max(2.0, (B - 1) / 3.0)))
        if impl == "opentsne":
            from openTSNE import affinity
            x = np.ascontiguousarray(emb.detach().cpu().numpy().astype(np.float32))
            P = np.asarray(affinity.PerplexityBasedNN(
                x, perplexity=perp, metric="euclidean").P.todense(), dtype=np.float64)
            P = P / (P.sum() + _EPS)
            return torch.from_numpy(P).to(dtype=emb.dtype, device=emb.device)
        emb_np = emb.detach().cpu().numpy().astype(np.float64)
        d2 = pairwise_distances(emb_np, squared=True).astype(np.float64)
        np.fill_diagonal(d2, 0.0)
        p_cond = _joint_probabilities(d2, desired_perplexity=perp, verbose=0)
        P = squareform(p_cond)                               # (B, B), sym, sums to 1
        return torch.from_numpy(P).to(dtype=emb.dtype, device=emb.device)


def student_t_affinity(z: torch.Tensor) -> torch.Tensor:
    """Student-t latent affinity ``Q`` (the t-SNE low-dim kernel).

    ``Q_ij = (1 + ||z_i - z_j||^2)^-1``, diagonal zeroed, normalised to sum 1.
    Differentiable in ``z``.

    Parameters
    ----------
    z : torch.Tensor, shape (B, d)
        Latent batch.

    Returns
    -------
    torch.Tensor, shape (B, B)
    """
    d2 = torch.cdist(z, z) ** 2
    q = 1.0 / (1.0 + d2)
    q = q - torch.diag(torch.diagonal(q))                    # zero diagonal
    return q / (q.sum() + _EPS)


def full_teacher_affinity(
    emb: "np.ndarray",
    perplexity: float = 30.0,
) -> "sparse.csr_matrix":
    """Teacher affinity ``P`` computed once over the whole dataset.

    This is MIND's default path: the affinity is calibrated on every cell with
    ``openTSNE.affinity.PerplexityBasedNN`` and each batch then takes a
    submatrix of it (:func:`subset_affinity`). Compared with recomputing ``P``
    per batch, the neighbourhoods are the cells' true nearest neighbours
    rather than their nearest neighbours within a random batch, and the
    per-step cost is much lower.

    ``PerplexityBasedNN`` is a sparse kNN estimator (about ``3 * perplexity``
    neighbours per row), so the result stays small for large ``n``.

    Parameters
    ----------
    emb : numpy.ndarray, shape (n, d)
        Teacher embedding of every cell.
    perplexity : float, default 30.0
        Target perplexity.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n, n)
        Symmetric affinity, normalised to sum 1 over the whole matrix.
    """
    from openTSNE import affinity

    x = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
    aff = affinity.PerplexityBasedNN(x, perplexity=float(perplexity), metric="euclidean")
    return sparse.csr_matrix(aff.P)


def subset_affinity(p_full: "sparse.csr_matrix", idx: "np.ndarray",
                    device: str = "cpu", dtype: "torch.dtype" = torch.float32) -> torch.Tensor:
    """Take a batch submatrix of a full-data affinity and renormalise it.

    Mirrors MIND's ``sub_data_similarity / sub_data_similarity.sum()``.

    Parameters
    ----------
    p_full : scipy.sparse.csr_matrix
        Full-data affinity from :func:`full_teacher_affinity`.
    idx : np.ndarray
        Batch row indices.
    device : str, default "cpu"
        Device for the returned tensor.
    dtype : torch.dtype, default torch.float32
        Dtype for the returned tensor.

    Returns
    -------
    torch.Tensor, shape (len(idx), len(idx))
        Dense submatrix normalised to sum 1.
    """
    sub = np.asarray(p_full[idx][:, idx].todense(), dtype=np.float32)
    s = sub.sum()
    if s > 0:
        sub /= s
    return torch.from_numpy(sub).to(device=device, dtype=dtype)


def tilt_loss_from_P(z: torch.Tensor, p_batch: torch.Tensor) -> torch.Tensor:
    """``-sum_ij P_ij log Q_ij(z)`` for an already-computed batch affinity.

    Parameters
    ----------
    z : torch.Tensor, shape (B, d)
        Latent batch (gradient flows here).
    p_batch : torch.Tensor, shape (B, B)
        Teacher affinity of the same cells, e.g. from :func:`subset_affinity`.

    Returns
    -------
    torch.Tensor
        Scalar tilt loss.
    """
    q = student_t_affinity(z)
    return -(p_batch * torch.log(q + _EPS)).sum()


def topology_tilt_loss(
    z: torch.Tensor,
    teacher_emb: torch.Tensor,
    perplexity: float = 30.0,
    impl: str = "sklearn",
) -> torch.Tensor:
    """``L_tilt = -sum_ij P_ij log Q_ij(z)`` for one batch.

    The teacher affinity is recomputed from ``teacher_emb`` on every call.

    Parameters
    ----------
    z : torch.Tensor, shape (B, latent_dim)
        Learned latent of the batch (gradient flows here).
    teacher_emb : torch.Tensor, shape (B, d)
        Teacher embedding of the same cells (fixed).
    perplexity : float, default 30.0
        Teacher-affinity perplexity.
    impl : {"sklearn", "opentsne"}, default "sklearn"
        Teacher-affinity estimator; see :func:`teacher_affinity`.

    Returns
    -------
    torch.Tensor
        Scalar tilt loss.
    """
    P = teacher_affinity(teacher_emb, perplexity=perplexity, impl=impl)  # fixed
    Q = student_t_affinity(z)                                 # learned
    return -(P * torch.log(Q + _EPS)).sum()
