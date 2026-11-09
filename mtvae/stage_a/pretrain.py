"""Per-modality autoencoder pre-training.

:func:`train_modality_ae` fits one
:class:`~mtvae.nn.autoencoders.ModalityAutoencoder` by reconstruction (MSE or
Huber) with Adam, holding out a validation fraction. Stage A uses it in
``staged`` mode to pre-train the trunks before the fused model is trained.
Mini-batches are drawn by sampling integer row indices, with no DataLoader.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from mtvae.nn.autoencoders import ModalityAutoencoder

log = logging.getLogger("mtvae.stage_a.pretrain")


@dataclass
class Phase1Config:
    """Settings of :func:`train_modality_ae`.

    Attributes
    ----------
    input_dim : int
        Number of features; must match the training matrix.
    latent_dim : int, default 48
        Code dimension.
    n_iters : int, default 1000
        Optimisation steps.
    batch_size : int, default 512
        Rows per step.
    lr : float, default 1e-3
        Adam learning rate.
    val_frac : float, default 0.10
        Fraction of rows held out for validation.
    seed : int, default 314159
        Seed of the row split, the batch draws and the weight initialisation.
    log_every : int, default 100
        Logging interval in steps.
    loss : {"mse", "huber"}, default "mse"
        Training loss.
    huber_delta : float, default 1.0
        Huber delta, used when ``loss="huber"``.
    """
    input_dim: int
    latent_dim: int = 48
    n_iters: int = 1_000
    batch_size: int = 512
    lr: float = 1e-3
    val_frac: float = 0.10
    seed: int = 314159
    log_every: int = 100
    loss: str = "mse"          # "mse" | "huber"
    huber_delta: float = 1.0   # only used when loss == "huber"


def train_modality_ae(
    matrix: np.ndarray,
    cfg: Phase1Config,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
) -> tuple[ModalityAutoencoder, dict]:
    """Train a ``ModalityAutoencoder`` on one modality.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (n_cells, n_features)
        Feature matrix; loaded onto ``device`` in full.
    cfg : Phase1Config
        Training settings.
    device : str, default "cpu"
        Compute device.
    checkpoint_path : pathlib.Path, optional
        If given, the state dict, config and history are saved there.

    Returns
    -------
    model : ModalityAutoencoder
        The trained autoencoder.
    history : dict
        Keys ``iter``, ``train_loss`` and ``val_loss`` (validation MSE on up
        to ``4 * batch_size`` held-out rows, whatever the training loss).

    Raises
    ------
    AssertionError
        If ``matrix`` does not have ``cfg.input_dim`` columns.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_cells, n_features = matrix.shape
    assert n_features == cfg.input_dim, (
        f"config input_dim {cfg.input_dim} != matrix features {n_features}"
    )
    n_val = int(n_cells * cfg.val_frac)
    perm = rng.permutation(n_cells)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    log.info("n_train = %d, n_val = %d", len(train_idx), len(val_idx))

    # The whole matrix is loaded once as float32.
    X = torch.from_numpy(np.asarray(matrix, dtype=np.float32)).to(device)

    model = ModalityAutoencoder(
        input_dim=cfg.input_dim, latent_dim=cfg.latent_dim,
    ).to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)

    history = {"iter": [], "train_loss": [], "val_loss": []}
    def loss_fn(x_rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Training loss selected by ``cfg.loss``."""
        if cfg.loss == "huber":
            return F.huber_loss(x_rec, x, delta=cfg.huber_delta)
        return F.mse_loss(x_rec, x)

    for step in range(1, cfg.n_iters + 1):
        model.train()
        batch_rows = rng.choice(train_idx, size=cfg.batch_size, replace=False)
        xb = X[batch_rows]
        x_rec, _ = model(xb)
        loss = loss_fn(x_rec, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == 1 or step == cfg.n_iters:
            model.eval()
            with torch.no_grad():
                # Eval on up to 4 * batch_size held-out cells
                eval_rows = val_idx[: min(len(val_idx), 4 * cfg.batch_size)]
                xv = X[eval_rows]
                x_rec_v, _ = model(xv)
                # Validation is always MSE, whatever the training loss.
                val_loss = F.mse_loss(x_rec_v, xv).item()
            log.info(
                "iter %4d  train_mse %.4f  val_mse %.4f", step, loss.item(), val_loss,
            )
            history["iter"].append(step)
            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss))

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(cfg),
                "history": history,
            },
            checkpoint_path,
        )
        log.info("Saved checkpoint to %s", checkpoint_path)

    return model, history


def knn_label_preservation(
    z: np.ndarray, labels: np.ndarray, k: int = 20,
) -> float:
    """Fraction of each cell's ``k`` nearest latent neighbours sharing its label.

    Parameters
    ----------
    z : numpy.ndarray, shape (n, d)
        Latent codes.
    labels : numpy.ndarray, shape (n,)
        Cell labels.
    k : int, default 20
        Number of neighbours (the cell itself excluded).

    Returns
    -------
    float
        Mean label agreement over all cells, in ``[0, 1]``.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="euclidean")
    nn.fit(z)
    _, idx = nn.kneighbors(z)
    neighbour_labels = labels[idx[:, 1:]]
    matches = (neighbour_labels == labels[:, None]).sum(axis=1)
    return float(matches.mean() / k)
