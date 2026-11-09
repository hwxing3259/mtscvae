"""Read the examples and the trained models from the figshare layout.

Every path is relative to one data root: the unzipped figshare item, passed as an
argument or through the ``MTVAE_DATA`` environment variable::

    <data_root>/data/pfc/                    PFC inputs
    <data_root>/data/fourds/{HC,HO,DR,MN}/   four-dataset inputs
    <data_root>/models/pfc_full/s<seed>/     PFC fits (seeds 22, 44, 88)
    <data_root>/models/fourds/<DS>/s<seed>/  four-dataset fits (seeds 0, 1, 2)
"""
from __future__ import annotations

import dataclasses
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch

from mtvae.data import Dataset
from mtvae.nn.sde import SDESampler
from mtvae.pipeline import build_dataset

#: Sub-folder of each example under ``<data_root>/data``.
EXAMPLE_DIRS = {"pfc": Path("data/pfc"), "fourds": Path("data/fourds")}


def data_root(path: Optional[Path | str] = None) -> Path:
    """Resolve the figshare data root.

    Parameters
    ----------
    path : path-like, optional
        Explicit root. Defaults to ``$MTVAE_DATA``.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        If no root is given and ``MTVAE_DATA`` is unset, or the root has no
        ``data/`` folder.
    """
    p = path if path is not None else os.environ.get("MTVAE_DATA")
    if p is None:
        raise FileNotFoundError(
            "no data root: pass one, or set MTVAE_DATA to the unzipped figshare item")
    p = Path(p).expanduser()
    if not (p / "data").is_dir():
        raise FileNotFoundError(f"{p} has no data/ folder; is it the figshare root?")
    return p


def load_example(name: str, root: Optional[Path | str] = None, *,
                 dataset: Optional[str] = None,
                 data: Optional[Dict[str, Any]] = None) -> Dataset:
    """Load one of the paper's examples as a :class:`~mtvae.data.Dataset`.

    The data handling (input clipping, training split) is the example's recipe,
    i.e. what the released models were trained on, unless ``data`` overrides it.

    Parameters
    ----------
    name : {"pfc", "fourds"}
    root : path-like, optional
        Figshare data root; defaults to ``$MTVAE_DATA``.
    dataset : {"HC", "HO", "DR", "MN"}, optional
        Required for ``"fourds"``.
    data : dict, optional
        Overrides of the ``data`` section (``clip``, ``train_mask``, ``n_rows``).

    Returns
    -------
    Dataset
    """
    from mtvae.pipeline import PipelineConfig
    cfg = PipelineConfig.for_example(name, seed=0, device="cpu", data=data)
    return build_dataset(name, data_root(root), dataset=dataset, data=cfg.data,
                         arch=cfg.arch)


@dataclasses.dataclass
class Model:
    """A trained model: co-adapted encoder, two decoders and the latent SDE.

    Modalities are addressed by the model's slot names, ``"rna"`` (slot 0) and
    ``"atac"`` (slot 1).
    """
    encoder: torch.nn.Module
    decoders: Dict[str, torch.nn.Module]
    sde: SDESampler
    stage_times: np.ndarray
    device: str = "cpu"

    def encode(self, data: Dataset, chunk: int = 4096) -> Dict[str, np.ndarray]:
        """Posterior-mean latents of every cell, per modality.

        Parameters
        ----------
        data : Dataset
            Inputs scaled and clipped as for training (see :func:`load_example`).
        chunk : int, optional

        Returns
        -------
        dict
            Modality name -> ``(n_cells, latent_dim)`` float32.
        """
        slots = ("rna", "atac")
        return {m.name: self.encoder.encode_all(m.x, s, device=self.device, chunk=chunk)
                for m, s in zip(data.modalities, slots)}

    @torch.no_grad()
    def decode(self, z: np.ndarray, modality: str = "rna") -> np.ndarray:
        """Map latents to feature space with one modality's decoder.

        Parameters
        ----------
        z : numpy.ndarray, shape (n, latent_dim)
        modality : {"rna", "atac"}

        Returns
        -------
        numpy.ndarray, shape (n, n_features)
        """
        dec = self.decoders[modality]
        return dec(torch.as_tensor(np.asarray(z, np.float32), device=self.device)).cpu().numpy()

    def rollout(self, seeds: "OrderedDict[str, np.ndarray]", *, n_rollouts: int = 1,
                decode: Optional[str] = None,
                reduce: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Any:
        """Transport seed latents from the first stage to every stage.

        A thin wrapper of :func:`mtvae.predict.rollouts`: per rollout ``r`` the
        global torch stream is seeded with ``r`` and the seed sets are
        transported in the order given, sharing that stream.

        Parameters
        ----------
        seeds : OrderedDict of str to numpy.ndarray
            Named ``(n_cells, latent_dim)`` blocks, in transport order.
        n_rollouts : int, optional
        decode : {"rna", "atac"}, optional
            Decode every stage with this modality's decoder; ``None`` keeps latents.
        reduce : callable, optional
            Applied per stage before anything is kept, e.g. ``lambda x: x.mean(0)``.

        Returns
        -------
        OrderedDict of str to RolloutResult
        """
        from mtvae.predict import rollouts
        return rollouts(self.sde, seeds, self.stage_times, n_rollouts=n_rollouts,
                        decode_with=None if decode is None else self.decoders[decode],
                        reduce=reduce, device=self.device)


def load_model(model_dir: Path | str, *, device: str = "cpu") -> Model:
    """Load a trained model from its ``model.pt``.

    ``model.pt`` holds the co-adapted encoder, both decoders, the SDE, and the
    architecture they are rebuilt from (Stage-A config and trunk entries).

    Parameters
    ----------
    model_dir : path-like
        A fit folder (e.g. ``<data_root>/models/pfc_full/s44``) or ``model.pt``.
    device : str, optional

    Returns
    -------
    Model
    """
    from mtvae.nn.autoencoders import ModalityAutoencoder
    from mtvae.stage_b.merged_loader import build_encoder

    p = Path(model_dir)
    ck = torch.load(p / "model.pt" if p.is_dir() else p, weights_only=False,
                    map_location="cpu")
    esd = ck["encoder_state_dict"]
    dims = {m: int(esd[f"{m}_ae.encoder.net.0.weight"].shape[1]) for m in ("rna", "atac")}
    latent_dim = int(ck["config"]["latent_dim"])
    enc = build_encoder(ck["stage_a_config"], ck["trunks"], dims, latent_dim)
    enc.load_state_dict(esd, strict=True)
    enc.to(device).eval()
    decoders = {}
    for m in ("rna", "atac"):
        dec = ModalityAutoencoder(input_dim=dims[m], latent_dim=latent_dim).decoder
        dec.load_state_dict(ck[f"decoder_{m}_state_dict"], strict=True)
        decoders[m] = dec.to(device).eval()
    geom = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"].items()
            if k not in ("sde_dt", "latent_dim")}
    sde = SDESampler(latent_dim=latent_dim, **geom).to(device)
    sde.load_state_dict(ck["sde_state_dict"], strict=True)
    sde.eval()
    return Model(encoder=enc, decoders=decoders, sde=sde,
                 stage_times=np.asarray(ck["stage_norm"], np.float32), device=device)


def model_dir(example: str, seed: int, root: Optional[Path | str] = None, *,
              dataset: Optional[str] = None) -> Path:
    """Path of one released fit.

    Parameters
    ----------
    example : {"pfc", "fourds"}
    seed : int
    root : path-like, optional
    dataset : str, optional
        Four-dataset label.

    Returns
    -------
    pathlib.Path
    """
    r = data_root(root)
    if example == "pfc":
        return r / "models" / "pfc_full" / f"s{seed}"
    return r / "models" / "fourds" / str(dataset) / f"s{seed}"


def stage_index(stage_id: Sequence[str], stages: Sequence[str]) -> np.ndarray:
    """Integer stage index of each cell.

    Parameters
    ----------
    stage_id : sequence of str
    stages : sequence of str
        Ordered vocabulary.

    Returns
    -------
    numpy.ndarray of int64
    """
    s2i = {s: i for i, s in enumerate(stages)}
    return np.asarray([s2i[s] for s in stage_id], dtype=np.int64)
