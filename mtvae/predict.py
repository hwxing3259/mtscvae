"""Transport cells through a trained SDE: :func:`predict` and :func:`rollouts`.
The seed sets of :func:`rollouts` share one Brownian stream, so their order changes results."""
from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class Prediction:
    """One transport's output, per stage.

    Attributes
    ----------
    per_stage : list of numpy.ndarray
        Per stage, the (reduced) latents or decoded features.
    z : list of numpy.ndarray, optional
        Per stage, the transported latents (only with ``keep_z=True``).
    """
    per_stage: list
    z: Optional[list] = None


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    """Several rollouts of one seed set, already reduced.

    Attributes
    ----------
    per_stage : list of numpy.ndarray
        Per stage, the mean over rollouts of what ``reduce`` returned.
    n_rollouts : int
        Number of rollouts.
    spread : list of numpy.ndarray, optional
        Per stage, the standard deviation across rollouts (``ddof=0``);
        ``None`` for a single rollout.
    """
    per_stage: list          #: mean over rollouts of whatever `reduce` returned
    n_rollouts: int
    spread: Optional[list] = None   #: per-stage sd across rollouts, if >1


def predict(sde: Any, z0: np.ndarray, stage_times: Sequence[float], *,
            decode_with: Optional[Any] = None,
            generator: Optional[torch.Generator] = None,
            reduce: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            keep_z: bool = False, device: str = "cpu") -> Prediction:
    """Transport ``z0`` to every stage and optionally decode.

    Parameters
    ----------
    sde : SDESampler
        The trained SDE.
    z0 : numpy.ndarray, shape (n_cells, latent_dim)
        Seed latents at the first stage.
    stage_times : sequence of float
        Normalised stage times; ``stage_times[0]`` is the origin.
    decode_with : torch.nn.Module, optional
        Decoder applied per stage. ``None`` returns latents only.
    generator : torch.Generator, optional
        Generator for the Brownian increments; ``None`` draws from the global
        torch stream. To share one stream across several seed sets, use
        :func:`rollouts`.
    reduce : callable, optional
        Applied per stage before anything is retained, e.g.
        ``lambda xp: xp.mean(0)``. Without it a full-atlas decode is large
        (6 stages x 154,748 cells x 5,000 features in float32 is 18.6 GB).
    keep_z : bool, default False
        Also retain the transported latents.
    device : str, default "cpu"
        Compute device.

    Returns
    -------
    Prediction
    """
    z0_t = torch.as_tensor(np.asarray(z0, np.float32), device=device)
    sn = [float(t) for t in stage_times]
    out, zs = [], ([] if keep_z else None)
    with torch.no_grad():
        for s in range(len(sn)):
            z_s = z0_t if s == 0 else sde.forward(
                z0_t, torch.tensor(sn[s]), torch.tensor(sn[0]),
                generator=generator)
            if keep_z:
                zs.append(z_s.cpu().numpy().copy())
            v = z_s if decode_with is None else decode_with(z_s)
            v = v.cpu().numpy()
            out.append(reduce(v) if reduce is not None else v.copy())
    return Prediction(per_stage=out, z=zs)


def rollouts(sde: Any, seeds: "OrderedDict[str, np.ndarray]",
             stage_times: Sequence[float], *, n_rollouts: int = 1,
             seed_of_rollout: Callable[[int], int] = lambda rep: rep,
             decode_with: Optional[Any] = None,
             reduce: Optional[Callable[[np.ndarray], np.ndarray]] = None,
             device: str = "cpu") -> "OrderedDict[str, RolloutResult]":
    """Repeat :func:`predict` over rollouts for several seed sets sharing one stream.

    Per rollout, the global torch stream is seeded once and each seed set is
    then transported in the order given (see the module docstring).

    Parameters
    ----------
    sde : SDESampler
        The trained SDE.
    seeds : OrderedDict of str to numpy.ndarray
        Named seed-latent blocks, in the order they are transported, e.g.
        ``OrderedDict(cognate=..., control=...)`` in the astrocyte protocol.
    stage_times : sequence of float
        Normalised stage times; ``stage_times[0]`` is the origin.
    n_rollouts : int, default 1
        Number of rollouts. The astrocyte protocol uses 5
        (:data:`mtvae.astro.N_ROLLOUTS`); rollout 0 alone is the
        single-rollout result.
    seed_of_rollout : callable, default identity
        ``rep -> torch seed`` of each rollout.
    decode_with : torch.nn.Module, optional
        Decoder applied per stage.
    reduce : callable, optional
        Applied per stage before results are retained (see :func:`predict`).
    device : str, default "cpu"
        Compute device.

    Returns
    -------
    OrderedDict of str to RolloutResult
        One result per seed set, in the same order.

    Raises
    ------
    TypeError
        If ``seeds`` is not an ``OrderedDict``.
    """
    if not isinstance(seeds, OrderedDict):
        raise TypeError(
            "seeds must be an OrderedDict: the seed sets share one random stream "
            "per rollout, so their order changes every result and must be explicit.")
    acc: dict[str, list] = {k: [] for k in seeds}
    for rep in range(n_rollouts):
        torch.manual_seed(seed_of_rollout(rep))
        for name, z0 in seeds.items():          # order matters: one shared stream
            p = predict(sde, z0, stage_times, decode_with=decode_with,
                        reduce=reduce, device=device)
            acc[name].append(p.per_stage)
    out: "OrderedDict[str, RolloutResult]" = OrderedDict()
    for name, reps in acc.items():
        n_stages = len(reps[0])
        mean = [np.mean([r[s] for r in reps], axis=0) for s in range(n_stages)]
        sd = ([np.std([r[s] for r in reps], axis=0) for s in range(n_stages)]
              if len(reps) > 1 else None)
        out[name] = RolloutResult(per_stage=mean, n_rollouts=len(reps), spread=sd)
    return out
