"""Astrocyte maturation curves for the PFC example.
Gestation cohorts are carried through the trained SDE and decoded; astrocytes vs a control."""
from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from mtvae.predict import rollouts

#: Seed of the generator that subsamples the cohorts.
SUB_SEED = 0
#: Default cohort sizes and number of rollouts of the astrocyte protocol.
N_COGNATE, N_CONTROL, N_ROLLOUTS = 300, 100, 5


@dataclasses.dataclass(frozen=True)
class AstroCurves:
    """Decoded per-gene trajectories and the margin they imply.

    Attributes
    ----------
    genes : tuple of str
        Genes, in column order.
    cognate, control : numpy.ndarray, shape (n_stages, n_genes)
        Mean decoded expression of each cohort, averaged over rollouts.
    margin : mapping of str to float
        Per gene, ``tool_window(cognate) - tool_window(control)``.
    n_rollouts : int
        Number of rollouts.
    """
    genes: tuple
    cognate: np.ndarray      #: (n_stages, n_genes)
    control: np.ndarray      #: (n_stages, n_genes)
    margin: Mapping[str, float]
    n_rollouts: int

    @property
    def mean_margin(self) -> float:
        """Margin averaged over genes."""
        return float(np.mean(list(self.margin.values())))


def tool_window(curve: np.ndarray, early=(0, 2), late=(3, 5)) -> np.ndarray:
    """Late-minus-early contrast along the stage axis.

    Parameters
    ----------
    curve : numpy.ndarray, shape (n_stages, n_genes)
        Per-stage values.
    early, late : tuple of int, default (0, 2) and (3, 5)
        Half-open stage ranges.

    Returns
    -------
    numpy.ndarray, shape (n_genes,)
    """
    c = np.asarray(curve)
    return c[late[0]:late[1]].mean(0) - c[early[0]:early[1]].mean(0)


def seed_cells(labels: Sequence[str], stage: np.ndarray, *,
               cognate_label: str = "Astro", stage_index: int = 0,
               n_cognate: int = N_COGNATE, n_control: int = N_CONTROL,
               sub_seed: int = SUB_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Choose cognate and control seed cells at the origin stage.

    Cognate = ``cognate_label`` cells at ``stage_index``; control = every
    other cell at that stage. Both are subsampled with one fixed generator
    (cognate first), so the choice is reproducible.

    Parameters
    ----------
    labels : sequence of str
        Cell type of every cell.
    stage : numpy.ndarray
        Stage index of every cell.
    cognate_label : str, default "Astro"
        Label of the cognate cell type.
    stage_index : int, default 0
        Origin stage.
    n_cognate, n_control : int, default N_COGNATE, N_CONTROL
        Cohort sizes.
    sub_seed : int, default SUB_SEED
        Seed of the subsampling generator.

    Returns
    -------
    cognate, control : numpy.ndarray
        Sorted cell indices.

    Raises
    ------
    ValueError
        If the origin stage has too few cognate or control cells.
    """
    labels = np.asarray(labels)
    at_stage = np.flatnonzero(stage == stage_index)
    is_cog = labels[at_stage] == cognate_label
    cog_pool, ctl_pool = at_stage[is_cog], at_stage[~is_cog]
    if len(cog_pool) < n_cognate or len(ctl_pool) < n_control:
        raise ValueError(
            f"stage {stage_index} has {len(cog_pool)} {cognate_label} and "
            f"{len(ctl_pool)} other cells; need {n_cognate}/{n_control}")
    g = np.random.default_rng(sub_seed)
    return (np.sort(g.choice(cog_pool, n_cognate, replace=False)),
            np.sort(g.choice(ctl_pool, n_control, replace=False)))


def curves(dyn: Any, latents: np.ndarray, stage_times: Sequence[float],
           labels: Sequence[str], stage: np.ndarray, *,
           decoder: Any, genes: Sequence[str], gene_index: Mapping[str, int],
           n_rollouts: int = N_ROLLOUTS, restrict: Optional[np.ndarray] = None,
           device: str = "cpu") -> AstroCurves:
    """Astrocyte curves and their margin, from one latent matrix.

    Seed cells are chosen with :func:`seed_cells` (default cohort sizes), and
    both cohorts are transported with :func:`mtvae.predict.rollouts`, cognate
    first.

    Parameters
    ----------
    dyn : SDESampler
        The trained SDE.
    latents : numpy.ndarray
        Latents of every cell, row-aligned with ``labels`` and ``stage``.
    stage_times : sequence of float
        Normalised stage times.
    labels : sequence of str
        Cell type of every cell.
    stage : numpy.ndarray
        Stage index of every cell.
    decoder : torch.nn.Module
        Decoder applied per stage.
    genes : sequence of str
        Genes to read.
    gene_index : mapping of str to int
        Column of each gene in the decoder output.
    n_rollouts : int, default N_ROLLOUTS
        Number of rollouts.
    restrict : numpy.ndarray of bool, optional
        Restrict seed selection, e.g. to held-out donors. ``None`` = all cells.
    device : str, default "cpu"
        Compute device.

    Returns
    -------
    AstroCurves
    """
    lab = np.asarray(labels)
    st = np.asarray(stage)
    if restrict is not None:
        keep = np.flatnonzero(restrict)
        lab, st, latents = lab[keep], st[keep], latents[keep]
    ci, oi = seed_cells(lab, st)

    gi = [gene_index[g] for g in genes]
    take = lambda xp: xp[:, gi].mean(0)          # per-gene population mean

    # Ordered: cognate then control, sharing one stream per rollout. Reversing
    # this changes every number and raises nothing (see mtvae.predict).
    res = rollouts(dyn, OrderedDict(cognate=latents[ci], control=latents[oi]),
                   stage_times, n_rollouts=n_rollouts, decode_with=decoder,
                   reduce=take, device=device)

    cog = np.stack(res["cognate"].per_stage)     # (n_stages, n_genes)
    ctl = np.stack(res["control"].per_stage)
    margin = {g: float(m) for g, m in
              zip(genes, tool_window(cog) - tool_window(ctl))}
    return AstroCurves(genes=tuple(genes), cognate=cog, control=ctl,
                       margin=margin, n_rollouts=res["cognate"].n_rollouts)


#: The eight marker genes of the astrocyte maturation panels.
IMMATURE_GENES = ("TNC", "ID4", "PTPRZ1", "SYNE2")
MATURE_GENES = ("NDRG2", "FGFR3", "ALDH1A1", "SLC1A2")


def gestation_pools(coarse: Mapping[str, np.ndarray], stage: Mapping[str, np.ndarray],
                    keep: Mapping[str, np.ndarray], *, modalities: Sequence[str] = ("rna", "atac"),
                    cognate_label: str = "Astro", stage_index: int = 0) -> dict:
    """Candidate cohort nuclei at the first stage, per modality.

    Parameters
    ----------
    coarse, stage : mapping of str to numpy.ndarray
        Per modality: coarse cell type and integer stage of every nucleus.
    keep : mapping of str to numpy.ndarray of bool
        Per modality: which nuclei may be drawn (e.g. the training-mask side).
    modalities : sequence of str, default ("rna", "atac")
        Pools to build, in order; ``("rna",)`` gives the RNA-only cohort.
    cognate_label : str, default "Astro"
        Label of the cognate cell type.
    stage_index : int, default 0
        Origin stage.

    Returns
    -------
    dict
        ``{"cognate": [(modality, idx), ...], "control": [...]}``, indices sorted.
    """
    out: dict = {"cognate": [], "control": []}
    for mod in modalities:
        at = (np.asarray(stage[mod]) == stage_index) & np.asarray(keep[mod], bool)
        is_cog = np.asarray(coarse[mod]) == cognate_label
        out["cognate"].append((mod, np.where(at & is_cog)[0]))
        out["control"].append((mod, np.where(at & ~is_cog)[0]))
    return out


def draw_cohort(rng: np.random.Generator, pool: Sequence, n: int) -> list:
    """Draw ``n`` nuclei uniformly from the union of per-modality pools.

    Parameters
    ----------
    rng : numpy.random.Generator
        Shared between the cognate and the control draw, cognate first.
    pool : sequence of (str, numpy.ndarray)
        One entry of :func:`gestation_pools`.
    n : int
        Cohort size; the whole pool is taken if it is smaller.

    Returns
    -------
    list of (str, numpy.ndarray)
        Per modality, the sorted indices drawn.
    """
    flat = np.concatenate([idx for _, idx in pool])
    pos = (np.sort(rng.choice(len(flat), n, replace=False)) if len(flat) > n
           else np.arange(len(flat)))
    out, start = [], 0
    for mod, idx in pool:
        sel = pos[(pos >= start) & (pos < start + len(idx))] - start
        out.append((mod, idx[sel]))
        start += len(idx)
    return out


def cohort_trajectories(model: Any, latents: Mapping[str, np.ndarray], pools: dict, *,
                        n_cognate: int = N_COGNATE, n_control: int = N_CONTROL,
                        n_rollouts: int = N_ROLLOUTS, sub_seed: int = SUB_SEED) -> dict:
    """Decoded RNA trajectories of a cognate and a control cohort.

    The protocol of the paper's astrocyte panels: draw the cognate cohort and
    then the control cohort with one ``default_rng(sub_seed)``; then, per
    rollout ``r``, seed torch with ``r`` and transport the cognate cohort and then
    the control cohort from the first stage to every stage through the SDE,
    decoding each stage with the RNA decoder and averaging over nuclei. A cohort
    drawn from both modalities starts each nucleus from its own modality's latent.

    Parameters
    ----------
    model : mtvae.io.Model
        Fitted model; its ``sde``, ``stage_times``, ``decoders["rna"]`` and
        ``device`` are used.
    latents : mapping of str to numpy.ndarray
        Per modality, the latents of every nucleus (row-aligned with the pools).
    pools : dict
        From :func:`gestation_pools`.
    n_cognate, n_control, n_rollouts, sub_seed : int, optional
        Cohort sizes, number of rollouts and cohort-sampling seed; default to
        the module constants.

    Returns
    -------
    dict
        ``cognate``/``control``: ``(n_stages, n_genes)`` means over rollouts;
        ``cognate_sd``/``control_sd``: their spread across rollouts (ddof 1);
        ``composition``: nuclei drawn per cohort and modality.
    """
    import torch
    from mtvae.predict import predict

    rng = np.random.default_rng(sub_seed)
    drawn = OrderedDict(cognate=draw_cohort(rng, pools["cognate"], n_cognate),
                        control=draw_cohort(rng, pools["control"], n_control))
    z0 = {k: np.concatenate([np.asarray(latents[m][i], np.float32) for m, i in v if len(i)])
          for k, v in drawn.items()}
    acc: dict = {k: [] for k in drawn}
    for rep in range(n_rollouts):
        torch.manual_seed(rep)
        for k in drawn:                              # cognate first (shared stream)
            p = predict(model.sde, z0[k], model.stage_times,
                        decode_with=model.decoders["rna"], reduce=lambda x: x.mean(0),
                        device=model.device)
            acc[k].append(np.stack(p.per_stage))
    out = {k: np.mean(v, axis=0) for k, v in acc.items()}
    out.update({f"{k}_sd": np.std(v, axis=0, ddof=1) for k, v in acc.items()})
    out["composition"] = {f"{k}_{m}": int(len(i)) for k, v in drawn.items() for m, i in v}
    return out
