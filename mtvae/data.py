"""The data contract: what an example supplies and what the model consumes.

A :class:`Dataset` holds exactly two :class:`Modality` objects and an ordered
stage vocabulary. Each modality carries its feature matrix, a per-cell stage
index, and optionally a teacher embedding (for the Stage-A tilt), a training
mask, group codes, a trunk declaration (:class:`~mtvae.nn.trunks.TrunkSpec`)
and a reconstruction loss.

The adapters (:mod:`mtvae.adapters`) own everything about the files: which
matrix and teacher, the stage vocabulary and index mapping, the training
mask, the modality order and the chromosome ids for a spatial trunk. The
model never opens a file and never transforms a feature, except for the
symmetric clip that :class:`Modality` applies itself (:data:`UNIFORM_CLIP`
unless set otherwise).

Two constraints:

* **Exactly two modalities.** The model has two slots, ``"rna"`` and
  ``"atac"``, and checkpoint keys use those names (``rna_ae.*``,
  ``atac_ae.*``) whatever a dataset calls its modalities.
* **Order is the slot assignment.** ``modalities[0]`` fills the ``"rna"``
  slot and ``modalities[1]`` the ``"atac"`` slot. Modality names are free.

Example::

    from mtvae import Dataset, Modality, stage_times_from
    rna = Modality("rna", x_rna, stage_rna, teacher=t_rna)
    atac = Modality("atac", x_atac, stage_atac, teacher=t_atac)
    ds = Dataset("mydata", stages=("E1", "E2", "E3"), modalities=(rna, atac),
                 stage_times=stage_times_from([stage_rna, stage_atac],
                                              [age_rna, age_atac], 3))
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from mtvae.nn.trunks import TrunkSpec
from mtvae.stage_a.losses import DEFAULT_SLOT_LOSSES


def stage_times_from(stage_idx: Sequence[np.ndarray],
                    continuous: Sequence[np.ndarray],
                    n_stages: int) -> np.ndarray:
    """Normalised stage times from a per-cell continuous coordinate.

    Stage B integrates between stages, so it needs each stage's position on
    a real axis, not just its ordinal. Both examples derive it from
    ``log_age``. The per-stage mean is taken over the concatenation of both
    modalities, and the result is min-max normalised; computing either step
    per modality would give a different grid.

    Parameters
    ----------
    stage_idx : sequence of numpy.ndarray
        Per-modality stage index arrays, ``(n_cells,)`` int.
    continuous : sequence of numpy.ndarray
        Per-modality continuous coordinate, ``(n_cells,)`` float, in the same
        modality order and the same row order as ``stage_idx``.
    n_stages : int
        Size of the stage vocabulary.

    Returns
    -------
    numpy.ndarray
        ``(n_stages,)`` float, min-max normalised to ``[0, 1]``.

    Raises
    ------
    ValueError
        If a stage has no cells in either modality (its mean would be ``nan``),
        or if all stages share one mean time.
    """
    all_c = np.concatenate([np.asarray(c, dtype=np.float64) for c in continuous])
    all_s = np.concatenate([np.asarray(s) for s in stage_idx])
    empty = [s for s in range(n_stages) if not (all_s == s).any()]
    if empty:
        raise ValueError(
            f"stages {empty} have no cells in either modality; their mean time "
            f"would be nan and would propagate silently through Stage B")
    means = np.array([all_c[all_s == s].mean() for s in range(n_stages)])
    span = means.max() - means.min()
    if span == 0:
        raise ValueError(
            "all stages share one mean time; the normalisation would divide by "
            "zero and Stage B would have no axis to integrate along")
    return (means - means.min()) / span


#: Default symmetric feature bound, applied by :class:`Modality` itself to
#: every dataset that reaches the model, however it was built.
UNIFORM_CLIP = 10.0


class _Unset:
    """Sentinel: ``clip`` was not specified, so the uniform bound applies.

    Distinct from ``None``, which means "explicitly do not clip".
    """

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return "UNSET"


UNSET = _Unset()


@dataclasses.dataclass(frozen=True)
class Modality:
    """One modality's cells.

    Parameters
    ----------
    name : str
        Modality name (``"rna"`` / ``"atac"`` in both examples).
    x : numpy.ndarray, shape (n_cells, n_features), float32
        Features, already scaled by the adapter. The model does not transform
        them apart from the clip below.
    stage : numpy.ndarray, shape (n_cells,), int64
        Stage index in ``[0, n_stages)``.
    teacher : numpy.ndarray, optional
        Fixed teacher embedding for the Stage-A topology tilt. Required when
        the matching ``lambda_topo_*`` is positive.
    train : numpy.ndarray of bool, optional
        Training mask. ``None`` means all cells train. Row order must match
        ``x``: latents, metadata and mask are joined by position.
    group : numpy.ndarray, optional
        Donor or batch codes.
    trunk : TrunkSpec, default TrunkSpec()
        Which encoder/decoder trunk this modality gets (the MLP by default).
    clip : float or None, optional
        Symmetric bound applied to ``x`` here and recorded on the object.
        Unset means :data:`UNIFORM_CLIP`; a float sets another bound; ``None``
        means explicitly unclipped. A matrix already within the bound is kept
        as is, without a copy.
    loss : {"mse", "huber"}, optional
        Reconstruction loss of this modality. ``None`` uses the slot default
        (``"mse"`` for slot 0, ``"huber"`` for slot 1); declare it when the
        modalities are not RNA and ATAC.

    Raises
    ------
    ValueError
        If ``stage``, ``teacher``, ``train`` or ``group`` has a different
        number of rows than ``x``, or a spatial trunk's ``chr_ids`` length
        differs from the feature count.
    """
    name: str
    x: np.ndarray
    stage: np.ndarray
    teacher: Optional[np.ndarray] = None
    train: Optional[np.ndarray] = None
    group: Optional[np.ndarray] = None
    trunk: TrunkSpec = TrunkSpec()
    clip: Any = UNSET
    loss: Optional[str] = None

    def __post_init__(self) -> None:
        # The bound is applied here so that no dataset reaches training
        # unbounded, however it was built.
        c = UNIFORM_CLIP if self.clip is UNSET else self.clip
        if c is not None:
            # Check before copying: select() re-runs this on already-bounded
            # slices, and clipping is idempotent, so a bounded matrix needs no
            # copy.
            x = np.asarray(self.x)
            if x.size and (x.max() > c or x.min() < -c):
                object.__setattr__(self, "x", np.clip(x, -c, c))
        object.__setattr__(self, "clip", c)

        n = len(self.x)
        for field in ("stage", "teacher", "train", "group"):
            v = getattr(self, field)
            if v is not None and len(v) != n:
                raise ValueError(
                    f"{self.name}.{field} has {len(v)} rows, x has {n}; row order "
                    f"is joined by POSITION downstream, so a mismatch silently "
                    f"pairs the wrong cells")
        if self.trunk.kind == "spatial" and len(self.trunk.chr_ids) != self.x.shape[1]:
            raise ValueError(
                f"{self.name}: chr_ids has {len(self.trunk.chr_ids)} entries but "
                f"the matrix has {self.x.shape[1]} features")

    @property
    def n_cells(self) -> int:
        """Number of cells (rows of ``x``)."""
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features (columns of ``x``)."""
        return int(self.x.shape[1])

    def select(self, mask: np.ndarray) -> "Modality":
        """Return this modality restricted to ``mask``, all per-cell fields aligned.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask or integer index over cells.

        Returns
        -------
        Modality
        """
        return dataclasses.replace(
            self, x=self.x[mask], stage=self.stage[mask],
            teacher=None if self.teacher is None else self.teacher[mask],
            train=None if self.train is None else self.train[mask],
            group=None if self.group is None else self.group[mask])


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Exactly two modalities plus a stage vocabulary.

    Parameters
    ----------
    name : str
        For records and logging.
    stages : tuple of str
        Ordered stage vocabulary.
    modalities : tuple of Modality
        Exactly two; the order is the slot assignment.
    stage_times : numpy.ndarray, optional
        ``(n_stages,)`` normalised times in ``[0, 1]``, required by Stage B
        (see :func:`stage_times_from`).

    Raises
    ------
    ValueError
        If there are not exactly two modalities, or a cell's stage index is
        outside ``[0, n_stages)``.
    """
    name: str
    stages: Tuple[str, ...]
    modalities: Tuple[Modality, ...]
    stage_times: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if len(self.modalities) != 2:
            raise ValueError(
                f"{self.name}: this is a two-modality suite ({len(self.modalities)} "
                f"given). MergedVAE dispatches on two slots and the reconstruction "
                f"losses are hard-coded per slot.")
        for m in self.modalities:
            bad = (m.stage < 0) | (m.stage >= self.n_stages)
            if bad.any():
                raise ValueError(
                    f"{self.name}/{m.name}: {int(bad.sum())} cells have a stage "
                    f"outside [0, {self.n_stages}); unmapped stages silently "
                    f"collapse into stage 0")

    @property
    def n_stages(self) -> int:
        """Size of the stage vocabulary."""
        return len(self.stages)

    @property
    def names(self) -> Tuple[str, ...]:
        """Modality names, in slot order."""
        return tuple(m.name for m in self.modalities)

    @property
    def losses(self) -> Tuple[str, str]:
        """Per-slot reconstruction losses, declared or inherited.

        A modality that declares no loss gets its slot's default
        (:data:`mtvae.stage_a.losses.DEFAULT_SLOT_LOSSES`).

        Returns
        -------
        tuple of str
            ``(slot0_loss, slot1_loss)``.
        """
        return tuple(m.loss or d for m, d in
                     zip(self.modalities, DEFAULT_SLOT_LOSSES))

    def __getitem__(self, name: str) -> Modality:
        """Return the modality called ``name``; raise ``KeyError`` if absent."""
        for m in self.modalities:
            if m.name == name:
                return m
        raise KeyError(f"{self.name} has no modality {name!r}; has {self.names}")

    def training_view(self) -> "Dataset":
        """Return the dataset restricted to each modality's training mask.

        Fitting uses this view, while encoding uses the full dataset: the PFC
        example trains on a donor subset and encodes every cell, including
        the held-out donors.
        """
        mods = tuple(m if m.train is None else m.select(m.train)
                     for m in self.modalities)
        return dataclasses.replace(self, modalities=mods)
