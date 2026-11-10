"""Swap the Stage-A ATAC trunk for the chromosome-aware Conv1d trunk.
An alternative to ``TrunkSpec("spatial", chr_ids=...)``, the route the pipeline uses."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import logging
import re
from typing import Iterator, Sequence

import numpy as np


from mtvae.nn.atac_spatial_encoder import (  # noqa: E402
    SpatialModalityAutoencoder, load_chromosome_ids,
)
from mtvae.stage_a import trainer as _vm  # noqa: E402

log = logging.getLogger("mtvae.nn.spatial_trunk")

__all__ = ["spatial_atac_trunk", "assert_trunk_contract", "TRUNK_CONTRACT_SHA",
           "load_chromosome_ids"]

#: SHA-256 of the whitespace-normalised trunk-construction block of
#: ``mtvae.stage_a.trainer.train_merged``. :func:`assert_trunk_contract`
#: prints the current value when it does not match.
TRUNK_CONTRACT_SHA = "759b5a0a8b026cb8e5b31a0c958375f1c92cec0adbc6ddb5495531e2fe82c3d9"

_TRUNK_RE = re.compile(
    r"rna_ae\s*=\s*ModalityAutoencoder\(.*?atac_ae\s*=\s*ModalityAutoencoder\(.*?\)",
    re.DOTALL,
)


def _trunk_block() -> str:
    """Return the whitespace-normalised trunk-construction source block.

    Raises
    ------
    RuntimeError
        If the block cannot be found in ``train_merged``.
    """
    src = inspect.getsource(_vm.train_merged)
    m = _TRUNK_RE.search(src)
    if m is None:
        raise RuntimeError(
            "Could not locate the trunk-construction block in "
            "mtvae.stage_a.trainer.train_merged. The spatial-trunk patch in "
            "mtvae.nn.spatial_trunk no longer applies -- re-read that function "
            "before training with a spatial trunk."
        )
    return " ".join(m.group(0).split())


def assert_trunk_contract(strict: bool = True) -> str:
    """Verify that the trunk construction this swap relies on is unchanged.

    Parameters
    ----------
    strict : bool, default True
        Raise on mismatch. ``False`` logs a warning and continues, for a change
        that has been reviewed and is known to be benign.

    Returns
    -------
    str
        The SHA-256 of the current trunk-construction block.

    Raises
    ------
    RuntimeError
        If ``strict`` and the block does not match :data:`TRUNK_CONTRACT_SHA`,
        or if the block cannot be found at all.
    """
    got = hashlib.sha256(_trunk_block().encode()).hexdigest()
    if got != TRUNK_CONTRACT_SHA:
        msg = (
            "mtvae.stage_a.trainer trunk construction has CHANGED.\n"
            f"  expected {TRUNK_CONTRACT_SHA}\n  got      {got}\n"
            f"  block:   {_trunk_block()}\n"
            "The spatial patch assumes RNA is built first and ATAC second. "
            "Re-read train_merged and update TRUNK_CONTRACT_SHA if the order "
            "and count still hold."
        )
        if strict:
            raise RuntimeError(msg)
        log.warning(msg)
    return got


@contextlib.contextmanager
def spatial_atac_trunk(chr_ids: np.ndarray,
                       atac_input_dim: int,
                       spatial_kernel_sizes: Sequence[int] | int = 3,
                       strict_contract: bool = True) -> Iterator[dict]:
    """Swap the ATAC trunk for a chromosome-aware one inside ``train_merged``.

    Parameters
    ----------
    chr_ids : numpy.ndarray
        Per-peak chromosome id, from :func:`load_chromosome_ids`. Its length
        must equal the ATAC feature count.
    atac_input_dim : int
        Expected ATAC feature count, used to check that the replaced trunk was
        the ATAC one.
    spatial_kernel_sizes : sequence of int or int, default 3
        Conv kernel widths. The PFC example uses ``3``.
    strict_contract : bool, default True
        Passed to :func:`assert_trunk_contract`.

    Yields
    ------
    dict
        Mutable record with keys ``n_built``, ``atac_spatial`` and ``dims``,
        populated as trunks are constructed.

    Raises
    ------
    RuntimeError
        If the source contract fails, if the number of trunks built is not two,
        or if the replaced trunk did not receive ``atac_input_dim`` features.
    ValueError
        If ``len(chr_ids)`` differs from ``atac_input_dim``.

    Examples
    --------
    >>> chr_ids = load_chromosome_ids(peak_list)          # doctest: +SKIP
    >>> with spatial_atac_trunk(chr_ids, 5000):           # doctest: +SKIP
    ...     model, history = train_merged(...)
    """
    assert_trunk_contract(strict=strict_contract)
    if len(chr_ids) != atac_input_dim:
        raise ValueError(
            f"chr_ids has {len(chr_ids)} entries but ATAC has {atac_input_dim} "
            "features; the peak list does not match the matrix")

    original = _vm.ModalityAutoencoder
    rec: dict = {"n_built": 0, "atac_spatial": False, "dims": []}

    def factory(input_dim: int, latent_dim: int, **kw):
        rec["n_built"] += 1
        rec["dims"].append(int(input_dim))
        # The trainer builds RNA first and ATAC second (checked above).
        if rec["n_built"] == 2:
            if int(input_dim) != atac_input_dim:
                raise RuntimeError(
                    f"second trunk got input_dim={input_dim}, expected the ATAC "
                    f"count {atac_input_dim}; construction order is not what "
                    "the patch assumes")
            rec["atac_spatial"] = True
            log.info("ATAC trunk: SpatialModalityAutoencoder(kernels=%s) over "
                     "%d peaks on %d chromosomes",
                     spatial_kernel_sizes, input_dim, len(set(chr_ids.tolist())))
            return SpatialModalityAutoencoder(
                input_dim=input_dim, latent_dim=latent_dim, chr_ids=chr_ids,
                spatial_kernel_sizes=spatial_kernel_sizes,
            )
        log.info("RNA trunk: ModalityAutoencoder over %d features", input_dim)
        return original(input_dim=input_dim, latent_dim=latent_dim, **kw)

    _vm.ModalityAutoencoder = factory
    try:
        yield rec
    finally:
        _vm.ModalityAutoencoder = original

    if rec["n_built"] != 2:
        raise RuntimeError(
            f"expected exactly 2 trunks to be built, saw {rec['n_built']} "
            f"(dims {rec['dims']}). The spatial swap may have hit the wrong "
            "modality -- do not trust this run.")
    if not rec["atac_spatial"]:
        raise RuntimeError("the spatial ATAC trunk was never constructed")
