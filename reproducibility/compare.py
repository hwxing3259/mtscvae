"""Compare a refitted model folder with a released one, tensor for tensor.
Exit 0 = all identical, 3 = something differs, 1 = not comparable; there is no tolerance."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

BLOCKS = ("sde_state_dict", "decoder_rna_state_dict", "decoder_atac_state_dict",
          "encoder_state_dict")

log = logging.getLogger("compare")


def model_file(d: Path) -> Path:
    """The checkpoint of a fit folder: ``model.pt``, else ``dynamics.pt``.

    Parameters
    ----------
    d : pathlib.Path

    Returns
    -------
    pathlib.Path
    """
    return d / "model.pt" if (d / "model.pt").exists() else d / "dynamics.pt"


def cmp_arrays(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Bit-identity and largest absolute difference of two arrays.

    Parameters
    ----------
    a, b : numpy.ndarray

    Returns
    -------
    dict
        ``identical`` (None when the shapes differ), ``max_abs`` and ``shape``.
    """
    if a.shape != b.shape:
        return {"identical": None, "max_abs": None, "shape": [list(a.shape), list(b.shape)]}
    same = bool(np.array_equal(a, b))
    d = 0.0 if same else float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return {"identical": same, "max_abs": d, "shape": list(a.shape)}


def compare(a: Path, b: Path) -> Dict[str, Any]:
    """Compare two fit folders.

    Parameters
    ----------
    a, b : pathlib.Path

    Returns
    -------
    dict
        Per-tensor results, a summary and ``verdict`` (with exit code ``rc``).
    """
    A = torch.load(model_file(a), map_location="cpu", weights_only=False)
    B = torch.load(model_file(b), map_location="cpu", weights_only=False)
    res: Dict[str, Any] = {"tensors": {}, "latents": {}}
    broken = False
    for blk in BLOCKS:
        ka, kb = set(A[blk]), set(B[blk])
        if ka != kb:
            broken = True
            res["tensors"][blk] = {"keys_only_a": sorted(ka - kb), "keys_only_b": sorted(kb - ka)}
            continue
        for k in sorted(ka):
            res["tensors"][f"{blk}/{k}"] = cmp_arrays(A[blk][k].numpy(), B[blk][k].numpy())
    res["stage_norm"] = cmp_arrays(np.asarray(A["stage_norm"], np.float32),
                                   np.asarray(B["stage_norm"], np.float32))
    for m in ("rna", "atac"):
        f = f"latents/z_coadapt_{m}.npy"
        res["latents"][m] = cmp_arrays(np.load(a / f), np.load(b / f))
    # Stage A too, when both folders hold it (a full refit writes it).
    if (a / "stage_a.pt").exists() and (b / "stage_a.pt").exists():
        sa = torch.load(a / "stage_a.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        sb = torch.load(b / "stage_a.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        if set(sa) != set(sb):
            broken = True
        for k in sorted(set(sa) & set(sb)):
            res["tensors"][f"stage_a/{k}"] = cmp_arrays(sa[k].numpy(), sb[k].numpy())
        za, zb = np.load(a / "stage_a_latents.npz"), np.load(b / "stage_a_latents.npz")
        for m in ("rna", "atac"):
            res["latents"][f"stage_a_{m}"] = cmp_arrays(za[m], zb[m])
    cells = list(res["tensors"].values()) + list(res["latents"].values()) + [res["stage_norm"]]
    if any(c.get("identical") is None for c in cells):
        broken = True
    n_same = sum(bool(c.get("identical")) for c in cells)
    worst = max((c["max_abs"] for c in cells if c.get("max_abs") is not None), default=0.0)
    rc = 1 if broken else (0 if n_same == len(cells) else 3)
    res["summary"] = {"n_compared": len(cells), "n_identical": n_same, "max_abs": worst}
    res["verdict"] = {0: "BIT-IDENTICAL", 3: "DIFFERS", 1: "NOT COMPARABLE"}[rc]
    res["rc"] = rc
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compare two folders and optionally write the verdict.

    Parameters
    ----------
    argv : sequence of str, optional

    Returns
    -------
    int
        0 identical, 3 differs, 1 not comparable.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    res = compare(args.a, args.b)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))
    s = res["summary"]
    log.info("%s: %d/%d identical, max|d| %.3e", res["verdict"], s["n_identical"],
             s["n_compared"], s["max_abs"])
    return res["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
