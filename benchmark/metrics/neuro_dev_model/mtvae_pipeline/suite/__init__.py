"""Shared method-layer utilities.

Everything here is dataset-agnostic and, where marked, dependency-light. The
rule that governs the package: **nothing in `suite/` may import torch unless it
has to.** Five scoring modules run in processes that must stay torch-free —
`data/stage_score.py`, `data/snr_prepare_ours.py`, `data/snr_score.py`,
`data/snr_metrics.py` and `pfc/bench/metrics.py` — because on macOS a torch
import alongside scanpy/umap produces a double-libomp clash that kills the
process with exit 139 inside `umap.fuzzy_simplicial_set`. `OMP_NUM_THREADS=1`
and `KMP_DUPLICATE_LIB_OK` do not fix it; `NUMBA_THREADING_LAYER=workqueue`
does, and is numerically inert.

`validation/test_imports.py` enforces this mechanically.
"""
