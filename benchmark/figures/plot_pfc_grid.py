"""``fourds_benchmark_grid_trimmed_pfc`` alone, drawn strictly from the CSV file next to this script.

One figure, written as ``.svg`` and ``.png`` next to this file:

``fourds_benchmark_grid_trimmed_pfc``
    The four-dataset grid with PFC (matched, n8000) as a fifth column
    (``grid_values.csv``): paper Fig 2, with ours' bars from the path-KL 1e-4
    fits (four datasets: seeds 0/1/2; PFC: seeds 22/44/88). Every competitor bar
    is the beta = 0 figure's, except two: the PFC bars are scored on the matched
    tier's 8,000 nuclei, as ours are; HO Pamona and UnionCom are scMultiNODE's
    published latents on all HO cells, scored as every other fit.

It is drawn by :func:`plot_pfc_benchmark.grid_figure` (byte copy of
``bench_fig/plot_pfc_benchmark.py``, which imports its byte copy of
``plot_fourds_benchmark.py``), with that script's style and checks. The CSV is
written by ``neuro_dev_model/pfc_scaling_bench/export_kl1e4_grid.py``.

Run from any directory::

    python plot_pfc_grid.py

Requires numpy, pandas and matplotlib. Rendered with matplotlib 3.11.1.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_pfc_benchmark as pb  # noqa: E402  (byte copy of bench_fig/)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the figure.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pb.grid_figure(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
