"""Fig 2 alone (``fourds_benchmark_grid_trimmed_pfc``), drawn strictly from ``grid_values.csv``.
Rendered with matplotlib 3.11.1."""
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
