"""PFC held-out-donor figure, drawn strictly from the CSV files next to this script.

One figure, written as ``.svg`` and ``.png`` next to this file:

``pfc_astro_maturation``
    Eight astrocyte developmental marker genes, four per row (immature
    programme, then mature programme), across the six developmental stages.
    Blue is the model on training donors and red on withheld donors; solid is
    astrocytes (mean over the seeds, band +-1 s.e.m. across seeds), dashed
    is the non-astrocyte control propagated through the same pipeline, dotted
    is the observed astrocyte curve.

Inputs are ``pfc_astro_panels.csv`` and ``pfc_astro_margins.csv`` and nothing
else. Their rows fix every order: genes and programmes in order of first
appearance, stages by ``stage_idx``, seeds sorted. The script refuses to draw
if a gene's observed curve differs across seeds, since the seeds are then not
on a common reference and must not be averaged.

:func:`draw_panels` draws onto a 2 x 4 axes array it is given, so a composite
figure can place this block beside others; :func:`astro_figure` is the
standalone figure.

Run from any directory::

    python plot_pfc_astro.py

Requires numpy, pandas and matplotlib. The figure was rendered with matplotlib
3.11.1, and is pixel-identical to the PFC example notebook's
``results/figures/pfc_astro_maturation.png``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

#: Styling only. The rc settings of the PFC example notebook.
RC = {"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10}
STYLE = {"train": ("#1b6ca8", "train"), "heldout": ("#c0392b", "held-out")}
ARROW = {"down": "↓", "up": "↑"}
#: Row 1 (immature) fails its lineage-blind null and is context only; row 2
#: (mature) is the claim. The two rows are never pooled.
ROW_NOTE = "row 1 (immature) is context only; row 2 (mature) is the claim"


def load(here: Path = HERE) -> Dict[str, pd.DataFrame]:
    """Read the two CSVs.

    Parameters
    ----------
    here : pathlib.Path, optional
        Folder holding them.

    Returns
    -------
    dict of str to pandas.DataFrame
        Keyed ``panels`` and ``margins``.
    """
    return {"panels": pd.read_csv(here / "pfc_astro_panels.csv"),
            "margins": pd.read_csv(here / "pfc_astro_margins.csv")}


def layout(panels: pd.DataFrame) -> Tuple[List[Tuple[str, List[str], str]], List[str], List[int]]:
    """Programmes, stages and seeds, in the CSV's order.

    Parameters
    ----------
    panels : pandas.DataFrame
        ``pfc_astro_panels.csv``.

    Returns
    -------
    tuple
        ``(rows, stages, seeds)``: one ``(programme, genes, arrow)`` per row,
        the stage labels by ``stage_idx``, and the sorted seeds.
    """
    rows = []
    for prog, sub in panels.groupby("program", sort=False):
        direction = sub["expected_dir"].unique()
        if len(direction) != 1:
            raise RuntimeError(f"{prog}: genes disagree on the expected direction")
        rows.append((prog, list(dict.fromkeys(sub["gene"])), ARROW[direction[0]]))
    stages = (panels.drop_duplicates("stage_idx").sort_values("stage_idx")["stage"].tolist())
    seeds = sorted(panels["seed"].unique())
    return rows, stages, seeds


def curves(panels: pd.DataFrame, gene: str, regime: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack one gene's per-seed curves for one regime.

    Parameters
    ----------
    panels : pandas.DataFrame
        ``pfc_astro_panels.csv``.
    gene : str
        Gene symbol.
    regime : {"train", "heldout"}
        Training donors or withheld donors.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cognate, control, observed)``; the first two are ``(n_seeds, 6)``,
        observed is ``(6,)``.

    Raises
    ------
    ValueError
        If the observed curve is not identical across seeds.
    """
    sub = panels[(panels["gene"] == gene) & (panels["regime"] == regime)]
    sub = sub.sort_values(["seed", "stage_idx"])
    cog = np.stack([g["model_cognate_mean"].values for _, g in sub.groupby("seed")])
    ctl = np.stack([g["model_control_mean"].values for _, g in sub.groupby("seed")])
    obs = np.stack([g["obs_within"].values for _, g in sub.groupby("seed")])
    if not np.allclose(obs, obs[0]):
        raise ValueError(f"{gene}/{regime}: observed curve differs across seeds")
    return cog, ctl, obs[0]


def draw_panels(axes: np.ndarray, data: Dict[str, pd.DataFrame],
                letters: str = "") -> Tuple[list, List[str]]:
    """Draw the eight gene panels onto a 2 x 4 axes array.

    Parameters
    ----------
    axes : numpy.ndarray
        ``(2, 4)`` axes, rows sharing x.
    data : dict of str to pandas.DataFrame
        Output of :func:`load`.
    letters : str, optional
        One panel letter per row, printed on the row's first panel; none in the
        standalone figure.

    Returns
    -------
    tuple
        Legend handles and labels, from the first panel of row 2.
    """
    panels = data["panels"]
    rows, stages, seeds = layout(panels)
    # Band is the standard error of the across-seed mean, sd / sqrt(n), with n
    # the number of SEEDS: each per-seed curve is already a mean over the SDE
    # rollouts, which share a fitted model and are not replicates of seeds.
    band_n = len(seeds)
    x = np.arange(len(stages))
    for row, (_, genes, arrow) in enumerate(rows):
        for col, gene in enumerate(genes):
            ax = axes[row, col]
            for regime in ("train", "heldout"):
                cog, ctl, obs = curves(panels, gene, regime)
                colour, label = STYLE[regime]
                mean, sd = cog.mean(0), cog.std(0, ddof=1)
                band = sd / np.sqrt(band_n)
                ax.plot(x, mean, "-o", color=colour, lw=2, ms=4, label=f"{label} astrocytes")
                ax.fill_between(x, mean - band, mean + band, color=colour, alpha=0.18, lw=0)
                ax.plot(x, ctl.mean(0), "--", color=colour, lw=1.2, alpha=0.75,
                        label=f"{label} control")
                if regime == "train":
                    ax.plot(x, obs, ":", color="0.35", lw=1.6, label="observed astrocytes")
            ax.set_xticks(x)
            ax.set_xticklabels(stages, fontsize=7, rotation=45)
            ax.set_title(f"{gene}  ({arrow})")
            ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
            ax.tick_params(labelsize=7)
    for row, (prog, _, _) in enumerate(rows):
        axes[row, 0].set_ylabel(f"{prog} programme\n(z, global per gene)", fontsize=8)
    for row, letter in zip(range(len(rows)), letters):
        axes[row, 0].text(-0.02, 1.02, f"({letter})", transform=axes[row, 0].transAxes,
                          fontsize=13, fontweight="bold", va="bottom", ha="left")
    return axes[1, 0].get_legend_handles_labels()


def titles(data: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    """The headline (seeds and rising margins) and the band note.

    Parameters
    ----------
    data : dict of str to pandas.DataFrame
        Output of :func:`load`.

    Returns
    -------
    tuple of str
        ``(headline, note)``.
    """
    _, _, seeds = layout(data["panels"])
    m = data["margins"]
    rise = m[m["quantity"] == "rising_margin"]
    mt = rise[rise["regime"] == "train"]["value"]
    mh = rise[rise["regime"] == "heldout"]["value"]
    headline = (f"astrocyte maturation, mean of seeds {', '.join(map(str, seeds))}"
                f"     rising margin  train {mt.mean():+.3f} ± {mt.std(ddof=1):.3f}"
                f"   held-out {mh.mean():+.3f} ± {mh.std(ddof=1):.3f}")
    note = (f"band = ±1 s.e.m. across the {len(seeds)} seeds (sd/√{len(seeds)}), "
            f"not across cells; margins above are ± sd   |   {ROW_NOTE}")
    return headline, note


def astro_figure(out_dir: Path, data: Dict[str, pd.DataFrame]) -> None:
    """Draw and write the standalone 2 x 4 figure.

    Parameters
    ----------
    out_dir : pathlib.Path
        Destination folder.
    data : dict of str to pandas.DataFrame
        Output of :func:`load`.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), sharex=True)
        h, l = draw_panels(axes, data)
        headline, note = titles(data)
        fig.suptitle(headline, y=0.995)
        fig.text(0.5, 0.945, note, ha="center", fontsize=8, color="0.35")
        fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.04, 1, 0.925))
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "png"):
            fig.savefig(out_dir / f"pfc_astro_maturation.{ext}", dpi=300, format=ext)
            logger.info("wrote %s", out_dir / f"pfc_astro_maturation.{ext}")
        plt.close(fig)


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
    astro_figure(args.out_dir, load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
