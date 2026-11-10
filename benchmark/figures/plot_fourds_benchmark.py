"""Four-dataset benchmark figure, drawn strictly from ``fourds_benchmark_values.csv``.
Rendered with matplotlib 3.11.1."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "fourds_benchmark_values.csv"
DEFAULT_STEM = "fourds_benchmark_grid"

#: Styling only. Every method and family in the CSV must have a colour here.
COLOURS: Dict[str, str] = {
    "Ours": "#1f4e9c", "scMultiNODE": "#e07b39", "SCOTv1": "#8aa4c8",
    "SCOTv2": "#5f7fae", "Pamona": "#9c9c9c", "UnionCom": "#bdbdbd",
    "uniPort": "#7ba98a",
}
FAMILY_COLOUR: Dict[str, str] = {
    "temporal consistency": "#f2e8dc", "latent structure": "#e4ecf4",
    "cross-modal transfer": "#e6f0e8", "rare-type resolution": "#f4e9ef",
}
FAMILY_RULE: Dict[str, str] = {
    "temporal consistency": "#c9a227", "latent structure": "#7fa8d0",
    "cross-modal transfer": "#79b58c", "rare-type resolution": "#c98bb0",
}
RULE_WIDTH_PT = 3.6
GAP_LABEL_TO_RULE_PT = 1.0
GAP_RULE_TO_AXES_PT = 3.5


def read_values(path: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Read the values and take every ordering from the file itself.

    Parameters
    ----------
    path : Path
        The values CSV.

    Returns
    -------
    df : pandas.DataFrame
        The rows.
    datasets, metrics, methods : list of str
        Orders of first appearance.

    Raises
    ------
    ValueError
        If the grid is incomplete, a cell is duplicated, a metric spans two
        families, or a method or family has no colour.
    """
    df = pd.read_csv(path)
    datasets = list(dict.fromkeys(df.dataset))
    metrics = list(dict.fromkeys(df.metric))
    methods = list(dict.fromkeys(df.method))
    if len(df) != len(datasets) * len(metrics) * len(methods):
        raise ValueError(f"{path}: expected a full {len(datasets)}x{len(metrics)}x"
                         f"{len(methods)} grid, found {len(df)} rows")
    if df.duplicated(["dataset", "metric", "method"]).any():
        raise ValueError(f"{path}: duplicated (dataset, metric, method) rows")
    if (df.groupby("metric").family.nunique() != 1).any():
        raise ValueError(f"{path}: a metric belongs to two families")
    missing = (set(methods) - set(COLOURS)) | (set(df.family) - set(FAMILY_COLOUR))
    if missing:
        raise ValueError(f"no colour for {sorted(missing)}")
    return df, datasets, metrics, methods


def two_lines(family: str) -> str:
    """Break a family name after its first word, for the rotated label."""
    return family.replace(" ", "\n", 1)


def draw(df: pd.DataFrame, datasets: Sequence[str], metrics: Sequence[str],
         methods: Sequence[str]) -> Tuple[plt.Figure, np.ndarray]:
    """Draw the metrics x datasets grid of bar panels.

    Parameters
    ----------
    df : pandas.DataFrame
        The values.
    datasets, metrics, methods : sequence of str
        Column, row and bar order.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray
        Panel axes, shape ``(len(metrics), len(datasets))``.
    """
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444", "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "savefig.bbox": "tight",
    })
    family = df.groupby("metric").family.first().to_dict()
    n_row, n_col = len(metrics), len(datasets)
    fig, axes = plt.subplots(n_row, n_col, figsize=(9.6, 14.0), squeeze=False)
    x = np.arange(len(methods))
    for i, metric in enumerate(metrics):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            ax.set_facecolor(FAMILY_COLOUR[family[metric]])
            cell = df[(df.dataset == ds) & (df.metric == metric)]
            heights = cell.set_index("method").reindex(list(methods))["value"].to_numpy(float)
            if i == 0:
                ax.set_title(ds, fontsize=11.0, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(metric, fontsize=9, fontweight="bold", labelpad=6)
            if np.isnan(heights).all():
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=10.0,
                        color="#8a8a8a", style="italic", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ("top", "right", "left", "bottom"):
                    ax.spines[side].set_visible(False)
                continue
            ax.bar(x, heights, width=0.76, color=[COLOURS[m] for m in methods],
                   edgecolor="#333333", linewidth=0.5, zorder=3)
            # Every endpoint is bounded by 1, so the top stops just past 1.0.
            lo = min(0.0, float(np.nanmin(heights)))
            hi = float(np.nanmax(heights))
            pad = 0.14 * (hi - lo if hi > lo else 1.0)
            top = min(hi + pad, 1.04) if hi <= 1.0 else hi + pad
            ax.set_ylim(lo - (0.08 * abs(lo) if lo < 0 else 0.0), top)
            if lo < 0:
                ax.axhline(0.0, color="#666666", linewidth=0.6, zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels(list(methods) if i == n_row - 1 else [],
                               rotation=90, fontsize=6.6)
            ax.tick_params(axis="y", labelsize=6.4)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis="y", color="white", linewidth=0.7, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.945, bottom=0.085,
                        wspace=0.34, hspace=0.42)
    # One column of metric names, so each strip sits at the same distance.
    fig.align_ylabels(axes[:, 0])
    return fig, axes


def add_family_labels(fig: plt.Figure, axes: np.ndarray, metrics: Sequence[str],
                      family: Dict[str, str]) -> List[Tuple[plt.Text, Line2D]]:
    """Put each family's label just left of its strip, and the strip by the axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure, already laid out.
    axes : numpy.ndarray
        Panel axes.
    metrics : sequence of str
        Row order.
    family : dict
        Family of each metric.

    Returns
    -------
    list of (Text, Line2D)
        Each family's label and strip.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w = fig.get_figwidth() * fig.dpi
    pt = fig.dpi / 72.0 / fig_w
    axes_left = min(axes[i, 0].get_tightbbox(renderer).x0 for i in range(len(metrics))) / fig_w
    x_rule = axes_left - (GAP_RULE_TO_AXES_PT + RULE_WIDTH_PT / 2) * pt
    x_text = x_rule - (RULE_WIDTH_PT / 2 + GAP_LABEL_TO_RULE_PT) * pt
    blocks: Dict[str, List[int]] = {}
    for i, metric in enumerate(metrics):
        blocks.setdefault(family[metric], []).append(i)
    out = []
    for fam, rows in blocks.items():
        top = axes[rows[0], 0].get_position().y1
        bottom = axes[rows[-1], 0].get_position().y0
        rule = Line2D([x_rule, x_rule], [bottom, top], transform=fig.transFigure,
                      color=FAMILY_RULE[fam], linewidth=RULE_WIDTH_PT,
                      solid_capstyle="butt")
        fig.lines.append(rule)
        # Rotated 90 degrees, the text's baseline faces right, so aligning its
        # right edge to the strip makes the strip read as its underline.
        text = fig.text(x_text, (top + bottom) / 2, two_lines(fam), rotation=90,
                        ha="right", va="center", multialignment="center",
                        fontsize=9.5, fontweight="bold", color="#333333",
                        linespacing=1.25)
        out.append((text, rule))
    return out


def add_legend(fig: plt.Figure, axes: np.ndarray, methods: Sequence[str]) -> None:
    """One row of method colours, centred under the panel grid.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    axes : numpy.ndarray
        Panel axes.
    methods : sequence of str
        Bar order.
    """
    left = axes[0, 0].get_position().x0
    right = axes[0, -1].get_position().x1
    handles = [Patch(facecolor=COLOURS[m], edgecolor="#333333", label=m) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), frameon=False,
               fontsize=7.6, bbox_to_anchor=((left + right) / 2, 0.03),
               handlelength=1.4, columnspacing=1.4)


def check(fig: plt.Figure, axes: np.ndarray, df: pd.DataFrame, datasets: Sequence[str],
          metrics: Sequence[str], methods: Sequence[str],
          families: Sequence[Tuple[plt.Text, Line2D]]) -> None:
    """Refuse to write a figure whose bars differ from the CSV, or whose labels collide.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The finished figure.
    axes : numpy.ndarray
        Panel axes.
    df : pandas.DataFrame
        The values.
    datasets, metrics, methods : sequence of str
        Orders.
    families : sequence of (Text, Line2D)
        Family labels and strips.

    Raises
    ------
    RuntimeError
        On any mismatch with the CSV, any surviving annotation, a legend that is
        not exactly the methods, or a family label touching its strip.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, metric in enumerate(metrics):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            want = (df[(df.dataset == ds) & (df.metric == metric)]
                    .set_index("method").reindex(list(methods))["value"].to_numpy(float))
            bars = list(ax.patches)
            if np.isnan(want).all():
                if bars or [t.get_text() for t in ax.texts] != ["n/a"]:
                    raise RuntimeError(f"{ds} {metric}: an empty panel must say only 'n/a'")
                continue
            got = np.array([b.get_height() for b in bars])
            if len(got) != len(methods) or not np.allclose(got, want, equal_nan=True):
                raise RuntimeError(f"{ds} {metric}: bar heights differ from the CSV")
            if any(b.get_hatch() for b in bars) or ax.containers[1:] or ax.texts:
                raise RuntimeError(f"{ds} {metric}: hatch, error bar or tag present")
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    if labels != list(methods):
        raise RuntimeError(f"legend is {labels}, expected {list(methods)}")
    axes_left = min(axes[i, 0].get_tightbbox(renderer).x0 for i in range(len(metrics)))
    half = RULE_WIDTH_PT / 2 * fig.dpi / 72.0
    for text, rule in families:
        tb = text.get_window_extent(renderer)
        rule_x = fig.transFigure.transform((rule.get_xdata()[0], 0))[0]
        if tb.x1 > rule_x - half or rule_x + half > axes_left:
            raise RuntimeError(f"family label {text.get_text()!r} collides with its strip")
    logger.info("checks passed: %d panels match the CSV; legend of %d", axes.size, len(labels))


def main(argv: Sequence[str] | None = None) -> int:
    """Draw, check and write the figure.

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
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df, datasets, metrics, methods = read_values(args.data)
    logger.info("read %d values from %s", len(df), args.data)
    fig, axes = draw(df, datasets, metrics, methods)
    family = df.groupby("metric").family.first().to_dict()
    families = add_family_labels(fig, axes, metrics, family)
    add_legend(fig, axes, methods)
    check(fig, axes, df, datasets, metrics, methods, families)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = args.out_dir / f"{args.stem}.{ext}"
        fig.savefig(out, dpi=300, format=ext)
        logger.info("wrote %s", out)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
