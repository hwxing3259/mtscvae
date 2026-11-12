"""Fig 2 regrouped by endpoint (``benchmark_dotplot``), drawn strictly from ``grid_values.csv``.
Rendered with matplotlib 3.11.1."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

logger = logging.getLogger("plot_benchmark_dotplot")
HERE = Path(__file__).resolve().parent
STEM = "benchmark_dotplot"
TITLE = "Multimodal developmental integration benchmarks"
#: Panel titles, keyed by the CSV's metric.
METRIC_TITLES = {
    "time_corr": "Time correlation", "nmi": "Cell-type NMI", "batch_entropy": "Batch entropy",
    "lta_time": "LTA: time", "lta_type": "LTA: cell type",
    "rna_rare_sil": "RNA rare silhouette", "atac_rare_sil": "ATAC rare silhouette",
}
#: Dataset colour and marker. The colours pass the categorical palette check with colour-blind
#: separation in the 6-8 band, which needs a second cue: the marker shape and the fixed offset.
DATASET_STYLE = {
    "HC": ("#3b7dc4", "o"), "HO": ("#e07b28", "s"), "DR": ("#1c9a6a", "D"),
    "MN": ("#8e5cc4", "^"), "PFC": ("#e04f8c", "v"),
}
#: Marker area per shape, so the five read as the same size.
MARKER_SIZE = {"o": 34, "s": 28, "D": 22, "^": 42, "v": 42}
OURS = "Ours"
SHADE, GRID, INK = "#e9eef6", "#e3e3e3", "#333333"
HALF_SPREAD = 0.28  # dataset offsets span +/- this around each method


def draw_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, methods: List[str],
               datasets: List[str]) -> Dict[str, PathCollection]:
    """Draw one endpoint: a dot per (method, dataset), our column shaded.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    df : pandas.DataFrame
        The rows of ``grid_values.csv``.
    metric : str
        The endpoint to draw.
    methods, datasets : list of str
        Column order and dot order.

    Returns
    -------
    dict of str to matplotlib.collections.PathCollection
        The dots of each scored dataset.
    """
    sub = df[df.metric == metric]
    offsets = np.linspace(-HALF_SPREAD, HALF_SPREAD, len(datasets))
    ax.axvspan(methods.index(OURS) - 0.5, methods.index(OURS) + 0.5, color=SHADE, lw=0,
               zorder=0)
    dots, missing = {}, []
    for k, ds in enumerate(datasets):
        v = sub[sub.dataset == ds].set_index("method").value.reindex(methods)
        if v.isna().all():
            missing.append(ds)
            continue
        ok = v.notna().values
        colour, marker = DATASET_STYLE[ds]
        dots[ds] = ax.scatter(np.arange(len(methods))[ok] + offsets[k], v.values[ok],
                              s=MARKER_SIZE[marker],
                              c=colour, marker=marker, edgecolors="white", linewidths=0.7,
                              zorder=3, label=ds)
    vals = sub.value.dropna()
    lo = min(0.0, np.floor((vals.min() - 0.02) * 20) / 20)
    ax.set_ylim(lo, min(1.05, vals.max() + 0.08))
    if lo < 0:
        ax.axhline(0, color="#999999", lw=0.6, zorder=1)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Score")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if missing:
        ax.text(0.99, 0.98, f"{', '.join(missing)}: not scored", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#777777")
    return dots


def check(drawn: Dict[str, Dict[str, PathCollection]], df: pd.DataFrame, methods: List[str],
          datasets: List[str], legend_labels: List[str]) -> None:
    """Refuse to write a figure whose dots or legend differ from the CSV.

    Parameters
    ----------
    drawn : dict
        Per metric, the dots of each dataset, as :func:`draw_panel` returns them.
    df : pandas.DataFrame
        The rows of ``grid_values.csv``.
    methods, datasets : list of str
        Column order and dot order.
    legend_labels : list of str
        The labels the legend shows.

    Raises
    ------
    RuntimeError
        On any difference.
    """
    offsets = dict(zip(datasets, np.linspace(-HALF_SPREAD, HALF_SPREAD, len(datasets))))
    n = 0
    for metric, dots in drawn.items():
        for ds, pc in dots.items():
            want = (df[(df.metric == metric) & (df.dataset == ds)].set_index("method")
                    .value.reindex(methods).dropna())
            got = pc.get_offsets()
            x_want = np.array([methods.index(m) for m in want.index]) + offsets[ds]
            if not (np.allclose(got[:, 0], x_want) and np.array_equal(got[:, 1], want.values)):
                raise RuntimeError(f"{metric}/{ds}: drawn dots differ from the CSV")
            n += len(got)
    if n != df.value.notna().sum():
        raise RuntimeError(f"drew {n} dots, the CSV has {df.value.notna().sum()} values")
    if legend_labels != datasets:
        raise RuntimeError(f"legend is {legend_labels}, expected {datasets}")
    logger.info("checks passed: %d dots in %d panels match the CSV", n, len(drawn))


def figure(out_dir: Path) -> None:
    """Draw the figure from ``HERE / grid_values.csv`` and write it as .svg and .png.

    Parameters
    ----------
    out_dir : pathlib.Path
        Folder the figure is written to. ``HERE`` is read at call time, so a caller can point
        the module at another folder holding ``grid_values.csv``.
    """
    df = pd.read_csv(HERE / "grid_values.csv")
    metrics, methods, datasets = (list(dict.fromkeys(df[c])) for c in
                                  ("metric", "method", "dataset"))
    if OURS not in methods:
        raise RuntimeError(f"no {OURS!r} rows in the CSV")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.bbox": "tight",
    })
    n_col = 2
    n_row = -(-(len(metrics) + 1) // n_col)  # one slot for the legend
    fig, axes = plt.subplots(n_row, n_col, figsize=(11.0, 2.75 * n_row), squeeze=False)
    flat = axes.ravel()
    drawn = {}
    for i, metric in enumerate(metrics):
        ax = flat[i]
        drawn[metric] = draw_panel(ax, df, metric, methods, datasets)
        ax.set_title(f"{chr(ord('a') + i)}  {METRIC_TITLES[metric]}", loc="left",
                     fontsize=11, fontweight="bold", color=INK)
    for ax in flat[len(metrics):]:
        ax.axis("off")
    handles = [Line2D([], [], ls="", marker=DATASET_STYLE[d][1], color=DATASET_STYLE[d][0],
                      markeredgecolor="white", markersize=8, label=d) for d in datasets]
    leg = flat[len(metrics)].legend(handles=handles, title="Dataset", loc="center", ncols=2,
                                    columnspacing=2.5, frameon=False, fontsize=11,
                                    title_fontsize=11)
    fig.suptitle(TITLE, fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(h_pad=1.6, w_pad=3.0)
    check(drawn, df, methods, datasets, [t.get_text() for t in leg.get_texts()])
    for ext in ("svg", "png"):
        out = out_dir / f"{STEM}.{ext}"
        fig.savefig(out, dpi=300, format=ext)
        logger.info("wrote %s", out)
    plt.close(fig)


def main() -> int:
    """Write the figure next to this script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    figure(HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
