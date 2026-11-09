"""PFC benchmark figures, drawn strictly from the CSV files next to this script.

Four figures, each written as ``.svg`` and ``.png`` next to this file:

``fourds_benchmark_grid_trimmed_pfc``
    The approved four-dataset grid with PFC as a fifth column (``grid_values.csv``),
    drawn by ``plot_fourds_benchmark.py`` -- a byte copy of the four-dataset
    deposit's script -- with the canvas widened so every panel keeps its
    four-column size.
``pfc_8k_vs_full``
    Three bars per endpoint: ours on the full training split, ours at the
    matched size, and uniPort -- the only competitor that reaches full -- on the
    full split (``8k_vs_full_values.csv``).
``pfc_scaling_runtime``
    Wall time and peak memory of one fit against cells fitted, measured and
    projected (``runtime_values.csv``, ``runtime_reference_lines.csv``).
``pfc_scaling_endpoints_vs_n``
    The seven endpoints against cells fitted (``endpoints_vs_n_values.csv``).

The CSVs are the only inputs. Their rows fix every order: method order is taken
from ``grid_values.csv``, metric order likewise. This script contributes only
styling (method colours and family colours come from ``plot_fourds_benchmark.py``;
markers are defined here) and refuses to write a figure whose drawn marks differ
from its CSV.

Run from any directory::

    python plot_pfc_benchmark.py

Requires numpy, pandas and matplotlib. The figures were rendered with
matplotlib 3.11.1.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_fourds_benchmark as pf  # noqa: E402  (byte copy of the approved script)

logger = logging.getLogger(__name__)

#: Styling only.  One marker per method, so line identity never rests on colour.
MARKERS: Dict[str, str] = {"Ours": "o", "scMultiNODE": "s", "SCOTv1": "^", "SCOTv2": "v",
                           "Pamona": "D", "UnionCom": "P", "uniPort": "X"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
#: Light tint of ours' colour (#1f4e9c mixed 55 % with white) for the matched bars.
MATCHED_TINT = "#9aafd2"
PFC_DATASET = "PFC"
#: Geometry of the approved four-column grid: 9.6 in wide,
#: subplots_adjust(left=0.13, right=0.995, wspace=0.34).
BASE_W, BASE_LEFT, BASE_RIGHT, WSPACE = 9.6, 0.13, 0.995, 0.34


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def orders(grid: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Method and metric order, and each metric's family, from the grid CSV.

    Parameters
    ----------
    grid : pandas.DataFrame
        ``grid_values.csv``.

    Returns
    -------
    tuple
        ``(methods, metrics, family)`` in order of first appearance.
    """
    methods = list(dict.fromkeys(grid.method))
    metrics = list(dict.fromkeys(grid.metric))
    family = grid.groupby("metric").family.first().to_dict()
    return methods, metrics, family


def reference(lines: pd.DataFrame, name: str) -> Tuple[float, str]:
    """One reference value and its label.

    Parameters
    ----------
    lines : pandas.DataFrame
        ``runtime_reference_lines.csv``.
    name : str
        Row name.

    Returns
    -------
    tuple
        ``(value, label)``.
    """
    r = lines.set_index("name").loc[name]
    return float(r["value"]), ("" if pd.isna(r["label"]) else str(r["label"]))


def _style() -> None:
    """Recessive axes, thin marks, text in ink."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.6, "svg.fonttype": "none", "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, format=ext)
        logger.info("wrote %s", out_dir / f"{stem}.{ext}")
    plt.close(fig)


def _same(a: Sequence[float], b: Sequence[float], what: str) -> None:
    """Refuse a figure whose drawn values differ from the CSV."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
        raise RuntimeError(f"{what}: drawn values differ from the CSV")


# ---------------------------------------------------------------------------
# 1. the grid with PFC as a fifth column
# ---------------------------------------------------------------------------
def grid_figure(out_dir: Path) -> None:
    """Draw the approved grid with the PFC column, by the approved script.

    Parameters
    ----------
    out_dir : pathlib.Path
        Where to write.
    """
    path = HERE / "grid_values.csv"
    df, datasets, metrics, methods = pf.read_values(path)
    fig, axes = pf.draw(df, datasets, metrics, methods)
    n = len(datasets)
    panel = (BASE_RIGHT - BASE_LEFT) * BASE_W / (4 + 3 * WSPACE)
    left_in, right_in = BASE_LEFT * BASE_W, (1 - BASE_RIGHT) * BASE_W
    width = left_in + right_in + panel * (n + (n - 1) * WSPACE)
    fig.set_size_inches(width, fig.get_figheight())
    fig.subplots_adjust(left=left_in / width, right=1 - right_in / width)
    family = df.groupby("metric").family.first().to_dict()
    families = pf.add_family_labels(fig, axes, metrics, family)
    pf.add_legend(fig, axes, methods)
    pf.check(fig, axes, df, datasets, metrics, methods, families)
    _save(fig, out_dir, "fourds_benchmark_grid_trimmed_pfc")


# ---------------------------------------------------------------------------
# 2. ours at the matched size vs on the full split
# ---------------------------------------------------------------------------
def draw_8k_vs_full(ax: plt.Axes, v: pd.DataFrame, metrics: List[str],
                    legend_ncol: int = 2,
                    legend_anchor: Tuple[float, float] = (0.5, -0.30)) -> None:
    """Draw the three bars per endpoint onto a given axes.

    Read left to right within a group: ours on the full training split, ours at
    the matched size (does more data help ours?), uniPort on the full split
    (the only competitor that reaches it -- how does ours compare at full
    scale?).  Plain bars with dark edges on each endpoint's family colour, as in
    the grid.  The matched bar is a light tint of ours' colour.  Call
    :func:`_style` before creating ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    v : pandas.DataFrame
        ``8k_vs_full_values.csv`` indexed by ``metric``.
    metrics : list of str
        Group order; must equal the CSV's row order.
    legend_ncol : int, optional
        Legend columns (2 in the standalone figure).
    legend_anchor : tuple of float, optional
        Legend anchor in axes coordinates, below the axes.
    """
    if list(v.index) != metrics:
        raise RuntimeError("8k_vs_full_values.csv rows are not in the grid's metric order")
    series = [("ours_full_mean", pf.COLOURS["Ours"]), ("ours_matched_mean", MATCHED_TINT),
              ("uniport_full_mean", pf.COLOURS["uniPort"])]
    x = np.arange(len(metrics))
    w = 0.26
    fam = v["family"].tolist()
    for i, f in enumerate(fam):
        ax.axvspan(i - 0.5, i + 0.5, color=pf.FAMILY_COLOUR[f], lw=0, zorder=0)
    for k, (col, colour) in enumerate(series):
        bars = ax.bar(x + (k - 1) * w, v[col], width=w, color=colour,
                      edgecolor="#333333", linewidth=0.5, zorder=3)
        _same([b.get_height() for b in bars], v[col], col)
    # one family label per run of equal families, centred over it
    i = 0
    while i < len(fam):
        j = i
        while j + 1 < len(fam) and fam[j + 1] == fam[i]:
            j += 1
        ax.text((i + j) / 2, 0.985, fam[i].replace(" ", "\n", 1), ha="center", va="top",
                fontsize=6.0, color=INK_2, transform=ax.get_xaxis_transform())
        i = j + 1
    ax.set_xticks(x, metrics, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="white", linewidth=0.7, zorder=1)
    ax.set_ylabel("score (higher is better)")
    mt = str(v["ours_matched_tier"].iloc[0])
    k8 = f"{int(mt[1:]) // 1000}k"
    nf = int(v["uniport_full_n_fits"].iloc[0])
    labels = [f"ours, full training split ({int(v['ours_full_n_seeds'].iloc[0])} seeds)",
              f"ours, matched {k8} + {k8} cells ({int(v['ours_matched_n_seeds'].iloc[0])} seeds)",
              f"uniPort, full training split ({nf} fit{'s' if nf != 1 else ''})"]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#333333", lw=0.5, label=l)
               for (_, c), l in zip(series, labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=legend_anchor,
              ncol=legend_ncol, frameon=False, fontsize=6.6)


def eightk_vs_full_figure(out_dir: Path, metrics: List[str]) -> None:
    """Write the standalone three-bar figure.

    Parameters
    ----------
    out_dir : pathlib.Path
        Where to write.
    metrics : list of str
        Group order (from the grid CSV).
    """
    _style()
    v = pd.read_csv(HERE / "8k_vs_full_values.csv").set_index("metric")
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    draw_8k_vs_full(ax, v, metrics)
    _save(fig, out_dir, "pfc_8k_vs_full")


# ---------------------------------------------------------------------------
# 3. time and memory against cells fitted
# ---------------------------------------------------------------------------
def _cells_ticks(ax: plt.Axes, cells: List[int], labels: List[str]) -> None:
    ax.set_xticks(cells, labels)
    ax.minorticks_off()
    if labels and labels[-1] == "full" and len(cells) > 1:
        ticks = ax.get_xticklabels()
        ticks[-2].set_ha("right")
        ticks[-1].set_ha("left")


def _time_ticks(ax: plt.Axes) -> None:
    lo, hi = ax.get_ylim()
    ticks = [10, 60, 600, 3600, 4 * 3600, 86400, 7 * 86400, 30 * 86400, 365 * 86400]
    labels = ["10 s", "1 min", "10 min", "1 h", "4 h", "1 d", "1 wk", "1 mo", "1 yr"]
    keep = [(t, l) for t, l in zip(ticks, labels) if lo <= t <= hi]
    ax.set_yticks([t for t, _ in keep], [l for _, l in keep])
    ax.minorticks_off()


def _place_end_labels(ax: plt.Axes, items: List[Tuple[float, float, str]]) -> None:
    """Direct-label line ends on a log-log axis without collisions."""
    if not items:
        return
    lo, hi = (math.log10(v) for v in ax.get_ylim())
    height_in = ax.get_position().height * ax.figure.get_figheight()
    sep = (6.5 * 1.25 / 72) / height_in * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    columns: List[List[Tuple[float, float, str]]] = []
    for it in items:
        if columns and it[0] / columns[-1][0][0] < 1.6:
            columns[-1].append(it)
        else:
            columns.append([it])
    for col in columns:
        col.sort(key=lambda t: t[1])
        ys = [math.log10(t[1]) for t in col]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + sep)
        for (x, y, txt), yl in zip(col, ys):
            moved = abs(yl - math.log10(y)) > 1e-6
            ax.annotate(txt, (x, y), xytext=(x * 1.12, 10 ** yl), textcoords="data",
                        va="center", fontsize=6.5, color=INK,
                        arrowprops=(dict(arrowstyle="-", lw=0.4, color=INK_2,
                                         shrinkA=0, shrinkB=2) if moved else None))


def runtime_figure(out_dir: Path, methods: List[str]) -> None:
    """Time and memory of one fit against cells fitted, measured and projected.

    Parameters
    ----------
    out_dir : pathlib.Path
        Where to write.
    methods : list of str
        Method order (from the grid CSV).
    """
    _style()
    rv = pd.read_csv(HERE / "runtime_values.csv")
    lines = pd.read_csv(HERE / "runtime_reference_lines.csv")
    cap, cap_label = reference(lines, "cap_seconds")
    mem_lim, mem_label = reference(lines, "container_memory_limit_gib")
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                     gridspec_kw={"width_ratios": [1.45, 1]})
    ticks = rv.drop_duplicates("cells_fitted").sort_values("cells_fitted")
    cells = ticks.cells_fitted.astype(int).tolist()
    labels = [("full" if t == "full" else f"{c / 1000:.0f}k")
              for t, c in zip(ticks.tier, ticks.cells_fitted)]
    ends: List[Tuple[float, float, str]] = []
    for m in methods:
        f = rv[rv.method == m]
        if f.empty:
            continue
        col, mk = pf.COLOURS[m], MARKERS[m]
        meas = f[f.kind == "measured"].sort_values("cells_fitted")
        proj = f[f.kind == "projected"].sort_values("cells_fitted")
        adop = f[f.kind == "adopted_other_pod"]
        lt = ax_t.plot(meas.cells_fitted, meas.seconds, "-", color=col, marker=mk, ms=4.5,
                       mec="white", mew=0.6, zorder=3)[0]
        lm = ax_m.plot(meas.cells_fitted, meas.peak_rss_gib, "-", color=col, marker=mk,
                       ms=4.5, mec="white", mew=0.6, zorder=3)[0]
        _same(lt.get_ydata(), meas.seconds, f"{m} measured time")
        _same(lm.get_ydata(), meas.peak_rss_gib, f"{m} measured memory")
        for _, r in adop.iterrows():
            ax_t.plot(r.cells_fitted, r.seconds, marker="*", ms=8, mfc="white", mec=col,
                      mew=1.0, ls="none", zorder=4)
        if len(meas) and len(proj):
            x0 = meas.cells_fitted.iloc[-1]
            ax_t.plot([x0, *proj.cells_fitted], [meas.seconds.iloc[-1], *proj.seconds],
                      ls=":", color=col, lw=1.0)
            pt = ax_t.plot(proj.cells_fitted, proj.seconds, marker=mk, ms=4.5, mfc="white",
                           mec=col, mew=0.9, ls="none", zorder=3)[0]
            _same(pt.get_ydata(), proj.seconds, f"{m} projected time")
            pm = proj.dropna(subset=["peak_rss_gib"])
            ax_m.plot([x0, *pm.cells_fitted], [meas.peak_rss_gib.iloc[-1], *pm.peak_rss_gib],
                      ls=":", color=col, lw=1.0)
            ax_m.plot(pm.cells_fitted, pm.peak_rss_gib, marker=mk, ms=4.5, mfc="white",
                      mec=col, mew=0.9, ls="none", zorder=3)
        if len(meas):
            dev = " (GPU)" if (meas.device == "gpu").any() else ""
            end = proj.iloc[-1] if len(proj) else meas.iloc[-1]
            ends.append((float(end.cells_fitted), float(end.seconds), m + dev))
    ax_t.axhline(cap, ls=(0, (1, 2)), color=INK_2, lw=0.8)
    ax_t.text(0.99, cap * 1.12, cap_label, fontsize=6.5, color=INK_2, ha="right",
              va="bottom", transform=ax_t.get_yaxis_transform())
    ax_m.axhline(mem_lim, ls=(0, (1, 2)), color=INK_2, lw=0.8)
    ax_m.text(0.02, mem_lim * 1.08, mem_label, fontsize=6.5, color=INK_2,
              transform=ax_m.get_yaxis_transform())
    for ax in (ax_t, ax_m):
        ax.set_xscale("log")
        ax.set_yscale("log")
        _cells_ticks(ax, cells, labels)
        ax.set_xlabel("cells fitted (RNA + ATAC)")
        ax.set_xlim(cells[0] / 1.3, cells[-1] * 1.3)
    ax_t.set_xlim(cells[0] / 1.3, cells[-1] * 3.2)          # room for end labels
    ymax = max(cap * 2, float(rv.seconds.max()))
    ymin = min(60.0, float(rv.seconds[rv.seconds > 0].min()))
    ax_t.set_ylim(ymin / 2, ymax * 2)
    _time_ticks(ax_t)
    ax_t.set_ylim(ymin / 2, ymax * 2)
    ax_m.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.subplots_adjust(wspace=0.32)
    _place_end_labels(ax_t, ends)
    ax_t.set_ylabel("wall-clock time of one fit")
    ax_m.set_ylabel("peak resident memory (GiB)")
    ax_t.set_title("a  time", loc="left", color=INK)
    ax_m.set_title("b  memory", loc="left", color=INK)
    handles = [plt.Line2D([], [], color=pf.COLOURS[m], marker=MARKERS[m], ls="-", ms=4.5,
                          label=m) for m in methods if (rv.method == m).any()]
    handles += [plt.Line2D([], [], color=INK_2, marker="o", mfc="white", ls=":",
                           label="projected (not run)"),
                plt.Line2D([], [], color=INK_2, marker="*", mfc="white", ms=8, ls="none",
                           label="ours, adopted full-atlas fits (fitted separately)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.12))
    _save(fig, out_dir, "pfc_scaling_runtime")


# ---------------------------------------------------------------------------
# 4. the seven endpoints against cells fitted
# ---------------------------------------------------------------------------
def endpoints_figure(out_dir: Path, methods: List[str], metrics: List[str],
                     n_eval: int) -> None:
    """Small multiples: each endpoint against cells fitted, mean and seed range.

    Parameters
    ----------
    out_dir : pathlib.Path
        Where to write.
    methods, metrics : list of str
        Orders (from the grid CSV).
    n_eval : int
        Evaluation cells per modality, for the axis note.
    """
    _style()
    ev = pd.read_csv(HERE / "endpoints_vs_n_values.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.0))
    axes = axes.ravel()
    cells = sorted(ev.cells_fitted.unique().astype(int))
    for ax, ep in zip(axes, metrics):
        for m in methods:
            g = ev[(ev.method == m) & (ev.metric == ep)].sort_values("cells_fitted")
            g = g.dropna(subset=["mean"])
            if g.empty:
                continue
            ln = ax.plot(g.cells_fitted, g["mean"], color=pf.COLOURS[m], marker=MARKERS[m],
                         ms=3.5, mec="white", mew=0.5, lw=1.2)[0]
            _same(ln.get_ydata(), g["mean"], f"{m} {ep}")
            if (g["max"] > g["min"]).any():
                ax.fill_between(g.cells_fitted, g["min"], g["max"], color=pf.COLOURS[m],
                                alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_title(ep, loc="left", color=INK)
        ax.set_xticks(cells[::2], [f"{c / 1000:.0f}k" for c in cells[::2]])
        ax.minorticks_off()
    axes[-1].axis("off")
    handles = [plt.Line2D([], [], color=pf.COLOURS[m], marker=MARKERS[m], ms=4, ls="-",
                          label=m) for m in methods if (ev.method == m).any()]
    axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=7)
    fig.supxlabel(f"cells fitted (RNA + ATAC); every point scored on the same "
                  f"{n_eval:,} + {n_eval:,} evaluation cells", fontsize=7.5, color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "pfc_scaling_endpoints_vs_n")


def main(argv: Sequence[str] | None = None) -> int:
    """Draw, check and write every figure.

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

    grid = pd.read_csv(HERE / "grid_values.csv")
    methods, metrics, _ = orders(grid)
    lines = pd.read_csv(HERE / "runtime_reference_lines.csv")
    n_eval = int(reference(lines, "n_eval_per_modality")[0])
    grid_figure(args.out_dir)
    eightk_vs_full_figure(args.out_dir, metrics)
    runtime_figure(args.out_dir, methods)
    endpoints_figure(args.out_dir, methods, metrics, n_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
