"""PFC integration figure, drawn strictly from the CSV files next to this script.

One figure, written as ``.svg`` and ``.png`` next to this file:

``pfc_integration_row``
    (a) one UMAP of the fused latent of both modalities (seed 44), with each
    type's RNA and ATAC centroids joined; (b, c) cosine UMAPs of the raw RNA
    and ATAC encoder inputs, with the medoids of the three types with the
    weakest cross-modal alignment starred.

Inputs are the six CSVs in this folder and nothing else:
``umap_joint_coords.csv``, ``umap_raw_rna_coords.csv``,
``umap_raw_atac_coords.csv`` (the points), ``cell_types.csv`` (type order, which
fixes the palette, and which types are starred), ``umap_markers.csv`` (every
centroid and medoid drawn, in drawing order) and ``umap_settings.csv`` (the
UMAP settings shown in the titles). The script recomputes every marker from the
coordinates and refuses to draw if one differs from its CSV, and refuses if
the starred types are not the lowest mean cosines.

:func:`draw_panels` draws onto three axes it is given, so a composite figure
can place this row beside others; :func:`integration_row_figure` is the
standalone figure.

Run from any directory::

    python plot_pfc_umap.py

Requires numpy, pandas, scipy and matplotlib. The figure was rendered with
matplotlib 3.11.1, and is pixel-identical to the PFC example notebook's
``results/figures/pfc_integration_row.png``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

HERE = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

#: Styling only. The rc settings of the PFC example notebook.
RC = {"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10}
#: Contour levels, as fractions of each modality's own density maximum.
LEVELS = [0.08, 0.25, 0.55]
MOD_LINESTYLE = {"RNA": "-", "ATAC": ":"}
MOD_MARKER = {"RNA": ("o", 130), "ATAC": ("^", 150)}
#: Star-label offsets in points. In (c) the three medoids fall within a few
#: points of one another, so each label takes its own side.
STAR_LABEL_OFFSET = {"ID2": (-30, 7), "VIP": (9, 5), "PV": (9, -13)}


def load(here: Path = HERE) -> Dict[str, pd.DataFrame]:
    """Read the six CSVs.

    Parameters
    ----------
    here : pathlib.Path, optional
        Folder holding them.

    Returns
    -------
    dict of str to pandas.DataFrame
        Keyed ``joint``, ``rna``, ``atac``, ``types``, ``markers``, ``settings``.
    """
    names = {"joint": "umap_joint_coords.csv", "rna": "umap_raw_rna_coords.csv",
             "atac": "umap_raw_atac_coords.csv", "types": "cell_types.csv",
             "markers": "umap_markers.csv", "settings": "umap_settings.csv"}
    return {k: pd.read_csv(here / v) for k, v in names.items()}


def palette(types: pd.DataFrame) -> Dict[str, tuple]:
    """tab20 by rank in the order of ``cell_types.csv``, as in the PFC example."""
    return {t: plt.get_cmap("tab20")(i % 20) for i, t in enumerate(types["cell_type"])}


def medoid(xy: np.ndarray) -> np.ndarray:
    """Return the point with the least summed euclidean distance to the rest.

    Parameters
    ----------
    xy : numpy.ndarray
        ``(n, 2)`` coordinates of one type.

    Returns
    -------
    numpy.ndarray
        ``(2,)`` coordinates of the medoid, one of the rows of ``xy``.
    """
    total = np.concatenate([cdist(xy[i:i + 2000], xy).sum(1)
                            for i in range(0, len(xy), 2000)])
    return xy[np.argmin(total)]


def _same(a: Sequence[float], b: Sequence[float], what: str) -> None:
    """Refuse a figure whose drawn values differ from the CSV."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-9):
        raise RuntimeError(f"{what}: drawn values differ from the CSV")


def check(data: Dict[str, pd.DataFrame]) -> None:
    """Recompute the starred set and every marker from the coordinates.

    Parameters
    ----------
    data : dict of str to pandas.DataFrame
        Output of :func:`load`.

    Raises
    ------
    RuntimeError
        If the starred types are not the lowest mean cosines, or a marker in
        ``umap_markers.csv`` differs from its recomputation.
    """
    types, mk = data["types"], data["markers"]
    starred = types[types["starred"]]
    lowest = types.nsmallest(len(starred), "cosine_mean")
    if set(starred["cell_type"]) != set(lowest["cell_type"]):
        raise RuntimeError("starred types are not the lowest mean cosines")
    src = {"a": data["joint"], "b": data["rna"], "c": data["atac"]}
    for r in mk.itertuples():
        df = src[r.panel]
        sel = df["cell_type"] == r.cell_type
        if r.panel == "a":
            sel &= df["modality"] == r.modality
        xy = df.loc[sel, ["x", "y"]].values
        got = xy.mean(0) if r.kind == "centroid_mean" else medoid(xy)
        _same([*got, len(xy)], [r.x, r.y, r.n_cells],
              f"panel ({r.panel}) {r.modality} {r.cell_type} {r.kind}")


def _density_contour(ax: plt.Axes, xy: np.ndarray, linestyle: str) -> None:
    """Draw one modality's global density contours on a common grid."""
    pad = 0.08
    (x0, x1), (y0, y1) = ((xy[:, i].min(), xy[:, i].max()) for i in (0, 1))
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    xe = np.linspace(x0 - dx, x1 + dx, 240)
    ye = np.linspace(y0 - dy, y1 + dy, 240)
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    hist = gaussian_filter(hist, 2.5)
    hist /= hist.max()
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), hist.T,
               levels=LEVELS, colors="k", linewidths=0.9, alpha=0.75,
               linestyles=linestyle, zorder=4)


def _points(ax: plt.Axes, df: pd.DataFrame, order: List[str], pal: Dict[str, tuple]) -> None:
    for t in order:
        m = df["cell_type"].values == t
        if m.any():
            ax.scatter(df["x"].values[m], df["y"].values[m], s=3, alpha=0.45,
                       linewidths=0, color=pal[t], rasterized=True)


def draw_panels(axes: Sequence[plt.Axes], data: Dict[str, pd.DataFrame],
                letters: str = "abc") -> List[Line2D]:
    """Draw panels (a), (b) and (c) onto three given axes.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Three axes, for (a), (b) and (c).
    data : dict of str to pandas.DataFrame
        Output of :func:`load`, already passed through :func:`check`.
    letters : str, optional
        Panel letters to print, one per axes; the star's legend entry names
        the last two.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Handles for a shared legend: one per cell type present, then the star.
    """
    types, mk = data["types"], data["markers"]
    order, pal = list(types["cell_type"]), palette(types)
    st = data["settings"].set_index("panel")

    ax = axes[0]
    joint = data["joint"]
    _points(ax, joint, order, pal)
    for mod in ("RNA", "ATAC"):
        sel = joint["modality"].values == mod
        _density_contour(ax, joint.loc[sel, ["x", "y"]].values, MOD_LINESTYLE[mod])
    # Connectors join each type's two centroids. Their length is not a measure
    # of alignment; they locate a type's two centroids and do not rank types.
    cen = mk[mk["panel"] == "a"]
    for t in dict.fromkeys(cen["cell_type"]):
        pair = cen[cen["cell_type"] == t].set_index("modality")
        rc, ac = pair.loc["RNA", ["x", "y"]].values, pair.loc["ATAC", ["x", "y"]].values
        ax.plot([rc[0], ac[0]], [rc[1], ac[1]], color="0.2", lw=1.2, zorder=5)
        for mod, xy in (("RNA", rc), ("ATAC", ac)):
            marker, size = MOD_MARKER[mod]
            ax.scatter(*xy, marker=marker, s=size, color=pal[t], edgecolors="k",
                       linewidths=1.0, zorder=6)
    ax.set_title(rf"fused latent $\mathbf{{z}}_{{\mathrm{{fus}}}}$ "
                 rf"(both modalities, seed {int(st.loc['a', 'fit_seed'])}, "
                 rf"n_neighbors = {int(st.loc['a', 'n_neighbors'])})")
    ax.legend(handles=[Line2D([], [], color="k", ls="-", label="RNA  (solid contour, ● centroid)"),
                       Line2D([], [], color="k", ls=":", label="ATAC (dotted contour, ▲ centroid)")],
              loc="lower right", fontsize=7.5, framealpha=0.9)

    for ax, key, panel, name in ((axes[1], "rna", "b", "raw RNA"),
                                 (axes[2], "atac", "c", "raw ATAC")):
        _points(ax, data[key], order, pal)
        for r in mk[mk["panel"] == panel].itertuples():
            ax.scatter(r.x, r.y, marker="*", s=320, color=pal[r.cell_type],
                       edgecolors="k", linewidths=1.2, zorder=6)
            ax.annotate(r.cell_type, (r.x, r.y), xytext=STAR_LABEL_OFFSET[r.cell_type],
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
        ax.set_title(f"{name} (encoder input, {st.loc[panel, 'metric']} UMAP)")

    for ax, letter in zip(axes, letters):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="left")

    present = [t for t in order if t in set(joint["cell_type"])]
    star = Line2D([], [], marker="*", ls="", markersize=11, markerfacecolor="w",
                  markeredgecolor="k", label=f"medoid, ({letters[1]}) and ({letters[2]})")
    return [Line2D([], [], marker="o", ls="", markersize=6, color=pal[t], label=t)
            for t in present] + [star]


def integration_row_figure(out_dir: Path, data: Dict[str, pd.DataFrame]) -> None:
    """Draw and write the standalone 1 x 3 figure.

    Parameters
    ----------
    out_dir : pathlib.Path
        Destination folder.
    data : dict of str to pandas.DataFrame
        Output of :func:`load`, already passed through :func:`check`.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))
        handles = draw_panels(axes, data)
        fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "png"):
            fig.savefig(out_dir / f"pfc_integration_row.{ext}", dpi=300, format=ext)
            logger.info("wrote %s", out_dir / f"pfc_integration_row.{ext}")
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Check and draw the figure.

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
    data = load()
    check(data)
    logger.info("checked %d markers against the coordinates", len(data["markers"]))
    integration_row_figure(args.out_dir, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
