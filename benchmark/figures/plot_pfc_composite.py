"""PFC composite figure, drawn strictly from the CSV files next to this script.
Rendered with matplotlib 3.11.1; Fig 3 uses ``--control-cohort rna_only``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_pfc_astro as pa  # noqa: E402  (byte copy of astro_mixed_fig/, = astro_fig/)
import plot_pfc_benchmark as pb  # noqa: E402  (byte copy of bench_fig/)
import plot_pfc_umap as pu  # noqa: E402  (byte copy of umap_fig/)

logger = logging.getLogger(__name__)

#: Layout only.  Inches; the width ratio puts the bars in one column of four.
FIGSIZE = (16.0, 12.6)
ROW_HEIGHTS = [1.0, 1.45]
#: Width of (a) against the three UMAPs; (a) needs more than a quarter once its
#: text is scaled up.
TOP_WIDTHS = [1.4, 3.0]
#: Every text in the figure is drawn at this multiple of its panel script's size.
TEXT_SCALE = 1.4
LETTER = dict(fontsize=13, fontweight="bold", va="bottom", ha="left")

# Composite labels.
BAR_YLABEL = "Scores"
#: Prefixes of the standalone bar legend, in bar order, that the labels replace.
BAR_LEGEND_FROM = ("ours, full", "ours, matched", "uniPort, full")
UMAP_TITLES = ("UMAP-latent", "UMAP-RNA data", "UMAP-ATAC data")
STAR_LABEL = "centroid"
ASTRO_TITLE = "astrocyte maturation"
ASTRO_YLABEL = "expression level"
ASTRO_LEGEND_ORDER = ("train astrocytes", "held-out astrocytes", "observed astrocytes",
                      "train control", "held-out control")
#: Seeds averaged in (e, f).
ASTRO_SEEDS = (22, 44, 88)
#: Where (e, f)'s control curves come from: the mixed-modality cohort of the
#: astrocyte curves (figure of record), or the RNA-only cohort of astro_fig/
#: (``pfc_astro_panels.csv``; the pfc_composite_rnactl variant).
CONTROL_COHORTS = ("mixed", "rna_only")
CONTROL_COLS = ("model_control_mean", "model_control_sd")


def rna_only_controls(panels: pd.DataFrame) -> pd.DataFrame:
    """Replace the control curves of ``panels`` with the RNA-only cohort's.

    Parameters
    ----------
    panels : pandas.DataFrame
        Mixed-cohort rows of ``astro_mixed_panels.csv``.

    Returns
    -------
    pandas.DataFrame
        The same rows, with :data:`CONTROL_COLS` taken from
        ``pfc_astro_panels.csv``; the astrocyte and observed curves untouched.

    Raises
    ------
    RuntimeError
        If a row has no RNA-only counterpart, or the two CSVs disagree on the
        observed curve (they would then not share a reference).
    """
    key = ["seed", "gene", "regime", "stage_idx"]
    rna = pd.read_csv(HERE / "pfc_astro_panels.csv")
    m = panels.merge(rna[key + ["obs_within", *CONTROL_COLS]], on=key, how="left",
                     suffixes=("", "_rna"), validate="one_to_one")
    if m["obs_within_rna"].isna().any():
        raise RuntimeError("RNA-only panels do not cover every mixed-cohort row")
    if not (m["obs_within"] == m["obs_within_rna"]).all():
        raise RuntimeError("the two cohorts' CSVs disagree on the observed curve")
    for c in CONTROL_COLS:
        m[c] = m[f"{c}_rna"]
    return m[panels.columns]
#: Family labels of (a) shortened to fit the panel.
FAMILY_SHORT = {"temporal\nconsistency": "temporal\ncons."}
#: Panel (b)'s modality key, shortened so it covers less of the panel, and moved
#: to the top left, which the embedding leaves empty.
MODALITY_KEY = {"RNA": "RNA (solid, ●)", "ATAC": "ATAC (dotted, ▲)"}
MODALITY_KEY_LOC = "upper left"
#: x of the UMAP panel letters, in axes fraction (the panel script uses -0.02);
#: moved left so a letter clears its scaled-up title.
UMAP_LETTER_X = -0.08
#: Star labels in (c, d): (dx, dy) in points, horizontal and vertical alignment.
#: In (c) PV and ID2 sit close, so PV takes its left and ID2 the space below;
#: VIP sits near the right edge, so its label goes above.
STAR_LABEL_PLACE = {
    1: {"PV": (-10, 0, "right", "center"), "ID2": (0, -12, "center", "top"),
        "VIP": (0, 12, "center", "bottom")},
    2: {"ID2": (-10, 2, "right", "center"), "VIP": (10, 4, "left", "center"),
        "PV": (10, -12, "left", "center")},
}


def scale_text(fig: plt.Figure, k: float) -> None:
    """Multiply the size of every text in the figure by ``k``, once each.

    Tick labels are scaled through each axis's tick parameters, so ticks that
    matplotlib creates at draw time get the new size too; every other text is
    scaled directly.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The drawn figure, subfigures included.
    k : float
        Scale factor.
    """
    done = set()
    for ax in fig.findobj(Axes):
        for axis in (ax.xaxis, ax.yaxis):
            labels = axis.get_ticklabels()
            if labels:
                axis.set_tick_params(labelsize=labels[0].get_fontsize() * k)
                done.update(id(t) for t in labels)
    for t in fig.findobj(Text):
        if id(t) not in done:
            t.set_fontsize(t.get_fontsize() * k)
            done.add(id(t))


def bar_labels(bars: pd.DataFrame) -> List[str]:
    """Legend labels for (a), in bar order; the matched size is read from the CSV.

    Parameters
    ----------
    bars : pandas.DataFrame
        ``8k_vs_full_values.csv``.

    Returns
    -------
    list of str
        ``ours_full``, ``ours_<k>`` and ``uniPort_full``.
    """
    k = int(str(bars["ours_matched_tier"].iloc[0]).lstrip("n")) // 1000
    return ["ours_full", f"ours_{k}k", "uniPort_full"]


def composite_figure(out_dir: Path, astro_seeds: Sequence[int] = ASTRO_SEEDS,
                     stem: str = "pfc_composite", control_cohort: str = "mixed") -> None:
    """Draw and write the composite figure.

    Parameters
    ----------
    out_dir : pathlib.Path
        Destination folder.
    astro_seeds : sequence of int, optional
        Seeds averaged in (e, f); the default is the figure of record.
    stem : str, optional
        Output file name without extension.
    control_cohort : {"mixed", "rna_only"}, optional
        Cohort of (e, f)'s control curves; see :data:`CONTROL_COHORTS`.
    """
    bars = pd.read_csv(HERE / "8k_vs_full_values.csv").set_index("metric")
    umap = pu.load(HERE)
    pu.check(umap)
    logger.info("checked %d UMAP markers against the coordinates", len(umap["markers"]))
    panels = pd.read_csv(HERE / "astro_mixed_panels.csv")
    margins = pd.read_csv(HERE / "astro_mixed_margins.csv")
    if not set(astro_seeds) <= set(panels["seed"]):
        raise RuntimeError(f"seeds {astro_seeds} are not all in astro_mixed_panels.csv")
    astro = {"panels": panels[panels["seed"].isin(astro_seeds)],
             "margins": margins[margins["seed"].isin(astro_seeds)]}
    if control_cohort == "rna_only":
        astro["panels"] = rna_only_controls(astro["panels"])

    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=ROW_HEIGHTS)
    bar_sf, umap_sf = top.subfigures(1, 2, width_ratios=TOP_WIDTHS)

    # (a) -- the benchmark script's style, scoped to this panel.
    with plt.rc_context():
        pb._style()
        ax = bar_sf.subplots()
        pb.draw_8k_vs_full(ax, bars, list(bars.index))
        ax.set_ylabel(BAR_YLABEL)
        for t in ax.texts:
            if t.get_text() in FAMILY_SHORT:
                t.set_text(FAMILY_SHORT[t.get_text()])
        if set(FAMILY_SHORT.values()) - {t.get_text() for t in ax.texts}:
            raise RuntimeError("a family label to shorten was not found in (a)")
        leg = ax.get_legend()
        old = [t.get_text() for t in leg.get_texts()]
        if not all(o.startswith(f) for o, f in zip(old, BAR_LEGEND_FROM, strict=True)):
            raise RuntimeError(f"unexpected bar legend: {old}")
        handles = leg.legend_handles
        leg.remove()
        ax.legend(handles, bar_labels(bars), loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  ncol=1, frameon=False, fontsize=8, handlelength=1.4)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_rotation_mode("anchor")
        ax.text(-0.02, 1.02, "(a)", transform=ax.transAxes, **LETTER)

    # (b-d)
    with plt.rc_context(pu.RC):
        axes = umap_sf.subplots(1, 3)
        handles = pu.draw_panels(axes, umap, letters="bcd")
        for ax, title in zip(axes, UMAP_TITLES, strict=True):
            ax.set_title(title)
        key = axes[0].get_legend()
        for t in key.get_texts():
            mod = t.get_text().split()[0]
            if mod not in MODALITY_KEY:
                raise RuntimeError(f"unexpected modality key in (b): {t.get_text()!r}")
            t.set_text(MODALITY_KEY[mod])
        key.set_loc(MODALITY_KEY_LOC)
        for ax, letter in zip(axes, "bcd"):
            found = [t for t in ax.texts if t.get_text() == f"({letter})"]
            if len(found) != 1:
                raise RuntimeError(f"panel letter ({letter}) not found")
            found[0].set_x(UMAP_LETTER_X)
        for i, place in STAR_LABEL_PLACE.items():
            anns = {a.get_text(): a for a in axes[i].texts if a.get_text() in place}
            if set(anns) != set(place):
                raise RuntimeError(f"star labels in panel {i}: {sorted(anns)}")
            for name, (dx, dy, ha, va) in place.items():
                anns[name].xyann = (dx, dy)
                anns[name].set_ha(ha)
                anns[name].set_va(va)
        if not handles[-1].get_label().startswith("medoid"):
            raise RuntimeError("the last UMAP legend handle is not the star")
        handles[-1].set_label(STAR_LABEL)
        umap_sf.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=8,
                       frameon=False)

    # (e, f)
    with plt.rc_context(pa.RC):
        axes = bottom.subplots(2, 4, sharex=True)
        h, l = pa.draw_panels(axes, astro, letters="ef")
        for ax in axes[:, 0]:
            ax.set_ylabel(ASTRO_YLABEL, fontsize=8)
        if sorted(l) != sorted(ASTRO_LEGEND_ORDER):
            raise RuntimeError(f"unexpected astrocyte legend: {l}")
        by_label = dict(zip(l, h))
        bottom.suptitle(ASTRO_TITLE, fontweight="bold", fontsize=11)
        bottom.legend([by_label[k] for k in ASTRO_LEGEND_ORDER], list(ASTRO_LEGEND_ORDER),
                      loc="outside lower center", ncol=5, fontsize=8, frameon=False)

    scale_text(fig, TEXT_SCALE)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, format=ext)
        logger.info("wrote %s", out_dir / f"{stem}.{ext}")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the composite figure.

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
    parser.add_argument("--astro-seeds", default=",".join(map(str, ASTRO_SEEDS)),
                        help="comma-separated seeds averaged in (e, f)")
    parser.add_argument("--stem", default="pfc_composite", help="output file name")
    parser.add_argument("--control-cohort", choices=CONTROL_COHORTS, default="mixed",
                        help="cohort of (e, f)'s control curves")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seeds = tuple(int(x) for x in args.astro_seeds.split(","))
    if (seeds != ASTRO_SEEDS or args.control_cohort != "mixed") and args.stem == "pfc_composite":
        parser.error("a variant needs its own --stem; pfc_composite is the figure of record")
    composite_figure(args.out_dir, seeds, args.stem, args.control_cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
