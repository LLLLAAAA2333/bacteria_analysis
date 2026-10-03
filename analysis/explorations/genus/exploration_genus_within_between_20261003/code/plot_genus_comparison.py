"""Plot saved genus distance summaries without recomputing the analysis.

Each row compares within-genus strain pairs with pairs joining that genus to
other genera. Bars show distribution quantiles, not confidence intervals.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"within": "#276A9B", "between": "#BD793E"}
QUANTILES = ("q10", "q25", "q50", "q75", "q90")


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_plot_data(out):
    summary = _read_csv(out / "tables" / "distribution_summary.csv")
    pairs = _read_csv(out / "tables" / "pair_catalogue.csv")
    lookup = {}
    for row in summary:
        converted = dict(row)
        for field in QUANTILES + ("minimum", "maximum", "mean"):
            converted[field] = float(row[field])
        for field in ("n_strains", "n_pairs"):
            converted[field] = int(float(row[field]))
        key = (row["genus"], row["modality"], row["relation"])
        if key in lookup:
            raise ValueError(f"Duplicate distribution summary: {key}")
        lookup[key] = converted
    genera = sorted(
        {row["genus"] for row in summary},
        key=lambda genus: (-lookup[(genus, "chemical", "within")]["n_strains"], genus),
    )
    within = {genus: {"chemical": [], "neural": []} for genus in genera}
    for row in pairs:
        genus = row["genus_a"]
        if genus == row["genus_b"] and genus in within:
            for modality in ("chemical", "neural"):
                within[genus][modality].append(float(row[f"{modality}_distance"]))
    for genus in genera:
        for modality in ("chemical", "neural"):
            expected = lookup[(genus, modality, "within")]["n_pairs"]
            if len(within[genus][modality]) != expected:
                raise ValueError(f"Within-pair count does not match summary: {genus}, {modality}")
    return genera, lookup, within


def _draw_summary(ax, summary, y, relation):
    color = COLORS[relation]
    if summary["n_pairs"] > 1:
        ax.plot(
            [summary["q10"], summary["q90"]], [y, y],
            color=color, linewidth=1.3, solid_capstyle="butt", zorder=3,
        )
        ax.plot(
            [summary["q25"], summary["q75"]], [y, y],
            color=color, linewidth=5, solid_capstyle="butt", zorder=4,
        )
    ax.scatter(
        [summary["q50"]], [y], s=33, color=color,
        edgecolors="white", linewidths=0.75, zorder=5,
    )


def _draw_panel(ax, genera, lookup, within, modality, jitter):
    visible_max = 0.0
    for index, genus in enumerate(genera):
        points = np.asarray(within[genus][modality], dtype=float)
        y_within = index - 0.14
        if len(points) > 1:
            ax.scatter(
                points, y_within + jitter[genus], color=COLORS["within"],
                s=9, alpha=0.19, linewidths=0, zorder=2,
            )
        for relation, offset in (("within", -0.14), ("between", 0.14)):
            summary = lookup[(genus, modality, relation)]
            _draw_summary(ax, summary, index + offset, relation)
            visible_max = max(visible_max, summary["q90"])
        if points.size:
            visible_max = max(visible_max, float(points.max()))
    ax.set_xlim(0, visible_max * 1.045 if visible_max else 1)
    ax.set_ylim(len(genera) - 0.48, -0.55)
    ax.set_yticks(np.arange(len(genera)))
    ax.grid(axis="x", color="#E7EAED", linewidth=0.65, zorder=0)
    for index in range(len(genera) - 1):
        ax.axhline(index + 0.5, color="#EFF1F3", linewidth=0.65, zorder=0)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", labelsize=10, color="#A7ADB4")
    for name in ("top", "right", "left"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#A7ADB4")


def make_figure(out):
    """Read ``out/tables`` and return the two saved figure paths.

    Horizontal ranges cover every within-genus pair and all displayed
    between-genus quantiles. The two modalities retain their own units.
    """
    out = Path(out)
    genera, lookup, within = _load_plot_data(out)
    rng = np.random.default_rng(20261003)
    jitter = {
        genus: rng.uniform(-0.065, 0.065, len(within[genus]["chemical"]))
        for genus in genera
    }
    labels = []
    for genus in genera:
        row = lookup[(genus, "chemical", "within")]
        labels.append(f"{genus}  (n={row['n_strains']}, pairs={row['n_pairs']})")
    style = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 13,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(15.3, max(7.8, len(genera) * 0.52 + 1.55)), sharey=True)
        fig.subplots_adjust(left=0.285, right=0.98, bottom=0.11, top=0.855, wspace=0.12)
        for ax, modality in zip(axes, ("chemical", "neural")):
            _draw_panel(ax, genera, lookup, within, modality, jitter)
        axes[0].set_yticklabels(labels, fontsize=10.5)
        axes[0].set_title("Chemical profiles", pad=13, fontweight="semibold")
        axes[1].set_title("Neural composition", pad=13, fontweight="semibold")
        axes[0].set_xlabel("RMS log2 concentration difference", labelpad=10)
        axes[1].set_xlabel("1 − cosine similarity", labelpad=10)
        fig.suptitle("Within- and between-genus distances", fontsize=18, x=0.62, y=0.976)
        handles = [
            Line2D([0], [0], color=COLORS[relation], linewidth=4, marker="o",
                   markersize=6, markeredgecolor="white", markeredgewidth=0.75, label=label)
            for relation, label in (("within", "Within genus"), ("between", "Between genera"))
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 0.942),
                   ncol=2, frameon=False, handlelength=2.4, columnspacing=2)
        figure_dir = out / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        paths = [figure_dir / f"01_within_between_by_genus.{suffix}" for suffix in ("png", "svg")]
        for path in paths:
            fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
    return paths
