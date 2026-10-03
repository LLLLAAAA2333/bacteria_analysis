"""Display existing full-vector slopes; no new fit, selection, or smoothing."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def make_figure(out):
    out = Path(out)
    source = out / "tables/neural_slopes.csv"
    slopes = pd.read_csv(source)
    slopes = slopes.assign(discovery_abs=slopes.discovery.abs()).sort_values(
        "discovery_abs", ascending=False, kind="stable")
    assert len(slopes) == 13 and slopes.cell.is_unique
    assert np.isfinite(slopes[["discovery", "holdout"]].to_numpy()).all()
    summary = json.loads((out / "results.json").read_text())
    discovery_color, holdout_color = "#426C7C", "#B97C59"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.labelsize": 11}):
        fig, ax = plt.subplots(figsize=(7.7, 6.5))
        fig.subplots_adjust(left=0.19, right=0.965, bottom=0.17, top=0.80)
        y = np.arange(len(slopes))
        values = slopes[["discovery", "holdout"]].to_numpy()
        bound = float(np.max(np.abs(values))) * 1.16
        ax.set_xlim(-bound, bound)
        ax.set_ylim(len(slopes) - .4, -.7)
        ax.axvline(0, color="#858D92", lw=1.1, zorder=1)
        for position, row in zip(y, slopes.itertuples()):
            ax.plot([row.discovery, row.holdout], [position, position],
                    color="#C7CDCF", lw=2.0, zorder=2)
        ax.scatter(slopes.discovery, y, s=77, facecolor="white", edgecolor=discovery_color,
                   linewidth=1.8, zorder=3)
        ax.scatter(slopes.holdout, y, s=33, color=holdout_color, edgecolor="white",
                   linewidth=.5, zorder=4)
        ax.set_yticks(y, slopes.cell)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.set_xticks([-.15, -.10, -.05, 0, .05, .10, .15],
                      ["−0.15", "−0.10", "−0.05", "0", "+0.05", "+0.10", "+0.15"])
        ax.tick_params(axis="x", length=3, color="#A5ADB0", labelsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#A5ADB0")
        ax.set_xlabel("Unit coefficient change per chemical-score SD", labelpad=11)
        ax.text(.24, 1.025, "Decreases", ha="center", va="bottom", transform=ax.transAxes,
                color="#616B70", fontsize=10)
        ax.text(.76, 1.025, "Increases", ha="center", va="bottom", transform=ax.transAxes,
                color="#616B70", fontsize=10)
        fig.text(.19, .955, "Neural changes with higher chemical levels", ha="left", va="top",
                 fontsize=14)
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=8,
                          markerfacecolor="white", markeredgecolor=discovery_color, markeredgewidth=1.8,
                          label=f"Discovery (n = {summary['n_discovery']})"),
                   Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                          markerfacecolor=holdout_color, markeredgecolor="white",
                          label=f"Held out (n = {summary['n_holdout']})")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.18, .919),
                   frameon=False, ncol=2, columnspacing=2.0, handletextpad=.5, fontsize=10)
        paths = []
        for extension in ["png", "svg"]:
            path = out / f"figures/04_neural_pattern_summary.{extension}"
            fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    caption = (
        "Full neural change-vector summary for the previously selected chemical trio: "
        "Glucaric acid, Lumichrome and Vitamin B1. Each row is one neuron class, with all 13 "
        "classes retained. The x coordinate is the existing fitted change in unit template "
        "coefficient per one discovery SD increase of the fixed chemical score. Positive and "
        "negative values indicate relative coefficient increases and decreases, not excitation "
        "and inhibition. The open discovery point and filled held-out point use the original "
        "saved fits; their connecting segment is not an uncertainty interval. No uncertainty "
        "intervals or significance labels are shown. Rows retain descending absolute discovery "
        "slope order, without using held-out strength or sign agreement to select or order cells. "
        "The graph summarizes fitted trends, not uniform behavior in individual strains. "
        "The retained 03_simple_story_centered heatmap shows all 36 held-out strain profiles "
        "and their exceptions. No measurements, fitted results, score, split, scale or cell "
        "selection were recomputed for this display. The previously seen dataset provides an "
        "internal exploratory comparison, not an independent experiment. Full-vector holdout "
        "prediction improvement remains 2.1%, with uneven within-genus correspondence.\n"
    )
    (out / "figures/neural_pattern_summary_caption.txt").write_text(caption)
    parameters = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "n_neurons": len(slopes), "neuron_order": slopes.cell.tolist(),
                  "order": "descending absolute discovery slope, stable ties",
                  "display": "existing discovery/holdout point estimates; connecting segments are not intervals",
                  "units": "unit coefficient per chemical-score discovery SD", "new_fitting": False}
    (out / "figures/neural_pattern_summary_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    return paths
