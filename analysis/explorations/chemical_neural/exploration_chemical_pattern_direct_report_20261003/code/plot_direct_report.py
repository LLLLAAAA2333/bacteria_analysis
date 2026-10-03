"""Display supplied direct-workbook discovery and held-out results.

Call ``make_figures(out)`` after the new analysis tables have been saved.
This module does not choose candidates, alter measurements, or fit models.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


DISCOVERY = "#426C7C"
HOLDOUT = "#B97C59"


def _save(fig: plt.Figure, path: Path) -> list[Path]:
    paths = [path.with_suffix(".png"), path.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low = min(0.0, float(finite.min())) if finite.size else -1.0
    high = max(0.0, float(finite.max())) if finite.size else 1.0
    pad = 0.08 * (high - low or 1.0)
    return low - pad, high + pad


def _symmetric_max(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(finite.max()), 0.001) if finite.size else 1.0


def _neat_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, color="#A7ADB1")
    ax.spines[["bottom", "left"]].set_color("#A7ADB1")


def make_figures(out: Path) -> list[Path]:
    """Render two figures plus captions from the direct-workbook output tables.

    Scatter values use the saved frozen discovery definitions without further
    centering. Chemistry heatmaps use the saved global discovery z scores;
    neural heatmaps use observed unit-vector coefficients. Both heatmap color
    scales span all observed values, and no display clustering is performed.
    """
    out = Path(out)
    tables, figures = out / "tables", out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(tables / "sample_scores.csv", dtype={"strain": str, "genus": str})
    members = pd.read_csv(tables / "selected_members.csv")
    slopes = pd.read_csv(tables / "neural_slopes.csv")
    chemistry = pd.read_csv(tables / "selected_chemical_standardized.csv", index_col=0)
    neural = pd.read_csv(tables / "neural_unit_coefficients.csv", index_col=0)
    chemistry.index = chemistry.index.astype(str)
    neural.index = neural.index.astype(str)
    summary = json.loads((out / "results.json").read_text())
    paths = []

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "svg.fonttype": "none", "axes.titlepad": 13,
    }):
        fig, axs = plt.subplots(1, 3, figsize=(14.4, 5.4),
                                gridspec_kw={"width_ratios": [1, 1, 1.08]})
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.87, wspace=0.36)
        xlim = _limits(scores["chemical_score"].to_numpy(float))
        ylim = _limits(scores["neural_projection"].to_numpy(float))
        for ax, split, color, panel in zip(axs[:2], ["discovery", "holdout"],
                                           [DISCOVERY, HOLDOUT], ["A", "B"]):
            subset = scores.loc[scores["split"].eq(split)]
            ax.scatter(subset["chemical_score"], subset["neural_projection"],
                       s=40, color=color, alpha=0.84, edgecolor="white", linewidth=0.6)
            ax.axhline(0, color="#D9DDDF", linewidth=0.8, zorder=0)
            ax.axvline(0, color="#D9DDDF", linewidth=0.8, zorder=0)
            ax.set(xlim=xlim, ylim=ylim, xlabel="Chemical score (discovery SD)",
                   ylabel="Neural projection",
                   title=f"{'Discovery' if split == 'discovery' else 'Held out'}  ·  n = {len(subset)}")
            ax.text(-0.16, 1.055, panel, transform=ax.transAxes, fontweight="bold", fontsize=13)
            _neat_axis(ax)
        axs[1].set_ylabel("")
        ax = axs[2]
        y = np.arange(len(slopes))
        ax.barh(y - 0.19, slopes["discovery"], height=0.35, color=DISCOVERY, label="Discovery")
        ax.barh(y + 0.19, slopes["holdout"], height=0.35, color=HOLDOUT, label="Held out")
        ax.set_yticks(y, slopes["cell"].astype(str))
        ax.invert_yaxis()
        ax.axvline(0, color="#8F999D", linewidth=0.8)
        ax.set(title="Neural combination", xlabel="Unit coefficient change / score SD")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.text(-0.16, 1.055, "C", transform=ax.transAxes, fontweight="bold", fontsize=13)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=2, fontsize=9)
        _neat_axis(ax)
        paths.extend(_save(fig, figures / "01_candidate_correspondence"))

        held = scores.loc[scores["split"].eq("holdout")].sort_values(["chemical_score", "strain"])
        ids = held["strain"].tolist()
        feature_columns = members["metabolite"].astype(str).tolist()
        chemical_values = chemistry.loc[ids, feature_columns].to_numpy(float)
        cells = slopes["cell"].astype(str).tolist()
        neural_values = neural.loc[ids, cells].to_numpy(float)
        chem_width = max(4.0, len(feature_columns) * 0.37)
        neural_width = max(4.3, len(cells) * 0.32)
        fig_height = max(6.0, len(ids) * 0.205 + 3.1)
        fig = plt.figure(figsize=(chem_width + neural_width + 3.3, fig_height))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.026],
                              width_ratios=[chem_width, neural_width], hspace=0.02, wspace=0.055)
        axc, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axcb, axnb = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        fig.subplots_adjust(left=0.205, right=0.985, top=0.94, bottom=0.25)
        chem_max = _symmetric_max(chemical_values)
        neural_max = _symmetric_max(neural_values)
        cm = plt.get_cmap("RdBu_r").copy()
        cm.set_bad("#D9DDDF")
        imc = axc.imshow(chemical_values, aspect="auto", interpolation="nearest",
                         cmap=cm, vmin=-chem_max, vmax=chem_max)
        imn = axn.imshow(neural_values, aspect="auto", interpolation="nearest",
                         cmap=cm, vmin=-neural_max, vmax=neural_max)
        labels = [f"{row.strain}  ·  {row.genus}" for row in held.itertuples()]
        axc.set_yticks(np.arange(len(ids)), labels, fontsize=8)
        axn.set_yticks([])
        feature_labels = ["\n".join(textwrap.wrap(name, width=22, break_long_words=False))
                          for name in feature_columns]
        axc.set_xticks(np.arange(len(feature_columns)), feature_labels,
                       rotation=60, ha="right", rotation_mode="anchor", fontsize=8)
        axn.set_xticks(np.arange(len(cells)), cells,
                       rotation=60, ha="right", rotation_mode="anchor", fontsize=8)
        for ax in (axc, axn):
            ax.tick_params(axis="both", length=0, pad=5)
            ax.tick_params(axis="x", pad=40)
            ax.spines[:].set_visible(False)
        cbc = fig.colorbar(imc, cax=axcb, orientation="horizontal", ticks=[-chem_max, 0, chem_max])
        cbn = fig.colorbar(imn, cax=axnb, orientation="horizontal", ticks=[-neural_max, 0, neural_max])
        for cb in (cbc, cbn):
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=2, labelsize=7, pad=2)
            cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        axc.set_title("Selected chemistry  ·  discovery z score", loc="left", pad=14)
        axn.set_title("Neural response  ·  unit coefficients", loc="left", pad=14)
        paths.extend(_save(fig, figures / "02_holdout_profiles"))

    captions = (
        "Figure 1. Discovery and held-out correspondence for the selected chemical module "
        f"({summary.get('selected_module', 'see results.json')}). Chemistry was rebuilt directly "
        "from the raw workbook (reported ng/mL, log2 transformed), with fresh quality control "
        "and no imputation. These independently cultured materials do not measure the "
        "concentrations delivered in the neural stimulus. "
        "Chemical candidates were constructed using discovery chemistry only; the selected "
        "candidate and neural projection direction were then determined in discovery data. "
        "The current cached neural templates were reused. Each point is one strain. Both "
        "scatter panels use the frozen discovery score and projection definitions and share "
        "axes, with no further centering for display. Bars show the supplied slopes for all "
        "13 neuron classes, in unit coefficient change per chemical score SD. No uncertainty "
        "interval is shown; inferential results and the selection rule are provided in the "
        "report. This is exploratory reuse of a previously seen dataset, not an independent "
        "experiment. Correspondence does not establish causation.\n\n"
        "Figure 2. Supporting profiles of every held-out strain, ordered by chemical score. "
        "Chemistry columns include every member of the selected module; neural columns include "
        "all 13 neuron classes. Chemistry values are global z scores calculated using discovery "
        "means and standard deviations. Neural values are observed unit-vector coefficients. "
        "Both matrices use the same strain order. Their color scales have different units and "
        "span the complete observed ranges without clipping. Missing values, if present, are "
        "gray. No additional centering or clustering is applied for display. The heatmaps are "
        "supporting inspection material, not an additional validation test.\n"
    )
    caption_path = figures / "captions.txt"
    caption_path.write_text(captions, encoding="utf-8")
    paths.append(caption_path)
    return paths
