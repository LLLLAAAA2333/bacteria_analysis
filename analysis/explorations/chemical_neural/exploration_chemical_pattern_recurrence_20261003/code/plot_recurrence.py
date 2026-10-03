"""Figures for the discovery/holdout chemical recurrence analysis.

Call ``make_figures(output_directory)`` after the analysis tables are saved.
This module only displays the supplied results; it does not select a candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DISCOVERY = "#27647B"
HOLDOUT = "#CB7548"
REFERENCE_COLORS = ["#377E91", "#E29A42", "#9364A7", "#5E8C51", "#C16374", "#686B8C"]


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


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


def _feature_label(row: pd.Series) -> str:
    value = row.get("metabolite")
    if pd.isna(value) or not str(value).strip():
        value = row.get("Mass", row["column"])
    return "\n".join(textwrap.wrap(str(value), width=22, break_long_words=False))


def make_figures(out: Path) -> list[Path]:
    """Render two figures and external captions from the analysis CSV contract.

    Scatter axes are centered separately within each split and reference group.
    Chemistry heatmap values are the saved discovery-standardized values; neural
    heatmap values are the saved unit-vector coefficients. No heatmap recentering
    or clustering is applied here, and both color scales span all observed values.
    """
    out = Path(out)
    tables, figures = out / "tables", out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(tables / "sample_scores.csv", dtype={"strain": str, "reference": str, "genus": str})
    members = pd.read_csv(tables / "selected_members.csv")
    if "metabolite" not in members and members.columns[0].startswith("Unnamed:"):
        members = members.rename(columns={members.columns[0]: "metabolite"})
    slopes = pd.read_csv(tables / "neural_slopes.csv")
    chemistry = pd.read_csv(tables / "selected_chemical_standardized.csv", index_col=0)
    neural = pd.read_csv(tables / "neural_unit_coefficients.csv", index_col=0)
    chemistry.index = chemistry.index.astype(str)
    neural.index = neural.index.astype(str)
    summary = json.loads((out / "results.json").read_text())
    refs = sorted(scores["reference"].dropna().unique())
    colors = {ref: REFERENCE_COLORS[i % len(REFERENCE_COLORS)] for i, ref in enumerate(refs)}
    display_refs = {ref: ref.replace("_", " ") for ref in refs}
    paths = []

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "svg.fonttype": "none", "axes.titlepad": 13,
    }):
        fig, axs = plt.subplots(1, 3, figsize=(14.4, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.08]})
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.21, top=0.86, wspace=0.36)
        xlim = _limits(scores["chemical_within"].to_numpy(float))
        ylim = _limits(scores["projection_within"].to_numpy(float))
        for ax, split, panel in zip(axs[:2], ["discovery", "holdout"], ["A", "B"]):
            subset = scores.loc[scores["split"].eq(split)]
            for ref in refs:
                group = subset.loc[subset["reference"].eq(ref)]
                ax.scatter(group["chemical_within"], group["projection_within"],
                           s=40, color=colors[ref], alpha=0.86, edgecolor="white", linewidth=0.6)
            ax.axhline(0, color="#D9DDDF", linewidth=0.8, zorder=0)
            ax.axvline(0, color="#D9DDDF", linewidth=0.8, zorder=0)
            ax.set(xlim=xlim, ylim=ylim, xlabel="Chemical score (within reference)",
                   ylabel="Neural projection (within reference)",
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
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.text(-0.16, 1.055, "C", transform=ax.transAxes, fontweight="bold", fontsize=13)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=2, fontsize=9)
        _neat_axis(ax)
        handles = [Line2D([], [], marker="o", linestyle="", color=colors[r], markersize=6,
                          label=display_refs[r]) for r in refs]
        fig.legend(handles=handles, title="Reference", frameon=False, loc="lower left",
                   bbox_to_anchor=(0.055, 0.008), ncol=min(4, len(handles)), fontsize=9, title_fontsize=9)
        path = figures / "01_candidate_validation"
        _save(fig, path)
        paths.extend([path.with_suffix(".png"), path.with_suffix(".svg")])

        held = scores.loc[scores["split"].eq("holdout")].sort_values(["reference", "chemical_score", "strain"])
        ids = held["strain"].tolist()
        # The `column` metadata describes C18/HILIC; metabolite identifies features.
        feature_columns = members["metabolite"].astype(str).tolist()
        missing = sorted(set(feature_columns) - set(chemistry.columns))
        if missing:
            raise ValueError(f"Selected chemistry columns missing from matrix: {missing}")
        chemical_values = chemistry.loc[ids, feature_columns].to_numpy(float)
        cells = slopes["cell"].astype(str).tolist()
        neural_values = neural.loc[ids, cells].to_numpy(float)
        n_features = len(feature_columns)
        chem_width = max(4.0, n_features * 0.37)
        neural_width = max(4.3, len(cells) * 0.32)
        fig_height = max(6.0, len(ids) * 0.205 + 3.8)
        fig = plt.figure(figsize=(chem_width + neural_width + 3.3, fig_height))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.026],
                              width_ratios=[chem_width, neural_width], hspace=0.02, wspace=0.055)
        axc, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axcb, axnb = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        fig.subplots_adjust(left=0.205, right=0.985, top=0.90, bottom=0.25)
        chem_max = _symmetric_max(chemical_values)
        neural_max = _symmetric_max(neural_values)
        cm = plt.get_cmap("RdBu_r").copy()
        cm.set_bad("#D9DDDF")
        imc = axc.imshow(chemical_values, aspect="auto", interpolation="nearest", cmap=cm, vmin=-chem_max, vmax=chem_max)
        imn = axn.imshow(neural_values, aspect="auto", interpolation="nearest", cmap=cm, vmin=-neural_max, vmax=neural_max)
        labels = [f"{row.strain}  ·  {row.genus}" for row in held.itertuples()]
        axc.set_yticks(np.arange(len(ids)), labels, fontsize=8)
        axn.set_yticks([])
        axc.set_xticks(np.arange(n_features), [_feature_label(row) for _, row in members.iterrows()],
                       rotation=60, ha="right", rotation_mode="anchor", fontsize=8)
        axn.set_xticks(np.arange(len(cells)), cells, rotation=60, ha="right", rotation_mode="anchor", fontsize=8)
        for ax in (axc, axn):
            ax.tick_params(axis="both", length=0, pad=5)
            ax.spines[:].set_visible(False)
        # Keep color bars above feature labels; labels start below the bars.
        axc.tick_params(axis="x", pad=40)
        axn.tick_params(axis="x", pad=40)
        cbc = fig.colorbar(imc, cax=axcb, orientation="horizontal", ticks=[-chem_max, 0, chem_max])
        cbn = fig.colorbar(imn, cax=axnb, orientation="horizontal", ticks=[-neural_max, 0, neural_max])
        for cb in (cbc, cbn):
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=2, labelsize=7, pad=2)
            cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        axc.set_title("Selected chemistry  ·  discovery SD", loc="left", pad=14)
        axn.set_title("Neural response  ·  unit coefficients", loc="left", pad=14)
        ref_values = held["reference"].to_numpy()
        boundaries = np.flatnonzero(ref_values[1:] != ref_values[:-1]) + 0.5
        for boundary in boundaries:
            axc.axhline(boundary, color="white", linewidth=2)
            axn.axhline(boundary, color="white", linewidth=2)
        for i, tick in enumerate(axc.get_yticklabels()):
            tick.set_color(colors[ref_values[i]])
        handles = [Line2D([], [], marker="s", linestyle="", color=colors[r], markersize=6,
                          label=display_refs[r]) for r in refs if r in ref_values]
        fig.legend(handles=handles, title="Reference", frameon=False, loc="upper center",
                   bbox_to_anchor=(0.59, 0.998), ncol=min(4, len(handles)), fontsize=8, title_fontsize=8)
        path = figures / "02_holdout_profiles"
        _save(fig, path)
        paths.extend([path.with_suffix(".png"), path.with_suffix(".svg")])

    captions = (
        "Figure 1. Discovery and held-out correspondence for the selected chemical module "
        f"({summary.get('selected_module', 'see results.json')}). Candidate selection and the neural "
        "projection direction were determined in discovery data; see the analysis report for the "
        "selection rule. Each point is one strain, colored by reference. Chemical scores and neural "
        "projections are centered separately within each split × reference group. Discovery and "
        "held-out scatter panels share axes. Bars display the supplied discovery and held-out "
        "slopes for all 13 neuron classes, in unit coefficient change per chemical score SD. "
        "No uncertainty interval is shown; the figure is descriptive, and inferential results "
        "are provided in the report. Correspondence does not establish causation.\n\n"
        "Figure 2. Supporting profiles of every held-out strain, sorted first by reference and "
        "then by chemical score within reference. Chemistry columns include all members of the "
        "selected module; neural columns include all 13 neuron classes. Chemistry values use "
        "the saved discovery reference means and discovery within-reference standard deviations. "
        "Neural values are the observed unit-vector coefficients. Both matrices use the same "
        "strain order; row-label colors and horizontal separators identify reference groups. "
        "The two color scales have different units and span their complete observed ranges, "
        "without clipping. Missing values, if present, are gray. Profiles are not centered "
        "within the held-out split or clustered for display. The heatmaps are supporting "
        "inspection material, not an independent validation test.\n"
    )
    caption_path = figures / "captions.txt"
    caption_path.write_text(captions, encoding="utf-8")
    paths.append(caption_path)
    return paths
