"""Poster panels from the five-sample, animal-paired analysis.

Run with the repository Python. Reads reviewed tables only; never fits a model.
Each PDF page also has a PNG preview. Main panels additionally have editable SVGs.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
NAVY, MUTED, LIGHT = "#132C4E", "#687F99", "#DDE4EB"
COLORS = {"A021": "#8794A5", "A022": "#207F9C", "A023": "#BC692B",
          "A007": "#207F9C", "A010": "#BC692B"}
SAMPLES = ["A021", "A022", "A023"]
CELLS = ["ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
         "ASEL", "ASER", "AWCON", "AWCOFF"]
MAIN_CELLS = ["AWA", "ASH", "AWCON"]
PAIRS = ["A022-A021", "A023-A021", "A023-A022"]
STYLE = {"font.family": "DejaVu Sans", "font.size": 11,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "text.color": NAVY,
         "xtick.color": NAVY, "ytick.color": NAVY, "axes.linewidth": .8,
         "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def tidy(ax):
    ax.tick_params(length=3, labelsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def legend(fig, samples, anchor, fontsize=11):
    handles = [Line2D([0], [0], color=COLORS[s], lw=2.5, label=s) for s in samples]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=anchor,
               frameon=False, ncol=len(samples), fontsize=fontsize,
               handlelength=1.6, columnspacing=1.4)


def trace(ax, curves, summary, cell, samples, end=25, individuals=False):
    selected = summary[summary.neuron_class.eq(cell) & summary.sample_id.isin(samples)
                       & summary.time_s.lt(end)]
    ax.axvspan(0, 10, color=LIGHT, alpha=.55, lw=0, zorder=-5)
    ax.axhline(0, color=NAVY, lw=.7, alpha=.5, zorder=-4)
    for sample in samples:
        s = selected[selected.sample_id.eq(sample)].sort_values("time_s")
        if individuals:
            raw = curves[curves.neuron_class.eq(cell) & curves.sample_id.eq(sample)
                         & curves.time_s.lt(end)]
            for _, animal in raw.groupby("animal_id"):
                animal = animal.sort_values("time_s")
                ax.plot(animal.time_s, animal.response, lw=.55,
                        color=COLORS[sample], alpha=.28)
        else:
            ax.fill_between(s.time_s.to_numpy(), (s["mean"]-s["sem"]).to_numpy(),
                            (s["mean"]+s["sem"]).to_numpy(),
                            color=COLORS[sample], alpha=.13, linewidth=0)
        ax.plot(s.time_s, s["mean"], color=COLORS[sample], lw=2)
    ax.set(xlim=(-5, end), xticks=[0, 10, end], xlabel="Time (s)")
    tidy(ax)
    return selected


def stage_points(ax, stages, cell, samples, stage="10-25s"):
    d = stages[stages.neuron_class.eq(cell) & stages.sample_id.isin(samples)
               & stages.stage.eq(stage) & stages.response.notna()]
    matrix = d.pivot(index="animal_id", columns="sample_id", values="response").reindex(columns=samples)
    assert matrix.notna().all().all(), "Main curves require identical animal support."
    offsets = np.linspace(-.10, .10, len(matrix))
    for j, (_, values) in enumerate(matrix.iterrows()):
        x = np.arange(len(samples)) + offsets[j]
        ax.plot(x, values, color="#B4BDC7", lw=.8, alpha=.8, zorder=1)
        for k, sample in enumerate(samples):
            ax.scatter(x[k], values[sample], s=27, color=COLORS[sample],
                       edgecolors="white", linewidths=.45, zorder=3)
    for k, sample in enumerate(samples):
        ax.plot([k-.18, k+.18], [matrix[sample].mean()]*2, color=NAVY, lw=2, zorder=4)
    ax.axhline(0, color=NAVY, lw=.65, alpha=.45, zorder=-3)
    ax.set(xlim=(-.4, len(samples)-.6), xticks=range(len(samples)), xticklabels=samples)
    tidy(ax)
    return d


def main_panel(curves, summary, stages, chemistry):
    fig = plt.figure(figsize=(14, 6.9), facecolor="white")
    fig.text(.04, .946, "Strain rankings differ across sensory neurons", fontsize=20, weight="bold")
    fig.text(.04, .900, "Bacteroides stercoris", style="italic", fontsize=12, color=MUTED)
    legend(fig, SAMPLES, (.735, .908))
    fig.text(.04, .828, "A   Reference chemistry", fontsize=13, weight="bold")
    fig.text(.345, .828, "B   Calcium responses", fontsize=13, weight="bold")
    fig.text(.345, .405, "C   Paired animal means, 10–25 s", fontsize=13, weight="bold")

    chem = chemistry[chemistry.scope.eq("primary")].set_index("pair_id").loc[
        ["A021_A022", "A021_A023", "A022_A023"]].reset_index()
    ax = fig.add_axes([.125, .35, .15, .35])
    for y, row in zip([2, 1, 0], chem.itertuples()):
        ax.plot([0, row.common_rms_log2fc], [y, y], color=LIGHT, lw=2)
        ax.scatter(row.common_rms_log2fc, y, color=NAVY, s=43, zorder=3)
        ax.text(row.common_rms_log2fc+.08, y, f"{row.common_rms_log2fc:.2f}", va="center", fontsize=10)
    ax.set(ylim=(-.6, 2.6), xlim=(0, 1.93), yticks=[2, 1, 0],
           yticklabels=["A021 / A022", "A021 / A023", "A022 / A023"],
           xticks=[0, .5, 1, 1.5], xlabel="RMS difference\n(log₂FC)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", labelsize=9, length=3)
    ax.set_title("322 shared features", fontsize=11, color=MUTED, pad=12)
    fig.add_artist(Line2D([.303, .303], [.14, .79], transform=fig.transFigure, color=LIGHT, lw=.8))

    records = []
    for x, cell in zip([.36, .58, .80], MAIN_CELLS):
        ax = fig.add_axes([x, .535, .165, .22])
        trace(ax, curves, summary, cell, SAMPLES)
        n = stages[stages.sample_id.eq("A021") & stages.neuron_class.eq(cell)
                   & stages.stage.eq("10-25s") & stages.response.notna()].animal_id.nunique()
        ax.set_title(f"{cell}  ·  n = {n}", fontsize=12, pad=8)
        if cell == MAIN_CELLS[0]:
            ax.set_ylabel("ΔF/F₀")
        ax2 = fig.add_axes([x, .12, .165, .22])
        records.append(stage_points(ax2, stages, cell, SAMPLES))
        if cell == MAIN_CELLS[0]:
            ax2.set_ylabel("Mean ΔF/F₀")
    pd.concat(records).to_csv(TABLES / "poster_main_animal_points.csv", index=False)
    summary[summary.sample_id.isin(SAMPLES) & summary.neuron_class.isin(MAIN_CELLS)
            & summary.time_s.lt(25)].to_csv(TABLES / "poster_main_curve_values.csv", index=False)
    chem.to_csv(TABLES / "poster_main_chemistry.csv", index=False)
    return fig


def timing_panel(curves, summary, stages, windows):
    fig = plt.figure(figsize=(11.3, 4.9), facecolor="white")
    fig.text(.05, .93, "Time averaging can conceal a local response difference", fontsize=18, weight="bold")
    fig.text(.05, .865, "AWB  ·  n = 5 animals", fontsize=12, color=MUTED)
    legend(fig, ["A007", "A010"], (.78, .87))
    ax = fig.add_axes([.075, .23, .255, .50])
    trace(ax, curves, summary, "AWB", ["A007", "A010"])
    ax.set_ylabel("ΔF/F₀")
    ax.set_title("A   Response time course", fontsize=12, loc="left", pad=12)
    d = windows[windows.pair.eq("A010-A007") & windows.neuron_class.eq("AWB")
                & windows.bin_index.lt(5) & windows.difference.notna()]
    matrix = d.pivot(index="animal_id", columns="bin_index", values="difference")
    assert matrix.shape == (5, 5) and matrix.notna().all().all()
    ax = fig.add_axes([.435, .23, .28, .50])
    ax.axvspan(-.5, 1.5, color=LIGHT, alpha=.55, linewidth=0, zorder=-4)
    ax.axhline(0, color=NAVY, lw=.8, alpha=.5)
    for j, (_, values) in enumerate(matrix.iterrows()):
        x = np.arange(5)+(j-2)*.035
        ax.plot(x, values, color="#A4B2C0", lw=.8, alpha=.75)
        ax.scatter(x, values, color="#71869D", s=19, edgecolors="white", linewidths=.4, zorder=3)
    for x, value in enumerate(matrix.mean()):
        ax.plot([x-.22, x+.22], [value]*2, color=NAVY, lw=2.2, zorder=4)
    ax.set(xticks=range(5), xticklabels=["0–5", "5–10", "10–15", "15–20", "20–25"],
           xlabel="Time window (s)", ylabel="Paired ΔF/F₀ difference", xlim=(-.5, 4.5))
    ax.set_title("B   A010 − A007", fontsize=12, loc="left", pad=12)
    tidy(ax)
    ax.tick_params(axis="x", labelsize=9)
    pair_stage = read("neural_paired_stages")
    whole = pair_stage[pair_stage.pair.eq("A010-A007") & pair_stage.neuron_class.eq("AWB")
                       & pair_stage.stage.eq("0-25s") & pair_stage.difference.notna()].sort_values("animal_id")
    np.testing.assert_allclose(whole.difference.to_numpy(), matrix.mean(axis=1).sort_index().to_numpy())
    ax2 = fig.add_axes([.83, .23, .115, .50])
    ax2.axhline(0, color=NAVY, lw=.8, alpha=.5)
    ax2.scatter(np.linspace(-.10, .10, len(whole)), whole.difference, s=29, color="#71869D",
                edgecolors="white", linewidths=.45)
    ax2.plot([-.23, .23], [whole.difference.mean()]*2, color=NAVY, lw=2.2)
    ax2.set(xlim=(-.5, .5), xticks=[0], xticklabels=["0–25"], xlabel="Window (s)", ylim=ax.get_ylim())
    ax2.set_title("C   Time average", fontsize=12, loc="center", pad=12)
    tidy(ax2)
    d.to_csv(TABLES / "poster_timing_animal_windows.csv", index=False)
    whole.to_csv(TABLES / "poster_timing_animal_average.csv", index=False)
    return fig


def all_cell_page(curves, summary, samples, title):
    fig, axes = plt.subplots(4, 4, figsize=(13.6, 10.1))
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.86, hspace=.64, wspace=.36)
    fig.suptitle(title, x=.05, ha="left", y=.965, fontsize=19, weight="bold")
    legend(fig, samples, (.51, .912))
    for ax, cell in zip(axes.flat, CELLS):
        trace(ax, curves, summary, cell, samples, end=40, individuals=True)
        n = summary[summary.sample_id.eq(samples[0]) & summary.neuron_class.eq(cell)].n.iloc[0]
        ax.set_title(f"{cell}  ·  n = {n}", fontsize=11)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("ΔF/F₀")
    for ax in list(axes.flat)[len(CELLS):]:
        ax.set_visible(False)
    fig.text(.07, .045, "Thin lines: animals. Thick lines: means. Shading: stimulation (0–10 s).\n"
             "Original baseline and units; cell-specific y scales. Missing cells are not filled.", fontsize=10, color=MUTED)
    return fig


def paired_page(paired):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 8.8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.85, wspace=.38)
    fig.suptitle("All neuron classes: paired differences over 0–25 s", x=.05, ha="left", y=.955,
                 fontsize=19, weight="bold")
    comparison = paired[paired.pair.isin(PAIRS) & paired.stage.eq("0-25s")].difference.dropna()
    bound = np.ceil(comparison.abs().max()*10)/10 + .03
    for ax, pair in zip(axes, PAIRS):
        sub = paired[paired.pair.eq(pair) & paired.stage.eq("0-25s") & paired.difference.notna()]
        ax.axvline(0, color=NAVY, lw=.8, alpha=.5)
        labels = []
        for y, cell in enumerate(CELLS):
            d = sub[sub.neuron_class.eq(cell)].sort_values("animal_id")
            labels.append(f"{cell}  ({len(d)})")
            ax.scatter(d.difference, y+np.linspace(-.12, .12, len(d)), s=20,
                       color="#71869D", edgecolors="white", linewidths=.4)
            ax.plot([d.difference.mean()]*2, [y-.22, y+.22], color=NAVY, lw=2)
        ax.set(yticks=range(13), yticklabels=labels, ylim=(12.6, -.6), xlim=(-bound, bound),
               xlabel="Paired ΔF/F₀ difference")
        ax.set_title(pair.replace("-", " − "), fontsize=13)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.spines["left"].set_visible(False)
    fig.text(.10, .06, "Each dot is one animal; bars are means. Parentheses give the paired animal count.\n"
             "The three contrasts share animals and strain endpoints; they are not independent replications.", fontsize=10, color=MUTED)
    return fig


def projection_page(projection):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 8.8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.81, wspace=.37)
    fig.suptitle("Descriptive decomposition of cross-animal differences", x=.05, ha="left", y=.955,
                 fontsize=19, weight="bold")
    handles = [Line2D([0], [0], color=c, marker="o", ls="none", label=l)
               for c, l in [("#207F9C", "Along shared shape"), ("#BC692B", "Outside shared shape")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .90), frameon=False, ncol=2)
    comparison = projection[projection.pair.isin(PAIRS) & projection.window.eq("0-25s")]
    vals = comparison[["along_agreement", "residual_agreement"]].to_numpy()
    margin = .06*(vals.max()-vals.min())
    limits = (vals.min()-margin, vals.max()+margin)
    for ax, pair in zip(axes, PAIRS):
        ax.axvline(0, color=NAVY, lw=.7, alpha=.5)
        for y, cell in enumerate(CELLS):
            d = projection[projection.pair.eq(pair) & projection.neuron_class.eq(cell)
                           & projection.window.eq("0-25s")]
            for offset, metric, color in [(-.13, "along_agreement", "#207F9C"),
                                          (.13, "residual_agreement", "#BC692B")]:
                ax.scatter(d[metric], y+offset+np.linspace(-.055, .055, len(d)), color=color, s=12, alpha=.35)
                ax.scatter(d[metric].mean(), y+offset, color=color, s=28, zorder=4)
        ax.set(yticks=range(13), yticklabels=CELLS, ylim=(12.6, -.6), xlim=limits,
               xlabel="Cross-animal product (ΔF/F₀)²")
        ax.set_title(pair.replace("-", " − "), fontsize=13)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.text(.07, .066, "Small points: held-out animal × mean of other paired animals; large points: means. 0–25 s templates exclude that animal.\n"
             "The two components sum to the observed cross-animal product. Negative estimates are retained.\n"
             "Observed test amplitudes are projected, not predicted. Overlapping folds are not independent tests; these are not variance fractions.",
             fontsize=9.5, color=MUTED)
    return fig


def chemistry_page(chemistry):
    fig = plt.figure(figsize=(13.6, 8.6))
    fig.text(.05, .95, "Chemical context and reporting limits", fontsize=19, weight="bold")
    p = chemistry[chemistry.scope.eq("primary")]
    ax = fig.add_axes([.13, .47, .28, .32])
    for y, row in enumerate(p.itertuples()):
        for off, col, color in [(-.14, "full_rms_log2fc", "#8794A5"),
                                (0, "common_rms_log2fc", "#207F9C"),
                                (.14, "common_qc_screen_rms_log2fc", "#BC692B")]:
            ax.scatter(getattr(row, col), y+off, color=color, s=35)
    ax.set(yticks=range(3), yticklabels=[s.replace("_", " / ") for s in p.pair_id],
           ylim=(2.5, -.5), xlim=(0, 3.3), xlabel="RMS difference (log₂FC)")
    ax.set_title("Distance depends on reporting support", fontsize=12, pad=14)
    handles = [Line2D([0], [0], marker="o", color=c, ls="none", label=l) for c,l in
               [("#8794A5", "All 380"), ("#207F9C", "Shared 322"), ("#BC692B", "Shared + QC: 286")]]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(.50, -.22), fontsize=9)
    cand = read("chemical_candidate_display").sort_values("display_rank")
    values = cand[[s+"_log2fc" for s in SAMPLES]].to_numpy()
    ax = fig.add_axes([.73, .43, .17, .39])
    cmap = LinearSegmentedColormap.from_list("chemical", ["#32738D", "#F5F5F2", "#BE733D"])
    limit = max(1, np.max(np.abs(values)))
    im = ax.imshow(values, vmin=-limit, vmax=limit, cmap=cmap, aspect="auto")
    ax.set(yticks=range(len(cand)), yticklabels=cand.metabolite.tolist(),
           xticks=range(3), xticklabels=SAMPLES)
    ax.tick_params(length=0, labelsize=9)
    ax.set_title("Chemical-only selection", fontsize=12, pad=14)
    cax = fig.add_axes([.924, .43, .015, .39])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Reported log₂FC", fontsize=10)
    fig.text(.08, .23, "Single-sided report missingness contributes 82.3%, 79.2% and 88.9% of the full squared distances, respectively.\n"
             "The 322-feature comparison uses exactly the same observed entries for all three pairs; the QC subset is sensitivity only.",
             fontsize=10.5, color=MUTED, linespacing=1.6)
    fig.text(.08, .11, "Six displayed entries: largest three-strain log₂FC ranges among shared features with QC RSD ≤ 0.30 and ≥ 2 QC observations.\n"
             "Report annotations lack identification-confidence levels. Selection does not use neural responses.\n"
             "Independent culture-batch reference spectra, not chemical measurements of the imaging aliquots. No molecular attribution.",
             fontsize=9.5, color=MUTED, linespacing=1.5)
    return fig


def main():
    FIGURES.mkdir(exist_ok=True)
    curves, summary, stages = read("neural_curves"), read("neural_curve_summary"), read("neural_animal_stages")
    windows, paired = read("neural_paired_windows"), read("neural_paired_stages")
    projection, chemistry = read("neural_oof_projection_animals"), read("chemical_pair_summary")
    with plt.rc_context(STYLE):
        with PdfPages(FIGURES / "poster_panels.pdf") as pdf:
            for name, fig in [("strain_response_poster", main_panel(curves, summary, stages, chemistry)),
                              ("timing_cancellation_poster", timing_panel(curves, summary, stages, windows))]:
                pdf.savefig(fig)
                fig.savefig(FIGURES / f"{name}.png", dpi=220)
                fig.savefig(FIGURES / f"{name}.svg")
                plt.close(fig)
        with PdfPages(FIGURES / "supporting_evidence.pdf") as pdf:
            pages = [all_cell_page(curves, summary, SAMPLES, "Three-strain comparison: all 13 neuron classes"),
                     paired_page(paired), projection_page(projection),
                     all_cell_page(curves, summary, ["A007", "A010"], "A007 / A010: all 13 neuron classes"),
                     chemistry_page(chemistry)]
            for i, fig in enumerate(pages, 1):
                pdf.savefig(fig)
                fig.savefig(FIGURES / f"supporting_evidence_{i:02d}.png", dpi=140)
                plt.close(fig)
    parameters = {
        "primary_samples": SAMPLES, "support_samples": ["A007", "A010"],
        "main_cells": MAIN_CELLS, "main_window": "0-25s", "paired_stage": "10-25s",
        "main_selection": "Post hoc illustrative selection after inspecting all 13 classes: AWA contrasts A021 with both other strains; ASH has consistent relative ordering; AWCON requires stage information. Not independent discoveries or a fixed neural module.",
        "support_selection": "AWB selected after inspection of the A007/A010 comparison for opposite-signed differences in the existing 10-15 and 20-25 s bins. All five bins and the whole-window mean are shown.",
        "curve_display": "Original 1-s samples, trial-averaged within animal. No smoothing or new centering. Main bands are pointwise SEM (not CI); supporting thin lines are animals. Y scales differ by cell.",
        "animal_display": "All matched animals; deterministic horizontal offsets and within-animal lines. Dark bars are means, not uncertainty intervals.",
        "chemical_main": "Fixed three-strain 322-feature reported intersection; no error bars or inference across features.",
        "pdf_pages": {"poster_panels.pdf": 2, "supporting_evidence.pdf": 5},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "table_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TABLES.glob("*.csv"))}
    }
    (ROOT / "logs/poster_parameters.json").write_text(json.dumps(parameters, indent=2)+"\n")
    print("Created 2 poster pages and 5 supporting pages; PNG/SVG and exact plot tables saved.")


if __name__ == "__main__":
    main()
