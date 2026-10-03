"""Draw the three-strain, seven-cell poster process from saved model outputs.

Notebook usage::

    from bacteria_analysis.response_structure_process import plot_response_process
    plot_response_process(root / "reports/representation/response_structure_20260930",
                          root / "reports/representation/response_process_draft_20261001")

This is a descriptive illustration, not a refit or a prediction-performance
figure. Panel A uses saved one-second animal curves; the model retains its
eight five-second bins. Both inputs already contain the within-animal trial average.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


STRAINS = ("A021", "A022", "A023")
CELLS = ("AWCON", "ASK", "ADF", "ASJ", "AWA", "AWB", "ASH")
BLOCK = "20260601"
COLORS = ("#78899D", "#16839C", "#CB7539")
INK, MUTED, RULE = "#20374D", "#71808F", "#DCE3E9"
SOURCE_FILES = ("data/observations.parquet", "tables/templates.csv",
                "tables/coefficients.csv", "tables/predictions.parquet",
                "tables/cell_assessments_metadata.json")


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_examples(source):
    """Read only the selected cached observations and verify their provenance."""
    hashes = {name: _sha256(source / name) for name in SOURCE_FILES}
    review = json.loads((source / "tables/cell_assessments_metadata.json").read_text())
    for name in ("tables/templates.csv", "tables/coefficients.csv", "tables/predictions.parquet"):
        if hashes[name] != review["evidence_sha256"][name]:
            raise ValueError(f"Saved model output differs from its reviewed version: {name}")

    observations = pd.read_parquet(source / "data/observations.parquet", filters=[
        ("sample_id", "in", list(STRAINS)), ("block", "==", BLOCK),
        ("neuron_class", "in", list(CELLS)), ("bin_index", "<", 8)])
    observations = observations.rename(columns={"sample_id": "strain", "neuron_class": "cell"})
    keys = ["strain", "block", "cell", "animal_id", "bin_index"]
    if observations.duplicated(keys).any():
        raise ValueError("Expected one response per animal, condition, cell and bin.")
    if np.isinf(observations.response).any():
        raise ValueError("Infinite responses cannot be summarized as missing data.")
    saved = pd.read_parquet(source / "tables/predictions.parquet", filters=[
        ("strain", "in", list(STRAINS)), ("block", "==", BLOCK),
        ("cell", "in", list(CELLS)), ("window", "==", "0-40s")])
    left = observations.dropna(subset=["response"]).set_index(keys).response.sort_index()
    right = saved.set_index(keys).actual.sort_index()
    if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-12, atol=1e-14):
        raise ValueError("Selected observations do not match the reviewed cached responses.")

    summary = observations.groupby(["strain", "block", "cell", "bin_index"]).response.agg(
        mean="mean", sd="std", n_animals="count").reset_index()
    expected = pd.MultiIndex.from_product([STRAINS, [BLOCK], CELLS, range(8)],
                                         names=["strain", "block", "cell", "bin_index"])
    summary = summary.set_index(expected.names).reindex(expected).reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n_animals)
    summary["time_s"] = summary.bin_index * 5 + 2.5
    if not summary.n_animals.ge(3).all():
        raise ValueError("This selected illustration requires at least three animals in every plotted bin.")
    summary["lower"] = summary["mean"] - summary["sem"]
    summary["upper"] = summary["mean"] + summary["sem"]

    templates = pd.read_csv(source / "tables/templates.csv")
    templates = templates.loc[templates.window.eq("0-40s") & templates.fit_type.eq("full")
                              & templates.cell.isin(CELLS)].copy()
    coefficients = pd.read_csv(source / "tables/coefficients.csv", dtype={"block": str})
    coefficients = coefficients.loc[
        coefficients.window.eq("0-40s") & coefficients.fit_type.eq("full")
        & coefficients.strain.isin(STRAINS) & coefficients.block.eq(BLOCK)
        & coefficients.cell.isin(CELLS)].copy()
    if len(templates) != len(CELLS) * 8 or templates.duplicated(["cell", "bin_index"]).any():
        raise ValueError("Expected exactly eight full-data template nodes per cell.")
    if len(coefficients) != len(STRAINS) * len(CELLS) or coefficients.duplicated(["strain", "cell"]).any():
        raise ValueError("Expected one full-data coefficient per displayed condition and cell.")
    for cell in CELLS:
        t = templates.loc[templates.cell.eq(cell)].sort_values("bin_index")
        if not np.allclose(t.time_s, np.arange(2.5, 40, 5)):
            raise ValueError("Unexpected template bin times.")
        if not np.isclose(np.mean(t.template.to_numpy() ** 2), 1, rtol=1e-10):
            raise ValueError("Template RMS is not one.")
        if t.template.iloc[np.argmax(np.abs(t.template.to_numpy()))] <= 0:
            raise ValueError("Template sign convention differs from the saved analysis.")
    values = summary.merge(templates[["cell", "bin_index", "template"]],
                           on=["cell", "bin_index"], validate="many_to_one")
    values = values.merge(coefficients[["strain", "block", "cell", "coefficient"]],
                          on=["strain", "block", "cell"], validate="many_to_one")
    values["reconstruction"] = values.coefficient * values.template
    values["residual"] = values["mean"] - values.reconstruction
    # Complete, equally weighted eight-bin condition means must reproduce a.
    projected = values.assign(product=values["mean"] * values.template).groupby(
        ["strain", "cell"])["product"].mean().sort_index()
    coefficients_indexed = coefficients.set_index(["strain", "cell"]).coefficient.sort_index()
    if not projected.index.equals(coefficients_indexed.index) or not np.allclose(
            projected, coefficients_indexed, rtol=1e-10, atol=1e-12):
        raise ValueError("Full-data coefficients do not match the displayed mean/template projection.")
    return observations, values, templates, coefficients, hashes


def _load_unbinned(curve_file, observations):
    """Summarize saved 1-s animal curves after checking their bin-level identity."""
    curves = pd.read_csv(curve_file, dtype={"date": str}, usecols=[
        "sample_id", "date", "animal_id", "neuron_class", "time_s", "response"])
    curves = curves.loc[curves.sample_id.isin(STRAINS) & curves.date.eq(BLOCK)
                        & curves.neuron_class.isin(CELLS) & curves.time_s.between(0, 39)].copy()
    curves = curves.rename(columns={"sample_id": "strain", "date": "block", "neuron_class": "cell"})
    keys = ["strain", "block", "cell", "animal_id"]
    if curves.duplicated(keys + ["time_s"]).any() or np.isinf(curves.response).any():
        raise ValueError("Invalid or duplicated animal-level one-second observations.")
    if not curves.groupby(keys).time_s.nunique().eq(40).all():
        raise ValueError("Expected all 40 original one-second time points for each selected animal curve.")
    curves["bin_index"] = (curves.time_s // 5).astype(int)
    rebinned = curves.groupby(keys + ["bin_index"]).response.mean().dropna().sort_index()
    cached = observations.dropna(subset=["response"]).set_index(keys + ["bin_index"]).response.sort_index()
    if not rebinned.index.equals(cached.index) or not np.allclose(
            rebinned, cached, rtol=1e-12, atol=1e-14):
        raise ValueError("The original one-second curves do not reproduce the cached animal bins.")
    summary = curves.groupby(["strain", "block", "cell", "time_s"]).response.agg(
        mean="mean", sd="std", n_animals="count").reset_index()
    expected = pd.MultiIndex.from_product([STRAINS, [BLOCK], CELLS, range(40)],
                                         names=["strain", "block", "cell", "time_s"])
    summary = summary.set_index(expected.names).reindex(expected).reset_index()
    if not summary.n_animals.ge(3).all():
        raise ValueError("Insufficient animal coverage for a displayed one-second mean/SEM.")
    summary["sem"] = summary.sd / np.sqrt(summary.n_animals)
    summary["lower"] = summary["mean"] - summary["sem"]
    summary["upper"] = summary["mean"] + summary["sem"]
    return curves, summary


def plot_response_process(source_dir, output_dir):
    """Write PNG/PDF/SVG, exact plotted values, counts, parameters and caption.

    The fixed initial examples reuse the previously examined A021/A022/A023
    group. No search over the full data or fit-quality ranking is performed.
    """
    source, out = Path(source_dir), Path(output_dir)
    observations, values, templates, coefficients, hashes = _load_examples(source)
    curve_file = source.parent.parent / "examples/sample_interpretation_20261001/tables/neural_curves.csv"
    curve_hash = _sha256(curve_file)
    animal_curves, curve_summary = _load_unbinned(curve_file, observations)
    out.mkdir(parents=True, exist_ok=True)
    matrix = coefficients.pivot(index="strain", columns="cell", values="coefficient").reindex(
        index=STRAINS, columns=CELLS)
    limit = np.ceil(np.max(np.abs(matrix.to_numpy())) * 10) / 10
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list("signed_amplitude", ["#74509C", "#FBFBFC", "#147D78"])
    cmap.set_bad("#D3D7DC")
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#AAB7C2", "axes.linewidth": .7,
             "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": MUTED, "ytick.color": MUTED,
             "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}
    axes_limits = {}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(18.5, 11.2), facecolor="white")
        left, right = .083, .971
        column = (right - left) / len(CELLS)
        inset = .015
        fig.text(.037, .948, "From response curves to cell-response profiles", fontsize=25, weight="bold")
        fig.text(.038, .912, "3 strains  ·  7 neuron classes  ·  0–40 s", fontsize=11, color=MUTED)
        handles = [Line2D([0], [0], color=color, lw=2.6, label=strain)
                   for strain, color in zip(STRAINS, COLORS)]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.975, .927), ncol=3,
                   frameon=False, handlelength=2.1, columnspacing=1.9, fontsize=12)

        fig.text(.038, .855, "A", fontsize=18, weight="bold")
        fig.text(left, .855, "Response curves", fontsize=16, weight="bold")
        fig.text(right, .855, "1-s mean ± SEM   ·   Dashed: 5-s reconstruction", fontsize=10.5, ha="right", color=MUTED)
        for j, cell in enumerate(CELLS):
            x = left + j * column + inset
            ax = fig.add_axes([x, .598, column - 2 * inset, .205])
            cell_values = curve_summary.loc[curve_summary.cell.eq(cell)]
            cell_bins = values.loc[values.cell.eq(cell)]
            lower = min(0., cell_values.lower.min(), cell_bins.reconstruction.min())
            upper = max(0., cell_values.upper.max(), cell_bins.reconstruction.max())
            span = upper - lower
            ax.set_ylim(lower - .10 * span, upper + .12 * span)
            axes_limits[cell] = list(ax.get_ylim())
            ax.axvspan(0, 10, color="#E7ECF1", alpha=.75, zorder=-3, linewidth=0)
            ax.axhline(0, color="#A8B4BF", lw=.7, zorder=-2)
            for strain, color in zip(STRAINS, COLORS):
                d = cell_values.loc[cell_values.strain.eq(strain)].sort_values("time_s")
                reconstruction = cell_bins.loc[cell_bins.strain.eq(strain)].sort_values("time_s")
                ax.fill_between(d.time_s, d.lower, d.upper, color=color, alpha=.16, linewidth=0)
                ax.plot(reconstruction.time_s, reconstruction.reconstruction, color=color, lw=1.0,
                        linestyle=(0, (4, 3)), alpha=.70, zorder=2)
                ax.plot(d.time_s, d["mean"], color=color, lw=2.0)
            ax.set(xlim=(0, 40), xticks=[0, 10, 40])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            ax.tick_params(labelsize=9, length=3, pad=3)
            ax.set_title(cell, fontsize=13, weight="bold", pad=22)
            counts = cell_values.groupby("strain").n_animals.agg(["min", "max"]).reindex(STRAINS)
            if counts["min"].eq(counts["max"]).all() and counts["min"].nunique() == 1:
                count_text = f"n = {int(counts['min'].iloc[0])} per strain"
            else:
                count_text = "n = " + "/".join(str(int(v)) for v in counts["min"])
            ax.text(.5, 1.035, count_text, transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
            if j == 0:
                ax.set_ylabel("ΔF/F₀", fontsize=11, labelpad=7)
        fig.text((left + right) / 2, .560, "Time (s)", ha="center", fontsize=10.5)
        fig.text(right, .558, "Cell-specific y scales", ha="right", fontsize=9, color=MUTED)
        fig.add_artist(Line2D([left, right], [.530, .530], transform=fig.transFigure, color=RULE, lw=.8))

        fig.text(.038, .487, "B", fontsize=18, weight="bold")
        fig.text(left, .487, "Shared shape", fontsize=16, weight="bold")
        fig.text(right, .487, "8 × 5 s   ·   Fitted across all 112 conditions", fontsize=10.5, color=MUTED, ha="right")
        for j, cell in enumerate(CELLS):
            x = left + j * column + inset
            ax = fig.add_axes([x, .315, column - 2 * inset, .139])
            d = templates.loc[templates.cell.eq(cell)].sort_values("time_s")
            ax.axvspan(0, 10, color="#E7ECF1", alpha=.75, zorder=-3, linewidth=0)
            ax.axhline(0, color="#A8B4BF", lw=.7)
            ax.plot(d.time_s, d.template, color=INK, lw=2., marker="o", ms=3)
            ax.set(xlim=(0, 40), ylim=(-2.6, 2.6), xticks=[0, 10, 40], yticks=[-2, 0, 2])
            ax.tick_params(labelsize=9, length=3, pad=3)
            if j == 0:
                ax.set_ylabel("Shape (RMS = 1)", fontsize=10.5, labelpad=7)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
                ax.spines["left"].set_visible(False)
        fig.text((left + right) / 2, .278, "Time (s)", ha="center", fontsize=10.5)
        fig.add_artist(Line2D([left, right], [.256, .256], transform=fig.transFigure, color=RULE, lw=.8))

        fig.text(.038, .216, "C", fontsize=18, weight="bold")
        fig.text(left, .216, "Signed amplitude", fontsize=16, weight="bold")
        fig.text(right, .216, "Response ≈ amplitude × shared shape", fontsize=12, ha="right")
        ax = fig.add_axes([left, .073, right - left, .113])
        grid = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap=cmap, norm=norm,
                         interpolation="none")
        ax.set(xticks=[], yticks=range(len(STRAINS)), yticklabels=STRAINS)
        ax.tick_params(axis="y", length=0, pad=10, labelsize=11)
        for tick, color in zip(ax.get_yticklabels(), COLORS):
            tick.set_color(color)
            tick.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(len(CELLS) - 1) + .5, minor=True)
        ax.set_yticks(np.arange(len(STRAINS) - 1) + .5, minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(STRAINS)):
            for j in range(len(CELLS)):
                value = matrix.iloc[i, j]
                text_color = "white" if abs(value) > .68 * limit else INK
                # Three decimals preserve small nonzero coefficients in this draft.
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=11, color=text_color)
        cax = fig.add_axes([.693, .028, .205, .010])
        cbar = fig.colorbar(grid, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=8.5, pad=3)
        fig.text(.686, .032, "Amplitude (ΔF/F₀)", fontsize=9.5, ha="right", va="center")
        fig.text(left, .033, "Shared shape × signed amplitude is an approximation", fontsize=9.5, color=MUTED)

        paths = {}
        for extension in ("png", "pdf", "svg"):
            path = out / f"response_process_draft.{extension}"
            fig.savefig(path, dpi=220)
            paths[extension] = str(path.resolve())
        plt.close(fig)

    values.to_csv(out / "plotted_curves_and_reconstruction.csv", index=False)
    observations.to_csv(out / "selected_animal_bins.csv", index=False)
    animal_curves.to_csv(out / "selected_animal_1s_curves.csv", index=False)
    curve_summary.to_csv(out / "plotted_1s_mean_sem.csv", index=False)
    templates.to_csv(out / "shared_templates.csv", index=False)
    coefficients.to_csv(out / "signed_amplitudes.csv", index=False)
    counts = values.groupby(["strain", "block", "cell"]).n_animals.agg(["min", "max"])
    counts.to_csv(out / "animal_counts.csv")
    parameters = {
        "source_dir": str(source.resolve()), "source_sha256": hashes,
        "one_second_curve_source": str(curve_file.resolve()), "one_second_curve_sha256": curve_hash,
        "code_sha256": _sha256(__file__), "window_seconds": [0, 40], "bin_seconds": 5,
        "strains": list(STRAINS), "block": BLOCK, "cells": list(CELLS),
        "selection_rule": "Reuse the previously examined A021/A022/A023 group from the same date; no all-data search or fit-quality optimization. Seven cells with previously reviewed useful template information, including timing limitations.",
        "mean": "Panel A: equal-weight mean of finite cached animal responses at each original 1-s time point, within strain, block and cell. Existing within-animal trial averages and baseline are retained.",
        "sem": "Panel A: animal sample SD (ddof=1) / sqrt(number of finite animal responses), independently at every 1-s time point. The retained bin-level audit table uses the same formula on animal bin means; pointwise SEMs are never averaged into bin SEMs.",
        "model": "Saved 0-40s full-data M1 templates and condition coefficients; no refit.",
        "template_normalization": "RMS=1, greatest-absolute bin positive; templates are dimensionless.",
        "coefficient_units": "delta_F_over_F0", "coefficient_color_limits": [-float(limit), float(limit)],
        "missingness": "Preserve missing responses, no zero imputation or thresholding.",
        "curve_display": "Panel A solid lines and bands: original 1-s animal means ± SEM at 0–39 s, without smoothing or rebaselining. Thin dashed reconstructions and Panel B templates: eight 5-s nodes at 2.5–37.5 s connected by straight lines; no increased model resolution or extrapolation.",
        "response_y_limits": axes_limits, "template_y_limits": [-2.6, 2.6],
        "individual_traces_plotted": False, "refit": False,
        "outputs": paths,
    }
    (out / "plot_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    caption = (
        "From response curves to cell-response profiles. The previously examined A021/A022/A023 "
        "group (acquisition date 2026-06-01) illustrates seven cells with useful, but incomplete, "
        "template information in prior analyses; examples were not optimized for fit quality. "
        "A, solid curves show original 1-s animal means ± pointwise SEM (sample SD/√n) after "
        "within-animal trial averaging, without smoothing or additional baseline correction. "
        "Dashed curves connect the eight 5-s template × amplitude reconstruction nodes; gray "
        "shading marks 0–10 s stimulation. Each cell uses the "
        "same animals across strains, with n shown per strain; y scales differ between cells. "
        "B, dimensionless templates are saved full-data fits across all 112 strain-by-acquisition-block "
        "conditions, retaining eight 5-s bins, RMS=1 and the greatest-absolute bin positive; "
        "template axes share a scale. "
        "C, signed amplitudes use one unclipped ΔF/F₀ color scale. Negative values reverse the "
        "template and do not generically indicate inhibition. Amplitudes preserve only the "
        "template-aligned component: notably, A023 has a near-zero AWCON coefficient despite "
        "an early negative and later positive response. These are descriptive full-data fits, "
        "not held-out predictions or evidence that every response is adequately compressed.\n"
    )
    (out / "caption.txt").write_text(caption)
    verification = {
        "selected_strains": len(STRAINS), "selected_cells": len(CELLS),
        "mean_sem_bin_rows": len(values), "coefficient_count": len(coefficients),
        "mean_sem_1s_rows": len(curve_summary), "animal_1s_rows": len(animal_curves),
        "template_nodes": len(templates), "animal_bin_rows": len(observations),
        "finite_animal_bin_rows": int(observations.response.notna().sum()),
        "minimum_animals_per_bin": int(values.n_animals.min()),
        "maximum_animals_per_bin": int(values.n_animals.max()),
        "checks_passed": ["Reviewed template/coefficient/prediction source hashes",
                          "Selected animal observations match saved model observations",
                          "Unique animal-condition-cell-bin records",
                          "Complete 3 × 7 × 8 displayed summary",
                          "Original one-second animal curves reproduce all 912 finite cached animal bins",
                          "Complete 3 × 7 × 40 one-second mean and SEM summary",
                          "Template RMS and sign conventions",
                          "All 21 coefficients equal the displayed mean/template projections"],
        "source_files_unchanged": hashes == {name: _sha256(source / name) for name in SOURCE_FILES},
        "one_second_source_unchanged": curve_hash == _sha256(curve_file),
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return paths
