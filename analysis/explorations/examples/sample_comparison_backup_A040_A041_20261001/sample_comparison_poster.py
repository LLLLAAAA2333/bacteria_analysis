"""Poster figure 5: A040/A041 reference chemistry and model amplitudes.

Notebook usage::
    from sample_comparison_poster import plot_sample_comparison
    plot_sample_comparison(root / "reports",
                           root / "reports/sample_comparison_draft_20261001")

Summarizes selected cached data only; no refit, raw processing or new imputation.
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


STRAINS = ("A040", "A041")
CELLS = ("AWCON", "ASK", "ADF", "ASJ", "AWA", "AWB", "ASH")
CURVE_CELLS = ("ADF", "ASH", "AWA")
DATE = "20260429"
COLORS = ("#16839C", "#CB7539")
INK, MUTED, RULE = "#20374D", "#71808F", "#DCE3E9"
CHEM = "population_first_20260930/tables"
MODEL = "response_structure_20260930"
CURVES = "exploration_20260929/tables/animal_curves.parquet"
SOURCES = (
    f"{CHEM}/aligned_chemical_log2fc_all.csv",
    f"{CHEM}/aligned_chemical_report_observed_all.parquet",
    f"{CHEM}/aligned_chemical_report_values_all.csv",
    f"{CHEM}/aligned_chemical_reference_groups_all.csv",
    f"{CHEM}/aligned_taxonomy_all.csv",
    "poster_neighborhoods_20260930/tables/pair_context.csv",
    f"{MODEL}/data/observations.parquet",
    f"{MODEL}/tables/coefficients.csv", f"{MODEL}/tables/templates.csv",
    f"{MODEL}/tables/cell_assessments_metadata.json", CURVES,
)


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_chemistry(reports):
    folder = reports / CHEM
    fc = pd.read_csv(folder / "aligned_chemical_log2fc_all.csv", index_col=0).loc[list(STRAINS)]
    raw = pd.read_csv(folder / "aligned_chemical_report_values_all.csv", index_col=0).loc[list(STRAINS)]
    observed = pd.read_parquet(folder / "aligned_chemical_report_observed_all.parquet").loc[list(STRAINS)]
    if not fc.columns.equals(raw.columns) or not fc.columns.equals(observed.columns):
        raise ValueError("Chemical feature order differs among the source tables.")
    if not observed.equals(raw.notna()):
        raise ValueError("Chemical mask differs from original report missingness.")
    refs = pd.read_csv(folder / "aligned_chemical_reference_groups_all.csv", index_col=0)
    if not refs.loc[list(STRAINS), "reference_group"].eq("ref12").all():
        raise ValueError("Expected the same numerical reference group, ref12.")
    a, b = observed.loc[STRAINS[0]], observed.loc[STRAINS[1]]
    states = np.select([a & b, a & ~b, ~a & b],
                       ["joint_reported", "only_a_reported", "only_b_reported"],
                       default="neither_reported")
    chemistry = pd.DataFrame({"feature": fc.columns, "a_log2fc": fc.iloc[0].to_numpy(),
                              "b_log2fc": fc.iloc[1].to_numpy(),
                              "a_report_ng_ml": raw.iloc[0].to_numpy(),
                              "b_report_ng_ml": raw.iloc[1].to_numpy(), "report_state": states})
    counts = chemistry.report_state.value_counts().to_dict()
    if counts != {"joint_reported": 336, "only_a_reported": 13,
                  "only_b_reported": 10, "neither_reported": 21}:
        raise ValueError("The selected chemical report coverage has changed.")
    joint = chemistry.loc[chemistry.report_state.eq("joint_reported")].copy()
    if not np.isfinite(joint[["a_log2fc", "b_log2fc"]]).all().all():
        raise ValueError("Nonfinite jointly reported chemical values.")
    rms = float(np.sqrt(np.mean((joint.a_log2fc - joint.b_log2fc) ** 2)))
    context = pd.read_csv(reports / "poster_neighborhoods_20260930/tables/pair_context.csv")
    context = context.loc[context.pair_id.eq(f"{DATE}_{STRAINS[0]}_{STRAINS[1]}")].iloc[0]
    if not np.isclose(rms, context.joint_reported_rms, rtol=1e-12) or context.mutual_nearest != 1:
        raise ValueError("Selected chemical context differs from the existing audit.")
    taxonomy = pd.read_csv(folder / "aligned_taxonomy_all.csv", index_col=0).loc[list(STRAINS)]
    return chemistry, joint, counts, rms, context, taxonomy


def _load_neural(reports, hashes):
    model = reports / MODEL
    review = json.loads((model / "tables/cell_assessments_metadata.json").read_text())
    for name in ("tables/templates.csv", "tables/coefficients.csv"):
        if hashes[f"{MODEL}/{name}"] != review["evidence_sha256"][name]:
            raise ValueError(f"Reviewed model source has changed: {name}")
    observations = pd.read_parquet(model / "data/observations.parquet", filters=[
        ("sample_id", "in", list(STRAINS)), ("block", "==", DATE),
        ("neuron_class", "in", list(CELLS)), ("bin_index", "<", 8)])
    wide = pd.read_parquet(reports / CURVES, filters=[
        ("sample_id", "in", list(STRAINS)), ("date", "==", DATE),
        ("neuron_class", "in", list(CELLS))]).reset_index()
    keys = ["sample_id", "date", "worm_key", "neuron_class"]
    if wide.duplicated(keys).any():
        raise ValueError("Duplicate original animal curves.")
    curves = wide.melt(id_vars=keys, value_vars=[str(i) for i in range(40)],
                       var_name="time_s", value_name="response")
    curves["time_s"] = curves.time_s.astype(int)
    curves["bin_index"] = curves.time_s // 5
    if np.isinf(curves.response).any():
        raise ValueError("Infinite responses cannot be treated as missing data.")
    if not curves.groupby(keys).response.count().isin([0, 40]).all():
        raise ValueError("Selected animal curves must be complete or entirely missing.")
    observations["date"] = observations.date.astype(str)
    cached = observations.set_index(keys + ["bin_index"]).response.dropna().sort_index()
    rebinned = curves.groupby(keys + ["bin_index"]).response.mean().dropna().sort_index()
    if not rebinned.index.equals(cached.index) or not np.allclose(rebinned, cached, rtol=1e-12, atol=1e-14):
        raise ValueError("Original 1-s curves do not reproduce the cached animal bins.")
    for cell in CELLS:
        sets = [set(curves.loc[curves.sample_id.eq(s) & curves.neuron_class.eq(cell)
                               & curves.response.notna(), "worm_key"]) for s in STRAINS]
        if sets[0] != sets[1]:
            raise ValueError(f"Different animal support between strains for {cell}.")
    summary = curves.groupby(["sample_id", "neuron_class", "time_s"]).response.agg(
        mean="mean", sd="std", n="count").reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    expected = pd.MultiIndex.from_product([STRAINS, CELLS, range(40)], names=["sample_id", "neuron_class", "time_s"])
    if not summary.set_index(expected.names).index.sort_values().equals(expected.sort_values()):
        raise ValueError("Missing condition-cell-time combinations.")
    if not summary.n.ge(3).all() or not summary.groupby(["sample_id", "neuron_class"]).n.nunique().eq(1).all():
        raise ValueError("Incomplete animal coverage within the selected curves.")
    coeff = pd.read_csv(model / "tables/coefficients.csv", dtype={"block": str})
    coeff = coeff.loc[coeff.window.eq("0-40s") & coeff.fit_type.eq("full")
                      & coeff.block.eq(DATE) & coeff.strain.isin(STRAINS) & coeff.cell.isin(CELLS)].copy()
    templates = pd.read_csv(model / "tables/templates.csv")
    templates = templates.loc[templates.window.eq("0-40s") & templates.fit_type.eq("full")
                              & templates.cell.isin(CELLS)].copy()
    if len(coeff) != 14 or len(templates) != 56:
        raise ValueError("Expected two amplitudes and eight template bins per cell.")
    for _, t in templates.groupby("cell"):
        if not np.isclose(np.mean(t.template ** 2), 1) or t.iloc[np.argmax(abs(t.template))].template <= 0:
            raise ValueError("Template normalization or sign convention changed.")
    means = observations.groupby(["sample_id", "neuron_class", "bin_index"]).response.mean().reset_index()
    means = means.rename(columns={"sample_id": "strain", "neuron_class": "cell", "response": "mean"})
    reconstruction = means.merge(templates[["cell", "bin_index", "time_s", "template"]],
                                 on=["cell", "bin_index"], validate="many_to_one")
    reconstruction = reconstruction.merge(coeff[["strain", "cell", "coefficient"]],
                                           on=["strain", "cell"], validate="many_to_one")
    projected = reconstruction.assign(product=lambda d: d["mean"] * d.template).groupby(["strain", "cell"]).product.mean().sort_index()
    saved = coeff.set_index(["strain", "cell"]).coefficient.sort_index()
    if not projected.index.equals(saved.index) or not np.allclose(projected, saved, rtol=1e-10, atol=1e-12):
        raise ValueError("Displayed amplitudes do not equal the observed mean/template projection.")
    reconstruction["reconstruction"] = reconstruction.coefficient * reconstruction.template
    reconstruction["residual"] = reconstruction["mean"] - reconstruction.reconstruction
    return curves, summary, coeff, templates, reconstruction


def plot_sample_comparison(reports_dir, output_dir):
    """Draw the user-selected example, with exact plotted values and provenance."""
    reports, out = Path(reports_dir), Path(output_dir)
    hashes = {name: _sha256(reports / name) for name in SOURCES}
    chemistry, joint, counts, rms, context, taxonomy = _load_chemistry(reports)
    curves, summary, coeff, templates, reconstruction = _load_neural(reports, hashes)
    out.mkdir(parents=True, exist_ok=True)
    matrix = coeff.pivot(index="strain", columns="cell", values="coefficient").reindex(index=STRAINS, columns=CELLS)
    limit = float(np.ceil(abs(matrix.to_numpy()).max() * 10) / 10)
    cmap = LinearSegmentedColormap.from_list("signed_amplitude", ["#74509C", "#FBFBFC", "#147D78"])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    chemistry_limits = [float(np.floor(joint[["a_log2fc", "b_log2fc"]].min().min() / 5) * 5),
                        float(np.ceil(joint[["a_log2fc", "b_log2fc"]].max().max() / 5) * 5)]
    curve_limits = {}
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#AAB7C2", "axes.linewidth": .8,
             "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": MUTED, "ytick.color": MUTED,
             "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(17.8, 10.4), facecolor="white")
        fig.text(.040, .942, "A040 and A041 differ in cell-response amplitudes", fontsize=25, weight="bold")
        fig.legend(handles=[Line2D([0], [0], color=c, lw=2.7, label=s) for s, c in zip(STRAINS, COLORS)],
                   loc="upper right", bbox_to_anchor=(.97, .915), ncol=2, frameon=False, fontsize=12)
        fig.text(.040, .838, "A", fontsize=18, weight="bold")
        fig.text(.078, .838, "Reference chemistry", fontsize=16, weight="bold")
        fig.text(.078, .802, "336 jointly reported features", fontsize=10.5, color=MUTED)
        ax = fig.add_axes([.083, .275, .283, .283 * 17.8 / 10.4])
        ax.plot(chemistry_limits, chemistry_limits, color="#A7B4BF", lw=1.1, zorder=1)
        ax.scatter(joint.a_log2fc, joint.b_log2fc, s=19, color="#536F86", alpha=.64,
                   edgecolors="white", linewidths=.3, zorder=3)
        ax.set(xlim=chemistry_limits, ylim=chemistry_limits, aspect="equal",
               xlabel="A040 log₂FC", ylabel="A041 log₂FC")
        ax.set_xticks(np.arange(chemistry_limits[0], chemistry_limits[1] + 1, 5))
        ax.set_yticks(np.arange(chemistry_limits[0], chemistry_limits[1] + 1, 5))
        ax.xaxis.label.set_color(COLORS[0])
        ax.yaxis.label.set_color(COLORS[1])
        ax.tick_params(labelsize=10, length=3.5, pad=5)
        fig.add_artist(Line2D([.410, .410], [.12, .854], transform=fig.transFigure, color=RULE, lw=.8))
        fig.text(.451, .838, "B", fontsize=18, weight="bold")
        fig.text(.489, .838, "Cell-response amplitudes", fontsize=16, weight="bold")
        fig.text(.962, .838, "Shared-shape model  ·  0–40 s", fontsize=10, color=MUTED, ha="right")
        ax = fig.add_axes([.506, .642, .447, .128])
        grid = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="none", cmap=cmap, norm=norm)
        cell_counts = summary.groupby("neuron_class").n.min()
        ax.set(xticks=range(7), xticklabels=[f"{c}\nn = {int(cell_counts[c])}" for c in CELLS],
               yticks=[0, 1], yticklabels=STRAINS)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=7, labelsize=10)
        ax.tick_params(axis="y", length=0, pad=9, labelsize=11)
        for tick, c in zip(ax.get_xticklabels(), CELLS):
            tick.set_color(INK if c in CURVE_CELLS else MUTED)
        for tick, c in zip(ax.get_yticklabels(), COLORS):
            tick.set_color(c)
            tick.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(6) + .5, minor=True)
        ax.set_yticks([.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", bottom=False, top=False, left=False)
        for i in range(2):
            for j in range(7):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > .68 * limit else INK)
        cax = fig.add_axes([.784, .590, .168, .010])
        cbar = fig.colorbar(grid, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=8.5, pad=3)
        fig.text(.774, .595, "Amplitude (ΔF/F₀)", fontsize=9.5, ha="right", va="center")
        fig.add_artist(Line2D([.451, .962], [.550, .550], transform=fig.transFigure, color=RULE, lw=.8))
        fig.text(.451, .505, "C", fontsize=18, weight="bold")
        fig.text(.489, .505, "Response curves", fontsize=16, weight="bold")
        fig.text(.962, .505, "1-s mean ± SEM", fontsize=10.5, ha="right", color=MUTED)
        for cell, x in zip(CURVE_CELLS, [.490, .657, .824]):
            ax = fig.add_axes([x, .157, .137, .282])
            d = summary.loc[summary.neuron_class.eq(cell)]
            bins = reconstruction.loc[reconstruction.cell.eq(cell)]
            lower = min(0., (d["mean"] - d["sem"]).min(), bins.reconstruction.min())
            upper = max(0., (d["mean"] + d["sem"]).max(), bins.reconstruction.max())
            span = upper - lower
            ax.set_ylim(lower - .10 * span, upper + .12 * span)
            curve_limits[cell] = list(ax.get_ylim())
            ax.axvspan(0, 10, color="#E7ECF1", alpha=.75, linewidth=0, zorder=-3)
            ax.axhline(0, color="#A8B4BF", lw=.8, zorder=-2)
            for strain, color in zip(STRAINS, COLORS):
                p = d.loc[d.sample_id.eq(strain)].sort_values("time_s")
                r = bins.loc[bins.strain.eq(strain)].sort_values("time_s")
                ax.fill_between(p.time_s, p["mean"] - p["sem"], p["mean"] + p["sem"], color=color, alpha=.16, lw=0)
                ax.plot(r.time_s, r.reconstruction, color=color, lw=1.0, linestyle=(0, (4, 3)), alpha=.72)
                ax.plot(p.time_s, p["mean"], color=color, lw=2.2)
            ax.set_title(cell, fontsize=14, pad=12)
            ax.set(xlim=(0, 40), xticks=[0, 10, 40], xlabel="Time (s)")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.tick_params(labelsize=9, length=3.5, pad=4)
            if cell == CURVE_CELLS[0]:
                ax.set_ylabel("ΔF/F₀", labelpad=7)
        fig.text(.490, .070, "Dashed: template × amplitude", fontsize=10, color=MUTED)
        fig.text(.962, .070, "Cell-specific y scales", fontsize=9.5, color=MUTED, ha="right")
        paths = {}
        for extension in ("png", "pdf", "svg"):
            path = out / f"sample_comparison_draft.{extension}"
            fig.savefig(path, dpi=220)
            paths[extension] = str(path.resolve())
        plt.close(fig)

    for name, frame in {"chemical_pair_all_features": chemistry, "chemical_scatter_values": joint,
                        "selected_animal_1s_curves": curves, "plotted_1s_mean_sem": summary,
                        "signed_amplitudes": coeff, "shared_templates": templates,
                        "plotted_curves_and_reconstruction": reconstruction}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame([dict(report_state=k, n=v) for k, v in counts.items()]).to_csv(out / "chemical_report_coverage.csv", index=False)
    taxonomy.to_csv(out / "selected_taxonomy_metadata.csv")
    parameters = {
        "reports_dir": str(reports.resolve()), "source_sha256": hashes, "code_sha256": _sha256(__file__),
        "strains": STRAINS, "date": DATE, "model_cells": CELLS, "curve_cells": CURVE_CELLS,
        "selection_rule": "User-selected A040/A041 after descriptive review of 28 existing chemical-neighbor pairs. ADF/ASH illustrate selective amplitude differences; AWA is a secondary early-response comparison with timing limitations. Post hoc illustration, not a confirmatory comparison.",
        "reference_group": "ref12", "chemical_report_counts": counts, "chemical_joint_rms_log2fc": rms,
        "chemical_mask": "Original report joint mask, 336 pair-specific features; no imputation for plotting.",
        "chemical_axes": {"limits": chemistry_limits, "aspect": "equal", "reference_line": "y=x"},
        "model_window_seconds": [0, 40], "model_bin_seconds": 5, "plotted_time_points": list(range(40)),
        "model": "Saved full-data per-cell shared templates across 112 strain-by-date conditions; RMS=1, largest absolute bin positive. Amplitude is a mean-curve projection, not a response-presence label. AWA/AWB timing is not fully represented.",
        "mean_sem": "Animal means at original 1-s points; SEM=sample SD(ddof=1)/sqrt(n) after existing within-animal trial averages. No smoothing or rebaselining; animals, not trials, are replicates.",
        "missingness": "Unreported chemistry excluded from scatter; neural missing animals remain NaN and are excluded from means/SEMs.",
        "amplitude_color_limits": [-limit, limit], "curve_y_limits": curve_limits,
        "sequence_mean_abs_trial_gap": float(context.sequence_mean_abs_gap),
        "limitations": "Fixed strain order; chemistry from separate cultures, not stimulus aliquots; no chemical biological-replicate uncertainty. Source taxonomy has source_note_check flags, so species names are omitted from the figure.",
        "outputs": paths,
    }
    (out / "plot_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    caption = (
        "A040 and A041 differ in cell-response amplitudes. A, 336 jointly reported chemical features "
        "are plotted as existing log₂FC relative to ref12; the line is y=x (RMS difference, 1.021). "
        "The strains are mutual nearest neighbors within their six-strain comparison group, not chemically "
        "equivalent. Features reported only in A040 (13), only in A041 (10), or neither (21) are excluded; "
        "upstream fill-zero/+1 processing is retained. B, signed amplitudes use the same 0–40 s shared "
        "templates as Fig. 4, fitted across 112 strain-by-date conditions. Templates have unit RMS and "
        "a positive largest absolute bin. The seven classes are a fixed subset of the atlas. C, original "
        "1-s animal means ± pointwise SEM; dashed curves show the saved eight-bin template × amplitude, "
        "with no refit. Gray shading marks 0–10 s stimulation; y scales differ between cells. Counts in "
        "B apply to both strains and the same animals within each class (ADF, 6; ASH/AWA, 7), recorded "
        "on 2026-04-29 after within-animal trial averaging. ADF/ASH illustrate similar strong ADF means "
        "and reduced ASH amplitude; AWA shows an early-response difference. Near-zero AWA amplitude "
        "does not imply no response, and AWA/AWB timing is not fully summarized by one coefficient. "
        "This example was selected after reviewing existing candidates. Fixed stimulus order (mean "
        "trial gap, about 21) limits strain attribution. Reference chemistry comes from separate cultures, "
        "not stimulus aliquots, and has no biological-replicate uncertainty here. The comparison does "
        "not establish chemical causation or independent neural information.\n"
    )
    (out / "caption.txt").write_text(caption)
    verification = {
        "chemical_scatter_points": len(joint), "chemical_report_counts": counts,
        "amplitude_matrix_shape": list(matrix.shape),
        "animal_counts": {cell: int(summary.loc[summary.neuron_class.eq(cell), "n"].min()) for cell in CELLS},
        "checks_passed": ["Original chemical report mask agrees with raw report missingness",
                          "Chemical reference groups and joint RMS match existing audit",
                          "Saved templates and amplitudes match reviewed source SHA256",
                          "Original 1-s curves reproduce cached animal bins",
                          "Each selected animal curve is complete or entirely missing",
                          "Both strains have identical animal support within each class",
                          "Displayed coefficients reproduce the mean/template projections"],
        "source_files_unchanged": hashes == {name: _sha256(reports / name) for name in SOURCES},
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return paths
