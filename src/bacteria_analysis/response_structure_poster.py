"""A compact poster bridge from response curves to signed template weights.

Reuses reviewed model results and reads only one raw-data condition to illustrate
binning. Smooth template interpolation is display-only, never used for scoring.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .response_structure_display import validate_evidence


GROUPS = [
    ("Useful approximation", ["AWCON"], "#207F9C"),
    ("Useful, with timing differences", ["ASK", "ADF", "ASJ", "AWA", "AWB", "ASH"], "#207F9C"),
    ("Timing beyond weights", ["AWCOFF"], "#BC692B"),
    ("Overall unresolved", ["ADL", "ASI", "ASER", "ASG", "ASEL"], "#838E97"),
]
SOURCE_FILES = ["tables/predictions.parquet", "tables/templates.csv", "tables/coefficients.csv",
                "tables/overall_summary.csv", "tables/cell_summary.csv", "tables/cell_assessments.csv",
                "data/preparation_metadata.json"]


def _raw_path(source):
    metadata = json.loads((source / "data/preparation_metadata.json").read_text())
    return Path(metadata["raw_file"])


def _unbinned_example(source, example):
    """Read one condition, retaining the original trial→animal→bin order."""
    columns = ["date", "worm_key", "segment_index", "stim_name", "neuron", "time_point",
               "delta_F_over_F0", "start_time", "end_time"]
    raw = pd.read_parquet(_raw_path(source), columns=columns, filters=[
        ("date", "==", "20260520"), ("stim_name", "==", "A300 stationary"),
        ("neuron", "==", "AWCON"), ("time_point", ">=", 0), ("time_point", "<=", 44)])
    if raw.empty or not raw.start_time.eq(5).all() or not raw.end_time.eq(15).all():
        raise ValueError("The raw example does not have the reviewed stimulus timing.")
    raw["delta_F_over_F0"] = raw.delta_F_over_F0.replace([np.inf, -np.inf], np.nan)
    keys = ["date", "stim_name", "worm_key", "segment_index", "neuron", "time_point"]
    trials = raw.groupby(keys, as_index=False).delta_F_over_F0.mean()
    animal = trials.groupby([key for key in keys if key != "segment_index"], as_index=False).delta_F_over_F0.mean()
    animal = animal.rename(columns={"delta_F_over_F0": "response"})
    animal["time_s"] = animal.time_point - 5
    animal["animal_id"] = animal.date + "|" + animal.worm_key
    post = animal.loc[animal.time_s.between(0, 39)].copy()
    post["bin_index"] = (post.time_s // 5).astype(int)
    bins = post.groupby(["animal_id", "bin_index"]).response.mean().sort_index()
    cached = example.set_index(["animal_id", "bin_index"]).actual.sort_index()
    if not bins.index.equals(cached.index) or not np.allclose(bins, cached, rtol=1e-12, atol=1e-14):
        raise ValueError("The unbinned example does not reproduce the cached animal-level bins.")
    mean = post.groupby("time_s", as_index=False).response.mean()
    counts = post.groupby("time_s").response.count()
    if not counts.eq(len(post.animal_id.unique())).all():
        raise ValueError("This schematic requires a complete example; do not swap aggregation under missingness.")
    return animal, mean, len(raw)


def _hashes(source):
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    with _raw_path(source).open("rb") as handle:
        hashes["raw_example_input"] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def poster_cache_current(source_dir, output_dir):
    """Check source and code identities before displaying a cached poster."""
    record = Path(output_dir) / "poster_parameters.json"
    if not record.exists():
        return False
    data = json.loads(record.read_text())
    return (data.get("source_sha256") == _hashes(Path(source_dir))
            and data.get("code_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def plot_poster(source_dir, output_dir):
    """Export one English poster figure, exact source values and an external caption."""
    source, out = Path(source_dir), Path(output_dir)
    reviewed = validate_evidence(source)
    predictions = pd.read_parquet(source / "tables/predictions.parquet")
    templates = pd.read_csv(source / "tables/templates.csv")
    coefficients = pd.read_csv(source / "tables/coefficients.csv", dtype={"block": str})
    overall = pd.read_csv(source / "tables/overall_summary.csv")
    summary = pd.read_csv(source / "tables/cell_summary.csv")
    assessments = pd.read_csv(source / "tables/cell_assessments.csv")
    condition = dict(cell="AWCON", strain="A300", block="20260520", window="0-40s")
    mask = np.ones(len(predictions), bool)
    for key, value in condition.items():
        mask &= predictions[key].astype(str).eq(value)
    example = predictions.loc[mask].copy()
    profile = templates.loc[templates.window.eq("0-40s") & templates.cell.eq("AWCON")
                            & templates.fit_type.eq("full")].sort_values("time_s")
    coef = coefficients.loc[coefficients.window.eq("0-40s") & coefficients.cell.eq("AWCON")
                            & coefficients.strain.eq("A300") & coefficients.block.eq("20260520")]
    if len(coef) != 1 or len(profile) != 8 or example.empty:
        raise ValueError("The reviewed illustration condition is unavailable.")
    weight = float(coef.coefficient.iloc[0])
    mean = example.groupby("time_s").actual.mean().reindex(profile.time_s)
    values = pd.DataFrame({"time_s": profile.time_s.to_numpy(), "condition_mean": mean.to_numpy(),
                           "template": profile.template.to_numpy(), "weight": weight,
                           "reconstruction": weight * profile.template.to_numpy()})
    for key, value in condition.items():
        values[key] = value
    values["n_animals"] = example.animal_id.nunique()
    animal_traces, raw_mean, raw_rows = _unbinned_example(source, example)
    shape_times = np.linspace(values.time_s.min(), values.time_s.max(), 351)
    shape_guide = PchipInterpolator(values.time_s, values.template, extrapolate=False)(shape_times)
    totals = overall.loc[overall.window.eq("0-40s")].iloc[0]
    models = ["B", "M1", "M2"]
    errors = pd.DataFrame({"model": models, "mse": [totals["mse_"+m] for m in models]})
    errors["rmse"] = np.sqrt(errors.mse)
    # Confirm the plotted overall estimator remains the equal-cell summary.
    for model in models:
        cell_mse = summary.loc[summary.window.eq("0-40s"), "mse_"+model].mean()
        if not np.isclose(cell_mse, totals["mse_"+model], rtol=1e-12, atol=1e-14):
            raise ValueError("Overall and cell-level cached errors are inconsistent.")
    group_records = [{"cell": cell, "poster_summary": label}
                     for label, cells, color in GROUPS for cell in cells]
    groups = pd.DataFrame(group_records).merge(
        assessments.loc[assessments.window.eq("0-40s")], on="cell", validate="one_to_one")
    if len(groups) != 13 or groups.cell.nunique() != 13:
        raise ValueError("The poster must account for all 13 neuron classes.")

    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#132C4E", "axes.labelcolor": "#132C4E",
             "text.color": "#132C4E", "xtick.color": "#132C4E", "ytick.color": "#132C4E",
             "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "path", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14.8, 8.1), facecolor="white")
        fig.text(.043, .952, "From response traces to a compact neural summary", fontsize=20, weight="bold")
        fig.text(.043, .91, "AWCON · A300", fontsize=10.5, color="#687F99")
        for x, label in [(.148, "Unbinned mean"), (.414, "8 time bins"), (.680, "Shared shape"), (.895, "Signed amplitude")]:
            fig.text(x, .849, label, fontsize=14, ha="center")

        def time_axis(ax, ylabel=None):
            ax.axvspan(0, 10, color="#D8DEE5", alpha=.45, zorder=-3)
            ax.axhline(0, color="#132C4E", alpha=.4, lw=.6, zorder=-2)
            ax.set(xlim=(0, 40), xticks=[0, 10, 40], xlabel="Time (s)")
            ax.tick_params(length=3, labelsize=9)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=10)

        limits = (-.18, max(float(raw_mean.response.max()), float(values.condition_mean.max()))*1.09)
        raw_ax = fig.add_axes([.055, .615, .185, .191])
        time_axis(raw_ax, "ΔF/F₀")
        raw_ax.set_ylim(limits)
        raw_ax.plot(raw_mean.time_s, raw_mean.response, color="#26354A", lw=2.1)
        raw_ax.set_title("", pad=0)
        bin_ax = fig.add_axes([.322, .615, .185, .191])
        time_axis(bin_ax)
        bin_ax.set_ylim(limits)
        # Eight horizontal bin means; no connected polyline or invented time points.
        for j, value in enumerate(values.condition_mean):
            bin_ax.fill_between([5*j, 5*(j+1)], [0, 0], [value, value],
                                color="#AFC4D4", alpha=.65, linewidth=0)
            bin_ax.plot([5*j+.25, 5*(j+1)-.25], [value, value], color="#4B718A", lw=2.4)
            bin_ax.axvline(5*j, color="white", lw=1.1, zorder=4)
        bin_ax.set_yticks([])
        bin_ax.spines["left"].set_visible(False)
        shape_ax = fig.add_axes([.593, .615, .174, .191])
        time_axis(shape_ax)
        shape_ax.set_ylim(-.1, 2.1)
        shape_ax.set_yticks([])
        shape_ax.spines["left"].set_visible(False)
        shape_ax.plot(shape_times, shape_guide, color="#207F9C", lw=2.4)
        shape_ax.scatter(values.time_s, values.template, s=10, color="#207F9C", zorder=3)
        fig.text(.679, .524, r"$h_c(t)$", fontsize=16, ha="center", color="#207F9C")
        fig.text(.679, .491, "shared · RMS = 1", fontsize=10, ha="center", color="#687F99")
        fig.text(.803, .708, "×", fontsize=26, ha="center", color="#132C4E")
        fig.text(.895, .704, f"{weight:+.2f}", fontsize=30, ha="center", color="#207F9C")
        fig.text(.895, .524, r"$a_{kc}$", fontsize=16, ha="center", color="#207F9C")
        fig.text(.895, .491, "one per condition", fontsize=10, ha="center", color="#687F99")
        fig.text(.148, .524, "1-s sampling", fontsize=10, ha="center", color="#687F99")
        fig.text(.414, .524, "8 × 5 s", fontsize=10, ha="center", color="#687F99")
        for start, end in [(.257, .302), (.525, .570)]:
            fig.add_artist(FancyArrowPatch((start, .716), (end, .716), transform=fig.transFigure,
                                          arrowstyle="-|>", mutation_scale=15, lw=1.3, color="#687F99"))
        fig.text(.548, .64, "fit across\nconditions", fontsize=9, ha="center", color="#687F99", va="top")
        fig.text(.047, .405, "Prediction in other animals", fontsize=14, weight="bold")
        fig.text(.533, .405, "Neuron-level evidence", fontsize=14, weight="bold")
        fig.add_artist(Line2D([.045, .966], [.451, .451], transform=fig.transFigure, color="#D8DEE5", lw=.7))

        ax = fig.add_axes([.179, .133, .265, .22])
        y = np.arange(3)[::-1]
        colors = ["#8B959D", "#207F9C", "#BC692B"]
        for yy, value, color in zip(y, errors.rmse, colors):
            ax.plot([0, value], [yy, yy], color="#DFE5E9", lw=1.3, zorder=1)
            ax.scatter(value, yy, s=45, color=color, zorder=3)
            ax.text(value+.01, yy, f"{value:.3f}", va="center", fontsize=10, color=color)
        ax.set(xlim=(0, .345), ylim=(-.5, 2.5), xticks=[0, .1, .2, .3], yticks=y,
               yticklabels=["Common response", "Shared shape", "Independent curves"], xlabel="Prediction error (ΔF/F₀)")
        ax.tick_params(axis="y", length=0, labelsize=10, pad=7)
        ax.tick_params(axis="x", length=3, labelsize=9)
        ax.spines["left"].set_visible(False)
        fig.text(.18, .366, "All 13 classes · lower is better", fontsize=9, color="#687F99")

        positions = [(.536, .344), (.760, .344), (.536, .222), (.760, .222)]
        short_labels = ["Useful approximation", "Useful + timing", "Timing beyond weights", "Overall unresolved"]
        for (x, ypos), short, (_, cells, color) in zip(positions, short_labels, GROUPS):
            fig.text(x, ypos, short, fontsize=11, color=color, weight="bold")
            names = [c + ("*" if c == "ASEL" else "") for c in cells]
            labels = "  ".join(names[:3]) + ("\n" + "  ".join(names[3:]) if len(names)>3 else "")
            fig.text(x, ypos-.032, labels, fontsize=11.5, va="top", linespacing=1.6)
        fig.text(.760, .095, "* Local timing evidence in ASEL", fontsize=8.5, color="#687F99")
        fig.text(.045, .034, "0–40 s  ·  13 neuron classes  ·  106 strains  ·  49 animals", fontsize=9.5, color="#687F99")
        out.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"curve_compression_poster.{extension}", dpi=220)
        plt.close(fig)

    values.to_csv(out / "compression_example.csv", index=False)
    animal_traces.to_csv(out / "example_unbinned_animal_traces.csv", index=False)
    raw_mean.to_csv(out / "example_unbinned_mean.csv", index=False)
    pd.DataFrame({"time_s": shape_times, "template_display_only": shape_guide}).to_csv(
        out / "template_display_guide.csv", index=False)
    example.to_csv(out / "example_animal_responses_and_predictions.csv", index=False)
    errors.to_csv(out / "heldout_prediction_errors.csv", index=False)
    groups.to_csv(out / "neuron_evidence_summary.csv", index=False)
    parameters = {"window": "0-40s", "refit": False, "source_dir": str(source.resolve()),
                  "source_sha256": _hashes(source), "reviewed_numeric_sha256": reviewed["numeric_sha256"],
                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "example": condition, "example_weight": weight,
                  "example_rule": "Previously reviewed strong positive AWCON example; chosen after inspecting results to illustrate compression, not a representative error estimate.",
                  "illustration": "Full-data unbinned condition mean, its eight 5-s means, and the cached shared template and coefficient; descriptive only.",
                  "raw_example": {"selected_rows": raw_rows, "n_animals": int(example.animal_id.nunique()),
                                  "sampling_s": 1, "trial_aggregation": "Within trial/volume, then within animal/volume, then time bins; matches cached animal bins.",
                                  "filters": "date=20260520, stim_name=A300 stationary, neuron=AWCON, time_point=0..44"},
                  "template_display": "PCHIP through the eight cached template nodes, 2.5–37.5 s only; no extrapolation, fitting or prediction at interpolated points.",
                  "error_metric": "sqrt(existing equal-cell hierarchical mean MSE); no cell normalization",
                  "evidence_groups": "Descriptive reviewed applicability, not discrete biological response types; groups are not created by an error threshold."}
    (out / "poster_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2)+"\n")
    caption = (
        "Top: AWCON–A300 condition mean (7 animals), eight 5-s averages, and a shared shape × signed amplitude. "
        "The unbinned trace retains the original ΔF/F₀ baseline; shading marks stimulation (0–10 s). "
        "The template and weights are fitted jointly across conditions (strain × acquisition block), not from this trace alone. "
        "Template dots are the eight model values; the smooth line is a display-only interpolation. "
        "Template RMS is one; signed amplitude retains ΔF/F₀ units. "
        "This strong-response example was selected after inspection to illustrate compression, not validation. "
        "Bottom left "
        "reports whole-animal held-out RMSE in original response units: a common response for all conditions, "
        "a shared shape with condition-specific weights, and independent condition curves. All prediction "
        "parameters use training animals only. Errors use the original hierarchical weights, including equal "
        "neuron-class weights; large responses can still influence the aggregate. Bottom right: exploratory "
        "neuron-level evidence, not response types. Independent curves are training means, not a noise ceiling. "
        "Timing evidence in ADF, ASK, ASJ and ASEL is local; "
        "AWCON also has a local late exception. Lack of extra prediction gain does not establish identical time courses."
    )
    (out / "caption.txt").write_text(caption+"\n")
    return {"figure": str(out / "curve_compression_poster.png"), "pdf": str(out / "curve_compression_poster.pdf"),
            "svg": str(out / "curve_compression_poster.svg"), "caption": str(out / "caption.txt")}
