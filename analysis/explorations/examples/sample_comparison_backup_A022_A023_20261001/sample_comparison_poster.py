"""Poster figure 5: chemistry, response curves, and paired animal differences.

Notebook usage::

    from sample_comparison_poster import plot_sample_comparison
    plot_sample_comparison(root / "reports/sample_interpretation_20261001",
                           root / "reports/sample_comparison_draft_20261001")

Only the selected cached A022/A023 examples are summarized. No raw-data
processing, model fitting, chemical imputation, or inferential tests are run.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


STRAINS = ("A022", "A023")
CELLS = ("ASH", "AWCON")
DATE = "20260601"
COLORS = ("#16839C", "#CB7539")
INK, MUTED, RULE = "#20374D", "#71808F", "#DCE3E9"
SOURCE_FILES = ("tables/chemical_pair_features.csv", "tables/chemical_pair_summary.csv",
                "tables/neural_curves.csv", "tables/neural_curve_summary.csv",
                "tables/neural_paired_stages.csv", "tables/neural_pair_stage_summary.csv")


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_selected(source):
    """Check cached source identities and the displayed small-sample summaries."""
    hashes = {name: _sha256(source / name) for name in SOURCE_FILES}
    manifest = json.loads((source / "logs/output_manifest.json").read_text())
    for name, value in hashes.items():
        if value != manifest[name]:
            raise ValueError(f"Source table differs from its recorded version: {name}")

    chemistry = pd.read_csv(source / "tables/chemical_pair_features.csv")
    chemistry = chemistry.loc[chemistry.pair_id.eq("A022_A023")].copy()
    if len(chemistry) != 380 or chemistry.metabolite.duplicated().any():
        raise ValueError("Expected 380 unique chemical report features for A022/A023.")
    a, b = chemistry.a_report_ng_ml.notna(), chemistry.b_report_ng_ml.notna()
    states = np.select([a & b, a & ~b, ~a & b],
                       ["joint_reported", "only_a_reported", "only_b_reported"],
                       default="neither_reported")
    if not np.array_equal(states, chemistry.missing_state):
        raise ValueError("Chemical reporting flags do not match the original report values.")
    counts = chemistry.missing_state.value_counts().to_dict()
    if counts != {"joint_reported": 332, "neither_reported": 31,
                  "only_a_reported": 10, "only_b_reported": 7}:
        raise ValueError("The selected chemical reporting coverage has changed.")
    joint = chemistry.loc[chemistry.missing_state.eq("joint_reported")].copy()
    if not np.isfinite(joint[["a_log2fc", "b_log2fc"]]).all().all():
        raise ValueError("Nonfinite jointly reported chemical values.")
    rms = float(np.sqrt(np.mean((joint.a_log2fc - joint.b_log2fc) ** 2)))
    chemical_summary = pd.read_csv(source / "tables/chemical_pair_summary.csv")
    saved_rms = chemical_summary.loc[chemical_summary.pair_id.eq("A022_A023"), "joint_rms_log2fc"].item()
    if not np.isclose(rms, saved_rms, rtol=1e-12):
        raise ValueError("Joint-report chemical RMS does not match the cached summary.")

    curves = pd.read_csv(source / "tables/neural_curves.csv", dtype={"date": str})
    curves = curves.loc[curves.sample_id.isin(STRAINS) & curves.date.eq(DATE)
                        & curves.neuron_class.isin(CELLS) & curves.time_s.between(0, 24)].copy()
    keys = ["sample_id", "date", "animal_id", "neuron_class"]
    if curves.duplicated(keys + ["time_s"]).any() or not np.isfinite(curves.response).all():
        raise ValueError("Selected observed animal curves are incomplete or duplicated.")
    if not curves.groupby(keys).time_s.nunique().eq(25).all():
        raise ValueError("Expected all 25 one-second samples per observed animal trace.")
    mean_keys = ["sample_id", "date", "neuron_class", "time_s"]
    summary = curves.groupby(mean_keys).response.agg(mean="mean", sd="std", n="count").reset_index()
    summary["sem"] = summary.sd / np.sqrt(summary.n)
    summary["lower"] = summary["mean"] - summary["sem"]
    summary["upper"] = summary["mean"] + summary["sem"]
    saved_summary = pd.read_csv(source / "tables/neural_curve_summary.csv", dtype={"date": str})
    saved_summary = saved_summary.loc[saved_summary.sample_id.isin(STRAINS) & saved_summary.date.eq(DATE)
                                      & saved_summary.neuron_class.isin(CELLS)
                                      & saved_summary.time_s.between(0, 24)]
    left = summary.set_index(mean_keys)[["mean", "sem", "n"]].sort_index()
    right = saved_summary.set_index(mean_keys)[["mean", "sem", "n"]].sort_index()
    if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-12, atol=1e-14):
        raise ValueError("One-second means/SEMs do not match the saved summaries.")

    pairs = pd.read_csv(source / "tables/neural_paired_stages.csv", dtype={"date": str})
    pairs = pairs.loc[pairs.sample_first.eq("A022") & pairs.sample_second.eq("A023")
                      & pairs.date.eq(DATE) & pairs.neuron_class.isin(CELLS)
                      & pairs.stage.eq("0-25s")].copy()
    pair_keys = ["date", "animal_id", "neuron_class"]
    if pairs.duplicated(pair_keys).any():
        raise ValueError("Duplicate animal-level paired difference.")
    animal_means = curves.groupby(keys).response.mean().unstack("sample_id")
    from_curves = animal_means.A023 - animal_means.A022
    cached = pairs.set_index(pair_keys).difference
    aligned = from_curves.reindex(cached.index)
    if not np.array_equal(aligned.isna(), cached.isna()) or not np.allclose(
            aligned, cached, equal_nan=True, rtol=1e-12, atol=1e-14):
        raise ValueError("Paired 0–25 s differences do not match the original one-second curves.")
    pair_stats = []
    saved_pairs = pd.read_csv(source / "tables/neural_pair_stage_summary.csv", dtype={"date": str})
    for cell in CELLS:
        sets = [set(curves.loc[curves.sample_id.eq(strain) & curves.neuron_class.eq(cell), "animal_id"])
                for strain in STRAINS]
        if sets[0] != sets[1]:
            raise ValueError("The two strain curves do not share the same animal support.")
        d = pairs.loc[pairs.neuron_class.eq(cell)].difference.dropna()
        existing = saved_pairs.loc[saved_pairs.sample_first.eq("A022") & saved_pairs.sample_second.eq("A023")
                                    & saved_pairs.date.eq(DATE) & saved_pairs.neuron_class.eq(cell)
                                    & saved_pairs.stage.eq("0-25s")].iloc[0]
        if len(d) != existing.n or not np.isclose(d.mean(), existing["mean"], rtol=1e-12):
            raise ValueError("Paired summary differs from the saved record.")
        pair_stats.append(dict(cell=cell, n=len(d), mean=d.mean(),
                               n_negative=int(d.lt(0).sum()), n_positive=int(d.gt(0).sum())))
    return chemistry, joint, counts, rms, curves, summary, pairs, pd.DataFrame(pair_stats), hashes


def plot_sample_comparison(source_dir, output_dir):
    """Draw the requested exploratory example using only saved selected inputs."""
    source, out = Path(source_dir), Path(output_dir)
    chemistry, joint, counts, rms, curves, summary, pairs, pair_stats, hashes = _load_selected(source)
    out.mkdir(parents=True, exist_ok=True)
    chemistry_limits = (-11., 15.5)
    difference_limit = np.ceil(pairs.difference.abs().max() * 10) / 10
    curve_limits = {}
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#AAB7C2", "axes.linewidth": .8,
             "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": MUTED, "ytick.color": MUTED,
             "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(17.8, 10.4), facecolor="white")
        fig.text(.039, .946, "A022 and A023 differ across sensory neurons", fontsize=25, weight="bold")
        fig.text(.040, .903, "Bacteroides stercoris", fontstyle="italic", fontsize=13, color=MUTED)
        handles = [Line2D([0], [0], color=color, lw=2.7, label=strain)
                   for strain, color in zip(STRAINS, COLORS)]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.966, .927),
                   ncol=2, frameon=False, handlelength=2.4, columnspacing=2.0, fontsize=13)

        fig.text(.040, .838, "A", fontsize=18, weight="bold")
        fig.text(.075, .838, "Reference chemistry", fontsize=16, weight="bold")
        fig.text(.075, .805, "332 jointly reported features", fontsize=10.5, color=MUTED)
        ax = fig.add_axes([.083, .313, .282, .48265])
        ax.plot(chemistry_limits, chemistry_limits, color="#A7B4BF", lw=1.1, zorder=1)
        ax.scatter(joint.a_log2fc, joint.b_log2fc, s=18, color="#536F86", alpha=.62,
                   edgecolors="white", linewidths=.3, zorder=3)
        ax.set(xlim=chemistry_limits, ylim=chemistry_limits, aspect="equal",
               xticks=[-10, -5, 0, 5, 10, 15], yticks=[-10, -5, 0, 5, 10, 15],
               xlabel="A022 log₂FC", ylabel="A023 log₂FC")
        ax.xaxis.label.set_color(COLORS[0])
        ax.yaxis.label.set_color(COLORS[1])
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.tick_params(labelsize=10, length=3.5, pad=5)
        fig.text(.083, .243, "Additional report coverage", fontsize=11, color=MUTED)
        for x, label, number, color in [(.114, "A022 only", counts["only_a_reported"], COLORS[0]),
                                          (.221, "A023 only", counts["only_b_reported"], COLORS[1]),
                                          (.327, "Neither", counts["neither_reported"], MUTED)]:
            fig.text(x, .208, label, fontsize=10.5, ha="center", color=color)
            fig.text(x, .169, str(number), fontsize=18, ha="center", color=INK)
        fig.add_artist(Line2D([.410, .410], [.139, .851], transform=fig.transFigure, color=RULE, lw=.8))

        fig.text(.451, .838, "B", fontsize=18, weight="bold")
        fig.text(.488, .838, "Calcium responses", fontsize=16, weight="bold")
        fig.text(.965, .838, "1-s mean ± SEM", fontsize=10.5, color=MUTED, ha="right")
        positions = (.495, .767)
        width = .196
        for cell, x in zip(CELLS, positions):
            ax = fig.add_axes([x, .527, width, .247])
            d = summary.loc[summary.neuron_class.eq(cell)]
            lower, upper = min(0., d.lower.min()), max(0., d.upper.max())
            span = upper - lower
            ax.set_ylim(lower - .10 * span, upper + .12 * span)
            curve_limits[cell] = list(ax.get_ylim())
            ax.axvspan(0, 10, color="#E7ECF1", alpha=.75, linewidth=0, zorder=-3)
            ax.axhline(0, color="#A8B4BF", lw=.8, zorder=-2)
            for strain, color in zip(STRAINS, COLORS):
                p = d.loc[d.sample_id.eq(strain)].sort_values("time_s")
                ax.fill_between(p.time_s, p.lower, p.upper, color=color, alpha=.16, linewidth=0)
                ax.plot(p.time_s, p["mean"], color=color, lw=2.4)
            n = int(pair_stats.loc[pair_stats.cell.eq(cell), "n"].item())
            ax.set_title(f"{cell}  ·  n = {n}", fontsize=14, pad=12)
            ax.set(xlim=(0, 25), xticks=[0, 10, 25], xlabel="Time (s)", ylabel="ΔF/F₀")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=10, length=3.5, pad=4)
        fig.text(.965, .464, "Cell-specific y scales", fontsize=9.5, color=MUTED, ha="right")

        fig.text(.451, .401, "C", fontsize=18, weight="bold")
        fig.text(.488, .401, "Paired animal differences", fontsize=16, weight="bold")
        fig.text(.488, .368, "A023 − A022  ·  Mean response over 0–25 s", fontsize=10.5, color=MUTED)
        offsets = {}
        for cell, x in zip(CELLS, positions):
            ax = fig.add_axes([x, .122, width, .206])
            d = pairs.loc[pairs.neuron_class.eq(cell) & pairs.difference.notna()].sort_values("animal_id")
            jitter = np.linspace(-.10, .10, len(d))
            offsets[cell] = dict(zip(d.animal_id, jitter.tolist()))
            ax.axhline(0, color="#9AAAB8", lw=1.1, zorder=1)
            ax.scatter(jitter, d.difference, s=60, color="#718BA2", edgecolors="white",
                       linewidth=.9, zorder=4)
            ax.plot([-.16, .16], [d.difference.mean()] * 2, color=INK, lw=3.0, zorder=3)
            ax.set(xlim=(-.43, .43), ylim=(-difference_limit, difference_limit),
                   xticks=[0], xticklabels=[cell], yticks=[-difference_limit, 0, difference_limit],
                   ylabel="Mean ΔF/F₀ difference")
            ax.tick_params(axis="x", length=0, labelsize=12, pad=8)
            ax.tick_params(axis="y", labelsize=10, length=3.5, pad=4)
            ax.spines["bottom"].set_visible(False)
        difference_handles = [Line2D([0], [0], color="#718BA2", marker="o", markersize=6,
                                     linestyle="none", label="Animal"),
                              Line2D([0], [0], color=INK, lw=3, label="Mean")]
        fig.legend(handles=difference_handles, loc="lower right", bbox_to_anchor=(.967, .027),
                   frameon=False, ncol=2, fontsize=10, handlelength=1.7, columnspacing=1.8)
        paths = {}
        for extension in ("png", "pdf", "svg"):
            path = out / f"sample_comparison_draft.{extension}"
            fig.savefig(path, dpi=220)
            paths[extension] = str(path.resolve())
        plt.close(fig)

    chemistry.to_csv(out / "chemical_pair_all_features.csv", index=False)
    joint.to_csv(out / "chemical_scatter_values.csv", index=False)
    pd.DataFrame([dict(report_state=key, n=value) for key, value in counts.items()]).to_csv(
        out / "chemical_report_coverage.csv", index=False)
    curves.to_csv(out / "selected_animal_1s_curves.csv", index=False)
    summary.to_csv(out / "plotted_1s_mean_sem.csv", index=False)
    pairs.to_csv(out / "paired_animal_differences.csv", index=False)
    pair_stats.to_csv(out / "paired_difference_summary.csv", index=False)
    parameters = {
        "source_dir": str(source.resolve()), "source_sha256": hashes, "code_sha256": _sha256(__file__),
        "strains": list(STRAINS), "cells": list(CELLS), "neural_date": DATE,
        "reference_group": "A050", "selection_rule": "Previously explored A022/A023 comparison; ASH and AWCON were selected after examining their contrasting paired-response directions. This is an illustrative, descriptive case.",
        "chemical_mask": "Original report jointly reported in A022 and A023; 332 pair-specific features. The fixed three-strain in_common_set mask is not used.",
        "chemical_report_counts": counts, "chemical_joint_rms_log2fc": rms,
        "chemical_axes": {"limits": list(chemistry_limits), "aspect": "equal", "reference_line": "y=x"},
        "neural_window_seconds": [0, 25], "plotted_time_points": list(range(25)),
        "mean_sem": "Animal means at each original 1-s sampling point; SEM=sample SD(ddof=1)/sqrt(n), after pre-existing within-animal trial averages. No smoothing, rebaselining, or normalization.",
        "paired_difference": "For each cell and the same date|animal, mean(A023 over 0–24 s) minus mean(A022 over 0–24 s); use the existing 0–25 s stage results, verified against the 1-s curves.",
        "missingness": "Chemical one-sided and jointly unreported features are counted separately and excluded from the scatter. Neural missing animals remain NaN in exports and are not plotted.",
        "curve_y_limits": curve_limits, "paired_difference_y_limits": [-float(difference_limit), float(difference_limit)],
        "animal_point_offsets": offsets, "outputs": paths,
    }
    (out / "plot_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    caption = (
        "A022 and A023 differ across sensory neurons. The previously explored Bacteroides stercoris "
        "pair is shown with ASH/AWCON, selected after observing their contrasting paired-response "
        "directions. A, each point is one "
        "of 332 jointly reported chemical features; the line is y=x on equal log₂FC axes "
        f"(RMS difference {rms:.3f}). The other 48 features are counted by report status; missing "
        "reports are not plotted as zeros. Existing FCs retain upstream fill-zero/+1 processing "
        "and reference A050. Chemistry comes from separate cultures, not the neural-stimulus aliquots, "
        "and has no biological-replicate uncertainty here. B, original 1-s responses show animal "
        "means ± pointwise SEM after within-animal trial averaging, without additional smoothing "
        "or baseline correction. Gray shading marks 0–10 s stimulation; y scales differ between "
        "cells. C, each point is the same animal's A023−A022 mean response difference over [0,25) s; "
        "short bars indicate means, and both panels use the same y scale. ASH has six paired animals "
        "and AWCON five; one missing AWCON animal is excluded. B and C use the same recordings "
        "from 2026-06-01, not independent validation. Differences are descriptive under the existing "
        "stimulus sequence and do not establish chemical causation or batch-independent strain effects.\n"
    )
    (out / "caption.txt").write_text(caption)
    verification = {
        "chemical_scatter_points": len(joint), "chemical_report_counts": counts,
        "one_second_animal_rows": len(curves), "mean_sem_points": len(summary),
        "paired_rows_including_missing": len(pairs), "finite_paired_points": int(pairs.difference.notna().sum()),
        "pair_summary": pair_stats.to_dict("records"),
        "checks_passed": ["Source SHA256 matches existing output manifest",
                          "Chemical original report flags and pair-specific joint mask agree",
                          "Joint-report RMS agrees with saved pair summary",
                          "One-second animal means and SEMs agree with cached summaries",
                          "Both strains use the same animals within each cell",
                          "Original one-second paired means agree with existing stage differences, including NaN mask"],
        "source_files_unchanged": hashes == {name: _sha256(source / name) for name in SOURCE_FILES},
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return paths
