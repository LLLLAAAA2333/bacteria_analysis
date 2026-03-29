"""Output writers and figures for Stage 1 reliability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bacteria_analysis.io import write_json, write_parquet


def ensure_stage1_output_dirs(output_root: str | Path) -> dict[str, Path]:
    """Create and return the standard Stage 1 output tree."""

    root = Path(output_root)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    qc_dir = root / "qc"

    for directory in (root, tables_dir, figures_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": root,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "qc_dir": qc_dir,
    }


def _prepare_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        if prepared[column].dtype != "object":
            continue
        prepared[column] = prepared[column].map(
            lambda value: "|".join(map(str, value)) if isinstance(value, (list, tuple, set)) else value
        )
    return prepared


def _write_markdown_summary(summary: dict[str, Any], path: str | Path) -> Path:
    lines = [
        "# Stage 1 Reliability Run Summary",
        "",
        "## Dataset",
        f"- Trials: {summary['n_trials']}",
        f"- Individuals: {summary['n_individuals']}",
        f"- Dates: {summary['n_dates']}",
        f"- Stimuli: {summary['n_stimuli']}",
        "",
        "## Primary View",
        f"- Primary view: {summary['primary_view']}",
        f"- Distance gap: {summary['primary_view_distance_gap']:.4f}",
        f"- LOIO mean accuracy: {summary['primary_view_loio_accuracy_mean']:.4f}",
        f"- LODO mean accuracy: {summary['primary_view_lodo_accuracy_mean']:.4f}",
        "",
        "## Strongest Views",
    ]
    strongest_views = summary.get("strongest_views", [])
    if strongest_views:
        for row in strongest_views:
            lines.append(f"- {row['view_name']}: distance_gap={row['distance_gap']:.4f}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Output Paths",
            f"- Final summary table: {summary['final_summary_path']}",
            f"- Figures directory: {summary['figures_dir']}",
            f"- QC directory: {summary['qc_dir']}",
        ]
    )

    output_path = Path(path)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _plot_same_vs_different(
    comparisons: pd.DataFrame,
    final_summary: pd.DataFrame,
    primary_view: str,
    path: Path,
) -> Path:
    valid = comparisons[
        (comparisons["comparison_status"] == "ok") & (comparisons["view_name"].astype(str) == primary_view)
    ].copy()
    valid["comparison_label"] = valid["same_stimulus"].map({True: "same", False: "different"})
    primary_row = final_summary.loc[final_summary["view_name"].astype(str) == primary_view].iloc[0]
    plt.figure(figsize=(6.5, 4.5))
    ax = sns.boxplot(
        data=valid,
        x="comparison_label",
        y="distance",
        order=["same", "different"],
        showfliers=False,
    )
    ax.text(
        0.02,
        0.98,
        f"permutation p = {float(primary_row['p_value']):.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    plt.xlabel("Comparison")
    plt.ylabel("Distance")
    plt.title(f"Same vs Different Trial Distances ({primary_view})")
    return _save_figure(path)


def _plot_holdout_summary(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    primary_view: str | None = None,
) -> Path:
    plot_summary = summary.copy()
    if primary_view is not None:
        plot_summary = plot_summary.loc[plot_summary["view_name"].astype(str) == primary_view].copy()
    plt.figure(figsize=(5.5 if primary_view is not None else 8, 4.5))
    ax = sns.barplot(data=plot_summary, x="view_name", y="accuracy_mean", color="#4c78a8")
    plt.ylim(0, 1)
    if primary_view is not None and not plot_summary.empty:
        value = float(plot_summary.iloc[0]["accuracy_mean"])
        ax.text(0, value + 0.03, f"{value:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("Primary View" if primary_view is not None else "View")
    plt.ylabel("Accuracy")
    plt.title(f"{title} ({primary_view})" if primary_view is not None else title)
    return _save_figure(path)


def _plot_cross_view(final_summary: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=final_summary, x="view_name", y="distance_gap", color="#f58518")
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("View")
    plt.ylabel("Different - Same Distance")
    plt.title("Cross-View Reliability Comparison")
    return _save_figure(path)


def _plot_within_date_cross_individual_same_vs_different(comparisons: pd.DataFrame, path: Path) -> Path:
    valid = comparisons[comparisons["comparison_status"] == "ok"].copy()
    valid["comparison_label"] = valid["same_stimulus"].map({True: "same", False: "different"})
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=valid, x="view_name", y="distance", hue="comparison_label", showfliers=False)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("View")
    plt.ylabel("Distance")
    plt.title("Within-Date Cross-Individual Same vs Different Distances")
    return _save_figure(path)


def _plot_overlap_summary(comparisons: pd.DataFrame, primary_view: str, path: Path) -> Path:
    valid = comparisons[
        (comparisons["comparison_status"] == "ok") & (comparisons["view_name"].astype(str) == primary_view)
    ].copy()
    plt.figure(figsize=(5.5, 4.5))
    sns.boxplot(data=valid, x="view_name", y="overlap_neuron_count", color="#54a24b", showfliers=False)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("Primary View")
    plt.ylabel("Overlap Neuron Count")
    plt.title(f"Overlap-Neuron QC ({primary_view})")
    return _save_figure(path)


def _plot_per_date_loio_overview(summary: pd.DataFrame, path: Path) -> Path:
    pivot = (
        summary.pivot(index="source_date", columns="view_name", values="accuracy_mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    plt.figure(figsize=(8, max(3.5, 0.6 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("View")
    plt.ylabel("Date")
    plt.title("Per-Date LOIO Accuracy Overview")
    return _save_figure(path)


def _build_stimulus_distance_matrix_frames(pair_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for view_name, group in pair_summary.groupby("view_name", sort=False, dropna=False):
        forward = group[["stimulus_left", "stimulus_right", "mean_distance"]].rename(
            columns={"stimulus_left": "stimulus_row", "stimulus_right": "stimulus_column"}
        )
        reverse = (
            group.loc[group["stimulus_left"] != group["stimulus_right"], ["stimulus_right", "stimulus_left", "mean_distance"]]
            .rename(columns={"stimulus_right": "stimulus_row", "stimulus_left": "stimulus_column"})
        )
        matrix_long = pd.concat([forward, reverse], ignore_index=True)
        stimuli = sorted(set(matrix_long["stimulus_row"]).union(matrix_long["stimulus_column"]))
        matrix = matrix_long.pivot(index="stimulus_row", columns="stimulus_column", values="mean_distance")
        matrix = matrix.reindex(index=stimuli, columns=stimuli)
        frames[str(view_name)] = matrix.reset_index()
    return frames


def _plot_stimulus_distance_matrix(matrix_frame: pd.DataFrame, view_name: str, path: Path) -> Path:
    heatmap_frame = matrix_frame.set_index("stimulus_row")
    size = max(6.5, 0.45 * len(heatmap_frame.columns) + 2.5)
    plt.figure(figsize=(size, size))
    sns.heatmap(heatmap_frame, cmap="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Stimulus")
    plt.ylabel("Stimulus")
    plt.title(f"Stimulus-by-Stimulus Distance Matrix ({view_name})")
    return _save_figure(path)


def build_run_summary(
    core_outputs: dict[str, pd.DataFrame],
    stats_outputs: dict[str, pd.DataFrame],
    written_paths: dict[str, Path],
    primary_view: str,
) -> dict[str, Any]:
    metadata_summary = core_outputs["metadata_summary"].iloc[0]
    final_summary = stats_outputs["final_summary"].sort_values("distance_gap", ascending=False, kind="stable")
    strongest = final_summary[["view_name", "distance_gap"]].head(2).to_dict(orient="records")
    primary_view_row = final_summary.loc[final_summary["view_name"] == primary_view].iloc[0]
    return {
        "n_trials": int(metadata_summary["n_trials"]),
        "n_individuals": int(metadata_summary["n_individuals"]),
        "n_dates": int(metadata_summary["n_dates"]),
        "n_stimuli": int(metadata_summary["n_stimuli"]),
        "primary_view": primary_view,
        "primary_view_distance_gap": float(primary_view_row["distance_gap"]),
        "primary_view_loio_accuracy_mean": float(primary_view_row["loio_accuracy_mean"]),
        "primary_view_lodo_accuracy_mean": float(primary_view_row["lodo_accuracy_mean"]),
        "strongest_views": strongest,
        "final_summary_path": str(written_paths["final_summary"]),
        "figures_dir": str(written_paths["figures_dir"]),
        "qc_dir": str(written_paths["qc_dir"]),
    }


def write_stage1_outputs(
    core_outputs: dict[str, pd.DataFrame],
    stats_outputs: dict[str, pd.DataFrame],
    output_root: str | Path,
    primary_view: str,
) -> dict[str, Path]:
    """Write all Stage 1 tables, QC outputs, figures, and summaries."""

    dirs = ensure_stage1_output_dirs(output_root)
    written: dict[str, Path] = {
        "figures_dir": dirs["figures_dir"],
        "qc_dir": dirs["qc_dir"],
    }
    final_summary_with_primary = stats_outputs["final_summary"].copy()
    final_summary_with_primary["is_primary_view"] = final_summary_with_primary["view_name"].astype(str) == primary_view

    for view_name, group in core_outputs["comparisons"].groupby("view_name", sort=False, dropna=False):
        written[f"comparisons_{view_name}"] = write_parquet(
            _prepare_for_parquet(group.reset_index(drop=True)),
            dirs["tables_dir"] / f"comparisons__{view_name}.parquet",
        )

    table_names = (
        "same_vs_different_summary",
        "loio_trials",
        "loio_groups",
        "loio_summary",
        "lodo_trials",
        "lodo_groups",
        "lodo_summary",
        "split_half_results",
        "split_half_summary",
        "permutation_iterations",
        "permutation_summary",
        "bootstrap_iterations",
        "bootstrap_summary",
        "final_summary",
    )
    for table_name in table_names:
        table_frame = stats_outputs.get(table_name, core_outputs.get(table_name)).reset_index(drop=True)
        if table_name == "final_summary":
            table_frame = final_summary_with_primary.reset_index(drop=True)
        written[table_name] = write_parquet(
            _prepare_for_parquet(table_frame),
            dirs["tables_dir"] / f"{table_name}.parquet",
        )

    written["reliability_summary"] = write_parquet(
        _prepare_for_parquet(final_summary_with_primary.reset_index(drop=True)),
        dirs["tables_dir"] / "reliability_summary.parquet",
    )
    written["primary_view_summary"] = write_parquet(
        _prepare_for_parquet(
            final_summary_with_primary.loc[final_summary_with_primary["is_primary_view"]].reset_index(drop=True)
        ),
        dirs["tables_dir"] / "primary_view_summary.parquet",
    )
    written["leave_one_individual_out"] = write_parquet(
        _prepare_for_parquet(core_outputs["loio_groups"].reset_index(drop=True)),
        dirs["tables_dir"] / "leave_one_individual_out.parquet",
    )
    written["leave_one_date_out"] = write_parquet(
        _prepare_for_parquet(core_outputs["lodo_groups"].reset_index(drop=True)),
        dirs["tables_dir"] / "leave_one_date_out.parquet",
    )
    written["split_half_reliability"] = write_parquet(
        _prepare_for_parquet(core_outputs["split_half_results"].reset_index(drop=True)),
        dirs["tables_dir"] / "split_half_reliability.parquet",
    )
    written["permutation_null"] = write_parquet(
        _prepare_for_parquet(stats_outputs["permutation_iterations"].reset_index(drop=True)),
        dirs["tables_dir"] / "permutation_null.parquet",
    )
    written["grouped_bootstrap"] = write_parquet(
        _prepare_for_parquet(stats_outputs["bootstrap_iterations"].reset_index(drop=True)),
        dirs["tables_dir"] / "grouped_bootstrap.parquet",
    )
    written["within_date_cross_individual_comparisons"] = write_parquet(
        _prepare_for_parquet(core_outputs["within_date_cross_individual_comparisons"].reset_index(drop=True)),
        dirs["tables_dir"] / "within_date_cross_individual_comparisons.parquet",
    )
    written["within_date_cross_individual_same_vs_different_summary"] = write_parquet(
        _prepare_for_parquet(core_outputs["within_date_cross_individual_summary"].reset_index(drop=True)),
        dirs["tables_dir"] / "within_date_cross_individual_same_vs_different_summary.parquet",
    )
    written["per_date_loio_trials"] = write_parquet(
        _prepare_for_parquet(core_outputs["per_date_loio_trials"].reset_index(drop=True)),
        dirs["tables_dir"] / "per_date_loio_trials.parquet",
    )
    written["per_date_loio_groups"] = write_parquet(
        _prepare_for_parquet(core_outputs["per_date_loio_groups"].reset_index(drop=True)),
        dirs["tables_dir"] / "per_date_loio_groups.parquet",
    )
    written["per_date_loio_summary"] = write_parquet(
        _prepare_for_parquet(core_outputs["per_date_loio_summary"].reset_index(drop=True)),
        dirs["tables_dir"] / "per_date_loio_summary.parquet",
    )
    written["stimulus_distance_pairs"] = write_parquet(
        _prepare_for_parquet(core_outputs["stimulus_distance_pairs"].reset_index(drop=True)),
        dirs["tables_dir"] / "stimulus_distance_pairs.parquet",
    )
    stimulus_distance_matrices = _build_stimulus_distance_matrix_frames(core_outputs["stimulus_distance_pairs"])
    for view_name, matrix_frame in stimulus_distance_matrices.items():
        written[f"stimulus_distance_matrix_{view_name}"] = write_parquet(
            _prepare_for_parquet(matrix_frame.reset_index(drop=True)),
            dirs["tables_dir"] / f"stimulus_distance_matrix__{view_name}.parquet",
        )

    written["overlap_qc_summary"] = write_parquet(
        _prepare_for_parquet(core_outputs["overlap_qc_summary"].reset_index(drop=True)),
        dirs["qc_dir"] / "overlap_neuron_summary.parquet",
    )
    written["overlap_neuron_counts"] = write_parquet(
        _prepare_for_parquet(core_outputs["overlap_qc_summary"].reset_index(drop=True)),
        dirs["qc_dir"] / "overlap_neuron_counts.parquet",
    )
    written["excluded_comparisons"] = write_parquet(
        _prepare_for_parquet(
            core_outputs["comparisons"]
            .loc[core_outputs["comparisons"]["comparison_status"] != "ok"]
            .reset_index(drop=True)
        ),
        dirs["qc_dir"] / "excluded_comparisons.parquet",
    )
    excluded_holdout = pd.concat(
        [
            core_outputs["loio_trials"].loc[core_outputs["loio_trials"]["score_status"] != "scored"],
            core_outputs["lodo_trials"].loc[core_outputs["lodo_trials"]["score_status"] != "scored"],
        ],
        ignore_index=True,
    )
    written["excluded_holdout_trials"] = write_parquet(
        _prepare_for_parquet(excluded_holdout.reset_index(drop=True)),
        dirs["qc_dir"] / "excluded_holdout_trials.parquet",
    )

    written["same_vs_different_figure"] = _plot_same_vs_different(
        core_outputs["comparisons"],
        final_summary_with_primary,
        primary_view,
        dirs["figures_dir"] / "same_vs_different_distributions.png",
    )
    written["loio_figure"] = _plot_holdout_summary(
        core_outputs["loio_summary"].loc[core_outputs["loio_summary"]["holdout_type"] == "individual"],
        dirs["figures_dir"] / "leave_one_individual_out_summary.png",
        title="Leave-One-Individual-Out Accuracy",
        primary_view=primary_view,
    )
    written["lodo_figure"] = _plot_holdout_summary(
        core_outputs["lodo_summary"].loc[core_outputs["lodo_summary"]["holdout_type"] == "date"],
        dirs["figures_dir"] / "leave_one_date_out_summary.png",
        title="Leave-One-Date-Out Accuracy",
        primary_view=primary_view,
    )
    written["split_half_figure"] = _plot_holdout_summary(
        core_outputs["split_half_summary"],
        dirs["figures_dir"] / "split_half_summary.png",
        title="Split-Half Accuracy",
    )
    written["cross_view_figure"] = _plot_cross_view(
        stats_outputs["final_summary"],
        dirs["figures_dir"] / "cross_view_reliability_comparison.png",
    )
    written["within_date_cross_individual_figure"] = _plot_within_date_cross_individual_same_vs_different(
        core_outputs["within_date_cross_individual_comparisons"],
        dirs["figures_dir"] / "within_date_cross_individual_same_vs_different.png",
    )
    written["per_date_loio_figure"] = _plot_per_date_loio_overview(
        core_outputs["per_date_loio_summary"],
        dirs["figures_dir"] / "per_date_loio_overview.png",
    )
    written["overlap_qc_figure"] = _plot_overlap_summary(
        core_outputs["comparisons"],
        primary_view,
        dirs["figures_dir"] / "overlap_neuron_qc_summary.png",
    )
    for view_name, matrix_frame in stimulus_distance_matrices.items():
        written[f"stimulus_distance_matrix_figure_{view_name}"] = _plot_stimulus_distance_matrix(
            matrix_frame,
            view_name,
            dirs["figures_dir"] / f"stimulus_distance_matrix__{view_name}.png",
        )

    stats_outputs_with_primary = dict(stats_outputs)
    stats_outputs_with_primary["final_summary"] = final_summary_with_primary
    summary = build_run_summary(core_outputs, stats_outputs_with_primary, written, primary_view=primary_view)
    written["run_summary_json"] = write_json(summary, dirs["output_root"] / "run_summary.json")
    written["run_summary_md"] = _write_markdown_summary(summary, dirs["output_root"] / "run_summary.md")
    return written
