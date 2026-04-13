"""Output writers and figures for geometry analysis."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from bacteria_analysis.io import write_json, write_parquet

PRIMARY_GEOMETRY_VIEW = "response_window"
PRIMARY_SUPPORT_ARTIFACT = "stimulus_overlap__date"
OVERLAP_FIGURE_TITLES = {
    "stimulus_overlap__date": "Stimulus Overlap By Date",
    "stimulus_overlap__individual": "Stimulus Overlap By Individual",
}


def ensure_geometry_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    return _mkdir_geometry_dirs(root)


def write_geometry_outputs(core_outputs: dict[str, pd.DataFrame], output_root: str | Path) -> dict[str, Path]:
    dirs = ensure_geometry_output_dirs(output_root)
    return _write_geometry_artifacts(core_outputs, dirs)


def _mkdir_geometry_dirs(root: Path) -> dict[str, Path]:
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


def _write_geometry_artifacts(core_outputs: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Path]:
    _remove_stale_geometry_figures(dirs["figures_dir"])
    written: dict[str, Path] = {
        "output_root": dirs["output_root"],
        "tables_dir": dirs["tables_dir"],
        "figures_dir": dirs["figures_dir"],
        "qc_dir": dirs["qc_dir"],
    }

    for artifact_name, frame in core_outputs.items():
        if _is_non_pooled_matrix_artifact(artifact_name):
            continue
        output_dir = dirs["qc_dir"] if _is_qc_artifact(artifact_name) else dirs["tables_dir"]
        written[artifact_name] = write_parquet(_prepare_for_parquet(frame), output_dir / f"{artifact_name}.parquet")

    focus_view = _choose_focus_view(core_outputs)
    focus_matrix_name = f"rdm_matrix__{focus_view}__pooled" if focus_view else None
    if focus_matrix_name:
        matrix_frame = core_outputs.get(focus_matrix_name)
        if matrix_frame is not None:
            written[f"{focus_matrix_name}_figure"] = _plot_rdm_matrix(
                matrix_frame,
                title=f"Pooled RDM ({focus_view})",
                path=dirs["figures_dir"] / f"{focus_matrix_name}.png",
            )

    support_artifact_name = _choose_support_artifact(core_outputs)
    if support_artifact_name:
        frame = core_outputs.get(support_artifact_name)
        if frame is not None:
            written[f"{support_artifact_name}_figure"] = _plot_stimulus_overlap(
                frame,
                title=OVERLAP_FIGURE_TITLES.get(support_artifact_name, support_artifact_name),
                path=dirs["figures_dir"] / f"{support_artifact_name}.png",
            )

    summary = _build_run_summary(
        core_outputs,
        written,
        focus_view=focus_view,
        support_artifact_name=support_artifact_name,
    )
    written["run_summary_json"] = write_json(summary, dirs["output_root"] / "run_summary.json")
    written["run_summary_md"] = _write_markdown_summary(summary, dirs["output_root"] / "run_summary.md")
    return written


def _prepare_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        if prepared[column].dtype != "object":
            continue
        prepared[column] = prepared[column].map(
            lambda value: "|".join(map(str, value)) if isinstance(value, (list, tuple, set)) else value
        )
    return prepared


def _plot_rdm_matrix(
    matrix_frame: pd.DataFrame,
    title: str,
    path: Path,
    *,
    reorder_by_similarity: bool = False,
) -> Path:
    heatmap_frame = _build_rdm_heatmap_frame(matrix_frame, reorder_by_similarity=reorder_by_similarity)
    if heatmap_frame.empty or len(heatmap_frame.columns) == 0:
        plt.figure(figsize=(6, 4.5))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        return _save_figure(path)
    size = max(5.5, 0.75 * len(heatmap_frame.columns) + 2.0)
    plt.figure(figsize=(size, size))
    sns.heatmap(heatmap_frame, cmap="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Stimulus Name")
    plt.ylabel("Stimulus Name")
    plt.title(title)
    return _save_figure(path)


def _build_rdm_heatmap_frame(
    matrix_frame: pd.DataFrame,
    *,
    reorder_by_similarity: bool = False,
) -> pd.DataFrame:
    heatmap_frame = matrix_frame.set_index("stimulus_row").copy()
    if heatmap_frame.empty:
        return heatmap_frame

    heatmap_frame.index = heatmap_frame.index.astype(str)
    heatmap_frame.columns = heatmap_frame.columns.astype(str)
    if set(heatmap_frame.index) != set(heatmap_frame.columns):
        raise ValueError("pooled RDM heatmap requires matching stimulus_row and column labels")
    heatmap_frame = heatmap_frame.reindex(columns=heatmap_frame.index)

    if reorder_by_similarity:
        heatmap_frame = _cluster_reorder_heatmap_frame(heatmap_frame)

    display_labels = _build_stimulus_display_labels(
        list(heatmap_frame.index),
        matrix_frame.attrs.get("stimulus_name_map"),
    )
    labeled = heatmap_frame.copy()
    labeled.index = display_labels
    labeled.columns = display_labels
    return labeled


def _plot_stimulus_overlap(matrix_frame: pd.DataFrame, title: str, path: Path) -> Path:
    heatmap_frame = _build_overlap_heatmap_frame(matrix_frame)
    if heatmap_frame.empty or len(heatmap_frame.columns) == 0:
        plt.figure(figsize=(6, 4.5))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        return _save_figure(path)

    width = max(7.0, 0.6 * len(heatmap_frame.columns) + 2.0)
    height = max(4.5, 0.3 * len(heatmap_frame.index) + 2.0)
    plt.figure(figsize=(width, height))
    cmap = sns.color_palette(["#f2f2f2", "#4c78a8"], as_cmap=True)
    sns.heatmap(
        heatmap_frame,
        cmap=cmap,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0.2,
        linecolor="#ffffff",
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Stimulus Name")
    plt.ylabel(str(matrix_frame.attrs.get("group_axis_label", "Group")))
    plt.title(title)
    return _save_figure(path)


def _build_overlap_heatmap_frame(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    heatmap_frame = matrix_frame.set_index("group_id").copy()
    if heatmap_frame.empty:
        return heatmap_frame

    heatmap_frame.index = heatmap_frame.index.astype(str)
    heatmap_frame.columns = heatmap_frame.columns.astype(str)
    numeric = heatmap_frame.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    display_labels = _build_stimulus_display_labels(
        list(numeric.columns),
        matrix_frame.attrs.get("stimulus_name_map"),
    )
    group_display_labels = matrix_frame.attrs.get("group_display_labels") or {}
    labeled = numeric.copy()
    labeled.index = [str(group_display_labels.get(group_id, group_id)) for group_id in numeric.index]
    labeled.columns = display_labels
    return labeled


def _build_stimulus_display_labels(
    stimulus_labels: list[str],
    stimulus_name_map: dict[str, str] | None,
) -> list[str]:
    if not stimulus_name_map:
        return stimulus_labels

    display_labels = [str(stimulus_name_map.get(label, label)) for label in stimulus_labels]
    if len(set(display_labels)) != len(display_labels):
        raise ValueError("stim_name labels must be unique for pooled RDM visualization")
    return display_labels


def _cluster_reorder_heatmap_frame(heatmap_frame: pd.DataFrame) -> pd.DataFrame:
    if len(heatmap_frame.index) < 3:
        return heatmap_frame.copy()

    numeric = heatmap_frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("clustered pooled RDM heatmap requires a complete numeric matrix")

    distances = numeric.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(distances, 0.0)
    linkage_matrix = linkage(squareform(distances, checks=False), method="average", optimal_ordering=True)
    order = leaves_list(linkage_matrix).tolist()
    return heatmap_frame.iloc[order, order]


def _plot_similarity_summary(frame: pd.DataFrame, title: str, path: Path) -> Path:
    if frame.empty:
        plt.figure(figsize=(7, 4.5))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        return _save_figure(path)

    panels = _build_similarity_plot_panels(frame)
    fig, axes = plt.subplots(1, len(panels), figsize=(max(7, 4.5 * len(panels)), 4.5), squeeze=False)
    axes_flat = list(axes.flat)
    for axis, panel in zip(axes_flat, panels, strict=True):
        sns.barplot(data=panel["frame"], x="view_label", y="similarity", color="#4c78a8", ax=axis)
        axis.tick_params(axis="x", rotation=15)
        for label in axis.get_xticklabels():
            label.set_ha("right")
        axis.set_xlabel("Comparison")
        axis.set_ylabel("Similarity")
        axis.set_title(_format_similarity_scope_label(panel["comparison_scope"]))
    fig.suptitle(title)
    return _save_figure(path)


def _build_similarity_plot_panels(frame: pd.DataFrame) -> list[dict[str, object]]:
    plot_frame = frame.copy()
    plot_frame["view_label"] = plot_frame["view_name"].astype(str)
    if "reference_view_name" in plot_frame.columns:
        reference = plot_frame["reference_view_name"].astype(str)
        different_reference = reference != plot_frame["view_label"]
        plot_frame.loc[different_reference, "view_label"] = (
            plot_frame.loc[different_reference, "view_label"] + " vs " + reference.loc[different_reference]
        )
    plot_frame["similarity"] = pd.to_numeric(plot_frame["similarity"], errors="coerce")

    panels: list[dict[str, object]] = []
    for comparison_scope, scope_frame in plot_frame.groupby("comparison_scope", sort=False, dropna=False):
        summary = (
            scope_frame.groupby("view_label", sort=False, dropna=False)["similarity"]
            .median()
            .reset_index()
        )
        summary["comparison_scope"] = str(comparison_scope)
        panels.append(
            {
                "comparison_scope": str(comparison_scope),
                "frame": summary[["comparison_scope", "view_label", "similarity"]],
            }
        )
    return panels


def _format_similarity_scope_label(comparison_scope: str) -> str:
    return str(comparison_scope).replace("_", " ").title()


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _remove_stale_geometry_figures(figures_dir: Path) -> None:
    for pattern in (
        "rdm_matrix__*__pooled.png",
        "rdm_matrix__*__pooled__clustered.png",
        "rdm_stability_by_*.png",
        "rdm_view_comparison.png",
        "stimulus_overlap__*.png",
    ):
        for stale_figure in figures_dir.glob(pattern):
            stale_figure.unlink()


def _is_non_pooled_matrix_artifact(artifact_name: str) -> bool:
    return artifact_name.startswith("rdm_matrix__") and not artifact_name.endswith("__pooled")


def _is_qc_artifact(artifact_name: str) -> bool:
    return artifact_name == "rdm_group_coverage" or artifact_name.startswith("stimulus_overlap__")


def _build_run_summary(
    core_outputs: dict[str, pd.DataFrame],
    written: dict[str, Path],
    *,
    focus_view: str | None,
    support_artifact_name: str | None,
) -> dict[str, Any]:
    pair_table_names = sorted(name for name in core_outputs if name.startswith("rdm_pairs__"))
    pooled_matrix_names = sorted(name for name in core_outputs if name.startswith("rdm_matrix__") and name.endswith("__pooled"))
    views = _ordered_views_for_summary(core_outputs, pair_table_names, pooled_matrix_names)
    pooled_matrix_views = [view_name for view_name in views if f"rdm_matrix__{view_name}__pooled" in core_outputs]
    stability_table_names = sorted(name for name in core_outputs if name.startswith("rdm_stability_by_"))
    qc_table_names = sorted(name for name in core_outputs if _is_qc_artifact(name))
    primary_figure = f"rdm_matrix__{focus_view}__pooled" if focus_view else None
    figure_names = [name for name in (primary_figure, support_artifact_name) if name]
    pooled_cross_view = _summarize_cross_view(core_outputs.get("rdm_view_comparison"))
    individual_stability = _summarize_stability_table(core_outputs.get("rdm_stability_by_individual"))
    date_stability = _summarize_stability_table(core_outputs.get("rdm_stability_by_date"))
    date_overlap = _summarize_overlap_frame(core_outputs.get("stimulus_overlap__date"))
    individual_overlap = _summarize_overlap_frame(core_outputs.get("stimulus_overlap__individual"))

    return {
        "views": views,
        "focus_view": focus_view,
        "sensitivity_views": [view_name for view_name in views if view_name != focus_view],
        "pair_table_names": pair_table_names,
        "pooled_matrix_views": pooled_matrix_views,
        "stability_table_names": stability_table_names,
        "view_comparison_table": "rdm_view_comparison" if "rdm_view_comparison" in core_outputs else None,
        "qc_table_names": qc_table_names,
        "primary_figure": primary_figure,
        "support_figure": support_artifact_name,
        "figure_names": figure_names,
        "pooled_cross_view": pooled_cross_view,
        "individual_stability": individual_stability,
        "date_stability": date_stability,
        "date_overlap": date_overlap,
        "individual_overlap": individual_overlap,
        "tables_dir": str(written["tables_dir"]),
        "figures_dir": str(written["figures_dir"]),
        "qc_dir": str(written["qc_dir"]),
    }


def _ordered_views_for_summary(
    core_outputs: dict[str, pd.DataFrame],
    pair_table_names: list[str],
    pooled_matrix_names: list[str],
) -> list[str]:
    coverage = core_outputs.get("rdm_group_coverage")
    if coverage is not None and not coverage.empty and "view_name" in coverage.columns:
        return coverage["view_name"].astype(str).drop_duplicates().tolist()

    views: list[str] = []
    for artifact_name in list(core_outputs) + pair_table_names + pooled_matrix_names:
        parts = artifact_name.split("__")
        if len(parts) < 3:
            continue
        view_name = parts[1]
        if view_name not in views:
            views.append(view_name)
    return views


def _choose_focus_view(core_outputs: dict[str, pd.DataFrame]) -> str | None:
    pooled_matrix_names = sorted(name for name in core_outputs if name.startswith("rdm_matrix__") and name.endswith("__pooled"))
    views = _ordered_views_for_summary(core_outputs, [], pooled_matrix_names)
    pooled_views = [view_name for view_name in views if f"rdm_matrix__{view_name}__pooled" in core_outputs]
    if PRIMARY_GEOMETRY_VIEW in pooled_views:
        return PRIMARY_GEOMETRY_VIEW
    if pooled_views:
        return pooled_views[0]
    return None


def _choose_support_artifact(core_outputs: dict[str, pd.DataFrame]) -> str | None:
    if PRIMARY_SUPPORT_ARTIFACT in core_outputs:
        return PRIMARY_SUPPORT_ARTIFACT
    for artifact_name in ("stimulus_overlap__date", "stimulus_overlap__individual"):
        if artifact_name in core_outputs:
            return artifact_name
    return None


def _summarize_cross_view(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "similarity": None,
            "n_shared_entries": None,
            "valid_count": 0,
            "total_count": 0,
        }

    summary = frame.copy()
    summary["similarity"] = pd.to_numeric(summary.get("similarity"), errors="coerce")
    summary["n_shared_entries"] = pd.to_numeric(summary.get("n_shared_entries"), errors="coerce")
    finite = summary.loc[np.isfinite(summary["similarity"])]
    source = finite if not finite.empty else summary
    return {
        "similarity": _median_or_none(finite["similarity"]) if not finite.empty else None,
        "n_shared_entries": _median_or_none(source["n_shared_entries"]),
        "valid_count": int(len(finite)),
        "total_count": int(len(summary)),
    }


def _summarize_stability_table(frame: pd.DataFrame | None) -> dict[str, Any]:
    empty_summary = {
        "views": [],
        "within_group_median_by_view": {},
        "pooled_vs_group_median_by_view": {},
        "within_group_valid_count": 0,
        "within_group_total_count": 0,
        "pooled_vs_group_valid_count": 0,
        "pooled_vs_group_total_count": 0,
        "within_group_median_shared_entries": None,
        "pooled_vs_group_median_shared_entries": None,
    }
    if frame is None or frame.empty:
        return empty_summary

    summary = frame.copy()
    if "view_name" not in summary.columns or "comparison_scope" not in summary.columns:
        return empty_summary

    summary["view_name"] = summary["view_name"].astype(str)
    summary["comparison_scope"] = summary["comparison_scope"].astype(str)
    summary["similarity"] = pd.to_numeric(summary.get("similarity"), errors="coerce")
    summary["n_shared_entries"] = pd.to_numeric(summary.get("n_shared_entries"), errors="coerce")

    views = summary["view_name"].drop_duplicates().tolist()
    within_scope = summary.loc[summary["comparison_scope"] == "within_group_type"].copy()
    pooled_scope = summary.loc[summary["comparison_scope"] == "pooled_vs_group"].copy()
    within_finite = within_scope.loc[np.isfinite(within_scope["similarity"])]
    pooled_finite = pooled_scope.loc[np.isfinite(pooled_scope["similarity"])]

    return {
        "views": views,
        "within_group_median_by_view": _median_similarity_by_view(within_finite, view_order=views),
        "pooled_vs_group_median_by_view": _median_similarity_by_view(pooled_finite, view_order=views),
        "within_group_valid_count": int(len(within_finite)),
        "within_group_total_count": int(len(within_scope)),
        "pooled_vs_group_valid_count": int(len(pooled_finite)),
        "pooled_vs_group_total_count": int(len(pooled_scope)),
        "within_group_median_shared_entries": _median_or_none(within_finite["n_shared_entries"]),
        "pooled_vs_group_median_shared_entries": _median_or_none(pooled_finite["n_shared_entries"]),
    }


def _median_similarity_by_view(frame: pd.DataFrame, *, view_order: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    if frame.empty:
        return medians
    grouped = frame.groupby("view_name", sort=False, dropna=False)["similarity"].median()
    for view_name in view_order:
        if view_name not in grouped.index:
            continue
        value = grouped.loc[view_name]
        if pd.isna(value):
            continue
        medians[str(view_name)] = float(value)
    return medians


def _summarize_overlap_frame(frame: pd.DataFrame | None) -> dict[str, Any]:
    empty_summary = {
        "group_count": 0,
        "stimuli_per_group_median": None,
        "stimuli_per_group_min": None,
        "stimuli_per_group_max": None,
        "pairwise_shared_stimuli_median": None,
        "pairwise_shared_stimuli_min": None,
        "pairwise_shared_stimuli_max": None,
    }
    if frame is None or frame.empty or "group_id" not in frame.columns:
        return empty_summary

    matrix = frame.set_index("group_id").copy()
    if matrix.empty:
        return empty_summary

    numeric = matrix.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    per_group_counts = numeric.sum(axis=1)
    pairwise_overlaps: list[int] = []
    if len(numeric.index) >= 2:
        for left_index, right_index in itertools.combinations(range(len(numeric.index)), 2):
            left = numeric.iloc[left_index].to_numpy(dtype=int, copy=False)
            right = numeric.iloc[right_index].to_numpy(dtype=int, copy=False)
            pairwise_overlaps.append(int(np.logical_and(left > 0, right > 0).sum()))

    return {
        "group_count": int(len(numeric.index)),
        "stimuli_per_group_median": _median_or_none(per_group_counts),
        "stimuli_per_group_min": int(per_group_counts.min()) if not per_group_counts.empty else None,
        "stimuli_per_group_max": int(per_group_counts.max()) if not per_group_counts.empty else None,
        "pairwise_shared_stimuli_median": _median_or_none(pairwise_overlaps),
        "pairwise_shared_stimuli_min": int(min(pairwise_overlaps)) if pairwise_overlaps else None,
        "pairwise_shared_stimuli_max": int(max(pairwise_overlaps)) if pairwise_overlaps else None,
    }


def _median_or_none(values: pd.Series | list[int] | list[float]) -> float | None:
    if isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None
        return float(numeric.median())
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _format_number(value: float | None, *, precision: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "None"
    return f"{value:.{precision}f}"


def _format_view_values(values: dict[str, float]) -> str:
    if not values:
        return "None"
    return ", ".join(f"{view_name} {_format_number(value)}" for view_name, value in values.items())


def _pooled_geometry_readout(summary: dict[str, Any]) -> str:
    pooled_cross_view = summary["pooled_cross_view"]
    similarity = pooled_cross_view["similarity"]
    if summary["focus_view"] is None:
        return "No pooled neural RDM figure was written."
    if similarity is None or len(summary["pooled_matrix_views"]) < 2:
        return "Stage 2 is acting as a single-view pooled geometry handoff."
    if similarity >= 0.8:
        return "Pooled neural geometry is consistent across the included views."
    if similarity >= 0.5:
        return "Pooled neural geometry is only moderately consistent across views."
    return "Pooled neural geometry differs meaningfully across views and needs caution."


def _date_support_readout(summary: dict[str, Any]) -> str:
    date_stability = summary["date_stability"]
    date_overlap = summary["date_overlap"]
    if date_overlap["group_count"] < 2:
        return "There are not enough date groups to judge date-level stability."
    if date_stability["within_group_valid_count"] == 0:
        return "Date-level stability is support-limited because no finite within-date comparisons were available."
    shared = date_overlap["pairwise_shared_stimuli_median"]
    if shared is not None and shared < 2:
        return "Date-level stability should be read cautiously because date panels barely overlap."
    return "Date-level stability has at least minimal support, but should still be interpreted with overlap in mind."


def _write_markdown_summary(summary: dict[str, Any], path: str | Path) -> Path:
    lines = [
        "# Geometry Analysis Run Summary",
        "",
        "## Bottom Line",
        f"- Included views: {', '.join(summary['views']) if summary['views'] else 'None'}",
        f"- Focus view: {summary['focus_view'] or 'None'}",
        f"- Sensitivity views: {', '.join(summary['sensitivity_views']) if summary['sensitivity_views'] else 'None'}",
        f"- Primary figure: {summary['primary_figure'] or 'None'}",
        f"- Support figure: {summary['support_figure'] or 'None'}",
        f"- Pooled cross-view similarity: {_format_number(summary['pooled_cross_view']['similarity'])} "
        f"(shared entries median {_format_number(summary['pooled_cross_view']['n_shared_entries'], precision=1)})",
        f"- Readout: {_pooled_geometry_readout(summary)}",
        "",
        "## Individual Stability",
        f"- Within-individual median similarity: {_format_view_values(summary['individual_stability']['within_group_median_by_view'])}",
        f"- Pooled-vs-individual median similarity: {_format_view_values(summary['individual_stability']['pooled_vs_group_median_by_view'])}",
        f"- Valid within-individual comparisons: {summary['individual_stability']['within_group_valid_count']} "
        f"of {summary['individual_stability']['within_group_total_count']}",
        f"- Median shared entries for valid individual comparisons: "
        f"{_format_number(summary['individual_stability']['within_group_median_shared_entries'], precision=1)}",
        "",
        "## Date Stability",
        f"- Within-date median similarity: {_format_view_values(summary['date_stability']['within_group_median_by_view'])}",
        f"- Pooled-vs-date median similarity: {_format_view_values(summary['date_stability']['pooled_vs_group_median_by_view'])}",
        f"- Valid within-date comparisons: {summary['date_stability']['within_group_valid_count']} "
        f"of {summary['date_stability']['within_group_total_count']}",
        f"- Median shared entries for valid date comparisons: "
        f"{_format_number(summary['date_stability']['within_group_median_shared_entries'], precision=1)}",
        f"- Readout: {_date_support_readout(summary)}",
        "",
        "## Overlap Limits",
        f"- Dates: {summary['date_overlap']['group_count']} groups; median stimuli per date "
        f"{_format_number(summary['date_overlap']['stimuli_per_group_median'], precision=1)}; "
        f"median shared stimuli across date pairs {_format_number(summary['date_overlap']['pairwise_shared_stimuli_median'], precision=1)}",
        f"- Individuals: {summary['individual_overlap']['group_count']} groups; median stimuli per individual "
        f"{_format_number(summary['individual_overlap']['stimuli_per_group_median'], precision=1)}; "
        f"median shared stimuli across individual pairs "
        f"{_format_number(summary['individual_overlap']['pairwise_shared_stimuli_median'], precision=1)}",
        f"- Readout: overlap remains the main constraint on grouped stability interpretation.",
        "",
        "## Output Paths",
        f"- Tables directory: {summary['tables_dir']}",
        f"- Figures directory: {summary['figures_dir']}",
        f"- QC directory: {summary['qc_dir']}",
    ]

    output_path = Path(path)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


ensure_stage2_output_dirs = ensure_geometry_output_dirs
write_stage2_outputs = write_geometry_outputs
