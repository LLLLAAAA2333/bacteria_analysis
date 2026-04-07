"""Output writers and figures for geometry analysis."""

from __future__ import annotations

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

    for artifact_name in ("rdm_matrix__response_window__pooled", "rdm_matrix__full_trajectory__pooled"):
        matrix_frame = core_outputs.get(artifact_name)
        if matrix_frame is None:
            continue
        view_name = artifact_name.split("__")[1]
        written[f"{artifact_name}_figure"] = _plot_rdm_matrix(
            matrix_frame,
            title=f"Pooled RDM ({view_name})",
            path=dirs["figures_dir"] / f"{artifact_name}.png",
        )
        written[f"{artifact_name}_clustered_figure"] = _plot_rdm_matrix(
            matrix_frame,
            title=f"Pooled RDM ({view_name}, hierarchical order)",
            path=dirs["figures_dir"] / f"{artifact_name}__clustered.png",
            reorder_by_similarity=True,
        )

    figure_specs = (
        ("rdm_stability_by_individual", "Individual RDM Stability"),
        ("rdm_stability_by_date", "Date RDM Stability"),
        ("rdm_view_comparison", "Pooled Cross-View RDM Stability"),
    )
    for artifact_name, title in figure_specs:
        frame = core_outputs.get(artifact_name)
        if frame is None:
            continue
        written[f"{artifact_name}_figure"] = _plot_similarity_summary(
            frame,
            title=title,
            path=dirs["figures_dir"] / f"{artifact_name}.png",
        )

    overlap_specs = (
        ("stimulus_overlap__date", "Stimulus Overlap By Date"),
        ("stimulus_overlap__individual", "Stimulus Overlap By Individual"),
    )
    for artifact_name, title in overlap_specs:
        frame = core_outputs.get(artifact_name)
        if frame is None:
            continue
        written[f"{artifact_name}_figure"] = _plot_stimulus_overlap(
            frame,
            title=title,
            path=dirs["figures_dir"] / f"{artifact_name}.png",
        )

    summary = _build_run_summary(core_outputs, written)
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


def _is_non_pooled_matrix_artifact(artifact_name: str) -> bool:
    return artifact_name.startswith("rdm_matrix__") and not artifact_name.endswith("__pooled")


def _is_qc_artifact(artifact_name: str) -> bool:
    return artifact_name == "rdm_group_coverage" or artifact_name.startswith("stimulus_overlap__")


def _build_run_summary(core_outputs: dict[str, pd.DataFrame], written: dict[str, Path]) -> dict[str, Any]:
    pair_table_names = sorted(name for name in core_outputs if name.startswith("rdm_pairs__"))
    pooled_matrix_names = sorted(name for name in core_outputs if name.startswith("rdm_matrix__") and name.endswith("__pooled"))
    views = _ordered_views_for_summary(core_outputs, pair_table_names, pooled_matrix_names)
    pooled_matrix_views = [view_name for view_name in views if f"rdm_matrix__{view_name}__pooled" in core_outputs]
    stability_table_names = sorted(name for name in core_outputs if name.startswith("rdm_stability_by_"))
    qc_table_names = sorted(name for name in core_outputs if _is_qc_artifact(name))

    return {
        "views": views,
        "pair_table_names": pair_table_names,
        "pooled_matrix_views": pooled_matrix_views,
        "stability_table_names": stability_table_names,
        "view_comparison_table": "rdm_view_comparison" if "rdm_view_comparison" in core_outputs else None,
        "qc_table_names": qc_table_names,
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


def _write_markdown_summary(summary: dict[str, Any], path: str | Path) -> Path:
    lines = [
        "# Geometry Analysis Run Summary",
        "",
        "## Views",
        f"- Views: {', '.join(summary['views']) if summary['views'] else 'None'}",
        f"- Pooled matrix views: {', '.join(summary['pooled_matrix_views']) if summary['pooled_matrix_views'] else 'None'}",
        "",
        "## Tables",
        f"- Pair tables: {', '.join(summary['pair_table_names']) if summary['pair_table_names'] else 'None'}",
        f"- Stability tables: {', '.join(summary['stability_table_names']) if summary['stability_table_names'] else 'None'}",
        f"- View comparison table: {summary['view_comparison_table'] or 'None'}",
        f"- QC tables: {', '.join(summary['qc_table_names']) if summary['qc_table_names'] else 'None'}",
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
