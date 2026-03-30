"""Output writers and figures for Stage 2 geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bacteria_analysis.io import write_json, write_parquet


def ensure_stage2_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    return _mkdir_stage2_dirs(root)


def write_stage2_outputs(core_outputs: dict[str, pd.DataFrame], output_root: str | Path) -> dict[str, Path]:
    dirs = ensure_stage2_output_dirs(output_root)
    return _write_stage2_artifacts(core_outputs, dirs)


def _mkdir_stage2_dirs(root: Path) -> dict[str, Path]:
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


def _write_stage2_artifacts(core_outputs: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Path]:
    written: dict[str, Path] = {
        "output_root": dirs["output_root"],
        "tables_dir": dirs["tables_dir"],
        "figures_dir": dirs["figures_dir"],
        "qc_dir": dirs["qc_dir"],
    }

    for artifact_name, frame in core_outputs.items():
        if _is_non_pooled_matrix_artifact(artifact_name):
            continue
        output_dir = dirs["qc_dir"] if artifact_name == "rdm_group_coverage" else dirs["tables_dir"]
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


def _plot_rdm_matrix(matrix_frame: pd.DataFrame, title: str, path: Path) -> Path:
    heatmap_frame = matrix_frame.set_index("stimulus_row")
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
    plt.xlabel("Stimulus")
    plt.ylabel("Stimulus")
    plt.title(title)
    return _save_figure(path)


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


def _build_run_summary(core_outputs: dict[str, pd.DataFrame], written: dict[str, Path]) -> dict[str, Any]:
    pair_table_names = sorted(name for name in core_outputs if name.startswith("rdm_pairs__"))
    pooled_matrix_names = sorted(name for name in core_outputs if name.startswith("rdm_matrix__") and name.endswith("__pooled"))
    views = _ordered_views_for_summary(core_outputs, pair_table_names, pooled_matrix_names)
    pooled_matrix_views = [view_name for view_name in views if f"rdm_matrix__{view_name}__pooled" in core_outputs]
    stability_table_names = sorted(name for name in core_outputs if name.startswith("rdm_stability_by_"))
    qc_table_names = sorted(name for name in core_outputs if name == "rdm_group_coverage")

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
        "# Stage 2 Geometry Run Summary",
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
