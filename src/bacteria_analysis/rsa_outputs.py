"""Output writers and figures for Stage 3 biochemical RSA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bacteria_analysis.io import write_json, write_parquet

TABLE_ARTIFACT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stimulus_sample_map", ("stimulus_sample_map",)),
    ("metabolite_annotation_resolved", ("metabolite_annotation_resolved", "metabolite_annotation")),
    ("model_registry_resolved", ("model_registry_resolved", "model_registry")),
    ("model_membership_resolved", ("model_membership_resolved", "model_membership")),
    ("model_feature_qc", ("model_feature_qc",)),
    ("model_rdm_summary", ("model_rdm_summary",)),
    ("rsa_results", ("rsa_results",)),
    ("rsa_leave_one_stimulus_out", ("rsa_leave_one_stimulus_out", "leave_one_stimulus_out")),
    ("rsa_view_comparison", ("rsa_view_comparison", "cross_view_comparison")),
)

QC_ARTIFACT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("model_input_coverage", ("model_input_coverage",)),
    ("model_feature_filtering", ("model_feature_filtering", "model_feature_qc")),
)

REQUIRED_FIGURES: tuple[str, ...] = (
    "ranked_primary_model_rsa",
    "leave_one_stimulus_out_robustness",
    "view_comparison_summary",
)


def _build_neural_vs_model_figure_names(view_names: list[str]) -> list[str]:
    return [f"neural_vs_top_model_rdm__{view_name}" for view_name in view_names]


def ensure_stage3_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    return _mkdir_stage3_dirs(root)


def write_stage3_outputs(core_outputs: dict[str, pd.DataFrame], output_root: str | Path) -> dict[str, Path]:
    dirs = ensure_stage3_output_dirs(output_root)
    return _write_stage3_artifacts(core_outputs, dirs)


def _mkdir_stage3_dirs(root: Path) -> dict[str, Path]:
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


def _write_stage3_artifacts(core_outputs: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Path]:
    written: dict[str, Path] = {
        "output_root": dirs["output_root"],
        "tables_dir": dirs["tables_dir"],
        "figures_dir": dirs["figures_dir"],
        "qc_dir": dirs["qc_dir"],
    }
    consumed_keys: set[str] = set()

    required_tables = _resolve_artifact_family(core_outputs, TABLE_ARTIFACT_SPECS, consumed_keys)
    required_qc = _resolve_artifact_family(core_outputs, QC_ARTIFACT_SPECS, consumed_keys)

    for artifact_name, frame in required_tables.items():
        written[artifact_name] = write_parquet(
            _prepare_for_parquet(frame),
            dirs["tables_dir"] / f"{artifact_name}.parquet",
        )
    for artifact_name, frame in required_qc.items():
        written[artifact_name] = write_parquet(
            _prepare_for_parquet(frame),
            dirs["qc_dir"] / f"{artifact_name}.parquet",
        )

    for artifact_name, frame in core_outputs.items():
        if artifact_name in consumed_keys or not isinstance(frame, pd.DataFrame):
            continue
        output_dir = dirs["qc_dir"] if _is_qc_artifact(artifact_name) else dirs["tables_dir"]
        written[artifact_name] = write_parquet(_prepare_for_parquet(frame), output_dir / f"{artifact_name}.parquet")

    registry = required_tables["model_registry_resolved"]
    rsa_results = required_tables["rsa_results"]
    leave_one_out = required_tables["rsa_leave_one_stimulus_out"]
    view_comparison = required_tables["rsa_view_comparison"]
    view_names = _ordered_views(rsa_results, view_comparison)
    family_summary = _collect_model_families(registry)
    top_primary_models = _build_top_primary_models_by_view(rsa_results, family_summary["primary_models"])
    primary_view = _choose_primary_view(rsa_results, view_candidates=view_names)

    written["ranked_primary_model_rsa"] = _plot_ranked_primary_model_rsa(
        rsa_results,
        family_summary["primary_models"],
        path=dirs["figures_dir"] / "ranked_primary_model_rsa.png",
        primary_view=primary_view,
    )
    for figure_name, view_name in zip(_build_neural_vs_model_figure_names(view_names), view_names, strict=False):
        written[figure_name] = _plot_neural_vs_top_model_rdm_view(
            core_outputs,
            top_primary_models,
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )
    written["leave_one_stimulus_out_robustness"] = _plot_leave_one_stimulus_out_robustness(
        leave_one_out,
        family_summary["primary_models"],
        dirs["figures_dir"] / "leave_one_stimulus_out_robustness.png",
    )
    written["view_comparison_summary"] = _plot_view_comparison_summary(
        view_comparison,
        dirs["figures_dir"] / "view_comparison_summary.png",
    )

    summary = _build_run_summary(
        required_tables=required_tables,
        required_qc=required_qc,
        written=written,
        family_summary=family_summary,
        top_primary_models=top_primary_models,
        primary_view=primary_view,
    )
    written["run_summary_json"] = write_json(summary, dirs["output_root"] / "run_summary.json")
    written["run_summary_md"] = _write_markdown_summary(summary, dirs["output_root"] / "run_summary.md")
    return written


def _resolve_artifact_family(
    core_outputs: dict[str, pd.DataFrame],
    specs: tuple[tuple[str, tuple[str, ...]], ...],
    consumed_keys: set[str],
) -> dict[str, pd.DataFrame]:
    resolved: dict[str, pd.DataFrame] = {}
    for canonical_name, aliases in specs:
        frame, source_key = _resolve_artifact_frame(core_outputs, aliases)
        resolved[canonical_name] = frame
        if source_key is not None:
            consumed_keys.add(source_key)
    return resolved


def _resolve_artifact_frame(
    core_outputs: dict[str, pd.DataFrame],
    aliases: tuple[str, ...],
) -> tuple[pd.DataFrame, str | None]:
    for alias in aliases:
        frame = core_outputs.get(alias)
        if isinstance(frame, pd.DataFrame):
            return frame.copy(), alias
    return pd.DataFrame(), None


def _prepare_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        if prepared[column].dtype != "object":
            continue
        prepared[column] = prepared[column].map(
            lambda value: "|".join(map(str, value)) if isinstance(value, (list, tuple, set)) else value
        )
    return prepared


def _is_qc_artifact(artifact_name: str) -> bool:
    return artifact_name in {"model_input_coverage", "model_feature_filtering"} or artifact_name.endswith("_qc")


def _collect_model_families(registry: pd.DataFrame) -> dict[str, list[str]]:
    if registry.empty or "model_id" not in registry.columns:
        return {
            "primary_models": [],
            "supplementary_models": [],
            "excluded_models": [],
        }

    model_ids = registry["model_id"].astype(str)
    model_tier = _string_column(registry, "model_tier")
    model_status = _string_column(registry, "model_status")
    excluded_from_primary_ranking = _bool_column(registry, "excluded_from_primary_ranking")
    hard_excluded_mask = model_status.eq("excluded")
    primary_excluded_mask = model_tier.eq("primary") & excluded_from_primary_ranking

    return {
        "primary_models": model_ids.loc[model_tier.eq("primary") & ~primary_excluded_mask & ~hard_excluded_mask].tolist(),
        "supplementary_models": model_ids.loc[model_tier.eq("supplementary") & ~hard_excluded_mask].tolist(),
        "excluded_models": model_ids.loc[hard_excluded_mask | primary_excluded_mask].tolist(),
    }


def _build_top_primary_models_by_view(rsa_results: pd.DataFrame, primary_models: list[str]) -> dict[str, str]:
    if rsa_results.empty or not primary_models:
        return {}

    required_columns = {"view_name", "model_id", "rsa_similarity"}
    if not required_columns.issubset(rsa_results.columns):
        return {}

    filtered = rsa_results.copy()
    filtered["view_name"] = filtered["view_name"].astype(str)
    filtered["model_id"] = filtered["model_id"].astype(str)
    filtered["rsa_similarity"] = pd.to_numeric(filtered["rsa_similarity"], errors="coerce")
    filtered = filtered.loc[filtered["model_id"].isin(primary_models)]
    if "score_status" in filtered.columns:
        filtered = filtered.loc[_string_column(filtered, "score_status").eq("ok")]
    filtered = filtered.loc[np.isfinite(filtered["rsa_similarity"])]
    if filtered.empty:
        return {}

    top_models: dict[str, str] = {}
    for view_name, group in filtered.groupby("view_name", sort=False):
        ranked = group.sort_values(["rsa_similarity", "model_id"], ascending=[False, True])
        top_models[str(view_name)] = str(ranked.iloc[0]["model_id"])
    return top_models


def _plot_ranked_primary_model_rsa(
    rsa_results: pd.DataFrame,
    primary_models: list[str],
    *,
    primary_view: str | None,
    path: Path,
) -> Path:
    if rsa_results.empty or not primary_models:
        return _plot_empty_figure(path, title="Ranked Primary-Model RSA", message="No primary RSA results")

    required_columns = {"model_id", "view_name", "rsa_similarity"}
    if not required_columns.issubset(rsa_results.columns):
        return _plot_empty_figure(path, title="Ranked Primary-Model RSA", message="Missing rsa_results columns")

    ranked = rsa_results.copy()
    ranked["model_id"] = ranked["model_id"].astype(str)
    ranked["view_name"] = ranked["view_name"].astype(str)
    ranked["rsa_similarity"] = pd.to_numeric(ranked["rsa_similarity"], errors="coerce")
    ranked = ranked.loc[ranked["model_id"].isin(primary_models)]
    if primary_view is not None:
        ranked = ranked.loc[ranked["view_name"] == primary_view]
    ranked = ranked.loc[np.isfinite(ranked["rsa_similarity"])]
    if ranked.empty:
        return _plot_empty_figure(path, title="Ranked Primary-Model RSA", message="No finite primary RSA values")

    ranked = ranked.sort_values(["rsa_similarity", "model_id"], ascending=[True, True])
    y_positions = np.arange(len(ranked), dtype=float)
    plt.figure(figsize=(7.5, max(3.5, 0.8 * len(ranked) + 1.5)))
    plt.barh(y_positions, ranked["rsa_similarity"].to_numpy(dtype=float))
    plt.yticks(y_positions, ranked["model_id"].tolist())
    plt.xlabel("RSA similarity")
    plt.ylabel("Model")
    if primary_view is None:
        plt.title("Ranked Primary-Model RSA")
    else:
        plt.title(f"Ranked Primary-Model RSA ({primary_view})")
    return _save_figure(path)


def _plot_neural_vs_top_model_rdm_view(
    core_outputs: dict[str, pd.DataFrame],
    top_primary_models: dict[str, str],
    *,
    view_name: str,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(
        ncols=2,
        figsize=(9.5, 4.0),
        squeeze=False,
    )
    figure.suptitle(f"Neural-Versus-Top-Model RDM Comparison ({view_name})", fontsize=12)

    top_model_id = top_primary_models.get(view_name)
    neural_matrix = _find_matrix_frame(
        core_outputs,
        (
            f"neural_rdm__{view_name}",
            f"neural_rdm__{view_name}__pooled",
            f"rdm_matrix__{view_name}__pooled",
        ),
    )
    model_matrix = None
    if top_model_id:
        model_matrix = _find_matrix_frame(
            core_outputs,
            (
                f"model_rdm__{top_model_id}__{view_name}",
                f"model_rdm__{view_name}__{top_model_id}",
                f"model_rdm__{top_model_id}",
            ),
        )

    _render_rdm_axis(
        axes[0, 0],
        neural_matrix,
        title=f"{view_name}: neural",
        fallback_message="No neural matrix provided",
    )
    _render_rdm_axis(
        axes[0, 1],
        model_matrix,
        title=f"{view_name}: {top_model_id or 'no top model'}",
        fallback_message="No top-model matrix provided",
    )

    return _save_figure(path)


def _plot_leave_one_stimulus_out_robustness(
    leave_one_out: pd.DataFrame,
    primary_models: list[str],
    path: Path,
) -> Path:
    if leave_one_out.empty:
        return _plot_empty_figure(path, title="Leave-One-Stimulus-Out Robustness", message="No robustness table")

    required_columns = {"excluded_stimulus", "model_id", "rsa_similarity"}
    if not required_columns.issubset(leave_one_out.columns):
        return _plot_empty_figure(path, title="Leave-One-Stimulus-Out Robustness", message="Missing robustness columns")

    summary = leave_one_out.copy()
    summary["excluded_stimulus"] = summary["excluded_stimulus"].astype(str)
    summary["model_id"] = summary["model_id"].astype(str)
    summary["rsa_similarity"] = pd.to_numeric(summary["rsa_similarity"], errors="coerce")
    summary = summary.loc[np.isfinite(summary["rsa_similarity"])]
    if primary_models:
        summary = summary.loc[summary["model_id"].isin(primary_models)]
    if summary.empty:
        return _plot_empty_figure(path, title="Leave-One-Stimulus-Out Robustness", message="No finite robustness data")

    figure_width = max(6.5, 0.9 * summary["excluded_stimulus"].nunique() + 2.0)
    plt.figure(figsize=(figure_width, 4.5))
    for model_id, group in summary.groupby("model_id", sort=False):
        ordered = group.sort_values("excluded_stimulus")
        plt.plot(
            ordered["excluded_stimulus"],
            ordered["rsa_similarity"],
            marker="o",
            label=model_id,
        )
    plt.xlabel("Excluded stimulus")
    plt.ylabel("RSA similarity")
    plt.title("Leave-One-Stimulus-Out Robustness")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    return _save_figure(path)


def _plot_view_comparison_summary(view_comparison: pd.DataFrame, path: Path) -> Path:
    if view_comparison.empty:
        return _plot_empty_figure(path, title="View Comparison Summary", message="No cross-view table")

    required_columns = {"view_name", "rsa_similarity"}
    if not required_columns.issubset(view_comparison.columns):
        return _plot_empty_figure(path, title="View Comparison Summary", message="Missing cross-view columns")

    summary = view_comparison.copy()
    summary["view_name"] = summary["view_name"].astype(str)
    if "reference_view_name" in summary.columns:
        summary["reference_view_name"] = summary["reference_view_name"].astype(str)
        labels = summary["view_name"] + " vs " + summary["reference_view_name"]
    else:
        labels = summary["view_name"]
    summary["plot_label"] = labels
    summary["rsa_similarity"] = pd.to_numeric(summary["rsa_similarity"], errors="coerce")
    summary = summary.loc[np.isfinite(summary["rsa_similarity"])]
    if summary.empty:
        return _plot_empty_figure(path, title="View Comparison Summary", message="No finite cross-view values")

    plt.figure(figsize=(max(6.0, 0.9 * len(summary) + 2.0), 4.5))
    plt.bar(summary["plot_label"], summary["rsa_similarity"].to_numpy(dtype=float))
    plt.xlabel("Comparison")
    plt.ylabel("RSA similarity")
    plt.title("View Comparison Summary")
    plt.xticks(rotation=45, ha="right")
    return _save_figure(path)


def _find_matrix_frame(core_outputs: dict[str, pd.DataFrame], aliases: tuple[str, ...]) -> pd.DataFrame | None:
    for alias in aliases:
        frame = core_outputs.get(alias)
        if isinstance(frame, pd.DataFrame):
            return frame
    return None


def _render_rdm_axis(
    axis: plt.Axes,
    matrix_frame: pd.DataFrame | None,
    *,
    title: str,
    fallback_message: str,
) -> None:
    axis.set_title(title)
    if matrix_frame is None:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return

    heatmap_frame = _coerce_rdm_heatmap_frame(matrix_frame)
    if heatmap_frame.empty:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return

    image = axis.imshow(heatmap_frame.to_numpy(dtype=float), cmap="viridis")
    axis.set_xticks(np.arange(len(heatmap_frame.columns)))
    axis.set_yticks(np.arange(len(heatmap_frame.index)))
    axis.set_xticklabels(heatmap_frame.columns.tolist(), rotation=45, ha="right")
    axis.set_yticklabels(heatmap_frame.index.tolist())
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _coerce_rdm_heatmap_frame(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    if "stimulus_row" in matrix_frame.columns:
        heatmap_frame = matrix_frame.set_index("stimulus_row").copy()
    else:
        heatmap_frame = matrix_frame.copy()

    if heatmap_frame.empty:
        return heatmap_frame

    heatmap_frame.index = pd.Index(heatmap_frame.index.astype(str))
    heatmap_frame.columns = pd.Index(heatmap_frame.columns.astype(str))
    if set(heatmap_frame.index) != set(heatmap_frame.columns):
        raise ValueError("RDM heatmap requires matching row and column labels")
    heatmap_frame = heatmap_frame.reindex(columns=heatmap_frame.index)
    return heatmap_frame.apply(pd.to_numeric, errors="coerce")


def _plot_empty_figure(path: Path, *, title: str, message: str) -> Path:
    plt.figure(figsize=(6.5, 4.0))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.title(title)
    return _save_figure(path)


def _build_run_summary(
    *,
    required_tables: dict[str, pd.DataFrame],
    required_qc: dict[str, pd.DataFrame],
    written: dict[str, Path],
    family_summary: dict[str, list[str]],
    top_primary_models: dict[str, str],
    primary_view: str | None,
) -> dict[str, Any]:
    table_names = [artifact_name for artifact_name, _ in TABLE_ARTIFACT_SPECS]
    qc_table_names = [artifact_name for artifact_name, _ in QC_ARTIFACT_SPECS]
    additional_table_names = sorted(
        key
        for key, path in written.items()
        if isinstance(path, Path)
        and path.parent == written["tables_dir"]
        and key not in table_names
        and key not in {"output_root", "tables_dir", "figures_dir", "qc_dir"}
    )
    view_names = _ordered_views(required_tables["rsa_results"], required_tables["rsa_view_comparison"])
    figure_names = [*REQUIRED_FIGURES, *_build_neural_vs_model_figure_names(view_names)]

    return {
        "views": view_names,
        "primary_view": primary_view,
        "sensitivity_views": [view_name for view_name in view_names if view_name != primary_view],
        "primary_models": family_summary["primary_models"],
        "supplementary_models": family_summary["supplementary_models"],
        "excluded_models": family_summary["excluded_models"],
        "top_primary_models_by_view": top_primary_models,
        "resolved_input_tables": [
            "stimulus_sample_map",
            "metabolite_annotation_resolved",
            "model_registry_resolved",
            "model_membership_resolved",
        ],
        "rsa_table_names": [
            "model_rdm_summary",
            "rsa_results",
            "rsa_leave_one_stimulus_out",
            "rsa_view_comparison",
        ],
        "qc_table_names": qc_table_names,
        "additional_table_names": additional_table_names,
        "figure_names": figure_names,
        "n_required_tables_written": sum(int(not frame.empty) for frame in required_tables.values()),
        "n_required_qc_tables_written": sum(int(not frame.empty) for frame in required_qc.values()),
        "tables_dir": str(written["tables_dir"]),
        "figures_dir": str(written["figures_dir"]),
        "qc_dir": str(written["qc_dir"]),
    }


def _ordered_views(rsa_results: pd.DataFrame, view_comparison: pd.DataFrame) -> list[str]:
    views: list[str] = []
    if not rsa_results.empty and "view_name" in rsa_results.columns:
        for view_name in rsa_results["view_name"].astype(str).tolist():
            if view_name not in views:
                views.append(view_name)
    if not view_comparison.empty and "view_name" in view_comparison.columns:
        for view_name in view_comparison["view_name"].astype(str).tolist():
            if view_name not in views:
                views.append(view_name)
    return views


def _choose_primary_view(rsa_results: pd.DataFrame, *, view_candidates: list[str] | None = None) -> str | None:
    views = view_candidates or _ordered_views(rsa_results, pd.DataFrame())
    if not views:
        return None
    if "response_window" in views:
        return "response_window"
    return views[0]


def _write_markdown_summary(summary: dict[str, Any], path: str | Path) -> Path:
    lines = [
        "# Stage 3 Biochemical RSA Run Summary",
        "",
        "## Views",
        f"- Primary view: {summary['primary_view'] or 'None'}",
        f"- Sensitivity views: {', '.join(summary['sensitivity_views']) if summary['sensitivity_views'] else 'None'}",
        "",
        "## Model Families",
        f"- Primary models: {', '.join(summary['primary_models']) if summary['primary_models'] else 'None'}",
        f"- Supplementary models: {', '.join(summary['supplementary_models']) if summary['supplementary_models'] else 'None'}",
        f"- Excluded models: {', '.join(summary['excluded_models']) if summary['excluded_models'] else 'None'}",
        "",
        "## Top Primary Models By View",
    ]

    top_models = summary["top_primary_models_by_view"]
    if top_models:
        lines.extend(f"- {view_name}: {model_id}" for view_name, model_id in top_models.items())
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Resolved input tables: {', '.join(summary['resolved_input_tables'])}",
            f"- RSA tables: {', '.join(summary['rsa_table_names'])}",
            f"- QC tables: {', '.join(summary['qc_table_names'])}",
            f"- Figures: {', '.join(summary['figure_names'])}",
            "",
            "## Output Paths",
            f"- Tables directory: {summary['tables_dir']}",
            f"- Figures directory: {summary['figures_dir']}",
            f"- QC directory: {summary['qc_dir']}",
        ]
    )

    output_path = Path(path)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _string_column(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column_name].fillna("").astype(str).str.strip().str.lower()


def _bool_column(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column_name]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes"})


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


__all__ = [
    "ensure_stage3_output_dirs",
    "write_stage3_outputs",
]

