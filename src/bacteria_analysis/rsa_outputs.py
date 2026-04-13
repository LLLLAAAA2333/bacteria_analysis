"""Output writers and figures for biochemical RSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

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
    "single_stimulus_sensitivity",
)
DEFAULT_FIGURE_VIEWS: tuple[str, ...] = ("response_window", "full_trajectory")
RANKED_MODEL_RSA_DETAIL_FIELDS: tuple[str, ...] = (
    "view_name",
    "model_id",
    "rsa_similarity",
    "p_value_raw",
    "p_value_fdr",
    "n_shared_entries",
    "score_status",
    "is_top_model",
)
VIEW_COMPARISON_DETAIL_FIELDS: tuple[str, ...] = (
    "view_name",
    "reference_view_name",
    "comparison_scope",
    "rsa_similarity",
    "p_value_raw",
    "p_value_fdr",
    "n_shared_entries",
    "score_status",
)
PROTOTYPE_QC_ARTIFACT_NAMES: tuple[str, ...] = (
    "aggregated_response_support__per_date",
    "aggregated_response_support__pooled",
)
INTERNAL_ONLY_ARTIFACT_PREFIX = "internal__"
INTERNAL_AGGREGATED_RESPONSE_PER_DATE_RDM_PREFIX = (
    f"{INTERNAL_ONLY_ARTIFACT_PREFIX}aggregated_response_rdm__per_date__"
)
INTERNAL_RESPONSE_AGGREGATION_KEY = f"{INTERNAL_ONLY_ARTIFACT_PREFIX}response_aggregation"
INTERNAL_PROTOTYPE_PER_DATE_RDM_PREFIX = INTERNAL_AGGREGATED_RESPONSE_PER_DATE_RDM_PREFIX
INTERNAL_PROTOTYPE_AGGREGATION_KEY = INTERNAL_RESPONSE_AGGREGATION_KEY


@dataclass(frozen=True)
class RdmDisplayParameters:
    vmin: float
    vmax: float
    norm: matplotlib.colors.PowerNorm


def _build_neural_vs_model_figure_names(view_names: list[str]) -> list[str]:
    return [f"neural_vs_top_model_rdm__{view_name}" for view_name in view_names]


def _build_aggregated_response_rsa_figure_names(view_names: list[str]) -> list[str]:
    return [f"aggregated_response_rsa__per_date__{view_name}" for view_name in view_names]


def _build_aggregated_response_rdm_comparison_figure_names(view_names: list[str]) -> list[str]:
    return [f"aggregated_response_rdm_comparison__per_date__{view_name}" for view_name in view_names]


def _build_aggregated_response_rdm_figure_names(view_names: list[str]) -> list[str]:
    return [f"aggregated_response_rdm__pooled__{view_name}" for view_name in view_names]


def _canonicalize_view_order(view_names: list[str]) -> list[str]:
    unique_views = sorted({str(view_name) for view_name in view_names})
    ordered_views: list[str] = []
    for preferred_view in ("response_window", "full_trajectory"):
        if preferred_view in unique_views:
            ordered_views.append(preferred_view)
            unique_views.remove(preferred_view)
    ordered_views.extend(unique_views)
    return ordered_views


def _internal_aggregated_response_per_date_rdm_key(view_name: str, date_value: str) -> str:
    return f"{INTERNAL_AGGREGATED_RESPONSE_PER_DATE_RDM_PREFIX}{view_name}__{date_value}"


def ensure_rsa_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    return _mkdir_rsa_dirs(root)


def write_rsa_outputs(core_outputs: dict[str, pd.DataFrame], output_root: str | Path) -> dict[str, Path]:
    dirs = ensure_rsa_output_dirs(output_root)
    return _write_rsa_artifacts(core_outputs, dirs)


def _mkdir_rsa_dirs(root: Path) -> dict[str, Path]:
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


def _write_rsa_artifacts(core_outputs: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> dict[str, Path]:
    written: dict[str, Path] = {
        "output_root": dirs["output_root"],
        "tables_dir": dirs["tables_dir"],
        "figures_dir": dirs["figures_dir"],
        "qc_dir": dirs["qc_dir"],
    }
    consumed_keys: set[str] = set()

    _remove_legacy_rsa_figures(dirs["figures_dir"])
    _remove_stale_prototype_parquets(dirs["tables_dir"], dirs["qc_dir"])

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
        if artifact_name in consumed_keys or _is_internal_only_artifact(artifact_name) or not isinstance(frame, pd.DataFrame):
            continue
        output_dir = dirs["qc_dir"] if _is_qc_artifact(artifact_name) else dirs["tables_dir"]
        written[artifact_name] = write_parquet(_prepare_for_parquet(frame), output_dir / f"{artifact_name}.parquet")

    registry = required_tables["model_registry_resolved"]
    rsa_results = required_tables["rsa_results"]
    leave_one_out = required_tables["rsa_leave_one_stimulus_out"]
    view_comparison = required_tables["rsa_view_comparison"]
    view_names = _ordered_views(rsa_results, view_comparison)
    figure_view_names = _figure_view_names(rsa_results, view_comparison)
    group_summary = _collect_model_groups(registry)
    top_models = _build_top_models_by_view(rsa_results, group_summary["ranked_models"])
    focus_view = _choose_focus_view(rsa_results, view_candidates=view_names)

    written["single_stimulus_sensitivity"] = _plot_single_stimulus_sensitivity(
        leave_one_out,
        group_summary["ranked_models"],
        dirs["figures_dir"] / "single_stimulus_sensitivity.png",
    )
    for figure_name, view_name in zip(
        _build_neural_vs_model_figure_names(figure_view_names), figure_view_names, strict=False
    ):
        written[figure_name] = _plot_neural_vs_top_model_rdm_view(
            core_outputs,
            top_models,
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )
    aggregated_response_summary = _write_aggregated_response_context_figures(
        core_outputs,
        dirs,
        written,
        top_models=top_models,
    )

    summary = _build_run_summary(
        required_tables=required_tables,
        required_qc=required_qc,
        written=written,
        group_summary=group_summary,
        top_models=top_models,
        focus_view=focus_view,
        aggregated_response_summary=aggregated_response_summary,
    )
    written["run_summary_json"] = write_json(summary, dirs["output_root"] / "run_summary.json")
    written["run_summary_md"] = _write_markdown_summary(summary, dirs["output_root"] / "run_summary.md")
    return written


def _remove_legacy_rsa_figures(figures_dir: Path) -> None:
    for legacy_name in (
        "neural_vs_top_model_rdm_panel.png",
        "ranked_primary_model_rsa.png",
        "ranked_model_rsa.png",
        "view_comparison_summary.png",
        "leave_one_stimulus_out_robustness.png",
    ):
        legacy_figure = figures_dir / legacy_name
        if legacy_figure.exists():
            legacy_figure.unlink()

    for stale_figure in figures_dir.glob("neural_vs_top_model_rdm__*.png"):
        stale_figure.unlink()
    for stale_figure in figures_dir.glob("aggregated_response_*.png"):
        stale_figure.unlink()
    for stale_figure in figures_dir.glob("prototype_*.png"):
        stale_figure.unlink()


def _remove_stale_prototype_parquets(tables_dir: Path, qc_dir: Path) -> None:
    for directory in (tables_dir, qc_dir):
        for stale_artifact in directory.glob("aggregated_response_*.parquet"):
            stale_artifact.unlink()
        for stale_artifact in directory.glob("prototype_*.parquet"):
            stale_artifact.unlink()


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
    return artifact_name in {
        "model_input_coverage",
        "model_feature_filtering",
        *PROTOTYPE_QC_ARTIFACT_NAMES,
    } or artifact_name.endswith("_qc")


def _is_internal_only_artifact(artifact_name: str) -> bool:
    return artifact_name.startswith(INTERNAL_ONLY_ARTIFACT_PREFIX)


def _collect_model_groups(registry: pd.DataFrame) -> dict[str, list[str]]:
    if registry.empty or "model_id" not in registry.columns:
        return {
            "ranked_models": [],
            "additional_models": [],
            "excluded_models": [],
        }

    model_ids = registry["model_id"].astype(str)
    model_tier = _string_column(registry, "model_tier")
    model_status = _string_column(registry, "model_status")
    excluded_from_primary_ranking = _bool_column(registry, "excluded_from_primary_ranking")
    hard_excluded_mask = model_status.eq("excluded")
    primary_excluded_mask = model_tier.eq("primary") & excluded_from_primary_ranking

    return {
        "ranked_models": model_ids.loc[model_tier.eq("primary") & ~primary_excluded_mask & ~hard_excluded_mask].tolist(),
        "additional_models": model_ids.loc[model_tier.eq("supplementary") & ~hard_excluded_mask].tolist(),
        "excluded_models": model_ids.loc[hard_excluded_mask | primary_excluded_mask].tolist(),
    }


def _build_top_models_by_view(rsa_results: pd.DataFrame, ranked_models: list[str]) -> dict[str, str]:
    if rsa_results.empty or not ranked_models:
        return {}

    required_columns = {"view_name", "model_id", "rsa_similarity"}
    if not required_columns.issubset(rsa_results.columns):
        return {}

    filtered = rsa_results.copy()
    filtered["view_name"] = filtered["view_name"].astype(str)
    filtered["model_id"] = filtered["model_id"].astype(str)
    filtered["rsa_similarity"] = pd.to_numeric(filtered["rsa_similarity"], errors="coerce")
    filtered = filtered.loc[filtered["model_id"].isin(ranked_models)]
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


def _build_ranked_model_rsa_details(
    rsa_results: pd.DataFrame,
    ranked_models: list[str],
    *,
    focus_view: str | None,
) -> list[dict[str, Any]]:
    if rsa_results.empty or not ranked_models:
        return []

    required_columns = {"view_name", "model_id", "rsa_similarity"}
    if not required_columns.issubset(rsa_results.columns):
        return []

    filtered = rsa_results.copy()
    filtered["view_name"] = filtered["view_name"].astype(str)
    filtered["model_id"] = filtered["model_id"].astype(str)
    filtered["rsa_similarity"] = pd.to_numeric(filtered["rsa_similarity"], errors="coerce")
    filtered = filtered.loc[filtered["model_id"].isin(ranked_models)]
    if focus_view is not None:
        filtered = filtered.loc[filtered["view_name"] == str(focus_view)]
    filtered = filtered.loc[np.isfinite(filtered["rsa_similarity"])]
    if filtered.empty:
        return []

    ordered = filtered.sort_values(["rsa_similarity", "model_id"], ascending=[False, True], kind="mergesort")
    return [_build_summary_detail_record(row, RANKED_MODEL_RSA_DETAIL_FIELDS) for _, row in ordered.iterrows()]


def _build_view_comparison_details(view_comparison: pd.DataFrame) -> list[dict[str, Any]]:
    if view_comparison.empty:
        return []

    required_columns = {"view_name", "reference_view_name", "rsa_similarity"}
    if not required_columns.issubset(view_comparison.columns):
        return []

    filtered = view_comparison.copy()
    filtered["view_name"] = filtered["view_name"].astype(str)
    filtered["reference_view_name"] = filtered["reference_view_name"].astype(str)
    filtered["rsa_similarity"] = pd.to_numeric(filtered["rsa_similarity"], errors="coerce")
    filtered = filtered.loc[np.isfinite(filtered["rsa_similarity"])]
    if filtered.empty:
        return []

    return [_build_summary_detail_record(row, VIEW_COMPARISON_DETAIL_FIELDS) for _, row in filtered.iterrows()]


def _build_summary_detail_record(row: pd.Series, field_names: tuple[str, ...]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for field_name in field_names:
        if field_name not in row.index:
            continue
        record[field_name] = _summary_json_value(row[field_name])
    return record


def _summary_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_top_aggregated_response_models_by_date_and_view(
    aggregated_response_rsa_results: pd.DataFrame | None,
) -> dict[tuple[str, str], str]:
    if aggregated_response_rsa_results is None or aggregated_response_rsa_results.empty:
        return {}

    required_columns = {"date", "view_name", "model_id", "is_top_model"}
    if not required_columns.issubset(aggregated_response_rsa_results.columns):
        return {}

    selected = aggregated_response_rsa_results.copy()
    selected["date"] = selected["date"].astype(str)
    selected["view_name"] = selected["view_name"].astype(str)
    selected["model_id"] = selected["model_id"].astype(str)
    selected = selected.loc[_bool_column(selected, "is_top_model")]
    if "excluded_from_primary_ranking" in selected.columns:
        selected = selected.loc[~_bool_column(selected, "excluded_from_primary_ranking")]
    if selected.empty:
        return {}

    top_models: dict[tuple[str, str], str] = {}
    for _, row in selected.iterrows():
        top_models[(str(row["date"]), str(row["view_name"]))] = str(row["model_id"])
    return top_models


def _plot_neural_vs_top_model_rdm_view(
    core_outputs: dict[str, pd.DataFrame],
    top_models: dict[str, str],
    *,
    view_name: str,
    path: Path,
) -> Path:
    figure, axes, colorbar_axes = _create_rdm_panel_figure(nrows=2, figsize=(10.2, 7.0))
    comparison_title = (
        "Aggregated-Response Versus Top-Model RDM Comparison"
        if _aggregated_response_context_enabled(core_outputs)
        else "Neural-Versus-Top-Model RDM Comparison"
    )
    left_label = "aggregated response" if _aggregated_response_context_enabled(core_outputs) else "neural"
    figure.suptitle(f"{comparison_title} ({view_name})", fontsize=12)

    top_model_id = top_models.get(view_name)
    stimulus_sample_map = core_outputs.get("stimulus_sample_map")
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
        model_matrix = _find_matrix_frame(core_outputs, _model_rdm_aliases(top_model_id, view_name))
        if model_matrix is not None and neural_matrix is not None:
            neural_labels = _coerce_rdm_heatmap_frame(neural_matrix).index.tolist()
            model_matrix = _restrict_rdm_to_labels(model_matrix, neural_labels)

    prepared_panels: list[tuple[int, int, pd.DataFrame | None, str, str]] = []
    for row_index, order_source in enumerate(("neural", "model")):
        order_labels_for_pair = _resolve_pair_order_labels(
            order_source=order_source,
            neural_matrix=neural_matrix,
            model_matrix=model_matrix,
            stimulus_sample_map=stimulus_sample_map,
        )
        prepared_panels.extend(
            [
                (
                    row_index,
                    0,
                    _prepare_rdm_display_frame(
                        neural_matrix,
                        stimulus_sample_map,
                        order_labels=order_labels_for_pair,
                    ),
                    f"{view_name}: {left_label} ({order_source} order)",
                    "No neural matrix provided",
                ),
                (
                    row_index,
                    1,
                    _prepare_rdm_display_frame(
                        model_matrix,
                        stimulus_sample_map,
                        order_labels=order_labels_for_pair,
                    ),
                    f"{view_name}: {top_model_id or 'no top model'} ({order_source} order)",
                    "No top-model matrix provided",
                ),
            ]
        )
    _render_prepared_rdm_panels(figure, axes, colorbar_axes, prepared_panels)

    return _save_figure(path, tight_layout=False)


def _plot_single_stimulus_sensitivity(
    leave_one_out: pd.DataFrame,
    ranked_models: list[str],
    path: Path,
) -> Path:
    if leave_one_out.empty:
        return _plot_empty_figure(path, title="Single-Stimulus Sensitivity", message="No sensitivity table")

    required_columns = {"excluded_stimulus", "model_id", "rsa_similarity"}
    if not required_columns.issubset(leave_one_out.columns):
        return _plot_empty_figure(path, title="Single-Stimulus Sensitivity", message="Missing sensitivity columns")

    summary = leave_one_out.copy()
    summary["excluded_stimulus"] = summary["excluded_stimulus"].astype(str)
    summary["model_id"] = summary["model_id"].astype(str)
    summary["rsa_similarity"] = pd.to_numeric(summary["rsa_similarity"], errors="coerce")
    summary = summary.loc[np.isfinite(summary["rsa_similarity"])]
    if ranked_models:
        summary = summary.loc[summary["model_id"].isin(ranked_models)]
    if summary.empty:
        return _plot_empty_figure(path, title="Single-Stimulus Sensitivity", message="No finite sensitivity data")

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
    plt.title("Single-Stimulus Sensitivity")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    return _save_figure(path)


def _write_aggregated_response_context_figures(
    core_outputs: dict[str, pd.DataFrame],
    dirs: dict[str, Path],
    written: dict[str, Path],
    *,
    top_models: dict[str, str],
) -> dict[str, Any]:
    aggregated_response_rsa_results = _dataframe_or_none(core_outputs, "aggregated_response_rsa_results__per_date")
    top_aggregated_response_models = _build_top_aggregated_response_models_by_date_and_view(
        aggregated_response_rsa_results
    )
    aggregated_response_rsa_views = _aggregated_response_rsa_views(aggregated_response_rsa_results)
    aggregated_response_comparison_views = _aggregated_response_views(core_outputs)
    aggregated_response_rdm_views = _aggregated_response_rdm_views(core_outputs)
    aggregated_response_figure_names = [
        *_build_aggregated_response_rsa_figure_names(aggregated_response_rsa_views),
        *_build_aggregated_response_rdm_comparison_figure_names(aggregated_response_comparison_views),
        *_build_aggregated_response_rdm_figure_names(aggregated_response_rdm_views),
    ]

    for figure_name, view_name in zip(
        _build_aggregated_response_rsa_figure_names(aggregated_response_rsa_views),
        aggregated_response_rsa_views,
        strict=False,
    ):
        written[figure_name] = _plot_aggregated_response_rsa_per_date(
            aggregated_response_rsa_results,
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )

    for figure_name, view_name in zip(
        _build_aggregated_response_rdm_comparison_figure_names(aggregated_response_comparison_views),
        aggregated_response_comparison_views,
        strict=False,
    ):
        written[figure_name] = _plot_aggregated_response_rdm_comparison_per_date(
            core_outputs,
            aggregated_response_rsa_results,
            top_aggregated_response_models,
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )

    for figure_name, view_name in zip(
        _build_aggregated_response_rdm_figure_names(aggregated_response_rdm_views),
        aggregated_response_rdm_views,
        strict=False,
    ):
        written[f"figure__{figure_name}"] = _plot_aggregated_response_pooled_rdm(
            core_outputs,
            _dataframe_or_none(core_outputs, figure_name),
            top_models,
            stimulus_sample_map=_dataframe_or_none(core_outputs, "stimulus_sample_map"),
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )

    return {
        "aggregated_response_context_enabled": _aggregated_response_context_enabled(core_outputs),
        "response_aggregation": _response_aggregation(core_outputs),
        "aggregated_response_views": _aggregated_response_views(core_outputs),
        "aggregated_response_dates": _aggregated_response_dates(core_outputs),
        "aggregated_response_table_names": _aggregated_response_table_names(core_outputs),
        "aggregated_response_figure_names": aggregated_response_figure_names,
        "aggregated_response_descriptive_outputs": _aggregated_response_descriptive_outputs(core_outputs),
    }


def _plot_aggregated_response_rsa_per_date(
    aggregated_response_rsa_results: pd.DataFrame | None,
    *,
    view_name: str,
    path: Path,
) -> Path:
    if aggregated_response_rsa_results is None or aggregated_response_rsa_results.empty:
        return _plot_empty_figure(
            path,
            title=f"Aggregated-Response RSA By Date ({view_name})",
            message="No aggregated-response RSA table",
        )

    required_columns = {"date", "view_name", "model_id", "rsa_similarity"}
    if not required_columns.issubset(aggregated_response_rsa_results.columns):
        return _plot_empty_figure(
            path,
            title=f"Aggregated-Response RSA By Date ({view_name})",
            message="Missing aggregated-response RSA columns",
        )

    summary = aggregated_response_rsa_results.copy()
    summary["date"] = summary["date"].astype(str)
    summary["view_name"] = summary["view_name"].astype(str)
    summary["model_id"] = summary["model_id"].astype(str)
    summary["rsa_similarity"] = pd.to_numeric(summary["rsa_similarity"], errors="coerce")
    summary = summary.loc[summary["view_name"] == view_name]
    if "score_status" in summary.columns:
        summary = summary.loc[_string_column(summary, "score_status").eq("ok")]
    summary = summary.loc[np.isfinite(summary["rsa_similarity"])]
    if summary.empty:
        return _plot_empty_figure(
            path,
            title=f"Aggregated-Response RSA By Date ({view_name})",
            message="No finite aggregated-response RSA values",
        )

    ordered_dates = sorted(summary["date"].unique().tolist())
    plt.figure(figsize=(max(6.0, 1.4 * len(ordered_dates) + 2.0), 4.5))
    for model_id, group in summary.groupby("model_id", sort=False):
        ordered = group.sort_values("date")
        plt.plot(
            ordered["date"],
            ordered["rsa_similarity"],
            marker="o",
            label=model_id,
        )
    plt.xlabel("Date")
    plt.ylabel("RSA similarity")
    plt.title(f"Aggregated-Response RSA By Date ({view_name})")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    return _save_figure(path)


def _plot_aggregated_response_rdm_comparison_per_date(
    core_outputs: dict[str, pd.DataFrame],
    aggregated_response_rsa_results: pd.DataFrame | None,
    top_aggregated_response_models: dict[tuple[str, str], str],
    *,
    view_name: str,
    path: Path,
) -> Path:
    ordered_dates = _aggregated_response_per_date_comparison_dates(
        core_outputs,
        aggregated_response_rsa_results,
        view_name=view_name,
    )
    if not ordered_dates:
        return _plot_empty_figure(
            path,
            title=f"Aggregated-Response RDM Comparison By Date ({view_name})",
            message="No aggregated-response data for this view",
        )

    stimulus_sample_map = _dataframe_or_none(core_outputs, "stimulus_sample_map")
    row_labels = ("neural", "model")
    figure, axes, colorbar_axes = _create_rdm_panel_figure(
        nrows=len(ordered_dates) * len(row_labels),
        figsize=(10.2, max(5.0, 3.4 * len(ordered_dates) * len(row_labels))),
    )
    figure.suptitle(f"Aggregated-Response RDM Comparison By Date ({view_name})", fontsize=12)

    prepared_panels: list[tuple[int, int, pd.DataFrame | None, str, str]] = []
    for date_offset, date_value in enumerate(ordered_dates):
        neural_matrix = _find_internal_aggregated_response_per_date_rdm(
            core_outputs,
            view_name=view_name,
            date_value=date_value,
        )
        top_model_id = top_aggregated_response_models.get((date_value, view_name))
        model_matrix = None
        if top_model_id:
            model_matrix = _find_matrix_frame(
                core_outputs,
                _model_rdm_aliases(top_model_id, view_name),
            )
            if model_matrix is not None and neural_matrix is not None:
                neural_labels = _coerce_rdm_heatmap_frame(neural_matrix).index.tolist()
                model_matrix = _restrict_rdm_to_labels(model_matrix, neural_labels)
        for order_offset, order_source in enumerate(row_labels):
            row_index = (date_offset * len(row_labels)) + order_offset
            order_labels_for_pair = _resolve_pair_order_labels(
                order_source=order_source,
                neural_matrix=neural_matrix,
                model_matrix=model_matrix,
                stimulus_sample_map=stimulus_sample_map,
            )
            prepared_panels.extend(
                [
                    (
                        row_index,
                        0,
                        _prepare_rdm_display_frame(
                            neural_matrix,
                            stimulus_sample_map,
                            order_labels=order_labels_for_pair,
                        ),
                        f"{date_value}: aggregated response ({order_source} order)",
                        "No per-date aggregated-response RDM",
                    ),
                    (
                        row_index,
                        1,
                        _prepare_rdm_display_frame(
                            model_matrix,
                            stimulus_sample_map,
                            order_labels=order_labels_for_pair,
                        ),
                        f"{date_value}: {top_model_id or 'no top model'} ({order_source} order)",
                        "No paired top-model RDM",
                    ),
                ]
            )

    _render_prepared_rdm_panels(figure, axes, colorbar_axes, prepared_panels)

    return _save_figure(path, tight_layout=False)


def _plot_aggregated_response_pooled_rdm(
    core_outputs: dict[str, pd.DataFrame],
    aggregated_response_rdm: pd.DataFrame | None,
    top_models: dict[str, str],
    *,
    stimulus_sample_map: pd.DataFrame | None,
    view_name: str,
    path: Path,
) -> Path:
    figure, axes, colorbar_axes = _create_rdm_panel_figure(nrows=2, figsize=(10.2, 7.0))
    figure.suptitle(f"Aggregated-Response Pooled RDM Comparison ({view_name})", fontsize=12)

    top_model_id = top_models.get(view_name)
    model_matrix = None
    if top_model_id:
        model_matrix = _find_matrix_frame(core_outputs, _model_rdm_aliases(top_model_id, view_name))
        if model_matrix is not None and aggregated_response_rdm is not None:
            response_labels = _coerce_rdm_heatmap_frame(aggregated_response_rdm).index.tolist()
            model_matrix = _restrict_rdm_to_labels(model_matrix, response_labels)

    prepared_panels: list[tuple[int, int, pd.DataFrame | None, str, str]] = []
    for row_index, order_source in enumerate(("neural", "model")):
        order_labels_for_pair = _resolve_pair_order_labels(
            order_source=order_source,
            neural_matrix=aggregated_response_rdm,
            model_matrix=model_matrix,
            stimulus_sample_map=stimulus_sample_map,
        )
        prepared_panels.extend(
            [
                (
                    row_index,
                    0,
                    _prepare_rdm_display_frame(
                        aggregated_response_rdm,
                        stimulus_sample_map,
                        order_labels=order_labels_for_pair,
                    ),
                    f"{view_name}: pooled aggregated response ({order_source} order)",
                    "No pooled aggregated-response RDM provided",
                ),
                (
                    row_index,
                    1,
                    _prepare_rdm_display_frame(
                        model_matrix,
                        stimulus_sample_map,
                        order_labels=order_labels_for_pair,
                    ),
                    f"{view_name}: {top_model_id or 'no top model'} ({order_source} order)",
                    "No paired top-model RDM",
                ),
            ]
        )
    _render_prepared_rdm_panels(figure, axes, colorbar_axes, prepared_panels)
    return _save_figure(path, tight_layout=False)


def _find_internal_aggregated_response_per_date_rdm(
    core_outputs: dict[str, pd.DataFrame],
    *,
    view_name: str,
    date_value: str,
) -> pd.DataFrame | None:
    return _dataframe_or_none(core_outputs, _internal_aggregated_response_per_date_rdm_key(view_name, date_value))


def _aggregated_response_per_date_comparison_dates(
    core_outputs: dict[str, pd.DataFrame],
    aggregated_response_rsa_results: pd.DataFrame | None,
    *,
    view_name: str,
) -> list[str]:
    dates = set(_aggregated_response_internal_per_date_dates(core_outputs, view_name=view_name))
    if aggregated_response_rsa_results is None or aggregated_response_rsa_results.empty:
        return sorted(dates)
    if not {"date", "view_name"}.issubset(aggregated_response_rsa_results.columns):
        return sorted(dates)

    view_rows = aggregated_response_rsa_results.copy()
    view_rows["date"] = view_rows["date"].astype(str)
    view_rows["view_name"] = view_rows["view_name"].astype(str)
    dates.update(view_rows.loc[view_rows["view_name"] == view_name, "date"].tolist())
    return sorted(date_value for date_value in dates if date_value)


def _aggregated_response_internal_per_date_dates(
    core_outputs: dict[str, pd.DataFrame],
    *,
    view_name: str,
) -> list[str]:
    prefix = f"{INTERNAL_AGGREGATED_RESPONSE_PER_DATE_RDM_PREFIX}{view_name}__"
    return sorted(
        artifact_name.removeprefix(prefix)
        for artifact_name, frame in core_outputs.items()
        if artifact_name.startswith(prefix) and isinstance(frame, pd.DataFrame)
    )


def _find_matrix_frame(core_outputs: dict[str, pd.DataFrame], aliases: tuple[str, ...]) -> pd.DataFrame | None:
    for alias in aliases:
        frame = core_outputs.get(alias)
        if isinstance(frame, pd.DataFrame):
            return frame
    return None


def _model_rdm_aliases(model_id: str, view_name: str) -> tuple[str, str, str]:
    return (
        f"model_rdm__{model_id}__{view_name}",
        f"model_rdm__{view_name}__{model_id}",
        f"model_rdm__{model_id}",
    )


def _restrict_rdm_to_labels(matrix_frame: pd.DataFrame, order_labels: list[str]) -> pd.DataFrame | None:
    if not order_labels:
        return None

    heatmap_frame = _coerce_rdm_heatmap_frame(matrix_frame)
    if heatmap_frame.empty or not set(order_labels).issubset(heatmap_frame.index):
        return None

    restricted = heatmap_frame.loc[order_labels, order_labels].copy()
    restricted.insert(0, "stimulus_row", restricted.index.astype(str))
    return restricted.reset_index(drop=True)


def _resolve_pair_order_labels(
    *,
    order_source: str,
    neural_matrix: pd.DataFrame | None,
    model_matrix: pd.DataFrame | None,
    stimulus_sample_map: pd.DataFrame | None,
) -> list[str]:
    preferred_matrix = neural_matrix if order_source == "neural" else model_matrix
    fallback_matrix = model_matrix if order_source == "neural" else neural_matrix
    for candidate in (preferred_matrix, fallback_matrix):
        if candidate is None:
            continue
        _, order_labels = _prepare_rdm_heatmap_frame(candidate, stimulus_sample_map)
        if order_labels:
            return order_labels
    return []


def _prepare_rdm_display_frame(
    matrix_frame: pd.DataFrame | None,
    stimulus_sample_map: pd.DataFrame | None,
    *,
    order_labels: list[str] | None = None,
) -> pd.DataFrame | None:
    if matrix_frame is None:
        return None

    heatmap_frame, _ = _prepare_rdm_heatmap_frame(matrix_frame, stimulus_sample_map, order_labels=order_labels)
    if heatmap_frame.empty:
        return None
    return _mask_rdm_diagonal(heatmap_frame)


def _resolve_rdm_heatmap_frame(
    matrix_frame: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame | None,
) -> pd.DataFrame:
    heatmap_frame = _coerce_rdm_heatmap_frame(matrix_frame)
    if heatmap_frame.empty or stimulus_sample_map is None or stimulus_sample_map.empty:
        return heatmap_frame

    resolved_labels = _resolve_display_labels(heatmap_frame.index.tolist(), stimulus_sample_map)
    if resolved_labels is None:
        return heatmap_frame

    resolved_frame = heatmap_frame.copy()
    resolved_frame.index = pd.Index(resolved_labels)
    resolved_frame.columns = pd.Index(resolved_labels)
    return resolved_frame


def _mask_rdm_diagonal(heatmap_frame: pd.DataFrame) -> pd.DataFrame:
    masked = heatmap_frame.copy()
    diagonal_length = min(masked.shape)
    for diagonal_index in range(diagonal_length):
        masked.iat[diagonal_index, diagonal_index] = np.nan
    return masked


def _render_prepared_rdm_panels(
    figure: plt.Figure,
    axes: np.ndarray,
    colorbar_axes: np.ndarray,
    panels: list[tuple[int, int, pd.DataFrame | None, str, str]],
) -> None:
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#f2f2f2")

    for row_index, col_index, frame, title, fallback_message in panels:
        axis = axes[row_index, col_index]
        colorbar_axis = colorbar_axes[row_index, col_index]
        axis.set_title(title)
        image = _render_prepared_rdm_axis(
            axis,
            frame,
            fallback_message=fallback_message,
            cmap=cmap,
        )
        if image is None:
            colorbar_axis.set_visible(False)
            continue
        try:
            colorbar_axis.set_visible(True)
            figure.colorbar(image, cax=colorbar_axis, label="RDM dissimilarity")
        except Exception as exc:
            warnings.warn(
                f"RDM colorbar failed for panel ({row_index}, {col_index}): {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            colorbar_axis.set_visible(False)


def _create_rdm_panel_figure(
    *,
    nrows: int,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(
        nrows=nrows,
        ncols=4,
        width_ratios=(1.0, 0.06, 1.0, 0.06),
        left=0.07,
        right=0.96,
        bottom=0.08,
        top=0.9,
        wspace=0.28,
        hspace=0.34,
    )
    axes = np.empty((nrows, 2), dtype=object)
    colorbar_axes = np.empty((nrows, 2), dtype=object)
    for row_index in range(nrows):
        axes[row_index, 0] = figure.add_subplot(grid[row_index, 0])
        colorbar_axes[row_index, 0] = figure.add_subplot(grid[row_index, 1])
        axes[row_index, 1] = figure.add_subplot(grid[row_index, 2])
        colorbar_axes[row_index, 1] = figure.add_subplot(grid[row_index, 3])
    return figure, axes, colorbar_axes


def _render_prepared_rdm_axis(
    axis: plt.Axes,
    heatmap_frame: pd.DataFrame | None,
    *,
    fallback_message: str,
    cmap: matplotlib.colors.Colormap,
) -> matplotlib.image.AxesImage | None:
    if heatmap_frame is None or heatmap_frame.empty:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return None

    display_frame = _coerce_rdm_heatmap_frame(heatmap_frame)
    display_parameters = _compute_rdm_display_parameters(display_frame)
    if display_parameters is None:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return None

    values = display_frame.to_numpy(dtype=float)
    axis.set_axis_on()
    image = axis.imshow(values, cmap=cmap, norm=display_parameters.norm)
    axis.set_xticks(np.arange(len(display_frame.columns)))
    axis.set_yticks(np.arange(len(display_frame.index)))
    axis.set_xticklabels(display_frame.columns.tolist(), rotation=45, ha="right")
    axis.set_yticklabels(display_frame.index.tolist())
    return image


def _compute_rdm_display_parameters(
    heatmap_frame: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> RdmDisplayParameters | None:
    finite_off_diagonal_values = _finite_off_diagonal_values(heatmap_frame)
    if finite_off_diagonal_values.size == 0:
        return None

    quantiles = np.quantile(finite_off_diagonal_values, [lower_quantile, upper_quantile])
    if not np.all(np.isfinite(quantiles)):
        vmin = float(np.min(finite_off_diagonal_values))
        vmax = float(np.max(finite_off_diagonal_values))
    else:
        vmin = float(quantiles[0])
        vmax = float(quantiles[1])

    if vmin > vmax:
        vmin, vmax = vmax, vmin

    finite_min = float(np.min(finite_off_diagonal_values))
    finite_max = float(np.max(finite_off_diagonal_values))
    if vmin == vmax:
        if finite_min != finite_max:
            vmin, vmax = finite_min, finite_max
        else:
            constant_value = finite_min
            padding = max(abs(constant_value) * 0.05, 1e-6)
            vmin = constant_value - padding
            vmax = constant_value + padding

    norm = matplotlib.colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax, clip=True)
    return RdmDisplayParameters(vmin=vmin, vmax=vmax, norm=norm)


def _finite_off_diagonal_values(heatmap_frame: pd.DataFrame) -> np.ndarray:
    values = _coerce_rdm_heatmap_frame(heatmap_frame).to_numpy(dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)

    finite_mask = np.isfinite(values)
    diagonal_length = min(values.shape)
    if diagonal_length:
        diagonal_indices = np.arange(diagonal_length)
        finite_mask[diagonal_indices, diagonal_indices] = False
    return values[finite_mask]


def _prepare_rdm_heatmap_frame(
    matrix_frame: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame | None,
    *,
    order_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    heatmap_frame = _coerce_rdm_heatmap_frame(matrix_frame)
    if heatmap_frame.empty:
        return heatmap_frame, []

    if order_labels is None:
        ordered_labels = _cluster_reorder_heatmap_labels(heatmap_frame)
    else:
        ordered_labels = [str(label) for label in order_labels]
        if set(ordered_labels) != set(heatmap_frame.index):
            ordered_labels = heatmap_frame.index.tolist()

    ordered_frame = heatmap_frame.reindex(index=ordered_labels, columns=ordered_labels)
    if stimulus_sample_map is None or stimulus_sample_map.empty:
        return ordered_frame, ordered_labels

    resolved_labels = _resolve_display_labels(ordered_labels, stimulus_sample_map)
    if resolved_labels is None:
        return ordered_frame, ordered_labels

    resolved_frame = ordered_frame.copy()
    resolved_frame.index = pd.Index(resolved_labels)
    resolved_frame.columns = pd.Index(resolved_labels)
    return resolved_frame, ordered_labels


def _cluster_reorder_heatmap_labels(heatmap_frame: pd.DataFrame) -> list[str]:
    original_order = heatmap_frame.index.tolist()
    if len(original_order) < 3:
        return original_order

    numeric = heatmap_frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=True)
    if np.isnan(values).any() or not np.isfinite(values).all():
        return original_order

    np.fill_diagonal(values, 0.0)
    try:
        linkage_matrix = linkage(squareform(values, checks=False), method="average", optimal_ordering=True)
        order = leaves_list(linkage_matrix).tolist()
    except Exception:
        return original_order
    return [original_order[position] for position in order]


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


def _resolve_display_labels(
    stimulus_order: list[str],
    stimulus_sample_map: pd.DataFrame,
) -> list[str] | None:
    if not stimulus_order:
        return None
    if "stimulus" not in stimulus_sample_map.columns:
        return None

    map_frame = stimulus_sample_map.copy()
    map_frame["stimulus"] = map_frame["stimulus"].fillna("").astype(str).str.strip()
    map_frame = map_frame.loc[map_frame["stimulus"] != ""]
    subset = map_frame.loc[map_frame["stimulus"].isin(stimulus_order)]
    if subset.empty:
        return None
    if len(subset["stimulus"].unique()) != len(stimulus_order):
        return None
    if subset["stimulus"].duplicated().any():
        return None

    stimulus_lookup = subset.set_index("stimulus", drop=False)

    for candidate_column in ("sample_id", "stim_name", "stimulus"):
        if candidate_column not in stimulus_lookup.columns:
            continue

        candidate_series = stimulus_lookup[candidate_column].reindex(stimulus_order)
        if candidate_series.isna().any():
            continue
        candidate_series = candidate_series.fillna("").astype(str).str.strip()
        if candidate_series.empty or (candidate_series == "").any():
            continue
        if candidate_series.duplicated().any():
            continue

        return candidate_series.tolist()

    return None


def _plot_empty_figure(path: Path, *, title: str, message: str) -> Path:
    plt.figure(figsize=(6.5, 4.0))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.title(title)
    return _save_figure(path)


def _dataframe_or_none(core_outputs: dict[str, pd.DataFrame], key: str) -> pd.DataFrame | None:
    frame = core_outputs.get(key)
    if isinstance(frame, pd.DataFrame):
        return frame
    return None


def _build_run_summary(
    *,
    required_tables: dict[str, pd.DataFrame],
    required_qc: dict[str, pd.DataFrame],
    written: dict[str, Path],
    group_summary: dict[str, list[str]],
    top_models: dict[str, str],
    focus_view: str | None,
    aggregated_response_summary: dict[str, Any],
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
    figure_view_names = _figure_view_names(required_tables["rsa_results"], required_tables["rsa_view_comparison"])
    figure_names = [
        *REQUIRED_FIGURES,
        *_build_neural_vs_model_figure_names(figure_view_names),
        *aggregated_response_summary["aggregated_response_figure_names"],
    ]
    ranked_model_rsa_details = _build_ranked_model_rsa_details(
        required_tables["rsa_results"],
        group_summary["ranked_models"],
        focus_view=focus_view,
    )
    view_comparison_details = _build_view_comparison_details(required_tables["rsa_view_comparison"])

    return {
        "views": view_names,
        "focus_view": focus_view,
        "sensitivity_views": [view_name for view_name in view_names if view_name != focus_view],
        "ranked_models": group_summary["ranked_models"],
        "additional_models": group_summary["additional_models"],
        "excluded_models": group_summary["excluded_models"],
        "ranked_model_rsa_details": ranked_model_rsa_details,
        "view_comparison_details": view_comparison_details,
        "top_models_by_view": top_models,
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
        "aggregated_response_context_enabled": aggregated_response_summary["aggregated_response_context_enabled"],
        "response_aggregation": aggregated_response_summary["response_aggregation"],
        "aggregated_response_views": aggregated_response_summary["aggregated_response_views"],
        "aggregated_response_dates": aggregated_response_summary["aggregated_response_dates"],
        "aggregated_response_table_names": aggregated_response_summary["aggregated_response_table_names"],
        "aggregated_response_figure_names": aggregated_response_summary["aggregated_response_figure_names"],
        "aggregated_response_descriptive_outputs": aggregated_response_summary[
            "aggregated_response_descriptive_outputs"
        ],
    }


def _aggregated_response_context_enabled(core_outputs: dict[str, pd.DataFrame]) -> bool:
    return any(
        key.startswith("aggregated_response_") and isinstance(value, pd.DataFrame) for key, value in core_outputs.items()
    )


def _response_aggregation(core_outputs: dict[str, pd.DataFrame]) -> str | None:
    config = _dataframe_or_none(core_outputs, INTERNAL_RESPONSE_AGGREGATION_KEY)
    if config is None or config.empty or "response_aggregation" not in config.columns:
        return None
    value = str(config["response_aggregation"].iloc[0]).strip().lower()
    return value or None


def _aggregated_response_views(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    view_names: list[str] = []
    aggregated_response_rsa_results = _dataframe_or_none(core_outputs, "aggregated_response_rsa_results__per_date")
    aggregated_response_support_per_date = _dataframe_or_none(core_outputs, "aggregated_response_support__per_date")
    aggregated_response_support_pooled = _dataframe_or_none(core_outputs, "aggregated_response_support__pooled")
    for frame in (
        aggregated_response_rsa_results,
        aggregated_response_support_per_date,
        aggregated_response_support_pooled,
    ):
        if frame is None or "view_name" not in frame.columns:
            continue
        view_names.extend(frame["view_name"].astype(str).tolist())
    view_names.extend(_aggregated_response_rdm_views(core_outputs))
    return _canonicalize_view_order(view_names)


def _aggregated_response_rsa_views(aggregated_response_rsa_results: pd.DataFrame | None) -> list[str]:
    if aggregated_response_rsa_results is None or "view_name" not in aggregated_response_rsa_results.columns:
        return []
    return _canonicalize_view_order(aggregated_response_rsa_results["view_name"].astype(str).tolist())


def _aggregated_response_rdm_views(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    view_names = [
        artifact_name.removeprefix("aggregated_response_rdm__pooled__")
        for artifact_name, frame in core_outputs.items()
        if artifact_name.startswith("aggregated_response_rdm__pooled__") and isinstance(frame, pd.DataFrame)
    ]
    return _canonicalize_view_order(view_names)


def _aggregated_response_dates(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    dates: set[str] = set()
    for artifact_name in ("aggregated_response_rsa_results__per_date", "aggregated_response_support__per_date"):
        frame = _dataframe_or_none(core_outputs, artifact_name)
        if frame is None or "date" not in frame.columns:
            continue
        for value in frame["date"].dropna().astype(str):
            if value:
                dates.add(value)
    return sorted(dates)


def _aggregated_response_table_names(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    table_names: list[str] = []
    if _dataframe_or_none(core_outputs, "aggregated_response_rsa_results__per_date") is not None:
        table_names.append("aggregated_response_rsa_results__per_date")
    table_names.extend(_build_aggregated_response_rdm_figure_names(_aggregated_response_rdm_views(core_outputs)))
    return table_names


def _aggregated_response_descriptive_outputs(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    return _build_aggregated_response_rdm_figure_names(_aggregated_response_rdm_views(core_outputs))


def _ordered_views(rsa_results: pd.DataFrame, view_comparison: pd.DataFrame) -> list[str]:
    views: list[str] = []
    if not rsa_results.empty and "view_name" in rsa_results.columns:
        views.extend(rsa_results["view_name"].astype(str).tolist())
    if not view_comparison.empty and "view_name" in view_comparison.columns:
        views.extend(view_comparison["view_name"].astype(str).tolist())
    return _canonicalize_view_order(views)


def _figure_view_names(rsa_results: pd.DataFrame, view_comparison: pd.DataFrame) -> list[str]:
    view_names = _ordered_views(rsa_results, view_comparison)
    if view_names:
        return view_names
    return _canonicalize_view_order(list(DEFAULT_FIGURE_VIEWS))


def _choose_focus_view(rsa_results: pd.DataFrame, *, view_candidates: list[str] | None = None) -> str | None:
    views = view_candidates or _ordered_views(rsa_results, pd.DataFrame())
    if not views:
        return None
    if "response_window" in views:
        return "response_window"
    return views[0]


def _write_markdown_summary(summary: dict[str, Any], path: str | Path) -> Path:
    lines = [
        "# Biochemical RSA Run Summary",
        "",
        "## Views",
        f"- Focus view: {summary['focus_view'] or 'None'}",
        f"- Sensitivity views: {', '.join(summary['sensitivity_views']) if summary['sensitivity_views'] else 'None'}",
        "",
        "## Model Groups",
        f"- Ranked models: {', '.join(summary['ranked_models']) if summary['ranked_models'] else 'None'}",
        f"- Additional models: {', '.join(summary['additional_models']) if summary['additional_models'] else 'None'}",
        f"- Excluded models: {', '.join(summary['excluded_models']) if summary['excluded_models'] else 'None'}",
        "",
        "## Top Models By View",
    ]

    top_models = summary["top_models_by_view"]
    if top_models:
        lines.extend(f"- {view_name}: {model_id}" for view_name, model_id in top_models.items())
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Ranked Model RSA Details",
        ]
    )
    if summary["ranked_model_rsa_details"]:
        lines.extend(_format_ranked_model_rsa_detail(detail) for detail in summary["ranked_model_rsa_details"])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## View Comparison Details",
        ]
    )
    if summary["view_comparison_details"]:
        lines.extend(_format_view_comparison_detail(detail) for detail in summary["view_comparison_details"])
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
        ]
    )

    if summary["aggregated_response_context_enabled"]:
        lines.extend(
            [
                "## Aggregated Response Context",
                f"- Response aggregation: {summary['response_aggregation'] or 'None'}",
                f"- Views: {', '.join(summary['aggregated_response_views']) if summary['aggregated_response_views'] else 'None'}",
                f"- Dates: {', '.join(summary['aggregated_response_dates']) if summary['aggregated_response_dates'] else 'None'}",
                f"- Aggregated-response tables: {', '.join(summary['aggregated_response_table_names']) if summary['aggregated_response_table_names'] else 'None'}",
                f"- Aggregated-response figures: {', '.join(summary['aggregated_response_figure_names']) if summary['aggregated_response_figure_names'] else 'None'}",
                f"- Aggregated-response descriptive outputs: {', '.join(summary['aggregated_response_descriptive_outputs']) if summary['aggregated_response_descriptive_outputs'] else 'None'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Paths",
            f"- Tables directory: {summary['tables_dir']}",
            f"- Figures directory: {summary['figures_dir']}",
            f"- QC directory: {summary['qc_dir']}",
        ]
    )

    output_path = Path(path)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _format_ranked_model_rsa_detail(detail: dict[str, Any]) -> str:
    return (
        f"- {detail.get('model_id', 'None')} | view={_summary_markdown_value(detail.get('view_name'))} | "
        f"rsa={_summary_markdown_value(detail.get('rsa_similarity'))} | "
        f"p_raw={_summary_markdown_value(detail.get('p_value_raw'))} | "
        f"p_fdr={_summary_markdown_value(detail.get('p_value_fdr'))} | "
        f"n={_summary_markdown_value(detail.get('n_shared_entries'))} | "
        f"status={_summary_markdown_value(detail.get('score_status'))} | "
        f"top={_summary_markdown_value(detail.get('is_top_model'))}"
    )


def _format_view_comparison_detail(detail: dict[str, Any]) -> str:
    return (
        f"- {_summary_markdown_value(detail.get('view_name'))} vs "
        f"{_summary_markdown_value(detail.get('reference_view_name'))} | "
        f"scope={_summary_markdown_value(detail.get('comparison_scope'))} | "
        f"rsa={_summary_markdown_value(detail.get('rsa_similarity'))} | "
        f"p_raw={_summary_markdown_value(detail.get('p_value_raw'))} | "
        f"p_fdr={_summary_markdown_value(detail.get('p_value_fdr'))} | "
        f"n={_summary_markdown_value(detail.get('n_shared_entries'))} | "
        f"status={_summary_markdown_value(detail.get('score_status'))}"
    )


def _summary_markdown_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "None"
    return str(value)


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
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes"})


def _save_figure(path: Path, *, tight_layout: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.gcf()
    if tight_layout:
        figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


__all__ = [
    "ensure_rsa_output_dirs",
    "write_rsa_outputs",
    "ensure_stage3_output_dirs",
    "write_stage3_outputs",
]

ensure_stage3_output_dirs = ensure_rsa_output_dirs
write_stage3_outputs = write_rsa_outputs

