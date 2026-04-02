"""Output writers and figures for Stage 3 biochemical RSA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    "ranked_primary_model_rsa",
    "leave_one_stimulus_out_robustness",
    "view_comparison_summary",
)
DEFAULT_FIGURE_VIEWS: tuple[str, ...] = ("response_window", "full_trajectory")
PROTOTYPE_QC_ARTIFACT_NAMES: tuple[str, ...] = (
    "prototype_support__per_date",
    "prototype_support__pooled",
)
INTERNAL_ONLY_ARTIFACT_PREFIX = "internal__"
INTERNAL_PROTOTYPE_PER_DATE_RDM_PREFIX = f"{INTERNAL_ONLY_ARTIFACT_PREFIX}prototype_rdm__per_date__"


def _build_neural_vs_model_figure_names(view_names: list[str]) -> list[str]:
    return [f"neural_vs_top_model_rdm__{view_name}" for view_name in view_names]


def _build_prototype_rsa_figure_names(view_names: list[str]) -> list[str]:
    return [f"prototype_rsa__per_date__{view_name}" for view_name in view_names]


def _build_prototype_rdm_comparison_figure_names(view_names: list[str]) -> list[str]:
    return [f"prototype_rdm_comparison__per_date__{view_name}" for view_name in view_names]


def _build_prototype_rdm_figure_names(view_names: list[str]) -> list[str]:
    return [f"prototype_rdm__pooled__{view_name}" for view_name in view_names]


def _canonicalize_view_order(view_names: list[str]) -> list[str]:
    unique_views = sorted({str(view_name) for view_name in view_names})
    ordered_views: list[str] = []
    for preferred_view in ("response_window", "full_trajectory"):
        if preferred_view in unique_views:
            ordered_views.append(preferred_view)
            unique_views.remove(preferred_view)
    ordered_views.extend(unique_views)
    return ordered_views


def _internal_prototype_per_date_rdm_key(view_name: str, date_value: str) -> str:
    return f"{INTERNAL_PROTOTYPE_PER_DATE_RDM_PREFIX}{view_name}__{date_value}"


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

    _remove_legacy_stage3_figures(dirs["figures_dir"])
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
    family_summary = _collect_model_families(registry)
    top_primary_models = _build_top_primary_models_by_view(rsa_results, family_summary["primary_models"])
    primary_view = _choose_primary_view(rsa_results, view_candidates=view_names)

    written["ranked_primary_model_rsa"] = _plot_ranked_primary_model_rsa(
        rsa_results,
        family_summary["primary_models"],
        path=dirs["figures_dir"] / "ranked_primary_model_rsa.png",
        primary_view=primary_view,
    )
    for figure_name, view_name in zip(
        _build_neural_vs_model_figure_names(figure_view_names), figure_view_names, strict=False
    ):
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
    prototype_summary = _write_prototype_supplementary_figures(core_outputs, dirs, written)

    summary = _build_run_summary(
        required_tables=required_tables,
        required_qc=required_qc,
        written=written,
        family_summary=family_summary,
        top_primary_models=top_primary_models,
        primary_view=primary_view,
        prototype_summary=prototype_summary,
    )
    written["run_summary_json"] = write_json(summary, dirs["output_root"] / "run_summary.json")
    written["run_summary_md"] = _write_markdown_summary(summary, dirs["output_root"] / "run_summary.md")
    return written


def _remove_legacy_stage3_figures(figures_dir: Path) -> None:
    legacy_figure = figures_dir / "neural_vs_top_model_rdm_panel.png"
    if legacy_figure.exists():
        legacy_figure.unlink()

    for stale_figure in figures_dir.glob("neural_vs_top_model_rdm__*.png"):
        stale_figure.unlink()
    for stale_figure in figures_dir.glob("prototype_*.png"):
        stale_figure.unlink()


def _remove_stale_prototype_parquets(tables_dir: Path, qc_dir: Path) -> None:
    for directory in (tables_dir, qc_dir):
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


def _build_top_prototype_models_by_date_and_view(prototype_rsa_results: pd.DataFrame | None) -> dict[tuple[str, str], str]:
    if prototype_rsa_results is None or prototype_rsa_results.empty:
        return {}

    required_columns = {"date", "view_name", "model_id", "is_top_model"}
    if not required_columns.issubset(prototype_rsa_results.columns):
        return {}

    selected = prototype_rsa_results.copy()
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
        model_matrix = _find_matrix_frame(
            core_outputs,
            (
                f"model_rdm__{top_model_id}__{view_name}",
                f"model_rdm__{view_name}__{top_model_id}",
                f"model_rdm__{top_model_id}",
            ),
        )

    neural_order_labels: list[str] | None = []
    if neural_matrix is not None:
        _, neural_order_labels = _prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)

    _render_rdm_axis(
        axes[0, 0],
        neural_matrix,
        stimulus_sample_map=stimulus_sample_map,
        title=f"{view_name}: neural",
        fallback_message="No neural matrix provided",
        order_labels=neural_order_labels,
    )
    _render_rdm_axis(
        axes[0, 1],
        model_matrix,
        stimulus_sample_map=stimulus_sample_map,
        title=f"{view_name}: {top_model_id or 'no top model'}",
        fallback_message="No top-model matrix provided",
        order_labels=neural_order_labels,
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


def _write_prototype_supplementary_figures(
    core_outputs: dict[str, pd.DataFrame],
    dirs: dict[str, Path],
    written: dict[str, Path],
) -> dict[str, Any]:
    prototype_rsa_results = _dataframe_or_none(core_outputs, "prototype_rsa_results__per_date")
    top_prototype_models = _build_top_prototype_models_by_date_and_view(prototype_rsa_results)
    prototype_rsa_views = _prototype_rsa_views(prototype_rsa_results)
    prototype_rdm_views = _prototype_rdm_views(core_outputs)
    prototype_figure_names = [
        *_build_prototype_rsa_figure_names(prototype_rsa_views),
        *_build_prototype_rdm_comparison_figure_names(prototype_rsa_views),
        *_build_prototype_rdm_figure_names(prototype_rdm_views),
    ]

    for figure_name, view_name in zip(
        _build_prototype_rsa_figure_names(prototype_rsa_views),
        prototype_rsa_views,
        strict=False,
    ):
        written[figure_name] = _plot_prototype_rsa_per_date(
            prototype_rsa_results,
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )

    for figure_name, view_name in zip(
        _build_prototype_rdm_comparison_figure_names(prototype_rsa_views),
        prototype_rsa_views,
        strict=False,
    ):
        written[figure_name] = _plot_prototype_rdm_comparison_per_date(
            core_outputs,
            prototype_rsa_results,
            top_prototype_models,
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )

    for figure_name, view_name in zip(
        _build_prototype_rdm_figure_names(prototype_rdm_views),
        prototype_rdm_views,
        strict=False,
    ):
        written[f"figure__{figure_name}"] = _plot_prototype_pooled_rdm(
            _dataframe_or_none(core_outputs, figure_name),
            stimulus_sample_map=_dataframe_or_none(core_outputs, "stimulus_sample_map"),
            view_name=view_name,
            path=dirs["figures_dir"] / f"{figure_name}.png",
        )

    return {
        "prototype_supplement_enabled": _prototype_supplement_enabled(core_outputs),
        "prototype_views": _prototype_views(core_outputs),
        "prototype_dates": _prototype_dates(core_outputs),
        "prototype_table_names": _prototype_table_names(core_outputs),
        "prototype_figure_names": prototype_figure_names,
        "prototype_descriptive_outputs": _prototype_descriptive_outputs(core_outputs),
    }


def _plot_prototype_rsa_per_date(
    prototype_rsa_results: pd.DataFrame | None,
    *,
    view_name: str,
    path: Path,
) -> Path:
    if prototype_rsa_results is None or prototype_rsa_results.empty:
        return _plot_empty_figure(path, title=f"Prototype RSA By Date ({view_name})", message="No prototype RSA table")

    required_columns = {"date", "view_name", "model_id", "rsa_similarity"}
    if not required_columns.issubset(prototype_rsa_results.columns):
        return _plot_empty_figure(
            path,
            title=f"Prototype RSA By Date ({view_name})",
            message="Missing prototype RSA columns",
        )

    summary = prototype_rsa_results.copy()
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
            title=f"Prototype RSA By Date ({view_name})",
            message="No finite prototype RSA values",
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
    plt.title(f"Prototype RSA By Date ({view_name})")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    return _save_figure(path)


def _plot_prototype_rdm_comparison_per_date(
    core_outputs: dict[str, pd.DataFrame],
    prototype_rsa_results: pd.DataFrame | None,
    top_prototype_models: dict[tuple[str, str], str],
    *,
    view_name: str,
    path: Path,
) -> Path:
    if prototype_rsa_results is None or prototype_rsa_results.empty:
        return _plot_empty_figure(
            path,
            title=f"Prototype RDM Comparison By Date ({view_name})",
            message="No prototype RSA table",
        )

    required_columns = {"date", "view_name"}
    if not required_columns.issubset(prototype_rsa_results.columns):
        return _plot_empty_figure(
            path,
            title=f"Prototype RDM Comparison By Date ({view_name})",
            message="Missing prototype RSA columns",
        )

    view_rows = prototype_rsa_results.copy()
    view_rows["date"] = view_rows["date"].astype(str)
    view_rows["view_name"] = view_rows["view_name"].astype(str)
    view_rows = view_rows.loc[view_rows["view_name"] == view_name]
    ordered_dates = sorted(view_rows["date"].unique().tolist())
    if not ordered_dates:
        return _plot_empty_figure(
            path,
            title=f"Prototype RDM Comparison By Date ({view_name})",
            message="No prototype data for this view",
        )

    stimulus_sample_map = _dataframe_or_none(core_outputs, "stimulus_sample_map")
    figure, axes = plt.subplots(
        nrows=len(ordered_dates),
        ncols=2,
        figsize=(9.5, max(4.0, 3.8 * len(ordered_dates))),
        squeeze=False,
    )
    figure.suptitle(f"Prototype RDM Comparison By Date ({view_name})", fontsize=12)

    for row_index, date_value in enumerate(ordered_dates):
        neural_matrix = _find_internal_prototype_per_date_rdm(core_outputs, view_name=view_name, date_value=date_value)
        neural_order_labels: list[str] = []
        if neural_matrix is not None:
            _, neural_order_labels = _prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)

        top_model_id = top_prototype_models.get((date_value, view_name))
        model_matrix = None
        if top_model_id and neural_order_labels:
            model_matrix = _find_matrix_frame(
                core_outputs,
                (
                    f"model_rdm__{top_model_id}__{view_name}",
                    f"model_rdm__{view_name}__{top_model_id}",
                    f"model_rdm__{top_model_id}",
                ),
            )
            if model_matrix is not None:
                model_matrix = _restrict_rdm_to_labels(model_matrix, neural_order_labels)

        _render_rdm_axis(
            axes[row_index, 0],
            neural_matrix,
            stimulus_sample_map=stimulus_sample_map,
            title=f"{date_value}: neural prototype",
            fallback_message="No per-date neural prototype RDM",
            order_labels=neural_order_labels,
        )
        _render_rdm_axis(
            axes[row_index, 1],
            model_matrix,
            stimulus_sample_map=stimulus_sample_map,
            title=f"{date_value}: {top_model_id or 'no top model'}",
            fallback_message="No paired top-model RDM",
            order_labels=neural_order_labels,
        )

    return _save_figure(path)


def _plot_prototype_pooled_rdm(
    prototype_rdm: pd.DataFrame | None,
    *,
    stimulus_sample_map: pd.DataFrame | None,
    view_name: str,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    figure.suptitle(f"Prototype Pooled RDM ({view_name})", fontsize=12)

    order_labels: list[str] | None = []
    if prototype_rdm is not None:
        order_labels = _coerce_rdm_heatmap_frame(prototype_rdm).index.tolist()

    _render_rdm_axis(
        axis,
        prototype_rdm,
        stimulus_sample_map=stimulus_sample_map,
        title=f"{view_name}: pooled prototype",
        fallback_message="No pooled prototype RDM provided",
        order_labels=order_labels,
    )
    return _save_figure(path)


def _find_internal_prototype_per_date_rdm(
    core_outputs: dict[str, pd.DataFrame],
    *,
    view_name: str,
    date_value: str,
) -> pd.DataFrame | None:
    return _dataframe_or_none(core_outputs, _internal_prototype_per_date_rdm_key(view_name, date_value))


def _find_matrix_frame(core_outputs: dict[str, pd.DataFrame], aliases: tuple[str, ...]) -> pd.DataFrame | None:
    for alias in aliases:
        frame = core_outputs.get(alias)
        if isinstance(frame, pd.DataFrame):
            return frame
    return None


def _restrict_rdm_to_labels(matrix_frame: pd.DataFrame, order_labels: list[str]) -> pd.DataFrame | None:
    if not order_labels:
        return None

    heatmap_frame = _coerce_rdm_heatmap_frame(matrix_frame)
    if heatmap_frame.empty or not set(order_labels).issubset(heatmap_frame.index):
        return None

    restricted = heatmap_frame.loc[order_labels, order_labels].copy()
    restricted.insert(0, "stimulus_row", restricted.index.astype(str))
    return restricted.reset_index(drop=True)


def _render_rdm_axis(
    axis: plt.Axes,
    matrix_frame: pd.DataFrame | None,
    *,
    stimulus_sample_map: pd.DataFrame | None,
    title: str,
    fallback_message: str,
    order_labels: list[str] | None = None,
) -> None:
    axis.set_title(title)
    if matrix_frame is None:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return

    heatmap_frame, _ = _prepare_rdm_heatmap_frame(matrix_frame, stimulus_sample_map, order_labels=order_labels)
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
    family_summary: dict[str, list[str]],
    top_primary_models: dict[str, str],
    primary_view: str | None,
    prototype_summary: dict[str, Any],
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
        *prototype_summary["prototype_figure_names"],
    ]

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
        "prototype_supplement_enabled": prototype_summary["prototype_supplement_enabled"],
        "prototype_views": prototype_summary["prototype_views"],
        "prototype_dates": prototype_summary["prototype_dates"],
        "prototype_table_names": prototype_summary["prototype_table_names"],
        "prototype_figure_names": prototype_summary["prototype_figure_names"],
        "prototype_descriptive_outputs": prototype_summary["prototype_descriptive_outputs"],
    }


def _prototype_supplement_enabled(core_outputs: dict[str, pd.DataFrame]) -> bool:
    return any(key.startswith("prototype_") and isinstance(value, pd.DataFrame) for key, value in core_outputs.items())


def _prototype_views(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    view_names: list[str] = []
    prototype_rsa_results = _dataframe_or_none(core_outputs, "prototype_rsa_results__per_date")
    prototype_support_per_date = _dataframe_or_none(core_outputs, "prototype_support__per_date")
    prototype_support_pooled = _dataframe_or_none(core_outputs, "prototype_support__pooled")
    for frame in (prototype_rsa_results, prototype_support_per_date, prototype_support_pooled):
        if frame is None or "view_name" not in frame.columns:
            continue
        view_names.extend(frame["view_name"].astype(str).tolist())
    view_names.extend(_prototype_rdm_views(core_outputs))
    return _canonicalize_view_order(view_names)


def _prototype_rsa_views(prototype_rsa_results: pd.DataFrame | None) -> list[str]:
    if prototype_rsa_results is None or "view_name" not in prototype_rsa_results.columns:
        return []
    return _canonicalize_view_order(prototype_rsa_results["view_name"].astype(str).tolist())


def _prototype_rdm_views(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    view_names = [
        artifact_name.removeprefix("prototype_rdm__pooled__")
        for artifact_name, frame in core_outputs.items()
        if artifact_name.startswith("prototype_rdm__pooled__") and isinstance(frame, pd.DataFrame)
    ]
    return _canonicalize_view_order(view_names)


def _prototype_dates(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    dates: set[str] = set()
    for artifact_name in ("prototype_rsa_results__per_date", "prototype_support__per_date"):
        frame = _dataframe_or_none(core_outputs, artifact_name)
        if frame is None or "date" not in frame.columns:
            continue
        for value in frame["date"].dropna().astype(str):
            if value:
                dates.add(value)
    return sorted(dates)


def _prototype_table_names(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    table_names: list[str] = []
    if _dataframe_or_none(core_outputs, "prototype_rsa_results__per_date") is not None:
        table_names.append("prototype_rsa_results__per_date")
    table_names.extend(_build_prototype_rdm_figure_names(_prototype_rdm_views(core_outputs)))
    return table_names


def _prototype_descriptive_outputs(core_outputs: dict[str, pd.DataFrame]) -> list[str]:
    return _build_prototype_rdm_figure_names(_prototype_rdm_views(core_outputs))


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
        ]
    )

    if summary["prototype_supplement_enabled"]:
        lines.extend(
            [
                "## Prototype Supplement",
                f"- Views: {', '.join(summary['prototype_views']) if summary['prototype_views'] else 'None'}",
                f"- Dates: {', '.join(summary['prototype_dates']) if summary['prototype_dates'] else 'None'}",
                f"- Prototype tables: {', '.join(summary['prototype_table_names']) if summary['prototype_table_names'] else 'None'}",
                f"- Prototype figures: {', '.join(summary['prototype_figure_names']) if summary['prototype_figure_names'] else 'None'}",
                f"- Prototype descriptive outputs: {', '.join(summary['prototype_descriptive_outputs']) if summary['prototype_descriptive_outputs'] else 'None'}",
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

