"""Stage 3 RSA statistics and robustness helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bacteria_analysis.io import read_parquet
from bacteria_analysis.model_space import build_model_rdm, summarize_model_input_coverage
from bacteria_analysis.rsa_prototypes import PrototypeSupplementInputs, build_grouped_prototypes, build_prototype_rdm

DEFAULT_CROSS_VIEW_NAMES = ("response_window", "full_trajectory")


def align_rdm_upper_triangles(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> pd.DataFrame:
    """Align shared upper-triangle entries by stimulus labels."""

    neural_triangle = _extract_upper_triangle(neural_matrix).rename(columns={"value": "neural_value"})
    model_triangle = _extract_upper_triangle(model_matrix).rename(columns={"value": "model_value"})
    shared = neural_triangle.merge(model_triangle, on=["stimulus_left", "stimulus_right"], how="inner")
    if shared.empty:
        return pd.DataFrame(columns=["stimulus_left", "stimulus_right", "neural_value", "model_value"])

    value_columns = ["neural_value", "model_value"]
    shared[value_columns] = shared[value_columns].apply(pd.to_numeric, errors="coerce")
    return shared.reset_index(drop=True)


def compute_rsa_score(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> dict[str, object]:
    shared = align_rdm_upper_triangles(neural_matrix, model_matrix)
    shared = shared.dropna(subset=["neural_value", "model_value"]).reset_index(drop=True)
    n_shared_entries = int(len(shared))
    if n_shared_entries < 2:
        return {
            "score_method": "spearman",
            "score_status": "invalid",
            "n_shared_entries": n_shared_entries,
            "rsa_similarity": np.nan,
            "p_value_raw": np.nan,
            "p_value_fdr": np.nan,
        }

    rsa_similarity = _spearman_similarity(
        shared["neural_value"].to_numpy(dtype=float),
        shared["model_value"].to_numpy(dtype=float),
    )
    score_status = "ok" if np.isfinite(rsa_similarity) else "invalid"
    return {
        "score_method": "spearman",
        "score_status": score_status,
        "n_shared_entries": n_shared_entries,
        "rsa_similarity": rsa_similarity,
        "p_value_raw": np.nan,
        "p_value_fdr": np.nan,
    }


def build_permutation_null(
    neural_matrix: pd.DataFrame,
    model_matrix: pd.DataFrame,
    n_iterations: int,
    seed: int,
) -> pd.DataFrame:
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    model_square = _prepare_square_matrix(model_matrix)
    for iteration in range(n_iterations):
        permuted_model = _matrix_to_frame(_permute_model_labels(model_square, rng))
        score = compute_rsa_score(neural_matrix, permuted_model)
        rows.append({"iteration": iteration, **score})
    return pd.DataFrame(
        rows,
        columns=[
            "iteration",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be a one-dimensional array")
    if values.size == 0:
        return values.copy()

    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return adjusted

    order = np.argsort(finite_values)
    ranked = finite_values[order]
    scale = finite_values.size / np.arange(1, finite_values.size + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    finite_indices = np.flatnonzero(finite_mask)
    adjusted[finite_indices[order]] = ranked_adjusted
    return adjusted


def summarize_leave_one_stimulus_out(
    neural_matrix: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    n_iterations: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    shared_labels = sorted(_shared_stimulus_labels(neural_matrix, model_matrix))
    rows: list[dict[str, object]] = []
    for offset, stimulus_label in enumerate(shared_labels):
        reduced_neural = _matrix_to_frame(_drop_stimulus(_prepare_square_matrix(neural_matrix), stimulus_label))
        reduced_model = _matrix_to_frame(_drop_stimulus(_prepare_square_matrix(model_matrix), stimulus_label))
        score = compute_rsa_score(reduced_neural, reduced_model)
        p_value_raw = np.nan
        if n_iterations > 0 and score["score_status"] == "ok":
            null = build_permutation_null(
                reduced_neural,
                reduced_model,
                n_iterations=n_iterations,
                seed=seed + offset,
            )
            p_value_raw = _empirical_p_value(score["rsa_similarity"], null["rsa_similarity"].to_numpy(dtype=float))
        rows.append(
            {
                "excluded_stimulus": stimulus_label,
                **score,
                "p_value_raw": p_value_raw,
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "excluded_stimulus",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )
    if summary.empty:
        return summary
    summary["p_value_fdr"] = benjamini_hochberg(summary["p_value_raw"].to_numpy(dtype=float))
    return summary


def summarize_cross_view_comparison(
    neural_matrices: dict[str, pd.DataFrame],
    model_matrix: pd.DataFrame,
    *,
    view_names: tuple[str, ...] | list[str] = DEFAULT_CROSS_VIEW_NAMES,
    n_iterations: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    requested_views: list[str] = []
    for view_name in view_names:
        normalized = str(view_name)
        if normalized not in neural_matrices:
            raise ValueError(f"missing neural matrix for view_name {normalized!r}")
        if normalized not in requested_views:
            requested_views.append(normalized)

    rows: list[dict[str, object]] = []
    for offset, view_name in enumerate(requested_views):
        score = compute_rsa_score(neural_matrices[view_name], model_matrix)
        p_value_raw = np.nan
        if n_iterations > 0 and score["score_status"] == "ok":
            null = build_permutation_null(
                neural_matrices[view_name],
                model_matrix,
                n_iterations=n_iterations,
                seed=seed + offset,
            )
            p_value_raw = _empirical_p_value(score["rsa_similarity"], null["rsa_similarity"].to_numpy(dtype=float))
        rows.append(
            {
                "view_name": view_name,
                **score,
                "p_value_raw": p_value_raw,
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "view_name",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )
    if summary.empty:
        return summary
    summary["p_value_fdr"] = benjamini_hochberg(summary["p_value_raw"].to_numpy(dtype=float))
    return summary


def load_stage2_pooled_neural_rdms(
    stage2_root: str | Path,
    *,
    view_names: tuple[str, ...] | list[str] = DEFAULT_CROSS_VIEW_NAMES,
) -> dict[str, pd.DataFrame]:
    root = Path(stage2_root)
    tables_dir = root / "tables"
    neural_rdms: dict[str, pd.DataFrame] = {}

    for view_name in view_names:
        normalized_view = str(view_name)
        matrix_path = tables_dir / f"rdm_matrix__{normalized_view}__pooled.parquet"
        if not matrix_path.exists():
            raise FileNotFoundError(f"missing Stage 2 pooled RDM for view {normalized_view!r}: {matrix_path}")
        neural_rdms[normalized_view] = read_parquet(matrix_path)

    return neural_rdms


def run_stage3_rsa(
    resolved_inputs: dict[str, pd.DataFrame],
    *,
    neural_matrices: dict[str, pd.DataFrame],
    prototype_inputs: PrototypeSupplementInputs | None = None,
    permutations: int = 0,
    seed: int = 0,
    view_names: tuple[str, ...] | list[str] = DEFAULT_CROSS_VIEW_NAMES,
    primary_view: str = "response_window",
) -> dict[str, pd.DataFrame]:
    requested_views = _resolve_requested_views(neural_matrices, view_names)
    resolved_primary_view = primary_view if primary_view in requested_views else requested_views[0]
    registry = resolved_inputs["model_registry_resolved"].copy()
    results_rows: list[dict[str, object]] = []
    leave_one_out_frames: list[pd.DataFrame] = []
    cross_view_frames: list[pd.DataFrame] = []
    model_summary_rows: list[dict[str, object]] = []

    core_outputs: dict[str, pd.DataFrame] = {
        "stimulus_sample_map": resolved_inputs["stimulus_sample_map"].copy(),
        "metabolite_annotation_resolved": resolved_inputs["metabolite_annotation"].copy(),
        "model_membership_resolved": resolved_inputs["model_membership_resolved"].copy(),
    }
    for view_name in requested_views:
        core_outputs[f"neural_rdm__{view_name}"] = neural_matrices[view_name].copy()

    for _, registry_row in registry.iterrows():
        model_id = str(registry_row["model_id"]).strip().lower()
        model_status = str(registry_row["model_status"]).strip().lower()
        if model_status == "excluded":
            continue

        try:
            model_matrix = build_model_rdm(resolved_inputs, model_id)
        except ValueError as exc:
            if model_id == "global_profile" or not _should_skip_model_build_error(exc):
                raise
            _mark_model_excluded_from_primary_ranking(resolved_inputs, model_id)
            continue
        core_outputs[f"model_rdm__{model_id}"] = model_matrix.copy()
        model_summary_rows.append(_build_model_rdm_summary_row(model_id, registry_row, model_matrix))

        for view_index, view_name in enumerate(requested_views):
            score = compute_rsa_score(neural_matrices[view_name], model_matrix)
            p_value_raw = np.nan
            if permutations > 0 and score["score_status"] == "ok":
                null = build_permutation_null(
                    neural_matrices[view_name],
                    model_matrix,
                    n_iterations=permutations,
                    seed=seed + (1000 * view_index),
                )
                p_value_raw = _empirical_p_value(
                    score["rsa_similarity"],
                    null["rsa_similarity"].to_numpy(dtype=float),
                )

            results_rows.append(
                {
                    "view_name": view_name,
                    "reference_view_name": "model_rdm",
                    "comparison_scope": "neural_vs_model",
                    "primary_view": resolved_primary_view,
                    "model_id": model_id,
                    "model_label": str(registry_row["model_label"]).strip(),
                    "model_tier": str(registry_row["model_tier"]).strip().lower(),
                    "model_status": model_status,
                    "feature_kind": str(registry_row["feature_kind"]).strip().lower(),
                    "distance_kind": str(registry_row["distance_kind"]).strip().lower(),
                    "excluded_from_primary_ranking": bool(registry_row.get("excluded_from_primary_ranking", False)),
                    **score,
                    "p_value_raw": p_value_raw,
                    "p_value_fdr": np.nan,
                }
            )

        leave_one_out = summarize_leave_one_stimulus_out(
            neural_matrices[resolved_primary_view],
            model_matrix,
            n_iterations=permutations,
            seed=seed,
        )
        if not leave_one_out.empty:
            leave_one_out = leave_one_out.assign(
                view_name=resolved_primary_view,
                reference_view_name="model_rdm",
                comparison_scope="neural_vs_model",
                primary_view=resolved_primary_view,
                model_id=model_id,
                model_label=str(registry_row["model_label"]).strip(),
                model_tier=str(registry_row["model_tier"]).strip().lower(),
                model_status=model_status,
                distance_kind=str(registry_row["distance_kind"]).strip().lower(),
            )
        leave_one_out_frames.append(leave_one_out)

        cross_view = summarize_cross_view_comparison(
            {view_name: neural_matrices[view_name] for view_name in requested_views},
            model_matrix,
            view_names=requested_views,
            n_iterations=permutations,
            seed=seed,
        )
        if not cross_view.empty:
            cross_view = cross_view.assign(
                reference_view_name="model_rdm",
                comparison_scope="neural_vs_model",
                primary_view=resolved_primary_view,
                model_id=model_id,
                model_label=str(registry_row["model_label"]).strip(),
                model_tier=str(registry_row["model_tier"]).strip().lower(),
                model_status=model_status,
                distance_kind=str(registry_row["distance_kind"]).strip().lower(),
            )
        cross_view_frames.append(cross_view)

    rsa_results = pd.DataFrame.from_records(results_rows)
    if not rsa_results.empty:
        rsa_results["p_value_fdr"] = benjamini_hochberg(rsa_results["p_value_raw"].to_numpy(dtype=float))
        rsa_results["is_top_model"] = False
        for view_name, group in rsa_results.groupby("view_name", sort=False):
            valid_group = group.loc[
                group["score_status"].astype(str).str.strip().str.lower().eq("ok")
                & np.isfinite(pd.to_numeric(group["rsa_similarity"], errors="coerce"))
            ]
            if valid_group.empty:
                continue
            top_index = valid_group["rsa_similarity"].astype(float).idxmax()
            rsa_results.loc[top_index, "is_top_model"] = True

    core_outputs["model_registry_resolved"] = resolved_inputs["model_registry_resolved"].copy()
    core_outputs["model_input_coverage"] = summarize_model_input_coverage(resolved_inputs)
    core_outputs["model_feature_qc"] = resolved_inputs.get("model_feature_qc", pd.DataFrame()).copy()
    core_outputs["model_feature_filtering"] = core_outputs["model_feature_qc"].copy()
    core_outputs["model_rdm_summary"] = pd.DataFrame.from_records(model_summary_rows)
    core_outputs["rsa_results"] = rsa_results
    core_outputs["rsa_leave_one_stimulus_out"] = _concat_summary_frames(
        leave_one_out_frames,
        columns=[
            "excluded_stimulus",
            "view_name",
            "reference_view_name",
            "comparison_scope",
            "primary_view",
            "model_id",
            "model_label",
            "model_tier",
            "model_status",
            "distance_kind",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )
    core_outputs["rsa_view_comparison"] = _concat_summary_frames(
        cross_view_frames,
        columns=[
            "view_name",
            "reference_view_name",
            "comparison_scope",
            "primary_view",
            "model_id",
            "model_label",
            "model_tier",
            "model_status",
            "distance_kind",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )
    if prototype_inputs is not None:
        core_outputs.update(
            _build_prototype_supplement_outputs(
                resolved_inputs,
                core_outputs=core_outputs,
                prototype_inputs=prototype_inputs,
                view_names=requested_views,
                permutations=permutations,
                seed=seed,
            )
        )
    return core_outputs


def _build_prototype_supplement_outputs(
    resolved_inputs: dict[str, pd.DataFrame],
    *,
    core_outputs: dict[str, pd.DataFrame],
    prototype_inputs: PrototypeSupplementInputs,
    view_names: list[str],
    permutations: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    registry = resolved_inputs["model_registry_resolved"].copy()
    if "excluded_from_primary_ranking" not in registry.columns:
        registry["excluded_from_primary_ranking"] = False

    built_model_ids = core_outputs.get("model_rdm_summary", pd.DataFrame()).get("model_id", pd.Series(dtype=object))
    built_model_ids = built_model_ids.astype(str).str.strip().str.lower().tolist()
    registry = registry.loc[
        registry["model_id"].astype(str).str.strip().str.lower().isin(built_model_ids)
    ].reset_index(drop=True)

    per_date_rows: list[dict[str, object]] = []
    per_date_support_frames: list[pd.DataFrame] = []
    pooled_support_frames: list[pd.DataFrame] = []
    pooled_rdms: dict[str, pd.DataFrame] = {}

    for view_index, view_name in enumerate(view_names):
        if view_name not in prototype_inputs.views:
            raise ValueError(f"missing prototype view for view_name {view_name!r}")

        view = prototype_inputs.views[view_name]
        per_date_prototypes, per_date_support = build_grouped_prototypes(
            view,
            group_columns=("date", "stimulus", "stim_name"),
        )
        pooled_prototypes, pooled_support = build_grouped_prototypes(
            view,
            group_columns=("stimulus", "stim_name"),
        )
        per_date_prototypes = _ensure_frame_schema(per_date_prototypes, ["date", "stimulus", "stim_name"])
        pooled_prototypes = _ensure_frame_schema(pooled_prototypes, ["stimulus", "stim_name"])
        per_date_support = _ensure_frame_schema(
            per_date_support,
            [
                "date",
                "stimulus",
                "stim_name",
                "n_trials",
                "n_total_features",
                "n_supported_features",
                "n_all_nan_features",
            ],
        )
        pooled_support = _ensure_frame_schema(
            pooled_support,
            [
                "stimulus",
                "stim_name",
                "n_trials",
                "n_dates_contributed",
                "n_total_features",
                "n_supported_features",
                "n_all_nan_features",
            ],
        )

        per_date_support_frames.append(
            per_date_support.assign(view_name=view_name)[
                [
                    "date",
                    "view_name",
                    "stimulus",
                    "stim_name",
                    "n_trials",
                    "n_total_features",
                    "n_supported_features",
                    "n_all_nan_features",
                ]
            ]
        )
        pooled_support_frames.append(
            pooled_support.assign(view_name=view_name)[
                [
                    "view_name",
                    "stimulus",
                    "stim_name",
                    "n_trials",
                    "n_dates_contributed",
                    "n_total_features",
                    "n_supported_features",
                    "n_all_nan_features",
                ]
            ]
        )
        pooled_rdms[f"prototype_rdm__pooled__{view_name}"] = (
            build_prototype_rdm(pooled_prototypes, id_columns=("stimulus",))
            if not pooled_prototypes.empty
            else pd.DataFrame(columns=["stimulus_row"])
        )

        for date_index, (date_value, date_prototypes) in enumerate(per_date_prototypes.groupby("date", sort=False)):
            date_prototypes = date_prototypes.reset_index(drop=True)
            stimulus_labels = date_prototypes["stimulus"].astype(str).tolist()
            neural_matrix = (
                build_prototype_rdm(date_prototypes, id_columns=("stimulus",))
                if not date_prototypes.empty
                else pd.DataFrame(columns=["stimulus_row"])
            )

            for model_index, registry_row in registry.iterrows():
                model_id = str(registry_row["model_id"]).strip().lower()
                model_matrix = _restrict_matrix_to_labels(
                    core_outputs[f"model_rdm__{model_id}"],
                    stimulus_labels,
                )
                score = compute_rsa_score(neural_matrix, model_matrix)
                p_value_raw = np.nan
                if permutations > 0 and score["score_status"] == "ok":
                    null = build_permutation_null(
                        neural_matrix,
                        model_matrix,
                        n_iterations=permutations,
                        seed=seed + (10000 * view_index) + (1000 * date_index) + model_index,
                    )
                    p_value_raw = _empirical_p_value(
                        score["rsa_similarity"],
                        null["rsa_similarity"].to_numpy(dtype=float),
                    )

                per_date_rows.append(
                    {
                        "date": str(date_value),
                        "view_name": view_name,
                        "reference_view_name": "model_rdm",
                        "comparison_scope": "prototype_neural_vs_model",
                        "model_id": model_id,
                        "model_label": str(registry_row["model_label"]).strip(),
                        "model_tier": str(registry_row["model_tier"]).strip().lower(),
                        "model_status": str(registry_row["model_status"]).strip().lower(),
                        "feature_kind": str(registry_row["feature_kind"]).strip().lower(),
                        "distance_kind": str(registry_row["distance_kind"]).strip().lower(),
                        "excluded_from_primary_ranking": bool(
                            registry_row.get("excluded_from_primary_ranking", False)
                        ),
                        "score_method": score["score_method"],
                        "score_status": score["score_status"],
                        "n_stimuli": int(len(stimulus_labels)),
                        "n_shared_entries": score["n_shared_entries"],
                        "rsa_similarity": score["rsa_similarity"],
                        "p_value_raw": p_value_raw,
                        "p_value_fdr": np.nan,
                        "is_top_model": False,
                    }
                )

    prototype_rsa_results = pd.DataFrame.from_records(
        per_date_rows,
        columns=[
            "date",
            "view_name",
            "reference_view_name",
            "comparison_scope",
            "model_id",
            "model_label",
            "model_tier",
            "model_status",
            "feature_kind",
            "distance_kind",
            "excluded_from_primary_ranking",
            "score_method",
            "score_status",
            "n_stimuli",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
            "is_top_model",
        ],
    )
    if not prototype_rsa_results.empty:
        prototype_rsa_results["p_value_fdr"] = benjamini_hochberg(
            prototype_rsa_results["p_value_raw"].to_numpy(dtype=float)
        )
        for _, group in prototype_rsa_results.groupby(["date", "view_name"], sort=False):
            eligible = group.loc[
                group["score_status"].astype(str).str.strip().str.lower().eq("ok")
                & ~group["excluded_from_primary_ranking"].astype(bool)
                & np.isfinite(pd.to_numeric(group["rsa_similarity"], errors="coerce"))
            ]
            if eligible.empty:
                continue
            top_index = eligible["rsa_similarity"].astype(float).idxmax()
            prototype_rsa_results.loc[top_index, "is_top_model"] = True

    outputs: dict[str, pd.DataFrame] = {
        "prototype_rsa_results__per_date": prototype_rsa_results,
        "prototype_support__per_date": _concat_summary_frames(
            per_date_support_frames,
            columns=[
                "date",
                "view_name",
                "stimulus",
                "stim_name",
                "n_trials",
                "n_total_features",
                "n_supported_features",
                "n_all_nan_features",
            ],
        ),
        "prototype_support__pooled": _concat_summary_frames(
            pooled_support_frames,
            columns=[
                "view_name",
                "stimulus",
                "stim_name",
                "n_trials",
                "n_dates_contributed",
                "n_total_features",
                "n_supported_features",
                "n_all_nan_features",
            ],
        ),
    }
    outputs.update(pooled_rdms)
    return outputs


def _prepare_square_matrix(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    if "stimulus_row" not in matrix_frame.columns:
        raise ValueError("matrix_frame must include a stimulus_row column")

    matrix = matrix_frame.set_index("stimulus_row").copy()
    if not matrix.index.is_unique:
        duplicates = matrix.index[matrix.index.duplicated()].unique().tolist()
        raise ValueError(f"matrix index contains duplicate stimulus labels: {duplicates}")
    if not matrix.columns.is_unique:
        duplicates = matrix.columns[matrix.columns.duplicated()].unique().tolist()
        raise ValueError(f"matrix columns contain duplicate stimulus labels: {duplicates}")

    normalized_index = pd.Index(matrix.index.map(str))
    normalized_columns = pd.Index(matrix.columns.map(str))
    if not normalized_index.is_unique:
        duplicates = normalized_index[normalized_index.duplicated()].unique().tolist()
        raise ValueError(f"matrix index labels collide after string normalization: {duplicates}")
    if not normalized_columns.is_unique:
        duplicates = normalized_columns[normalized_columns.duplicated()].unique().tolist()
        raise ValueError(f"matrix columns labels collide after string normalization: {duplicates}")

    matrix.index = normalized_index
    matrix.columns = normalized_columns
    if set(matrix.index) != set(matrix.columns):
        raise ValueError(
            "matrix columns must match stimulus_row labels; "
            f"index={matrix.index.tolist()} columns={matrix.columns.tolist()}"
        )

    matrix = matrix.reindex(columns=matrix.index)
    return matrix.apply(pd.to_numeric, errors="coerce")


def _extract_upper_triangle(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    matrix = _prepare_square_matrix(matrix_frame)
    rows: list[dict[str, object]] = []
    labels = matrix.index.tolist()
    for row_idx, stimulus_left in enumerate(labels):
        for col_idx in range(row_idx + 1, len(labels)):
            stimulus_pair = sorted((stimulus_left, labels[col_idx]))
            rows.append(
                {
                    "stimulus_left": stimulus_pair[0],
                    "stimulus_right": stimulus_pair[1],
                    "value": matrix.iloc[row_idx, col_idx],
                }
            )
    return pd.DataFrame(rows, columns=["stimulus_left", "stimulus_right", "value"])


def _spearman_similarity(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_ranks = pd.Series(left_values, copy=False).rank(method="average").to_numpy(dtype=float)
    right_ranks = pd.Series(right_values, copy=False).rank(method="average").to_numpy(dtype=float)
    if np.unique(left_ranks).size < 2 or np.unique(right_ranks).size < 2:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _shared_stimulus_labels(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> list[str]:
    neural_labels = _prepare_square_matrix(neural_matrix).index.tolist()
    model_labels = set(_prepare_square_matrix(model_matrix).index.tolist())
    return [label for label in neural_labels if label in model_labels]


def _drop_stimulus(matrix: pd.DataFrame, stimulus_label: str) -> pd.DataFrame:
    if stimulus_label not in matrix.index:
        return matrix.copy()
    return matrix.drop(index=stimulus_label, columns=stimulus_label)


def _restrict_matrix_to_labels(matrix_frame: pd.DataFrame, stimulus_labels: list[str]) -> pd.DataFrame:
    matrix = _prepare_square_matrix(matrix_frame)
    shared_labels = [label for label in stimulus_labels if label in matrix.index]
    if not shared_labels:
        return pd.DataFrame(columns=["stimulus_row"])
    return _matrix_to_frame(matrix.loc[shared_labels, shared_labels].copy())


def _matrix_to_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    frame = matrix.copy()
    frame.insert(0, "stimulus_row", frame.index.astype(str))
    return frame.reset_index(drop=True)


def _permute_model_labels(model_matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    labels = np.array(sorted(model_matrix.index.astype(str).tolist()), dtype=object)
    canonical = model_matrix.loc[labels, labels].copy()
    permuted_labels = rng.permutation(labels)
    permuted = canonical.loc[permuted_labels, permuted_labels].copy()
    permuted.index = labels
    permuted.columns = labels
    return permuted


def _empirical_p_value(observed_similarity: float, null_values: np.ndarray) -> float:
    finite_null = np.asarray(null_values, dtype=float)
    finite_null = finite_null[np.isfinite(finite_null)]
    if finite_null.size == 0 or not np.isfinite(observed_similarity):
        return float("nan")
    return float((1 + np.sum(finite_null >= observed_similarity)) / (finite_null.size + 1))


def _resolve_requested_views(
    neural_matrices: dict[str, pd.DataFrame],
    view_names: tuple[str, ...] | list[str],
) -> list[str]:
    requested_views: list[str] = []
    for view_name in view_names:
        normalized_view = str(view_name)
        if normalized_view not in neural_matrices:
            raise ValueError(f"missing neural matrix for view_name {normalized_view!r}")
        if normalized_view not in requested_views:
            requested_views.append(normalized_view)
    return requested_views


def _build_model_rdm_summary_row(
    model_id: str,
    registry_row: pd.Series,
    model_matrix: pd.DataFrame,
) -> dict[str, object]:
    distances = _extract_upper_triangle(model_matrix)["value"].to_numpy(dtype=float, copy=False)
    finite_distances = distances[np.isfinite(distances)]
    return {
        "model_id": model_id,
        "model_label": str(registry_row["model_label"]).strip(),
        "model_tier": str(registry_row["model_tier"]).strip().lower(),
        "model_status": str(registry_row["model_status"]).strip().lower(),
        "distance_kind": str(registry_row["distance_kind"]).strip().lower(),
        "n_stimuli": int(len(model_matrix)),
        "n_pairwise_distances": int(finite_distances.size),
        "mean_distance": float(finite_distances.mean()) if finite_distances.size else np.nan,
    }


def _concat_summary_frames(frames: list[pd.DataFrame], *, columns: list[str]) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty_frames, ignore_index=True)[columns]


def _ensure_frame_schema(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = pd.Series(dtype=object)
    ordered_columns = columns + [column for column in normalized.columns if column not in columns]
    return normalized[ordered_columns]


def _should_skip_model_build_error(exc: ValueError) -> bool:
    message = str(exc)
    return any(
        snippet in message
        for snippet in (
            "at least 1 retained feature",
            "at least 2 retained features",
            "non-constant stimulus rows",
        )
    )


def _mark_model_excluded_from_primary_ranking(resolved_inputs: dict[str, pd.DataFrame], model_id: str) -> None:
    registry = resolved_inputs["model_registry_resolved"].copy()
    mask = registry["model_id"].astype(str).str.strip().str.lower() == model_id
    if "excluded_from_primary_ranking" not in registry.columns:
        registry["excluded_from_primary_ranking"] = False
    registry.loc[mask, "excluded_from_primary_ranking"] = True
    resolved_inputs["model_registry_resolved"] = registry


__all__ = [
    "DEFAULT_CROSS_VIEW_NAMES",
    "align_rdm_upper_triangles",
    "benjamini_hochberg",
    "build_permutation_null",
    "compute_rsa_score",
    "load_stage2_pooled_neural_rdms",
    "run_stage3_rsa",
    "summarize_cross_view_comparison",
    "summarize_leave_one_stimulus_out",
]
