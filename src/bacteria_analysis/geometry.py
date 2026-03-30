"""Stage 2 geometry aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bacteria_analysis.reliability import (
    DEFAULT_DISTANCE_METRIC,
    VALID_COMPARISON_STATUS,
    VIEW_WINDOWS,
    build_trial_views,
    compute_pairwise_distances,
    load_reliability_inputs,
)

GROUP_TYPES = ("pooled", "individual", "date")
DEFAULT_STAGE2_VIEWS = ("response_window", "full_trajectory")


def parse_stage2_views(view_names: str | list[str] | tuple[str, ...] | None = None) -> list[str]:
    available_views = set(VIEW_WINDOWS)
    if view_names is None:
        requested = list(DEFAULT_STAGE2_VIEWS)
    elif isinstance(view_names, str):
        requested = [part.strip() for part in view_names.split(",")]
    else:
        requested = [str(part).strip() for part in view_names]

    normalized: list[str] = []
    for view_name in requested:
        if not view_name:
            continue
        if view_name not in available_views:
            supported = ", ".join(sorted(available_views))
            raise ValueError(f"unsupported Stage 2 view: {view_name!r}. Supported views: {supported}")
        if view_name not in normalized:
            normalized.append(view_name)

    if not normalized:
        raise ValueError("at least one Stage 2 view is required")
    return normalized


def run_geometry_pipeline(
    input_root: str | Path,
    view_names: str | list[str] | tuple[str, ...] | None = None,
    metric: str = DEFAULT_DISTANCE_METRIC,
) -> dict[str, pd.DataFrame]:
    input_root = Path(input_root)
    input_paths = {
        "metadata": input_root / "trial_level" / "trial_metadata.parquet",
        "tensor": input_root / "trial_level" / "trial_tensor_baseline_centered.npz",
    }
    missing = [f"{name}={path}" for name, path in input_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Stage 0 inputs: " + ", ".join(missing))

    inputs = load_reliability_inputs(
        metadata_path=input_paths["metadata"],
        tensor_path=input_paths["tensor"],
    )
    selected_view_names = parse_stage2_views(view_names)
    view_windows = {view_name: VIEW_WINDOWS[view_name] for view_name in selected_view_names}
    views = build_trial_views(inputs.metadata, inputs.tensor, view_windows=view_windows)
    comparisons = pd.concat(
        [
            compute_pairwise_distances(views[view_name], metric=metric)
            for view_name in selected_view_names
        ],
        ignore_index=True,
    )

    core_outputs: dict[str, pd.DataFrame] = {}
    pair_summary_frames: list[pd.DataFrame] = []
    for view_name in selected_view_names:
        for group_type in GROUP_TYPES:
            pair_summary = summarize_grouped_stimulus_pairs(comparisons, view_name=view_name, group_type=group_type)
            core_outputs[f"rdm_pairs__{view_name}__{group_type}"] = pair_summary
            pair_summary_frames.append(pair_summary)
        core_outputs[f"rdm_matrix__{view_name}__pooled"] = build_rdm_matrix(
            core_outputs[f"rdm_pairs__{view_name}__pooled"],
            group_id="pooled",
        )

    pair_summary_by_group = _concat_frames(pair_summary_frames)
    stability = summarize_rdm_stability(pair_summary_by_group)
    core_outputs["rdm_stability_by_individual"] = _filter_stability_table(stability, group_type="individual")
    core_outputs["rdm_stability_by_date"] = _filter_stability_table(stability, group_type="date")
    core_outputs["rdm_view_comparison"] = _filter_stability_table(stability, comparison_scope="pooled_cross_view")
    core_outputs["rdm_group_coverage"] = build_rdm_group_coverage(pair_summary_by_group, view_names=selected_view_names)
    return core_outputs


def summarize_grouped_stimulus_pairs(comparisons: pd.DataFrame, view_name: str, group_type: str) -> pd.DataFrame:
    if group_type not in GROUP_TYPES:
        raise ValueError(f"unsupported group_type: {group_type}")
    view_frame = comparisons.loc[comparisons["view_name"] == view_name].copy()
    return _aggregate_grouped_pairs(view_frame, group_type=group_type)


def build_rdm_matrix(pair_summary: pd.DataFrame, group_id: str) -> pd.DataFrame:
    group = pair_summary.loc[pair_summary["group_id"] == group_id].copy()
    if group.empty:
        return pd.DataFrame(columns=["stimulus_row"])

    combinations = group[["view_name", "group_type"]].drop_duplicates()
    if len(combinations) != 1:
        found = combinations.to_dict("records")
        raise ValueError(
            "build_rdm_matrix requires exactly one (view_name, group_type) combination "
            f"for group_id {group_id!r}; found {found}"
        )

    matrix = _pivot_symmetric_distance_matrix(group, value_column="mean_distance")
    if matrix.empty:
        return pd.DataFrame(columns=["stimulus_row"])
    frame = matrix.copy()
    frame.insert(0, "stimulus_row", frame.index.astype(str))
    return frame.reset_index(drop=True)


def extract_upper_triangle(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix_frame.set_index("stimulus_row")
    return _matrix_upper_triangle_records(matrix)


def score_rdm_similarity(left_matrix: pd.DataFrame, right_matrix: pd.DataFrame) -> dict[str, object]:
    shared = _shared_upper_triangle(left_matrix, right_matrix)
    return _score_shared_triangle(shared, method="spearman")


def summarize_rdm_stability(pair_summary_by_group: pd.DataFrame) -> pd.DataFrame:
    grouped_matrices = build_group_matrices(pair_summary_by_group)
    return _score_group_matrix_sets(grouped_matrices)


def build_group_matrices(pair_summary_by_group: pd.DataFrame) -> pd.DataFrame:
    columns = ["view_name", "group_type", "group_id", "matrix_frame"]
    if pair_summary_by_group.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (view_name, group_type, group_id), group in pair_summary_by_group.groupby(
        ["view_name", "group_type", "group_id"],
        sort=False,
        dropna=False,
    ):
        rows.append(
            {
                "view_name": view_name,
                "group_type": group_type,
                "group_id": group_id,
                "matrix_frame": build_rdm_matrix(group, group_id=group_id),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_rdm_group_coverage(pair_summary_by_group: pd.DataFrame, view_names: list[str]) -> pd.DataFrame:
    columns = ["view_name", "group_type", "n_groups"]
    rows: list[dict[str, object]] = []
    for view_name in view_names:
        for group_type in ("individual", "date"):
            subset = pair_summary_by_group.loc[
                (pair_summary_by_group["view_name"] == view_name) & (pair_summary_by_group["group_type"] == group_type)
            ].copy()
            rows.append(
                {
                    "view_name": view_name,
                    "group_type": group_type,
                    "n_groups": int(subset["group_id"].nunique()) if not subset.empty else 0,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _aggregate_grouped_pairs(comparisons: pd.DataFrame, group_type: str) -> pd.DataFrame:
    columns = [
        "view_name",
        "group_type",
        "group_id",
        "stimulus_left",
        "stimulus_right",
        "same_stimulus",
        "n_pairs",
        "mean_distance",
        "median_distance",
    ]
    if comparisons.empty:
        return pd.DataFrame(columns=columns)

    valid = comparisons.loc[comparisons["comparison_status"] == VALID_COMPARISON_STATUS].copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    if group_type == "pooled":
        valid["group_id"] = "pooled"
    elif group_type == "individual":
        valid = valid.loc[valid["same_individual"].astype(bool)].copy()
        valid["group_id"] = valid["individual_id_a"].astype(str)
    else:
        valid = valid.loc[valid["same_date"].astype(bool)].copy()
        valid["group_id"] = valid["date_a"].astype(str)

    if valid.empty:
        return pd.DataFrame(columns=columns)

    stimulus_a = valid["stimulus_a"].astype(str)
    stimulus_b = valid["stimulus_b"].astype(str)
    valid["stimulus_left"] = np.where(stimulus_a <= stimulus_b, stimulus_a, stimulus_b)
    valid["stimulus_right"] = np.where(stimulus_a <= stimulus_b, stimulus_b, stimulus_a)

    rows: list[dict[str, object]] = []
    for (view, group_id, stimulus_left, stimulus_right), group in valid.groupby(
        ["view_name", "group_id", "stimulus_left", "stimulus_right"],
        sort=False,
        dropna=False,
    ):
        rows.append(
            {
                "view_name": view,
                "group_type": group_type,
                "group_id": group_id,
                "stimulus_left": stimulus_left,
                "stimulus_right": stimulus_right,
                "same_stimulus": bool(stimulus_left == stimulus_right),
                "n_pairs": int(len(group)),
                "mean_distance": float(group["distance"].mean()),
                "median_distance": float(group["distance"].median()),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_stability_table(
    stability: pd.DataFrame,
    *,
    group_type: str | None = None,
    comparison_scope: str | None = None,
) -> pd.DataFrame:
    if stability.empty:
        return stability.copy()

    filtered = stability.copy()
    if group_type is not None:
        filtered = filtered.loc[filtered["group_type"] == group_type]
    if comparison_scope is not None:
        filtered = filtered.loc[filtered["comparison_scope"] == comparison_scope]
    return filtered.reset_index(drop=True)


def _pivot_symmetric_distance_matrix(group: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()

    forward = group[["stimulus_left", "stimulus_right", value_column]].rename(
        columns={"stimulus_left": "stimulus_row", "stimulus_right": "stimulus_column"}
    )
    reverse = group.loc[
        group["stimulus_left"] != group["stimulus_right"], ["stimulus_right", "stimulus_left", value_column]
    ].rename(columns={"stimulus_right": "stimulus_row", "stimulus_left": "stimulus_column"})
    matrix_long = pd.concat([forward, reverse], ignore_index=True)
    if matrix_long.empty:
        return pd.DataFrame()

    stimuli = sorted(set(matrix_long["stimulus_row"]).union(matrix_long["stimulus_column"]))
    columns = pd.Index(stimuli, dtype=object)
    matrix = matrix_long.pivot(index="stimulus_row", columns="stimulus_column", values=value_column)
    matrix = matrix.reindex(index=columns, columns=columns)
    matrix.index.name = "stimulus_row"
    matrix.columns.name = "stimulus_column"
    return matrix


def _matrix_upper_triangle_records(matrix: pd.DataFrame) -> pd.DataFrame:
    columns = ["stimulus_left", "stimulus_right", "value"]
    if matrix.empty:
        return pd.DataFrame(columns=columns)

    matrix = _align_matrix_columns_to_index(matrix)

    rows: list[dict[str, object]] = []
    labels = list(matrix.index.astype(str))
    for row_idx, stimulus_left in enumerate(labels):
        for col_idx in range(row_idx + 1, len(labels)):
            stimulus_right = labels[col_idx]
            rows.append(
                {
                    "stimulus_left": stimulus_left,
                    "stimulus_right": stimulus_right,
                    "value": matrix.iloc[row_idx, col_idx],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _shared_upper_triangle(left_matrix: pd.DataFrame, right_matrix: pd.DataFrame) -> pd.DataFrame:
    left = extract_upper_triangle(left_matrix).rename(columns={"value": "value_left"})
    right = extract_upper_triangle(right_matrix).rename(columns={"value": "value_right"})
    shared = left.merge(right, on=["stimulus_left", "stimulus_right"], how="inner")
    if shared.empty:
        return shared

    value_columns = ["value_left", "value_right"]
    shared[value_columns] = shared[value_columns].apply(pd.to_numeric, errors="coerce")
    return shared.dropna(subset=value_columns).reset_index(drop=True)


def _score_shared_triangle(shared: pd.DataFrame, method: str = "spearman") -> dict[str, object]:
    n_shared_entries = int(len(shared))
    if n_shared_entries < 2:
        return {
            "score_method": method,
            "score_status": "invalid",
            "n_shared_entries": n_shared_entries,
            "similarity": np.nan,
        }

    left_values = shared["value_left"]
    right_values = shared["value_right"]
    if method == "spearman":
        similarity = _spearman_similarity(left_values, right_values)
    else:
        raise ValueError(f"unsupported similarity method: {method}")

    score_status = "ok" if pd.notna(similarity) else "invalid"
    return {
        "score_method": method,
        "score_status": score_status,
        "n_shared_entries": n_shared_entries,
        "similarity": similarity,
    }


def _score_group_matrix_sets(grouped_matrices: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison_scope",
        "view_name",
        "reference_view_name",
        "group_type",
        "group_id",
        "reference_group_id",
        "score_method",
        "score_status",
        "n_shared_entries",
        "similarity",
    ]
    if grouped_matrices.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    rows.extend(_score_within_group_type(grouped_matrices))
    rows.extend(_score_pooled_vs_group(grouped_matrices))
    rows.extend(_score_pooled_cross_view(grouped_matrices))
    return pd.DataFrame(rows, columns=columns)


def _score_within_group_type(grouped_matrices: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eligible = grouped_matrices.loc[grouped_matrices["group_type"].isin(["individual", "date"])].copy()
    for (view_name, group_type), subset in eligible.groupby(["view_name", "group_type"], sort=False, dropna=False):
        records = subset.to_dict("records")
        for left_idx, left in enumerate(records):
            for right in records[left_idx + 1 :]:
                score = score_rdm_similarity(left["matrix_frame"], right["matrix_frame"])
                rows.append(
                    {
                        "comparison_scope": "within_group_type",
                        "view_name": view_name,
                        "reference_view_name": view_name,
                        "group_type": group_type,
                        "group_id": left["group_id"],
                        "reference_group_id": right["group_id"],
                        **score,
                    }
                )
    return rows


def _score_pooled_vs_group(grouped_matrices: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled = grouped_matrices.loc[grouped_matrices["group_type"] == "pooled"].copy()
    for _, group in grouped_matrices.loc[grouped_matrices["group_type"].isin(["individual", "date"])].iterrows():
        pooled_match = pooled.loc[pooled["view_name"] == group["view_name"]]
        if pooled_match.empty:
            continue
        if len(pooled_match) != 1:
            raise ValueError(
                f"expected exactly one pooled matrix for view_name {group['view_name']!r}; "
                f"found {len(pooled_match)}"
            )
        pooled_matrix = pooled_match.iloc[0]
        score = score_rdm_similarity(group["matrix_frame"], pooled_matrix["matrix_frame"])
        rows.append(
            {
                "comparison_scope": "pooled_vs_group",
                "view_name": group["view_name"],
                "reference_view_name": pooled_matrix["view_name"],
                "group_type": group["group_type"],
                "group_id": group["group_id"],
                "reference_group_id": pooled_matrix["group_id"],
                **score,
            }
        )
    return rows


def _score_pooled_cross_view(grouped_matrices: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled = grouped_matrices.loc[grouped_matrices["group_type"] == "pooled"].copy()
    records = pooled.to_dict("records")
    for left_idx, left in enumerate(records):
        for right in records[left_idx + 1 :]:
            score = score_rdm_similarity(left["matrix_frame"], right["matrix_frame"])
            rows.append(
                {
                    "comparison_scope": "pooled_cross_view",
                    "view_name": left["view_name"],
                    "reference_view_name": right["view_name"],
                    "group_type": "pooled",
                    "group_id": left["group_id"],
                    "reference_group_id": right["group_id"],
                    **score,
                }
            )
    return rows


def _spearman_similarity(left_values: pd.Series, right_values: pd.Series) -> float:
    left_ranks = pd.Series(left_values, copy=False).rank(method="average")
    right_ranks = pd.Series(right_values, copy=False).rank(method="average")
    if left_ranks.nunique(dropna=True) < 2 or right_ranks.nunique(dropna=True) < 2:
        return np.nan
    return float(left_ranks.corr(right_ranks, method="pearson"))


def _align_matrix_columns_to_index(matrix: pd.DataFrame) -> pd.DataFrame:
    index_labels = pd.Index(matrix.index.astype(str))
    column_labels = pd.Index(matrix.columns.astype(str))
    if not index_labels.is_unique:
        duplicates = index_labels[index_labels.duplicated()].unique().tolist()
        raise ValueError(f"matrix index contains duplicate stimulus labels: {duplicates}")
    if not column_labels.is_unique:
        duplicates = column_labels[column_labels.duplicated()].unique().tolist()
        raise ValueError(f"matrix columns contain duplicate stimulus labels: {duplicates}")
    if set(index_labels) != set(column_labels):
        raise ValueError(
            "matrix columns must match stimulus_row labels; "
            f"index={index_labels.tolist()} columns={column_labels.tolist()}"
        )
    aligned = matrix.copy()
    aligned.index = index_labels
    aligned.columns = column_labels
    return aligned.reindex(columns=index_labels)
