"""Stage 2 geometry aggregation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.reliability import VALID_COMPARISON_STATUS

GROUP_TYPES = ("pooled", "individual", "date")


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
