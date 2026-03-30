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
