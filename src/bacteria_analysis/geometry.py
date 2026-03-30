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
