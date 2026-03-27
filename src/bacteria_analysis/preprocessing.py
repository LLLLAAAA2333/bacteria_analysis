"""Structural preprocessing helpers for trial validation."""

from __future__ import annotations

import pandas as pd

from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS

TRACE_GROUP_COLUMNS = ("worm_key", "segment_index", "date", "neuron")


def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a normalized trial_id column."""

    out = df.copy()
    date = pd.to_datetime(out["date"], errors="raise").dt.strftime("%Y%m%d")
    out["trial_id"] = date + "__" + out["worm_key"].astype(str) + "__" + out["segment_index"].astype(str)
    return out


def annotate_trace_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add trace-level QC flags and counts."""

    out = df.copy()

    def _trace_quality_stats(group: pd.DataFrame) -> pd.Series:
        baseline_values = group.loc[group["time_point"].isin(BASELINE_TIMEPOINTS), "delta_F_over_F0"]
        valid_points = group["delta_F_over_F0"].notna().sum()
        return pd.Series(
            {
                "n_valid_points": valid_points,
                "has_any_nan_trace": group["delta_F_over_F0"].isna().any(),
                "is_all_nan_trace": valid_points == 0,
                "n_valid_baseline_points": baseline_values.notna().sum(),
            }
        )

    stats = (
        out.groupby(list(TRACE_GROUP_COLUMNS), sort=False, dropna=False)
        .apply(_trace_quality_stats, include_groups=False)
        .reset_index()
    )
    out = out.merge(stats, on=list(TRACE_GROUP_COLUMNS), how="left")
    out["n_valid_points"] = out["n_valid_points"].astype("int64")
    out["n_valid_baseline_points"] = out["n_valid_baseline_points"].astype("int64")
    out["has_any_nan_trace"] = out["has_any_nan_trace"].fillna(False).astype(bool)
    out["is_all_nan_trace"] = out["is_all_nan_trace"].fillna(False).astype(bool)
    return out


def filter_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Drop traces that are entirely NaN."""

    out = df if "is_all_nan_trace" in df.columns else annotate_trace_quality(df)
    return out.loc[~out["is_all_nan_trace"]].copy()


def center_by_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Center traces by their baseline mean when the full baseline window is present."""

    out = df if "n_valid_points" in df.columns else annotate_trace_quality(df)

    def _baseline_stats(group: pd.DataFrame) -> pd.Series:
        baseline_values = group.loc[group["time_point"].isin(BASELINE_TIMEPOINTS), "delta_F_over_F0"]
        baseline_valid = baseline_values.notna().sum() == len(BASELINE_TIMEPOINTS)
        baseline_mean = baseline_values.mean() if baseline_valid else float("nan")
        return pd.Series(
            {
                "baseline_mean": baseline_mean,
                "baseline_valid": baseline_valid,
            }
        )

    stats = (
        out.groupby(list(TRACE_GROUP_COLUMNS), sort=False, dropna=False)
        .apply(_baseline_stats, include_groups=False)
        .reset_index()
    )
    out = out.merge(stats, on=list(TRACE_GROUP_COLUMNS), how="left")
    out["baseline_valid"] = out["baseline_valid"].fillna(False).astype(bool)
    out["dff_baseline_centered"] = out["delta_F_over_F0"] - out["baseline_mean"]
    return out


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate the structural integrity of the preprocessing input frame."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

    unique_timepoints = tuple(sorted(pd.unique(df["time_point"])))
    if unique_timepoints != EXPECTED_TIMEPOINTS:
        raise ValueError("expected 45 unique time_point values covering 0..44")

    validated = add_trial_id(df)

    if validated.groupby("trial_id")["stimulus"].nunique().ne(1).any():
        raise ValueError("each trial must map to exactly one stimulus")

    group_sizes = validated.groupby(["trial_id", "stimulus", "neuron"], sort=False)
    for _, group in group_sizes:
        if len(group) != len(EXPECTED_TIMEPOINTS):
            raise ValueError("each trial_id x stimulus x neuron must have 45 rows")
        if group["time_point"].nunique() != len(EXPECTED_TIMEPOINTS):
            raise ValueError("each trial_id x stimulus x neuron must have 45 unique time_point values")
