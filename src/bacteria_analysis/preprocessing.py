"""Structural preprocessing helpers for trial validation."""

from __future__ import annotations

import pandas as pd

from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS


def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a normalized trial_id column."""

    out = df.copy()
    date = pd.to_datetime(out["date"], errors="raise").dt.strftime("%Y%m%d")
    out["trial_id"] = date + "__" + out["worm_key"].astype(str) + "__" + out["segment_index"].astype(str)
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

