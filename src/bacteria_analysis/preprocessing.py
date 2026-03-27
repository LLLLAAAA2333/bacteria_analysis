"""Structural preprocessing helpers for trial validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER, REQUIRED_COLUMNS

TRACE_GROUP_COLUMNS = ("trial_id", "stimulus", "neuron")


def _ensure_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trial_id present."""

    return df if "trial_id" in df.columns else add_trial_id(df)


def add_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a normalized trial_id column."""

    out = df.copy()
    date = pd.to_datetime(out["date"], errors="raise").dt.strftime("%Y%m%d")
    out["trial_id"] = date + "__" + out["worm_key"].astype(str) + "__" + out["segment_index"].astype(str)
    return out


def annotate_trace_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add trace-level QC flags and counts."""

    out = _ensure_trial_id(df).copy()

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
    removed_counts = (
        out.groupby(list(TRACE_GROUP_COLUMNS), sort=False, dropna=False)["is_all_nan_trace"]
        .first()
        .groupby(level=0, sort=False, dropna=False)
        .sum()
        .astype("int64")
        if "trial_id" in out.columns
        else pd.Series(dtype="int64")
    )
    kept = out.loc[~out["is_all_nan_trace"]].copy()
    kept["n_all_nan_traces_removed"] = kept["trial_id"].map(removed_counts).fillna(0).astype("int64")
    return kept


def center_by_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Center traces by the mean of available non-NaN baseline points."""

    out = _ensure_trial_id(df)
    out = out if "n_valid_points" in out.columns else annotate_trace_quality(out)

    def _baseline_stats(group: pd.DataFrame) -> pd.Series:
        baseline_values = group.loc[group["time_point"].isin(BASELINE_TIMEPOINTS), "delta_F_over_F0"]
        valid_baseline_values = baseline_values.dropna()
        baseline_valid = not valid_baseline_values.empty
        baseline_mean = valid_baseline_values.mean() if baseline_valid else float("nan")
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
    out.loc[~out["baseline_valid"], "dff_baseline_centered"] = float("nan")
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


def build_trial_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Build one metadata row per trial in deterministic trial order."""

    out = _ensure_trial_id(df).copy()
    if "has_any_nan_trace" not in out.columns or "is_all_nan_trace" not in out.columns:
        out = annotate_trace_quality(out)

    rows = []
    trial_columns = [
        "trial_id",
        "date",
        "worm_key",
        "segment_index",
        "stimulus",
        "stim_name",
        "stim_color",
        "n_observed_neurons",
        "n_missing_neurons",
        "n_all_nan_traces_removed",
        "has_partial_nan_trace",
    ]

    for trial_id, trial in out.groupby("trial_id", sort=False, dropna=False):
        first_row = trial.iloc[0]
        trace_flags = trial.drop_duplicates(["stimulus", "neuron"], keep="first")
        surviving_traces = trace_flags.loc[~trace_flags["is_all_nan_trace"].fillna(False)]
        observed_neurons = {neuron for neuron in surviving_traces["neuron"].dropna().astype(str) if neuron in NEURON_ORDER}
        if "n_all_nan_traces_removed" in trial.columns:
            n_all_nan_traces_removed = int(trial["n_all_nan_traces_removed"].iloc[0])
        else:
            n_all_nan_traces_removed = int(trace_flags["is_all_nan_trace"].fillna(False).sum())
        has_partial_nan_trace = bool((trace_flags["has_any_nan_trace"].fillna(False) & ~trace_flags["is_all_nan_trace"].fillna(False)).any())

        rows.append(
            {
                "trial_id": trial_id,
                "date": first_row["date"],
                "worm_key": first_row["worm_key"],
                "segment_index": first_row["segment_index"],
                "stimulus": first_row["stimulus"],
                "stim_name": first_row["stim_name"],
                "stim_color": first_row["stim_color"],
                "n_observed_neurons": len(observed_neurons),
                "n_missing_neurons": len(NEURON_ORDER) - len(observed_neurons),
                "n_all_nan_traces_removed": n_all_nan_traces_removed,
                "has_partial_nan_trace": has_partial_nan_trace,
            }
        )

    metadata = pd.DataFrame(rows, columns=trial_columns)
    metadata["__trial_sort_date"] = pd.to_datetime(metadata["date"], errors="raise")
    metadata = metadata.sort_values(
        ["__trial_sort_date", "worm_key", "segment_index", "trial_id"],
        kind="stable",
    ).drop(columns=["__trial_sort_date"])
    return metadata.reset_index(drop=True)


def build_trial_wide_table(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Build a trial-major wide table using baseline-centered values."""

    if "dff_baseline_centered" not in df.columns:
        raise ValueError("build_trial_wide_table requires dff_baseline_centered values")
    if "trial_id" not in metadata.columns:
        raise ValueError("metadata must include trial_id")

    source = _ensure_trial_id(df).copy()
    wide = metadata.copy().reset_index(drop=True)
    trial_positions = {trial_id: position for position, trial_id in enumerate(wide["trial_id"])}
    feature_columns = [f"{neuron}__t{time_point:02d}" for neuron in NEURON_ORDER for time_point in EXPECTED_TIMEPOINTS]
    feature_frame = pd.DataFrame(np.nan, index=wide.index, columns=feature_columns)

    for row in source.itertuples(index=False):
        trial_position = trial_positions.get(row.trial_id)
        if trial_position is None:
            continue
        if row.neuron not in NEURON_ORDER:
            continue
        if row.time_point not in EXPECTED_TIMEPOINTS:
            continue
        feature_frame.at[trial_position, f"{row.neuron}__t{int(row.time_point):02d}"] = row.dff_baseline_centered

    return pd.concat([wide, feature_frame], axis=1)


def build_trial_tensor(df: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
    """Build a trial-major tensor ordered by metadata, neurons, and time points."""

    if "dff_baseline_centered" not in df.columns:
        raise ValueError("build_trial_tensor requires dff_baseline_centered values")
    if "trial_id" not in metadata.columns:
        raise ValueError("metadata must include trial_id")

    source = _ensure_trial_id(df).copy()
    tensor = np.full((len(metadata), len(NEURON_ORDER), len(EXPECTED_TIMEPOINTS)), np.nan, dtype=float)
    trial_positions = {trial_id: position for position, trial_id in enumerate(metadata["trial_id"])}
    neuron_positions = {neuron: position for position, neuron in enumerate(NEURON_ORDER)}
    time_positions = {time_point: position for position, time_point in enumerate(EXPECTED_TIMEPOINTS)}

    for row in source.itertuples(index=False):
        trial_position = trial_positions.get(row.trial_id)
        neuron_position = neuron_positions.get(row.neuron)
        time_position = time_positions.get(row.time_point)
        if trial_position is None or neuron_position is None or time_position is None:
            continue
        tensor[trial_position, neuron_position, time_position] = row.dff_baseline_centered

    return tensor


def build_qc_report(raw_df: pd.DataFrame, processed_df: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Build a data-only QC summary for JSON and markdown rendering."""

    raw = _ensure_trial_id(raw_df).copy()
    if "has_any_nan_trace" not in raw.columns or "is_all_nan_trace" not in raw.columns:
        raw = annotate_trace_quality(raw)

    processed = _ensure_trial_id(processed_df).copy()
    if "has_any_nan_trace" not in processed.columns or "is_all_nan_trace" not in processed.columns:
        processed = annotate_trace_quality(processed)

    trace_flags = (
        raw.drop_duplicates(list(TRACE_GROUP_COLUMNS), keep="first")[
            ["trial_id", "stimulus", "neuron", "has_any_nan_trace", "is_all_nan_trace"]
        ]
        .copy()
    )

    fully_nan_traces_removed = int(trace_flags["is_all_nan_trace"].sum())
    partially_nan_traces_retained = int(
        (trace_flags["has_any_nan_trace"] & ~trace_flags["is_all_nan_trace"]).sum()
    )

    neuron_coverage = (
        metadata.groupby("n_observed_neurons", sort=True, dropna=False)["trial_id"]
        .nunique()
        .reset_index(name="n_trials")
        .sort_values("n_observed_neurons", kind="stable")
    )
    neuron_coverage_distribution = [
        {
            "n_observed_neurons": int(row.n_observed_neurons),
            "n_trials": int(row.n_trials),
        }
        for row in neuron_coverage.itertuples(index=False)
    ]

    trials_per_stimulus = (
        metadata.groupby(["stimulus", "stim_name"], sort=True, dropna=False)["trial_id"]
        .nunique()
        .reset_index(name="n_trials")
        .sort_values(["stimulus", "stim_name"], kind="stable")
    )
    trials_per_stimulus_summary = [
        {
            "stimulus": str(row.stimulus),
            "stim_name": str(row.stim_name),
            "n_trials": int(row.n_trials),
        }
        for row in trials_per_stimulus.itertuples(index=False)
    ]

    return {
        "input_rows": int(len(raw_df)),
        "output_rows": int(len(processed_df)),
        "n_unique_trials": int(metadata["trial_id"].nunique()),
        "n_unique_stimuli": int(metadata["stimulus"].nunique()),
        "n_unique_neurons": int(processed["neuron"].nunique()),
        "n_fully_nan_traces_removed": fully_nan_traces_removed,
        "n_partially_nan_traces_retained": partially_nan_traces_retained,
        "neuron_coverage_distribution": neuron_coverage_distribution,
        "trials_per_stimulus_summary": trials_per_stimulus_summary,
    }
