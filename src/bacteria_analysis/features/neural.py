"""Neural feature construction from raw trial rows."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from bacteria_analysis.constants import NEURON_ORDER
from bacteria_analysis.preprocessing import (
    add_trial_id,
    annotate_trace_quality,
    build_trial_metadata,
    build_trial_tensor,
    center_by_baseline,
    filter_traces,
    validate_input_dataframe,
)
VIEW_TIMEPOINTS: dict[str, tuple[int, ...]] = {
    "response_window": tuple(range(5, 25)),
    "full_trajectory": tuple(range(45)),
}

MERGED_NEURON_ORDER = (
    "ADF",
    "ADL",
    "ASEL",
    "ASER",
    "ASG",
    "ASH",
    "ASI",
    "ASJ",
    "ASK",
    "AWA",
    "AWB",
    "AWCOFF",
    "AWCON",
)

LR_MERGE_PAIRS = {
    "ADF": ("ADFL", "ADFR"),
    "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"),
    "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"),
    "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"),
    "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}

FEATURE_ID_COLUMNS = ("trial_id", "stimulus", "stim_name", "date")
SUPPORTED_AGGREGATIONS = ("median", "mean")
FEATURE_COLUMN_PATTERN = re.compile(r"^[A-Za-z0-9]+__t\d{2}$")


def build_trial_feature_matrix(dataset_or_frame, *, view: str, merge_lr: bool = True) -> pd.DataFrame:
    """Build one baseline-centered neural feature row per trial from raw neural rows."""

    raw = _extract_neural_frame(dataset_or_frame)
    if view not in VIEW_TIMEPOINTS:
        raise ValueError(f"unknown neural feature view {view!r}")

    validated = add_trial_id(raw)
    validate_input_dataframe(validated)
    annotated = annotate_trace_quality(validated)
    filtered = filter_traces(annotated)
    centered = center_by_baseline(filtered)
    metadata = build_trial_metadata(centered)
    tensor = build_trial_tensor(centered, metadata)

    timepoints = VIEW_TIMEPOINTS[view]
    view_tensor = tensor[:, :, list(timepoints)]
    neuron_labels, feature_tensor = _select_neuron_features(view_tensor, merge_lr=merge_lr)

    feature_columns = [
        f"{neuron}__t{time_point:02d}"
        for neuron in neuron_labels
        for time_point in timepoints
    ]
    feature_frame = pd.DataFrame(
        feature_tensor.reshape(feature_tensor.shape[0], -1),
        columns=feature_columns,
    )

    id_columns = [column for column in FEATURE_ID_COLUMNS if column in metadata.columns]
    return pd.concat([metadata.loc[:, id_columns].reset_index(drop=True), feature_frame], axis=1)


def build_stimulus_prototypes(features: pd.DataFrame, *, aggregation: str = "median") -> pd.DataFrame:
    """Aggregate trial feature rows into one prototype vector per stimulus."""

    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"unknown stimulus aggregation {aggregation!r}")
    if "stimulus" not in features.columns:
        raise ValueError("features must include a stimulus column")

    feature_columns = neural_feature_columns(features)
    if not feature_columns:
        raise ValueError("features must include at least one neural feature column")

    rows: list[dict[str, object]] = []
    for stimulus, group in features.groupby("stimulus", sort=True, dropna=False):
        values = group.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if aggregation == "median":
                prototype = np.nanmedian(values, axis=0)
            else:
                prototype = np.nanmean(values, axis=0)
        row: dict[str, object] = {"stimulus": str(stimulus), "n_trials": int(len(group))}
        if "stim_name" in group.columns:
            row["stim_name"] = str(group["stim_name"].iloc[0])
        row.update(dict(zip(feature_columns, prototype, strict=True)))
        rows.append(row)

    id_columns = ["stimulus", "stim_name", "n_trials"]
    ordered_columns = [column for column in id_columns if any(column in row for row in rows)] + feature_columns
    return pd.DataFrame.from_records(rows, columns=ordered_columns)


def _extract_neural_frame(dataset_or_frame) -> pd.DataFrame:
    if isinstance(dataset_or_frame, pd.DataFrame):
        return dataset_or_frame.copy()
    if hasattr(dataset_or_frame, "neural"):
        neural = getattr(dataset_or_frame, "neural")
        if isinstance(neural, pd.DataFrame):
            return neural.copy()
    raise ValueError("expected a raw neural DataFrame or an object with a neural DataFrame")


def _select_neuron_features(tensor: np.ndarray, *, merge_lr: bool) -> tuple[tuple[str, ...], np.ndarray]:
    if not merge_lr:
        return NEURON_ORDER, tensor

    neuron_positions = {neuron: index for index, neuron in enumerate(NEURON_ORDER)}
    merged = np.full((tensor.shape[0], len(MERGED_NEURON_ORDER), tensor.shape[2]), np.nan, dtype=float)
    for output_index, label in enumerate(MERGED_NEURON_ORDER):
        if label in LR_MERGE_PAIRS:
            left, right = LR_MERGE_PAIRS[label]
            pair = tensor[:, [neuron_positions[left], neuron_positions[right]], :]
            counts = np.isfinite(pair).sum(axis=1)
            totals = np.nansum(pair, axis=1)
            np.divide(totals, counts, out=merged[:, output_index, :], where=counts > 0)
        else:
            merged[:, output_index, :] = tensor[:, neuron_positions[label], :]
    return MERGED_NEURON_ORDER, merged


def neural_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return neural feature columns from a trial or stimulus feature table."""

    return [column for column in frame.columns if FEATURE_COLUMN_PATTERN.match(str(column))]


__all__ = [
    "build_stimulus_prototypes",
    "build_trial_feature_matrix",
    "neural_feature_columns",
]
