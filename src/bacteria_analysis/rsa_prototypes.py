"""Prototype helpers for biochemical RSA context outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from bacteria_analysis.reliability import (
    DEFAULT_DISTANCE_METRIC,
    MIN_VALID_VALUES,
    TrialView,
    VIEW_WINDOWS,
    build_trial_views,
    compute_vector_distance,
    load_reliability_inputs,
)

PROTOTYPE_AGGREGATIONS: tuple[str, str] = ("mean", "median")


@dataclass(frozen=True)
class PrototypeContextInputs:
    """Preprocessed inputs used to build prototype-level context outputs."""

    metadata: pd.DataFrame
    views: dict[str, TrialView]


def load_prototype_context_inputs(
    preprocess_root: str | Path,
    view_names: tuple[str, ...] | list[str],
) -> PrototypeContextInputs:
    """Load preprocessing outputs required for prototype-level context outputs."""

    root = Path(preprocess_root)
    wide_path = root / "trial_level" / "trial_wide_baseline_centered.parquet"
    inputs = load_reliability_inputs(
        metadata_path=root / "trial_level" / "trial_metadata.parquet",
        tensor_path=root / "trial_level" / "trial_tensor_baseline_centered.npz",
        wide_path=wide_path if wide_path.exists() else None,
    )

    selected_views = tuple(view_names)
    missing_views = [view_name for view_name in selected_views if view_name not in VIEW_WINDOWS]
    if missing_views:
        raise ValueError(f"unknown prototype view_names: {', '.join(missing_views)}")

    view_windows = {view_name: VIEW_WINDOWS[view_name] for view_name in selected_views}
    views = build_trial_views(inputs.metadata, inputs.tensor, view_windows=view_windows)
    return PrototypeContextInputs(metadata=inputs.metadata, views=views)


def build_grouped_prototypes(
    view: TrialView,
    group_columns: tuple[str, ...],
    *,
    aggregation: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build grouped prototype vectors and support QC for a grouped trial view."""

    metadata = view.metadata.reset_index(drop=True)
    values = np.asarray(view.values, dtype=float)
    if values.ndim < 2:
        raise ValueError("prototype construction requires at least 2D trial values")
    if aggregation not in PROTOTYPE_AGGREGATIONS:
        raise ValueError(f"unknown prototype aggregation {aggregation!r}")

    records: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    grouped = metadata.groupby(list(group_columns), sort=True, dropna=False)

    for group_key, group_metadata in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        trial_indices = group_metadata.index.to_numpy()
        grouped_values = values[trial_indices].reshape(len(trial_indices), -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prototype_vector = _aggregate_grouped_values(grouped_values, aggregation=aggregation)

        feature_names = [f"f{index:03d}" for index in range(prototype_vector.shape[0])]
        record = {column: value for column, value in zip(group_columns, group_key, strict=True)}
        record.update(dict(zip(feature_names, prototype_vector, strict=True)))
        records.append(record)

        finite_counts = np.isfinite(grouped_values).sum(axis=0)
        n_total_features = int(grouped_values.shape[1])
        n_supported_features = int(np.count_nonzero(finite_counts > 0))
        support_row = {
            **{column: value for column, value in zip(group_columns, group_key, strict=True)},
            "n_trials": int(len(trial_indices)),
            "n_total_features": n_total_features,
            "n_supported_features": n_supported_features,
            "n_all_nan_features": int(n_total_features - n_supported_features),
        }
        if "date" not in group_columns and "date" in metadata.columns:
            support_row["n_dates_contributed"] = int(group_metadata["date"].astype(str).nunique())
        support_rows.append(support_row)

    prototypes = pd.DataFrame.from_records(records)
    support = pd.DataFrame.from_records(support_rows)
    return prototypes, support


def _aggregate_grouped_values(grouped_values: np.ndarray, *, aggregation: str) -> np.ndarray:
    if aggregation == "mean":
        return np.nanmean(grouped_values, axis=0)
    if aggregation == "median":
        return np.nanmedian(grouped_values, axis=0)
    raise ValueError(f"unknown prototype aggregation {aggregation!r}")


def build_prototype_rdm(prototypes: pd.DataFrame, *, id_columns: tuple[str, ...]) -> pd.DataFrame:
    """Build an overlap-aware correlation-distance RDM from prototype vectors."""

    if not id_columns:
        raise ValueError("id_columns must include at least one column")

    prototype_frame = prototypes.reset_index(drop=True).copy()
    feature_columns = _select_prototype_feature_columns(prototype_frame, id_columns)
    if not feature_columns:
        raise ValueError("prototype RDMs require at least one feature column")

    row_labels = _build_prototype_labels(prototype_frame, id_columns)
    if pd.Index(row_labels).duplicated().any():
        raise ValueError("prototype id labels must be unique")

    values = prototype_frame[feature_columns].to_numpy(dtype=float, copy=False)
    n_rows = values.shape[0]
    distance = np.full((n_rows, n_rows), np.nan, dtype=float)
    np.fill_diagonal(distance, 0.0)

    for left_index in range(n_rows):
        left_values = values[left_index]
        for right_index in range(left_index + 1, n_rows):
            right_values = values[right_index]
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            if int(valid.sum()) < MIN_VALID_VALUES:
                continue

            pair_distance, status = compute_vector_distance(
                left_values[valid],
                right_values[valid],
                metric=DEFAULT_DISTANCE_METRIC,
            )
            if status != "ok" or not np.isfinite(pair_distance):
                continue

            distance[left_index, right_index] = pair_distance
            distance[right_index, left_index] = pair_distance

    frame = pd.DataFrame(distance, index=row_labels, columns=row_labels)
    frame.insert(0, "stimulus_row", frame.index.astype(str))
    return frame.reset_index(drop=True)


def _build_prototype_labels(frame: pd.DataFrame, id_columns: tuple[str, ...]) -> list[str]:
    if len(id_columns) == 1:
        return frame[id_columns[0]].astype(str).tolist()

    return frame.loc[:, list(id_columns)].astype(str).agg("__".join, axis=1).tolist()


def _select_prototype_feature_columns(frame: pd.DataFrame, id_columns: tuple[str, ...]) -> list[str]:
    excluded = set(id_columns)
    feature_columns = [column for column in frame.columns if column not in excluded and _is_prototype_feature_column(column)]
    return sorted(feature_columns, key=_prototype_feature_column_key)


def _is_prototype_feature_column(column: object) -> bool:
    return bool(re.fullmatch(r"f\d{3}", str(column)))


def _prototype_feature_column_key(column: str) -> int:
    return int(str(column)[1:])


PrototypeSupplementInputs = PrototypeContextInputs
load_prototype_supplement_inputs = load_prototype_context_inputs
