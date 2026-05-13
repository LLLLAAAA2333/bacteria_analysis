"""RDM builders from shared neural and chemical feature tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bacteria_analysis._vector_distance import compute_vector_distance
from bacteria_analysis.features.chemical import (
    build_chemical_class_feature_matrices,
    build_chemical_feature_matrix,
)
from bacteria_analysis.features.neural import (
    VIEW_TIMEPOINTS,
    build_stimulus_prototypes,
    build_trial_feature_matrix,
    neural_feature_columns,
)

SUPPORTED_DISTANCES = ("correlation", "euclidean")


@dataclass(frozen=True)
class RdmResult:
    matrix: pd.DataFrame
    metadata: dict[str, object]


def build_neural_rdm(
    dataset,
    *,
    view: str,
    aggregation: str = "median",
    merge_lr: bool = True,
    distance: str = "correlation",
) -> RdmResult:
    """Build a stimulus-level neural RDM from shared neural features."""

    _validate_distance(distance, label="neural RDM")
    features = build_trial_feature_matrix(dataset, view=view, merge_lr=merge_lr)
    prototypes = build_stimulus_prototypes(features, aggregation=aggregation)
    matrix = _build_rdm_matrix(
        prototypes,
        label_column="stimulus",
        feature_columns=neural_feature_columns(prototypes),
        distance=distance,
        euclidean_min_valid=2,
    )
    metadata = {
        "view": view,
        "aggregation": aggregation,
        "merge_lr": merge_lr,
        "distance": distance,
        "n_trials": int(len(features)),
        "n_stimuli": int(len(prototypes)),
        "feature_count": int(len(neural_feature_columns(prototypes))),
        "timepoints": VIEW_TIMEPOINTS[view],
    }
    return RdmResult(matrix=matrix, metadata=metadata)


def build_chemical_rdm(
    dataset,
    *,
    qc_threshold: float = 0.2,
    transform: str = "log2",
    distance: str = "euclidean",
) -> RdmResult:
    """Build a stimulus-level chemical RDM from shared chemical features."""

    _validate_distance(distance, label="chemical RDM")
    features = build_chemical_feature_matrix(
        dataset,
        qc_threshold=qc_threshold,
        transform=transform,
    )
    return RdmResult(
        matrix=_build_rdm_matrix(
            features.matrix.reset_index(),
            label_column="stimulus",
            feature_columns=features.matrix.columns.astype(str).tolist(),
            distance=distance,
            euclidean_min_valid=1,
        ),
        metadata={**features.metadata, "distance": distance},
    )


def build_chemical_class_rdms(
    dataset,
    *,
    taxonomy_level: str = "Class",
    qc_threshold: float = 0.2,
    min_features: int = 3,
    transform: str = "log2",
    distance: str = "euclidean",
) -> dict[str, RdmResult]:
    """Build chemical RDMs for each retained taxonomy category."""

    _validate_distance(distance, label="chemical class RDM")
    feature_results = build_chemical_class_feature_matrices(
        dataset,
        taxonomy_level=taxonomy_level,
        qc_threshold=qc_threshold,
        min_features=min_features,
        transform=transform,
    )
    return {
        category: RdmResult(
            matrix=_build_rdm_matrix(
                result.matrix.reset_index(),
                label_column="stimulus",
                feature_columns=result.matrix.columns.astype(str).tolist(),
                distance=distance,
                euclidean_min_valid=1,
            ),
            metadata={**result.metadata, "distance": distance},
        )
        for category, result in feature_results.items()
    }


def _build_rdm_matrix(
    values: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    distance: str,
    euclidean_min_valid: int,
) -> pd.DataFrame:
    labels = values[label_column].astype(str).tolist()
    array = values.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
    distances = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)

    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            pair_distance = _pair_distance(
                array[left_index],
                array[right_index],
                distance=distance,
                euclidean_min_valid=euclidean_min_valid,
            )
            distances[left_index, right_index] = pair_distance
            distances[right_index, left_index] = pair_distance

    return pd.DataFrame(distances, index=labels, columns=labels)


def _pair_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    distance: str,
    euclidean_min_valid: int,
) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left_values = left[valid]
    right_values = right[valid]
    if distance == "euclidean":
        if left_values.size < euclidean_min_valid:
            return float("nan")
        return float(np.linalg.norm(left_values - right_values))
    pair_distance, status = compute_vector_distance(left_values, right_values, metric=distance)
    if status == "ok" and np.isfinite(pair_distance):
        return pair_distance
    return float("nan")


def _validate_distance(distance: str, *, label: str) -> None:
    if distance not in SUPPORTED_DISTANCES:
        raise ValueError(f"unsupported {label} distance {distance!r}")


__all__ = [
    "RdmResult",
    "build_chemical_class_rdms",
    "build_chemical_rdm",
    "build_neural_rdm",
]
