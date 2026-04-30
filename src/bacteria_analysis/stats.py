"""Shared statistics helpers for RSA workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.rdm import align_square_rdms, spearman_similarity


def empirical_p_value(observed: float, null_values: np.ndarray, side: str = "greater") -> float:
    """Return a permutation-style empirical p-value with +1 correction."""

    finite_null = np.asarray(null_values, dtype=float)
    finite_null = finite_null[np.isfinite(finite_null)]
    if finite_null.size == 0 or not np.isfinite(observed):
        return float("nan")

    if side == "greater":
        count = np.sum(finite_null >= observed)
    elif side == "less":
        count = np.sum(finite_null <= observed)
    elif side == "two-sided":
        count = np.sum(np.abs(finite_null) >= abs(observed))
    else:
        raise ValueError("side must be one of 'greater', 'less', or 'two-sided'")
    return float((count + 1) / (finite_null.size + 1))


def label_shuffle_null(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Return RSA similarities after shuffling chemical labels."""

    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    aligned_neural, aligned_chemical = align_square_rdms(neural, chemical)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        permuted = _permute_square_labels(aligned_chemical, rng.permutation(aligned_chemical.index.to_numpy()))
        null[iteration] = _rdm_similarity(aligned_neural, permuted)
    return null


def date_preserving_label_shuffle_null(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    date_map,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Return RSA null values after permuting labels within date groups."""

    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")
    aligned_neural, aligned_chemical = align_square_rdms(neural, chemical)
    aligned_dates = _aligned_date_series(date_map, aligned_neural.index)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        permuted_labels = aligned_chemical.index.to_numpy().copy()
        for date_value in aligned_dates.drop_duplicates().tolist():
            positions = np.flatnonzero(aligned_dates.to_numpy() == date_value)
            permuted_labels[positions] = rng.permutation(permuted_labels[positions])
        permuted = _permute_square_labels(aligned_chemical, permuted_labels)
        null[iteration] = _rdm_similarity(aligned_neural, permuted)
    return null


def stimulus_subset_rsa(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    subset_count: int,
    subset_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Sample stimulus subsets and return RSA similarity per subset."""

    if subset_count < 0:
        raise ValueError("subset_count must be non-negative")
    if not 0 < subset_fraction <= 1:
        raise ValueError("subset_fraction must be greater than 0 and at most 1")

    aligned_neural, aligned_chemical = align_square_rdms(neural, chemical)
    labels = np.asarray(aligned_neural.index.tolist(), dtype=object)
    if labels.size < 2:
        raise ValueError("at least 2 shared labels are required for stimulus subset RSA")
    subset_size = int(np.floor(labels.size * subset_fraction))
    subset_size = max(2, min(labels.size, subset_size))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for iteration in range(subset_count):
        subset = rng.choice(labels, size=subset_size, replace=False)
        subset_labels = [label for label in labels if label in set(subset)]
        rows.append(
            {
                "iteration": iteration,
                "n_stimuli": len(subset_labels),
                "rsa_similarity": _rdm_similarity(
                    aligned_neural.loc[subset_labels, subset_labels],
                    aligned_chemical.loc[subset_labels, subset_labels],
                ),
            }
        )
    return pd.DataFrame(rows, columns=["iteration", "n_stimuli", "rsa_similarity"])


def _rdm_similarity(left: pd.DataFrame, right: pd.DataFrame) -> float:
    rows, cols = np.triu_indices(len(left), k=1)
    return spearman_similarity(left.to_numpy()[rows, cols], right.to_numpy()[rows, cols])


def _permute_square_labels(matrix: pd.DataFrame, permuted_labels: np.ndarray) -> pd.DataFrame:
    permuted = matrix.loc[permuted_labels, permuted_labels].copy()
    permuted.index = matrix.index
    permuted.columns = matrix.columns
    return permuted


def _aligned_date_series(date_map, labels: pd.Index) -> pd.Series:
    if isinstance(date_map, pd.Series):
        dates = date_map.copy()
    else:
        dates = pd.Series(date_map)
    dates.index = dates.index.map(str)
    aligned = dates.reindex(labels)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise ValueError(f"date_map is missing labels: {missing}")
    return aligned


__all__ = [
    "date_preserving_label_shuffle_null",
    "empirical_p_value",
    "label_shuffle_null",
    "stimulus_subset_rsa",
]
