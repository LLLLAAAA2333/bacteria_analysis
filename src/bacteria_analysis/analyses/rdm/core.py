"""Shared representational dissimilarity matrix helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_square_rdms(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return shared-label square RDMs ordered by ``left`` labels."""

    left_square = _prepare_square_rdm(left, name="left")
    right_square = _prepare_square_rdm(right, name="right")
    shared_labels = [label for label in left_square.index if label in set(right_square.index)]
    if not shared_labels:
        return (
            left_square.loc[[], []].copy(),
            right_square.loc[[], []].copy(),
        )
    return (
        left_square.loc[shared_labels, shared_labels].copy(),
        right_square.loc[shared_labels, shared_labels].copy(),
    )


def upper_triangle_values(matrix: pd.DataFrame) -> pd.Series:
    """Return off-diagonal upper-triangle values with stimulus pair labels."""

    square = _prepare_square_rdm(matrix, name="matrix")
    labels = square.index.tolist()
    pairs: list[tuple[str, str]] = []
    values: list[float] = []
    for row_idx, left_label in enumerate(labels):
        for col_idx in range(row_idx + 1, len(labels)):
            pairs.append((left_label, labels[col_idx]))
            values.append(square.iloc[row_idx, col_idx])
    index = pd.MultiIndex.from_tuples(pairs, names=["stimulus_left", "stimulus_right"])
    return pd.Series(values, index=index, name="distance")


def rdm_pair_values(neural: pd.DataFrame, chemical: pd.DataFrame) -> pd.DataFrame:
    """Align RDMs and return paired upper-triangle distances."""

    aligned_neural, aligned_chemical = align_square_rdms(neural, chemical)
    neural_values = upper_triangle_values(aligned_neural)
    chemical_values = upper_triangle_values(aligned_chemical)
    rows = [
        {
            "stimulus_left": pair[0],
            "stimulus_right": pair[1],
            "neural_distance": neural_values.loc[pair],
            "chemical_distance": chemical_values.loc[pair],
        }
        for pair in neural_values.index
    ]
    return pd.DataFrame(
        rows,
        columns=["stimulus_left", "stimulus_right", "neural_distance", "chemical_distance"],
    )


def spearman_similarity(left, right) -> float:
    """Return Pearson correlation of average ranks after pairwise NaN removal."""

    left_values, right_values = _paired_finite_values(left, right)
    if left_values.size < 2:
        return float("nan")
    return pearson_similarity(rank_normalize(left_values), rank_normalize(right_values))


def pearson_similarity(left, right) -> float:
    """Return Pearson correlation after pairwise NaN removal."""

    left_values, right_values = _paired_finite_values(left, right)
    if left_values.size < 2:
        return float("nan")
    if np.unique(left_values).size < 2 or np.unique(right_values).size < 2:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def rank_normalize(values) -> np.ndarray:
    """Return one-based average ranks, preserving NaN positions."""

    return pd.Series(values, copy=False).rank(method="average").to_numpy(dtype=float)


def _prepare_square_rdm(matrix: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")

    if "stimulus_row" in matrix.columns:
        square = matrix.set_index("stimulus_row").copy()
    else:
        square = matrix.copy()

    if square.shape[0] != square.shape[1]:
        raise ValueError(f"{name} RDM must be square; found shape {square.shape}")
    if not square.index.is_unique:
        duplicates = square.index[square.index.duplicated()].unique().tolist()
        raise ValueError(f"{name} RDM index contains duplicate stimulus labels: {duplicates}")
    if not square.columns.is_unique:
        duplicates = square.columns[square.columns.duplicated()].unique().tolist()
        raise ValueError(f"{name} RDM columns contain duplicate stimulus labels: {duplicates}")

    index_labels = pd.Index(square.index.map(str))
    column_labels = pd.Index(square.columns.map(str))
    if not index_labels.is_unique:
        duplicates = index_labels[index_labels.duplicated()].unique().tolist()
        raise ValueError(f"{name} RDM index labels collide after string normalization: {duplicates}")
    if not column_labels.is_unique:
        duplicates = column_labels[column_labels.duplicated()].unique().tolist()
        raise ValueError(f"{name} RDM column labels collide after string normalization: {duplicates}")
    if set(index_labels) != set(column_labels):
        raise ValueError(
            f"{name} RDM columns must match index labels; "
            f"index={index_labels.tolist()} columns={column_labels.tolist()}"
        )

    square.index = index_labels
    square.columns = column_labels
    square = square.reindex(columns=index_labels)
    numeric = square.apply(pd.to_numeric, errors="coerce")
    bad_mask = numeric.isna() & square.notna()
    if bad_mask.to_numpy().any():
        bad_positions = [
            f"{row_label},{column_label}"
            for row_label, row in bad_mask.iterrows()
            for column_label, is_bad in row.items()
            if bool(is_bad)
        ]
        raise ValueError(f"{name} RDM contains non-numeric values at: {', '.join(bad_positions)}")
    return numeric


def _paired_finite_values(left, right) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.asarray(left, dtype=float).ravel()
    right_values = np.asarray(right, dtype=float).ravel()
    if left_values.shape != right_values.shape:
        raise ValueError(f"values must have the same shape; left={left_values.shape} right={right_values.shape}")
    finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
    return left_values[finite_mask], right_values[finite_mask]


__all__ = [
    "align_square_rdms",
    "pearson_similarity",
    "rank_normalize",
    "rdm_pair_values",
    "spearman_similarity",
    "upper_triangle_values",
]
