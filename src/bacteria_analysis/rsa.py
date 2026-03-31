"""Stage 3 RSA statistics and robustness helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CROSS_VIEW_NAMES = ("response_window", "full_trajectory")


def align_rdm_upper_triangles(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> pd.DataFrame:
    """Align shared upper-triangle entries by stimulus labels."""

    neural_triangle = _extract_upper_triangle(neural_matrix).rename(columns={"value": "neural_value"})
    model_triangle = _extract_upper_triangle(model_matrix).rename(columns={"value": "model_value"})
    shared = neural_triangle.merge(model_triangle, on=["stimulus_left", "stimulus_right"], how="inner")
    if shared.empty:
        return pd.DataFrame(columns=["stimulus_left", "stimulus_right", "neural_value", "model_value"])

    value_columns = ["neural_value", "model_value"]
    shared[value_columns] = shared[value_columns].apply(pd.to_numeric, errors="coerce")
    return shared.reset_index(drop=True)


def compute_rsa_score(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> dict[str, object]:
    shared = align_rdm_upper_triangles(neural_matrix, model_matrix)
    shared = shared.dropna(subset=["neural_value", "model_value"]).reset_index(drop=True)
    n_shared_entries = int(len(shared))
    if n_shared_entries < 2:
        return {
            "score_method": "spearman",
            "score_status": "invalid",
            "n_shared_entries": n_shared_entries,
            "rsa_similarity": np.nan,
            "p_value_raw": np.nan,
            "p_value_fdr": np.nan,
        }

    rsa_similarity = _spearman_similarity(
        shared["neural_value"].to_numpy(dtype=float),
        shared["model_value"].to_numpy(dtype=float),
    )
    score_status = "ok" if np.isfinite(rsa_similarity) else "invalid"
    return {
        "score_method": "spearman",
        "score_status": score_status,
        "n_shared_entries": n_shared_entries,
        "rsa_similarity": rsa_similarity,
        "p_value_raw": np.nan,
        "p_value_fdr": np.nan,
    }


def build_permutation_null(
    neural_matrix: pd.DataFrame,
    model_matrix: pd.DataFrame,
    n_iterations: int,
    seed: int,
) -> pd.DataFrame:
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    model_square = _prepare_square_matrix(model_matrix)
    for iteration in range(n_iterations):
        permuted_model = _matrix_to_frame(_permute_model_labels(model_square, rng))
        score = compute_rsa_score(neural_matrix, permuted_model)
        rows.append({"iteration": iteration, **score})
    return pd.DataFrame(
        rows,
        columns=[
            "iteration",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be a one-dimensional array")
    if values.size == 0:
        return values.copy()

    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return adjusted

    order = np.argsort(finite_values)
    ranked = finite_values[order]
    scale = finite_values.size / np.arange(1, finite_values.size + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    finite_indices = np.flatnonzero(finite_mask)
    adjusted[finite_indices[order]] = ranked_adjusted
    return adjusted


def summarize_leave_one_stimulus_out(
    neural_matrix: pd.DataFrame,
    model_matrix: pd.DataFrame,
    *,
    n_iterations: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    shared_labels = sorted(_shared_stimulus_labels(neural_matrix, model_matrix))
    rows: list[dict[str, object]] = []
    for offset, stimulus_label in enumerate(shared_labels):
        reduced_neural = _matrix_to_frame(_drop_stimulus(_prepare_square_matrix(neural_matrix), stimulus_label))
        reduced_model = _matrix_to_frame(_drop_stimulus(_prepare_square_matrix(model_matrix), stimulus_label))
        score = compute_rsa_score(reduced_neural, reduced_model)
        p_value_raw = np.nan
        if n_iterations > 0 and score["score_status"] == "ok":
            null = build_permutation_null(
                reduced_neural,
                reduced_model,
                n_iterations=n_iterations,
                seed=seed + offset,
            )
            p_value_raw = _empirical_p_value(score["rsa_similarity"], null["rsa_similarity"].to_numpy(dtype=float))
        rows.append(
            {
                "excluded_stimulus": stimulus_label,
                **score,
                "p_value_raw": p_value_raw,
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "excluded_stimulus",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )
    if summary.empty:
        return summary
    summary["p_value_fdr"] = benjamini_hochberg(summary["p_value_raw"].to_numpy(dtype=float))
    return summary


def summarize_cross_view_comparison(
    neural_matrices: dict[str, pd.DataFrame],
    model_matrix: pd.DataFrame,
    *,
    view_names: tuple[str, ...] | list[str] = DEFAULT_CROSS_VIEW_NAMES,
    n_iterations: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    requested_views: list[str] = []
    for view_name in view_names:
        normalized = str(view_name)
        if normalized not in neural_matrices:
            raise ValueError(f"missing neural matrix for view_name {normalized!r}")
        if normalized not in requested_views:
            requested_views.append(normalized)

    rows: list[dict[str, object]] = []
    for offset, view_name in enumerate(requested_views):
        score = compute_rsa_score(neural_matrices[view_name], model_matrix)
        p_value_raw = np.nan
        if n_iterations > 0 and score["score_status"] == "ok":
            null = build_permutation_null(
                neural_matrices[view_name],
                model_matrix,
                n_iterations=n_iterations,
                seed=seed + offset,
            )
            p_value_raw = _empirical_p_value(score["rsa_similarity"], null["rsa_similarity"].to_numpy(dtype=float))
        rows.append(
            {
                "view_name": view_name,
                **score,
                "p_value_raw": p_value_raw,
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "view_name",
            "score_method",
            "score_status",
            "n_shared_entries",
            "rsa_similarity",
            "p_value_raw",
            "p_value_fdr",
        ],
    )
    if summary.empty:
        return summary
    summary["p_value_fdr"] = benjamini_hochberg(summary["p_value_raw"].to_numpy(dtype=float))
    return summary


def _prepare_square_matrix(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    if "stimulus_row" not in matrix_frame.columns:
        raise ValueError("matrix_frame must include a stimulus_row column")

    matrix = matrix_frame.set_index("stimulus_row").copy()
    if not matrix.index.is_unique:
        duplicates = matrix.index[matrix.index.duplicated()].unique().tolist()
        raise ValueError(f"matrix index contains duplicate stimulus labels: {duplicates}")
    if not matrix.columns.is_unique:
        duplicates = matrix.columns[matrix.columns.duplicated()].unique().tolist()
        raise ValueError(f"matrix columns contain duplicate stimulus labels: {duplicates}")

    normalized_index = pd.Index(matrix.index.map(str))
    normalized_columns = pd.Index(matrix.columns.map(str))
    if not normalized_index.is_unique:
        duplicates = normalized_index[normalized_index.duplicated()].unique().tolist()
        raise ValueError(f"matrix index labels collide after string normalization: {duplicates}")
    if not normalized_columns.is_unique:
        duplicates = normalized_columns[normalized_columns.duplicated()].unique().tolist()
        raise ValueError(f"matrix columns labels collide after string normalization: {duplicates}")

    matrix.index = normalized_index
    matrix.columns = normalized_columns
    if set(matrix.index) != set(matrix.columns):
        raise ValueError(
            "matrix columns must match stimulus_row labels; "
            f"index={matrix.index.tolist()} columns={matrix.columns.tolist()}"
        )

    matrix = matrix.reindex(columns=matrix.index)
    return matrix.apply(pd.to_numeric, errors="coerce")


def _extract_upper_triangle(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    matrix = _prepare_square_matrix(matrix_frame)
    rows: list[dict[str, object]] = []
    labels = matrix.index.tolist()
    for row_idx, stimulus_left in enumerate(labels):
        for col_idx in range(row_idx + 1, len(labels)):
            stimulus_pair = sorted((stimulus_left, labels[col_idx]))
            rows.append(
                {
                    "stimulus_left": stimulus_pair[0],
                    "stimulus_right": stimulus_pair[1],
                    "value": matrix.iloc[row_idx, col_idx],
                }
            )
    return pd.DataFrame(rows, columns=["stimulus_left", "stimulus_right", "value"])


def _spearman_similarity(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_ranks = pd.Series(left_values, copy=False).rank(method="average").to_numpy(dtype=float)
    right_ranks = pd.Series(right_values, copy=False).rank(method="average").to_numpy(dtype=float)
    if np.unique(left_ranks).size < 2 or np.unique(right_ranks).size < 2:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _shared_stimulus_labels(neural_matrix: pd.DataFrame, model_matrix: pd.DataFrame) -> list[str]:
    neural_labels = _prepare_square_matrix(neural_matrix).index.tolist()
    model_labels = set(_prepare_square_matrix(model_matrix).index.tolist())
    return [label for label in neural_labels if label in model_labels]


def _drop_stimulus(matrix: pd.DataFrame, stimulus_label: str) -> pd.DataFrame:
    if stimulus_label not in matrix.index:
        return matrix.copy()
    return matrix.drop(index=stimulus_label, columns=stimulus_label)


def _matrix_to_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    frame = matrix.copy()
    frame.insert(0, "stimulus_row", frame.index.astype(str))
    return frame.reset_index(drop=True)


def _permute_model_labels(model_matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    labels = np.array(sorted(model_matrix.index.astype(str).tolist()), dtype=object)
    canonical = model_matrix.loc[labels, labels].copy()
    permuted_labels = rng.permutation(labels)
    permuted = canonical.loc[permuted_labels, permuted_labels].copy()
    permuted.index = labels
    permuted.columns = labels
    return permuted


def _empirical_p_value(observed_similarity: float, null_values: np.ndarray) -> float:
    finite_null = np.asarray(null_values, dtype=float)
    finite_null = finite_null[np.isfinite(finite_null)]
    if finite_null.size == 0 or not np.isfinite(observed_similarity):
        return float("nan")
    return float((1 + np.sum(finite_null >= observed_similarity)) / (finite_null.size + 1))


__all__ = [
    "DEFAULT_CROSS_VIEW_NAMES",
    "align_rdm_upper_triangles",
    "benjamini_hochberg",
    "build_permutation_null",
    "compute_rsa_score",
    "summarize_cross_view_comparison",
    "summarize_leave_one_stimulus_out",
]
