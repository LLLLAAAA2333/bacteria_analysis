import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.rsa import (
    align_rdm_upper_triangles,
    benjamini_hochberg,
    build_permutation_null,
    compute_rsa_score,
    summarize_cross_view_comparison,
    summarize_leave_one_stimulus_out,
)


def _matrix(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_align_rdm_upper_triangles_uses_stimulus_labels_not_row_positions():
    neural = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 1.0, "s3": 5.0},
            {"stimulus_row": "s2", "s1": 1.0, "s2": 0.0, "s3": 3.0},
            {"stimulus_row": "s3", "s1": 5.0, "s2": 3.0, "s3": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "s3", "s2": 7.0, "s3": 0.0, "s1": 9.0},
            {"stimulus_row": "s1", "s2": 2.0, "s3": 9.0, "s1": 0.0},
            {"stimulus_row": "s2", "s2": 0.0, "s3": 7.0, "s1": 2.0},
        ]
    )

    shared = align_rdm_upper_triangles(neural, model)

    assert shared.to_dict("records") == [
        {"stimulus_left": "s1", "stimulus_right": "s2", "neural_value": 1.0, "model_value": 2.0},
        {"stimulus_left": "s1", "stimulus_right": "s3", "neural_value": 5.0, "model_value": 9.0},
        {"stimulus_left": "s2", "stimulus_right": "s3", "neural_value": 3.0, "model_value": 7.0},
    ]


def test_compute_rsa_score_uses_shared_upper_triangle_entries_only():
    neural = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 0.1, "s3": 0.8},
            {"stimulus_row": "s2", "s1": 0.1, "s2": 0.0, "s3": 0.3},
            {"stimulus_row": "s3", "s1": 0.8, "s2": 0.3, "s3": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 0.2, "s3": 0.4},
            {"stimulus_row": "s2", "s1": 0.2, "s2": 0.0, "s3": np.nan},
            {"stimulus_row": "s3", "s1": 0.4, "s2": np.nan, "s3": 0.0},
        ]
    )

    result = compute_rsa_score(neural, model)

    assert result["score_status"] == "ok"
    assert result["n_shared_entries"] == 2
    assert result["rsa_similarity"] == pytest.approx(1.0)
    assert np.isnan(result["p_value_raw"])
    assert np.isnan(result["p_value_fdr"])


def test_benjamini_hochberg_returns_monotonic_adjusted_values():
    adjusted = benjamini_hochberg(np.array([0.01, 0.03, 0.2]))

    assert np.all(np.diff(adjusted) >= 0)
    assert adjusted.tolist() == pytest.approx([0.03, 0.045, 0.2])


def test_leave_one_stimulus_out_summary_records_excluded_stimulus():
    neural = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 0.2, "b3_1": 0.4},
            {"stimulus_row": "b2_1", "b1_1": 0.2, "b2_1": 0.0, "b3_1": 0.3},
            {"stimulus_row": "b3_1", "b1_1": 0.4, "b2_1": 0.3, "b3_1": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 1.0, "b3_1": 2.0},
            {"stimulus_row": "b2_1", "b1_1": 1.0, "b2_1": 0.0, "b3_1": 3.0},
            {"stimulus_row": "b3_1", "b1_1": 2.0, "b2_1": 3.0, "b3_1": 0.0},
        ]
    )

    summary = summarize_leave_one_stimulus_out(neural, model)

    assert set(summary["excluded_stimulus"]) == {"b1_1", "b2_1", "b3_1"}
    assert summary["n_shared_entries"].tolist() == [1, 1, 1]


def test_build_permutation_null_is_reproducible_for_fixed_seed():
    neural = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 0.2, "s3": 0.8, "s4": 0.4},
            {"stimulus_row": "s2", "s1": 0.2, "s2": 0.0, "s3": 0.5, "s4": 0.1},
            {"stimulus_row": "s3", "s1": 0.8, "s2": 0.5, "s3": 0.0, "s4": 0.9},
            {"stimulus_row": "s4", "s1": 0.4, "s2": 0.1, "s3": 0.9, "s4": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 1.0, "s3": 4.0, "s4": 2.0},
            {"stimulus_row": "s2", "s1": 1.0, "s2": 0.0, "s3": 3.0, "s4": 5.0},
            {"stimulus_row": "s3", "s1": 4.0, "s2": 3.0, "s3": 0.0, "s4": 6.0},
            {"stimulus_row": "s4", "s1": 2.0, "s2": 5.0, "s3": 6.0, "s4": 0.0},
        ]
    )

    first = build_permutation_null(neural, model, n_iterations=6, seed=13)
    second = build_permutation_null(neural, model, n_iterations=6, seed=13)

    pd.testing.assert_frame_equal(first, second)


def test_summarize_cross_view_comparison_returns_requested_views():
    model = _matrix(
        [
            {"stimulus_row": "A001", "A001": 0.0, "A002": 1.0, "A003": 2.0},
            {"stimulus_row": "A002", "A001": 1.0, "A002": 0.0, "A003": 3.0},
            {"stimulus_row": "A003", "A001": 2.0, "A002": 3.0, "A003": 0.0},
        ]
    )
    neural_matrices = {
        "response_window": _matrix(
            [
                {"stimulus_row": "A001", "A001": 0.0, "A002": 0.2, "A003": 0.4},
                {"stimulus_row": "A002", "A001": 0.2, "A002": 0.0, "A003": 0.3},
                {"stimulus_row": "A003", "A001": 0.4, "A002": 0.3, "A003": 0.0},
            ]
        ),
        "full_trajectory": _matrix(
            [
                {"stimulus_row": "A001", "A001": 0.0, "A002": 0.1, "A003": 0.5},
                {"stimulus_row": "A002", "A001": 0.1, "A002": 0.0, "A003": 0.2},
                {"stimulus_row": "A003", "A001": 0.5, "A002": 0.2, "A003": 0.0},
            ]
        ),
    }

    summary = summarize_cross_view_comparison(neural_matrices, model)

    assert summary["view_name"].tolist() == ["response_window", "full_trajectory"]
    assert summary["score_status"].tolist() == ["ok", "ok"]


def test_compute_rsa_score_rejects_labels_that_collide_after_string_normalization():
    matrix = _matrix(
        [
            {"stimulus_row": 1, 1: 0.0, "1": 0.2},
            {"stimulus_row": "1", 1: 0.2, "1": 0.0},
        ]
    )

    with pytest.raises(ValueError, match="collide after string normalization"):
        compute_rsa_score(matrix, matrix)
