import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.analyses.rdm.core import (
    align_square_rdms,
    pearson_similarity,
    rank_normalize,
    rdm_pair_values,
    spearman_similarity,
    upper_triangle_values,
)


def _rdm(labels, values):
    return pd.DataFrame(values, index=labels, columns=labels)


def test_align_square_rdms_preserves_left_order_and_drops_non_shared_labels():
    left = _rdm(["b", "a", "c"], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    right = _rdm(["c", "b", "d"], [[0, 20, 9], [20, 0, 8], [9, 8, 0]])

    aligned_left, aligned_right = align_square_rdms(left, right)

    assert aligned_left.index.tolist() == ["b", "c"]
    assert aligned_left.columns.tolist() == ["b", "c"]
    assert aligned_right.index.tolist() == ["b", "c"]
    assert aligned_right.columns.tolist() == ["b", "c"]
    assert aligned_right.loc["b", "c"] == 20


def test_align_square_rdms_accepts_stimulus_row_matrix_frames():
    left = pd.DataFrame({"stimulus_row": ["a", "b"], "a": [0, 1], "b": [1, 0]})
    right = pd.DataFrame({"stimulus_row": ["b", "a"], "b": [0, 5], "a": [5, 0]})

    _, aligned_right = align_square_rdms(left, right)

    assert aligned_right.index.tolist() == ["a", "b"]
    assert aligned_right.loc["a", "b"] == 5


def test_align_square_rdms_rejects_invalid_square_matrix():
    matrix = pd.DataFrame([[0, 1, 2], [1, 0, 3]], index=["a", "b"], columns=["a", "b", "c"])

    with pytest.raises(ValueError, match="must be square"):
        align_square_rdms(matrix, matrix)


def test_align_square_rdms_rejects_mismatched_labels():
    matrix = pd.DataFrame([[0, 1], [1, 0]], index=["a", "b"], columns=["a", "c"])

    with pytest.raises(ValueError, match="columns must match index labels"):
        align_square_rdms(matrix, matrix)


def test_align_square_rdms_rejects_nonnumeric_values():
    matrix = pd.DataFrame([[0, "bad"], [1, 0]], index=["a", "b"], columns=["a", "b"])

    with pytest.raises(ValueError, match="non-numeric"):
        align_square_rdms(matrix, matrix)


def test_upper_triangle_values_excludes_diagonal_and_uses_pair_index():
    matrix = _rdm(["a", "b", "c"], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])

    result = upper_triangle_values(matrix)

    assert result.index.names == ["stimulus_left", "stimulus_right"]
    assert result.to_dict() == {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 3}


def test_rdm_pair_values_aligns_before_returning_distances():
    neural = _rdm(["b", "a", "c"], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    chemical = _rdm(["c", "b"], [[0, 20], [20, 0]])

    result = rdm_pair_values(neural, chemical)

    assert result.to_dict("records") == [
        {
            "stimulus_left": "b",
            "stimulus_right": "c",
            "neural_distance": 2,
            "chemical_distance": 20,
        }
    ]


def test_spearman_and_pearson_similarity_expected_values():
    left = np.array([1, 2, 3, 4], dtype=float)
    right = np.array([2, 4, 6, 8], dtype=float)
    reversed_right = right[::-1]

    assert pearson_similarity(left, right) == pytest.approx(1.0)
    assert spearman_similarity(left, right) == pytest.approx(1.0)
    assert spearman_similarity(left, reversed_right) == pytest.approx(-1.0)


def test_rank_normalize_uses_average_ranks_for_ties_and_preserves_nan():
    result = rank_normalize([10, 10, 30, np.nan])

    np.testing.assert_allclose(result[:3], [1.5, 1.5, 3.0])
    assert np.isnan(result[3])


def test_similarity_drops_paired_nans_and_returns_nan_for_constant_values():
    assert pearson_similarity([1, np.nan, 3], [1, 100, 3]) == pytest.approx(1.0)
    assert np.isnan(pearson_similarity([1, 1, 1], [1, 2, 3]))
    assert np.isnan(spearman_similarity([np.nan, 1], [2, 3]))
