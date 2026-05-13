import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.analyses.rdm.stats import (
    date_preserving_label_shuffle_null,
    empirical_p_value,
    label_shuffle_null,
    stimulus_subset_rsa,
)


def _rdm(labels, values):
    return pd.DataFrame(values, index=labels, columns=labels)


def test_empirical_p_value_supports_sides_with_plus_one_correction():
    null = np.array([0.1, 0.2, 0.3, np.nan])

    assert empirical_p_value(0.25, null, side="greater") == pytest.approx(2 / 4)
    assert empirical_p_value(0.25, null, side="less") == pytest.approx(3 / 4)
    assert empirical_p_value(-0.25, null, side="two-sided") == pytest.approx(2 / 4)


def test_empirical_p_value_rejects_unknown_side():
    with pytest.raises(ValueError, match="side must be"):
        empirical_p_value(0.1, np.array([0.1]), side="middle")


def test_label_shuffle_null_is_reproducible():
    neural = _rdm(["a", "b", "c", "d"], [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
    chemical = _rdm(["a", "b", "c", "d"], [[0, 6, 5, 4], [6, 0, 3, 2], [5, 3, 0, 1], [4, 2, 1, 0]])

    first = label_shuffle_null(neural, chemical, n_permutations=5, seed=42)
    second = label_shuffle_null(neural, chemical, n_permutations=5, seed=42)

    np.testing.assert_allclose(first, second)
    assert first.shape == (5,)


def test_date_preserving_label_shuffle_null_preserves_within_date_counts():
    labels = ["a", "b", "c", "d"]
    neural = _rdm(labels, [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
    chemical = _rdm(labels, [[0, 6, 5, 4], [6, 0, 3, 2], [5, 3, 0, 1], [4, 2, 1, 0]])
    date_map = pd.Series({"a": "day1", "b": "day1", "c": "day2", "d": "day2"})

    result = date_preserving_label_shuffle_null(neural, chemical, date_map, n_permutations=4, seed=7)
    repeat = date_preserving_label_shuffle_null(neural, chemical, date_map, n_permutations=4, seed=7)
    unrestricted = label_shuffle_null(neural, chemical, n_permutations=4, seed=7)

    np.testing.assert_allclose(result, repeat)
    assert result.shape == (4,)
    assert not np.allclose(result, unrestricted, equal_nan=True)


def test_date_preserving_label_shuffle_null_rejects_missing_dates():
    neural = _rdm(["a", "b"], [[0, 1], [1, 0]])
    chemical = _rdm(["a", "b"], [[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="date_map is missing labels"):
        date_preserving_label_shuffle_null(neural, chemical, {"a": "day1"}, n_permutations=1, seed=0)


def test_stimulus_subset_rsa_is_reproducible():
    labels = ["a", "b", "c", "d", "e"]
    neural = _rdm(
        labels,
        [
            [0, 1, 2, 3, 4],
            [1, 0, 5, 6, 7],
            [2, 5, 0, 8, 9],
            [3, 6, 8, 0, 10],
            [4, 7, 9, 10, 0],
        ],
    )
    chemical = _rdm(
        labels,
        [
            [0, 10, 9, 8, 7],
            [10, 0, 6, 5, 4],
            [9, 6, 0, 3, 2],
            [8, 5, 3, 0, 1],
            [7, 4, 2, 1, 0],
        ],
    )

    first = stimulus_subset_rsa(neural, chemical, subset_count=4, subset_fraction=0.6, seed=11)
    second = stimulus_subset_rsa(neural, chemical, subset_count=4, subset_fraction=0.6, seed=11)

    pd.testing.assert_frame_equal(first, second)
    assert first["iteration"].tolist() == [0, 1, 2, 3]
    assert first["n_stimuli"].tolist() == [3, 3, 3, 3]
    assert first["rsa_similarity"].notna().all()


def test_stimulus_subset_rsa_validates_arguments():
    neural = _rdm(["a", "b"], [[0, 1], [1, 0]])
    chemical = _rdm(["a", "b"], [[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="subset_count"):
        stimulus_subset_rsa(neural, chemical, subset_count=-1, subset_fraction=0.5, seed=0)
    with pytest.raises(ValueError, match="subset_fraction"):
        stimulus_subset_rsa(neural, chemical, subset_count=1, subset_fraction=0, seed=0)


def test_stimulus_subset_rsa_rejects_too_few_shared_labels():
    neural = _rdm(["a"], [[0]])
    chemical = _rdm(["b"], [[0]])

    with pytest.raises(ValueError, match="at least 2 shared labels"):
        stimulus_subset_rsa(neural, chemical, subset_count=1, subset_fraction=1, seed=0)
