import pandas as pd
import pytest

from bacteria_analysis.geometry import (
    build_rdm_matrix,
    extract_upper_triangle,
    score_rdm_similarity,
    summarize_grouped_stimulus_pairs,
    summarize_rdm_stability,
)


def test_individual_grouped_stimulus_pair_summary_excludes_cross_individual_pairs(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="individual",
    )

    assert set(summary["group_id"]) == {"2026-01-01__worm_a", "2026-01-02__worm_b"}
    assert (summary["group_type"] == "individual").all()
    assert summary["view_name"].eq("response_window").all()
    assert summary["n_pairs"].sum() == 3


def test_date_grouped_stimulus_pair_summary_uses_date_groups(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="full_trajectory",
        group_type="date",
    )

    assert set(summary["group_id"]) == {"2026-01-01"}
    assert (summary["group_type"] == "date").all()
    assert summary["view_name"].eq("full_trajectory").all()
    assert summary["n_pairs"].sum() == 1


def test_pooled_grouped_stimulus_pair_summary_uses_single_group(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="pooled",
    )

    assert summary["group_id"].nunique() == 1
    assert summary["group_id"].iat[0] == "pooled"
    assert summary["group_type"].eq("pooled").all()
    assert summary["view_name"].eq("response_window").all()
    assert set(zip(summary["stimulus_left"], summary["stimulus_right"])) == {
        ("b1_1", "b1_1"),
        ("b1_1", "b2_1"),
        ("b2_1", "b2_1"),
    }
    assert summary["n_pairs"].sum() == 4


def test_build_pooled_rdm_matrix_is_symmetric(synthetic_geometry_comparisons):
    pair_summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="pooled",
    )

    matrix = build_rdm_matrix(pair_summary, group_id="pooled").set_index("stimulus_row")

    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.equals(matrix.T)
    assert matrix.loc["b1_1", "b1_1"] == pytest.approx(0.10)
    assert matrix.loc["b2_1", "b2_1"] == pytest.approx(0.20)
    assert matrix.loc["b1_1", "b2_1"] == pytest.approx(0.875)


def test_build_rdm_matrix_rejects_mixed_view_summary(synthetic_geometry_comparisons):
    pooled_summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="pooled",
    )
    mixed_summary = pooled_summary.copy()
    mixed_summary.loc[mixed_summary.index[0], "view_name"] = "full_trajectory"

    with pytest.raises(ValueError, match="exactly one \\(view_name, group_type\\) combination"):
        build_rdm_matrix(mixed_summary, group_id="pooled")


def test_build_rdm_matrix_returns_stable_empty_schema_for_missing_group_id(synthetic_geometry_comparisons):
    pair_summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="pooled",
    )

    empty = build_rdm_matrix(pair_summary, group_id="missing")

    assert list(empty.columns) == ["stimulus_row"]
    assert empty.empty


def test_invalid_group_type_raises_value_error(synthetic_geometry_comparisons):
    with pytest.raises(ValueError, match="unsupported group_type: invalid"):
        summarize_grouped_stimulus_pairs(
            synthetic_geometry_comparisons,
            view_name="response_window",
            group_type="invalid",
        )


def test_extract_upper_triangle_returns_strict_upper_entries():
    matrix = pd.DataFrame(
        {
            "stimulus_row": ["b1_1", "b2_1", "b3_1"],
            "b1_1": [0.0, 0.3, 0.5],
            "b2_1": [0.3, 0.0, 0.7],
            "b3_1": [0.5, 0.7, 0.0],
        }
    )

    result = extract_upper_triangle(matrix)

    assert result.to_dict("records") == [
        {"stimulus_left": "b1_1", "stimulus_right": "b2_1", "value": 0.3},
        {"stimulus_left": "b1_1", "stimulus_right": "b3_1", "value": 0.5},
        {"stimulus_left": "b2_1", "stimulus_right": "b3_1", "value": 0.7},
    ]


def test_score_rdm_similarity_uses_shared_non_missing_entries_only():
    left = pd.DataFrame(
        {
            "stimulus_row": ["s1", "s2", "s3"],
            "s1": [0.0, 0.1, 0.8],
            "s2": [0.1, 0.0, 0.3],
            "s3": [0.8, 0.3, 0.0],
        }
    )
    right = pd.DataFrame(
        {
            "stimulus_row": ["s1", "s2", "s3"],
            "s1": [0.0, 0.2, 0.4],
            "s2": [0.2, 0.0, pd.NA],
            "s3": [0.4, pd.NA, 0.0],
        }
    )

    result = score_rdm_similarity(left, right)

    assert result["score_status"] == "ok"
    assert result["n_shared_entries"] == 2
    assert result["similarity"] == pytest.approx(1.0)


def test_score_rdm_similarity_marks_invalid_when_no_shared_entries():
    left_empty = pd.DataFrame(
        {
            "stimulus_row": ["s1", "s2"],
            "s1": [0.0, pd.NA],
            "s2": [pd.NA, 0.0],
        }
    )
    right_empty = pd.DataFrame(
        {
            "stimulus_row": ["s1", "s2"],
            "s1": [0.0, pd.NA],
            "s2": [pd.NA, 0.0],
        }
    )

    result = score_rdm_similarity(left_empty, right_empty)

    assert result["score_status"] == "invalid"
    assert result["n_shared_entries"] == 0
    assert pd.isna(result["similarity"])


def test_summarize_rdm_stability_returns_minimum_similarity_views():
    pair_summary = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_a",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.2,
                "median_distance": 0.2,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_a",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.9,
                "median_distance": 0.9,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_a",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.4,
                "median_distance": 0.4,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_b",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.3,
                "median_distance": 0.3,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_b",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.8,
                "median_distance": 0.8,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_b",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.6,
                "median_distance": 0.6,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "date",
                "group_id": "2026-01-01",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.1,
                "median_distance": 0.1,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "date",
                "group_id": "2026-01-01",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.7,
                "median_distance": 0.7,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "date",
                "group_id": "2026-01-01",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.5,
                "median_distance": 0.5,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "date",
                "group_id": "2026-01-02",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.2,
                "median_distance": 0.2,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "date",
                "group_id": "2026-01-02",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.6,
                "median_distance": 0.6,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "date",
                "group_id": "2026-01-02",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.4,
                "median_distance": 0.4,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.25,
                "median_distance": 0.25,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.85,
                "median_distance": 0.85,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.50,
                "median_distance": 0.50,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.15,
                "median_distance": 0.15,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.65,
                "median_distance": 0.65,
            },
            {
                "view_name": "full_trajectory",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.45,
                "median_distance": 0.45,
            },
        ]
    )

    result = summarize_rdm_stability(pair_summary)

    assert set(result["comparison_scope"]) == {
        "within_group_type",
        "pooled_vs_group",
        "pooled_cross_view",
    }
    assert set(result["group_type"]) == {"individual", "date", "pooled"}
    assert result["score_status"].eq("ok").all()
    assert result["n_shared_entries"].eq(3).all()
    assert result["similarity"].tolist() == pytest.approx([1.0] * len(result))
