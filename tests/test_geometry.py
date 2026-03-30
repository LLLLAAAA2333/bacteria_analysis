import json

import pandas as pd
import pytest

from bacteria_analysis.geometry import (
    build_rdm_matrix,
    extract_upper_triangle,
    score_rdm_similarity,
    summarize_grouped_stimulus_pairs,
    summarize_rdm_stability,
)
from bacteria_analysis.geometry_outputs import ensure_stage2_output_dirs, write_stage2_outputs


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


def test_extract_upper_triangle_aligns_columns_to_stimulus_row_labels():
    matrix = pd.DataFrame(
        {
            "stimulus_row": ["b1_1", "b2_1", "b3_1"],
            "b3_1": [0.5, 0.7, 0.0],
            "b1_1": [0.0, 0.3, 0.5],
            "b2_1": [0.3, 0.0, 0.7],
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
    assert len(result) == 7
    assert result["score_status"].eq("ok").all()
    assert result["n_shared_entries"].eq(3).all()
    assert result["similarity"].tolist() == pytest.approx([1.0] * len(result))
    assert len(result.loc[result["comparison_scope"] == "within_group_type"]) == 2
    assert len(result.loc[result["comparison_scope"] == "pooled_vs_group"]) == 4
    assert len(result.loc[result["comparison_scope"] == "pooled_cross_view"]) == 1


def test_summarize_rdm_stability_rejects_duplicate_pooled_matrices_for_view():
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
                "mean_distance": 0.8,
                "median_distance": 0.8,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_a",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.5,
                "median_distance": 0.5,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.2,
                "median_distance": 0.2,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.8,
                "median_distance": 0.8,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.5,
                "median_distance": 0.5,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled_copy",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.2,
                "median_distance": 0.2,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled_copy",
                "stimulus_left": "s1",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.8,
                "median_distance": 0.8,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled_copy",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 2,
                "mean_distance": 0.5,
                "median_distance": 0.5,
            },
        ]
    )

    with pytest.raises(ValueError, match="exactly one pooled matrix for view_name 'response_window'"):
        summarize_rdm_stability(pair_summary)


@pytest.fixture
def synthetic_geometry_outputs() -> dict[str, pd.DataFrame]:
    outputs = {
        "rdm_pairs__response_window__pooled": pd.DataFrame.from_records(
            [
                {
                    "view_name": "response_window",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b1_1",
                    "same_stimulus": True,
                    "n_pairs": 1,
                    "mean_distance": 0.10,
                    "median_distance": 0.10,
                },
                {
                    "view_name": "response_window",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": False,
                    "n_pairs": 2,
                    "mean_distance": 0.85,
                    "median_distance": 0.85,
                },
                {
                    "view_name": "response_window",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "stimulus_left": "b2_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": True,
                    "n_pairs": 1,
                    "mean_distance": 0.20,
                    "median_distance": 0.20,
                },
            ]
        ),
        "rdm_pairs__response_window__individual": pd.DataFrame.from_records(
            [
                {
                    "view_name": "response_window",
                    "group_type": "individual",
                    "group_id": "2026-01-01__worm_a",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": False,
                    "n_pairs": 1,
                    "mean_distance": 0.80,
                    "median_distance": 0.80,
                }
            ]
        ),
        "rdm_pairs__response_window__date": pd.DataFrame.from_records(
            [
                {
                    "view_name": "response_window",
                    "group_type": "date",
                    "group_id": "2026-01-01",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": False,
                    "n_pairs": 1,
                    "mean_distance": 0.82,
                    "median_distance": 0.82,
                }
            ]
        ),
        "rdm_pairs__full_trajectory__pooled": pd.DataFrame.from_records(
            [
                {
                    "view_name": "full_trajectory",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b1_1",
                    "same_stimulus": True,
                    "n_pairs": 1,
                    "mean_distance": 0.05,
                    "median_distance": 0.05,
                },
                {
                    "view_name": "full_trajectory",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": False,
                    "n_pairs": 2,
                    "mean_distance": 0.65,
                    "median_distance": 0.65,
                },
                {
                    "view_name": "full_trajectory",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "stimulus_left": "b2_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": True,
                    "n_pairs": 1,
                    "mean_distance": 0.15,
                    "median_distance": 0.15,
                },
            ]
        ),
        "rdm_pairs__full_trajectory__individual": pd.DataFrame.from_records(
            [
                {
                    "view_name": "full_trajectory",
                    "group_type": "individual",
                    "group_id": "2026-01-02__worm_b",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": False,
                    "n_pairs": 1,
                    "mean_distance": 0.60,
                    "median_distance": 0.60,
                }
            ]
        ),
        "rdm_pairs__full_trajectory__date": pd.DataFrame.from_records(
            [
                {
                    "view_name": "full_trajectory",
                    "group_type": "date",
                    "group_id": "2026-01-02",
                    "stimulus_left": "b1_1",
                    "stimulus_right": "b2_1",
                    "same_stimulus": False,
                    "n_pairs": 1,
                    "mean_distance": 0.62,
                    "median_distance": 0.62,
                }
            ]
        ),
        "rdm_matrix__response_window__pooled": pd.DataFrame(
            {
                "stimulus_row": ["b1_1", "b2_1"],
                "b1_1": [0.10, 0.85],
                "b2_1": [0.85, 0.20],
            }
        ),
        "rdm_matrix__full_trajectory__pooled": pd.DataFrame(
            {
                "stimulus_row": ["b1_1", "b2_1"],
                "b1_1": [0.05, 0.65],
                "b2_1": [0.65, 0.15],
            }
        ),
        "rdm_matrix__response_window__individual": pd.DataFrame(
            {
                "stimulus_row": ["b1_1", "b2_1"],
                "b1_1": [0.11, 0.81],
                "b2_1": [0.81, 0.21],
            }
        ),
        "rdm_stability_by_individual": pd.DataFrame.from_records(
            [
                {
                    "comparison_scope": "within_group_type",
                    "view_name": "response_window",
                    "reference_view_name": "response_window",
                    "group_type": "individual",
                    "group_id": "2026-01-01__worm_a",
                    "reference_group_id": "2026-01-02__worm_b",
                    "score_method": "spearman",
                    "score_status": "ok",
                    "n_shared_entries": 3,
                    "similarity": 0.95,
                },
                {
                    "comparison_scope": "pooled_vs_group",
                    "view_name": "full_trajectory",
                    "reference_view_name": "full_trajectory",
                    "group_type": "individual",
                    "group_id": "2026-01-02__worm_b",
                    "reference_group_id": "pooled",
                    "score_method": "spearman",
                    "score_status": "ok",
                    "n_shared_entries": 3,
                    "similarity": 0.90,
                },
            ]
        ),
        "rdm_stability_by_date": pd.DataFrame.from_records(
            [
                {
                    "comparison_scope": "within_group_type",
                    "view_name": "response_window",
                    "reference_view_name": "response_window",
                    "group_type": "date",
                    "group_id": "2026-01-01",
                    "reference_group_id": "2026-01-02",
                    "score_method": "spearman",
                    "score_status": "ok",
                    "n_shared_entries": 3,
                    "similarity": 0.88,
                }
            ]
        ),
        "rdm_view_comparison": pd.DataFrame.from_records(
            [
                {
                    "comparison_scope": "pooled_cross_view",
                    "view_name": "response_window",
                    "reference_view_name": "full_trajectory",
                    "group_type": "pooled",
                    "group_id": "pooled",
                    "reference_group_id": "pooled",
                    "score_method": "spearman",
                    "score_status": "ok",
                    "n_shared_entries": 3,
                    "similarity": 0.99,
                }
            ]
        ),
        "rdm_group_coverage": pd.DataFrame.from_records(
            [
                {"view_name": "response_window", "group_type": "individual", "n_groups": 2},
                {"view_name": "response_window", "group_type": "date", "n_groups": 2},
                {"view_name": "full_trajectory", "group_type": "individual", "n_groups": 1},
                {"view_name": "full_trajectory", "group_type": "date", "n_groups": 1},
            ]
        ),
    }
    return outputs


def test_ensure_stage2_output_dirs_creates_expected_tree(tmp_path):
    dirs = ensure_stage2_output_dirs(tmp_path / "stage2_geometry")

    assert dirs["output_root"].exists()
    assert dirs["tables_dir"].exists()
    assert dirs["figures_dir"].exists()
    assert dirs["qc_dir"].exists()


def test_write_stage2_outputs_writes_required_tables(tmp_path, synthetic_geometry_outputs):
    written = write_stage2_outputs(synthetic_geometry_outputs, tmp_path / "stage2_geometry")

    required_pair_tables = [
        "rdm_pairs__response_window__pooled.parquet",
        "rdm_pairs__response_window__individual.parquet",
        "rdm_pairs__response_window__date.parquet",
        "rdm_pairs__full_trajectory__pooled.parquet",
        "rdm_pairs__full_trajectory__individual.parquet",
        "rdm_pairs__full_trajectory__date.parquet",
    ]
    for name in required_pair_tables:
        assert (written["tables_dir"] / name).exists()

    assert (written["tables_dir"] / "rdm_stability_by_individual.parquet").exists()
    assert (written["tables_dir"] / "rdm_stability_by_date.parquet").exists()
    assert (written["tables_dir"] / "rdm_view_comparison.parquet").exists()
    assert (written["tables_dir"] / "rdm_matrix__response_window__pooled.parquet").exists()
    assert (written["tables_dir"] / "rdm_matrix__full_trajectory__pooled.parquet").exists()
    assert (written["qc_dir"] / "rdm_group_coverage.parquet").exists()
    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled.png").exists()
    assert (written["figures_dir"] / "rdm_matrix__full_trajectory__pooled.png").exists()
    assert (written["figures_dir"] / "rdm_stability_by_individual.png").exists()
    assert (written["figures_dir"] / "rdm_stability_by_date.png").exists()
    assert (written["figures_dir"] / "rdm_view_comparison.png").exists()


def test_write_stage2_outputs_skips_non_pooled_matrix_parquet_artifacts(tmp_path, synthetic_geometry_outputs):
    written = write_stage2_outputs(synthetic_geometry_outputs, tmp_path / "stage2_geometry")

    assert "rdm_matrix__response_window__individual" not in written
    assert not (written["tables_dir"] / "rdm_matrix__response_window__individual.parquet").exists()
    assert (written["tables_dir"] / "rdm_matrix__response_window__pooled.parquet").exists()


def test_write_stage2_outputs_handles_empty_pooled_matrix_frames(tmp_path, synthetic_geometry_outputs):
    outputs = dict(synthetic_geometry_outputs)
    outputs["rdm_matrix__response_window__pooled"] = pd.DataFrame(columns=["stimulus_row"])

    written = write_stage2_outputs(outputs, tmp_path / "stage2_geometry")

    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled.png").exists()
    assert (written["output_root"] / "run_summary.json").exists()

    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_matrix_views"] == ["full_trajectory", "response_window"]


def test_write_stage2_outputs_excludes_non_pooled_views_from_pooled_matrix_summary(tmp_path, synthetic_geometry_outputs):
    outputs = dict(synthetic_geometry_outputs)
    outputs["rdm_matrix__novel_view__individual"] = pd.DataFrame(
        {
            "stimulus_row": ["b1_1", "b2_1"],
            "b1_1": [0.12, 0.72],
            "b2_1": [0.72, 0.18],
        }
    )

    written = write_stage2_outputs(outputs, tmp_path / "stage2_geometry")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["pooled_matrix_views"] == ["full_trajectory", "response_window"]
    assert "novel_view" not in summary["pooled_matrix_views"]


def test_write_stage2_outputs_writes_run_summary(tmp_path, synthetic_geometry_outputs):
    written = write_stage2_outputs(synthetic_geometry_outputs, tmp_path / "stage2_geometry")

    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["views"] == ["full_trajectory", "response_window"]
    assert summary["pooled_matrix_views"] == ["full_trajectory", "response_window"]
    assert summary["pair_table_names"] == [
        "rdm_pairs__full_trajectory__date",
        "rdm_pairs__full_trajectory__individual",
        "rdm_pairs__full_trajectory__pooled",
        "rdm_pairs__response_window__date",
        "rdm_pairs__response_window__individual",
        "rdm_pairs__response_window__pooled",
    ]
    assert summary["stability_table_names"] == ["rdm_stability_by_date", "rdm_stability_by_individual"]
    assert summary["view_comparison_table"] == "rdm_view_comparison"
    assert summary["qc_table_names"] == ["rdm_group_coverage"]
    assert summary["tables_dir"].endswith("stage2_geometry\\tables")
    assert summary["figures_dir"].endswith("stage2_geometry\\figures")

    markdown = (written["output_root"] / "run_summary.md").read_text(encoding="utf-8")
    assert "# Stage 2 Geometry Run Summary" in markdown
    assert "- Views: full_trajectory, response_window" in markdown
