import json

import bacteria_analysis.geometry as geometry_module
import bacteria_analysis.geometry_outputs as geometry_outputs_module
import pandas as pd
import pytest

from bacteria_analysis.geometry import (
    build_rdm_matrix,
    build_rdm_group_coverage,
    build_stimulus_name_map,
    build_stimulus_overlap_matrix,
    extract_upper_triangle,
    score_rdm_similarity,
    summarize_grouped_stimulus_pairs,
    summarize_rdm_stability,
)
from bacteria_analysis.geometry_outputs import (
    _build_overlap_heatmap_frame,
    _build_rdm_heatmap_frame,
    _build_similarity_plot_panels,
    _cluster_reorder_heatmap_frame,
    ensure_geometry_output_dirs,
    write_geometry_outputs,
)


def test_geometry_aliases_remain_available():
    assert geometry_module.parse_stage2_views is geometry_module.parse_geometry_views
    assert geometry_outputs_module.ensure_stage2_output_dirs is geometry_outputs_module.ensure_geometry_output_dirs
    assert geometry_outputs_module.write_stage2_outputs is geometry_outputs_module.write_geometry_outputs


def test_individual_grouped_stimulus_pair_summary_excludes_cross_individual_pairs(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="individual",
    )

    assert set(summary["group_id"]) == {"2026-01-01__worm_a", "2026-01-02__worm_b"}
    assert (summary["group_type"] == "individual").all()
    assert summary["view_name"].eq("response_window").all()
    assert summary["pair_count"].sum() == 3
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
    assert summary["pair_count"].sum() == 1


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
    assert summary["pair_count"].sum() == 4

    support = summary.set_index(["stimulus_left", "stimulus_right"])
    assert support.loc[("b1_1", "b1_1"), "stimulus_left_trial_count"] == 2
    assert support.loc[("b1_1", "b1_1"), "stimulus_right_trial_count"] == 2
    assert support.loc[("b1_1", "b2_1"), "pair_count"] == 2
    assert support.loc[("b1_1", "b2_1"), "stimulus_left_trial_count"] == 1
    assert support.loc[("b1_1", "b2_1"), "stimulus_right_trial_count"] == 2
    assert support.loc[("b2_1", "b2_1"), "stimulus_left_trial_count"] == 2
    assert support.loc[("b2_1", "b2_1"), "stimulus_right_trial_count"] == 2


def test_build_rdm_group_coverage_surfaces_present_sparse_and_invalid_groups():
    metadata = pd.DataFrame.from_records(
        [
            {"trial_id": "t1", "individual_id": "ind_ok", "date": "2026-01-01"},
            {"trial_id": "t2", "individual_id": "ind_ok", "date": "2026-01-01"},
            {"trial_id": "t3", "individual_id": "ind_single", "date": "2026-01-01"},
            {"trial_id": "t4", "individual_id": "ind_invalid", "date": "2026-01-02"},
            {"trial_id": "t5", "individual_id": "ind_invalid", "date": "2026-01-02"},
        ]
    )
    comparisons = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "trial_id_a": "t1",
                "trial_id_b": "t2",
                "individual_id_a": "ind_ok",
                "individual_id_b": "ind_ok",
                "date_a": "2026-01-01",
                "date_b": "2026-01-01",
                "same_individual": True,
                "same_date": True,
                "comparison_status": "ok",
            },
            {
                "view_name": "response_window",
                "trial_id_a": "t1",
                "trial_id_b": "t3",
                "individual_id_a": "ind_ok",
                "individual_id_b": "ind_single",
                "date_a": "2026-01-01",
                "date_b": "2026-01-01",
                "same_individual": False,
                "same_date": True,
                "comparison_status": "ok",
            },
            {
                "view_name": "response_window",
                "trial_id_a": "t2",
                "trial_id_b": "t3",
                "individual_id_a": "ind_ok",
                "individual_id_b": "ind_single",
                "date_a": "2026-01-01",
                "date_b": "2026-01-01",
                "same_individual": False,
                "same_date": True,
                "comparison_status": "ok",
            },
            {
                "view_name": "response_window",
                "trial_id_a": "t4",
                "trial_id_b": "t5",
                "individual_id_a": "ind_invalid",
                "individual_id_b": "ind_invalid",
                "date_a": "2026-01-02",
                "date_b": "2026-01-02",
                "same_individual": True,
                "same_date": True,
                "comparison_status": "insufficient_overlap_neurons",
            },
        ]
    )

    coverage = build_rdm_group_coverage(metadata, comparisons, view_names=["response_window"])

    individual = coverage.loc[coverage["group_type"] == "individual"].set_index("group_id")
    assert individual.loc["ind_ok", "metadata_trial_count"] == 2
    assert individual.loc["ind_ok", "same_group_pair_count"] == 1
    assert individual.loc["ind_ok", "valid_same_group_pair_count"] == 1
    assert individual.loc["ind_single", "metadata_trial_count"] == 1
    assert individual.loc["ind_single", "same_group_pair_count"] == 0
    assert individual.loc["ind_single", "valid_same_group_pair_count"] == 0
    assert individual.loc["ind_invalid", "metadata_trial_count"] == 2
    assert individual.loc["ind_invalid", "same_group_pair_count"] == 1
    assert individual.loc["ind_invalid", "valid_same_group_pair_count"] == 0

    date = coverage.loc[coverage["group_type"] == "date"].set_index("group_id")
    assert date.loc["2026-01-01", "metadata_trial_count"] == 3
    assert date.loc["2026-01-01", "same_group_pair_count"] == 3
    assert date.loc["2026-01-01", "valid_same_group_pair_count"] == 3
    assert date.loc["2026-01-02", "metadata_trial_count"] == 2
    assert date.loc["2026-01-02", "same_group_pair_count"] == 1
    assert date.loc["2026-01-02", "valid_same_group_pair_count"] == 0


def test_build_stimulus_name_map_requires_one_name_per_stimulus():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b1_1", "stim_name": "A001 stationary"},
            {"stimulus": "b1_1", "stim_name": "A001 stationary"},
            {"stimulus": "b2_1", "stim_name": "A002 stationary"},
        ]
    )

    mapping = build_stimulus_name_map(metadata)

    assert mapping == {
        "b1_1": "A001 stationary",
        "b2_1": "A002 stationary",
    }


def test_build_stimulus_name_map_rejects_conflicting_names_for_one_stimulus():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b5_2", "stim_name": "A137 stationary"},
            {"stimulus": "b5_2", "stim_name": "A138 stationary"},
        ]
    )

    with pytest.raises(ValueError, match="stimulus to stim_name mapping must be one-to-one"):
        build_stimulus_name_map(metadata)


def test_build_stimulus_overlap_matrix_for_dates_is_binary_and_date_ordered():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "A002 stationary", "date": "2026-01-02", "individual_id": "2026-01-02__w2"},
            {"stimulus": "b1_1", "stim_name": "A001 stationary", "date": "2026-01-01", "individual_id": "2026-01-01__w1"},
            {"stimulus": "b2_1", "stim_name": "A002 stationary", "date": "2026-01-01", "individual_id": "2026-01-01__w1"},
        ]
    )

    matrix = build_stimulus_overlap_matrix(metadata, group_type="date", stimulus_name_map=build_stimulus_name_map(metadata))

    assert matrix.columns.tolist() == ["group_id", "b1_1", "b2_1"]
    assert matrix["group_id"].tolist() == ["2026-01-01", "2026-01-02"]
    assert matrix.loc[0, ["b1_1", "b2_1"]].tolist() == [1, 1]
    assert matrix.loc[1, ["b1_1", "b2_1"]].tolist() == [0, 1]
    assert matrix.attrs["group_axis_label"] == "Date"
    assert matrix.attrs["group_display_labels"] == {
        "2026-01-01": "2026-01-01 (2)",
        "2026-01-02": "2026-01-02 (1)",
    }


def test_build_stimulus_overlap_matrix_for_individuals_is_date_then_individual_ordered():
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "A002 stationary", "date": "2026-01-02", "individual_id": "2026-01-02__w2"},
            {"stimulus": "b1_1", "stim_name": "A001 stationary", "date": "2026-01-01", "individual_id": "2026-01-01__w2"},
            {"stimulus": "b1_1", "stim_name": "A001 stationary", "date": "2026-01-01", "individual_id": "2026-01-01__w1"},
        ]
    )

    matrix = build_stimulus_overlap_matrix(
        metadata,
        group_type="individual",
        stimulus_name_map=build_stimulus_name_map(metadata),
    )

    assert matrix["group_id"].tolist() == ["2026-01-01__w1", "2026-01-01__w2", "2026-01-02__w2"]
    assert matrix.columns.tolist() == ["group_id", "b1_1", "b2_1"]
    assert matrix.loc[0, ["b1_1", "b2_1"]].tolist() == [1, 0]
    assert matrix.loc[2, ["b1_1", "b2_1"]].tolist() == [0, 1]
    assert matrix.attrs["group_axis_label"] == "Individual"


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


def test_summarize_rdm_stability_marks_sparse_pooled_vs_group_scores_invalid():
    pair_summary = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 3,
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
                "n_pairs": 3,
                "mean_distance": 0.6,
                "median_distance": 0.6,
            },
            {
                "view_name": "response_window",
                "group_type": "pooled",
                "group_id": "pooled",
                "stimulus_left": "s2",
                "stimulus_right": "s3",
                "same_stimulus": False,
                "n_pairs": 3,
                "mean_distance": 0.4,
                "median_distance": 0.4,
            },
            {
                "view_name": "response_window",
                "group_type": "individual",
                "group_id": "ind_sparse",
                "stimulus_left": "s1",
                "stimulus_right": "s2",
                "same_stimulus": False,
                "n_pairs": 1,
                "mean_distance": 0.3,
                "median_distance": 0.3,
            },
        ]
    )

    result = summarize_rdm_stability(pair_summary)

    assert len(result) == 1
    assert result.loc[0, "comparison_scope"] == "pooled_vs_group"
    assert result.loc[0, "group_type"] == "individual"
    assert result.loc[0, "group_id"] == "ind_sparse"
    assert result.loc[0, "reference_group_id"] == "pooled"
    assert result.loc[0, "score_status"] == "invalid"
    assert result.loc[0, "n_shared_entries"] == 1
    assert pd.isna(result.loc[0, "similarity"])


def test_build_similarity_plot_panels_keeps_scopes_separate():
    frame = pd.DataFrame.from_records(
        [
            {
                "comparison_scope": "within_group_type",
                "view_name": "response_window",
                "reference_view_name": "response_window",
                "similarity": 0.9,
            },
            {
                "comparison_scope": "within_group_type",
                "view_name": "response_window",
                "reference_view_name": "response_window",
                "similarity": 0.1,
            },
            {
                "comparison_scope": "pooled_vs_group",
                "view_name": "response_window",
                "reference_view_name": "response_window",
                "similarity": 0.3,
            },
            {
                "comparison_scope": "pooled_vs_group",
                "view_name": "full_trajectory",
                "reference_view_name": "full_trajectory",
                "similarity": 0.5,
            },
        ]
    )

    panels = _build_similarity_plot_panels(frame)

    assert [panel["comparison_scope"] for panel in panels] == ["within_group_type", "pooled_vs_group"]
    within = panels[0]["frame"]
    pooled = panels[1]["frame"]
    assert within.to_dict("records") == [
        {
            "comparison_scope": "within_group_type",
            "view_label": "response_window",
            "similarity": 0.5,
        }
    ]
    assert pooled.to_dict("records") == [
        {
            "comparison_scope": "pooled_vs_group",
            "view_label": "response_window",
            "similarity": 0.3,
        },
        {
            "comparison_scope": "pooled_vs_group",
            "view_label": "full_trajectory",
            "similarity": 0.5,
        },
    ]


def test_build_rdm_heatmap_frame_uses_stim_name_labels():
    matrix = pd.DataFrame(
        {
            "stimulus_row": ["b1_1", "b2_1", "b3_1"],
            "b1_1": [0.0, 0.2, 0.8],
            "b2_1": [0.2, 0.0, 0.7],
            "b3_1": [0.8, 0.7, 0.0],
        }
    )
    matrix.attrs["stimulus_name_map"] = {
        "b1_1": "A001 stationary",
        "b2_1": "A002 stationary",
        "b3_1": "A003 stationary",
    }

    result = _build_rdm_heatmap_frame(matrix)

    assert result.index.tolist() == ["A001 stationary", "A002 stationary", "A003 stationary"]
    assert result.columns.tolist() == ["A001 stationary", "A002 stationary", "A003 stationary"]


def test_build_rdm_heatmap_frame_rejects_duplicate_stim_name_labels():
    matrix = pd.DataFrame(
        {
            "stimulus_row": ["b22_0", "b5_2"],
            "b22_0": [0.0, 0.4],
            "b5_2": [0.4, 0.0],
        }
    )
    matrix.attrs["stimulus_name_map"] = {
        "b22_0": "A137 stationary",
        "b5_2": "A137 stationary",
    }

    with pytest.raises(ValueError, match="stim_name labels must be unique"):
        _build_rdm_heatmap_frame(matrix)


def test_build_overlap_heatmap_frame_uses_stim_name_and_group_display_labels():
    matrix = pd.DataFrame(
        {
            "group_id": ["2026-01-01", "2026-01-02"],
            "b1_1": [1, 0],
            "b2_1": [1, 1],
        }
    )
    matrix.attrs["stimulus_name_map"] = {
        "b1_1": "A001 stationary",
        "b2_1": "A002 stationary",
    }
    matrix.attrs["group_display_labels"] = {
        "2026-01-01": "2026-01-01 (2)",
        "2026-01-02": "2026-01-02 (1)",
    }

    result = _build_overlap_heatmap_frame(matrix)

    assert result.index.tolist() == ["2026-01-01 (2)", "2026-01-02 (1)"]
    assert result.columns.tolist() == ["A001 stationary", "A002 stationary"]


def test_cluster_reorder_heatmap_frame_keeps_closest_pair_adjacent():
    heatmap_frame = pd.DataFrame(
        [
            [0.0, 0.1, 0.9, 0.8],
            [0.1, 0.0, 0.85, 0.75],
            [0.9, 0.85, 0.0, 0.2],
            [0.8, 0.75, 0.2, 0.0],
        ],
        index=["A001 stationary", "A002 stationary", "A003 stationary", "A004 stationary"],
        columns=["A001 stationary", "A002 stationary", "A003 stationary", "A004 stationary"],
    )

    result = _cluster_reorder_heatmap_frame(heatmap_frame)
    ordered = result.index.tolist()

    assert set(ordered) == set(heatmap_frame.index.tolist())
    assert abs(ordered.index("A001 stationary") - ordered.index("A002 stationary")) == 1
    assert abs(ordered.index("A003 stationary") - ordered.index("A004 stationary")) == 1


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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 2,
                    "stimulus_right_trial_count": 2,
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
                    "pair_count": 2,
                    "n_pairs": 2,
                    "stimulus_left_trial_count": 1,
                    "stimulus_right_trial_count": 2,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 2,
                    "stimulus_right_trial_count": 2,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 1,
                    "stimulus_right_trial_count": 1,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 1,
                    "stimulus_right_trial_count": 1,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 2,
                    "stimulus_right_trial_count": 2,
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
                    "pair_count": 2,
                    "n_pairs": 2,
                    "stimulus_left_trial_count": 1,
                    "stimulus_right_trial_count": 2,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 2,
                    "stimulus_right_trial_count": 2,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 1,
                    "stimulus_right_trial_count": 1,
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
                    "pair_count": 1,
                    "n_pairs": 1,
                    "stimulus_left_trial_count": 1,
                    "stimulus_right_trial_count": 1,
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
                {
                    "view_name": "response_window",
                    "group_type": "individual",
                    "group_id": "2026-01-01__worm_a",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 1,
                    "valid_same_group_pair_count": 1,
                },
                {
                    "view_name": "response_window",
                    "group_type": "individual",
                    "group_id": "2026-01-02__worm_b",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 1,
                    "valid_same_group_pair_count": 1,
                },
                {
                    "view_name": "response_window",
                    "group_type": "date",
                    "group_id": "2026-01-01",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 1,
                    "valid_same_group_pair_count": 1,
                },
                {
                    "view_name": "response_window",
                    "group_type": "date",
                    "group_id": "2026-01-02",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 0,
                    "valid_same_group_pair_count": 0,
                },
                {
                    "view_name": "full_trajectory",
                    "group_type": "individual",
                    "group_id": "2026-01-01__worm_a",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 1,
                    "valid_same_group_pair_count": 1,
                },
                {
                    "view_name": "full_trajectory",
                    "group_type": "individual",
                    "group_id": "2026-01-02__worm_b",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 0,
                    "valid_same_group_pair_count": 0,
                },
                {
                    "view_name": "full_trajectory",
                    "group_type": "date",
                    "group_id": "2026-01-01",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 1,
                    "valid_same_group_pair_count": 1,
                },
                {
                    "view_name": "full_trajectory",
                    "group_type": "date",
                    "group_id": "2026-01-02",
                    "metadata_trial_count": 2,
                    "same_group_pair_count": 0,
                    "valid_same_group_pair_count": 0,
                },
            ]
        ),
        "stimulus_overlap__date": pd.DataFrame(
            {
                "group_id": ["2026-01-01", "2026-01-02"],
                "b1_1": [1, 0],
                "b2_1": [1, 1],
            }
        ),
        "stimulus_overlap__individual": pd.DataFrame(
            {
                "group_id": ["2026-01-01__worm_a", "2026-01-02__worm_b"],
                "b1_1": [1, 0],
                "b2_1": [1, 1],
            }
        ),
    }
    stimulus_name_map = {
        "b1_1": "A001 stationary",
        "b2_1": "A002 stationary",
    }
    outputs["rdm_matrix__response_window__pooled"].attrs["stimulus_name_map"] = stimulus_name_map
    outputs["rdm_matrix__full_trajectory__pooled"].attrs["stimulus_name_map"] = stimulus_name_map
    outputs["stimulus_overlap__date"].attrs["stimulus_name_map"] = stimulus_name_map
    outputs["stimulus_overlap__date"].attrs["group_axis_label"] = "Date"
    outputs["stimulus_overlap__date"].attrs["group_display_labels"] = {
        "2026-01-01": "2026-01-01 (2)",
        "2026-01-02": "2026-01-02 (1)",
    }
    outputs["stimulus_overlap__individual"].attrs["stimulus_name_map"] = stimulus_name_map
    outputs["stimulus_overlap__individual"].attrs["group_axis_label"] = "Individual"
    outputs["stimulus_overlap__individual"].attrs["group_display_labels"] = {
        "2026-01-01__worm_a": "2026-01-01__worm_a (2)",
        "2026-01-02__worm_b": "2026-01-02__worm_b (1)",
    }
    return outputs


def test_ensure_geometry_output_dirs_creates_expected_tree(tmp_path):
    dirs = ensure_geometry_output_dirs(tmp_path / "geometry")

    assert dirs["output_root"].exists()
    assert dirs["tables_dir"].exists()
    assert dirs["figures_dir"].exists()
    assert dirs["qc_dir"].exists()


def test_write_geometry_outputs_writes_required_tables(tmp_path, synthetic_geometry_outputs):
    written = write_geometry_outputs(synthetic_geometry_outputs, tmp_path / "geometry")

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
    assert (written["qc_dir"] / "stimulus_overlap__date.parquet").exists()
    assert (written["qc_dir"] / "stimulus_overlap__individual.parquet").exists()
    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled.png").exists()
    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled__clustered.png").exists()
    assert (written["figures_dir"] / "rdm_matrix__full_trajectory__pooled.png").exists()
    assert (written["figures_dir"] / "rdm_matrix__full_trajectory__pooled__clustered.png").exists()
    assert (written["figures_dir"] / "rdm_stability_by_individual.png").exists()
    assert (written["figures_dir"] / "rdm_stability_by_date.png").exists()
    assert (written["figures_dir"] / "rdm_view_comparison.png").exists()
    assert (written["figures_dir"] / "stimulus_overlap__date.png").exists()
    assert (written["figures_dir"] / "stimulus_overlap__individual.png").exists()


def test_write_geometry_outputs_skips_non_pooled_matrix_parquet_artifacts(tmp_path, synthetic_geometry_outputs):
    written = write_geometry_outputs(synthetic_geometry_outputs, tmp_path / "geometry")

    assert "rdm_matrix__response_window__individual" not in written
    assert not (written["tables_dir"] / "rdm_matrix__response_window__individual.parquet").exists()
    assert (written["tables_dir"] / "rdm_matrix__response_window__pooled.parquet").exists()


def test_write_geometry_outputs_handles_empty_pooled_matrix_frames(tmp_path, synthetic_geometry_outputs):
    outputs = dict(synthetic_geometry_outputs)
    outputs["rdm_matrix__response_window__pooled"] = pd.DataFrame(columns=["stimulus_row"])

    written = write_geometry_outputs(outputs, tmp_path / "geometry")

    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled.png").exists()
    assert (written["figures_dir"] / "rdm_matrix__response_window__pooled__clustered.png").exists()
    assert (written["output_root"] / "run_summary.json").exists()

    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_matrix_views"] == ["response_window", "full_trajectory"]


def test_write_geometry_outputs_excludes_non_pooled_views_from_pooled_matrix_summary(tmp_path, synthetic_geometry_outputs):
    outputs = dict(synthetic_geometry_outputs)
    outputs["rdm_matrix__novel_view__individual"] = pd.DataFrame(
        {
            "stimulus_row": ["b1_1", "b2_1"],
            "b1_1": [0.12, 0.72],
            "b2_1": [0.72, 0.18],
        }
    )

    written = write_geometry_outputs(outputs, tmp_path / "geometry")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["pooled_matrix_views"] == ["response_window", "full_trajectory"]
    assert "novel_view" not in summary["pooled_matrix_views"]


def test_write_geometry_outputs_records_only_actual_included_views(tmp_path, synthetic_geometry_outputs):
    outputs = {
        "rdm_pairs__full_trajectory__pooled": synthetic_geometry_outputs["rdm_pairs__full_trajectory__pooled"],
        "rdm_pairs__full_trajectory__individual": synthetic_geometry_outputs["rdm_pairs__full_trajectory__individual"],
        "rdm_pairs__full_trajectory__date": synthetic_geometry_outputs["rdm_pairs__full_trajectory__date"],
        "rdm_matrix__full_trajectory__pooled": synthetic_geometry_outputs["rdm_matrix__full_trajectory__pooled"],
        "rdm_stability_by_individual": synthetic_geometry_outputs["rdm_stability_by_individual"].loc[
            lambda frame: frame["view_name"] == "full_trajectory"
        ],
        "rdm_stability_by_date": pd.DataFrame(columns=synthetic_geometry_outputs["rdm_stability_by_date"].columns),
        "rdm_view_comparison": pd.DataFrame(columns=synthetic_geometry_outputs["rdm_view_comparison"].columns),
        "rdm_group_coverage": synthetic_geometry_outputs["rdm_group_coverage"].loc[
            lambda frame: frame["view_name"] == "full_trajectory"
        ],
        "stimulus_overlap__date": synthetic_geometry_outputs["stimulus_overlap__date"],
        "stimulus_overlap__individual": synthetic_geometry_outputs["stimulus_overlap__individual"],
    }

    written = write_geometry_outputs(outputs, tmp_path / "geometry")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["views"] == ["full_trajectory"]
    assert summary["pooled_matrix_views"] == ["full_trajectory"]
    assert summary["pair_table_names"] == [
        "rdm_pairs__full_trajectory__date",
        "rdm_pairs__full_trajectory__individual",
        "rdm_pairs__full_trajectory__pooled",
    ]


def test_write_geometry_outputs_writes_run_summary(tmp_path, synthetic_geometry_outputs):
    written = write_geometry_outputs(synthetic_geometry_outputs, tmp_path / "geometry")

    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["views"] == ["response_window", "full_trajectory"]
    assert summary["pooled_matrix_views"] == ["response_window", "full_trajectory"]
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
    assert summary["qc_table_names"] == [
        "rdm_group_coverage",
        "stimulus_overlap__date",
        "stimulus_overlap__individual",
    ]
    assert summary["tables_dir"].endswith("geometry\\tables")
    assert summary["figures_dir"].endswith("geometry\\figures")

    markdown = (written["output_root"] / "run_summary.md").read_text(encoding="utf-8")
    assert "# Geometry Analysis Run Summary" in markdown
    assert "- Views: response_window, full_trajectory" in markdown

