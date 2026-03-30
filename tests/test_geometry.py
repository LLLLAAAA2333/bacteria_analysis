import pytest

from bacteria_analysis.geometry import build_rdm_matrix, summarize_grouped_stimulus_pairs


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
