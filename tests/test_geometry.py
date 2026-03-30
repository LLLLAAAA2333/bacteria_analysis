import pytest

from bacteria_analysis.geometry import summarize_grouped_stimulus_pairs


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
        view_name="response_window",
        group_type="date",
    )

    assert set(summary["group_id"]) == {"2026-01-01", "2026-01-02"}
    assert (summary["group_type"] == "date").all()
    assert summary["view_name"].eq("response_window").all()
    assert summary["n_pairs"].sum() == 3


def test_pooled_grouped_stimulus_pair_summary_uses_single_group(synthetic_geometry_comparisons):
    summary = summarize_grouped_stimulus_pairs(
        synthetic_geometry_comparisons,
        view_name="response_window",
        group_type="pooled",
    )

    assert len(summary) == 1
    assert summary["group_id"].tolist() == ["pooled"]
    assert summary["group_type"].tolist() == ["pooled"]
    assert summary["view_name"].tolist() == ["response_window"]
    assert summary["n_pairs"].iat[0] == 4


def test_invalid_group_type_raises_value_error(synthetic_geometry_comparisons):
    with pytest.raises(ValueError, match="unsupported group_type: invalid"):
        summarize_grouped_stimulus_pairs(
            synthetic_geometry_comparisons,
            view_name="response_window",
            group_type="invalid",
        )
