import pytest
import pandas as pd

from bacteria_analysis import reliability_outputs


def test_compute_quantile_axis_limits_uses_central_window_with_padding():
    values = pd.Series([0.1, 0.2, 0.3, 0.4, 5.0])

    lower, upper = reliability_outputs._compute_quantile_axis_limits(values, lower_quantile=0.2, upper_quantile=0.8)

    assert lower == pytest.approx(0.0888)
    assert upper == pytest.approx(1.4112)


def test_build_focus_view_same_vs_different_plot_frame_filters_and_labels():
    comparisons = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "distance": 0.20,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "distance": 0.80,
            },
            {
                "view_name": "response_window",
                "comparison_status": "excluded",
                "same_stimulus": True,
                "distance": 0.10,
            },
            {
                "view_name": "on_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "distance": 0.05,
            },
        ]
    )

    plot_frame = reliability_outputs._build_focus_view_same_vs_different_plot_frame(
        comparisons,
        focus_view="response_window",
    )

    assert list(plot_frame["comparison_label"].astype(str)) == ["same", "different"]
    assert list(plot_frame["comparison_label"].cat.categories) == ["same", "different"]
    assert plot_frame["distance"].tolist() == [0.20, 0.80]


def test_build_focus_view_same_vs_different_plot_frame_for_date_filters_to_within_date_pairs():
    comparisons = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "same_date": True,
                "date_a": "2026-03-27",
                "date_b": "2026-03-27",
                "distance": 0.20,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "same_date": True,
                "date_a": "2026-03-27",
                "date_b": "2026-03-27",
                "distance": 0.80,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "same_date": False,
                "date_a": "2026-03-27",
                "date_b": "2026-03-28",
                "distance": 0.95,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "same_date": True,
                "date_a": "2026-03-28",
                "date_b": "2026-03-28",
                "distance": 0.15,
            },
        ]
    )

    plot_frame = reliability_outputs._build_focus_view_same_vs_different_plot_frame_for_date(
        comparisons,
        focus_view="response_window",
        date_value="2026-03-27",
    )

    assert plot_frame["distance"].tolist() == [0.20, 0.80]
    assert plot_frame["date_a"].astype(str).tolist() == ["2026-03-27", "2026-03-27"]
    assert plot_frame["date_b"].astype(str).tolist() == ["2026-03-27", "2026-03-27"]
    assert list(plot_frame["comparison_label"].astype(str)) == ["same", "different"]


def test_sample_same_vs_different_points_caps_each_group():
    plot_frame = pd.DataFrame.from_records(
        [
            {"comparison_label": "same", "distance": index / 10}
            for index in range(5)
        ]
        + [
            {"comparison_label": "different", "distance": index / 10}
            for index in range(8)
        ]
    )
    plot_frame["comparison_label"] = pd.Categorical(
        plot_frame["comparison_label"],
        categories=["same", "different"],
        ordered=True,
    )

    sampled = reliability_outputs._sample_same_vs_different_points(
        plot_frame,
        max_points_per_group=3,
        random_state=7,
    )
    counts = sampled.groupby("comparison_label", observed=False).size()

    assert counts.loc["same"] == 3
    assert counts.loc["different"] == 3
    assert sampled["distance"].tolist() == pytest.approx([0.0, 0.3, 0.2, 0.2, 0.5, 0.0])


def test_compute_same_vs_different_center_gap_uses_group_medians():
    summary = pd.DataFrame(
        {
            "median": {"same": 0.42, "different": 0.71},
            "q1": {"same": 0.30, "different": 0.55},
            "q3": {"same": 0.60, "different": 0.88},
            "count": {"same": 10, "different": 12},
        }
    )

    center_gap = reliability_outputs._compute_same_vs_different_center_gap(summary, metric="median")

    assert center_gap == pytest.approx(0.29)


def test_build_focus_view_stimulus_gap_summary_aggregates_same_and_different_means():
    comparisons = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "stimulus_a": "b1_1",
                "stimulus_b": "b1_1",
                "distance": 0.20,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "stimulus_a": "b2_1",
                "stimulus_b": "b2_1",
                "distance": 0.40,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "stimulus_a": "b1_1",
                "stimulus_b": "b2_1",
                "distance": 0.80,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "stimulus_a": "b1_1",
                "stimulus_b": "b3_1",
                "distance": 0.90,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "stimulus_a": "b2_1",
                "stimulus_b": "b3_1",
                "distance": 0.70,
            },
            {
                "view_name": "full_trajectory",
                "comparison_status": "ok",
                "same_stimulus": True,
                "stimulus_a": "b1_1",
                "stimulus_b": "b1_1",
                "distance": 0.05,
            },
        ]
    )
    metadata = pd.DataFrame.from_records(
        [
            {"stimulus": "b1_1", "stim_name": "Stim 1", "stim_color": "#111111"},
            {"stimulus": "b2_1", "stim_name": "Stim 2", "stim_color": "#222222"},
            {"stimulus": "b3_1", "stim_name": "Stim 3", "stim_color": "#333333"},
        ]
    )

    summary = reliability_outputs._build_focus_view_stimulus_gap_summary(
        comparisons,
        metadata,
        focus_view="response_window",
    ).set_index("stimulus")

    assert summary.loc["b1_1", "same_count"] == 1
    assert summary.loc["b1_1", "same_mean_distance"] == pytest.approx(0.20)
    assert summary.loc["b1_1", "different_count"] == 2
    assert summary.loc["b1_1", "different_mean_distance"] == pytest.approx(0.85)
    assert summary.loc["b1_1", "distance_gap"] == pytest.approx(0.65)
    assert summary.loc["b1_1", "stim_color"] == "#111111"

    assert summary.loc["b3_1", "same_count"] == 0
    assert pd.isna(summary.loc["b3_1", "same_mean_distance"])
    assert summary.loc["b3_1", "different_count"] == 2
    assert summary.loc["b3_1", "different_mean_distance"] == pytest.approx(0.80)


def test_build_focus_view_stimulus_gap_summary_for_date_filters_to_within_date_pairs():
    comparisons = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "same_date": True,
                "date_a": "2026-03-27",
                "date_b": "2026-03-27",
                "stimulus_a": "b1_1",
                "stimulus_b": "b1_1",
                "distance": 0.20,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "same_date": True,
                "date_a": "2026-03-27",
                "date_b": "2026-03-27",
                "stimulus_a": "b1_1",
                "stimulus_b": "b2_1",
                "distance": 0.80,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": False,
                "same_date": False,
                "date_a": "2026-03-27",
                "date_b": "2026-03-28",
                "stimulus_a": "b1_1",
                "stimulus_b": "b3_1",
                "distance": 0.95,
            },
            {
                "view_name": "response_window",
                "comparison_status": "ok",
                "same_stimulus": True,
                "same_date": True,
                "date_a": "2026-03-28",
                "date_b": "2026-03-28",
                "stimulus_a": "b3_1",
                "stimulus_b": "b3_1",
                "distance": 0.30,
            },
        ]
    )
    metadata = pd.DataFrame.from_records(
        [
            {"date": "2026-03-27", "stimulus": "b1_1", "stim_name": "Stim 1", "stim_color": "#111111"},
            {"date": "2026-03-27", "stimulus": "b2_1", "stim_name": "Stim 2", "stim_color": "#222222"},
            {"date": "2026-03-28", "stimulus": "b3_1", "stim_name": "Stim 3", "stim_color": "#333333"},
        ]
    )

    summary = reliability_outputs._build_focus_view_stimulus_gap_summary(
        comparisons,
        metadata,
        focus_view="response_window",
        date_value="2026-03-27",
    ).set_index("stimulus")

    assert list(summary.index) == ["b1_1", "b2_1"]
    assert summary.loc["b1_1", "same_count"] == 1
    assert summary.loc["b1_1", "same_mean_distance"] == pytest.approx(0.20)
    assert summary.loc["b1_1", "different_count"] == 1
    assert summary.loc["b1_1", "different_mean_distance"] == pytest.approx(0.80)
    assert summary.loc["b2_1", "same_count"] == 0
    assert summary.loc["b2_1", "different_count"] == 1


def test_build_stimulus_availability_matrix_counts_unique_trials():
    metadata = pd.DataFrame.from_records(
        [
            {"date": "2026-03-11", "stimulus": "b1_1", "trial_id": "t1"},
            {"date": "2026-03-11", "stimulus": "b1_1", "trial_id": "t1"},
            {"date": "2026-03-11", "stimulus": "b2_1", "trial_id": "t2"},
            {"date": "2026-03-12", "stimulus": "b1_1", "trial_id": "t3"},
            {"date": "2026-03-12", "stimulus": "b3_1", "trial_id": "t4"},
        ]
    )

    matrix = reliability_outputs._build_stimulus_availability_matrix(metadata)

    assert list(matrix.index) == ["2026-03-11", "2026-03-12"]
    assert list(matrix.columns) == ["b1_1", "b2_1", "b3_1"]
    assert matrix.loc["2026-03-11", "b1_1"] == 1
    assert matrix.loc["2026-03-11", "b2_1"] == 1
    assert matrix.loc["2026-03-11", "b3_1"] == 0
    assert matrix.loc["2026-03-12", "b1_1"] == 1
    assert matrix.loc["2026-03-12", "b3_1"] == 1
