import pandas as pd
import pytest

from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER, REQUIRED_COLUMNS
from bacteria_analysis.preprocessing import (
    add_trial_id,
    annotate_trace_quality,
    center_by_baseline,
    filter_traces,
    validate_input_dataframe,
)


@pytest.fixture
def sample_df():
    rows = []
    traces = (
        {
            "neuron": "ADFL",
            "stimulus": "b1_1",
            "worm_key": "worm_001",
            "segment_index": 0,
            "date": "2026-03-27",
            "stim_name": "Bacteria 1",
            "stim_color": "#1f77b4",
            "values": [float(time_point) for time_point in EXPECTED_TIMEPOINTS],
        },
        {
            "neuron": "ADFR",
            "stimulus": "b1_1",
            "worm_key": "worm_001",
            "segment_index": 0,
            "date": "2026-03-27",
            "stim_name": "Bacteria 1",
            "stim_color": "#1f77b4",
            "values": [float("nan") for _ in EXPECTED_TIMEPOINTS],
        },
        {
            "neuron": "ASEL",
            "stimulus": "b2_1",
            "worm_key": "worm_002",
            "segment_index": 1,
            "date": "2026-03-27",
            "stim_name": "Bacteria 2",
            "stim_color": "#ff7f0e",
            "values": [
                float("nan") if time_point in {0, 1, 2, 10, 11} else (time_point + 5) / 20
                for time_point in EXPECTED_TIMEPOINTS
            ],
        },
    )

    for trace in traces:
        for time_point, value in zip(EXPECTED_TIMEPOINTS, trace["values"], strict=True):
            rows.append(
                {
                    "neuron": trace["neuron"],
                    "stimulus": trace["stimulus"],
                    "time_point": time_point,
                    "delta_F_over_F0": value,
                    "worm_key": trace["worm_key"],
                    "segment_index": trace["segment_index"],
                    "date": trace["date"],
                    "stim_name": trace["stim_name"],
                    "stim_color": trace["stim_color"],
                }
            )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


@pytest.fixture
def sample_df_missing_baseline(sample_df):
    broken = sample_df.copy()
    missing_baseline_mask = (broken["neuron"] == "ASEL") & broken["time_point"].isin(BASELINE_TIMEPOINTS)
    broken.loc[missing_baseline_mask, "delta_F_over_F0"] = float("nan")
    return broken


@pytest.fixture
def same_trial_dual_stimulus_df():
    rows = []
    for stimulus, values in (
        ("b1_1", [1.0 for _ in EXPECTED_TIMEPOINTS]),
        ("b1_2", [float("nan") for _ in EXPECTED_TIMEPOINTS]),
    ):
        for time_point, value in zip(EXPECTED_TIMEPOINTS, values, strict=True):
            rows.append(
                {
                    "neuron": "ADFL",
                    "stimulus": stimulus,
                    "time_point": time_point,
                    "delta_F_over_F0": value,
                    "worm_key": "worm_003",
                    "segment_index": 2,
                    "date": "2026-03-27",
                    "stim_name": "Bacteria 1",
                    "stim_color": "#1f77b4",
                }
            )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_expected_timepoints_cover_full_window():
    assert EXPECTED_TIMEPOINTS == tuple(range(45))


def test_baseline_window_matches_spec():
    assert BASELINE_TIMEPOINTS == (0, 1, 2, 3, 4, 5)


def test_neuron_order_matches_canonical_tuple():
    assert NEURON_ORDER == (
        "ADFL",
        "ADFR",
        "ADLL",
        "ADLR",
        "ASEL",
        "ASER",
        "ASGL",
        "ASGR",
        "ASHL",
        "ASHR",
        "ASIL",
        "ASIR",
        "ASJL",
        "ASJR",
        "ASKL",
        "ASKR",
        "AWAL",
        "AWAR",
        "AWBL",
        "AWBR",
        "AWCOFF",
        "AWCON",
    )


def test_synthetic_fixture_has_exactly_two_trials(synthetic_neuron_segments_df):
    trials = synthetic_neuron_segments_df[["worm_key", "segment_index", "date"]].drop_duplicates()
    assert len(trials) == 2


def test_synthetic_fixture_traces_cover_full_time_grid(synthetic_neuron_segments_df):
    trace_keys = ["worm_key", "segment_index", "date", "neuron"]

    for _, trace in synthetic_neuron_segments_df.groupby(trace_keys, sort=False):
        assert tuple(trace["time_point"]) == EXPECTED_TIMEPOINTS


def test_synthetic_fixture_includes_complete_missing_and_partial_traces(
    synthetic_neuron_segments_df,
):
    trace_keys = ["worm_key", "segment_index", "date", "neuron"]
    nan_counts = sorted(
        trace["delta_F_over_F0"].isna().sum()
        for _, trace in synthetic_neuron_segments_df.groupby(trace_keys, sort=False)
    )

    assert nan_counts == [0, 5, len(EXPECTED_TIMEPOINTS)]


def test_add_trial_id_uses_date_worm_segment():
    frame = pd.DataFrame(
        [
            {
                "neuron": "ADFL",
                "stimulus": "b1_1",
                "time_point": 0,
                "delta_F_over_F0": 1.0,
                "worm_key": "wormA",
                "segment_index": 1,
                "date": "2026-01-06",
                "stim_name": "Bacteria 1",
                "stim_color": "#1f77b4",
            }
        ],
        columns=REQUIRED_COLUMNS,
    )

    out = add_trial_id(frame)
    assert out["trial_id"].iloc[0] == "20260106__wormA__1"


def test_validate_rejects_missing_required_columns(synthetic_neuron_segments_df):
    broken = synthetic_neuron_segments_df.drop(columns=["stimulus"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_input_dataframe(broken)


def test_validate_rejects_broken_time_grid(synthetic_neuron_segments_df):
    broken = synthetic_neuron_segments_df[synthetic_neuron_segments_df["time_point"] != 44]
    with pytest.raises(ValueError, match="45 unique time_point"):
        validate_input_dataframe(broken)


def test_validate_rejects_trials_with_multiple_stimuli(synthetic_neuron_segments_df):
    broken = synthetic_neuron_segments_df.copy()
    first_trial_mask = (
        (broken["worm_key"] == "worm_001")
        & (broken["segment_index"] == 0)
        & (broken["date"] == "2026-03-27")
    )
    broken.loc[first_trial_mask & (broken["neuron"] == "ADFL") & (broken["time_point"] == 0), "stimulus"] = "b1_2"

    with pytest.raises(ValueError, match="exactly one stimulus"):
        validate_input_dataframe(broken)


def test_validate_rejects_trial_neuron_with_too_few_rows(synthetic_neuron_segments_df):
    broken = synthetic_neuron_segments_df[
        ~(
            (synthetic_neuron_segments_df["worm_key"] == "worm_001")
            & (synthetic_neuron_segments_df["segment_index"] == 0)
            & (synthetic_neuron_segments_df["date"] == "2026-03-27")
            & (synthetic_neuron_segments_df["neuron"] == "ADFL")
            & (synthetic_neuron_segments_df["time_point"] == 44)
        )
    ]

    with pytest.raises(ValueError, match="45 rows"):
        validate_input_dataframe(broken)


def test_validate_rejects_trial_neuron_with_duplicate_time_points(synthetic_neuron_segments_df):
    broken = synthetic_neuron_segments_df.copy()
    target_mask = (
        (broken["worm_key"] == "worm_001")
        & (broken["segment_index"] == 0)
        & (broken["date"] == "2026-03-27")
        & (broken["neuron"] == "ADFL")
    )
    broken.loc[target_mask & (broken["time_point"] == 44), "time_point"] = 43

    with pytest.raises(ValueError, match="45 unique time_point"):
        validate_input_dataframe(broken)


def test_full_nan_trace_is_flagged(sample_df):
    annotated = annotate_trace_quality(sample_df)
    assert annotated.loc[annotated["neuron"] == "ADFR", "is_all_nan_trace"].all()


def test_trace_quality_counts_are_recorded(sample_df):
    annotated = annotate_trace_quality(sample_df)

    adfl = annotated[annotated["neuron"] == "ADFL"]
    asel = annotated[annotated["neuron"] == "ASEL"]

    assert adfl["n_valid_points"].iloc[0] == 45
    assert not adfl["has_any_nan_trace"].iloc[0]
    assert adfl["n_valid_baseline_points"].iloc[0] == 6
    assert asel["n_valid_points"].iloc[0] == 40
    assert asel["has_any_nan_trace"].iloc[0]
    assert asel["n_valid_baseline_points"].iloc[0] == 3


def test_filter_drops_fully_nan_trace_only(sample_df):
    annotated = annotate_trace_quality(sample_df)
    filtered = filter_traces(annotated)
    assert "ADFR" not in filtered["neuron"].unique()
    assert "ADFL" in filtered["neuron"].unique()


def test_baseline_centering_subtracts_trace_baseline(sample_df):
    centered = center_by_baseline(filter_traces(annotate_trace_quality(sample_df)))
    trace = centered[centered["neuron"] == "ADFL"].sort_values("time_point")
    assert trace["baseline_mean"].iloc[0] == pytest.approx(2.5)
    assert trace.loc[trace["time_point"] == 0, "dff_baseline_centered"].iloc[0] == pytest.approx(-2.5)


def test_partial_baseline_uses_available_values(sample_df):
    centered = center_by_baseline(filter_traces(annotate_trace_quality(sample_df)))
    trace = centered[centered["neuron"] == "ASEL"].sort_values("time_point")
    assert trace["baseline_valid"].all()
    assert trace["baseline_mean"].iloc[0] == pytest.approx(0.45)
    assert trace.loc[trace["time_point"] == 3, "dff_baseline_centered"].iloc[0] == pytest.approx(-0.05)


def test_dff_baseline_centered_is_nan_when_baseline_invalid(sample_df_missing_baseline):
    centered = center_by_baseline(filter_traces(annotate_trace_quality(sample_df_missing_baseline)))
    trace = centered[centered["neuron"] == "ASEL"]
    assert not trace["baseline_valid"].any()
    assert trace["dff_baseline_centered"].isna().all()


def test_missing_baseline_marks_baseline_invalid(sample_df_missing_baseline):
    centered = center_by_baseline(filter_traces(annotate_trace_quality(sample_df_missing_baseline)))
    assert not centered["baseline_valid"].all()


def test_stimulus_is_part_of_trace_grouping(same_trial_dual_stimulus_df):
    annotated = annotate_trace_quality(same_trial_dual_stimulus_df)
    grouped = annotated.groupby("stimulus", sort=False)

    assert grouped["is_all_nan_trace"].first().to_dict() == {"b1_1": False, "b1_2": True}
    assert grouped["n_valid_points"].first().to_dict() == {"b1_1": 45, "b1_2": 0}
