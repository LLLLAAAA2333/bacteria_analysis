import pandas as pd
import pytest

from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER, REQUIRED_COLUMNS
from bacteria_analysis.preprocessing import add_trial_id, validate_input_dataframe


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
