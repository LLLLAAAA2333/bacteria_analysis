from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER


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
