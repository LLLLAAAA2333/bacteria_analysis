from bacteria_analysis.constants import BASELINE_TIMEPOINTS, EXPECTED_TIMEPOINTS, NEURON_ORDER


def test_expected_timepoints_cover_full_window():
    assert EXPECTED_TIMEPOINTS == tuple(range(45))


def test_baseline_window_matches_spec():
    assert BASELINE_TIMEPOINTS == (0, 1, 2, 3, 4, 5)


def test_neuron_order_has_22_entries():
    assert len(NEURON_ORDER) == 22
