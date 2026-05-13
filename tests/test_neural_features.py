import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS
from bacteria_analysis.analyses.rdm.builders import build_neural_rdm
from bacteria_analysis.io import AnalysisDataset
from bacteria_analysis.features.neural import (
    build_stimulus_prototypes,
    build_trial_feature_matrix,
)


def _raw_frame(traces):
    rows = []
    for trace in traces:
        values = trace["values"]
        for time_point, value in zip(EXPECTED_TIMEPOINTS, values, strict=True):
            rows.append(
                {
                    "neuron": trace["neuron"],
                    "stimulus": trace["stimulus"],
                    "time_point": time_point,
                    "delta_F_over_F0": value,
                    "worm_key": trace["worm_key"],
                    "segment_index": trace["segment_index"],
                    "date": trace.get("date", "2026-03-27"),
                    "stim_name": trace.get("stim_name", trace["stimulus"]),
                    "stim_color": trace.get("stim_color", "#111111"),
                }
            )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _linear_trace(offset):
    return [float(offset + time_point) for time_point in EXPECTED_TIMEPOINTS]


def test_build_trial_feature_matrix_uses_raw_baseline_centering():
    raw = _raw_frame(
        [
            {
                "neuron": "ADFL",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": _linear_trace(10.0),
            }
        ]
    )

    features = build_trial_feature_matrix(raw, view="response_window", merge_lr=False)

    assert features.loc[0, "ADFL__t06"] == pytest.approx(3.5)
    assert "ADFL__t05" not in features.columns
    assert "ADFL__t20" in features.columns


def test_all_nan_traces_are_removed_before_feature_construction():
    raw = _raw_frame(
        [
            {
                "neuron": "ADFL",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [float("nan") for _ in EXPECTED_TIMEPOINTS],
            },
            {
                "neuron": "ADFR",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": _linear_trace(20.0),
            },
        ]
    )

    features = build_trial_feature_matrix(raw, view="response_window", merge_lr=False)

    assert np.isnan(features.loc[0, "ADFL__t06"])
    assert features.loc[0, "ADFR__t06"] == pytest.approx(3.5)


def test_non_ase_left_right_neurons_merge_by_mean():
    raw = _raw_frame(
        [
            {
                "neuron": "ADFL",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [10.0 if t < 6 else 20.0 for t in EXPECTED_TIMEPOINTS],
            },
            {
                "neuron": "ADFR",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [10.0 if t < 6 else 30.0 for t in EXPECTED_TIMEPOINTS],
            },
        ]
    )

    features = build_trial_feature_matrix(raw, view="response_window", merge_lr=True)

    assert features.loc[0, "ADF__t06"] == pytest.approx(15.0)
    assert "ADFL__t06" not in features.columns
    assert "ADFR__t06" not in features.columns


def test_asel_and_aser_remain_separate_when_merging_lr():
    raw = _raw_frame(
        [
            {
                "neuron": "ASEL",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [1.0 if t < 6 else 4.0 for t in EXPECTED_TIMEPOINTS],
            },
            {
                "neuron": "ASER",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [1.0 if t < 6 else 9.0 for t in EXPECTED_TIMEPOINTS],
            },
        ]
    )

    features = build_trial_feature_matrix(raw, view="response_window", merge_lr=True)

    assert features.loc[0, "ASEL__t06"] == pytest.approx(3.0)
    assert features.loc[0, "ASER__t06"] == pytest.approx(8.0)


def test_stimulus_prototypes_support_median_and_mean_aggregation():
    features = pd.DataFrame(
        {
            "trial_id": ["t1", "t2", "t3"],
            "stimulus": ["s1", "s1", "s2"],
            "ADF__t06": [1.0, 5.0, 10.0],
            "ADF__t07": [2.0, 8.0, 20.0],
        }
    )

    median = build_stimulus_prototypes(features, aggregation="median").set_index("stimulus")
    mean = build_stimulus_prototypes(features, aggregation="mean").set_index("stimulus")

    assert median.loc["s1", "ADF__t06"] == pytest.approx(3.0)
    assert mean.loc["s1", "ADF__t07"] == pytest.approx(5.0)
    assert median.loc["s2", "n_trials"] == 1


def test_views_select_expected_timepoints():
    raw = _raw_frame(
        [
            {
                "neuron": "ADFL",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": _linear_trace(0.0),
            }
        ]
    )

    response = build_trial_feature_matrix(raw, view="response_window", merge_lr=False)
    full = build_trial_feature_matrix(raw, view="full_trajectory", merge_lr=False)

    response_adfl = [column for column in response.columns if column.startswith("ADFL__")]
    full_adfl = [column for column in full.columns if column.startswith("ADFL__")]
    assert response_adfl[0] == "ADFL__t06"
    assert response_adfl[-1] == "ADFL__t20"
    assert len(response_adfl) == 15
    assert full_adfl[0] == "ADFL__t00"
    assert full_adfl[-1] == "ADFL__t44"
    assert len(full_adfl) == 45


def test_build_neural_rdm_returns_square_matrix_with_labels_and_metadata():
    raw = _raw_frame(
        [
            {
                "neuron": "ADFL",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [0.0 if t < 6 else float(t) for t in EXPECTED_TIMEPOINTS],
            },
            {
                "neuron": "ADFL",
                "stimulus": "s2",
                "worm_key": "worm_002",
                "segment_index": 0,
                "values": [0.0 if t < 6 else float(2 * t) for t in EXPECTED_TIMEPOINTS],
            },
            {
                "neuron": "ASER",
                "stimulus": "s1",
                "worm_key": "worm_001",
                "segment_index": 0,
                "values": [0.0 if t < 6 else float(2 * t) for t in EXPECTED_TIMEPOINTS],
            },
            {
                "neuron": "ASER",
                "stimulus": "s2",
                "worm_key": "worm_002",
                "segment_index": 0,
                "values": [0.0 if t < 6 else float(t) for t in EXPECTED_TIMEPOINTS],
            },
        ]
    )
    dataset = AnalysisDataset(
        neural=raw,
        matrix=pd.DataFrame(),
        metadata=pd.DataFrame(),
        stimulus_sample_map=pd.DataFrame(),
        included_dates=("20260327",),
        excluded_dates=(),
        parameters={},
    )

    result = build_neural_rdm(dataset, view="response_window", aggregation="median", merge_lr=True)

    assert result.matrix.index.tolist() == ["s1", "s2"]
    assert result.matrix.columns.tolist() == ["s1", "s2"]
    assert result.matrix.shape == (2, 2)
    assert result.matrix.loc["s1", "s1"] == pytest.approx(0.0)
    assert np.isfinite(result.matrix.loc["s1", "s2"])
    assert result.metadata["view"] == "response_window"
    assert result.metadata["aggregation"] == "median"
    assert result.metadata["merge_lr"] is True
    assert result.metadata["distance"] == "correlation"
