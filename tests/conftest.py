from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS
from bacteria_analysis.preprocessing import run_preprocessing_pipeline

STAGE1_STIMULI = ("b1_1", "b2_1", "b3_1")
STAGE1_STIMULUS_META = {
    "b1_1": {"stim_name": "Bacteria 1", "stim_color": "#1f77b4"},
    "b2_1": {"stim_name": "Bacteria 2", "stim_color": "#ff7f0e"},
    "b3_1": {"stim_name": "Bacteria 3", "stim_color": "#2ca02c"},
}
STAGE1_STIMULUS_NEURONS = {
    "b1_1": ("ADFL", "ASEL", "ASER"),
    "b2_1": ("ADFL", "ASEL", "ASGL"),
    "b3_1": ("ADFR", "ASEL", "ASER"),
}
STAGE1_NEURON_SCALES = {
    "ADFL": 1.0,
    "ADFR": 0.85,
    "ASEL": 0.6,
    "ASER": 1.2,
    "ASGL": 0.9,
}


def _build_synthetic_waveform(stimulus: str) -> np.ndarray:
    values = np.zeros(len(EXPECTED_TIMEPOINTS), dtype=float)

    if stimulus == "b1_1":
        values[6:16] = np.linspace(0.2, 1.1, 10)
        values[16:21] = np.linspace(1.0, 0.6, 5)
        values[21:] = 0.4
    elif stimulus == "b2_1":
        values[6:16] = np.linspace(-0.1, -0.9, 10)
        values[16:21] = np.linspace(-1.0, -0.4, 5)
        values[21:] = -0.2
    elif stimulus == "b3_1":
        values[6:11] = np.linspace(0.0, 0.8, 5)
        values[11:16] = np.linspace(0.9, 0.2, 5)
        values[16:21] = np.linspace(0.1, -0.4, 5)
        values[21:] = -0.1
    else:
        raise ValueError(f"unknown stimulus: {stimulus}")

    return values


def _build_synthetic_raw_frame() -> pd.DataFrame:
    rows = []
    trial_specs = (
        ("2026-03-27", "worm_001", 0, "b1_1"),
        ("2026-03-27", "worm_001", 1, "b2_1"),
        ("2026-03-27", "worm_001", 2, "b3_1"),
        ("2026-03-27", "worm_002", 0, "b1_1"),
        ("2026-03-27", "worm_002", 1, "b2_1"),
        ("2026-03-27", "worm_002", 2, "b3_1"),
        ("2026-03-28", "worm_001", 0, "b1_1"),
        ("2026-03-28", "worm_001", 1, "b2_1"),
        ("2026-03-28", "worm_001", 2, "b3_1"),
        ("2026-03-28", "worm_002", 0, "b1_1"),
        ("2026-03-28", "worm_002", 1, "b2_1"),
        ("2026-03-28", "worm_002", 2, "b3_1"),
    )

    for date, worm_key, segment_index, stimulus in trial_specs:
        waveform = _build_synthetic_waveform(stimulus)
        metadata = STAGE1_STIMULUS_META[stimulus]
        for neuron in STAGE1_STIMULUS_NEURONS[stimulus]:
            scaled_values = waveform * STAGE1_NEURON_SCALES[neuron]
            for time_point, value in zip(EXPECTED_TIMEPOINTS, scaled_values, strict=True):
                rows.append(
                    {
                        "neuron": neuron,
                        "stimulus": stimulus,
                        "time_point": time_point,
                        "delta_F_over_F0": float(value),
                        "worm_key": worm_key,
                        "segment_index": segment_index,
                        "date": date,
                        "stim_name": metadata["stim_name"],
                        "stim_color": metadata["stim_color"],
                    }
                )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


@pytest.fixture
def synthetic_neuron_segments_df():
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
            "values": [time_point / 10 for time_point in EXPECTED_TIMEPOINTS],
        },
        {
            "neuron": "ADFR",
            "stimulus": "b1_1",
            "worm_key": "worm_001",
            "segment_index": 0,
            "date": "2026-03-27",
            "stim_name": "Bacteria 1",
            "stim_color": "#1f77b4",
            "values": [np.nan for _ in EXPECTED_TIMEPOINTS],
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
                np.nan if time_point in {0, 1, 2, 10, 11} else (time_point + 5) / 20
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

    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    return frame


@pytest.fixture
def synthetic_raw_df():
    return _build_synthetic_raw_frame()
