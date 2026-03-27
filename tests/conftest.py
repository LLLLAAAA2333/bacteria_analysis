from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS


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
