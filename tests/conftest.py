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
from bacteria_analysis.io import write_json, write_markdown_report, write_parquet, write_tensor_npz

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


def _build_stage1_waveform(stimulus: str) -> np.ndarray:
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


def _build_stage1_raw_frame() -> pd.DataFrame:
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
        waveform = _build_stage1_waveform(stimulus)
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
def stage1_raw_df():
    return _build_stage1_raw_frame()


@pytest.fixture
def stage1_preprocessing_outputs(stage1_raw_df):
    return run_preprocessing_pipeline(stage1_raw_df)


@pytest.fixture
def stage1_clean_df(stage1_preprocessing_outputs):
    return stage1_preprocessing_outputs["clean_df"].copy()


@pytest.fixture
def stage1_trial_metadata(stage1_preprocessing_outputs):
    return stage1_preprocessing_outputs["metadata"].copy()


@pytest.fixture
def stage1_trial_wide(stage1_preprocessing_outputs):
    return stage1_preprocessing_outputs["wide"].copy()


@pytest.fixture
def stage1_trial_tensor(stage1_preprocessing_outputs):
    return stage1_preprocessing_outputs["tensor"].copy()


@pytest.fixture
def stage1_stage0_root(tmp_path, stage1_preprocessing_outputs):
    root = tmp_path / "stage0"
    clean_dir = root / "clean"
    trial_level_dir = root / "trial_level"
    qc_dir = root / "qc"

    write_parquet(stage1_preprocessing_outputs["clean_df"], clean_dir / "neuron_segments_clean.parquet")
    write_parquet(stage1_preprocessing_outputs["metadata"], trial_level_dir / "trial_metadata.parquet")
    write_parquet(stage1_preprocessing_outputs["wide"], trial_level_dir / "trial_wide_baseline_centered.parquet")
    write_tensor_npz(
        trial_level_dir / "trial_tensor_baseline_centered.npz",
        stage1_preprocessing_outputs["tensor"],
        stage1_preprocessing_outputs["metadata"]["trial_id"].tolist(),
        stage1_preprocessing_outputs["metadata"]["stimulus"].tolist(),
        stage1_preprocessing_outputs["metadata"]["stim_name"].tolist(),
    )
    write_json(stage1_preprocessing_outputs["report"], qc_dir / "preprocessing_report.json")
    write_markdown_report(stage1_preprocessing_outputs["report"], qc_dir / "preprocessing_report.md")

    return root


@pytest.fixture
def synthetic_geometry_comparisons() -> pd.DataFrame:
    rows = [
        {
            "view_name": "response_window",
            "trial_id_a": "20260101__worm_a__0",
            "trial_id_b": "20260101__worm_a__1",
            "stimulus_a": "b1_1",
            "stimulus_b": "b1_1",
            "individual_id_a": "2026-01-01__worm_a",
            "individual_id_b": "2026-01-01__worm_a",
            "date_a": "2026-01-01",
            "date_b": "2026-01-01",
            "same_stimulus": True,
            "same_individual": True,
            "same_date": True,
            "comparison_status": "ok",
            "distance": 0.10,
        },
        {
            "view_name": "response_window",
            "trial_id_a": "20260101__worm_a__0",
            "trial_id_b": "20260101__worm_a__2",
            "stimulus_a": "b1_1",
            "stimulus_b": "b2_1",
            "individual_id_a": "2026-01-01__worm_a",
            "individual_id_b": "2026-01-01__worm_a",
            "date_a": "2026-01-01",
            "date_b": "2026-01-01",
            "same_stimulus": False,
            "same_individual": True,
            "same_date": True,
            "comparison_status": "ok",
            "distance": 0.80,
        },
        {
            "view_name": "response_window",
            "trial_id_a": "20260102__worm_b__0",
            "trial_id_b": "20260102__worm_b__1",
            "stimulus_a": "b2_1",
            "stimulus_b": "b2_1",
            "individual_id_a": "2026-01-02__worm_b",
            "individual_id_b": "2026-01-02__worm_b",
            "date_a": "2026-01-02",
            "date_b": "2026-01-02",
            "same_stimulus": True,
            "same_individual": True,
            "same_date": True,
            "comparison_status": "ok",
            "distance": 0.20,
        },
        {
            "view_name": "response_window",
            "trial_id_a": "20260102__worm_b__0",
            "trial_id_b": "20260102__worm_b__2",
            "stimulus_a": "b2_1",
            "stimulus_b": "b3_1",
            "individual_id_a": "2026-01-02__worm_b",
            "individual_id_b": "2026-01-02__worm_b",
            "date_a": "2026-01-02",
            "date_b": "2026-01-02",
            "same_stimulus": False,
            "same_individual": True,
            "same_date": True,
            "comparison_status": "insufficient_overlap_neurons",
            "distance": np.nan,
        },
        {
            "view_name": "response_window",
            "trial_id_a": "20260101__worm_a__0",
            "trial_id_b": "20260102__worm_b__0",
            "stimulus_a": "b1_1",
            "stimulus_b": "b2_1",
            "individual_id_a": "2026-01-01__worm_a",
            "individual_id_b": "2026-01-02__worm_b",
            "date_a": "2026-01-01",
            "date_b": "2026-01-02",
            "same_stimulus": False,
            "same_individual": False,
            "same_date": False,
            "comparison_status": "ok",
            "distance": 0.95,
        },
        {
            "view_name": "full_trajectory",
            "trial_id_a": "20260101__worm_a__0",
            "trial_id_b": "20260101__worm_a__1",
            "stimulus_a": "b1_1",
            "stimulus_b": "b1_1",
            "individual_id_a": "2026-01-01__worm_a",
            "individual_id_b": "2026-01-01__worm_a",
            "date_a": "2026-01-01",
            "date_b": "2026-01-01",
            "same_stimulus": True,
            "same_individual": True,
            "same_date": True,
            "comparison_status": "ok",
            "distance": 0.05,
        },
    ]

    return pd.DataFrame.from_records(rows)
