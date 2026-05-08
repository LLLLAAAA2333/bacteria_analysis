from __future__ import annotations

import pandas as pd

from bacteria_analysis.io import (
    AnalysisDataset,
    AnchorDataset,
    build_analysis_dataset,
    build_anchor_dataset,
)


def _write_neural_parquet(tmp_path, rows):
    path = tmp_path / "raw_neural.parquet"
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)
    return path


def _write_matrix_xlsx(tmp_path):
    path = tmp_path / "matrix.xlsx"
    pd.DataFrame.from_records(
        [
            {"sample_id": "A001", "feature_1": 0.1, "feature_2": 1.0},
            {"sample_id": "A002", "feature_1": 0.2, "feature_2": 0.8},
            {"sample_id": "A003", "feature_1": 0.3, "feature_2": 0.6},
        ]
    ).to_excel(path, index=False, engine="openpyxl")
    return path


def _write_metadata_csv(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame.from_records(
        [
            {"trial_id": "t1", "date": "2026-03-11", "operator": "one"},
            {"trial_id": "t2", "date": "20260312", "operator": "two"},
            {"trial_id": "t3", "date": "2026/03/13", "operator": "three"},
        ]
    ).to_csv(path, index=False)
    return path


def _write_metadata_xlsx(tmp_path):
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame.from_records([{"trial_id": "t1", "date": "2026-03-11"}]).to_excel(
        path,
        index=False,
        engine="openpyxl",
    )
    return path


def _raw_like_row(trial_id, date, stimulus, stim_name, *, neuron="ADFL", time_point=0, value=1.0):
    return {
        "trial_id": trial_id,
        "date": date,
        "stimulus": stimulus,
        "stim_name": stim_name,
        "stim_color": "#111111",
        "worm_key": "worm_001",
        "segment_index": 0,
        "neuron": neuron,
        "time_point": time_point,
        "delta_F_over_F0": value,
    }


def test_build_analysis_dataset_loads_raw_parquet_matrix_and_metadata(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            _raw_like_row("t1", "2026-03-11", "b1_1", "A001 stationary", neuron="ADFL", time_point=0, value=1.0),
            _raw_like_row("t1", "2026-03-11", "b1_1", "A001 stationary", neuron="ADFL", time_point=1, value=1.1),
            _raw_like_row("t2", "20260312", "b2_1", "A002 stationary", neuron="ASEL", time_point=0, value=2.0),
            _raw_like_row("t2", "20260312", "b2_1", "A002 stationary", neuron="ASEL", time_point=1, value=2.1),
        ],
    )

    dataset = build_analysis_dataset(neural_path, _write_matrix_xlsx(tmp_path), _write_metadata_csv(tmp_path))

    assert isinstance(dataset, AnalysisDataset)
    assert dataset.neural["trial_id"].tolist() == ["t1", "t1", "t2", "t2"]
    assert dataset.neural["date"].tolist() == ["20260311", "20260311", "20260312", "20260312"]
    assert {"worm_key", "segment_index", "neuron", "time_point", "delta_F_over_F0"}.issubset(dataset.neural.columns)
    assert dataset.matrix.index.tolist() == ["A001", "A002", "A003"]
    assert dataset.metadata["date"].tolist() == ["20260311", "20260312", "20260313"]
    assert dataset.stimulus_sample_map.to_dict(orient="records") == [
        {"stimulus": "b1_1", "stim_name": "A001 stationary", "sample_id": "A001"},
        {"stimulus": "b2_1", "stim_name": "A002 stationary", "sample_id": "A002"},
    ]
    assert dataset.included_dates == ("20260311", "20260312")
    assert dataset.excluded_dates == ()


def test_build_analysis_dataset_excludes_dates_without_preprocess_root(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            {"trial_id": "t1", "date": "2026-03-11", "stimulus": "b1_1", "stim_name": "A001 stationary"},
            {"trial_id": "t2", "date": "20260312", "stimulus": "b2_1", "stim_name": "A002 stationary"},
            {"trial_id": "t3", "date": "2026/03/13", "stimulus": "b3_1", "stim_name": "A003 stationary"},
        ],
    )

    dataset = build_analysis_dataset(
        neural_path,
        _write_matrix_xlsx(tmp_path),
        _write_metadata_csv(tmp_path),
        exclude_dates=["2026-03-12"],
    )

    assert dataset.neural["trial_id"].tolist() == ["t1", "t3"]
    assert dataset.metadata["trial_id"].tolist() == ["t1", "t3"]
    assert dataset.included_dates == ("20260311", "20260313")
    assert dataset.excluded_dates == ("20260312",)


def test_build_analysis_dataset_keep_dates_filters_locally(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            {"trial_id": "t1", "date": pd.Timestamp("2026-03-11"), "stimulus": "b1_1", "stim_name": "A001 stationary"},
            {"trial_id": "t2", "date": pd.Timestamp("2026-03-12"), "stimulus": "b2_1", "stim_name": "A002 stationary"},
            {"trial_id": "t3", "date": pd.Timestamp("2026-03-13"), "stimulus": "b3_1", "stim_name": "A003 stationary"},
        ],
    )

    dataset = build_analysis_dataset(
        neural_path,
        _write_matrix_xlsx(tmp_path),
        _write_metadata_xlsx(tmp_path),
        keep_dates=["20260311", "2026-03-13"],
    )

    assert dataset.neural["trial_id"].tolist() == ["t1", "t3"]
    assert dataset.metadata["trial_id"].tolist() == ["t1"]
    assert dataset.included_dates == ("20260311", "20260313")
    assert dataset.excluded_dates == ("20260312",)
    assert dataset.parameters["keep_dates"] == ("20260311", "20260313")


def test_build_analysis_dataset_builds_best_effort_map_from_stimulus_only(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            {"trial_id": "t1", "date": "20260311", "stimulus": "anchor_a"},
            {"trial_id": "t2", "date": "20260312", "stimulus": "anchor_b"},
        ],
    )

    dataset = build_analysis_dataset(neural_path, _write_matrix_xlsx(tmp_path), _write_metadata_csv(tmp_path))

    assert dataset.stimulus_sample_map.to_dict(orient="records") == [
        {"stimulus": "anchor_a", "stim_name": "anchor_a", "sample_id": ""},
        {"stimulus": "anchor_b", "stim_name": "anchor_b", "sample_id": ""},
    ]


def test_build_analysis_dataset_rejects_invalid_stimulus_sample_mapping(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            {"trial_id": "t1", "date": "20260311", "stimulus": "b1_1", "stim_name": "A999 stationary"},
        ],
    )

    try:
        build_analysis_dataset(neural_path, _write_matrix_xlsx(tmp_path), _write_metadata_csv(tmp_path))
    except ValueError as exc:
        assert "must exist in the matrix" in str(exc)
    else:
        raise AssertionError("Expected invalid sample mapping to fail")


def test_build_anchor_dataset_loads_requested_anchors_separately(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            {"trial_id": "t1", "date": "2026-03-11", "stimulus": "anchor_a", "stim_name": "Anchor A"},
            {"trial_id": "t2", "date": "2026-03-12", "stimulus": "b2_1", "stim_name": "A002 stationary"},
            {"trial_id": "t3", "date": "2026-03-13", "stimulus": "anchor_b", "stim_name": "Anchor B"},
        ],
    )

    dataset = build_anchor_dataset(neural_path, ["anchor_a", "Anchor B"], exclude_dates=["2026-03-12"])

    assert isinstance(dataset, AnchorDataset)
    assert dataset.neural["trial_id"].tolist() == ["t1", "t3"]
    assert dataset.anchor_stimuli == ("anchor_a", "Anchor B")
    assert dataset.included_dates == ("20260311", "20260313")
    assert dataset.excluded_dates == ("20260312",)


def test_build_anchor_dataset_reports_dates_after_anchor_filtering(tmp_path):
    neural_path = _write_neural_parquet(
        tmp_path,
        [
            {"trial_id": "t1", "date": "2026-03-11", "stimulus": "anchor_a", "stim_name": "Anchor A"},
            {"trial_id": "t2", "date": "2026-03-12", "stimulus": "non_anchor", "stim_name": "Other"},
            {"trial_id": "t3", "date": "2026-03-13", "stimulus": "anchor_b", "stim_name": "Anchor B"},
        ],
    )

    dataset = build_anchor_dataset(neural_path, ["anchor_a", "anchor_b"])

    assert dataset.neural["trial_id"].tolist() == ["t1", "t3"]
    assert dataset.included_dates == ("20260311", "20260313")
