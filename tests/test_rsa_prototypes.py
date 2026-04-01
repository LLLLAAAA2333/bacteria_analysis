import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.reliability import TrialView


def _trial_view(metadata: pd.DataFrame, values: np.ndarray, name: str = "response_window") -> TrialView:
    return TrialView(name=name, timepoints=(0, 1), metadata=metadata, values=values)


def test_build_grouped_prototypes_uses_nanmean_and_tracks_support():
    from bacteria_analysis import rsa_prototypes

    metadata = pd.DataFrame(
        {
            "date": ["2026-03-11", "2026-03-11", "2026-03-11"],
            "stimulus": ["b1_1", "b1_1", "b2_1"],
            "stim_name": ["A001 stationary", "A001 stationary", "A002 stationary"],
        }
    )
    values = np.array(
        [
            [[1.0, np.nan], [2.0, 4.0]],
            [[3.0, 5.0], [np.nan, 6.0]],
            [[7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=float,
    )
    view = _trial_view(metadata, values)

    prototypes, support = rsa_prototypes.build_grouped_prototypes(
        view,
        group_columns=("date", "stimulus", "stim_name"),
    )

    row = prototypes.loc[(prototypes["date"] == "2026-03-11") & (prototypes["stimulus"] == "b1_1")].iloc[0]
    assert row["f000"] == pytest.approx(2.0)
    assert row["f001"] == pytest.approx(5.0)
    support_row = support.loc[(support["date"] == "2026-03-11") & (support["stimulus"] == "b1_1")].iloc[0]
    assert support_row["n_trials"] == 2
    assert support_row["n_total_features"] == 4
    assert support_row["n_supported_features"] == 4
    assert support_row["n_all_nan_features"] == 0


def test_build_pooled_prototype_support_tracks_contributing_dates():
    from bacteria_analysis import rsa_prototypes

    metadata = pd.DataFrame(
        {
            "date": ["2026-03-11", "2026-03-13", "2026-03-13"],
            "stimulus": ["b1_1", "b1_1", "b2_1"],
            "stim_name": ["A001 stationary", "A001 stationary", "A002 stationary"],
        }
    )
    values = np.array(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=float,
    )
    view = _trial_view(metadata, values)

    _, support = rsa_prototypes.build_grouped_prototypes(view, group_columns=("stimulus", "stim_name"))

    support_row = support.loc[support["stimulus"] == "b1_1"].iloc[0]
    assert support_row["n_trials"] == 2
    assert support_row["n_dates_contributed"] == 2


def test_build_prototype_rdm_ignores_grouped_metadata_columns():
    from bacteria_analysis import rsa_prototypes

    metadata = pd.DataFrame(
        {
            "date": ["2026-03-11", "2026-03-12"],
            "stimulus": ["b1_1", "b2_1"],
            "stim_name": ["A001 stationary", "A002 stationary"],
        }
    )
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=float,
    )
    view = _trial_view(metadata, values)
    prototypes, _ = rsa_prototypes.build_grouped_prototypes(
        view,
        group_columns=("date", "stimulus", "stim_name"),
    )

    matrix = rsa_prototypes.build_prototype_rdm(prototypes, id_columns=("date", "stimulus", "stim_name"))

    assert set(matrix["stimulus_row"]) == {
        "2026-03-11__b1_1__A001 stationary",
        "2026-03-12__b2_1__A002 stationary",
    }
    assert matrix.shape == (2, 3)
    assert (
        matrix.loc[matrix["stimulus_row"] == "2026-03-11__b1_1__A001 stationary", "2026-03-12__b2_1__A002 stationary"].iloc[0]
        == pytest.approx(0.0)
    )


def test_load_prototype_supplement_inputs_allows_missing_wide_table(stage1_stage0_root):
    from bacteria_analysis import rsa_prototypes

    wide_path = stage1_stage0_root / "trial_level" / "trial_wide_baseline_centered.parquet"
    wide_path.unlink()

    inputs = rsa_prototypes.load_prototype_supplement_inputs(stage1_stage0_root, view_names=("response_window",))

    assert inputs.metadata.shape[0] > 0
    assert "response_window" in inputs.views
    assert inputs.views["response_window"].name == "response_window"


def test_build_prototype_rdm_uses_overlap_aware_correlation_distance_and_nan_for_invalid_pairs():
    from bacteria_analysis import rsa_prototypes

    prototypes = pd.DataFrame.from_records(
        [
            {"stimulus": "b1_1", "f000": 1.0, "f001": 2.0, "f002": 3.0},
            {"stimulus": "b2_1", "f000": 2.0, "f001": 4.0, "f002": 6.0},
            {"stimulus": "b3_1", "f000": 1.0, "f001": 1.0, "f002": 1.0},
        ]
    )

    matrix = rsa_prototypes.build_prototype_rdm(prototypes, id_columns=("stimulus",))

    assert matrix.loc[matrix["stimulus_row"] == "b1_1", "b2_1"].iloc[0] == pytest.approx(0.0)
    assert np.isnan(matrix.loc[matrix["stimulus_row"] == "b1_1", "b3_1"].iloc[0])
    assert matrix.loc[matrix["stimulus_row"] == "b1_1", "stimulus_row"].iloc[0] == "b1_1"
