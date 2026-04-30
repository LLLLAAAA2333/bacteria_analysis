import numpy as np
import pandas as pd

from bacteria_analysis.analysis_dataset import AnchorDataset
from bacteria_analysis.analysis_results import AnalysisResult, save_analysis_result
from bacteria_analysis.analyses.anchor_batch_effect import run_anchor_batch_effect
from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS


def _anchor_patterns():
    return {
        ("anchor_a", "20260401"): np.linspace(0.1, 1.5, 15),
        ("anchor_a", "20260402"): np.linspace(1.5, 0.1, 15),
        ("anchor_b", "20260401"): 0.8 + 0.4 * np.sin(np.linspace(0, np.pi, 15)),
        ("anchor_b", "20260402"): np.array(
            [0.1, 0.2, 0.4, 0.7, 1.0, 1.2, 1.3, 1.25, 1.1, 0.9, 0.7, 0.5, 0.35, 0.2, 0.1]
        ),
    }


def _raw_anchor_neural():
    rows = []
    for index, ((stimulus, date), response) in enumerate(_anchor_patterns().items(), start=1):
        response_by_time = dict(zip(range(6, 21), response, strict=True))
        for time_point in EXPECTED_TIMEPOINTS:
            rows.append(
                {
                    "neuron": "ADFL",
                    "stimulus": stimulus,
                    "time_point": time_point,
                    "delta_F_over_F0": float(response_by_time.get(time_point, 0.0)),
                    "worm_key": f"worm_{index:03d}",
                    "segment_index": 0,
                    "date": date,
                    "stim_name": stimulus,
                    "stim_color": "#111111",
                }
            )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _anchor_dataset():
    return AnchorDataset(
        neural=_raw_anchor_neural(),
        anchor_stimuli=("anchor_a", "anchor_b"),
        included_dates=("20260401", "20260402"),
        excluded_dates=(),
        parameters={"neural_path": "anchor.parquet"},
    )


def test_run_anchor_batch_effect_reports_coverage_distances_and_contrast():
    result = run_anchor_batch_effect(
        _anchor_dataset(),
        views=("response_window",),
        include_debug=True,
    )

    assert isinstance(result, AnalysisResult)
    assert result.summary["anchor_count"] == 2
    assert result.summary["view_count"] == 1

    coverage = result.tables["anchor_coverage"]
    assert coverage.to_dict("records") == [
        {"view": "response_window", "stimulus": "anchor_a", "date": "20260401", "n_trials": 1},
        {"view": "response_window", "stimulus": "anchor_a", "date": "20260402", "n_trials": 1},
        {"view": "response_window", "stimulus": "anchor_b", "date": "20260401", "n_trials": 1},
        {"view": "response_window", "stimulus": "anchor_b", "date": "20260402", "n_trials": 1},
    ]

    pairwise = result.tables["anchor_pairwise_distances"]
    assert len(pairwise) == 6
    assert pairwise["same_anchor"].any()
    assert pairwise["same_date"].any()
    assert pairwise["distance"].notna().all()

    same_anchor_summary = result.tables["same_anchor_cross_date_summary"]
    assert same_anchor_summary.loc[0, "view"] == "response_window"
    assert same_anchor_summary.loc[0, "n_pairs"] == 2

    contrast = result.tables["stimulus_vs_date_contrast"]
    assert contrast.loc[0, "n_same_anchor_cross_date"] == 2
    assert contrast.loc[0, "n_different_anchor_within_date"] == 2
    assert "date_effect_minus_stimulus_effect" in contrast.columns

    assert "response_window_anchor" in result.rdms
    assert "response_window_anchor_rdm" in result.figures
    assert "response_window_prototypes" in result.debug_tables


def test_anchor_batch_effect_can_be_saved_without_debug_by_default(tmp_path):
    result = run_anchor_batch_effect(
        _anchor_dataset(),
        views=("response_window",),
        include_debug=True,
    )

    save_analysis_result(result, tmp_path / "anchor")

    assert (tmp_path / "anchor" / "tables" / "anchor_coverage.csv").exists()
    assert (tmp_path / "anchor" / "tables" / "anchor_pairwise_distances.csv").exists()
    assert (tmp_path / "anchor" / "rdms" / "response_window_anchor.csv").exists()
    assert (tmp_path / "anchor" / "figures" / "response_window_anchor_rdm.png").exists()
    assert not (tmp_path / "anchor" / "debug").exists()


def test_anchor_batch_effect_omits_debug_tables_by_default():
    result = run_anchor_batch_effect(_anchor_dataset(), views=("response_window",))

    assert result.debug_tables == {}


def test_anchor_batch_effect_respects_declared_anchor_stimuli():
    dataset = _anchor_dataset()
    extra = dataset.neural.copy()
    extra["stimulus"] = "not_an_anchor"
    dataset = AnchorDataset(
        neural=pd.concat([dataset.neural, extra], ignore_index=True),
        anchor_stimuli=("anchor_a", "anchor_b"),
        included_dates=dataset.included_dates,
        excluded_dates=dataset.excluded_dates,
        parameters=dataset.parameters,
    )

    result = run_anchor_batch_effect(dataset, views=("response_window",))

    assert set(result.tables["anchor_coverage"]["stimulus"]) == {"anchor_a", "anchor_b"}
