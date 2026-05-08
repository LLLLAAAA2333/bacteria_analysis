from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bacteria_analysis.analyses import anchor_batch_effect as anchor_batch_effect_module
from bacteria_analysis.analyses.anchor_batch_effect import run_anchor_batch_effect
from bacteria_analysis.features.anchor import merge_neurons
from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS
from bacteria_analysis.io import AnchorDataset, AnalysisResult, save_analysis_result


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
        include_debug=True,
    )

    assert isinstance(result, AnalysisResult)
    assert result.summary["anchor_count"] == 2
    assert result.summary["view_count"] == 2

    coverage = result.tables["anchor_stimulus_coverage"]
    assert coverage.to_dict("records") == [
        {"stimulus": "anchor_a", "stim_name": "anchor_a", "date": "20260401", "n_trials": 1, "n_worms": 1},
        {"stimulus": "anchor_a", "stim_name": "anchor_a", "date": "20260402", "n_trials": 1, "n_worms": 1},
        {"stimulus": "anchor_b", "stim_name": "anchor_b", "date": "20260401", "n_trials": 1, "n_worms": 1},
        {"stimulus": "anchor_b", "stim_name": "anchor_b", "date": "20260402", "n_trials": 1, "n_worms": 1},
    ]

    pairwise = result.debug_tables["anchor_stimulus_pairwise_prototype_distances"]
    assert len(pairwise) == 12
    assert pairwise["same_stimulus"].any()
    assert pairwise["same_date"].any()
    assert pairwise["distance"].notna().all()

    same_anchor_summary = result.tables["anchor_stimulus_cross_date_anchor_summary"]
    response_summary = same_anchor_summary.loc[same_anchor_summary["view_name"] == "response_window"]
    assert len(response_summary) == 2
    assert response_summary["n_cross_date_pairs"].eq(1).all()

    contrast = result.tables["anchor_stimulus_date_pair_same_vs_other_contrasts"]
    response_contrast = contrast.loc[contrast["view_name"] == "response_window"]
    assert set(response_contrast["contrast"]) == {"same", "different"}

    assert "response_window_anchor" in result.rdms
    assert "full_trajectory_anchor" in result.rdms
    assert "anchor_stimulus_date_prototype_rdms.png" in result.figures
    assert "anchor_stimulus_date_mds__response_window.png" in result.figures
    assert "anchor_stimulus_trial_features" in result.debug_tables

    figure = result.figures["anchor_stimulus_date_prototype_rdms.png"](None)
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_merge_neurons_uses_generic_group_mapping():
    centered = pd.DataFrame(
        [
            {
                "trial_id": "trial_1",
                "date": "20260401",
                "stimulus": "s1",
                "stim_name": "s1",
                "worm_key": "worm_1",
                "segment_index": 0,
                "time_point": 6,
                "neuron": "LEFT",
                "dff_baseline_centered": 1.0,
            },
            {
                "trial_id": "trial_1",
                "date": "20260401",
                "stimulus": "s1",
                "stim_name": "s1",
                "worm_key": "worm_1",
                "segment_index": 0,
                "time_point": 6,
                "neuron": "RIGHT",
                "dff_baseline_centered": 3.0,
            },
            {
                "trial_id": "trial_1",
                "date": "20260401",
                "stimulus": "s1",
                "stim_name": "s1",
                "worm_key": "worm_1",
                "segment_index": 0,
                "time_point": 6,
                "neuron": "SINGLE",
                "dff_baseline_centered": 5.0,
            },
        ]
    )

    merged = merge_neurons(centered, {"PAIR": ("LEFT", "RIGHT"), "SINGLE": ("SINGLE",)})

    assert merged.set_index("merged_neuron")["dff_baseline_centered"].to_dict() == {
        "PAIR": 2.0,
        "SINGLE": 5.0,
    }


def test_anchor_batch_effect_can_be_saved_without_debug_by_default(tmp_path):
    result = run_anchor_batch_effect(
        _anchor_dataset(),
        include_debug=True,
    )

    save_analysis_result(result, tmp_path / "anchor")

    assert (tmp_path / "anchor" / "tables" / "anchor_stimulus_coverage.csv").exists()
    assert not (tmp_path / "anchor" / "tables" / "anchor_stimulus_pairwise_prototype_distances.csv").exists()
    assert (tmp_path / "anchor" / "rdms" / "response_window_anchor.csv").exists()
    assert (tmp_path / "anchor" / "figures" / "anchor_stimulus_date_prototype_rdms.png").exists()
    assert (
        tmp_path / "anchor" / "figures" / "anchor_stimulus_neuron_time_heatmap__full_trajectory.png"
    ).exists()
    assert not (tmp_path / "anchor" / "debug").exists()


def test_anchor_batch_effect_omits_debug_tables_by_default():
    result = run_anchor_batch_effect(_anchor_dataset())

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

    result = run_anchor_batch_effect(dataset)

    assert set(result.tables["anchor_stimulus_coverage"]["stimulus"]) == {"anchor_a", "anchor_b"}


def test_anchor_batch_effect_does_not_load_legacy_plot_scripts():
    source = Path(anchor_batch_effect_module.__file__).read_text(encoding="utf-8")

    assert "analysis_plot_scripts" not in source
    assert "load_plot_script" not in source
