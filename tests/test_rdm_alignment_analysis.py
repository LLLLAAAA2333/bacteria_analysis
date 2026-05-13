from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bacteria_analysis.analyses.rdm import neural_chemical as rdm_alignment_module
from bacteria_analysis.analyses.rdm.neural_chemical import run_rdm_alignment
from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, REQUIRED_COLUMNS
from bacteria_analysis.io import AnalysisDataset, AnalysisResult, save_analysis_result


def _patterns():
    return {
        "s1": np.linspace(0.1, 1.5, 15),
        "s2": np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.2, 1.3, 1.25, 1.1, 0.9, 0.7, 0.5, 0.35, 0.2, 0.1]),
        "s3": np.linspace(1.5, 0.1, 15),
        "s4": 0.8 + 0.4 * np.sin(np.linspace(0, np.pi, 15)),
    }


def _raw_neural(patterns):
    rows = []
    dates = {"s1": "20260401", "s2": "20260401", "s3": "20260402", "s4": "20260402"}
    for index, (stimulus, response) in enumerate(patterns.items(), start=1):
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
                    "date": dates[stimulus],
                    "stim_name": stimulus,
                    "stim_color": "#111111",
                }
            )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _dataset():
    patterns = _patterns()
    feature_names = [f"m{index:02d}" for index in range(15)]
    matrix = pd.DataFrame.from_dict(
        {f"sample_{stimulus}": pattern for stimulus, pattern in patterns.items()},
        orient="index",
        columns=feature_names,
    )
    metadata = pd.DataFrame(
        {
            "metabolite_name": feature_names,
            "QCRSD": [0.1] * len(feature_names),
            "Class": ["synthetic"] * len(feature_names),
        }
    )
    mapping = pd.DataFrame(
        {
            "stimulus": list(patterns),
            "stim_name": list(patterns),
            "sample_id": [f"sample_{stimulus}" for stimulus in patterns],
        }
    )
    return AnalysisDataset(
        neural=_raw_neural(patterns),
        matrix=matrix,
        metadata=metadata,
        stimulus_sample_map=mapping,
        included_dates=("20260401", "20260402"),
        excluded_dates=(),
        parameters={"neural_path": "raw_neural.parquet", "matrix_path": "matrix.xlsx", "seed": 999},
    )


def test_run_rdm_alignment_returns_result_with_expected_summary_and_audit():
    result = run_rdm_alignment(
        _dataset(),
        chemical_transform="none",
        chemical_distance="correlation",
        permutations=8,
        subset_count=6,
        subset_fraction=0.75,
        seed=3,
        include_debug=True,
    )

    assert isinstance(result, AnalysisResult)
    assert result.parameters["seed"] == 3
    assert result.summary["all_pairs_rsa"] == pytest.approx(1.0)
    assert "within_date_rsa" in result.summary
    assert "cross_date_rsa" in result.summary
    assert result.summary["n_pairs_all"] == 6
    assert result.summary["n_pairs_within_date"] == 2
    assert result.summary["n_pairs_cross_date"] == 4
    assert "stress test, not proof of generalization" in result.summary["date_structure_caveat"]
    assert "label_shuffle_p_value" in result.summary
    assert "date_preserving_p_value" in result.summary
    assert "subset_rsa_median" in result.summary
    assert "subset_rsa_q01" in result.summary
    assert "subset_rsa_q99" in result.summary
    assert not any("date_stratified_rank" in key for key in result.summary)

    assert result.rdms["neural"].index.tolist() == ["s1", "s2", "s3", "s4"]
    assert result.rdms["chemical"].index.tolist() == ["s1", "s2", "s3", "s4"]
    assert result.audit["aligned_stimulus_order"] == ["s1", "s2", "s3", "s4"]
    assert result.audit["retained_features"] == [f"m{index:02d}" for index in range(15)]
    assert result.audit["date_coverage"]["n_stimuli"].tolist() == [2, 2]
    assert set(result.audit["date_pair_coverage"]["date_scope"]) == {"within_date", "cross_date"}

    assert "rsa_summary_by_scope" in result.tables
    assert "neural_chemical_rdm_foundation__response_window_label_shuffle_0k_rdms.png" in result.figures
    assert "neural_chemical_rdm_foundation__response_window_label_shuffle_0k_distribution.png" in result.figures
    assert "neural_chemical_rdm_foundation__response_window_label_shuffle_0k_distribution_fraction.png" in result.figures
    assert (
        "neural_chemical_rdm_foundation__response_window_random_subsets_6x500_frac75_distribution_fraction.png"
        in result.figures
    )
    figure = result.figures["neural_chemical_rdm_foundation__response_window_label_shuffle_0k_rdms.png"](None)
    assert isinstance(figure, Figure)
    plt.close(figure)

    assert "pair_values" in result.debug_tables
    assert "label_shuffle_null" in result.debug_tables
    assert "subset_rsa" in result.debug_tables


def test_run_rdm_alignment_omits_debug_tables_by_default():
    result = run_rdm_alignment(
        _dataset(),
        chemical_transform="none",
        chemical_distance="correlation",
        permutations=2,
        subset_count=2,
        seed=1,
    )

    assert result.debug_tables == {}


def test_run_rdm_alignment_saved_result_keeps_final_rdms_and_skips_debug_by_default(tmp_path):
    result = run_rdm_alignment(
        _dataset(),
        chemical_transform="none",
        chemical_distance="correlation",
        permutations=2,
        subset_count=2,
        seed=2,
        include_debug=True,
    )

    save_analysis_result(result, tmp_path / "alignment")

    assert (tmp_path / "alignment" / "rdms" / "neural.csv").exists()
    assert (tmp_path / "alignment" / "rdms" / "chemical.csv").exists()
    assert not (tmp_path / "alignment" / "audit").exists()
    assert not (tmp_path / "alignment" / "debug").exists()


def test_rdm_alignment_does_not_load_legacy_plot_scripts():
    source = Path(rdm_alignment_module.__file__).read_text(encoding="utf-8")

    assert "analysis_plot_scripts" not in source
    assert "load_plot_script" not in source
