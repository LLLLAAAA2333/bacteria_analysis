from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bacteria_analysis.analyses import chemical_class_rsa as chemical_class_rsa_module
from bacteria_analysis.analyses.chemical_class_rsa import run_chemical_class_rsa
from bacteria_analysis.io import AnalysisDataset, AnalysisResult, save_analysis_result


def _target_rdm():
    labels = ["s1", "s2", "s3", "s4"]
    values = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    return pd.DataFrame(values, index=labels, columns=labels)


def _dataset():
    matrix = pd.DataFrame(
        {
            "match_1": [1.0, 2.0, 4.0, 8.0],
            "match_2": [1.0, 1.0, 1.0, 1.0],
            "match_3": [1.0, 1.0, 1.0, 1.0],
            "noise_1": [1.0, 8.0, 1.0, 2.0],
            "noise_2": [1.0, 1.0, 8.0, 2.0],
            "noise_3": [8.0, 1.0, 1.0, 2.0],
            "third_1": [1.0, 1.0, 8.0, 8.0],
            "third_2": [1.0, 8.0, 1.0, 8.0],
            "third_3": [8.0, 1.0, 1.0, 8.0],
        },
        index=["sample_s1", "sample_s2", "sample_s3", "sample_s4"],
    )
    metadata = pd.DataFrame(
        {
            "metabolite_name": matrix.columns.tolist(),
            "QCRSD": [0.1] * matrix.shape[1],
            "Class": ["match"] * 3 + ["noise"] * 3 + ["third"] * 3,
        }
    )
    mapping = pd.DataFrame(
        {
            "stimulus": ["s1", "s2", "s3", "s4"],
            "stim_name": ["s1", "s2", "s3", "s4"],
            "sample_id": ["sample_s1", "sample_s2", "sample_s3", "sample_s4"],
        }
    )
    neural = pd.DataFrame(
        {
            "stimulus": ["s1", "s2", "s3", "s4"],
            "date": ["20260401", "20260401", "20260402", "20260402"],
        }
    )
    return AnalysisDataset(
        neural=neural,
        matrix=matrix,
        metadata=metadata,
        stimulus_sample_map=mapping,
        included_dates=("20260401", "20260402"),
        excluded_dates=(),
        parameters={"matrix_path": "matrix.xlsx", "metadata_path": "metabolism.xlsx"},
    )


def test_run_chemical_class_rsa_separates_evidence_layers_and_ranks_top_class():
    result = run_chemical_class_rsa(
        _dataset(),
        neural_rdm=_target_rdm(),
        fixed_permutations=8,
        resamples=5,
        search_permutations=8,
        top_k=2,
        seed=4,
        include_debug=True,
    )

    assert isinstance(result, AnalysisResult)
    assert result.summary["top_class"] == "match"
    assert result.summary["top_class_feature_count"] == 3
    assert result.summary["search_corrected_results_are_diagnostic"] is True

    observed = result.tables["observed_class_scores"]
    assert observed["class"].tolist()[0] == "match"
    assert observed.loc[0, "feature_count"] == 3

    fixed = result.tables["fixed_class_permutation_summary"]
    assert {"p_value", "q_value"}.issubset(fixed.columns)
    assert fixed["class"].tolist()[0] == "match"

    reselection = result.tables["reselection_stability_summary"]
    assert {"top_fraction", "date_aware_resampling"}.issubset(reselection.columns)
    assert reselection.loc[reselection["class"] == "match", "top_fraction"].iloc[0] > 0
    assert reselection["date_aware_resampling"].all()
    assert "reselection_date_composition" in result.audit
    assert result.audit["reselection_date_composition"]["date_composition"].str.contains("20260401").all()

    search = result.debug_tables["search_corrected_diagnostic_summary"]
    assert search["diagnostic"].all()
    assert "search_corrected_p_value" in search.columns

    shortlist = result.tables["final_class_shortlist"]
    assert shortlist["selected_for_audit"].all()
    assert shortlist["class"].tolist()[0] == "match"
    assert len(shortlist) == 2
    assert shortlist.loc[shortlist["class"] == "match", "feature_count"].iloc[0] == 3

    assert "class_vs_full_chemical_rdm_similarity" in result.tables
    assert "class_to_class_chemical_rdm_similarity" in result.debug_tables
    assert "fixed_class_permutation.png" in result.figures
    assert "reselection_stability.png" in result.figures
    assert "top_class_rdm_comparison.png" in result.figures
    assert "taxonomy_class_stability_summary.png" in result.figures
    assert "class_chemical_rdm_similarity_matrix.png" in result.figures
    assert "class_vs_full_chemical_rdm_similarity.png" in result.figures
    figure = result.figures["top_class_rdm_comparison.png"](None)
    assert isinstance(figure, Figure)
    plt.close(figure)

    assert set(result.rdms).issuperset({"neural", "chemical_full", "class_match"})
    assert len([key for key in result.rdms if key.startswith("class_")]) == 2
    assert "fixed_class_null" in result.debug_tables
    assert "search_max_null" in result.debug_tables
    assert "reselection_draws" in result.debug_tables


def test_chemical_class_rsa_saved_result_does_not_write_all_candidate_rdms(tmp_path):
    result = run_chemical_class_rsa(
        _dataset(),
        neural_rdm=_target_rdm(),
        fixed_permutations=2,
        resamples=2,
        search_permutations=2,
        top_k=1,
        seed=1,
        include_debug=True,
    )

    save_analysis_result(result, tmp_path / "class_rsa")

    assert (tmp_path / "class_rsa" / "rdms" / "neural.csv").exists()
    assert (tmp_path / "class_rsa" / "rdms" / "chemical_full.csv").exists()
    assert (tmp_path / "class_rsa" / "rdms" / "class_match.csv").exists()
    assert (tmp_path / "class_rsa" / "figures" / "top_class_rdm_comparison.png").exists()
    assert not (tmp_path / "class_rsa" / "rdms" / "class_noise.csv").exists()
    assert not (tmp_path / "class_rsa" / "rdms" / "class_third.csv").exists()
    assert not (tmp_path / "class_rsa" / "debug").exists()


def test_chemical_class_rsa_omits_debug_tables_by_default():
    result = run_chemical_class_rsa(
        _dataset(),
        neural_rdm=_target_rdm(),
        fixed_permutations=1,
        resamples=1,
        search_permutations=1,
        top_k=1,
    )

    assert result.debug_tables == {}


def test_chemical_class_rsa_does_not_load_legacy_plot_scripts():
    source = Path(chemical_class_rsa_module.__file__).read_text(encoding="utf-8")

    assert "analysis_plot_scripts" not in source
    assert "load_plot_script" not in source
