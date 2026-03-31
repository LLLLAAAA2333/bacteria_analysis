import json

import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.rsa import (
    align_rdm_upper_triangles,
    benjamini_hochberg,
    build_permutation_null,
    compute_rsa_score,
    summarize_cross_view_comparison,
    summarize_leave_one_stimulus_out,
)
from bacteria_analysis.rsa_outputs import write_stage3_outputs


def _matrix(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


@pytest.fixture
def synthetic_stage3_outputs() -> dict[str, pd.DataFrame]:
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "A001"},
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "A002"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "A003"},
        ]
    )
    metabolite_annotation_resolved = pd.DataFrame.from_records(
        [
            {"metabolite_name": "feature_1", "superclass": "lipid", "subclass": "bile_acid"},
            {"metabolite_name": "feature_2", "superclass": "lipid", "subclass": "fatty_acid"},
            {"metabolite_name": "feature_3", "superclass": "lipid", "subclass": "bile_acid"},
        ]
    )
    model_registry = pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "model_label": "Global Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Global reference model",
                "authority": "user",
                "notes": "",
            },
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "euclidean",
                "description": "Focused primary model",
                "authority": "user",
                "notes": "",
            },
            {
                "model_id": "lipid_panel",
                "model_label": "Lipid Panel",
                "model_tier": "supplementary",
                "model_status": "supplementary",
                "feature_kind": "binary_presence",
                "distance_kind": "jaccard",
                "description": "Supplementary comparison model",
                "authority": "user",
                "notes": "",
            },
            {
                "model_id": "excluded_sparse",
                "model_label": "Excluded Sparse",
                "model_tier": "supplementary",
                "model_status": "excluded",
                "feature_kind": "binary_presence",
                "distance_kind": "jaccard",
                "description": "Excluded from RSA ranking",
                "authority": "user",
                "notes": "",
            },
        ]
    )
    model_registry_resolved = model_registry.copy()
    model_registry_resolved["is_primary_family"] = model_registry_resolved["model_tier"].eq("primary")
    model_registry_resolved["is_supplementary_family"] = model_registry_resolved["model_tier"].eq("supplementary")
    model_registry_resolved["excluded_from_primary_ranking"] = model_registry_resolved["model_id"].eq(
        "excluded_sparse"
    )
    model_membership_resolved = pd.DataFrame.from_records(
        [
            {"model_id": "bile_acid", "metabolite_name": "feature_1"},
            {"model_id": "bile_acid", "metabolite_name": "feature_3"},
            {"model_id": "lipid_panel", "metabolite_name": "feature_2"},
        ]
    )
    model_input_coverage = pd.DataFrame.from_records(
        [
            {"model_id": "global_profile", "n_resolved_metabolites": 3, "coverage_status": "ok"},
            {"model_id": "bile_acid", "n_resolved_metabolites": 2, "coverage_status": "ok"},
            {"model_id": "lipid_panel", "n_resolved_metabolites": 1, "coverage_status": "ok"},
            {"model_id": "excluded_sparse", "n_resolved_metabolites": 1, "coverage_status": "excluded"},
        ]
    )

    model_feature_qc = pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "metabolite_name": "feature_1",
                "feature_kind": "continuous_abundance",
                "retained": True,
                "status": "retained",
                "filter_reason": "retained",
                "threshold": np.nan,
            },
            {
                "model_id": "global_profile",
                "metabolite_name": "feature_2",
                "feature_kind": "continuous_abundance",
                "retained": False,
                "status": "dropped",
                "filter_reason": "zero_variance",
                "threshold": np.nan,
            },
            {
                "model_id": "bile_acid",
                "metabolite_name": "feature_3",
                "feature_kind": "continuous_abundance",
                "retained": True,
                "status": "retained",
                "filter_reason": "retained",
                "threshold": np.nan,
            },
        ]
    )

    model_rdm_summary = pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "view_name": "response_window",
                "distance_kind": "correlation",
                "n_stimuli": 3,
                "mean_distance": 0.42,
            },
            {
                "model_id": "bile_acid",
                "view_name": "full_trajectory",
                "distance_kind": "euclidean",
                "n_stimuli": 3,
                "mean_distance": 0.37,
            },
        ]
    )

    rsa_results = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "reference_view_name": "model_rdm",
                "comparison_scope": "neural_vs_model",
                "model_id": "global_profile",
                "model_tier": "primary",
                "model_status": "primary",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.94,
                "p_value_raw": 0.010,
                "p_value_fdr": 0.015,
                "is_top_model": True,
            },
            {
                "view_name": "response_window",
                "reference_view_name": "model_rdm",
                "comparison_scope": "neural_vs_model",
                "model_id": "bile_acid",
                "model_tier": "primary",
                "model_status": "primary",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.88,
                "p_value_raw": 0.020,
                "p_value_fdr": 0.025,
                "is_top_model": False,
            },
            {
                "view_name": "response_window",
                "reference_view_name": "model_rdm",
                "comparison_scope": "neural_vs_model",
                "model_id": "lipid_panel",
                "model_tier": "supplementary",
                "model_status": "supplementary",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.41,
                "p_value_raw": 0.120,
                "p_value_fdr": 0.130,
                "is_top_model": False,
            },
            {
                "view_name": "full_trajectory",
                "reference_view_name": "model_rdm",
                "comparison_scope": "neural_vs_model",
                "model_id": "global_profile",
                "model_tier": "primary",
                "model_status": "primary",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.97,
                "p_value_raw": 0.005,
                "p_value_fdr": 0.010,
                "is_top_model": True,
            },
            {
                "view_name": "full_trajectory",
                "reference_view_name": "model_rdm",
                "comparison_scope": "neural_vs_model",
                "model_id": "bile_acid",
                "model_tier": "primary",
                "model_status": "primary",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.91,
                "p_value_raw": 0.015,
                "p_value_fdr": 0.020,
                "is_top_model": False,
            },
            {
                "view_name": "full_trajectory",
                "reference_view_name": "model_rdm",
                "comparison_scope": "neural_vs_model",
                "model_id": "lipid_panel",
                "model_tier": "supplementary",
                "model_status": "supplementary",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.47,
                "p_value_raw": 0.140,
                "p_value_fdr": 0.150,
                "is_top_model": False,
            },
        ]
    )

    leave_one_stimulus_out = pd.DataFrame.from_records(
        [
            {
                "excluded_stimulus": "b1_1",
                "view_name": "response_window",
                "model_id": "global_profile",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 2,
                "rsa_similarity": 0.86,
                "p_value_raw": 0.030,
                "p_value_fdr": 0.040,
            },
            {
                "excluded_stimulus": "b2_1",
                "view_name": "response_window",
                "model_id": "global_profile",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 2,
                "rsa_similarity": 0.82,
                "p_value_raw": 0.050,
                "p_value_fdr": 0.060,
            },
            {
                "excluded_stimulus": "b3_1",
                "view_name": "response_window",
                "model_id": "global_profile",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 2,
                "rsa_similarity": 0.79,
                "p_value_raw": 0.070,
                "p_value_fdr": 0.080,
            },
        ]
    )

    cross_view_comparison = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "reference_view_name": "full_trajectory",
                "comparison_scope": "neural_vs_neural",
                "model_id": "global_profile",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.98,
                "p_value_raw": 0.010,
                "p_value_fdr": 0.015,
            },
            {
                "view_name": "full_trajectory",
                "reference_view_name": "response_window",
                "comparison_scope": "neural_vs_neural",
                "model_id": "global_profile",
                "score_method": "spearman",
                "score_status": "ok",
                "n_shared_entries": 3,
                "rsa_similarity": 0.98,
                "p_value_raw": 0.010,
                "p_value_fdr": 0.015,
            },
        ]
    )

    model_rdm_summary__global_profile = pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "view_name": "response_window",
                "n_shared_entries": 3,
                "rsa_similarity": 0.94,
            }
        ]
    )
    model_rdm_summary__bile_acid = pd.DataFrame.from_records(
        [
            {
                "model_id": "bile_acid",
                "view_name": "full_trajectory",
                "n_shared_entries": 3,
                "rsa_similarity": 0.91,
            }
        ]
    )
    neural_rdm__response_window = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 0.2, "b3_1": 0.5},
            {"stimulus_row": "b2_1", "b1_1": 0.2, "b2_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b1_1": 0.5, "b2_1": 0.4, "b3_1": 0.0},
        ]
    )
    neural_rdm__full_trajectory = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 0.1, "b3_1": 0.6},
            {"stimulus_row": "b2_1", "b1_1": 0.1, "b2_1": 0.0, "b3_1": 0.3},
            {"stimulus_row": "b3_1", "b1_1": 0.6, "b2_1": 0.3, "b3_1": 0.0},
        ]
    )
    model_rdm__global_profile__response_window = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 0.3, "b3_1": 0.6},
            {"stimulus_row": "b2_1", "b1_1": 0.3, "b2_1": 0.0, "b3_1": 0.5},
            {"stimulus_row": "b3_1", "b1_1": 0.6, "b2_1": 0.5, "b3_1": 0.0},
        ]
    )
    model_rdm__global_profile__full_trajectory = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 0.2, "b3_1": 0.7},
            {"stimulus_row": "b2_1", "b1_1": 0.2, "b2_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b1_1": 0.7, "b2_1": 0.4, "b3_1": 0.0},
        ]
    )

    return {
        "stimulus_sample_map": stimulus_sample_map,
        "metabolite_annotation_resolved": metabolite_annotation_resolved,
        "model_registry": model_registry,
        "model_registry_resolved": model_registry_resolved,
        "model_membership_resolved": model_membership_resolved,
        "model_feature_qc": model_feature_qc,
        "model_input_coverage": model_input_coverage,
        "model_rdm_summary": model_rdm_summary,
        "model_rdm_summary__global_profile": model_rdm_summary__global_profile,
        "model_rdm_summary__bile_acid": model_rdm_summary__bile_acid,
        "rsa_results": rsa_results,
        "leave_one_stimulus_out": leave_one_stimulus_out,
        "cross_view_comparison": cross_view_comparison,
        "neural_rdm__response_window": neural_rdm__response_window,
        "neural_rdm__full_trajectory": neural_rdm__full_trajectory,
        "model_rdm__global_profile__response_window": model_rdm__global_profile__response_window,
        "model_rdm__global_profile__full_trajectory": model_rdm__global_profile__full_trajectory,
    }


def test_align_rdm_upper_triangles_uses_stimulus_labels_not_row_positions():
    neural = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 1.0, "s3": 5.0},
            {"stimulus_row": "s2", "s1": 1.0, "s2": 0.0, "s3": 3.0},
            {"stimulus_row": "s3", "s1": 5.0, "s2": 3.0, "s3": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "s3", "s2": 7.0, "s3": 0.0, "s1": 9.0},
            {"stimulus_row": "s1", "s2": 2.0, "s3": 9.0, "s1": 0.0},
            {"stimulus_row": "s2", "s2": 0.0, "s3": 7.0, "s1": 2.0},
        ]
    )

    shared = align_rdm_upper_triangles(neural, model)

    assert shared.to_dict("records") == [
        {"stimulus_left": "s1", "stimulus_right": "s2", "neural_value": 1.0, "model_value": 2.0},
        {"stimulus_left": "s1", "stimulus_right": "s3", "neural_value": 5.0, "model_value": 9.0},
        {"stimulus_left": "s2", "stimulus_right": "s3", "neural_value": 3.0, "model_value": 7.0},
    ]


def test_compute_rsa_score_uses_shared_upper_triangle_entries_only():
    neural = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 0.1, "s3": 0.8},
            {"stimulus_row": "s2", "s1": 0.1, "s2": 0.0, "s3": 0.3},
            {"stimulus_row": "s3", "s1": 0.8, "s2": 0.3, "s3": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 0.2, "s3": 0.4},
            {"stimulus_row": "s2", "s1": 0.2, "s2": 0.0, "s3": np.nan},
            {"stimulus_row": "s3", "s1": 0.4, "s2": np.nan, "s3": 0.0},
        ]
    )

    result = compute_rsa_score(neural, model)

    assert result["score_status"] == "ok"
    assert result["n_shared_entries"] == 2
    assert result["rsa_similarity"] == pytest.approx(1.0)
    assert np.isnan(result["p_value_raw"])
    assert np.isnan(result["p_value_fdr"])


def test_benjamini_hochberg_returns_monotonic_adjusted_values():
    adjusted = benjamini_hochberg(np.array([0.01, 0.03, 0.2]))

    assert np.all(np.diff(adjusted) >= 0)
    assert adjusted.tolist() == pytest.approx([0.03, 0.045, 0.2])


def test_leave_one_stimulus_out_summary_records_excluded_stimulus():
    neural = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 0.2, "b3_1": 0.4},
            {"stimulus_row": "b2_1", "b1_1": 0.2, "b2_1": 0.0, "b3_1": 0.3},
            {"stimulus_row": "b3_1", "b1_1": 0.4, "b2_1": 0.3, "b3_1": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "b1_1", "b1_1": 0.0, "b2_1": 1.0, "b3_1": 2.0},
            {"stimulus_row": "b2_1", "b1_1": 1.0, "b2_1": 0.0, "b3_1": 3.0},
            {"stimulus_row": "b3_1", "b1_1": 2.0, "b2_1": 3.0, "b3_1": 0.0},
        ]
    )

    summary = summarize_leave_one_stimulus_out(neural, model)

    assert set(summary["excluded_stimulus"]) == {"b1_1", "b2_1", "b3_1"}
    assert summary["n_shared_entries"].tolist() == [1, 1, 1]


def test_build_permutation_null_is_reproducible_for_fixed_seed():
    neural = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 0.2, "s3": 0.8, "s4": 0.4},
            {"stimulus_row": "s2", "s1": 0.2, "s2": 0.0, "s3": 0.5, "s4": 0.1},
            {"stimulus_row": "s3", "s1": 0.8, "s2": 0.5, "s3": 0.0, "s4": 0.9},
            {"stimulus_row": "s4", "s1": 0.4, "s2": 0.1, "s3": 0.9, "s4": 0.0},
        ]
    )
    model = _matrix(
        [
            {"stimulus_row": "s1", "s1": 0.0, "s2": 1.0, "s3": 4.0, "s4": 2.0},
            {"stimulus_row": "s2", "s1": 1.0, "s2": 0.0, "s3": 3.0, "s4": 5.0},
            {"stimulus_row": "s3", "s1": 4.0, "s2": 3.0, "s3": 0.0, "s4": 6.0},
            {"stimulus_row": "s4", "s1": 2.0, "s2": 5.0, "s3": 6.0, "s4": 0.0},
        ]
    )

    first = build_permutation_null(neural, model, n_iterations=6, seed=13)
    second = build_permutation_null(neural, model, n_iterations=6, seed=13)

    pd.testing.assert_frame_equal(first, second)


def test_summarize_cross_view_comparison_returns_requested_views():
    model = _matrix(
        [
            {"stimulus_row": "A001", "A001": 0.0, "A002": 1.0, "A003": 2.0},
            {"stimulus_row": "A002", "A001": 1.0, "A002": 0.0, "A003": 3.0},
            {"stimulus_row": "A003", "A001": 2.0, "A002": 3.0, "A003": 0.0},
        ]
    )
    neural_matrices = {
        "response_window": _matrix(
            [
                {"stimulus_row": "A001", "A001": 0.0, "A002": 0.2, "A003": 0.4},
                {"stimulus_row": "A002", "A001": 0.2, "A002": 0.0, "A003": 0.3},
                {"stimulus_row": "A003", "A001": 0.4, "A002": 0.3, "A003": 0.0},
            ]
        ),
        "full_trajectory": _matrix(
            [
                {"stimulus_row": "A001", "A001": 0.0, "A002": 0.1, "A003": 0.5},
                {"stimulus_row": "A002", "A001": 0.1, "A002": 0.0, "A003": 0.2},
                {"stimulus_row": "A003", "A001": 0.5, "A002": 0.2, "A003": 0.0},
            ]
        ),
    }

    summary = summarize_cross_view_comparison(neural_matrices, model)

    assert summary["view_name"].tolist() == ["response_window", "full_trajectory"]
    assert summary["score_status"].tolist() == ["ok", "ok"]


def test_compute_rsa_score_rejects_labels_that_collide_after_string_normalization():
    matrix = _matrix(
        [
            {"stimulus_row": 1, 1: 0.0, "1": 0.2},
            {"stimulus_row": "1", 1: 0.2, "1": 0.0},
        ]
    )

    with pytest.raises(ValueError, match="collide after string normalization"):
        compute_rsa_score(matrix, matrix)


def test_write_stage3_outputs_writes_required_tables(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")

    assert (written["tables_dir"] / "stimulus_sample_map.parquet").exists()
    assert (written["tables_dir"] / "rsa_results.parquet").exists()
    assert (written["tables_dir"] / "model_registry_resolved.parquet").exists()
    assert (written["tables_dir"] / "rsa_leave_one_stimulus_out.parquet").exists()
    assert (written["qc_dir"] / "model_input_coverage.parquet").exists()
    assert (written["qc_dir"] / "model_feature_filtering.parquet").exists()
    assert (written["figures_dir"] / "ranked_primary_model_rsa.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm_panel.png").exists()
    assert (written["figures_dir"] / "leave_one_stimulus_out_robustness.png").exists()
    assert (written["figures_dir"] / "view_comparison_summary.png").exists()
    assert (written["output_root"] / "run_summary.json").exists()
    assert (written["output_root"] / "run_summary.md").exists()


def test_write_stage3_outputs_records_primary_and_supplementary_models(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["primary_models"] == ["global_profile", "bile_acid"]
    assert summary["supplementary_models"] == ["lipid_panel"]
    assert summary["excluded_models"] == ["excluded_sparse"]
    assert summary["top_primary_models_by_view"] == {
        "response_window": "global_profile",
        "full_trajectory": "global_profile",
    }
