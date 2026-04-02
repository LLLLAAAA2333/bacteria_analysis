import json

import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.rsa import (
    align_rdm_upper_triangles,
    benjamini_hochberg,
    build_permutation_null,
    compute_rsa_score,
    run_stage3_rsa,
    summarize_cross_view_comparison,
    summarize_leave_one_stimulus_out,
)
from bacteria_analysis import rsa_outputs
from bacteria_analysis.model_space import build_model_rdm
from bacteria_analysis.reliability import TrialView
from bacteria_analysis.rsa_prototypes import PrototypeSupplementInputs, build_grouped_prototypes, build_prototype_rdm
from bacteria_analysis.rsa_outputs import write_stage3_outputs


def _matrix(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _stage3_neural_rdms() -> dict[str, pd.DataFrame]:
    return {
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


def _resolved_stage3_inputs(
    *,
    matrix_rows: list[dict[str, object]],
    mapping_rows: list[dict[str, object]],
    registry_rows: list[dict[str, object]],
    membership_rows: list[dict[str, object]],
) -> dict[str, pd.DataFrame]:
    matrix = pd.DataFrame.from_records(matrix_rows).set_index("sample_id")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    mapping = pd.DataFrame.from_records(mapping_rows)
    registry = pd.DataFrame.from_records(registry_rows)
    registry["model_id"] = registry["model_id"].astype(str).str.strip().str.lower()
    if "excluded_from_primary_ranking" not in registry.columns:
        registry["excluded_from_primary_ranking"] = False
    registry["is_primary_family"] = registry["model_tier"].astype(str).str.strip().str.lower().eq("primary")
    registry["is_supplementary_family"] = registry["model_tier"].astype(str).str.strip().str.lower().eq(
        "supplementary"
    )

    membership = pd.DataFrame.from_records(membership_rows)
    if membership.empty:
        membership = pd.DataFrame(columns=["model_id", "metabolite_name"])
    else:
        membership["model_id"] = membership["model_id"].astype(str).str.strip().str.lower()
        membership["metabolite_name"] = membership["metabolite_name"].astype(str)

    annotation = pd.DataFrame(
        {
            "metabolite_name": matrix.columns.tolist(),
            "superclass": "",
            "subclass": "",
            "pathway_tag": "",
            "annotation_source": "",
            "review_status": "",
            "ambiguous_flag": False,
            "notes": "",
        }
    )

    return {
        "matrix": matrix,
        "stimulus_sample_map": mapping,
        "metabolite_annotation": annotation,
        "model_registry": registry.copy(),
        "model_membership": membership.copy(),
        "model_registry_resolved": registry.copy(),
        "model_membership_resolved": membership.copy(),
    }


def _stage3_prototype_resolved_inputs() -> dict[str, pd.DataFrame]:
    return _resolved_stage3_inputs(
        matrix_rows=[
            {"sample_id": "A001", "f1": 0.0, "f2": 1.0, "f3": 2.0, "f4": 3.0, "f5": 0.0, "f6": 1.0},
            {"sample_id": "A002", "f1": 0.0, "f2": 1.0, "f3": 3.0, "f4": 2.0, "f5": 1.0, "f6": 2.0},
            {"sample_id": "A003", "f1": 3.0, "f2": 2.0, "f3": 1.0, "f4": 0.0, "f5": 2.0, "f6": 3.0},
            {"sample_id": "A004", "f1": 1.0, "f2": 0.0, "f3": 2.0, "f4": 1.0, "f5": 3.0, "f6": 0.0},
        ],
        mapping_rows=[
            {"stimulus": "A001", "stim_name": "Stimulus A001", "sample_id": "A001"},
            {"stimulus": "A002", "stim_name": "Stimulus A002", "sample_id": "A002"},
            {"stimulus": "A003", "stim_name": "Stimulus A003", "sample_id": "A003"},
            {"stimulus": "A004", "stim_name": "Stimulus A004", "sample_id": "A004"},
        ],
        registry_rows=[
            {
                "model_id": "global_profile",
                "model_label": "Global Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
            },
            {
                "model_id": "focused_panel",
                "model_label": "Focused Panel",
                "model_tier": "supplementary",
                "model_status": "supplementary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "euclidean",
            },
            {
                "model_id": "excluded_panel",
                "model_label": "Excluded Panel",
                "model_tier": "supplementary",
                "model_status": "supplementary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "euclidean",
                "excluded_from_primary_ranking": True,
            },
        ],
        membership_rows=[
            {"model_id": "focused_panel", "metabolite_name": "f5"},
            {"model_id": "focused_panel", "metabolite_name": "f6"},
            {"model_id": "excluded_panel", "metabolite_name": "f1"},
            {"model_id": "excluded_panel", "metabolite_name": "f2"},
        ],
    )


def _stage3_prototype_inputs() -> PrototypeSupplementInputs:
    metadata = pd.DataFrame.from_records(
        [
            {"date": "2026-03-11", "stimulus": "A001", "stim_name": "Stimulus A001"},
            {"date": "2026-03-11", "stimulus": "A002", "stim_name": "Stimulus A002"},
            {"date": "2026-03-11", "stimulus": "A003", "stim_name": "Stimulus A003"},
            {"date": "2026-03-13", "stimulus": "A001", "stim_name": "Stimulus A001"},
            {"date": "2026-03-13", "stimulus": "A004", "stim_name": "Stimulus A004"},
        ]
    )
    response_window = np.array(
        [
            [[0.0, 1.0, 2.0, 3.0]],
            [[0.0, 1.0, 3.0, 2.0]],
            [[3.0, 2.0, 1.0, 0.0]],
            [[0.0, 1.0, np.nan, 0.0]],
            [[1.0, 0.0, np.nan, 1.0]],
        ],
        dtype=float,
    )
    full_trajectory = np.array(
        [
            [[0.0, 2.0, 1.0, 3.0]],
            [[1.0, 3.0, 0.0, 2.0]],
            [[3.0, 1.0, 2.0, 0.0]],
            [[0.0, 2.0, np.nan, 1.0]],
            [[1.0, 0.0, np.nan, 2.0]],
        ],
        dtype=float,
    )
    return PrototypeSupplementInputs(
        metadata=metadata,
        views={
            "response_window": TrialView(
                name="response_window",
                timepoints=(0,),
                metadata=metadata,
                values=response_window,
            ),
            "full_trajectory": TrialView(
                name="full_trajectory",
                timepoints=(0,),
                metadata=metadata,
                values=full_trajectory,
            ),
        },
    )


def _empty_stage3_prototype_inputs() -> PrototypeSupplementInputs:
    metadata = pd.DataFrame(columns=["date", "stimulus", "stim_name"])
    empty_values = np.empty((0, 1, 4), dtype=float)
    return PrototypeSupplementInputs(
        metadata=metadata,
        views={
            "response_window": TrialView(
                name="response_window",
                timepoints=(0,),
                metadata=metadata,
                values=empty_values,
            ),
            "full_trajectory": TrialView(
                name="full_trajectory",
                timepoints=(0,),
                metadata=metadata,
                values=empty_values,
            ),
        },
    )


def _stage3_outputs_with_prototype_supplement() -> dict[str, pd.DataFrame]:
    return run_stage3_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        prototype_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )


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
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert (written["figures_dir"] / "leave_one_stimulus_out_robustness.png").exists()
    assert (written["figures_dir"] / "view_comparison_summary.png").exists()
    assert (written["output_root"] / "run_summary.json").exists()
    assert (written["output_root"] / "run_summary.md").exists()

def test_write_stage3_outputs_writes_per_view_neural_model_figures(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")

    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert not (written["figures_dir"] / "neural_vs_top_model_rdm_panel.png").exists()


def test_write_stage3_outputs_removes_legacy_panel_figure(tmp_path, synthetic_stage3_outputs):
    output_root = tmp_path / "stage3_rsa"
    legacy_panel = output_root / "figures" / "neural_vs_top_model_rdm_panel.png"
    legacy_panel.parent.mkdir(parents=True, exist_ok=True)
    legacy_panel.write_bytes(b"legacy panel")

    written = write_stage3_outputs(synthetic_stage3_outputs, output_root)

    assert not legacy_panel.exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()


def test_write_stage3_outputs_removes_stale_per_view_figures_when_view_set_narrows(
    tmp_path, synthetic_stage3_outputs
):
    output_root = tmp_path / "stage3_rsa"

    first_written = write_stage3_outputs(synthetic_stage3_outputs, output_root)
    stale_full_trajectory = first_written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png"
    assert stale_full_trajectory.exists()

    narrowed_outputs = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in synthetic_stage3_outputs.items()
    }
    narrowed_outputs["rsa_results"] = narrowed_outputs["rsa_results"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs["cross_view_comparison"] = narrowed_outputs["cross_view_comparison"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()

    second_written = write_stage3_outputs(narrowed_outputs, output_root)

    assert (second_written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert not stale_full_trajectory.exists()
    assert not (second_written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert not (second_written["figures_dir"] / "neural_vs_top_model_rdm_panel.png").exists()


def test_write_stage3_outputs_reports_per_view_figure_names(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["figure_names"] == [
        "ranked_primary_model_rsa",
        "leave_one_stimulus_out_robustness",
        "view_comparison_summary",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]


def test_write_stage3_outputs_reports_canonical_per_view_figure_name_order(tmp_path, synthetic_stage3_outputs):
    rsa_results = synthetic_stage3_outputs["rsa_results"].copy()
    rsa_results["view_name"] = pd.Categorical(
        rsa_results["view_name"],
        categories=["sorted_view", "full_trajectory", "response_window", "alpha_view"],
        ordered=True,
    )
    rsa_results = pd.concat(
        [
            pd.DataFrame.from_records(
                [
                    {
                        "view_name": "sorted_view",
                        "reference_view_name": "model_rdm",
                        "comparison_scope": "neural_vs_model",
                        "model_id": "global_profile",
                        "model_tier": "primary",
                        "model_status": "primary",
                        "score_method": "spearman",
                        "score_status": "ok",
                        "n_shared_entries": 3,
                        "rsa_similarity": 0.89,
                        "p_value_raw": 0.011,
                        "p_value_fdr": 0.016,
                        "is_top_model": True,
                    },
                    {
                        "view_name": "alpha_view",
                        "reference_view_name": "model_rdm",
                        "comparison_scope": "neural_vs_model",
                        "model_id": "global_profile",
                        "model_tier": "primary",
                        "model_status": "primary",
                        "score_method": "spearman",
                        "score_status": "ok",
                        "n_shared_entries": 3,
                        "rsa_similarity": 0.87,
                        "p_value_raw": 0.012,
                        "p_value_fdr": 0.017,
                        "is_top_model": True,
                    },
                ]
            ),
            rsa_results,
        ],
        ignore_index=True,
    )
    cross_view_comparison = synthetic_stage3_outputs["cross_view_comparison"].copy()
    cross_view_comparison = pd.concat(
        [
            pd.DataFrame.from_records(
                [
                    {
                        "view_name": "sorted_view",
                        "reference_view_name": "response_window",
                        "comparison_scope": "neural_vs_neural",
                        "model_id": "global_profile",
                        "score_method": "spearman",
                        "score_status": "ok",
                        "n_shared_entries": 3,
                        "rsa_similarity": 0.95,
                        "p_value_raw": 0.013,
                        "p_value_fdr": 0.018,
                    },
                    {
                        "view_name": "alpha_view",
                        "reference_view_name": "full_trajectory",
                        "comparison_scope": "neural_vs_neural",
                        "model_id": "global_profile",
                        "score_method": "spearman",
                        "score_status": "ok",
                        "n_shared_entries": 3,
                        "rsa_similarity": 0.93,
                        "p_value_raw": 0.014,
                        "p_value_fdr": 0.019,
                    },
                ]
            ),
            cross_view_comparison,
        ],
        ignore_index=True,
    )
    synthetic_stage3_outputs["rsa_results"] = rsa_results
    synthetic_stage3_outputs["cross_view_comparison"] = cross_view_comparison

    for view_name in ("sorted_view", "alpha_view"):
        synthetic_stage3_outputs[f"neural_rdm__{view_name}"] = synthetic_stage3_outputs["neural_rdm__response_window"].copy()
        synthetic_stage3_outputs[f"model_rdm__global_profile__{view_name}"] = synthetic_stage3_outputs[
            "model_rdm__global_profile__response_window"
        ].copy()

    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["figure_names"] == [
        "ranked_primary_model_rsa",
        "leave_one_stimulus_out_robustness",
        "view_comparison_summary",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
        "neural_vs_top_model_rdm__alpha_view",
        "neural_vs_top_model_rdm__sorted_view",
    ]


def test_write_stage3_outputs_writes_default_per_view_placeholders_when_views_missing(tmp_path, synthetic_stage3_outputs):
    synthetic_stage3_outputs["rsa_results"] = pd.DataFrame()
    synthetic_stage3_outputs["cross_view_comparison"] = pd.DataFrame()

    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert summary["views"] == []
    assert summary["figure_names"] == [
        "ranked_primary_model_rsa",
        "leave_one_stimulus_out_robustness",
        "view_comparison_summary",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]


def test_write_stage3_outputs_records_primary_and_supplementary_models(tmp_path, synthetic_stage3_outputs):
    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["primary_view"] == "response_window"
    assert summary["primary_models"] == ["global_profile", "bile_acid"]
    assert summary["supplementary_models"] == ["lipid_panel"]
    assert summary["excluded_models"] == ["excluded_sparse"]
    assert summary["top_primary_models_by_view"] == {
        "response_window": "global_profile",
        "full_trajectory": "global_profile",
    }


def test_resolve_rdm_heatmap_frame_prefers_sample_id_labels_and_preserves_order():
    matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b1_1": 0.2, "b3_1": 0.5},
            {"stimulus_row": "b1_1", "b2_1": 0.2, "b1_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b2_1": 0.5, "b1_1": 0.4, "b3_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
        ]
    )

    resolved = rsa_outputs._resolve_rdm_heatmap_frame(matrix, stimulus_sample_map)

    assert resolved.index.tolist() == ["S002", "S001", "S003"]
    assert resolved.columns.tolist() == ["S002", "S001", "S003"]
    assert resolved.loc["S002", "S001"] == pytest.approx(0.2)
    assert resolved.loc["S001", "S003"] == pytest.approx(0.4)


def test_resolve_rdm_heatmap_frame_falls_back_when_sample_id_is_not_one_to_one():
    matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b1_1": 0.2, "b3_1": 0.5},
            {"stimulus_row": "b1_1", "b2_1": 0.2, "b1_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b2_1": 0.5, "b1_1": 0.4, "b3_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "shared"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "shared"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "shared"},
        ]
    )

    resolved = rsa_outputs._resolve_rdm_heatmap_frame(matrix, stimulus_sample_map)

    assert resolved.index.tolist() == ["Stimulus B2", "Stimulus B1", "Stimulus B3"]
    assert resolved.columns.tolist() == ["Stimulus B2", "Stimulus B1", "Stimulus B3"]
    assert resolved.loc["Stimulus B2", "Stimulus B1"] == pytest.approx(0.2)
    assert resolved.loc["Stimulus B1", "Stimulus B3"] == pytest.approx(0.4)


def test_resolve_rdm_heatmap_frame_falls_back_when_sample_id_is_incomplete():
    matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b1_1": 0.2, "b3_1": 0.5},
            {"stimulus_row": "b1_1", "b2_1": 0.2, "b1_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b2_1": 0.5, "b1_1": 0.4, "b3_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": None},
        ]
    )

    resolved = rsa_outputs._resolve_rdm_heatmap_frame(matrix, stimulus_sample_map)

    assert resolved.index.tolist() == ["Stimulus B2", "Stimulus B1", "Stimulus B3"]
    assert resolved.columns.tolist() == ["Stimulus B2", "Stimulus B1", "Stimulus B3"]
    assert resolved.loc["Stimulus B2", "Stimulus B1"] == pytest.approx(0.2)
    assert resolved.loc["Stimulus B1", "Stimulus B3"] == pytest.approx(0.4)


def test_resolve_rdm_heatmap_frame_ignores_duplicate_map_rows_outside_displayed_subset():
    matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b1_1": 0.2, "b3_1": 0.5},
            {"stimulus_row": "b1_1", "b2_1": 0.2, "b1_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b2_1": 0.5, "b1_1": 0.4, "b3_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
            {"stimulus": "x1", "stim_name": "Extra Stimulus", "sample_id": "dup"},
            {"stimulus": "x2", "stim_name": "Extra Stimulus", "sample_id": "dup"},
        ]
    )

    resolved = rsa_outputs._resolve_rdm_heatmap_frame(matrix, stimulus_sample_map)

    assert resolved.index.tolist() == ["S002", "S001", "S003"]
    assert resolved.columns.tolist() == ["S002", "S001", "S003"]
    assert resolved.loc["S002", "S001"] == pytest.approx(0.2)


def test_resolve_rdm_heatmap_frame_falls_back_to_raw_stimulus_when_other_layers_fail():
    matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b1_1": 0.2, "b3_1": 0.5},
            {"stimulus_row": "b1_1", "b2_1": 0.2, "b1_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b2_1": 0.5, "b1_1": 0.4, "b3_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "Shared Label", "sample_id": "dup"},
            {"stimulus": "b1_1", "stim_name": "Shared Label", "sample_id": "dup"},
            {"stimulus": "b3_1", "stim_name": "Shared Label", "sample_id": "dup"},
        ]
    )

    resolved = rsa_outputs._resolve_rdm_heatmap_frame(matrix, stimulus_sample_map)

    assert resolved.index.tolist() == ["b2_1", "b1_1", "b3_1"]
    assert resolved.columns.tolist() == ["b2_1", "b1_1", "b3_1"]
    assert resolved.loc["b2_1", "b1_1"] == pytest.approx(0.2)


def test_prepare_rdm_heatmap_frame_reuses_neural_cluster_order_for_model_matrix():
    neural_matrix = _matrix(
        [
            {"stimulus_row": "b3_1", "b3_1": 0.0, "b1_1": 8.0, "b4_1": 1.0, "b2_1": 8.0},
            {"stimulus_row": "b1_1", "b3_1": 8.0, "b1_1": 0.0, "b4_1": 8.0, "b2_1": 1.0},
            {"stimulus_row": "b4_1", "b3_1": 1.0, "b1_1": 8.0, "b4_1": 0.0, "b2_1": 8.0},
            {"stimulus_row": "b2_1", "b3_1": 8.0, "b1_1": 1.0, "b4_1": 8.0, "b2_1": 0.0},
        ]
    )
    model_matrix = _matrix(
        [
            {"stimulus_row": "b3_1", "b3_1": 0.0, "b1_1": 30.0, "b4_1": 10.0, "b2_1": 40.0},
            {"stimulus_row": "b1_1", "b3_1": 30.0, "b1_1": 0.0, "b4_1": 50.0, "b2_1": 20.0},
            {"stimulus_row": "b4_1", "b3_1": 10.0, "b1_1": 50.0, "b4_1": 0.0, "b2_1": 60.0},
            {"stimulus_row": "b2_1", "b3_1": 40.0, "b1_1": 20.0, "b4_1": 60.0, "b2_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
            {"stimulus": "b4_1", "stim_name": "Stimulus B4", "sample_id": "S004"},
        ]
    )

    neural_heatmap, order_labels = rsa_outputs._prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)
    model_heatmap, model_order_labels = rsa_outputs._prepare_rdm_heatmap_frame(
        model_matrix,
        stimulus_sample_map,
        order_labels=order_labels,
    )

    assert order_labels != ["b3_1", "b1_1", "b4_1", "b2_1"]
    assert order_labels == model_order_labels
    assert neural_heatmap.index.tolist() == model_heatmap.index.tolist()
    assert neural_heatmap.columns.tolist() == model_heatmap.columns.tolist()
    assert model_heatmap.loc["S003", "S004"] == pytest.approx(10.0)
    assert model_heatmap.loc["S001", "S002"] == pytest.approx(20.0)


def test_prepare_rdm_heatmap_frame_falls_back_to_aligned_original_order_when_neural_matrix_is_not_clusterable():
    neural_matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b1_1": np.nan, "b3_1": 0.5},
            {"stimulus_row": "b1_1", "b2_1": np.nan, "b1_1": 0.0, "b3_1": 0.4},
            {"stimulus_row": "b3_1", "b2_1": 0.5, "b1_1": 0.4, "b3_1": 0.0},
        ]
    )
    model_matrix = _matrix(
        [
            {"stimulus_row": "b3_1", "b3_1": 0.0, "b2_1": 0.7, "b1_1": 0.6},
            {"stimulus_row": "b2_1", "b3_1": 0.7, "b2_1": 0.0, "b1_1": 0.2},
            {"stimulus_row": "b1_1", "b3_1": 0.6, "b2_1": 0.2, "b1_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
        ]
    )

    neural_heatmap, order_labels = rsa_outputs._prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)
    model_heatmap, model_order_labels = rsa_outputs._prepare_rdm_heatmap_frame(
        model_matrix,
        stimulus_sample_map,
        order_labels=order_labels,
    )

    assert order_labels == ["b2_1", "b1_1", "b3_1"]
    assert model_order_labels == ["b2_1", "b1_1", "b3_1"]
    assert neural_heatmap.index.tolist() == ["S002", "S001", "S003"]
    assert model_heatmap.index.tolist() == ["S002", "S001", "S003"]
    assert model_heatmap.loc["S002", "S001"] == pytest.approx(0.2)


def test_prepare_rdm_heatmap_frame_falls_back_when_linkage_raises_on_invalid_distances():
    neural_matrix = _matrix(
        [
            {"stimulus_row": "b3_1", "b3_1": 0.0, "b1_1": -1.0, "b2_1": 0.6},
            {"stimulus_row": "b1_1", "b3_1": -1.0, "b1_1": 0.0, "b2_1": 0.4},
            {"stimulus_row": "b2_1", "b3_1": 0.6, "b1_1": 0.4, "b2_1": 0.0},
        ]
    )
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
        ]
    )

    heatmap_frame, order_labels = rsa_outputs._prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)

    assert order_labels == ["b3_1", "b1_1", "b2_1"]
    assert heatmap_frame.index.tolist() == ["S003", "S001", "S002"]
    assert heatmap_frame.columns.tolist() == ["S003", "S001", "S002"]
    assert heatmap_frame.loc["S003", "S001"] == pytest.approx(-1.0)


def test_plot_neural_vs_top_model_rdm_view_keeps_model_in_original_order_when_neural_matrix_is_missing(tmp_path):
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b4_1", "stim_name": "Stimulus B4", "sample_id": "S004"},
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
        ]
    )
    model_matrix = _matrix(
        [
            {"stimulus_row": "b3_1", "b3_1": 0.0, "b1_1": 30.0, "b4_1": 10.0, "b2_1": 40.0},
            {"stimulus_row": "b1_1", "b3_1": 30.0, "b1_1": 0.0, "b4_1": 50.0, "b2_1": 20.0},
            {"stimulus_row": "b4_1", "b3_1": 10.0, "b1_1": 50.0, "b4_1": 0.0, "b2_1": 60.0},
            {"stimulus_row": "b2_1", "b3_1": 40.0, "b1_1": 20.0, "b4_1": 60.0, "b2_1": 0.0},
        ]
    )
    core_outputs = {
        "stimulus_sample_map": stimulus_sample_map,
        "model_rdm__global_profile__response_window": model_matrix,
    }
    captured: dict[str, object] = {}
    original_render = rsa_outputs._render_rdm_axis

    def _capturing_render(axis, matrix_frame, *, stimulus_sample_map, title, fallback_message, order_labels=None):
        if title.endswith("global_profile"):
            heatmap_frame, resolved_order = rsa_outputs._prepare_rdm_heatmap_frame(
                matrix_frame,
                stimulus_sample_map,
                order_labels=order_labels,
            )
            captured["order_labels"] = order_labels
            captured["resolved_order"] = resolved_order
            captured["index_labels"] = heatmap_frame.index.tolist()
        return original_render(
            axis,
            matrix_frame,
            stimulus_sample_map=stimulus_sample_map,
            title=title,
            fallback_message=fallback_message,
            order_labels=order_labels,
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rsa_outputs, "_render_rdm_axis", _capturing_render)
    try:
        rsa_outputs._plot_neural_vs_top_model_rdm_view(
            core_outputs,
            {"response_window": "global_profile"},
            view_name="response_window",
            path=tmp_path / "missing_neural.png",
        )
    finally:
        monkeypatch.undo()

    assert captured["order_labels"] == []
    assert captured["resolved_order"] == ["b3_1", "b1_1", "b4_1", "b2_1"]
    assert captured["index_labels"] == ["S003", "S001", "S004", "S002"]


def test_build_top_prototype_models_by_date_and_view():
    prototype_rsa_results = pd.DataFrame.from_records(
        [
            {
                "date": "2026-03-11",
                "view_name": "response_window",
                "model_id": "global_profile",
                "is_top_model": True,
                "excluded_from_primary_ranking": False,
            },
            {
                "date": "2026-03-11",
                "view_name": "response_window",
                "model_id": "excluded_panel",
                "is_top_model": True,
                "excluded_from_primary_ranking": True,
            },
            {
                "date": "2026-03-13",
                "view_name": "response_window",
                "model_id": "focused_panel",
                "is_top_model": False,
                "excluded_from_primary_ranking": False,
            },
            {
                "date": "2026-03-13",
                "view_name": "full_trajectory",
                "model_id": "global_profile",
                "is_top_model": True,
                "excluded_from_primary_ranking": False,
            },
        ]
    )

    top_models = rsa_outputs._build_top_prototype_models_by_date_and_view(prototype_rsa_results)

    assert top_models == {
        ("2026-03-11", "response_window"): "global_profile",
        ("2026-03-13", "full_trajectory"): "global_profile",
    }


def test_plot_prototype_rdm_comparison_per_date_uses_internal_rdms_when_rsa_table_is_empty(tmp_path):
    core_outputs = _stage3_outputs_with_prototype_supplement()
    captured: list[tuple[str, bool]] = []
    original_render = rsa_outputs._render_rdm_axis

    def _capturing_render(axis, matrix_frame, *, stimulus_sample_map, title, fallback_message, order_labels=None):
        captured.append((title, matrix_frame is not None))
        return original_render(
            axis,
            matrix_frame,
            stimulus_sample_map=stimulus_sample_map,
            title=title,
            fallback_message=fallback_message,
            order_labels=order_labels,
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rsa_outputs, "_render_rdm_axis", _capturing_render)
    try:
        written_path = rsa_outputs._plot_prototype_rdm_comparison_per_date(
            core_outputs,
            pd.DataFrame(),
            {},
            view_name="response_window",
            path=tmp_path / "prototype_rdm_comparison__per_date__response_window.png",
        )
    finally:
        monkeypatch.undo()

    assert written_path.exists()
    assert captured == [
        ("2026-03-11: neural prototype", True),
        ("2026-03-11: no top model", False),
        ("2026-03-13: neural prototype", True),
        ("2026-03-13: no top model", False),
    ]


def test_write_stage3_outputs_keeps_skipped_supplementary_models_in_supplementary_bucket(
    tmp_path, synthetic_stage3_outputs
):
    registry = synthetic_stage3_outputs["model_registry_resolved"].copy()
    registry.loc[registry["model_id"] == "lipid_panel", "excluded_from_primary_ranking"] = True
    synthetic_stage3_outputs["model_registry_resolved"] = registry

    written = write_stage3_outputs(synthetic_stage3_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["supplementary_models"] == ["lipid_panel"]
    assert summary["excluded_models"] == ["excluded_sparse"]


def test_write_stage3_outputs_writes_per_date_prototype_comparison_figures(tmp_path):
    written = write_stage3_outputs(_stage3_outputs_with_prototype_supplement(), tmp_path / "stage3_rsa")

    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png").exists()


def test_write_stage3_outputs_writes_prototype_supplementary_artifacts(tmp_path):
    core_outputs = _stage3_outputs_with_prototype_supplement()

    written = write_stage3_outputs(core_outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (written["tables_dir"] / "prototype_rsa_results__per_date.parquet").exists()
    assert (written["tables_dir"] / "prototype_rdm__pooled__response_window.parquet").exists()
    assert (written["tables_dir"] / "prototype_rdm__pooled__full_trajectory.parquet").exists()
    assert (written["qc_dir"] / "prototype_support__per_date.parquet").exists()
    assert (written["qc_dir"] / "prototype_support__pooled.parquet").exists()
    assert (written["figures_dir"] / "prototype_rsa__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rsa__per_date__full_trajectory.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png").exists()
    assert (written["figures_dir"] / "prototype_rdm__pooled__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png").exists()
    assert summary["rsa_table_names"] == [
        "model_rdm_summary",
        "rsa_results",
        "rsa_leave_one_stimulus_out",
        "rsa_view_comparison",
    ]
    assert summary["additional_table_names"] == [
        "model_rdm__excluded_panel",
        "model_rdm__focused_panel",
        "model_rdm__global_profile",
        "neural_rdm__full_trajectory",
        "neural_rdm__response_window",
        "prototype_rdm__pooled__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rsa_results__per_date",
    ]
    assert not any(key.startswith("internal__") for key in written)
    assert not any(name.startswith("internal__") for name in summary["additional_table_names"])
    assert not list((written["tables_dir"]).glob("internal__*.parquet"))
    assert summary["prototype_supplement_enabled"] is True
    assert summary["prototype_views"] == ["response_window", "full_trajectory"]
    assert summary["prototype_dates"] == ["2026-03-11", "2026-03-13"]
    assert summary["prototype_table_names"] == [
        "prototype_rsa_results__per_date",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["prototype_figure_names"] == [
        "prototype_rsa__per_date__response_window",
        "prototype_rsa__per_date__full_trajectory",
        "prototype_rdm_comparison__per_date__response_window",
        "prototype_rdm_comparison__per_date__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["prototype_descriptive_outputs"] == [
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["figure_names"] == [
        "ranked_primary_model_rsa",
        "leave_one_stimulus_out_robustness",
        "view_comparison_summary",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
        "prototype_rsa__per_date__response_window",
        "prototype_rsa__per_date__full_trajectory",
        "prototype_rdm_comparison__per_date__response_window",
        "prototype_rdm_comparison__per_date__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]


def test_write_stage3_outputs_removes_stale_prototype_figures_when_view_set_narrows(tmp_path):
    output_root = tmp_path / "stage3_rsa"

    first_written = write_stage3_outputs(_stage3_outputs_with_prototype_supplement(), output_root)
    stale_prototype_rsa = first_written["figures_dir"] / "prototype_rsa__per_date__full_trajectory.png"
    stale_prototype_rdm_comparison = first_written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png"
    stale_prototype_rdm = first_written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png"
    stale_prototype_rsa_table = first_written["tables_dir"] / "prototype_rsa_results__per_date.parquet"
    stale_prototype_rdm_table = first_written["tables_dir"] / "prototype_rdm__pooled__full_trajectory.parquet"
    stale_prototype_support_qc = first_written["qc_dir"] / "prototype_support__per_date.parquet"

    assert stale_prototype_rsa.exists()
    assert stale_prototype_rdm_comparison.exists()
    assert stale_prototype_rdm.exists()
    assert stale_prototype_rsa_table.exists()
    assert stale_prototype_rdm_table.exists()
    assert stale_prototype_support_qc.exists()

    narrowed_outputs = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in _stage3_outputs_with_prototype_supplement().items()
    }
    narrowed_outputs["prototype_rsa_results__per_date"] = narrowed_outputs["prototype_rsa_results__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs["prototype_support__per_date"] = narrowed_outputs["prototype_support__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs["prototype_support__pooled"] = narrowed_outputs["prototype_support__pooled"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs.pop("prototype_rdm__pooled__full_trajectory")

    second_written = write_stage3_outputs(narrowed_outputs, output_root)
    summary = json.loads((second_written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (second_written["figures_dir"] / "prototype_rsa__per_date__response_window.png").exists()
    assert (second_written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
    assert (second_written["figures_dir"] / "prototype_rdm__pooled__response_window.png").exists()
    assert not stale_prototype_rsa.exists()
    assert not stale_prototype_rdm_comparison.exists()
    assert not stale_prototype_rdm.exists()
    assert not (second_written["figures_dir"] / "prototype_rsa__per_date__full_trajectory.png").exists()
    assert not (second_written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png").exists()
    assert not (second_written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png").exists()
    assert (second_written["tables_dir"] / "prototype_rsa_results__per_date.parquet").exists()
    assert (second_written["tables_dir"] / "prototype_rdm__pooled__response_window.parquet").exists()
    assert not stale_prototype_rdm_table.exists()
    assert (second_written["qc_dir"] / "prototype_support__per_date.parquet").exists()
    assert (second_written["qc_dir"] / "prototype_support__pooled.parquet").exists()
    assert stale_prototype_support_qc.exists()
    narrowed_prototype_rsa = pd.read_parquet(second_written["tables_dir"] / "prototype_rsa_results__per_date.parquet")
    narrowed_prototype_support = pd.read_parquet(second_written["qc_dir"] / "prototype_support__per_date.parquet")
    narrowed_prototype_support_pooled = pd.read_parquet(second_written["qc_dir"] / "prototype_support__pooled.parquet")
    assert narrowed_prototype_rsa["view_name"].astype(str).unique().tolist() == ["response_window"]
    assert narrowed_prototype_support["view_name"].astype(str).unique().tolist() == ["response_window"]
    assert narrowed_prototype_support_pooled["view_name"].astype(str).unique().tolist() == ["response_window"]
    assert summary["prototype_views"] == ["response_window"]
    assert summary["prototype_figure_names"] == [
        "prototype_rsa__per_date__response_window",
        "prototype_rdm_comparison__per_date__response_window",
        "prototype_rdm__pooled__response_window",
    ]
    assert summary["prototype_descriptive_outputs"] == [
        "prototype_rdm__pooled__response_window",
    ]


def test_write_stage3_outputs_writes_empty_per_date_prototype_comparison_for_view_without_per_date_data(tmp_path):
    outputs = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in _stage3_outputs_with_prototype_supplement().items()
    }
    outputs["prototype_rsa_results__per_date"] = outputs["prototype_rsa_results__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    outputs["prototype_support__per_date"] = outputs["prototype_support__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    for key in list(outputs):
        if key.startswith("internal__prototype_rdm__per_date__full_trajectory__"):
            outputs.pop(key)

    written = write_stage3_outputs(outputs, tmp_path / "stage3_rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "prototype_rdm_comparison__per_date__full_trajectory.png").exists()
    assert summary["prototype_figure_names"] == [
        "prototype_rsa__per_date__response_window",
        "prototype_rdm_comparison__per_date__response_window",
        "prototype_rdm_comparison__per_date__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]


def test_write_stage3_outputs_removes_stale_prototype_artifacts_when_rerun_without_supplement(
    tmp_path, synthetic_stage3_outputs
):
    output_root = tmp_path / "stage3_rsa"

    first_written = write_stage3_outputs(_stage3_outputs_with_prototype_supplement(), output_root)
    stale_paths = [
        first_written["tables_dir"] / "prototype_rsa_results__per_date.parquet",
        first_written["tables_dir"] / "prototype_rdm__pooled__response_window.parquet",
        first_written["tables_dir"] / "prototype_rdm__pooled__full_trajectory.parquet",
        first_written["qc_dir"] / "prototype_support__per_date.parquet",
        first_written["qc_dir"] / "prototype_support__pooled.parquet",
        first_written["figures_dir"] / "prototype_rsa__per_date__response_window.png",
        first_written["figures_dir"] / "prototype_rsa__per_date__full_trajectory.png",
        first_written["figures_dir"] / "prototype_rdm__pooled__response_window.png",
        first_written["figures_dir"] / "prototype_rdm__pooled__full_trajectory.png",
    ]

    for stale_path in stale_paths:
        assert stale_path.exists()

    second_written = write_stage3_outputs(synthetic_stage3_outputs, output_root)
    summary = json.loads((second_written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    for stale_path in stale_paths:
        assert not stale_path.exists()
    assert summary["prototype_supplement_enabled"] is False
    assert summary["prototype_views"] == []
    assert summary["prototype_dates"] == []
    assert summary["prototype_table_names"] == []
    assert summary["prototype_figure_names"] == []
    assert summary["prototype_descriptive_outputs"] == []


def test_run_stage3_rsa_marks_tiny_primary_models_excluded_from_primary_ranking():
    resolved_inputs = _resolved_stage3_inputs(
        matrix_rows=[
            {"sample_id": "A001", "f1": 1.0, "f2": 2.0, "f3": 0.5},
            {"sample_id": "A002", "f1": 2.0, "f2": 1.5, "f3": 1.5},
            {"sample_id": "A003", "f1": 3.0, "f2": 4.0, "f3": 2.5},
        ],
        mapping_rows=[
            {"stimulus": "A001", "stim_name": "Stimulus A001", "sample_id": "A001"},
            {"stimulus": "A002", "stim_name": "Stimulus A002", "sample_id": "A002"},
            {"stimulus": "A003", "stim_name": "Stimulus A003", "sample_id": "A003"},
        ],
        registry_rows=[
            {
                "model_id": "global_profile",
                "model_label": "Global Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
            },
            {
                "model_id": "tiny_primary",
                "model_label": "Tiny Primary",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
            },
        ],
        membership_rows=[
            {"model_id": "tiny_primary", "metabolite_name": "f1"},
            {"model_id": "tiny_primary", "metabolite_name": "f2"},
        ],
    )

    results = run_stage3_rsa(resolved_inputs, neural_matrices=_stage3_neural_rdms(), permutations=10, seed=0)

    excluded = results["model_registry_resolved"].loc[lambda df: df["model_id"] == "tiny_primary"]
    assert bool(excluded["excluded_from_primary_ranking"].iloc[0])


def test_run_stage3_rsa_keeps_global_profile_when_curated_subset_membership_is_empty():
    resolved_inputs = _resolved_stage3_inputs(
        matrix_rows=[
            {"sample_id": "A001", "f1": 1.0, "f2": 2.0, "f3": 0.5, "f4": 3.0, "f5": 4.5},
            {"sample_id": "A002", "f1": 2.5, "f2": 1.0, "f3": 1.5, "f4": 2.0, "f5": 3.0},
            {"sample_id": "A003", "f1": 3.0, "f2": 4.0, "f3": 2.5, "f4": 1.0, "f5": 2.0},
        ],
        mapping_rows=[
            {"stimulus": "A001", "stim_name": "Stimulus A001", "sample_id": "A001"},
            {"stimulus": "A002", "stim_name": "Stimulus A002", "sample_id": "A002"},
            {"stimulus": "A003", "stim_name": "Stimulus A003", "sample_id": "A003"},
        ],
        registry_rows=[
            {
                "model_id": "global_profile",
                "model_label": "Global Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
            },
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "draft",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
            },
        ],
        membership_rows=[],
    )

    results = run_stage3_rsa(resolved_inputs, neural_matrices=_stage3_neural_rdms(), permutations=10, seed=0)

    assert "global_profile" in set(results["rsa_results"]["model_id"])
    assert "bile_acid" not in set(results["rsa_results"]["model_id"])


def test_run_stage3_rsa_adds_prototype_tables_when_prototype_inputs_are_present():
    results = run_stage3_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        prototype_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    assert "prototype_rsa_results__per_date" in results
    assert "prototype_support__per_date" in results
    assert "prototype_support__pooled" in results
    assert "prototype_rdm__pooled__response_window" in results
    assert "prototype_rdm__pooled__full_trajectory" in results
    assert {
        "internal__prototype_rdm__per_date__response_window__2026-03-11",
        "internal__prototype_rdm__per_date__response_window__2026-03-13",
        "internal__prototype_rdm__per_date__full_trajectory__2026-03-11",
        "internal__prototype_rdm__per_date__full_trajectory__2026-03-13",
    }.issubset(results)
    assert results["internal__prototype_rdm__per_date__response_window__2026-03-11"].columns.tolist() == [
        "stimulus_row",
        "A001",
        "A002",
        "A003",
    ]
    assert "n_dates_contributed" in results["prototype_support__pooled"].columns
    assert {
        "date",
        "view_name",
        "reference_view_name",
        "comparison_scope",
        "model_id",
        "model_label",
        "model_tier",
        "model_status",
        "feature_kind",
        "distance_kind",
        "excluded_from_primary_ranking",
        "score_method",
        "score_status",
        "n_stimuli",
        "n_shared_entries",
        "rsa_similarity",
        "p_value_raw",
        "p_value_fdr",
        "is_top_model",
    }.issubset(results["prototype_rsa_results__per_date"].columns)


def test_run_stage3_rsa_returns_empty_prototype_outputs_with_expected_schema_for_empty_views():
    results = run_stage3_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        prototype_inputs=_empty_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    assert results["prototype_rsa_results__per_date"].empty
    assert results["prototype_rsa_results__per_date"].columns.tolist() == [
        "date",
        "view_name",
        "reference_view_name",
        "comparison_scope",
        "model_id",
        "model_label",
        "model_tier",
        "model_status",
        "feature_kind",
        "distance_kind",
        "excluded_from_primary_ranking",
        "score_method",
        "score_status",
        "n_stimuli",
        "n_shared_entries",
        "rsa_similarity",
        "p_value_raw",
        "p_value_fdr",
        "is_top_model",
    ]
    assert results["prototype_support__per_date"].empty
    assert results["prototype_support__per_date"].columns.tolist() == [
        "date",
        "view_name",
        "stimulus",
        "stim_name",
        "n_trials",
        "n_total_features",
        "n_supported_features",
        "n_all_nan_features",
    ]
    assert results["prototype_support__pooled"].empty
    assert results["prototype_support__pooled"].columns.tolist() == [
        "view_name",
        "stimulus",
        "stim_name",
        "n_trials",
        "n_dates_contributed",
        "n_total_features",
        "n_supported_features",
        "n_all_nan_features",
    ]
    assert results["prototype_rdm__pooled__response_window"].columns.tolist() == ["stimulus_row"]
    assert results["prototype_rdm__pooled__full_trajectory"].columns.tolist() == ["stimulus_row"]


def test_run_stage3_rsa_keeps_primary_rsa_results_unchanged_when_prototype_inputs_are_present():
    baseline = run_stage3_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        permutations=10,
        seed=0,
    )
    supplemented = run_stage3_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        prototype_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    pd.testing.assert_frame_equal(baseline["rsa_results"], supplemented["rsa_results"])
    pd.testing.assert_frame_equal(baseline["rsa_view_comparison"], supplemented["rsa_view_comparison"])
    pd.testing.assert_frame_equal(baseline["rsa_leave_one_stimulus_out"], supplemented["rsa_leave_one_stimulus_out"])


def test_run_stage3_rsa_corrects_prototype_fdr_across_full_table_and_ignores_excluded_rows_for_top_model():
    results = run_stage3_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        prototype_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    prototype = results["prototype_rsa_results__per_date"]
    finite = prototype.loc[prototype["p_value_raw"].notna()].reset_index(drop=True)
    expected_fdr = benjamini_hochberg(finite["p_value_raw"].to_numpy(dtype=float))

    np.testing.assert_allclose(finite["p_value_fdr"].to_numpy(dtype=float), expected_fdr)
    assert not prototype.loc[prototype["excluded_from_primary_ranking"].astype(bool), "is_top_model"].any()

    for _, group in prototype.groupby(["date", "view_name"], sort=False):
        eligible = group.loc[
            group["score_status"].astype(str).eq("ok")
            & ~group["excluded_from_primary_ranking"].astype(bool)
            & np.isfinite(pd.to_numeric(group["rsa_similarity"], errors="coerce"))
        ]
        if eligible.empty:
            continue
        assert int(eligible["is_top_model"].sum()) == 1


def test_run_stage3_rsa_restricts_model_rdms_by_date_stimulus_set_and_marks_sparse_rows_invalid():
    resolved_inputs = _stage3_prototype_resolved_inputs()
    prototype_inputs = _stage3_prototype_inputs()
    results = run_stage3_rsa(
        resolved_inputs,
        neural_matrices=_stage3_neural_rdms(),
        prototype_inputs=prototype_inputs,
        permutations=10,
        seed=0,
    )

    prototype = results["prototype_rsa_results__per_date"]
    restricted_row = prototype.loc[
        prototype["date"].eq("2026-03-11")
        & prototype["view_name"].eq("response_window")
        & prototype["model_id"].eq("global_profile")
    ].iloc[0]
    view = prototype_inputs.views["response_window"]
    per_date_prototypes, _ = build_grouped_prototypes(
        view,
        group_columns=("date", "stimulus", "stim_name"),
    )
    date_prototypes = per_date_prototypes.loc[per_date_prototypes["date"].eq("2026-03-11")].reset_index(drop=True)
    stimulus_labels = date_prototypes["stimulus"].astype(str).tolist()
    neural_matrix = build_prototype_rdm(date_prototypes, id_columns=("stimulus",))
    full_model_matrix = build_model_rdm(resolved_inputs, "global_profile")
    restricted_model_matrix = (
        full_model_matrix.set_index("stimulus_row")
        .loc[stimulus_labels, stimulus_labels]
        .pipe(lambda frame: frame.assign(stimulus_row=frame.index).reset_index(drop=True))
    )

    restricted_score = compute_rsa_score(neural_matrix, restricted_model_matrix)
    restricted_null = build_permutation_null(neural_matrix, restricted_model_matrix, n_iterations=10, seed=0)
    restricted_null_values = restricted_null["rsa_similarity"].to_numpy(dtype=float)
    restricted_null_values = restricted_null_values[np.isfinite(restricted_null_values)]
    expected_restricted_p = float(
        (1 + np.sum(restricted_null_values >= restricted_score["rsa_similarity"])) / (restricted_null_values.size + 1)
    )

    unrestricted_null = build_permutation_null(neural_matrix, full_model_matrix, n_iterations=10, seed=0)
    unrestricted_null_values = unrestricted_null["rsa_similarity"].to_numpy(dtype=float)
    unrestricted_null_values = unrestricted_null_values[np.isfinite(unrestricted_null_values)]
    unrestricted_p = float(
        (1 + np.sum(unrestricted_null_values >= restricted_score["rsa_similarity"])) / (unrestricted_null_values.size + 1)
    )

    assert restricted_row["n_stimuli"] == 3
    assert restricted_row["score_status"] == "ok"
    assert restricted_row["n_shared_entries"] == 3
    assert restricted_row["p_value_raw"] == pytest.approx(expected_restricted_p)
    assert unrestricted_p != pytest.approx(expected_restricted_p)

    sparse_rows = prototype.loc[prototype["date"].eq("2026-03-13")].reset_index(drop=True)

    assert not sparse_rows.empty
    assert sparse_rows["n_stimuli"].eq(2).all()
    assert sparse_rows["score_status"].eq("invalid").all()
    assert sparse_rows["n_shared_entries"].fillna(0).astype(int).max() < 2

