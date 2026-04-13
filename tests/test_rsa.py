import json

import bacteria_analysis.rsa as rsa_module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.rsa import (
    align_rdm_upper_triangles,
    benjamini_hochberg,
    build_permutation_null,
    compute_rsa_score,
    run_biochemical_rsa,
    summarize_cross_view_comparison,
    summarize_leave_one_stimulus_out,
)
from bacteria_analysis import rsa_outputs
from bacteria_analysis.model_space import build_model_rdm
from bacteria_analysis.reliability import TrialView
from bacteria_analysis.rsa_prototypes import PrototypeContextInputs, build_grouped_prototypes, build_prototype_rdm
from bacteria_analysis.rsa_outputs import write_rsa_outputs


def test_rsa_aliases_remain_available():
    assert rsa_module.run_stage3_rsa is rsa_module.run_biochemical_rsa
    assert rsa_module.load_stage2_pooled_neural_rdms is rsa_module.load_geometry_pooled_neural_rdms
    assert rsa_outputs.write_stage3_outputs is write_rsa_outputs
    assert rsa_outputs.ensure_stage3_output_dirs is rsa_outputs.ensure_rsa_output_dirs


def _matrix(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _markdown_section(markdown: str, heading: str) -> list[str]:
    lines = markdown.splitlines()
    start_index = lines.index(heading)
    section_lines: list[str] = []
    for line in lines[start_index + 1 :]:
        if line.startswith("## "):
            break
        if line:
            section_lines.append(line)
    return section_lines


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


def _stage3_prototype_inputs() -> PrototypeContextInputs:
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
    return PrototypeContextInputs(
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


def _empty_stage3_prototype_inputs() -> PrototypeContextInputs:
    metadata = pd.DataFrame(columns=["date", "stimulus", "stim_name"])
    empty_values = np.empty((0, 1, 4), dtype=float)
    return PrototypeContextInputs(
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
    return run_biochemical_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        aggregated_response_inputs=_stage3_prototype_inputs(),
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


def test_write_rsa_outputs_writes_required_tables(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")

    assert (written["tables_dir"] / "stimulus_sample_map.parquet").exists()
    assert (written["tables_dir"] / "rsa_results.parquet").exists()
    assert (written["tables_dir"] / "model_registry_resolved.parquet").exists()
    assert (written["tables_dir"] / "rsa_leave_one_stimulus_out.parquet").exists()
    assert (written["qc_dir"] / "model_input_coverage.parquet").exists()
    assert (written["qc_dir"] / "model_feature_filtering.parquet").exists()
    assert (written["figures_dir"] / "single_stimulus_sensitivity.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert (written["output_root"] / "run_summary.json").exists()
    assert (written["output_root"] / "run_summary.md").exists()


def test_write_rsa_outputs_removes_stale_summary_figures(tmp_path, synthetic_stage3_outputs):
    output_root = tmp_path / "rsa"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stale_paths = [
        figures_dir / "ranked_model_rsa.png",
        figures_dir / "view_comparison_summary.png",
        figures_dir / "leave_one_stimulus_out_robustness.png",
    ]
    for stale_path in stale_paths:
        stale_path.write_bytes(b"legacy figure")

    written = write_rsa_outputs(synthetic_stage3_outputs, output_root)

    assert (written["figures_dir"] / "single_stimulus_sensitivity.png").exists()
    for stale_path in stale_paths:
        assert not stale_path.exists()


def test_write_rsa_outputs_writes_per_view_neural_model_figures(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")

    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert not (written["figures_dir"] / "neural_vs_top_model_rdm_panel.png").exists()


def test_write_rsa_outputs_removes_legacy_panel_figure(tmp_path, synthetic_stage3_outputs):
    output_root = tmp_path / "rsa"
    legacy_panel = output_root / "figures" / "neural_vs_top_model_rdm_panel.png"
    legacy_panel.parent.mkdir(parents=True, exist_ok=True)
    legacy_panel.write_bytes(b"legacy panel")

    written = write_rsa_outputs(synthetic_stage3_outputs, output_root)

    assert not legacy_panel.exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()


def test_write_rsa_outputs_removes_stale_per_view_figures_when_view_set_narrows(
    tmp_path, synthetic_stage3_outputs
):
    output_root = tmp_path / "rsa"

    first_written = write_rsa_outputs(synthetic_stage3_outputs, output_root)
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

    second_written = write_rsa_outputs(narrowed_outputs, output_root)

    assert (second_written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert not stale_full_trajectory.exists()
    assert not (second_written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert not (second_written["figures_dir"] / "neural_vs_top_model_rdm_panel.png").exists()


def test_write_rsa_outputs_reports_per_view_figure_names(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["figure_names"] == [
        "single_stimulus_sensitivity",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]


def test_write_rsa_outputs_reports_canonical_per_view_figure_name_order(tmp_path, synthetic_stage3_outputs):
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

    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["figure_names"] == [
        "single_stimulus_sensitivity",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
        "neural_vs_top_model_rdm__alpha_view",
        "neural_vs_top_model_rdm__sorted_view",
    ]


def test_write_rsa_outputs_writes_default_per_view_placeholders_when_views_missing(tmp_path, synthetic_stage3_outputs):
    synthetic_stage3_outputs["rsa_results"] = pd.DataFrame()
    synthetic_stage3_outputs["cross_view_comparison"] = pd.DataFrame()

    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))
    markdown = (written["output_root"] / "run_summary.md").read_text(encoding="utf-8")

    assert (written["figures_dir"] / "neural_vs_top_model_rdm__response_window.png").exists()
    assert (written["figures_dir"] / "neural_vs_top_model_rdm__full_trajectory.png").exists()
    assert summary["views"] == []
    assert summary["ranked_model_rsa_details"] == []
    assert summary["view_comparison_details"] == []
    assert summary["figure_names"] == [
        "single_stimulus_sensitivity",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]
    assert _markdown_section(markdown, "## Ranked Model RSA Details") == ["- None"]
    assert _markdown_section(markdown, "## View Comparison Details") == ["- None"]


def test_write_rsa_outputs_markdown_summary_includes_detail_sections(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    markdown = (written["output_root"] / "run_summary.md").read_text(encoding="utf-8")

    ranked_section = _markdown_section(markdown, "## Ranked Model RSA Details")
    view_comparison_section = _markdown_section(markdown, "## View Comparison Details")

    assert ranked_section == [
        "- global_profile | view=response_window | rsa=0.94 | p_raw=0.01 | p_fdr=0.015 | n=3 | status=ok | top=True",
        "- bile_acid | view=response_window | rsa=0.88 | p_raw=0.02 | p_fdr=0.025 | n=3 | status=ok | top=False",
    ]
    assert view_comparison_section == [
        "- response_window vs full_trajectory | scope=neural_vs_neural | rsa=0.98 | p_raw=0.01 | p_fdr=0.015 | n=3 | status=ok",
        "- full_trajectory vs response_window | scope=neural_vs_neural | rsa=0.98 | p_raw=0.01 | p_fdr=0.015 | n=3 | status=ok",
    ]


def test_write_rsa_outputs_records_ranked_and_additional_models(tmp_path, synthetic_stage3_outputs):
    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["focus_view"] == "response_window"
    assert summary["ranked_models"] == ["global_profile", "bile_acid"]
    assert summary["additional_models"] == ["lipid_panel"]
    assert summary["excluded_models"] == ["excluded_sparse"]
    assert summary["ranked_model_rsa_details"][0]["view_name"] == "response_window"
    assert summary["ranked_model_rsa_details"][0]["model_id"] == "global_profile"
    assert summary["view_comparison_details"][0]["view_name"] == "response_window"
    assert summary["top_models_by_view"] == {
        "response_window": "global_profile",
        "full_trajectory": "global_profile",
    }


def test_ranked_model_rsa_details_filters_focus_view_and_sorts_by_similarity_then_model_id():
    rsa_results = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "model_id": "model_b",
                "rsa_similarity": 0.91,
                "p_value_raw": 0.020,
                "p_value_fdr": 0.030,
                "n_shared_entries": 3,
                "score_status": "ok",
                "is_top_model": False,
            },
            {
                "view_name": "response_window",
                "model_id": "model_a",
                "rsa_similarity": 0.91,
                "p_value_raw": 0.010,
                "p_value_fdr": 0.015,
                "n_shared_entries": 3,
                "score_status": "ok",
                "is_top_model": True,
            },
            {
                "view_name": "response_window",
                "model_id": "model_c",
                "rsa_similarity": np.inf,
                "p_value_raw": 0.5,
                "p_value_fdr": 0.5,
                "n_shared_entries": 3,
                "score_status": "ok",
                "is_top_model": False,
            },
            {
                "view_name": "full_trajectory",
                "model_id": "model_a",
                "rsa_similarity": 0.99,
                "p_value_raw": 0.001,
                "p_value_fdr": 0.002,
                "n_shared_entries": 3,
                "score_status": "ok",
                "is_top_model": True,
            },
            {
                "view_name": "response_window",
                "model_id": "model_z",
                "rsa_similarity": 0.95,
                "p_value_raw": 0.5,
                "p_value_fdr": 0.5,
                "n_shared_entries": 3,
                "score_status": "ok",
                "is_top_model": False,
            },
        ]
    )

    details = rsa_outputs._build_ranked_model_rsa_details(
        rsa_results,
        ["model_a", "model_b"],
        focus_view="response_window",
    )

    assert details == [
        {
            "view_name": "response_window",
            "model_id": "model_a",
            "rsa_similarity": 0.91,
            "p_value_raw": 0.01,
            "p_value_fdr": 0.015,
            "n_shared_entries": 3,
            "score_status": "ok",
            "is_top_model": True,
        },
        {
            "view_name": "response_window",
            "model_id": "model_b",
            "rsa_similarity": 0.91,
            "p_value_raw": 0.02,
            "p_value_fdr": 0.03,
            "n_shared_entries": 3,
            "score_status": "ok",
            "is_top_model": False,
        },
    ]


def test_view_comparison_details_preserves_input_order_and_filters_finite_rows():
    view_comparison = pd.DataFrame.from_records(
        [
            {
                "view_name": "response_window",
                "reference_view_name": "full_trajectory",
                "comparison_scope": "neural_vs_neural",
                "rsa_similarity": 0.82,
                "p_value_raw": 0.030,
                "p_value_fdr": 0.040,
                "n_shared_entries": 3,
                "score_status": "ok",
            },
            {
                "view_name": "full_trajectory",
                "reference_view_name": "response_window",
                "comparison_scope": "neural_vs_neural",
                "rsa_similarity": np.nan,
                "p_value_raw": 0.10,
                "p_value_fdr": 0.20,
                "n_shared_entries": 3,
                "score_status": "ok",
            },
            {
                "view_name": "alpha_view",
                "reference_view_name": "response_window",
                "comparison_scope": "neural_vs_neural",
                "rsa_similarity": 0.91,
                "p_value_raw": 0.020,
                "p_value_fdr": 0.025,
                "n_shared_entries": 3,
                "score_status": "ok",
            },
        ]
    )

    details = rsa_outputs._build_view_comparison_details(view_comparison)

    assert details == [
        {
            "view_name": "response_window",
            "reference_view_name": "full_trajectory",
            "comparison_scope": "neural_vs_neural",
            "rsa_similarity": 0.82,
            "p_value_raw": 0.03,
            "p_value_fdr": 0.04,
            "n_shared_entries": 3,
            "score_status": "ok",
        },
        {
            "view_name": "alpha_view",
            "reference_view_name": "response_window",
            "comparison_scope": "neural_vs_neural",
            "rsa_similarity": 0.91,
            "p_value_raw": 0.02,
            "p_value_fdr": 0.025,
            "n_shared_entries": 3,
            "score_status": "ok",
        },
    ]


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

    def _capturing_panels(figure, axes, colorbar_axes, panels):
        model_panels = [panel for panel in panels if "global_profile" in panel[3]]
        captured["titles"] = [panel[3] for panel in model_panels]
        captured["index_labels"] = [panel[2].index.tolist() if panel[2] is not None else None for panel in model_panels]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rsa_outputs, "_render_prepared_rdm_panels", _capturing_panels)
    try:
        rsa_outputs._plot_neural_vs_top_model_rdm_view(
            core_outputs,
            {"response_window": "global_profile"},
            view_name="response_window",
            path=tmp_path / "missing_neural.png",
        )
    finally:
        monkeypatch.undo()

    assert captured["titles"] == [
        "response_window: global_profile (neural order)",
        "response_window: global_profile (model order)",
    ]
    assert captured["index_labels"][0] == captured["index_labels"][1]
    assert set(captured["index_labels"][0]) == {"S001", "S002", "S003", "S004"}


def test_create_rdm_panel_figure_reserves_separate_colorbar_column():
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=2, figsize=(10.2, 7.0))
    try:
        assert axes.shape == (2, 2)
        assert colorbar_axes.shape == (2, 2)
        assert len(figure.axes) == 8
        for row_index in range(2):
            left_panel = axes[row_index, 0].get_position()
            left_colorbar = colorbar_axes[row_index, 0].get_position()
            right_panel = axes[row_index, 1].get_position()
            right_colorbar = colorbar_axes[row_index, 1].get_position()
            assert left_panel.x1 < left_colorbar.x0
            assert left_colorbar.x1 < right_panel.x0
            assert right_panel.x1 < right_colorbar.x0
    finally:
        plt.close(figure)


def test_compute_rdm_display_parameters_uses_custom_quantiles_from_off_diagonal_values():
    heatmap_frame = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 1.0, "S3": 7.0},
            {"stimulus_row": "S2", "S1": 2.0, "S2": 0.0, "S3": 8.0},
            {"stimulus_row": "S3", "S1": 3.0, "S2": 4.0, "S3": 0.0},
        ]
    )

    display_parameters = rsa_outputs._compute_rdm_display_parameters(
        heatmap_frame,
        lower_quantile=0.25,
        upper_quantile=0.75,
    )

    assert display_parameters is not None
    expected_vmin, expected_vmax = np.quantile(np.array([1.0, 7.0, 2.0, 8.0, 3.0, 4.0]), [0.25, 0.75])
    assert display_parameters.vmin == pytest.approx(expected_vmin)
    assert display_parameters.vmax == pytest.approx(expected_vmax)
    assert isinstance(display_parameters.norm, matplotlib.colors.PowerNorm)
    assert display_parameters.norm.gamma == pytest.approx(0.7)
    assert display_parameters.norm.clip is True


def test_compute_rdm_display_parameters_widens_constant_off_diagonal_values():
    heatmap_frame = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 2.0, "S3": 2.0},
            {"stimulus_row": "S2", "S1": 2.0, "S2": 0.0, "S3": 2.0},
            {"stimulus_row": "S3", "S1": 2.0, "S2": 2.0, "S3": 0.0},
        ]
    )

    display_parameters = rsa_outputs._compute_rdm_display_parameters(heatmap_frame)

    assert display_parameters is not None
    assert display_parameters.vmin < 2.0 < display_parameters.vmax
    assert display_parameters.vmin < display_parameters.vmax
    assert display_parameters.norm.vmin == pytest.approx(display_parameters.vmin)
    assert display_parameters.norm.vmax == pytest.approx(display_parameters.vmax)


def test_compute_rdm_display_parameters_returns_none_when_no_finite_off_diagonal_values():
    heatmap_frame = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": np.nan},
            {"stimulus_row": "S2", "S1": np.nan, "S2": 0.0},
        ]
    )

    assert rsa_outputs._compute_rdm_display_parameters(heatmap_frame) is None


def test_render_prepared_rdm_panels_uses_raw_value_colorbar_label(monkeypatch):
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(10.2, 3.6))
    heatmap_frame = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 0.12, "S3": 0.91},
            {"stimulus_row": "S2", "S1": 0.12, "S2": 0.0, "S3": 0.44},
            {"stimulus_row": "S3", "S1": 0.91, "S2": 0.44, "S3": 0.0},
        ]
    )
    panels = [
        (0, 0, heatmap_frame, "left", "empty"),
        (0, 1, heatmap_frame, "right", "empty"),
    ]
    captured_labels: list[str | None] = []
    original_colorbar = figure.colorbar

    def _capturing_colorbar(*args, **kwargs):
        captured_labels.append(kwargs.get("label"))
        return original_colorbar(*args, **kwargs)

    monkeypatch.setattr(figure, "colorbar", _capturing_colorbar)
    try:
        rsa_outputs._render_prepared_rdm_panels(figure, axes, colorbar_axes, panels)
    finally:
        plt.close(figure)

    assert captured_labels == ["RDM dissimilarity", "RDM dissimilarity"]


def test_render_prepared_rdm_panels_uses_panel_local_norms():
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(10.2, 3.6))
    left_panel = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 0.10, "S3": 0.30},
            {"stimulus_row": "S2", "S1": 0.10, "S2": 0.0, "S3": 0.20},
            {"stimulus_row": "S3", "S1": 0.30, "S2": 0.20, "S3": 0.0},
        ]
    )
    right_panel = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 10.0, "S3": 30.0},
            {"stimulus_row": "S2", "S1": 10.0, "S2": 0.0, "S3": 20.0},
            {"stimulus_row": "S3", "S1": 30.0, "S2": 20.0, "S3": 0.0},
        ]
    )
    try:
        rsa_outputs._render_prepared_rdm_panels(
            figure,
            axes,
            colorbar_axes,
            [
                (0, 0, left_panel, "left", "No data"),
                (0, 1, right_panel, "right", "No data"),
            ],
        )
        left_norm = axes[0, 0].images[0].norm
        right_norm = axes[0, 1].images[0].norm
        assert left_norm.vmin != right_norm.vmin
        assert left_norm.vmax != right_norm.vmax
        assert left_norm.vmax < right_norm.vmax
    finally:
        plt.close(figure)


def test_render_prepared_rdm_panels_hides_empty_panel_colorbar_axis():
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(10.2, 3.6))
    empty_panel = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": np.nan},
            {"stimulus_row": "S2", "S1": np.nan, "S2": 0.0},
        ]
    )
    non_empty_panel = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 0.3},
            {"stimulus_row": "S2", "S1": 0.3, "S2": 0.0},
        ]
    )
    panels = [
        (0, 0, empty_panel, "empty", "No data"),
        (0, 1, non_empty_panel, "full", "No data"),
    ]
    try:
        rsa_outputs._render_prepared_rdm_panels(figure, axes, colorbar_axes, panels)
        assert not colorbar_axes[0, 0].get_visible()
        assert colorbar_axes[0, 1].get_visible()
    finally:
        plt.close(figure)


def test_render_prepared_rdm_panels_hides_colorbar_axis_when_colorbar_creation_fails(monkeypatch):
    figure, axes, colorbar_axes = rsa_outputs._create_rdm_panel_figure(nrows=1, figsize=(10.2, 3.6))
    left_panel = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 0.1},
            {"stimulus_row": "S2", "S1": 0.1, "S2": 0.0},
        ]
    )
    right_panel = _matrix(
        [
            {"stimulus_row": "S1", "S1": 0.0, "S2": 0.4},
            {"stimulus_row": "S2", "S1": 0.4, "S2": 0.0},
        ]
    )
    original_colorbar = figure.colorbar
    call_count = {"value": 0}

    def _failing_first_colorbar(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("colorbar failed")
        return original_colorbar(*args, **kwargs)

    monkeypatch.setattr(figure, "colorbar", _failing_first_colorbar)
    try:
        with pytest.warns(RuntimeWarning, match="RDM colorbar failed for panel"):
            rsa_outputs._render_prepared_rdm_panels(
                figure,
                axes,
                colorbar_axes,
                [
                    (0, 0, left_panel, "left", "No data"),
                    (0, 1, right_panel, "right", "No data"),
                ],
            )
        assert not colorbar_axes[0, 0].get_visible()
        assert colorbar_axes[0, 1].get_visible()
        assert len(axes[0, 0].images) == 1
        assert len(axes[0, 1].images) == 1
    finally:
        plt.close(figure)


def test_build_top_aggregated_response_models_by_date_and_view():
    aggregated_response_rsa_results = pd.DataFrame.from_records(
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

    top_models = rsa_outputs._build_top_aggregated_response_models_by_date_and_view(
        aggregated_response_rsa_results
    )

    assert top_models == {
        ("2026-03-11", "response_window"): "global_profile",
        ("2026-03-13", "full_trajectory"): "global_profile",
    }


def test_plot_aggregated_response_rdm_comparison_per_date_uses_internal_rdms_when_rsa_table_is_empty(tmp_path):
    core_outputs = _stage3_outputs_with_prototype_supplement()
    captured: list[tuple[str, bool]] = []

    def _capturing_panels(figure, axes, colorbar_axes, panels):
        captured.extend((title, frame is not None) for _, _, frame, title, _ in panels)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rsa_outputs, "_render_prepared_rdm_panels", _capturing_panels)
    try:
        written_path = rsa_outputs._plot_aggregated_response_rdm_comparison_per_date(
            core_outputs,
            pd.DataFrame(),
            {},
            view_name="response_window",
            path=tmp_path / "aggregated_response_rdm_comparison__per_date__response_window.png",
        )
    finally:
        monkeypatch.undo()

    assert written_path.exists()
    assert captured == [
        ("2026-03-11: aggregated response (neural order)", True),
        ("2026-03-11: no top model (neural order)", False),
        ("2026-03-11: aggregated response (model order)", True),
        ("2026-03-11: no top model (model order)", False),
        ("2026-03-13: aggregated response (neural order)", True),
        ("2026-03-13: no top model (neural order)", False),
        ("2026-03-13: aggregated response (model order)", True),
        ("2026-03-13: no top model (model order)", False),
    ]


def test_plot_aggregated_response_pooled_rdm_uses_both_neural_and_model_order_rows(tmp_path):
    stimulus_sample_map = pd.DataFrame.from_records(
        [
            {"stimulus": "b3_1", "stim_name": "Stimulus B3", "sample_id": "S003"},
            {"stimulus": "b1_1", "stim_name": "Stimulus B1", "sample_id": "S001"},
            {"stimulus": "b2_1", "stim_name": "Stimulus B2", "sample_id": "S002"},
        ]
    )
    aggregated_response_rdm = _matrix(
        [
            {"stimulus_row": "b3_1", "b3_1": 0.0, "b1_1": 0.6},
            {"stimulus_row": "b1_1", "b3_1": 0.6, "b1_1": 0.0},
        ]
    )
    model_matrix = _matrix(
        [
            {"stimulus_row": "b2_1", "b2_1": 0.0, "b3_1": 0.2, "b1_1": 0.5},
            {"stimulus_row": "b3_1", "b2_1": 0.2, "b3_1": 0.0, "b1_1": 0.3},
            {"stimulus_row": "b1_1", "b2_1": 0.5, "b3_1": 0.3, "b1_1": 0.0},
        ]
    )
    core_outputs = {
        "stimulus_sample_map": stimulus_sample_map,
        "model_rdm__response_window__global_profile": model_matrix,
    }
    captured: dict[str, object] = {}

    def _capturing_panels(figure, axes, colorbar_axes, panels):
        captured["titles"] = [panel[3] for panel in panels]
        captured["model_indexes"] = [
            panel[2].index.tolist()
            for panel in panels
            if panel[2] is not None and "global_profile" in panel[3]
        ]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rsa_outputs, "_render_prepared_rdm_panels", _capturing_panels)
    try:
        written_path = rsa_outputs._plot_aggregated_response_pooled_rdm(
            core_outputs,
            aggregated_response_rdm,
            {"response_window": "global_profile"},
            stimulus_sample_map=stimulus_sample_map,
            view_name="response_window",
            path=tmp_path / "aggregated_response_rdm__pooled__response_window.png",
        )
    finally:
        monkeypatch.undo()

    assert written_path.exists()
    assert captured["titles"] == [
        "response_window: pooled aggregated response (neural order)",
        "response_window: global_profile (neural order)",
        "response_window: pooled aggregated response (model order)",
        "response_window: global_profile (model order)",
    ]
    assert captured["model_indexes"] == [
        ["S003", "S001"],
        ["S003", "S001"],
    ]


def test_write_rsa_outputs_keeps_skipped_additional_models_in_additional_bucket(
    tmp_path, synthetic_stage3_outputs
):
    registry = synthetic_stage3_outputs["model_registry_resolved"].copy()
    registry.loc[registry["model_id"] == "lipid_panel", "excluded_from_primary_ranking"] = True
    synthetic_stage3_outputs["model_registry_resolved"] = registry

    written = write_rsa_outputs(synthetic_stage3_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert summary["additional_models"] == ["lipid_panel"]
    assert summary["excluded_models"] == ["excluded_sparse"]


def test_write_rsa_outputs_writes_per_date_prototype_comparison_figures(tmp_path):
    written = write_rsa_outputs(_stage3_outputs_with_prototype_supplement(), tmp_path / "rsa")

    assert (written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__full_trajectory.png").exists()


def test_write_rsa_outputs_keeps_pooled_prototype_filenames(tmp_path):
    written = write_rsa_outputs(_stage3_outputs_with_prototype_supplement(), tmp_path / "rsa")

    assert sorted(path.name for path in written["figures_dir"].glob("aggregated_response_rdm__pooled__*.png")) == [
        "aggregated_response_rdm__pooled__full_trajectory.png",
        "aggregated_response_rdm__pooled__response_window.png",
    ]


def test_write_rsa_outputs_writes_prototype_context_artifacts(tmp_path):
    core_outputs = _stage3_outputs_with_prototype_supplement()

    written = write_rsa_outputs(core_outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (written["tables_dir"] / "aggregated_response_rsa_results__per_date.parquet").exists()
    assert (written["tables_dir"] / "aggregated_response_rdm__pooled__response_window.parquet").exists()
    assert (written["tables_dir"] / "aggregated_response_rdm__pooled__full_trajectory.parquet").exists()
    assert (written["qc_dir"] / "aggregated_response_support__per_date.parquet").exists()
    assert (written["qc_dir"] / "aggregated_response_support__pooled.parquet").exists()
    assert (written["figures_dir"] / "aggregated_response_rsa__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rsa__per_date__full_trajectory.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__full_trajectory.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rdm__pooled__response_window.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rdm__pooled__full_trajectory.png").exists()
    assert summary["rsa_table_names"] == [
        "model_rdm_summary",
        "rsa_results",
        "rsa_leave_one_stimulus_out",
        "rsa_view_comparison",
    ]
    assert summary["additional_table_names"] == [
        "aggregated_response_rdm__pooled__full_trajectory",
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rsa_results__per_date",
        "model_rdm__excluded_panel",
        "model_rdm__focused_panel",
        "model_rdm__global_profile",
        "neural_rdm__full_trajectory",
        "neural_rdm__response_window",
    ]
    assert not any(key.startswith("internal__") for key in written)
    assert not any(name.startswith("internal__") for name in summary["additional_table_names"])
    assert not list((written["tables_dir"]).glob("internal__*.parquet"))
    assert summary["aggregated_response_context_enabled"] is True
    assert summary["response_aggregation"] == "mean"
    assert summary["aggregated_response_views"] == ["response_window", "full_trajectory"]
    assert summary["aggregated_response_dates"] == ["2026-03-11", "2026-03-13"]
    assert summary["aggregated_response_table_names"] == [
        "aggregated_response_rsa_results__per_date",
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rdm__pooled__full_trajectory",
    ]
    assert summary["aggregated_response_figure_names"] == [
        "aggregated_response_rsa__per_date__response_window",
        "aggregated_response_rsa__per_date__full_trajectory",
        "aggregated_response_rdm_comparison__per_date__response_window",
        "aggregated_response_rdm_comparison__per_date__full_trajectory",
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rdm__pooled__full_trajectory",
    ]
    assert summary["aggregated_response_descriptive_outputs"] == [
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rdm__pooled__full_trajectory",
    ]
    assert summary["figure_names"] == [
        "single_stimulus_sensitivity",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
        "aggregated_response_rsa__per_date__response_window",
        "aggregated_response_rsa__per_date__full_trajectory",
        "aggregated_response_rdm_comparison__per_date__response_window",
        "aggregated_response_rdm_comparison__per_date__full_trajectory",
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rdm__pooled__full_trajectory",
    ]


def test_write_rsa_outputs_removes_stale_prototype_figures_when_view_set_narrows(tmp_path):
    output_root = tmp_path / "rsa"

    first_written = write_rsa_outputs(_stage3_outputs_with_prototype_supplement(), output_root)
    stale_prototype_rsa = first_written["figures_dir"] / "aggregated_response_rsa__per_date__full_trajectory.png"
    stale_prototype_rdm_comparison = first_written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__full_trajectory.png"
    stale_prototype_rdm = first_written["figures_dir"] / "aggregated_response_rdm__pooled__full_trajectory.png"
    stale_prototype_rsa_table = first_written["tables_dir"] / "aggregated_response_rsa_results__per_date.parquet"
    stale_prototype_rdm_table = first_written["tables_dir"] / "aggregated_response_rdm__pooled__full_trajectory.parquet"
    stale_prototype_support_qc = first_written["qc_dir"] / "aggregated_response_support__per_date.parquet"

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
    narrowed_outputs["aggregated_response_rsa_results__per_date"] = narrowed_outputs["aggregated_response_rsa_results__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs["aggregated_response_support__per_date"] = narrowed_outputs["aggregated_response_support__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs["aggregated_response_support__pooled"] = narrowed_outputs["aggregated_response_support__pooled"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    narrowed_outputs.pop("aggregated_response_rdm__pooled__full_trajectory")

    second_written = write_rsa_outputs(narrowed_outputs, output_root)
    summary = json.loads((second_written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (second_written["figures_dir"] / "aggregated_response_rsa__per_date__response_window.png").exists()
    assert (second_written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__response_window.png").exists()
    assert (second_written["figures_dir"] / "aggregated_response_rdm__pooled__response_window.png").exists()
    assert not stale_prototype_rsa.exists()
    assert not stale_prototype_rdm_comparison.exists()
    assert not stale_prototype_rdm.exists()
    assert not (second_written["figures_dir"] / "aggregated_response_rsa__per_date__full_trajectory.png").exists()
    assert not (second_written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__full_trajectory.png").exists()
    assert not (second_written["figures_dir"] / "aggregated_response_rdm__pooled__full_trajectory.png").exists()
    assert (second_written["tables_dir"] / "aggregated_response_rsa_results__per_date.parquet").exists()
    assert (second_written["tables_dir"] / "aggregated_response_rdm__pooled__response_window.parquet").exists()
    assert not stale_prototype_rdm_table.exists()
    assert (second_written["qc_dir"] / "aggregated_response_support__per_date.parquet").exists()
    assert (second_written["qc_dir"] / "aggregated_response_support__pooled.parquet").exists()
    assert stale_prototype_support_qc.exists()
    narrowed_prototype_rsa = pd.read_parquet(second_written["tables_dir"] / "aggregated_response_rsa_results__per_date.parquet")
    narrowed_prototype_support = pd.read_parquet(second_written["qc_dir"] / "aggregated_response_support__per_date.parquet")
    narrowed_prototype_support_pooled = pd.read_parquet(second_written["qc_dir"] / "aggregated_response_support__pooled.parquet")
    assert narrowed_prototype_rsa["view_name"].astype(str).unique().tolist() == ["response_window"]
    assert narrowed_prototype_support["view_name"].astype(str).unique().tolist() == ["response_window"]
    assert narrowed_prototype_support_pooled["view_name"].astype(str).unique().tolist() == ["response_window"]
    assert summary["aggregated_response_views"] == ["response_window"]
    assert summary["aggregated_response_figure_names"] == [
        "aggregated_response_rsa__per_date__response_window",
        "aggregated_response_rdm_comparison__per_date__response_window",
        "aggregated_response_rdm__pooled__response_window",
    ]
    assert summary["aggregated_response_descriptive_outputs"] == [
        "aggregated_response_rdm__pooled__response_window",
    ]


def test_write_rsa_outputs_writes_empty_per_date_prototype_comparison_for_view_without_per_date_data(tmp_path):
    outputs = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in _stage3_outputs_with_prototype_supplement().items()
    }
    outputs["aggregated_response_rsa_results__per_date"] = outputs["aggregated_response_rsa_results__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    outputs["aggregated_response_support__per_date"] = outputs["aggregated_response_support__per_date"].loc[
        lambda frame: frame["view_name"] == "response_window"
    ].copy()
    for key in list(outputs):
        if key.startswith("internal__aggregated_response_rdm__per_date__full_trajectory__"):
            outputs.pop(key)

    written = write_rsa_outputs(outputs, tmp_path / "rsa")
    summary = json.loads((written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    assert (written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__response_window.png").exists()
    assert (written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__full_trajectory.png").exists()
    assert summary["aggregated_response_figure_names"] == [
        "aggregated_response_rsa__per_date__response_window",
        "aggregated_response_rdm_comparison__per_date__response_window",
        "aggregated_response_rdm_comparison__per_date__full_trajectory",
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rdm__pooled__full_trajectory",
    ]


def test_write_rsa_outputs_removes_stale_prototype_artifacts_when_rerun_without_supplement(
    tmp_path, synthetic_stage3_outputs
):
    output_root = tmp_path / "rsa"

    first_written = write_rsa_outputs(_stage3_outputs_with_prototype_supplement(), output_root)
    stale_paths = [
        first_written["tables_dir"] / "aggregated_response_rsa_results__per_date.parquet",
        first_written["tables_dir"] / "aggregated_response_rdm__pooled__response_window.parquet",
        first_written["tables_dir"] / "aggregated_response_rdm__pooled__full_trajectory.parquet",
        first_written["qc_dir"] / "aggregated_response_support__per_date.parquet",
        first_written["qc_dir"] / "aggregated_response_support__pooled.parquet",
        first_written["figures_dir"] / "aggregated_response_rsa__per_date__response_window.png",
        first_written["figures_dir"] / "aggregated_response_rsa__per_date__full_trajectory.png",
        first_written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__response_window.png",
        first_written["figures_dir"] / "aggregated_response_rdm_comparison__per_date__full_trajectory.png",
        first_written["figures_dir"] / "aggregated_response_rdm__pooled__response_window.png",
        first_written["figures_dir"] / "aggregated_response_rdm__pooled__full_trajectory.png",
    ]

    for stale_path in stale_paths:
        assert stale_path.exists()

    second_written = write_rsa_outputs(synthetic_stage3_outputs, output_root)
    summary = json.loads((second_written["output_root"] / "run_summary.json").read_text(encoding="utf-8"))

    for stale_path in stale_paths:
        assert not stale_path.exists()
    assert summary["aggregated_response_context_enabled"] is False
    assert summary["aggregated_response_views"] == []
    assert summary["aggregated_response_dates"] == []
    assert summary["aggregated_response_table_names"] == []
    assert summary["aggregated_response_figure_names"] == []
    assert summary["aggregated_response_descriptive_outputs"] == []


def test_run_biochemical_rsa_marks_tiny_primary_models_excluded_from_primary_ranking():
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

    results = run_biochemical_rsa(resolved_inputs, neural_matrices=_stage3_neural_rdms(), permutations=10, seed=0)

    excluded = results["model_registry_resolved"].loc[lambda df: df["model_id"] == "tiny_primary"]
    assert bool(excluded["excluded_from_primary_ranking"].iloc[0])


def test_run_biochemical_rsa_keeps_global_profile_when_curated_subset_membership_is_empty():
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

    results = run_biochemical_rsa(resolved_inputs, neural_matrices=_stage3_neural_rdms(), permutations=10, seed=0)

    assert "global_profile" in set(results["rsa_results"]["model_id"])
    assert "bile_acid" not in set(results["rsa_results"]["model_id"])


def test_run_biochemical_rsa_adds_prototype_tables_when_prototype_inputs_are_present():
    results = run_biochemical_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        aggregated_response_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    assert "aggregated_response_rsa_results__per_date" in results
    assert "aggregated_response_support__per_date" in results
    assert "aggregated_response_support__pooled" in results
    assert "aggregated_response_rdm__pooled__response_window" in results
    assert "aggregated_response_rdm__pooled__full_trajectory" in results
    assert {
        "internal__aggregated_response_rdm__per_date__response_window__2026-03-11",
        "internal__aggregated_response_rdm__per_date__response_window__2026-03-13",
        "internal__aggregated_response_rdm__per_date__full_trajectory__2026-03-11",
        "internal__aggregated_response_rdm__per_date__full_trajectory__2026-03-13",
    }.issubset(results)
    assert "internal__response_aggregation" in results
    assert results["internal__response_aggregation"]["response_aggregation"].iloc[0] == "mean"
    assert results["internal__aggregated_response_rdm__per_date__response_window__2026-03-11"].columns.tolist() == [
        "stimulus_row",
        "A001",
        "A002",
        "A003",
    ]
    assert "n_dates_contributed" in results["aggregated_response_support__pooled"].columns
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
    }.issubset(results["aggregated_response_rsa_results__per_date"].columns)


def test_run_biochemical_rsa_returns_empty_prototype_outputs_with_expected_schema_for_empty_views():
    results = run_biochemical_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        aggregated_response_inputs=_empty_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    assert results["aggregated_response_rsa_results__per_date"].empty
    assert results["aggregated_response_rsa_results__per_date"].columns.tolist() == [
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
    assert results["aggregated_response_support__per_date"].empty
    assert results["aggregated_response_support__per_date"].columns.tolist() == [
        "date",
        "view_name",
        "stimulus",
        "stim_name",
        "n_trials",
        "n_total_features",
        "n_supported_features",
        "n_all_nan_features",
    ]
    assert results["aggregated_response_support__pooled"].empty
    assert results["aggregated_response_support__pooled"].columns.tolist() == [
        "view_name",
        "stimulus",
        "stim_name",
        "n_trials",
        "n_dates_contributed",
        "n_total_features",
        "n_supported_features",
        "n_all_nan_features",
    ]
    assert results["aggregated_response_rdm__pooled__response_window"].columns.tolist() == ["stimulus_row"]
    assert results["aggregated_response_rdm__pooled__full_trajectory"].columns.tolist() == ["stimulus_row"]


def test_run_biochemical_rsa_uses_aggregated_response_rdms_as_primary_neural_matrices_when_inputs_are_present():
    baseline = run_biochemical_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        permutations=10,
        seed=0,
    )
    supplemented = run_biochemical_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        aggregated_response_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    pd.testing.assert_frame_equal(
        supplemented["neural_rdm__response_window"],
        supplemented["aggregated_response_rdm__pooled__response_window"],
    )
    pd.testing.assert_frame_equal(
        supplemented["neural_rdm__full_trajectory"],
        supplemented["aggregated_response_rdm__pooled__full_trajectory"],
    )
    assert not baseline["rsa_results"].equals(supplemented["rsa_results"])
    assert not baseline["rsa_view_comparison"].equals(supplemented["rsa_view_comparison"])
    assert not baseline["rsa_leave_one_stimulus_out"].equals(supplemented["rsa_leave_one_stimulus_out"])


def test_run_biochemical_rsa_corrects_prototype_fdr_across_full_table_and_ignores_excluded_rows_for_top_model():
    results = run_biochemical_rsa(
        _stage3_prototype_resolved_inputs(),
        neural_matrices=_stage3_neural_rdms(),
        aggregated_response_inputs=_stage3_prototype_inputs(),
        permutations=10,
        seed=0,
    )

    prototype = results["aggregated_response_rsa_results__per_date"]
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


def test_run_biochemical_rsa_restricts_model_rdms_by_date_stimulus_set_and_marks_sparse_rows_invalid():
    resolved_inputs = _stage3_prototype_resolved_inputs()
    prototype_inputs = _stage3_prototype_inputs()
    results = run_biochemical_rsa(
        resolved_inputs,
        neural_matrices=_stage3_neural_rdms(),
        aggregated_response_inputs=prototype_inputs,
        permutations=10,
        seed=0,
    )

    prototype = results["aggregated_response_rsa_results__per_date"]
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



