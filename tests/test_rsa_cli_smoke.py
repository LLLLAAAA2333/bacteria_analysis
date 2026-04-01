import json
import importlib.util
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from bacteria_analysis.io import write_parquet


def _load_run_rsa_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_rsa.py"
    spec = importlib.util.spec_from_file_location("run_rsa_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_RSA = _load_run_rsa_module()


@pytest.fixture
def stage3_fixture_root(tmp_path):
    root = tmp_path / "stage3_fixture"
    stage2_tables_dir = root / "stage2_geometry" / "tables"
    model_input_root = root / "model_space"

    for directory in (stage2_tables_dir, model_input_root):
        directory.mkdir(parents=True, exist_ok=True)

    response_window_matrix = pd.DataFrame.from_records(
        [
            {"stimulus_row": "A001", "A001": 0.0, "A002": 0.2, "A003": 0.4},
            {"stimulus_row": "A002", "A001": 0.2, "A002": 0.0, "A003": 0.3},
            {"stimulus_row": "A003", "A001": 0.4, "A002": 0.3, "A003": 0.0},
        ]
    )
    full_trajectory_matrix = pd.DataFrame.from_records(
        [
            {"stimulus_row": "A001", "A001": 0.0, "A002": 0.1, "A003": 0.5},
            {"stimulus_row": "A002", "A001": 0.1, "A002": 0.0, "A003": 0.2},
            {"stimulus_row": "A003", "A001": 0.5, "A002": 0.2, "A003": 0.0},
        ]
    )
    write_parquet(response_window_matrix, stage2_tables_dir / "rdm_matrix__response_window__pooled.parquet")
    write_parquet(full_trajectory_matrix, stage2_tables_dir / "rdm_matrix__full_trajectory__pooled.parquet")

    pd.DataFrame.from_records(
        [
            {"stimulus": "A001", "stim_name": "Stimulus A001", "sample_id": "A001"},
            {"stimulus": "A002", "stim_name": "Stimulus A002", "sample_id": "A002"},
            {"stimulus": "A003", "stim_name": "Stimulus A003", "sample_id": "A003"},
        ]
    ).to_csv(model_input_root / "stimulus_sample_map.csv", index=False)

    pd.DataFrame.from_records(
        [
            {
                "metabolite_name": "feature_1",
                "superclass": "lipid",
                "subclass": "bile_acid",
                "pathway_tag": "bile_acid",
                "annotation_source": "user",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_2",
                "superclass": "lipid",
                "subclass": "bile_acid",
                "pathway_tag": "bile_acid",
                "annotation_source": "user",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_3",
                "superclass": "energy",
                "subclass": "organic_acid",
                "pathway_tag": "tca",
                "annotation_source": "user",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_4",
                "superclass": "lipid",
                "subclass": "bile_acid",
                "pathway_tag": "bile_acid",
                "annotation_source": "user",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "metabolite_name": "feature_5",
                "superclass": "lipid",
                "subclass": "bile_acid",
                "pathway_tag": "bile_acid",
                "annotation_source": "user",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
        ]
    ).to_csv(model_input_root / "metabolite_annotation.csv", index=False)

    pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "model_label": "Global Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "All metabolites",
                "authority": "user",
                "notes": "",
            },
            {
                "model_id": "bile_acid",
                "model_label": "Bile Acid",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "Bile acid subset",
                "authority": "user",
                "notes": "",
            },
        ]
    ).to_csv(model_input_root / "model_registry.csv", index=False)

    pd.DataFrame.from_records(
        [
            {
                "model_id": "bile_acid",
                "metabolite_name": "feature_1",
                "membership_source": "manual",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "model_id": "bile_acid",
                "metabolite_name": "feature_2",
                "membership_source": "manual",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "model_id": "bile_acid",
                "metabolite_name": "feature_3",
                "membership_source": "manual",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "model_id": "bile_acid",
                "metabolite_name": "feature_4",
                "membership_source": "manual",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
            {
                "model_id": "bile_acid",
                "metabolite_name": "feature_5",
                "membership_source": "manual",
                "review_status": "reviewed",
                "ambiguous_flag": False,
                "notes": "",
            },
        ]
    ).to_csv(model_input_root / "model_membership.csv", index=False)

    pd.DataFrame.from_records(
        [
            {
                "sample_id": "A001",
                "feature_1": 1.0,
                "feature_2": 2.0,
                "feature_3": 0.5,
                "feature_4": 3.0,
                "feature_5": 4.5,
            },
            {
                "sample_id": "A002",
                "feature_1": 2.5,
                "feature_2": 1.0,
                "feature_3": 1.5,
                "feature_4": 2.0,
                "feature_5": 3.0,
            },
            {
                "sample_id": "A003",
                "feature_1": 3.0,
                "feature_2": 4.0,
                "feature_3": 2.5,
                "feature_4": 1.0,
                "feature_5": 2.0,
            },
        ]
    ).to_excel(root / "matrix.xlsx", index=False)

    return root


def test_cli_runs_and_writes_stage3_outputs(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--stage2-root",
            str(stage3_fixture_root / "stage2_geometry"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--model-input-root",
            str(stage3_fixture_root / "model_space"),
            "--output-root",
            str(output_root),
            "--permutations",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    stage3_root = output_root / "stage3_rsa"
    expected_paths = [
        stage3_root / "tables" / "rsa_results.parquet",
        stage3_root / "tables" / "model_registry_resolved.parquet",
        stage3_root / "tables" / "rsa_leave_one_stimulus_out.parquet",
        stage3_root / "qc" / "model_input_coverage.parquet",
        stage3_root / "qc" / "model_feature_filtering.parquet",
        stage3_root / "figures" / "ranked_primary_model_rsa.png",
        stage3_root / "figures" / "neural_vs_top_model_rdm__response_window.png",
        stage3_root / "figures" / "neural_vs_top_model_rdm__full_trajectory.png",
        stage3_root / "run_summary.json",
        stage3_root / "run_summary.md",
    ]

    for path in expected_paths:
        assert path.exists(), path

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["primary_view"] == "response_window"
    assert summary["primary_models"] == ["global_profile", "bile_acid"]
    assert summary["figure_names"] == [
        "ranked_primary_model_rsa",
        "leave_one_stimulus_out_robustness",
        "view_comparison_summary",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]
    assert not (stage3_root / "figures" / "neural_vs_top_model_rdm_panel.png").exists()
    assert "Included primary models: global_profile, bile_acid" in result.stdout


def test_resolve_default_paths_use_shared_repo_locations_for_worktrees(tmp_path):
    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "stage3-rsa"
    shared_stage2_root = repo_root / "results" / "stage2_geometry"
    shared_model_input_root = repo_root / "data" / "model_space"
    shared_matrix_path = repo_root / "data" / "matrix.xlsx"

    shared_stage2_root.mkdir(parents=True)
    shared_model_input_root.mkdir(parents=True)
    shared_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    shared_matrix_path.write_bytes(b"matrix")

    assert RUN_RSA.resolve_stage2_root(str(RUN_RSA.DEFAULT_STAGE2_ROOT), root_dir=worktree_root) == shared_stage2_root
    assert RUN_RSA.resolve_model_input_root(str(RUN_RSA.DEFAULT_MODEL_INPUT_ROOT), root_dir=worktree_root) == shared_model_input_root
    assert RUN_RSA.resolve_matrix_path(str(RUN_RSA.DEFAULT_MATRIX_PATH), root_dir=worktree_root) == shared_matrix_path
