import json
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, NEURON_ORDER
from bacteria_analysis.io import write_parquet
from bacteria_analysis.io import write_tensor_npz


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
    geometry_tables_dir = root / "geometry" / "tables"
    model_input_root = root / "model_space"
    preprocess_trial_level_dir = root / "preprocess" / "trial_level"

    for directory in (geometry_tables_dir, model_input_root, preprocess_trial_level_dir):
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
    write_parquet(response_window_matrix, geometry_tables_dir / "rdm_matrix__response_window__pooled.parquet")
    write_parquet(full_trajectory_matrix, geometry_tables_dir / "rdm_matrix__full_trajectory__pooled.parquet")

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

    trial_rows = []
    trial_ids = []
    stimulus_labels = []
    stim_name_labels = []
    tensor = np.full((6, len(NEURON_ORDER), len(EXPECTED_TIMEPOINTS)), np.nan, dtype=float)
    stimulus_waveforms = {
        "A001": np.linspace(0.0, 1.0, len(EXPECTED_TIMEPOINTS), dtype=float),
        "A002": np.concatenate(
            [
                np.linspace(0.4, -0.3, 20, dtype=float),
                np.linspace(-0.3, 0.6, len(EXPECTED_TIMEPOINTS) - 20, dtype=float),
            ]
        ),
        "A003": np.sin(np.linspace(0.0, np.pi, len(EXPECTED_TIMEPOINTS), dtype=float)),
    }
    trial_specs = [
        ("2026-03-11", "worm_001", 0, "A001", "Stimulus A001", 1.00),
        ("2026-03-11", "worm_001", 1, "A002", "Stimulus A002", 0.95),
        ("2026-03-11", "worm_001", 2, "A003", "Stimulus A003", 1.05),
        ("2026-03-13", "worm_002", 0, "A001", "Stimulus A001", 1.10),
        ("2026-03-13", "worm_002", 1, "A002", "Stimulus A002", 1.00),
        ("2026-03-13", "worm_002", 2, "A003", "Stimulus A003", 0.90),
    ]
    for trial_index, (date, worm_key, segment_index, stimulus, stim_name, scale) in enumerate(trial_specs):
        trial_id = f"{date.replace('-', '')}__{worm_key}__{segment_index}"
        waveform = stimulus_waveforms[stimulus] * scale
        tensor[trial_index, 0, :] = waveform
        tensor[trial_index, 1, :] = (waveform * 0.7) + 0.1
        tensor[trial_index, 4, :] = np.roll(waveform, 1)
        trial_rows.append(
            {
                "trial_id": trial_id,
                "stimulus": stimulus,
                "stim_name": stim_name,
                "worm_key": worm_key,
                "segment_index": segment_index,
                "date": date,
                "stim_color": "#000000",
            }
        )
        trial_ids.append(trial_id)
        stimulus_labels.append(stimulus)
        stim_name_labels.append(stim_name)

    trial_metadata = pd.DataFrame.from_records(trial_rows)
    write_parquet(trial_metadata, preprocess_trial_level_dir / "trial_metadata.parquet")
    write_parquet(
        trial_metadata.loc[:, ["trial_id", "stimulus", "stim_name", "date", "worm_key", "segment_index"]],
        preprocess_trial_level_dir / "trial_wide_baseline_centered.parquet",
    )
    write_tensor_npz(
        preprocess_trial_level_dir / "trial_tensor_baseline_centered.npz",
        tensor,
        trial_ids,
        stimulus_labels,
        stim_name_labels,
    )

    return root


def test_cli_runs_and_writes_rsa_outputs(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--geometry-root",
            str(stage3_fixture_root / "geometry"),
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

    stage3_root = output_root / "rsa"
    expected_paths = [
        stage3_root / "tables" / "rsa_results.parquet",
        stage3_root / "tables" / "model_registry_resolved.parquet",
        stage3_root / "tables" / "rsa_leave_one_stimulus_out.parquet",
        stage3_root / "qc" / "model_input_coverage.parquet",
        stage3_root / "qc" / "model_feature_filtering.parquet",
        stage3_root / "figures" / "ranked_model_rsa.png",
        stage3_root / "figures" / "neural_vs_top_model_rdm__response_window.png",
        stage3_root / "figures" / "neural_vs_top_model_rdm__full_trajectory.png",
        stage3_root / "run_summary.json",
        stage3_root / "run_summary.md",
    ]

    for path in expected_paths:
        assert path.exists(), path

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["focus_view"] == "response_window"
    assert summary["ranked_models"] == ["global_profile", "bile_acid"]
    assert summary["prototype_context_enabled"] is False
    assert summary["prototype_views"] == []
    assert summary["prototype_dates"] == []
    assert summary["figure_names"] == [
        "ranked_model_rsa",
        "leave_one_stimulus_out_robustness",
        "view_comparison_summary",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]
    assert not (stage3_root / "figures" / "neural_vs_top_model_rdm_panel.png").exists()
    assert not (stage3_root / "tables" / "prototype_rsa_results__per_date.parquet").exists()
    assert "Included ranked models: global_profile, bile_acid" in result.stdout


def test_cli_runs_and_writes_rsa_prototype_context_outputs(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--geometry-root",
            str(stage3_fixture_root / "geometry"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--model-input-root",
            str(stage3_fixture_root / "model_space"),
            "--preprocess-root",
            str(stage3_fixture_root / "preprocess"),
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

    stage3_root = output_root / "rsa"
    expected_paths = [
        stage3_root / "tables" / "prototype_rsa_results__per_date.parquet",
        stage3_root / "tables" / "prototype_rdm__pooled__response_window.parquet",
        stage3_root / "tables" / "prototype_rdm__pooled__full_trajectory.parquet",
        stage3_root / "qc" / "prototype_support__per_date.parquet",
        stage3_root / "qc" / "prototype_support__pooled.parquet",
        stage3_root / "figures" / "prototype_rsa__per_date__response_window.png",
        stage3_root / "figures" / "prototype_rsa__per_date__full_trajectory.png",
        stage3_root / "figures" / "prototype_rdm_comparison__per_date__response_window.png",
        stage3_root / "figures" / "prototype_rdm_comparison__per_date__full_trajectory.png",
        stage3_root / "figures" / "prototype_rdm__pooled__response_window.png",
        stage3_root / "figures" / "prototype_rdm__pooled__full_trajectory.png",
    ]

    for path in expected_paths:
        assert path.exists(), path

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["prototype_context_enabled"] is True
    assert summary["prototype_aggregation"] == "mean"
    assert summary["prototype_views"] == ["response_window", "full_trajectory"]
    assert summary["prototype_dates"] == ["2026-03-11", "2026-03-13"]
    assert summary["prototype_table_names"] == [
        "prototype_rsa_results__per_date",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]
    assert summary["figure_names"] == [
        "ranked_model_rsa",
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
    assert summary["prototype_figure_names"] == [
        "prototype_rsa__per_date__response_window",
        "prototype_rsa__per_date__full_trajectory",
        "prototype_rdm_comparison__per_date__response_window",
        "prototype_rdm_comparison__per_date__full_trajectory",
        "prototype_rdm__pooled__response_window",
        "prototype_rdm__pooled__full_trajectory",
    ]


def test_cli_runs_and_writes_rsa_prototype_context_outputs_with_median_aggregation(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--geometry-root",
            str(stage3_fixture_root / "geometry"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--model-input-root",
            str(stage3_fixture_root / "model_space"),
            "--preprocess-root",
            str(stage3_fixture_root / "preprocess"),
            "--prototype-aggregation",
            "median",
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

    stage3_root = output_root / "rsa"
    assert (stage3_root / "figures" / "prototype_rdm_comparison__per_date__response_window.png").exists()

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["prototype_context_enabled"] is True
    assert summary["prototype_aggregation"] == "median"


def test_resolve_default_paths_use_shared_repo_locations_for_worktrees(tmp_path):
    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "stage3-rsa"
    shared_geometry_root = repo_root / "results" / "geometry"
    shared_model_input_root = repo_root / "data" / "model_space"
    shared_matrix_path = repo_root / "data" / "matrix.xlsx"
    shared_preprocess_root = repo_root / "data" / "preprocess"

    shared_geometry_root.mkdir(parents=True)
    shared_model_input_root.mkdir(parents=True)
    shared_preprocess_root.mkdir(parents=True)
    shared_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    shared_matrix_path.write_bytes(b"matrix")

    assert RUN_RSA.resolve_geometry_root(str(RUN_RSA.DEFAULT_GEOMETRY_ROOT), root_dir=worktree_root) == shared_geometry_root
    assert RUN_RSA.resolve_stage2_root(str(RUN_RSA.DEFAULT_STAGE2_ROOT), root_dir=worktree_root) == shared_geometry_root
    assert RUN_RSA.resolve_model_input_root(str(RUN_RSA.DEFAULT_MODEL_INPUT_ROOT), root_dir=worktree_root) == shared_model_input_root
    assert RUN_RSA.resolve_matrix_path(str(RUN_RSA.DEFAULT_MATRIX_PATH), root_dir=worktree_root) == shared_matrix_path
    assert RUN_RSA.resolve_preprocess_root("data/preprocess", root_dir=worktree_root) == shared_preprocess_root

