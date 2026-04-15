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


def _ranked_model_detail_line(detail: dict[str, object]) -> str:
    return (
        f"- {detail['model_id']} | view={detail['view_name']} | rsa={detail['rsa_similarity']} | "
        f"p_raw={detail['p_value_raw']} | p_fdr={detail['p_value_fdr']} | "
        f"n={detail['n_shared_entries']} | status={detail['score_status']} | top={detail['is_top_model']}"
    )


def _view_comparison_detail_line(detail: dict[str, object]) -> str:
    return (
        f"- {detail['view_name']} vs {detail['reference_view_name']} | scope={detail['comparison_scope']} | "
        f"rsa={detail['rsa_similarity']} | p_raw={detail['p_value_raw']} | p_fdr={detail['p_value_fdr']} | "
        f"n={detail['n_shared_entries']} | status={detail['score_status']}"
    )


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
            {"stimulus": "A001", "stim_name": "A001", "sample_id": "A001"},
            {"stimulus": "A002", "stim_name": "A002", "sample_id": "A002"},
            {"stimulus": "A003", "stim_name": "A003", "sample_id": "A003"},
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
        ("2026-03-11", "worm_001", 0, "A001", "A001", 1.00),
        ("2026-03-11", "worm_001", 1, "A002", "A002", 0.95),
        ("2026-03-11", "worm_001", 2, "A003", "A003", 1.05),
        ("2026-03-13", "worm_002", 0, "A001", "A001", 1.10),
        ("2026-03-13", "worm_002", 1, "A002", "A002", 1.00),
        ("2026-03-13", "worm_002", 2, "A003", "A003", 0.90),
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
        stage3_root / "figures" / "single_stimulus_sensitivity.png",
        stage3_root / "figures" / "neural_vs_top_model_rdm__response_window.png",
        stage3_root / "figures" / "neural_vs_top_model_rdm__full_trajectory.png",
        stage3_root / "run_summary.json",
        stage3_root / "run_summary.md",
    ]

    for path in expected_paths:
        assert path.exists(), path

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    markdown = (stage3_root / "run_summary.md").read_text(encoding="utf-8")
    assert summary["focus_view"] == "response_window"
    assert summary["ranked_models"] == ["global_profile", "bile_acid"]
    assert summary["ranked_model_rsa_details"]
    assert summary["view_comparison_details"]
    assert summary["aggregated_response_context_enabled"] is False
    assert summary["aggregated_response_views"] == []
    assert summary["aggregated_response_dates"] == []
    ranked_section = _markdown_section(markdown, "## Ranked Model RSA Details")
    view_comparison_section = _markdown_section(markdown, "## View Comparison Details")
    assert _ranked_model_detail_line(summary["ranked_model_rsa_details"][0]) in ranked_section
    assert _view_comparison_detail_line(summary["view_comparison_details"][0]) in view_comparison_section
    assert summary["figure_names"] == [
        "single_stimulus_sensitivity",
        "neural_vs_top_model_rdm__response_window",
        "neural_vs_top_model_rdm__full_trajectory",
    ]
    assert not (stage3_root / "figures" / "ranked_model_rsa.png").exists()
    assert not (stage3_root / "figures" / "view_comparison_summary.png").exists()
    assert not (stage3_root / "figures" / "leave_one_stimulus_out_robustness.png").exists()
    assert not (stage3_root / "figures" / "neural_vs_top_model_rdm_panel.png").exists()
    assert not (stage3_root / "tables" / "aggregated_response_rsa_results__per_date.parquet").exists()
    assert "## Ranked Model RSA Details" in markdown
    assert "## View Comparison Details" in markdown
    assert "Included ranked models: global_profile, bile_acid" in result.stdout


def test_cli_legacy_curated_fallback_uses_geometry_root_without_preprocess(tmp_path, monkeypatch):
    geometry_root = tmp_path / "geometry"
    model_input_root = tmp_path / "model_space"
    matrix_path = tmp_path / "matrix.xlsx"
    output_root = tmp_path / "results"
    geometry_root.mkdir()
    model_input_root.mkdir()
    matrix_path.write_bytes(b"matrix")

    resolved_inputs = object()
    neural_matrices = object()
    model_registry_resolved = pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "model_tier": "primary",
                "excluded_from_primary_ranking": False,
            }
        ]
    )

    def _resolve_model_inputs(actual_model_input_root, actual_matrix_path):
        assert actual_model_input_root == model_input_root
        assert actual_matrix_path == matrix_path
        return resolved_inputs

    def _load_geometry_pooled_neural_rdms(actual_geometry_root):
        assert actual_geometry_root == geometry_root
        return neural_matrices

    def _unexpected_direct_mode(*args, **kwargs):
        raise AssertionError("direct-mode preprocessing path should not be used for legacy curated fallback")

    def _unexpected_aggregated_context(*args, **kwargs):
        raise AssertionError("preprocess-backed aggregation should not be used for legacy curated fallback")

    def _run_biochemical_rsa(actual_inputs, **kwargs):
        assert actual_inputs is resolved_inputs
        assert kwargs["neural_matrices"] is neural_matrices
        assert "aggregated_response_inputs" not in kwargs
        assert "response_aggregation" not in kwargs
        return {"model_registry_resolved": model_registry_resolved}

    def _write_rsa_outputs(core_outputs, rsa_output_root):
        assert core_outputs["model_registry_resolved"].equals(model_registry_resolved)
        assert rsa_output_root == output_root / "rsa"
        return {
            "tables_dir": rsa_output_root / "tables",
            "figures_dir": rsa_output_root / "figures",
            "run_summary_json": rsa_output_root / "run_summary.json",
        }

    monkeypatch.setattr(RUN_RSA, "resolve_model_inputs", _resolve_model_inputs)
    monkeypatch.setattr(RUN_RSA, "load_geometry_pooled_neural_rdms", _load_geometry_pooled_neural_rdms)
    monkeypatch.setattr(RUN_RSA, "resolve_direct_global_profile_inputs", _unexpected_direct_mode)
    monkeypatch.setattr(RUN_RSA, "load_aggregated_response_context_inputs", _unexpected_aggregated_context)
    monkeypatch.setattr(RUN_RSA, "run_biochemical_rsa", _run_biochemical_rsa)
    monkeypatch.setattr(RUN_RSA, "write_rsa_outputs", _write_rsa_outputs)

    assert (
        RUN_RSA.main(
            [
                "--geometry-root",
                str(geometry_root),
                "--matrix",
                str(matrix_path),
                "--model-input-root",
                str(model_input_root),
                "--output-root",
                str(output_root),
                "--permutations",
                "0",
            ]
        )
        == 0
    )


def test_parse_args_defaults_model_input_root_to_none():
    args = RUN_RSA.parse_args([])

    assert args.model_input_root is None
    assert args.preprocess_root is None


def test_cli_direct_mode_runs_without_model_input_root(tmp_path, stage3_fixture_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_rsa.py",
            "--preprocess-root",
            str(stage3_fixture_root / "preprocess"),
            "--matrix",
            str(stage3_fixture_root / "matrix.xlsx"),
            "--output-root",
            str(output_root),
            "--permutations",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads((output_root / "rsa" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["ranked_models"] == ["global_profile"]
    assert summary["aggregated_response_context_enabled"] is True
    assert "Included ranked models: global_profile" in result.stdout


def test_cli_requires_preprocess_root_when_model_input_root_is_omitted(tmp_path, capsys):
    exit_code = RUN_RSA.cli(
        [
            "--matrix",
            str(tmp_path / "matrix.xlsx"),
            "--output-root",
            str(tmp_path / "results"),
            "--permutations",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Direct RSA requires --preprocess-root when --model-input-root is omitted" in captured.err


def test_direct_mode_does_not_resolve_default_model_input_root(tmp_path, monkeypatch):
    preprocess_root = tmp_path / "preprocess"
    preprocess_root.mkdir()
    matrix_path = tmp_path / "matrix.xlsx"
    matrix_path.write_bytes(b"matrix")

    def _unexpected_model_root_resolution(*args, **kwargs):
        raise AssertionError("resolve_model_input_root should not be called in direct mode")

    def _direct_mode_reached(*args, **kwargs):
        raise RuntimeError("direct mode reached")

    monkeypatch.setattr(RUN_RSA, "resolve_model_input_root", _unexpected_model_root_resolution)
    monkeypatch.setattr(RUN_RSA, "resolve_direct_global_profile_inputs", _direct_mode_reached)

    with pytest.raises(RuntimeError, match="direct mode reached"):
        RUN_RSA.main(
            [
                "--preprocess-root",
                str(preprocess_root),
                "--matrix",
                str(matrix_path),
                "--output-root",
                str(tmp_path / "results"),
                "--permutations",
                "0",
            ]
        )


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
        stage3_root / "tables" / "aggregated_response_rsa_results__per_date.parquet",
        stage3_root / "tables" / "aggregated_response_rdm__pooled__response_window.parquet",
        stage3_root / "tables" / "aggregated_response_rdm__pooled__full_trajectory.parquet",
        stage3_root / "qc" / "aggregated_response_support__per_date.parquet",
        stage3_root / "qc" / "aggregated_response_support__pooled.parquet",
        stage3_root / "figures" / "aggregated_response_rsa__per_date__response_window.png",
        stage3_root / "figures" / "aggregated_response_rsa__per_date__full_trajectory.png",
        stage3_root / "figures" / "aggregated_response_rdm_comparison__per_date__response_window.png",
        stage3_root / "figures" / "aggregated_response_rdm_comparison__per_date__full_trajectory.png",
        stage3_root / "figures" / "aggregated_response_rdm__pooled__response_window.png",
        stage3_root / "figures" / "aggregated_response_rdm__pooled__full_trajectory.png",
    ]

    for path in expected_paths:
        assert path.exists(), path

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    markdown = (stage3_root / "run_summary.md").read_text(encoding="utf-8")
    assert summary["aggregated_response_context_enabled"] is True
    assert summary["response_aggregation"] == "mean"
    assert summary["aggregated_response_views"] == ["response_window", "full_trajectory"]
    assert summary["aggregated_response_dates"] == ["2026-03-11", "2026-03-13"]
    assert summary["aggregated_response_table_names"] == [
        "aggregated_response_rsa_results__per_date",
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
    assert summary["aggregated_response_figure_names"] == [
        "aggregated_response_rsa__per_date__response_window",
        "aggregated_response_rsa__per_date__full_trajectory",
        "aggregated_response_rdm_comparison__per_date__response_window",
        "aggregated_response_rdm_comparison__per_date__full_trajectory",
        "aggregated_response_rdm__pooled__response_window",
        "aggregated_response_rdm__pooled__full_trajectory",
    ]
    assert not (stage3_root / "figures" / "ranked_model_rsa.png").exists()
    assert not (stage3_root / "figures" / "view_comparison_summary.png").exists()
    assert not (stage3_root / "figures" / "leave_one_stimulus_out_robustness.png").exists()
    assert _ranked_model_detail_line(summary["ranked_model_rsa_details"][0]) in _markdown_section(
        markdown, "## Ranked Model RSA Details"
    )
    assert _view_comparison_detail_line(summary["view_comparison_details"][0]) in _markdown_section(
        markdown, "## View Comparison Details"
    )


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
            "--response-aggregation",
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
    assert (stage3_root / "figures" / "aggregated_response_rdm_comparison__per_date__response_window.png").exists()

    summary = json.loads((stage3_root / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregated_response_context_enabled"] is True
    assert summary["response_aggregation"] == "median"


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


