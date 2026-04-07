import json
import importlib.util
from pathlib import Path
import subprocess


def _load_run_geometry_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_geometry.py"
    spec = importlib.util.spec_from_file_location("run_geometry_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN_GEOMETRY = _load_run_geometry_module()


def test_cli_runs_and_writes_geometry_outputs(tmp_path, synthetic_preprocess_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_geometry.py",
            "--input-root",
            str(synthetic_preprocess_root),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    geometry_root = output_root / "geometry"
    expected_paths = [
        geometry_root / "tables" / "rdm_pairs__response_window__pooled.parquet",
        geometry_root / "tables" / "rdm_pairs__response_window__individual.parquet",
        geometry_root / "tables" / "rdm_pairs__response_window__date.parquet",
        geometry_root / "tables" / "rdm_pairs__full_trajectory__pooled.parquet",
        geometry_root / "tables" / "rdm_pairs__full_trajectory__individual.parquet",
        geometry_root / "tables" / "rdm_pairs__full_trajectory__date.parquet",
        geometry_root / "tables" / "rdm_matrix__response_window__pooled.parquet",
        geometry_root / "tables" / "rdm_matrix__full_trajectory__pooled.parquet",
        geometry_root / "tables" / "rdm_stability_by_individual.parquet",
        geometry_root / "tables" / "rdm_stability_by_date.parquet",
        geometry_root / "tables" / "rdm_view_comparison.parquet",
        geometry_root / "qc" / "rdm_group_coverage.parquet",
        geometry_root / "qc" / "stimulus_overlap__date.parquet",
        geometry_root / "qc" / "stimulus_overlap__individual.parquet",
        geometry_root / "figures" / "rdm_matrix__response_window__pooled.png",
        geometry_root / "figures" / "rdm_matrix__response_window__pooled__clustered.png",
        geometry_root / "figures" / "rdm_matrix__full_trajectory__pooled.png",
        geometry_root / "figures" / "rdm_matrix__full_trajectory__pooled__clustered.png",
        geometry_root / "figures" / "rdm_stability_by_individual.png",
        geometry_root / "figures" / "rdm_stability_by_date.png",
        geometry_root / "figures" / "rdm_view_comparison.png",
        geometry_root / "figures" / "stimulus_overlap__date.png",
        geometry_root / "figures" / "stimulus_overlap__individual.png",
        geometry_root / "run_summary.json",
        geometry_root / "run_summary.md",
    ]

    for path in expected_paths:
        assert path.exists(), path

    run_summary = json.loads((geometry_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["views"] == ["response_window", "full_trajectory"]


def test_cli_preserves_requested_view_order_in_run_summary(tmp_path, synthetic_preprocess_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_geometry.py",
            "--input-root",
            str(synthetic_preprocess_root),
            "--output-root",
            str(output_root),
            "--views",
            "full_trajectory,response_window",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    geometry_root = output_root / "geometry"
    run_summary = json.loads((geometry_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["views"] == ["full_trajectory", "response_window"]
    assert run_summary["pooled_matrix_views"] == ["full_trajectory", "response_window"]


def test_cli_rejects_non_mvp_views(tmp_path, synthetic_preprocess_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_geometry.py",
            "--input-root",
            str(synthetic_preprocess_root),
            "--output-root",
            str(output_root),
            "--views",
            "on_window",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported geometry view" in result.stderr


def test_resolve_input_root_uses_shared_repo_data_processed_for_worktrees(tmp_path):
    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "stage2-geometry"
    shared_input_root = repo_root / "data" / "processed"
    shared_input_root.mkdir(parents=True)

    resolved = RUN_GEOMETRY.resolve_input_root(None, root_dir=worktree_root)

    assert resolved == shared_input_root


def test_resolve_input_root_prefers_explicit_value(tmp_path):
    explicit_root = tmp_path / "explicit-input"

    resolved = RUN_GEOMETRY.resolve_input_root(str(explicit_root), root_dir=Path("ignored"))

    assert resolved == explicit_root

