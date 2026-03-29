import subprocess
import json

import pytest


@pytest.fixture
def tiny_stage0_root(stage1_stage0_root):
    return stage1_stage0_root


def test_cli_runs_and_writes_stage1_outputs(tmp_path, tiny_stage0_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_reliability.py",
            "--input-root",
            str(tiny_stage0_root),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    stage1_root = output_root / "stage1_reliability"
    expected_paths = [
        stage1_root / "tables" / "reliability_summary.parquet",
        stage1_root / "tables" / "primary_view_summary.parquet",
        stage1_root / "tables" / "same_vs_different_summary.parquet",
        stage1_root / "tables" / "within_date_cross_individual_comparisons.parquet",
        stage1_root / "tables" / "within_date_cross_individual_same_vs_different_summary.parquet",
        stage1_root / "tables" / "leave_one_individual_out.parquet",
        stage1_root / "tables" / "leave_one_date_out.parquet",
        stage1_root / "tables" / "per_date_loio_trials.parquet",
        stage1_root / "tables" / "per_date_loio_groups.parquet",
        stage1_root / "tables" / "per_date_loio_summary.parquet",
        stage1_root / "tables" / "split_half_reliability.parquet",
        stage1_root / "tables" / "stimulus_distance_pairs.parquet",
        stage1_root / "tables" / "stimulus_distance_matrix__full_trajectory.parquet",
        stage1_root / "tables" / "stimulus_distance_matrix__on_window.parquet",
        stage1_root / "tables" / "stimulus_distance_matrix__response_window.parquet",
        stage1_root / "tables" / "stimulus_distance_matrix__post_window.parquet",
        stage1_root / "tables" / "permutation_null.parquet",
        stage1_root / "tables" / "grouped_bootstrap.parquet",
        stage1_root / "qc" / "overlap_neuron_counts.parquet",
        stage1_root / "run_summary.json",
        stage1_root / "run_summary.md",
        stage1_root / "figures" / "within_date_cross_individual_same_vs_different.png",
        stage1_root / "figures" / "per_date_loio_overview.png",
        stage1_root / "figures" / "stimulus_distance_matrix__response_window.png",
    ]

    for path in expected_paths:
        assert path.exists(), path

    run_summary = json.loads((stage1_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["primary_view"] == "response_window"

    figure_files = list((stage1_root / "figures").glob("*"))
    assert figure_files, "expected at least one Stage 1 figure"


def test_cli_allows_primary_view_override(tmp_path, tiny_stage0_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_reliability.py",
            "--input-root",
            str(tiny_stage0_root),
            "--output-root",
            str(output_root),
            "--primary-view",
            "on_window",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    stage1_root = output_root / "stage1_reliability"
    run_summary = json.loads((stage1_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["primary_view"] == "on_window"
