import subprocess
import json

import pytest


@pytest.fixture
def tiny_preprocess_root(synthetic_preprocess_root):
    return synthetic_preprocess_root


def test_cli_runs_and_writes_reliability_outputs(tmp_path, tiny_preprocess_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_reliability.py",
            "--input-root",
            str(tiny_preprocess_root),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    reliability_root = output_root / "reliability"
    expected_paths = [
        reliability_root / "tables" / "reliability_summary.parquet",
        reliability_root / "tables" / "focus_view_summary.parquet",
        reliability_root / "tables" / "same_vs_different_summary.parquet",
        reliability_root / "tables" / "within_date_cross_individual_comparisons.parquet",
        reliability_root / "tables" / "within_date_cross_individual_same_vs_different_summary.parquet",
        reliability_root / "tables" / "leave_one_individual_out.parquet",
        reliability_root / "tables" / "leave_one_date_out.parquet",
        reliability_root / "tables" / "per_date_loio_trials.parquet",
        reliability_root / "tables" / "per_date_loio_groups.parquet",
        reliability_root / "tables" / "per_date_loio_summary.parquet",
        reliability_root / "tables" / "split_half_reliability.parquet",
        reliability_root / "tables" / "stimulus_distance_pairs.parquet",
        reliability_root / "tables" / "stimulus_distance_matrix__full_trajectory.parquet",
        reliability_root / "tables" / "stimulus_distance_matrix__on_window.parquet",
        reliability_root / "tables" / "stimulus_distance_matrix__response_window.parquet",
        reliability_root / "tables" / "stimulus_distance_matrix__post_window.parquet",
        reliability_root / "tables" / "permutation_null.parquet",
        reliability_root / "tables" / "grouped_bootstrap.parquet",
        reliability_root / "qc" / "overlap_neuron_counts.parquet",
        reliability_root / "run_summary.json",
        reliability_root / "run_summary.md",
        reliability_root / "figures" / "same_vs_different_distributions__boxen_points.png",
        reliability_root / "figures" / "same_vs_different_distributions__ecdf.png",
        reliability_root / "figures" / "same_vs_different_by_date__2026-03-27.png",
        reliability_root / "figures" / "same_vs_different_by_date__2026-03-28.png",
        reliability_root / "figures" / "per_stimulus_same_vs_different__pooled.png",
        reliability_root / "figures" / "per_stimulus_same_vs_different__2026-03-27.png",
        reliability_root / "figures" / "per_stimulus_same_vs_different__2026-03-28.png",
        reliability_root / "figures" / "overlap_neuron_qc_summary.png",
        reliability_root / "figures" / "within_date_cross_individual_same_vs_different.png",
        reliability_root / "figures" / "per_date_loio_overview.png",
    ]

    for path in expected_paths:
        assert path.exists(), path

    run_summary = json.loads((reliability_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["focus_view"] == "response_window"
    assert run_summary["per_date_same_vs_different_figure_names"] == [
        "same_vs_different_by_date__2026-03-27.png",
        "same_vs_different_by_date__2026-03-28.png",
    ]
    assert run_summary["pooled_per_stimulus_same_vs_different_figure_name"] == "per_stimulus_same_vs_different__pooled.png"
    assert run_summary["per_date_per_stimulus_same_vs_different_figure_names"] == [
        "per_stimulus_same_vs_different__2026-03-27.png",
        "per_stimulus_same_vs_different__2026-03-28.png",
    ]
    assert not (reliability_root / "figures" / "same_vs_different_distributions.png").exists()
    assert not (reliability_root / "figures" / "same_vs_different_distributions__raincloud.png").exists()
    assert not (reliability_root / "figures" / "same_vs_different_distributions__violin_clean.png").exists()
    assert not (reliability_root / "figures" / "cross_view_reliability_comparison.png").exists()
    assert not (reliability_root / "figures" / "leave_one_individual_out_summary.png").exists()
    assert not (reliability_root / "figures" / "leave_one_date_out_summary.png").exists()
    assert not (reliability_root / "figures" / "split_half_summary.png").exists()
    assert not list((reliability_root / "figures").glob("stimulus_distance_matrix__*.png"))

    figure_files = list((reliability_root / "figures").glob("*"))
    assert figure_files, "expected at least one reliability figure"


def test_cli_allows_focus_view_override(tmp_path, tiny_preprocess_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_reliability.py",
            "--input-root",
            str(tiny_preprocess_root),
            "--output-root",
            str(output_root),
            "--focus-view",
            "on_window",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    reliability_root = output_root / "reliability"
    run_summary = json.loads((reliability_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["focus_view"] == "on_window"


def test_cli_keeps_primary_view_alias_for_focus_view(tmp_path, tiny_preprocess_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_reliability.py",
            "--input-root",
            str(tiny_preprocess_root),
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

    reliability_root = output_root / "reliability"
    run_summary = json.loads((reliability_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["focus_view"] == "on_window"

