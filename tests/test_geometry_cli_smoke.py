import json
import subprocess


def test_cli_runs_and_writes_stage2_outputs(tmp_path, stage1_stage0_root):
    output_root = tmp_path / "results"
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_geometry.py",
            "--input-root",
            str(stage1_stage0_root),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    stage2_root = output_root / "stage2_geometry"
    expected_paths = [
        stage2_root / "tables" / "rdm_pairs__response_window__pooled.parquet",
        stage2_root / "tables" / "rdm_pairs__response_window__individual.parquet",
        stage2_root / "tables" / "rdm_pairs__response_window__date.parquet",
        stage2_root / "tables" / "rdm_pairs__full_trajectory__pooled.parquet",
        stage2_root / "tables" / "rdm_pairs__full_trajectory__individual.parquet",
        stage2_root / "tables" / "rdm_pairs__full_trajectory__date.parquet",
        stage2_root / "tables" / "rdm_matrix__response_window__pooled.parquet",
        stage2_root / "tables" / "rdm_matrix__full_trajectory__pooled.parquet",
        stage2_root / "tables" / "rdm_stability_by_individual.parquet",
        stage2_root / "tables" / "rdm_stability_by_date.parquet",
        stage2_root / "tables" / "rdm_view_comparison.parquet",
        stage2_root / "qc" / "rdm_group_coverage.parquet",
        stage2_root / "figures" / "rdm_matrix__response_window__pooled.png",
        stage2_root / "figures" / "rdm_matrix__full_trajectory__pooled.png",
        stage2_root / "figures" / "rdm_stability_by_individual.png",
        stage2_root / "figures" / "rdm_stability_by_date.png",
        stage2_root / "figures" / "rdm_view_comparison.png",
        stage2_root / "run_summary.json",
        stage2_root / "run_summary.md",
    ]

    for path in expected_paths:
        assert path.exists(), path

    run_summary = json.loads((stage2_root / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["views"] == ["full_trajectory", "response_window"]
