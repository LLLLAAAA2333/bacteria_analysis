import subprocess

import pytest


@pytest.fixture
def tiny_parquet_path(tmp_path, synthetic_neuron_segments_df):
    path = tmp_path / "tiny_input.parquet"
    synthetic_neuron_segments_df.to_parquet(path, index=False)
    return path


def test_cli_runs_and_writes_outputs(tmp_path, tiny_parquet_path):
    result = subprocess.run(
        [
            "pixi",
            "run",
            "python",
            "scripts/run_preprocessing.py",
            "--input",
            str(tiny_parquet_path),
            "--output-root",
            str(tmp_path / "processed"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output_root = tmp_path / "processed"
    expected_paths = [
        output_root / "clean" / "neuron_segments_clean.parquet",
        output_root / "trial_level" / "trial_metadata.parquet",
        output_root / "trial_level" / "trial_wide_baseline_centered.parquet",
        output_root / "trial_level" / "trial_tensor_baseline_centered.npz",
        output_root / "qc" / "preprocessing_report.json",
        output_root / "qc" / "preprocessing_report.md",
    ]

    for path in expected_paths:
        assert path.exists(), path
