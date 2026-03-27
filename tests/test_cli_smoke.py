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
    assert (tmp_path / "processed" / "clean" / "neuron_segments_clean.parquet").exists()
