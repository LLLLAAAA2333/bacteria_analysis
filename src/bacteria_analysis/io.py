"""Filesystem helpers for preprocessing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file from disk."""

    return pd.read_parquet(Path(path))


def ensure_output_dirs(output_root: str | Path) -> dict[str, Path]:
    """Create and return the standard preprocessing output tree."""

    root = Path(output_root)
    clean_dir = root / "clean"
    trial_level_dir = root / "trial_level"
    qc_dir = root / "qc"

    for directory in (root, clean_dir, trial_level_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": root,
        "clean_dir": clean_dir,
        "trial_level_dir": trial_level_dir,
        "qc_dir": qc_dir,
    }


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe to parquet and return the resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def write_tensor_npz(
    path: str | Path,
    tensor: np.ndarray,
    trial_ids: list[str] | np.ndarray,
    stimulus_labels: list[str] | np.ndarray,
    stim_name_labels: list[str] | np.ndarray,
) -> Path:
    """Write tensor outputs and aligned labels to a compressed NPZ."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tensor=tensor,
        trial_ids=np.asarray(trial_ids, dtype=str),
        stimulus_labels=np.asarray(stimulus_labels, dtype=str),
        stim_name_labels=np.asarray(stim_name_labels, dtype=str),
    )
    return output_path


def write_json(obj: Any, path: str | Path) -> Path:
    """Write JSON with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_markdown_report(report: dict[str, Any], path: str | Path) -> Path:
    """Render the preprocessing QC schema as a readable markdown report."""

    required_keys = (
        "input_rows",
        "output_rows",
        "n_unique_trials",
        "n_unique_stimuli",
        "n_unique_neurons",
        "n_fully_nan_traces_removed",
        "n_partially_nan_traces_retained",
        "neuron_coverage_distribution",
        "trials_per_stimulus_summary",
    )
    missing_keys = [key for key in required_keys if key not in report]
    if missing_keys:
        raise ValueError(f"report is missing required keys: {', '.join(missing_keys)}")

    lines = [
        "# Preprocessing QC Report",
        "",
        "## Summary",
        f"- Input rows: {report['input_rows']}",
        f"- Output rows: {report['output_rows']}",
        f"- Unique trials: {report['n_unique_trials']}",
        f"- Unique stimuli: {report['n_unique_stimuli']}",
        f"- Unique neurons: {report['n_unique_neurons']}",
        "",
        "## Trace Filtering",
        f"- Fully-NaN traces removed: {report['n_fully_nan_traces_removed']}",
        f"- Partially-NaN traces retained: {report['n_partially_nan_traces_retained']}",
        "",
        "## Neuron Coverage Distribution",
    ]

    coverage_rows = report["neuron_coverage_distribution"]
    if coverage_rows:
        lines.extend(["", "| Observed neurons | Trials |", "| --- | ---: |"])
        for row in coverage_rows:
            lines.append(f"| {row['n_observed_neurons']} | {row['n_trials']} |")
    else:
        lines.append("")
        lines.append("- None")

    lines.extend(["", "## Trials Per Stimulus"])
    stimulus_rows = report["trials_per_stimulus_summary"]
    if stimulus_rows:
        lines.extend(["", "| Stimulus | Stim name | Trials |", "| --- | --- | ---: |"])
        for row in stimulus_rows:
            lines.append(f"| {row['stimulus']} | {row['stim_name']} | {row['n_trials']} |")
    else:
        lines.append("")
        lines.append("- None")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
