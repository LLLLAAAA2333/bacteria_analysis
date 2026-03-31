"""Loaders for Stage 3 model-space inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_stimulus_sample_map(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), dtype=str).fillna("")
    return _validate_stimulus_sample_map(frame)


def read_metabolite_matrix(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(Path(path), engine="openpyxl")
    return _normalize_matrix_frame(frame)


def _validate_stimulus_sample_map(frame: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in frame.columns:
        raise ValueError("stimulus_sample_map must include sample_id")

    normalized = frame.copy()
    normalized["sample_id"] = normalized["sample_id"].astype(str).str.strip()

    if (normalized["sample_id"] == "").any():
        raise ValueError("sample_id values must be non-empty")
    if normalized["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")

    return normalized


def _normalize_matrix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return normalized

    if "sample_id" in normalized.columns:
        sample_column = "sample_id"
    else:
        sample_column = normalized.columns[0]

    normalized[sample_column] = normalized[sample_column].astype(str).str.strip()
    if normalized[sample_column].duplicated().any():
        raise ValueError("sample_id values must be unique")

    normalized = normalized.set_index(sample_column)
    normalized.index.name = "sample_id"
    return normalized
