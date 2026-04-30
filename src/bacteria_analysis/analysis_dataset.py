"""Function-first dataset loaders for analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from bacteria_analysis.model_space import build_stimulus_sample_map, read_metabolite_matrix


@dataclass(frozen=True)
class AnalysisDataset:
    neural: pd.DataFrame
    matrix: pd.DataFrame
    metadata: pd.DataFrame
    stimulus_sample_map: pd.DataFrame
    included_dates: tuple[str, ...]
    excluded_dates: tuple[str, ...]
    parameters: dict[str, object]


@dataclass(frozen=True)
class AnchorDataset:
    neural: pd.DataFrame
    anchor_stimuli: tuple[str, ...]
    included_dates: tuple[str, ...]
    excluded_dates: tuple[str, ...]
    parameters: dict[str, object]


def build_analysis_dataset(
    neural_path: str | Path,
    matrix_path: str | Path,
    metadata_path: str | Path,
    exclude_dates: Iterable[object] | None = None,
    keep_dates: Iterable[object] | None = None,
) -> AnalysisDataset:
    """Load raw neural, metabolite matrix, and metadata inputs for analysis."""
    neural = pd.read_parquet(neural_path)
    matrix = read_metabolite_matrix(matrix_path)
    metadata = _read_table(metadata_path)

    neural, metadata, included_dates, excluded_dates_tuple = _apply_date_filters(
        neural,
        metadata,
        exclude_dates=exclude_dates,
        keep_dates=keep_dates,
    )
    stimulus_sample_map = _build_minimal_stimulus_sample_map(neural, matrix)

    return AnalysisDataset(
        neural=neural.reset_index(drop=True),
        matrix=matrix,
        metadata=metadata.reset_index(drop=True),
        stimulus_sample_map=stimulus_sample_map,
        included_dates=included_dates,
        excluded_dates=excluded_dates_tuple,
        parameters={
            "neural_path": str(Path(neural_path)),
            "matrix_path": str(Path(matrix_path)),
            "metadata_path": str(Path(metadata_path)),
            "exclude_dates": excluded_dates_tuple,
            "keep_dates": _normalize_filter_dates(keep_dates),
        },
    )


def build_anchor_dataset(
    neural_path: str | Path,
    anchor_stimuli: Iterable[object],
    exclude_dates: Iterable[object] | None = None,
    keep_dates: Iterable[object] | None = None,
) -> AnchorDataset:
    """Load raw neural rows for a requested set of anchor stimuli."""
    anchors = tuple(str(value).strip() for value in anchor_stimuli if str(value).strip())
    neural = pd.read_parquet(neural_path)
    neural, included_dates, excluded_dates_tuple = _filter_neural_dates(
        neural,
        exclude_dates=exclude_dates,
        keep_dates=keep_dates,
    )
    neural = _filter_anchor_stimuli(neural, anchors)
    included_dates = _unique_dates(neural)

    return AnchorDataset(
        neural=neural.reset_index(drop=True),
        anchor_stimuli=anchors,
        included_dates=included_dates,
        excluded_dates=excluded_dates_tuple,
        parameters={
            "neural_path": str(Path(neural_path)),
            "anchor_stimuli": anchors,
            "exclude_dates": excluded_dates_tuple,
            "keep_dates": _normalize_filter_dates(keep_dates),
        },
    )


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=separator)
    raise ValueError(f"unsupported metadata file type: {path.suffix}")


def _apply_date_filters(
    neural: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    exclude_dates: Iterable[object] | None,
    keep_dates: Iterable[object] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    neural, included_dates, excluded_dates_tuple = _filter_neural_dates(
        neural,
        exclude_dates=exclude_dates,
        keep_dates=keep_dates,
    )
    metadata = _normalize_date_column(metadata)
    metadata = _filter_frame_by_date(metadata, exclude_dates=exclude_dates, keep_dates=keep_dates)
    return neural, metadata, included_dates, excluded_dates_tuple


def _filter_neural_dates(
    neural: pd.DataFrame,
    *,
    exclude_dates: Iterable[object] | None,
    keep_dates: Iterable[object] | None,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    normalized = _normalize_date_column(neural)
    original_dates = _unique_dates(normalized)
    filtered = _filter_frame_by_date(normalized, exclude_dates=exclude_dates, keep_dates=keep_dates)
    included_dates = _unique_dates(filtered)
    excluded_dates_tuple = tuple(date for date in original_dates if date not in set(included_dates))
    return filtered, included_dates, excluded_dates_tuple


def _normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        return frame.copy()
    normalized = frame.copy()
    normalized["date"] = normalized["date"].map(_normalize_date_value)
    return normalized


def _filter_frame_by_date(
    frame: pd.DataFrame,
    *,
    exclude_dates: Iterable[object] | None,
    keep_dates: Iterable[object] | None,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        return frame.copy()

    filtered = frame.copy()
    keep = set(_normalize_filter_dates(keep_dates))
    exclude = set(_normalize_filter_dates(exclude_dates))
    if keep:
        filtered = filtered.loc[filtered["date"].isin(keep)]
    if exclude:
        filtered = filtered.loc[~filtered["date"].isin(exclude)]
    return filtered


def _normalize_filter_dates(values: Iterable[object] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = [_normalize_date_value(value) for value in values]
    return tuple(dict.fromkeys(value for value in normalized if value))


def _normalize_date_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit() and len(text) == 8:
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%Y%m%d")
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return text


def _unique_dates(frame: pd.DataFrame) -> tuple[str, ...]:
    if "date" not in frame.columns:
        return ()
    return tuple(sorted(value for value in frame["date"].dropna().astype(str).unique().tolist() if value))


def _build_minimal_stimulus_sample_map(neural: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    columns = ["stimulus", "stim_name", "sample_id"]
    if {"stimulus", "stim_name"}.issubset(neural.columns):
        return build_stimulus_sample_map(neural, matrix_sample_ids=matrix.index)

    if "stimulus" not in neural.columns:
        return pd.DataFrame(columns=columns)

    mapping = neural.loc[:, ["stimulus"]].drop_duplicates().copy()
    mapping["stimulus"] = mapping["stimulus"].fillna("").astype(str).str.strip()
    mapping = mapping.loc[mapping["stimulus"] != ""]
    if "stim_name" in neural.columns:
        names = (
            neural.loc[:, ["stimulus", "stim_name"]]
            .dropna(subset=["stimulus"])
            .drop_duplicates(subset=["stimulus"], keep="first")
        )
        names["stimulus"] = names["stimulus"].astype(str).str.strip()
        names["stim_name"] = names["stim_name"].fillna("").astype(str).str.strip()
        mapping = mapping.merge(names, on="stimulus", how="left")
    else:
        mapping["stim_name"] = mapping["stimulus"]
    mapping["sample_id"] = ""
    return mapping.loc[:, columns].reset_index(drop=True)


def _filter_anchor_stimuli(neural: pd.DataFrame, anchors: tuple[str, ...]) -> pd.DataFrame:
    if not anchors:
        return neural.iloc[0:0].copy()
    anchor_set = set(anchors)
    masks = []
    for column in ("stimulus", "stim_name"):
        if column in neural.columns:
            masks.append(neural[column].astype(str).isin(anchor_set))
    if not masks:
        return neural.iloc[0:0].copy()
    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    return neural.loc[mask].copy()
