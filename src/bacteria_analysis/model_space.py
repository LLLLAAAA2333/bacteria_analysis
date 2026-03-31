"""Loaders for Stage 3 model-space inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

STIMULUS_SAMPLE_MAP_REQUIRED_COLUMNS = ("stimulus", "stim_name", "sample_id")
METABOLITE_ANNOTATION_REQUIRED_COLUMNS = (
    "metabolite_name",
    "superclass",
    "subclass",
    "pathway_tag",
    "annotation_source",
    "review_status",
    "ambiguous_flag",
    "notes",
)
MODEL_REGISTRY_REQUIRED_COLUMNS = (
    "model_id",
    "model_label",
    "model_tier",
    "model_status",
    "feature_kind",
    "distance_kind",
    "description",
    "authority",
    "notes",
)
MODEL_MEMBERSHIP_REQUIRED_COLUMNS = ("model_id", "metabolite_name")
MODEL_MEMBERSHIP_OPTIONAL_COLUMNS = ("membership_source", "review_status", "ambiguous_flag", "notes")
PRIMARY_TIER_VALUE = "primary"
SUPPLEMENTARY_TIER_VALUE = "supplementary"
MODEL_TIER_ALLOWED_VALUES = (PRIMARY_TIER_VALUE, SUPPLEMENTARY_TIER_VALUE)
MODEL_STATUS_ALLOWED_VALUES = (PRIMARY_TIER_VALUE, SUPPLEMENTARY_TIER_VALUE, "draft", "excluded")
FEATURE_KIND_ALLOWED_VALUES = ("continuous_abundance", "binary_presence")
DISTANCE_KIND_ALLOWED_VALUES = ("correlation", "euclidean", "jaccard")
BOOL_TRUE_VALUES = {"true", "1", "yes", "y", "t"}
BOOL_FALSE_VALUES = {"false", "0", "no", "n", "f", ""}


def load_stimulus_sample_map(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), dtype=str).fillna("")
    return _validate_stimulus_sample_map(frame)


def load_metabolite_annotation(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), dtype=str).fillna("")
    return _validate_metabolite_annotation(frame)


def load_model_registry(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), dtype=str).fillna("")
    return _validate_model_registry(frame)


def load_model_membership(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), dtype=str).fillna("")
    return _validate_model_membership(frame)


def read_metabolite_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        first_sheet = workbook.worksheets[0]
        header_row = next(first_sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
    finally:
        workbook.close()

    header_values = ["" if value is None else str(value).strip() for value in header_row]
    if header_values and pd.Index(header_values).duplicated().any():
        raise ValueError("metabolite column names must be unique")

    frame = pd.read_excel(path, engine="openpyxl", sheet_name=0)
    return _normalize_matrix_frame(frame)


def build_metabolite_annotation_skeleton(matrix_path: str | Path) -> pd.DataFrame:
    matrix = read_metabolite_matrix(matrix_path)
    metabolite_names = matrix.columns.tolist()
    return pd.DataFrame(
        {
            "metabolite_name": metabolite_names,
            "superclass": [""] * len(metabolite_names),
            "subclass": [""] * len(metabolite_names),
            "pathway_tag": [""] * len(metabolite_names),
            "annotation_source": [""] * len(metabolite_names),
            "review_status": [""] * len(metabolite_names),
            "ambiguous_flag": [False] * len(metabolite_names),
            "notes": [""] * len(metabolite_names),
        }
    )


def resolve_model_inputs(model_input_root: str | Path, matrix_path: str | Path) -> dict[str, pd.DataFrame]:
    model_input_root = Path(model_input_root)
    mapping = load_stimulus_sample_map(model_input_root / "stimulus_sample_map.csv")
    annotation = load_metabolite_annotation(model_input_root / "metabolite_annotation.csv")
    registry = load_model_registry(model_input_root / "model_registry.csv")
    membership = load_model_membership(model_input_root / "model_membership.csv")
    return _resolve_stage3_inputs(mapping, annotation, registry, membership, matrix_path)


def _validate_stimulus_sample_map(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _require_columns(frame, STIMULUS_SAMPLE_MAP_REQUIRED_COLUMNS, "stimulus_sample_map")
    for column in STIMULUS_SAMPLE_MAP_REQUIRED_COLUMNS:
        normalized[column] = normalized[column].astype(str).str.strip()

    for column in STIMULUS_SAMPLE_MAP_REQUIRED_COLUMNS:
        _require_non_empty(normalized, column)
    for column in ("stimulus", "sample_id"):
        _require_unique(normalized, column, "stimulus_sample_map")

    return normalized


def _validate_metabolite_annotation(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _require_columns(frame, METABOLITE_ANNOTATION_REQUIRED_COLUMNS, "metabolite_annotation")
    for column in METABOLITE_ANNOTATION_REQUIRED_COLUMNS:
        if column == "ambiguous_flag":
            continue
        normalized[column] = normalized[column].astype(str).str.strip()

    normalized["ambiguous_flag"] = _coerce_boolean_column(normalized["ambiguous_flag"], "ambiguous_flag")
    _require_non_empty(normalized, "metabolite_name")
    _require_unique(normalized, "metabolite_name", "metabolite_annotation")
    return normalized


def _validate_model_registry(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _require_columns(frame, MODEL_REGISTRY_REQUIRED_COLUMNS, "model_registry")
    for column in MODEL_REGISTRY_REQUIRED_COLUMNS:
        normalized[column] = normalized[column].astype(str).str.strip()

    normalized["model_id"] = normalized["model_id"].astype(str).str.strip().str.lower()
    for column in ("model_id", "model_tier", "model_status"):
        _require_non_empty(normalized, column)
    _require_allowed_values(normalized, "model_tier", MODEL_TIER_ALLOWED_VALUES)
    _require_allowed_values(normalized, "model_status", MODEL_STATUS_ALLOWED_VALUES)
    _require_allowed_values(normalized, "feature_kind", FEATURE_KIND_ALLOWED_VALUES)
    _require_allowed_values(normalized, "distance_kind", DISTANCE_KIND_ALLOWED_VALUES)
    for column in ("model_tier", "model_status", "feature_kind", "distance_kind"):
        normalized[column] = normalized[column].astype(str).str.strip().str.lower()
    _require_unique(normalized, "model_id", "model_registry")
    return normalized


def _validate_model_membership(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _require_columns(frame, MODEL_MEMBERSHIP_REQUIRED_COLUMNS, "model_membership")

    for column in MODEL_MEMBERSHIP_REQUIRED_COLUMNS:
        normalized[column] = normalized[column].astype(str).str.strip()
    for column in MODEL_MEMBERSHIP_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
        elif column != "ambiguous_flag":
            normalized[column] = normalized[column].astype(str).str.strip()

    normalized["model_id"] = normalized["model_id"].astype(str).str.strip().str.lower()
    normalized["ambiguous_flag"] = _coerce_boolean_column(normalized["ambiguous_flag"], "ambiguous_flag")
    if not normalized.empty:
        _require_non_empty(normalized, "model_id")
        _require_non_empty(normalized, "metabolite_name")
        _require_unique_pairs(normalized, ("model_id", "metabolite_name"), "model_membership")

    return normalized


def _resolve_stage3_inputs(
    mapping: pd.DataFrame,
    annotation: pd.DataFrame,
    registry: pd.DataFrame,
    membership: pd.DataFrame,
    matrix_path: str | Path,
) -> dict[str, pd.DataFrame]:
    matrix = read_metabolite_matrix(matrix_path)
    _validate_mapping_against_matrix(mapping, matrix)
    _validate_annotation_against_matrix(annotation, matrix)
    _validate_membership_against_matrix(membership, matrix)

    resolved_registry = _seed_global_profile_registry(registry)
    resolved_membership = _seed_global_profile_membership(membership, matrix)
    resolved_membership = _validate_membership_against_registry(resolved_membership, resolved_registry)
    resolved_registry["is_primary_family"] = resolved_registry["model_tier"].str.lower().eq(PRIMARY_TIER_VALUE)
    resolved_registry["is_supplementary_family"] = resolved_registry["model_tier"].str.lower().eq(
        SUPPLEMENTARY_TIER_VALUE
    )

    return {
        "matrix": matrix,
        "stimulus_sample_map": mapping.copy(),
        "metabolite_annotation": annotation.copy(),
        "model_registry": registry.copy(),
        "model_membership": membership.copy(),
        "model_registry_resolved": resolved_registry,
        "model_membership_resolved": resolved_membership,
    }


def _normalize_matrix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return normalized

    if "sample_id" in normalized.columns:
        sample_column = "sample_id"
    else:
        sample_column = normalized.columns[0]

    sample_ids = normalized[sample_column]
    if sample_ids.isna().any():
        raise ValueError("sample_id values must be non-empty")

    sample_ids = sample_ids.astype(str).str.strip()
    if (sample_ids == "").any():
        raise ValueError("sample_id values must be non-empty")
    if sample_ids.duplicated().any():
        raise ValueError("sample_id values must be unique")

    metabolite_columns = [column for column in normalized.columns if column != sample_column]
    metabolite_column_names = ["" if column is None else str(column).strip() for column in metabolite_columns]
    if any(not name or name.lower().startswith("unnamed:") for name in metabolite_column_names):
        raise ValueError("metabolite column names must be non-empty")
    if pd.Index(metabolite_column_names).duplicated().any():
        raise ValueError("metabolite column names must be unique")

    normalized[sample_column] = sample_ids
    normalized = normalized.set_index(sample_column)
    normalized.index.name = "sample_id"
    return normalized


def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} must include columns: {', '.join(missing)}")
    return frame.copy()


def _require_non_empty(frame: pd.DataFrame, column: str) -> None:
    values = frame[column].astype(str).str.strip()
    if (values == "").any():
        raise ValueError(f"{column} values must be non-empty")


def _require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].duplicated().any():
        raise ValueError(f"{label} {column} values must be unique")


def _require_allowed_values(frame: pd.DataFrame, column: str, allowed_values: tuple[str, ...]) -> None:
    normalized = frame[column].astype(str).str.strip().str.lower()
    if not normalized.isin(allowed_values).all():
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{column} values must be one of: {allowed}")


def _require_unique_pairs(frame: pd.DataFrame, columns: tuple[str, str], label: str) -> None:
    duplicated = frame.duplicated(list(columns))
    if duplicated.any():
        left, right = columns
        raise ValueError(f"{label} {left} and {right} pairs must be unique")


def _coerce_boolean_column(series: pd.Series, column_name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(BOOL_TRUE_VALUES | BOOL_FALSE_VALUES)
    if invalid.any():
        raise ValueError(f"{column_name} values must be boolean-like")
    return normalized.isin(BOOL_TRUE_VALUES)
def _validate_mapping_against_matrix(mapping: pd.DataFrame, matrix: pd.DataFrame) -> None:
    matrix_sample_ids = set(matrix.index.astype(str))
    mapped_sample_ids = set(mapping["sample_id"].astype(str))
    missing = sorted(mapped_sample_ids.difference(matrix_sample_ids))
    if missing:
        raise ValueError(f"mapped sample_id values must exist in the matrix: {', '.join(missing)}")


def _validate_annotation_against_matrix(annotation: pd.DataFrame, matrix: pd.DataFrame) -> None:
    matrix_metabolites = set(matrix.columns.astype(str))
    annotation_metabolites = set(annotation["metabolite_name"].astype(str))
    missing = sorted(matrix_metabolites.difference(annotation_metabolites))
    if missing:
        raise ValueError(f"annotation metabolites must cover all matrix metabolites: {', '.join(missing)}")

    extra = sorted(annotation_metabolites.difference(matrix_metabolites))
    if extra:
        raise ValueError(f"annotation metabolites must exist in the matrix: {', '.join(extra)}")


def _seed_global_profile_membership(membership: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    resolved = _ensure_membership_columns(membership)
    matrix_metabolites = matrix.columns.astype(str).tolist()
    global_profile_mask = resolved["model_id"].astype(str).str.strip().str.lower() == "global_profile"
    global_profile_rows = resolved.loc[global_profile_mask]

    seed_model_id = "global_profile"
    if not global_profile_rows.empty:
        seed_model_id = global_profile_rows["model_id"].astype(str).iloc[0].strip()

    existing_metabolites = set(global_profile_rows["metabolite_name"].astype(str))
    missing_metabolites = [name for name in matrix_metabolites if name not in existing_metabolites]

    if not missing_metabolites:
        return resolved

    seeded_rows = pd.DataFrame(
        {
            "model_id": [seed_model_id] * len(missing_metabolites),
            "metabolite_name": missing_metabolites,
            "membership_source": ["matrix_all_columns"] * len(missing_metabolites),
            "review_status": [""] * len(missing_metabolites),
            "ambiguous_flag": [False] * len(missing_metabolites),
            "notes": [""] * len(missing_metabolites),
        }
    )
    if resolved.empty:
        return seeded_rows
    return pd.concat([resolved, seeded_rows], ignore_index=True)


def _seed_global_profile_registry(registry: pd.DataFrame) -> pd.DataFrame:
    resolved = registry.copy()
    if resolved["model_id"].astype(str).str.strip().str.lower().eq("global_profile").any():
        return resolved

    global_profile = pd.DataFrame.from_records(
        [
            {
                "model_id": "global_profile",
                "model_label": "Global Metabolite Profile",
                "model_tier": "primary",
                "model_status": "primary",
                "feature_kind": "continuous_abundance",
                "distance_kind": "correlation",
                "description": "All matrix metabolites",
                "authority": "user",
                "notes": "",
            }
        ]
    )
    return pd.concat([resolved, global_profile], ignore_index=True)


def _ensure_membership_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in MODEL_MEMBERSHIP_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "" if column != "ambiguous_flag" else False

    normalized["membership_source"] = normalized["membership_source"].astype(str).str.strip()
    normalized["review_status"] = normalized["review_status"].astype(str).str.strip()
    normalized["notes"] = normalized["notes"].astype(str).str.strip()
    normalized["ambiguous_flag"] = normalized["ambiguous_flag"].astype(bool)
    return normalized[list(MODEL_MEMBERSHIP_REQUIRED_COLUMNS + MODEL_MEMBERSHIP_OPTIONAL_COLUMNS)]


def _validate_membership_against_registry(membership: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return membership

    registry_model_ids = set(registry["model_id"].astype(str))
    membership_model_ids = set(membership["model_id"].astype(str))
    unknown = sorted(membership_model_ids.difference(registry_model_ids))
    if unknown:
        raise ValueError(f"model_membership references unknown model_id values: {', '.join(unknown)}")

    return membership


def _validate_membership_against_matrix(membership: pd.DataFrame, matrix: pd.DataFrame) -> None:
    matrix_metabolites = set(matrix.columns.astype(str))
    membership_metabolites = set(membership["metabolite_name"].astype(str))
    unknown = sorted(membership_metabolites.difference(matrix_metabolites))
    if unknown:
        raise ValueError(f"model_membership metabolites must exist in the matrix: {', '.join(unknown)}")
