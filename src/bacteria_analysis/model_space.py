"""Loaders for Stage 3 model-space inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

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
BOOL_TRUE_VALUES = {"true", "1", "yes", "y", "t"}
BOOL_FALSE_VALUES = {"false", "0", "no", "n", "f", ""}
UNION_LIKE_KEYWORDS = ("union", "broad", "combined", "mixture", "mix")


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
    try:
        frame = pd.read_excel(path, engine="openpyxl")
    except (ImportError, ModuleNotFoundError, ValueError):
        frame = _read_minimal_xlsx(path)
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

    _require_non_empty(normalized, "model_id")
    _require_unique(normalized, "model_id", "model_registry")
    _reject_primary_union_like_models(normalized)
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

    normalized[sample_column] = sample_ids
    normalized = normalized.set_index(sample_column)
    normalized.index.name = "sample_id"
    return normalized


def _read_minimal_xlsx(path: Path) -> pd.DataFrame:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in archive.namelist():
            raise ValueError("matrix workbook must include xl/worksheets/sheet1.xml")
        root = ET.fromstring(archive.read(sheet_path))

    rows: list[list[object]] = []
    for row_element in root.findall(".//main:sheetData/main:row", namespace):
        values_by_index: dict[int, object] = {}
        max_index = 0
        for cell in row_element.findall("main:c", namespace):
            ref = cell.attrib.get("r", "")
            column_index = _excel_column_to_index(_split_cell_reference(ref))
            max_index = max(max_index, column_index)
            values_by_index[column_index] = _read_xlsx_cell_value(cell, namespace)
        rows.append([values_by_index.get(index) for index in range(1, max_index + 1)])

    if not rows:
        return pd.DataFrame()

    headers = ["" if value is None else str(value).strip() for value in rows[0]]
    data = []
    for row_values in rows[1:]:
        padded = list(row_values) + [None] * max(0, len(headers) - len(row_values))
        data.append({headers[index]: padded[index] for index in range(len(headers))})

    return pd.DataFrame.from_records(data, columns=headers)


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


def _reject_primary_union_like_models(registry: pd.DataFrame) -> None:
    primary_rows = registry.loc[registry["model_tier"].str.lower().eq(PRIMARY_TIER_VALUE)].copy()
    if primary_rows.empty:
        return

    status_is_supplementary = primary_rows["model_status"].str.lower().eq(SUPPLEMENTARY_TIER_VALUE)
    status_is_other_non_primary = ~primary_rows["model_status"].str.lower().isin({PRIMARY_TIER_VALUE, SUPPLEMENTARY_TIER_VALUE})
    broad_text = primary_rows[["model_id", "model_label", "description"]].apply(
        lambda column: column.str.contains("|".join(UNION_LIKE_KEYWORDS), case=False, na=False)
    ).any(axis=1)
    invalid = primary_rows.loc[status_is_supplementary | status_is_other_non_primary | broad_text]
    if not invalid.empty:
        raise ValueError("broad union models must be marked supplementary instead of primary")


def _validate_mapping_against_matrix(mapping: pd.DataFrame, matrix: pd.DataFrame) -> None:
    matrix_sample_ids = set(matrix.index.astype(str))
    mapped_sample_ids = set(mapping["sample_id"].astype(str))
    missing = sorted(mapped_sample_ids.difference(matrix_sample_ids))
    if missing:
        raise ValueError(f"mapped sample_id values must exist in the matrix: {', '.join(missing)}")


def _validate_annotation_against_matrix(annotation: pd.DataFrame, matrix: pd.DataFrame) -> None:
    matrix_metabolites = set(matrix.columns.astype(str))
    annotation_metabolites = set(annotation["metabolite_name"].astype(str))
    if annotation_metabolites and not annotation_metabolites.issubset(matrix_metabolites):
        missing = sorted(annotation_metabolites.difference(matrix_metabolites))
        raise ValueError(f"annotation metabolites must exist in the matrix: {', '.join(missing)}")


def _seed_global_profile_membership(membership: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    resolved = membership.copy()
    if "global_profile" in set(resolved["model_id"].astype(str)):
        return resolved

    resolved = _ensure_membership_columns(resolved)
    global_rows = pd.DataFrame(
        {
            "model_id": ["global_profile"] * len(matrix.columns),
            "metabolite_name": matrix.columns.astype(str).tolist(),
            "membership_source": ["matrix_all_columns"] * len(matrix.columns),
            "review_status": [""] * len(matrix.columns),
            "ambiguous_flag": [False] * len(matrix.columns),
            "notes": [""] * len(matrix.columns),
        }
    )

    if resolved.empty:
        return global_rows

    return pd.concat([resolved, global_rows], ignore_index=True)


def _seed_global_profile_registry(registry: pd.DataFrame) -> pd.DataFrame:
    resolved = registry.copy()
    if "global_profile" in set(resolved["model_id"].astype(str)):
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


def _split_cell_reference(reference: str) -> str:
    letters = "".join(character for character in reference if character.isalpha())
    if not letters:
        raise ValueError("xlsx cell reference is missing a column")
    return letters


def _excel_column_to_index(column_reference: str) -> int:
    index = 0
    for character in column_reference.upper():
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index


def _read_xlsx_cell_value(cell: ET.Element, namespace: dict[str, str]) -> object:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        text_element = cell.find("main:is/main:t", namespace)
        return "" if text_element is None or text_element.text is None else text_element.text

    value_element = cell.find("main:v", namespace)
    if value_element is None or value_element.text is None:
        return None

    text = value_element.text.strip()
    if cell_type == "b":
        return text == "1"

    try:
        number = float(text)
    except ValueError:
        return text

    if number.is_integer():
        return int(number)
    return number
