"""Chemical feature and RDM construction from metabolite matrices."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from bacteria_analysis.neural_features import RdmResult

SUPPORTED_TRANSFORMS = ("none", "log2")
SUPPORTED_DISTANCES = ("euclidean", "correlation")

NAME_COLUMN_ALIASES = ("metabolite_name", "metabolitename", "name")
QCRSD_COLUMN_ALIASES = ("QCRSD", "qcrsd", "qc_rsd", "qc rsd", "qcrsd_percent")
TAXONOMY_COLUMN_ALIASES = {
    "superclass": ("SuperClass", "superclass", "super_class", "super class"),
    "class": ("Class", "class"),
    "subclass": ("SubClass", "subclass", "sub_class", "sub class"),
}
UNKNOWN_TAXONOMY_VALUES = {
    "",
    "-",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "unassigned",
    "unclassified",
    "not assigned",
}


def build_chemical_rdm(
    dataset,
    *,
    qc_threshold: float = 0.2,
    transform: str = "log2",
    distance: str = "euclidean",
) -> RdmResult:
    """Build a stimulus-level chemical RDM from a sample-by-feature matrix."""

    _validate_options(qc_threshold=qc_threshold, transform=transform, distance=distance)
    matrix = _numeric_matrix(_dataset_matrix(dataset))
    annotations = _feature_annotations(_dataset_metadata(dataset), require_name=False)
    retained_features = _retained_features(
        matrix.columns.astype(str).tolist(),
        annotations,
        qc_threshold=qc_threshold,
    )
    if not retained_features:
        raise ValueError("no chemical features passed QC")

    stimulus_matrix = _stimulus_matrix(dataset, matrix.loc[:, retained_features])
    transformed = _transform_matrix(stimulus_matrix, transform=transform)
    rdm = _build_rdm_matrix(transformed, distance=distance)
    return RdmResult(
        matrix=rdm,
        metadata={
            "feature_count": int(len(retained_features)),
            "retained_features": tuple(retained_features),
            "qc_threshold": float(qc_threshold),
            "transform": transform,
            "distance": distance,
            "n_stimuli": int(len(stimulus_matrix)),
            "n_samples": int(stimulus_matrix.index.nunique()),
            "qcrsd_filter_applied": bool(_has_qcrsd(annotations)),
        },
    )


def build_chemical_class_rdms(
    dataset,
    *,
    taxonomy_level: str = "Class",
    qc_threshold: float = 0.2,
    min_features: int = 3,
    transform: str = "log2",
    distance: str = "euclidean",
) -> dict[str, RdmResult]:
    """Build chemical RDMs for each retained taxonomy category."""

    _validate_options(qc_threshold=qc_threshold, transform=transform, distance=distance)
    if min_features < 1:
        raise ValueError("min_features must be at least 1")

    matrix = _numeric_matrix(_dataset_matrix(dataset))
    annotations = _feature_annotations(_dataset_metadata(dataset), require_name=True)
    taxonomy_column = _taxonomy_column(annotations, taxonomy_level)
    taxonomy_key = _canonical_taxonomy_key(taxonomy_level)

    results: dict[str, RdmResult] = {}
    for category in sorted({_clean_category(value) for value in annotations[taxonomy_column].unique()}):
        if not category:
            continue

        category_annotations = annotations.loc[annotations[taxonomy_column].map(_clean_category) == category]
        category_features = set(category_annotations["_feature_name"])
        candidate_features = [
            feature
            for feature in matrix.columns.astype(str).tolist()
            if feature in category_features
        ]
        retained_features = _retained_features(
            candidate_features,
            category_annotations,
            qc_threshold=qc_threshold,
        )
        if len(retained_features) < min_features:
            continue

        stimulus_matrix = _stimulus_matrix(dataset, matrix.loc[:, retained_features])
        transformed = _transform_matrix(stimulus_matrix, transform=transform)
        results[category] = RdmResult(
            matrix=_build_rdm_matrix(transformed, distance=distance),
            metadata={
                "taxonomy_level": taxonomy_key,
                "category": category,
                "feature_count": int(len(retained_features)),
                "retained_features": tuple(retained_features),
                "qc_threshold": float(qc_threshold),
                "transform": transform,
                "distance": distance,
                "n_stimuli": int(len(stimulus_matrix)),
                "n_samples": int(stimulus_matrix.index.nunique()),
                "qcrsd_filter_applied": bool(_has_qcrsd(category_annotations)),
            },
        )

    return results


def _dataset_matrix(dataset) -> pd.DataFrame:
    if hasattr(dataset, "matrix") and isinstance(dataset.matrix, pd.DataFrame):
        return dataset.matrix
    raise ValueError("dataset must expose a matrix DataFrame")


def _dataset_metadata(dataset) -> pd.DataFrame:
    if hasattr(dataset, "metadata") and isinstance(dataset.metadata, pd.DataFrame):
        return dataset.metadata
    return pd.DataFrame()


def _validate_options(*, qc_threshold: float, transform: str, distance: str) -> None:
    if not 0 <= qc_threshold <= 1:
        raise ValueError("qc_threshold must be a fraction between 0 and 1")
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"unsupported chemical transform {transform!r}")
    if distance not in SUPPORTED_DISTANCES:
        raise ValueError(f"unsupported chemical distance {distance!r}")


def _numeric_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("matrix must be a pandas DataFrame")
    normalized = matrix.copy()
    normalized.index = pd.Index(_clean_label(value) for value in normalized.index)
    normalized.columns = pd.Index(_clean_label(value) for value in normalized.columns)

    if normalized.index.has_duplicates:
        duplicates = normalized.index[normalized.index.duplicated()].unique().tolist()
        raise ValueError(f"matrix sample_id labels must be unique: {duplicates}")
    if normalized.columns.has_duplicates:
        duplicates = normalized.columns[normalized.columns.duplicated()].unique().tolist()
        raise ValueError(f"matrix feature labels must be unique: {duplicates}")
    if any(label == "" for label in normalized.index):
        raise ValueError("matrix sample_id labels must be non-empty")
    if any(label == "" for label in normalized.columns):
        raise ValueError("matrix feature labels must be non-empty")

    numeric = normalized.apply(pd.to_numeric, errors="coerce")
    bad_mask = numeric.isna() & normalized.notna()
    if bad_mask.to_numpy().any():
        bad_features = sorted({column for column in bad_mask.columns[bad_mask.any(axis=0)]})
        raise ValueError(f"matrix contains non-numeric values in features: {', '.join(bad_features)}")
    numeric.index = normalized.index
    numeric.columns = normalized.columns
    return numeric


def _feature_annotations(metadata: pd.DataFrame, *, require_name: bool) -> pd.DataFrame:
    columns = [
        "_feature_name",
        "_qcrsd_fraction",
        "_has_qcrsd_metadata",
        *metadata.columns.astype(str).tolist(),
    ]
    if metadata.empty:
        if require_name:
            raise ValueError("chemical metadata must include metabolite_name or name")
        return pd.DataFrame(columns=columns)

    name_column = _find_column(metadata, NAME_COLUMN_ALIASES)
    if name_column is None:
        if require_name or _find_column(metadata, QCRSD_COLUMN_ALIASES) is not None:
            raise ValueError("chemical metadata must include metabolite_name or name")
        return pd.DataFrame(columns=columns)

    annotations = metadata.copy()
    annotations["_feature_name"] = annotations[name_column].map(_clean_label)
    annotations = annotations.loc[annotations["_feature_name"] != ""].copy()
    if annotations["_feature_name"].duplicated().any():
        duplicates = (
            annotations.loc[annotations["_feature_name"].duplicated(), "_feature_name"]
            .unique()
            .tolist()
        )
        raise ValueError(f"chemical metadata has duplicate feature names: {duplicates}")

    qcrsd_column = _find_column(annotations, QCRSD_COLUMN_ALIASES)
    if qcrsd_column is not None:
        annotations["_qcrsd_fraction"] = _qcrsd_fraction(annotations[qcrsd_column], annotations["_feature_name"])
        annotations["_has_qcrsd_metadata"] = True
    else:
        annotations["_qcrsd_fraction"] = np.nan
        annotations["_has_qcrsd_metadata"] = False
    return annotations


def _retained_features(
    candidate_features: list[str],
    annotations: pd.DataFrame,
    *,
    qc_threshold: float,
) -> list[str]:
    if annotations.empty or not _has_qcrsd(annotations):
        return list(candidate_features)

    qcrsd_by_feature = annotations.set_index("_feature_name")["_qcrsd_fraction"]
    retained: list[str] = []
    for feature in candidate_features:
        value = qcrsd_by_feature.get(feature, np.nan)
        if np.isfinite(value) and float(value) <= qc_threshold:
            retained.append(feature)
    return retained


def _has_qcrsd(annotations: pd.DataFrame) -> bool:
    return "_has_qcrsd_metadata" in annotations.columns and bool(annotations["_has_qcrsd_metadata"].any())


def _qcrsd_fraction(raw: pd.Series, feature_names: pd.Series) -> pd.Series:
    values = pd.to_numeric(raw, errors="coerce")
    bad_mask = values.isna() & raw.notna() & raw.astype(str).str.strip().ne("")
    if bad_mask.any():
        bad_features = feature_names.loc[bad_mask].astype(str).tolist()
        raise ValueError(f"QCRSD values must be numeric for features: {', '.join(bad_features)}")
    values = values.astype(float)
    negative_mask = values.lt(0)
    if negative_mask.any():
        bad_features = feature_names.loc[negative_mask].astype(str).tolist()
        raise ValueError(f"QCRSD values must be non-negative for features: {', '.join(bad_features)}")
    return values.where(values <= 1.0, values / 100.0)


def _taxonomy_column(annotations: pd.DataFrame, taxonomy_level: str) -> str:
    taxonomy_key = _canonical_taxonomy_key(taxonomy_level)
    column = _find_column(annotations, TAXONOMY_COLUMN_ALIASES[taxonomy_key])
    if column is None:
        raise ValueError(f"chemical metadata must include a {taxonomy_level} taxonomy column")
    return column


def _canonical_taxonomy_key(taxonomy_level: str) -> str:
    normalized = _normalize_column_key(taxonomy_level)
    for key, aliases in TAXONOMY_COLUMN_ALIASES.items():
        if normalized in {_normalize_column_key(alias) for alias in aliases}:
            return key
    raise ValueError(f"unsupported taxonomy_level {taxonomy_level!r}")


def _stimulus_matrix(dataset, matrix: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(dataset, "stimulus_sample_map") or not isinstance(dataset.stimulus_sample_map, pd.DataFrame):
        raise ValueError("dataset must expose a stimulus_sample_map DataFrame")

    mapping = dataset.stimulus_sample_map.copy()
    required = {"stimulus", "sample_id"}
    if not required.issubset(mapping.columns):
        raise ValueError("stimulus_sample_map must include stimulus and sample_id columns")

    mapping["stimulus"] = mapping["stimulus"].map(_clean_label)
    mapping["sample_id"] = mapping["sample_id"].map(_clean_label)
    mapping = mapping.loc[(mapping["stimulus"] != "") & (mapping["sample_id"] != "")].copy()
    if mapping.empty:
        raise ValueError("stimulus_sample_map has no stimulus-to-sample rows")
    if mapping["stimulus"].duplicated().any():
        duplicates = mapping.loc[mapping["stimulus"].duplicated(), "stimulus"].unique().tolist()
        raise ValueError(f"stimulus_sample_map has duplicate stimuli: {duplicates}")
    if mapping["sample_id"].duplicated().any():
        duplicates = mapping.loc[mapping["sample_id"].duplicated(), "sample_id"].unique().tolist()
        raise ValueError(f"stimulus_sample_map has duplicate sample_ids: {duplicates}")

    sample_ids = mapping["sample_id"].tolist()
    missing_sample_ids = sorted(set(sample_ids).difference(matrix.index))
    if missing_sample_ids:
        raise ValueError(f"stimulus_sample_map sample_ids missing from matrix: {', '.join(missing_sample_ids)}")

    stimulus_matrix = matrix.loc[sample_ids].copy()
    stimulus_matrix.index = pd.Index(mapping["stimulus"].tolist(), name="stimulus")
    return stimulus_matrix


def _transform_matrix(matrix: pd.DataFrame, *, transform: str) -> pd.DataFrame:
    if transform == "none":
        return matrix.copy()

    nonpositive = matrix.le(0) & matrix.notna()
    if nonpositive.to_numpy().any():
        bad_features = sorted({column for column in nonpositive.columns[nonpositive.any(axis=0)]})
        raise ValueError(f"log2 transform requires positive values in retained features: {', '.join(bad_features)}")

    transformed = np.log2(matrix.astype(float))
    transformed.index = matrix.index
    transformed.columns = matrix.columns
    return transformed


def _build_rdm_matrix(values: pd.DataFrame, *, distance: str) -> pd.DataFrame:
    labels = values.index.astype(str).tolist()
    array = values.to_numpy(dtype=float, copy=False)
    distances = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)

    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            pair_distance = _pair_distance(array[left_index], array[right_index], distance=distance)
            distances[left_index, right_index] = pair_distance
            distances[right_index, left_index] = pair_distance

    return pd.DataFrame(distances, index=labels, columns=labels)


def _pair_distance(left: np.ndarray, right: np.ndarray, *, distance: str) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left_values = left[valid]
    right_values = right[valid]
    if distance == "euclidean":
        if left_values.size == 0:
            return float("nan")
        return float(np.linalg.norm(left_values - right_values))

    if left_values.size < 2:
        return float("nan")
    if np.std(left_values) == 0.0 or np.std(right_values) == 0.0:
        return float("nan")
    correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    if not np.isfinite(correlation):
        return float("nan")
    return 1.0 - float(np.clip(correlation, -1.0, 1.0))


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    alias_keys = {_normalize_column_key(alias) for alias in aliases}
    for column in frame.columns:
        if _normalize_column_key(column) in alias_keys:
            return str(column)
    return None


def _normalize_column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_category(value: object) -> str:
    text = _clean_label(value)
    if text.strip().lower() in UNKNOWN_TAXONOMY_VALUES:
        return ""
    return text


__all__ = [
    "build_chemical_class_rdms",
    "build_chemical_rdm",
]
