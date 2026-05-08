"""Shared helpers for biological-subspace neural/chemical RDM reviews."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from bacteria_analysis.constants import NEURON_ORDER
from bacteria_analysis.model_space import build_stimulus_sample_map
from bacteria_analysis.model_space_seed import RAW_METADATA_SHEET_NAME, _normalize_header_text
from bacteria_analysis.reliability import TrialView
from bacteria_analysis.rsa_aggregated_responses import (
    build_aggregated_response_rdm,
    build_grouped_aggregated_responses,
    load_aggregated_response_context_inputs,
)

VIEW_NAMES: tuple[str, str] = ("response_window", "full_trajectory")
KEEP_SEPARATE_NEURONS = {"ASEL", "ASER"}


def build_neural_rdms(preprocess_root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    context = load_aggregated_response_context_inputs(preprocess_root, view_names=VIEW_NAMES)

    rdms: dict[str, pd.DataFrame] = {}
    supports: list[pd.DataFrame] = []
    for view_name in VIEW_NAMES:
        merged_view = merge_lr_view(context.views[view_name])
        pooled_responses, support = build_grouped_aggregated_responses(
            merged_view,
            group_columns=("stimulus", "stim_name"),
            aggregation="median",
        )
        rdms[view_name] = build_aggregated_response_rdm(pooled_responses, id_columns=("stimulus",))
        supports.append(support.assign(view_name=view_name))
    return rdms, pd.concat(supports, ignore_index=True)


def merge_lr_view(view: TrialView) -> TrialView:
    _, merged_indices = build_lr_merge_plan()
    merged_slices: list[np.ndarray] = []
    for neuron_indices in merged_indices:
        subset = view.values[:, neuron_indices, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            merged_slices.append(np.nanmean(subset, axis=1))
    merged_values = np.stack(merged_slices, axis=1)
    return TrialView(
        name=view.name,
        timepoints=view.timepoints,
        metadata=view.metadata.reset_index(drop=True),
        values=merged_values,
    )


def build_lr_merge_plan() -> tuple[list[str], list[list[int]]]:
    neurons = list(NEURON_ORDER)
    present = set(neurons)
    merged_names: list[str] = []
    merged_indices: list[list[int]] = []
    seen_bases: set[str] = set()

    for index, neuron in enumerate(neurons):
        if neuron in KEEP_SEPARATE_NEURONS:
            merged_names.append(neuron)
            merged_indices.append([index])
            continue

        if neuron.endswith("L") or neuron.endswith("R"):
            base_name = neuron[:-1]
            left_name = f"{base_name}L"
            right_name = f"{base_name}R"
            if base_name in seen_bases:
                continue
            if left_name in present and right_name in present and base_name != "ASE":
                merged_names.append(base_name)
                merged_indices.append([neurons.index(left_name), neurons.index(right_name)])
                seen_bases.add(base_name)
                continue

        merged_names.append(neuron)
        merged_indices.append([index])

    return merged_names, merged_indices


def load_taxonomy_qc(raw_metadata_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(
        raw_metadata_path,
        sheet_name=RAW_METADATA_SHEET_NAME,
        usecols=lambda column: column in {"name", "QCRSD", "SuperClass", "Class", "SubClass"},
        dtype=str,
    ).fillna("")
    frame["name"] = frame["name"].astype(str).str.strip()
    frame = frame.loc[frame["name"] != ""].copy()
    frame["normalized_name"] = frame["name"].map(lambda value: _normalize_header_text(str(value))[0])
    frame["QCRSD"] = pd.to_numeric(frame["QCRSD"], errors="coerce")
    for column in ("SuperClass", "Class", "SubClass"):
        frame[column] = frame[column].astype(str).str.strip()
    return frame.loc[:, ["normalized_name", "QCRSD", "SuperClass", "Class", "SubClass"]].drop_duplicates(
        subset=["normalized_name"]
    )


def build_stimulus_mapping(preprocess_root: Path, matrix: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_parquet(preprocess_root / "trial_level" / "trial_metadata.parquet")
    return build_stimulus_sample_map(metadata, matrix_sample_ids=matrix.index)


def build_chemical_rdm(
    matrix: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame,
    metabolite_names: list[str],
) -> pd.DataFrame:
    feature_frame = matrix.loc[stimulus_sample_map["sample_id"].astype(str).tolist(), metabolite_names].copy()
    feature_frame.index = pd.Index(stimulus_sample_map["stimulus"].astype(str).tolist(), name="stimulus")
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    finite_feature_mask = np.isfinite(feature_frame).all(axis=0)
    feature_frame = feature_frame.loc[:, finite_feature_mask].copy()
    if feature_frame.shape[1] == 0:
        raise ValueError("selected metabolite set has no finite retained features")

    values = np.log2(feature_frame.to_numpy(dtype=float, copy=False))
    deltas = values[:, np.newaxis, :] - values[np.newaxis, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    np.fill_diagonal(distances, 0.0)

    frame = pd.DataFrame(distances, index=feature_frame.index, columns=feature_frame.index)
    frame.insert(0, "stimulus_row", frame.index.astype(str))
    return frame.reset_index(drop=True)


def prepare_display_frames(
    neural_matrix: pd.DataFrame,
    model_matrices: dict[str, pd.DataFrame],
    stimulus_sample_map: pd.DataFrame,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    neural_display, order_labels = prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)
    neural_display = mask_rdm_diagonal(neural_display)
    displays = {"neural": neural_display}
    for model_id, model_matrix in model_matrices.items():
        display, _ = prepare_rdm_heatmap_frame(
            model_matrix,
            stimulus_sample_map,
            order_labels=order_labels,
        )
        displays[model_id] = mask_rdm_diagonal(display)
    return order_labels, displays


def prepare_rdm_heatmap_frame(
    matrix_frame: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame | None,
    *,
    order_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    heatmap_frame = coerce_rdm_heatmap_frame(matrix_frame)
    if heatmap_frame.empty:
        return heatmap_frame, []

    if order_labels is None:
        ordered_labels = cluster_reorder_heatmap_labels(heatmap_frame)
    else:
        ordered_labels = [str(label) for label in order_labels]
        if set(ordered_labels) != set(heatmap_frame.index):
            ordered_labels = heatmap_frame.index.tolist()

    ordered_frame = heatmap_frame.reindex(index=ordered_labels, columns=ordered_labels)
    if stimulus_sample_map is None or stimulus_sample_map.empty:
        return ordered_frame, ordered_labels

    resolved_labels = resolve_display_labels(ordered_labels, stimulus_sample_map)
    if resolved_labels is None:
        return ordered_frame, ordered_labels

    resolved_frame = ordered_frame.copy()
    resolved_frame.index = pd.Index(resolved_labels)
    resolved_frame.columns = pd.Index(resolved_labels)
    return resolved_frame, ordered_labels


def coerce_rdm_heatmap_frame(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    if "stimulus_row" in matrix_frame.columns:
        heatmap_frame = matrix_frame.set_index("stimulus_row").copy()
    else:
        heatmap_frame = matrix_frame.copy()

    if heatmap_frame.empty:
        return heatmap_frame

    heatmap_frame.index = pd.Index(heatmap_frame.index.astype(str))
    heatmap_frame.columns = pd.Index(heatmap_frame.columns.astype(str))
    if set(heatmap_frame.index) != set(heatmap_frame.columns):
        raise ValueError("RDM heatmap requires matching row and column labels")
    heatmap_frame = heatmap_frame.reindex(columns=heatmap_frame.index)
    return heatmap_frame.apply(pd.to_numeric, errors="coerce")


def cluster_reorder_heatmap_labels(heatmap_frame: pd.DataFrame) -> list[str]:
    original_order = heatmap_frame.index.tolist()
    if len(original_order) < 3:
        return original_order

    numeric = heatmap_frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float, copy=True)
    if np.isnan(values).any() or not np.isfinite(values).all():
        return original_order

    np.fill_diagonal(values, 0.0)
    try:
        linkage_matrix = linkage(squareform(values, checks=False), method="average", optimal_ordering=True)
        order = leaves_list(linkage_matrix).tolist()
    except Exception:
        return original_order
    return [original_order[position] for position in order]


def resolve_display_labels(stimulus_order: list[str], stimulus_sample_map: pd.DataFrame) -> list[str] | None:
    if not stimulus_order or "stimulus" not in stimulus_sample_map.columns:
        return None

    map_frame = stimulus_sample_map.copy()
    map_frame["stimulus"] = map_frame["stimulus"].fillna("").astype(str).str.strip()
    map_frame = map_frame.loc[map_frame["stimulus"] != ""]
    subset = map_frame.loc[map_frame["stimulus"].isin(stimulus_order)]
    if subset.empty or len(subset["stimulus"].unique()) != len(stimulus_order):
        return None
    if subset["stimulus"].duplicated().any():
        return None

    stimulus_lookup = subset.set_index("stimulus", drop=False)
    for candidate_column in ("sample_id", "stim_name", "stimulus"):
        if candidate_column not in stimulus_lookup.columns:
            continue
        candidate_series = stimulus_lookup[candidate_column].reindex(stimulus_order)
        if candidate_series.isna().any():
            continue
        candidate_series = candidate_series.fillna("").astype(str).str.strip()
        if candidate_series.empty or (candidate_series == "").any():
            continue
        if candidate_series.duplicated().any():
            continue
        return candidate_series.tolist()
    return None


def mask_rdm_diagonal(frame: pd.DataFrame) -> pd.DataFrame:
    masked = coerce_rdm_heatmap_frame(frame).copy()
    diagonal_length = min(masked.shape)
    for index in range(diagonal_length):
        masked.iat[index, index] = np.nan
    return masked


__all__ = [
    "VIEW_NAMES",
    "build_chemical_rdm",
    "build_lr_merge_plan",
    "build_neural_rdms",
    "build_stimulus_mapping",
    "cluster_reorder_heatmap_labels",
    "coerce_rdm_heatmap_frame",
    "load_taxonomy_qc",
    "mask_rdm_diagonal",
    "merge_lr_view",
    "prepare_display_frames",
    "prepare_rdm_heatmap_frame",
    "resolve_display_labels",
]
