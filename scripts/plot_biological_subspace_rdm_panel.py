"""Plot neural-order RDM panels for a focused set of biology-driven metabolite subspaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.constants import NEURON_ORDER
from bacteria_analysis.model_space import build_stimulus_sample_map, read_metabolite_matrix
from bacteria_analysis.model_space_seed import RAW_METADATA_SHEET_NAME, _normalize_header_text
from bacteria_analysis.reliability import TrialView
from bacteria_analysis.rsa import compute_rsa_score
from bacteria_analysis.rsa_aggregated_responses import (
    build_aggregated_response_rdm,
    build_grouped_aggregated_responses,
    load_aggregated_response_context_inputs,
)
from bacteria_analysis.rsa_outputs import _coerce_rdm_heatmap_frame, _prepare_rdm_heatmap_frame

DEFAULT_SELECTED_MODELS: tuple[tuple[str, str], ...] = (
    ("Class", "Pyridines and derivatives"),
    ("SuperClass", "Organic oxygen compounds"),
    ("Class", "Purine nucleosides"),
    ("SubClass", "Indolyl carboxylic acids and derivatives"),
)
VIEW_NAMES: tuple[str, str] = ("response_window", "full_trajectory")
KEEP_SEPARATE_NEURONS = {"ASEL", "ASER"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocess-root",
        default="data/202604/202604_preprocess_without_20260331",
        help="Preprocess root that provides the neural trial tensor.",
    )
    parser.add_argument(
        "--matrix-path",
        default="data/matrix.xlsx",
        help="Matrix workbook used for metabolite distances.",
    )
    parser.add_argument(
        "--raw-metadata-path",
        default="data/metabolism_raw_data.xlsx",
        help="Raw workbook that provides taxonomy and QCRSD metadata.",
    )
    parser.add_argument(
        "--output-root",
        default="results/202604_without_20260331/biological_subspace_rdm_panel_neural_order",
        help="Directory to write the figure and summary tables.",
    )
    parser.add_argument(
        "--qc-threshold",
        type=float,
        default=0.2,
        help="Maximum raw-workbook QCRSD retained for chemistry features.",
    )
    return parser.parse_args()


def _build_lr_merge_plan() -> tuple[list[str], list[list[int]]]:
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


def merge_lr_view(view: TrialView) -> TrialView:
    _, merged_indices = _build_lr_merge_plan()
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
    neural_display, order_labels = _prepare_rdm_heatmap_frame(neural_matrix, stimulus_sample_map)
    neural_display = _mask_diagonal(neural_display)
    displays = {"neural": neural_display}
    for model_id, model_matrix in model_matrices.items():
        display, _ = _prepare_rdm_heatmap_frame(
            model_matrix,
            stimulus_sample_map,
            order_labels=order_labels,
        )
        displays[model_id] = _mask_diagonal(display)
    return order_labels, displays


def _mask_diagonal(frame: pd.DataFrame) -> pd.DataFrame:
    masked = _coerce_rdm_heatmap_frame(frame).copy()
    diagonal_length = min(masked.shape)
    for index in range(diagonal_length):
        masked.iat[index, index] = np.nan
    return masked


def _shared_norm(frames: list[pd.DataFrame]) -> matplotlib.colors.PowerNorm:
    finite_values: list[np.ndarray] = []
    for frame in frames:
        values = _coerce_rdm_heatmap_frame(frame).to_numpy(dtype=float, copy=False)
        if values.size == 0:
            continue
        finite_mask = np.isfinite(values)
        diagonal_length = min(values.shape)
        if diagonal_length:
            diagonal_indices = np.arange(diagonal_length)
            finite_mask[diagonal_indices, diagonal_indices] = False
        if np.any(finite_mask):
            finite_values.append(values[finite_mask])

    if not finite_values:
        return matplotlib.colors.PowerNorm(gamma=0.7, vmin=0.0, vmax=1.0, clip=True)

    concatenated = np.concatenate(finite_values)
    quantiles = np.quantile(concatenated, [0.05, 0.95])
    vmin = float(quantiles[0])
    vmax = float(quantiles[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(concatenated))
        vmax = float(np.max(concatenated))
    if vmin == vmax:
        padding = max(abs(vmin) * 0.05, 1e-6)
        vmin -= padding
        vmax += padding
    return matplotlib.colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax, clip=True)


def render_panel(
    output_path: Path,
    displays_by_view: dict[str, dict[str, pd.DataFrame]],
    panel_titles: list[tuple[str, str]],
) -> None:
    ncols = len(panel_titles)
    figure = plt.figure(figsize=(4.4 * ncols + 0.8, 9.0))
    grid = figure.add_gridspec(
        nrows=2,
        ncols=ncols + 1,
        width_ratios=[1.0] * ncols + [0.05],
        left=0.05,
        right=0.96,
        bottom=0.08,
        top=0.90,
        wspace=0.20,
        hspace=0.24,
    )

    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#f2f2f2")
    colorbar_axes = []

    for row_index, view_name in enumerate(VIEW_NAMES):
        row_displays = displays_by_view[view_name]
        row_norm = _shared_norm([row_displays[panel_id] for panel_id, _ in panel_titles])
        last_image = None
        for col_index, (panel_id, title) in enumerate(panel_titles):
            axis = figure.add_subplot(grid[row_index, col_index])
            display = row_displays[panel_id]
            values = _coerce_rdm_heatmap_frame(display).to_numpy(dtype=float, copy=False)
            last_image = axis.imshow(values, cmap=cmap, norm=row_norm)
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            if col_index == 0:
                axis.set_ylabel(view_name, fontsize=10)
            axis.set_xticks(np.arange(len(display.columns)))
            axis.set_yticks(np.arange(len(display.index)))
            axis.set_xticklabels(display.columns.tolist(), rotation=45, ha="right", fontsize=7)
            axis.set_yticklabels(display.index.tolist(), fontsize=7)

        colorbar_axis = figure.add_subplot(grid[row_index, -1])
        figure.colorbar(last_image, cax=colorbar_axis, label="RDM dissimilarity")
        colorbar_axes.append(colorbar_axis)

    figure.suptitle(
        "Biologically interpretable chemical subspaces vs neural RDM\n"
        "Neural order: L/R-merged trial-median correlation geometry | Chemical contract: QCRSD<=0.2 + log2 + Euclidean",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    preprocess_root = Path(args.preprocess_root)
    matrix_path = Path(args.matrix_path)
    raw_metadata_path = Path(args.raw_metadata_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    matrix = read_metabolite_matrix(matrix_path)
    stimulus_sample_map = build_stimulus_mapping(preprocess_root, matrix)
    neural_rdms, support = build_neural_rdms(preprocess_root)

    taxonomy_qc = load_taxonomy_qc(raw_metadata_path)
    retained_taxonomy = taxonomy_qc.loc[
        taxonomy_qc["normalized_name"].astype(str).isin(matrix.columns.astype(str))
        & taxonomy_qc["QCRSD"].le(float(args.qc_threshold))
    ].copy()

    model_rows: list[dict[str, object]] = []
    model_rdms_by_view: dict[str, dict[str, pd.DataFrame]] = {view_name: {} for view_name in VIEW_NAMES}
    panel_titles: list[tuple[str, str]] = [("neural", "Neural reference")]

    for taxonomy_level, category in DEFAULT_SELECTED_MODELS:
        model_id = f"{taxonomy_level}::{category}"
        metabolites = (
            retained_taxonomy.loc[
                retained_taxonomy[taxonomy_level].astype(str).str.strip().eq(category),
                "normalized_name",
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        chemical_rdm = build_chemical_rdm(matrix, stimulus_sample_map, metabolites)
        per_view_scores: dict[str, float] = {}
        for view_name in VIEW_NAMES:
            per_view_scores[view_name] = float(compute_rsa_score(neural_rdms[view_name], chemical_rdm)["rsa_similarity"])
            model_rdms_by_view[view_name][model_id] = chemical_rdm.copy()

        model_rows.append(
            {
                "model_id": model_id,
                "taxonomy_level": taxonomy_level,
                "category": category,
                "n_features": len(metabolites),
                "response_window_rsa": per_view_scores["response_window"],
                "full_trajectory_rsa": per_view_scores["full_trajectory"],
                "metabolites": " | ".join(metabolites),
            }
        )
        panel_titles.append(
            (
                model_id,
                f"{category}\n"
                f"n={len(metabolites)} | rw={per_view_scores['response_window']:.3f} | ft={per_view_scores['full_trajectory']:.3f}",
            )
        )

    displays_by_view: dict[str, dict[str, pd.DataFrame]] = {}
    for view_name in VIEW_NAMES:
        _, displays = prepare_display_frames(
            neural_matrix=neural_rdms[view_name],
            model_matrices=model_rdms_by_view[view_name],
            stimulus_sample_map=stimulus_sample_map,
        )
        displays_by_view[view_name] = displays

    figure_path = output_root / "biological_subspace_rdm_panel__neural_order.png"
    render_panel(
        output_path=figure_path,
        displays_by_view=displays_by_view,
        panel_titles=panel_titles,
    )

    summary = pd.DataFrame.from_records(model_rows)
    summary.to_csv(output_root / "selected_subspace_summary.csv", index=False)
    support.to_csv(output_root / "neural_support_summary.csv", index=False)

    run_summary = "\n".join(
        [
            "# Biological Subspace RDM Panel",
            "",
            f"- Preprocess root: `{preprocess_root}`",
            f"- Matrix path: `{matrix_path}`",
            f"- Raw metadata path: `{raw_metadata_path}`",
            f"- Neural reference: `non-ASE L/R merge + trial median + correlation distance`",
            f"- Chemical contract: `QCRSD <= {args.qc_threshold:.1f} + log2(matrix) + Euclidean`",
            "- Selected subspaces: `Pyridines and derivatives`, `Organic oxygen compounds`, `Purine nucleosides`, `Indolyl carboxylic acids and derivatives`",
            f"- Figure: `{figure_path}`",
            "",
            "## Selected Subspace Summary",
            "",
            "```text",
            summary.to_string(index=False),
            "```",
        ]
    )
    (output_root / "run_summary.md").write_text(run_summary, encoding="utf-8")


if __name__ == "__main__":
    main()
