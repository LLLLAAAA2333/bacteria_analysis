"""Plot neural-order RDM panels for a focused set of biology-driven metabolite subspaces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.biological_subspace import (
    VIEW_NAMES,
    build_chemical_rdm,
    build_neural_rdms,
    build_stimulus_mapping,
    coerce_rdm_heatmap_frame,
    load_taxonomy_qc,
    prepare_display_frames,
)
from bacteria_analysis.model_space import read_metabolite_matrix
from bacteria_analysis.rsa import compute_rsa_score

DEFAULT_SELECTED_MODELS: tuple[tuple[str, str], ...] = (
    ("Class", "Pyridines and derivatives"),
    ("SuperClass", "Organic oxygen compounds"),
    ("Class", "Purine nucleosides"),
    ("SubClass", "Indolyl carboxylic acids and derivatives"),
)


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


def _shared_norm(frames: list[pd.DataFrame]) -> matplotlib.colors.PowerNorm:
    finite_values: list[np.ndarray] = []
    for frame in frames:
        values = coerce_rdm_heatmap_frame(frame).to_numpy(dtype=float, copy=False)
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
            values = coerce_rdm_heatmap_frame(display).to_numpy(dtype=float, copy=False)
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
