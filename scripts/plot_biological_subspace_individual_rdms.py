"""Render one neural-vs-chemical RDM comparison figure per selected biological subspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.rsa import compute_rsa_score
from bacteria_analysis.rsa_outputs import _create_rdm_panel_figure, _render_prepared_rdm_panels
from plot_biological_subspace_rdm_panel import (
    VIEW_NAMES,
    build_chemical_rdm,
    build_neural_rdms,
    build_stimulus_mapping,
    load_taxonomy_qc,
    prepare_display_frames,
)
from bacteria_analysis.model_space import read_metabolite_matrix

DEFAULT_SELECTED_MODELS: tuple[str, ...] = (
    "Class::Indoles and derivatives",
    "SubClass::Pyridinecarboxylic acids and derivatives",
    "SuperClass::Organic oxygen compounds",
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
        default="results/202604_without_20260331/biological_subspace_individual_rdms_neural_order",
        help="Directory to write figures and summary tables.",
    )
    parser.add_argument(
        "--qc-threshold",
        type=float,
        default=0.2,
        help="Maximum raw-workbook QCRSD retained for chemistry features.",
    )
    parser.add_argument(
        "--selected-model",
        action="append",
        dest="selected_models",
        default=None,
        help="Repeatable taxonomy selector in the form Level::Category.",
    )
    return parser.parse_args()


def parse_selected_models(raw_models: list[str] | None) -> list[tuple[str, str]]:
    selected = list(raw_models) if raw_models else list(DEFAULT_SELECTED_MODELS)
    parsed: list[tuple[str, str]] = []
    for value in selected:
        if "::" not in value:
            raise ValueError(f"selected model must look like Level::Category, got: {value!r}")
        taxonomy_level, category = value.split("::", 1)
        taxonomy_level = taxonomy_level.strip()
        category = category.strip()
        if not taxonomy_level or not category:
            raise ValueError(f"selected model must look like Level::Category, got: {value!r}")
        parsed.append((taxonomy_level, category))
    return parsed


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def render_single_figure(
    *,
    output_path: Path,
    displays_by_view: dict[str, dict[str, pd.DataFrame]],
    model_id: str,
    category: str,
    per_view_scores: dict[str, float],
    n_features: int,
) -> None:
    figure, axes, colorbar_axes = _create_rdm_panel_figure(nrows=len(VIEW_NAMES), figsize=(9.6, 8.8))
    panels: list[tuple[int, int, pd.DataFrame | None, str, str]] = []
    for row_index, view_name in enumerate(VIEW_NAMES):
        panels.append(
            (
                row_index,
                0,
                displays_by_view[view_name]["neural"],
                f"Neural RDM\n{view_name}",
                "Neural RDM unavailable",
            )
        )
        panels.append(
            (
                row_index,
                1,
                displays_by_view[view_name][model_id],
                f"Chemical RDM\n{view_name} | RSA={per_view_scores[view_name]:.3f}",
                "Chemical RDM unavailable",
            )
        )

    _render_prepared_rdm_panels(figure, axes, colorbar_axes, panels)
    figure.suptitle(
        f"{category} vs neural RDM\n"
        f"Neural order: non-ASE L/R merge + trial median + correlation distance | n={n_features}",
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
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected_models = parse_selected_models(args.selected_models)

    matrix = read_metabolite_matrix(matrix_path)
    stimulus_sample_map = build_stimulus_mapping(preprocess_root, matrix)
    neural_rdms, neural_support = build_neural_rdms(preprocess_root)

    taxonomy_qc = load_taxonomy_qc(raw_metadata_path)
    retained_taxonomy = taxonomy_qc.loc[
        taxonomy_qc["normalized_name"].astype(str).isin(matrix.columns.astype(str))
        & taxonomy_qc["QCRSD"].le(float(args.qc_threshold))
    ].copy()

    summary_rows: list[dict[str, object]] = []

    for taxonomy_level, category in selected_models:
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
        if not metabolites:
            raise ValueError(f"no retained metabolites found for {model_id}")

        chemical_rdm = build_chemical_rdm(matrix, stimulus_sample_map, metabolites)
        per_view_scores = {
            view_name: float(compute_rsa_score(neural_rdms[view_name], chemical_rdm)["rsa_similarity"])
            for view_name in VIEW_NAMES
        }

        displays_by_view: dict[str, dict[str, pd.DataFrame]] = {}
        for view_name in VIEW_NAMES:
            _, displays = prepare_display_frames(
                neural_matrix=neural_rdms[view_name],
                model_matrices={model_id: chemical_rdm},
                stimulus_sample_map=stimulus_sample_map,
            )
            displays_by_view[view_name] = displays

        figure_name = (
            f"rdm_comparison__{slugify(taxonomy_level)}__{slugify(category)}__neural_order.png"
        )
        render_single_figure(
            output_path=figures_dir / figure_name,
            displays_by_view=displays_by_view,
            model_id=model_id,
            category=category,
            per_view_scores=per_view_scores,
            n_features=len(metabolites),
        )

        summary_rows.append(
            {
                "model_id": model_id,
                "taxonomy_level": taxonomy_level,
                "category": category,
                "n_features": len(metabolites),
                "response_window_rsa": per_view_scores["response_window"],
                "full_trajectory_rsa": per_view_scores["full_trajectory"],
                "figure_path": str((figures_dir / figure_name).as_posix()),
                "metabolites": " | ".join(metabolites),
            }
        )

    summary = pd.DataFrame.from_records(summary_rows)
    summary.to_csv(output_root / "selected_subspace_summary.csv", index=False)
    neural_support.to_csv(output_root / "neural_support_summary.csv", index=False)

    selected_model_text = ", ".join(f"`{level}::{category}`" for level, category in selected_models)
    run_summary = "\n".join(
        [
            "# Biological Subspace Individual RDMs",
            "",
            f"- Preprocess root: `{preprocess_root}`",
            f"- Matrix path: `{matrix_path}`",
            f"- Raw metadata path: `{raw_metadata_path}`",
            f"- Neural reference: `non-ASE L/R merge + trial median + correlation distance`",
            f"- Chemical contract: `QCRSD <= {args.qc_threshold:.1f} + log2(matrix) + Euclidean`",
            "- Selection rule: `subspaces drawn from the previously strongest non-redundant pair/triple union combinations`",
            f"- Selected subspaces: {selected_model_text}",
            f"- Figures dir: `{figures_dir}`",
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
