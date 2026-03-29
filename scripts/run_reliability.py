"""CLI entry point for Stage 1 reliability analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.reliability import build_trial_views, load_reliability_inputs, run_reliability_pipeline
from bacteria_analysis.reliability_outputs import write_stage1_outputs
from bacteria_analysis.reliability_stats import (
    build_final_summary_table,
    build_permutation_null,
    build_grouped_bootstrap_from_scores,
    score_permutation_null,
    summarize_permutation_null,
)

PRIMARY_VIEW_CHOICES = (
    "full_trajectory",
    "on_window",
    "response_window",
    "post_window",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 reliability analysis from Stage 0 outputs.")
    parser.add_argument(
        "--input-root",
        help="Root directory containing Stage 0 clean/, trial_level/, and qc/ outputs.",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/trial_level/trial_metadata.parquet",
        help="Path to Stage 0 trial metadata parquet.",
    )
    parser.add_argument(
        "--wide",
        default="data/processed/trial_level/trial_wide_baseline_centered.parquet",
        help="Path to Stage 0 trial wide parquet.",
    )
    parser.add_argument(
        "--tensor",
        default="data/processed/trial_level/trial_tensor_baseline_centered.npz",
        help="Path to Stage 0 trial tensor npz.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Base directory for Stage 1 outputs.",
    )
    parser.add_argument("--permutations", type=int, default=100, help="Number of permutation iterations.")
    parser.add_argument("--bootstrap-iterations", type=int, default=500, help="Number of grouped bootstrap iterations.")
    parser.add_argument("--split-half-repeats", type=int, default=100, help="Number of split-half repeats.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split-half and inference.")
    parser.add_argument(
        "--primary-view",
        choices=PRIMARY_VIEW_CHOICES,
        default="response_window",
        help="Primary reporting view to record in Stage 1 summaries.",
    )
    return parser.parse_args(argv)


def _validate_input_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Stage 0 inputs: " + ", ".join(missing))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_root:
        input_root = Path(args.input_root)
        input_paths = {
            "metadata": input_root / "trial_level" / "trial_metadata.parquet",
            "wide": input_root / "trial_level" / "trial_wide_baseline_centered.parquet",
            "tensor": input_root / "trial_level" / "trial_tensor_baseline_centered.npz",
        }
    else:
        input_paths = {
            "metadata": Path(args.metadata),
            "wide": Path(args.wide),
            "tensor": Path(args.tensor),
        }

    try:
        _validate_input_paths(input_paths)
        inputs = load_reliability_inputs(
            metadata_path=input_paths["metadata"],
            wide_path=input_paths["wide"],
            tensor_path=input_paths["tensor"],
        )
        core_outputs = run_reliability_pipeline(
            inputs,
            split_half_repeats=args.split_half_repeats,
            seed=args.seed,
        )
        views = build_trial_views(inputs.metadata, inputs.tensor)
        permutation_samples = build_permutation_null(
            view=views["full_trajectory"],
            n_iterations=args.permutations,
            seed=args.seed,
        )
        permutation_iterations = score_permutation_null(core_outputs["comparisons"], permutation_samples)
        permutation_summary = summarize_permutation_null(
            observed_summary=core_outputs["same_vs_different_summary"],
            permutation_scores=permutation_iterations,
        )
        bootstrap_iterations, bootstrap_summary = build_grouped_bootstrap_from_scores(
            core_outputs["loio_trials"],
            n_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        final_summary = build_final_summary_table(
            same_vs_different_summary=core_outputs["same_vs_different_summary"],
            loio_summary=core_outputs["loio_summary"],
            lodo_summary=core_outputs["lodo_summary"],
            split_half_summary=core_outputs["split_half_summary"],
            permutation_summary=permutation_summary,
            bootstrap_summary=bootstrap_summary,
        )
        stats_outputs = {
            "permutation_iterations": permutation_iterations,
            "permutation_summary": permutation_summary,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_summary": bootstrap_summary,
            "final_summary": final_summary,
        }
        stage1_output_root = Path(args.output_root) / "stage1_reliability"
        written = write_stage1_outputs(
            core_outputs,
            stats_outputs,
            stage1_output_root,
            primary_view=args.primary_view,
        )
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke tests
        print(f"Stage 1 reliability failed: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {core_outputs['metadata_summary'].iloc[0]['n_trials']} trials")
    print(f"Primary reporting view: {args.primary_view}")
    print(f"Wrote final summary table to {written['final_summary']}")
    print(f"Wrote figures to {written['figures_dir']}")
    print(f"Wrote QC tables to {written['qc_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
