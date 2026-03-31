"""CLI entry point for Stage 3 biochemical RSA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.model_space import resolve_model_inputs
from bacteria_analysis.rsa import load_stage2_pooled_neural_rdms, run_stage3_rsa
from bacteria_analysis.rsa_outputs import write_stage3_outputs

DEFAULT_STAGE2_ROOT = Path("results/stage2_geometry")
DEFAULT_MATRIX_PATH = Path("data/matrix.xlsx")
DEFAULT_MODEL_INPUT_ROOT = Path("data/model_space")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 pooled biochemical RSA from Stage 2 pooled RDM outputs.")
    parser.add_argument(
        "--stage2-root",
        default=str(DEFAULT_STAGE2_ROOT),
        help="Stage 2 output root containing pooled RDM parquet files.",
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH), help="Path to the metabolite matrix workbook.")
    parser.add_argument(
        "--model-input-root",
        default=str(DEFAULT_MODEL_INPUT_ROOT),
        help="Directory containing Stage 3 model input CSVs.",
    )
    parser.add_argument("--output-root", default="results", help="Base directory for Stage 3 outputs.")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of stimulus-label permutations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutation and robustness summaries.")
    return parser.parse_args(argv)


def resolve_stage2_root(stage2_root: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    return _resolve_repo_path(stage2_root, default_relative=DEFAULT_STAGE2_ROOT, root_dir=root_dir)


def resolve_matrix_path(matrix_path: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    return _resolve_repo_path(matrix_path, default_relative=DEFAULT_MATRIX_PATH, root_dir=root_dir)


def resolve_model_input_root(model_input_root: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    return _resolve_repo_path(model_input_root, default_relative=DEFAULT_MODEL_INPUT_ROOT, root_dir=root_dir)


def _resolve_repo_path(explicit_path: str | None, *, default_relative: Path, root_dir: Path) -> Path:
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.is_absolute() or explicit != default_relative:
            return explicit

    candidates = [root_dir / default_relative]
    if root_dir.parent.name == ".worktrees":
        candidates.append(root_dir.parents[1] / default_relative)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _included_primary_models(model_registry: pd.DataFrame) -> list[str]:
    excluded_mask = model_registry.get("excluded_from_primary_ranking", False)
    if not isinstance(excluded_mask, pd.Series):
        excluded_mask = pd.Series(bool(excluded_mask), index=model_registry.index, dtype=bool)
    else:
        excluded_mask = excluded_mask.fillna(False).astype(bool)

    primary_mask = model_registry["model_tier"].astype(str).str.strip().str.lower().eq("primary") & ~excluded_mask
    return model_registry.loc[primary_mask, "model_id"].astype(str).tolist()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        stage2_root = resolve_stage2_root(args.stage2_root)
        matrix_path = resolve_matrix_path(args.matrix)
        model_input_root = resolve_model_input_root(args.model_input_root)
        neural_matrices = load_stage2_pooled_neural_rdms(stage2_root)
        resolved_inputs = resolve_model_inputs(model_input_root, matrix_path)
        core_outputs = run_stage3_rsa(
            resolved_inputs,
            neural_matrices=neural_matrices,
            permutations=args.permutations,
            seed=args.seed,
        )
        stage3_output_root = Path(args.output_root) / "stage3_rsa"
        written = write_stage3_outputs(core_outputs, stage3_output_root)
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke tests
        print(f"Stage 3 RSA failed: {exc}", file=sys.stderr)
        return 1

    primary_models = _included_primary_models(core_outputs["model_registry_resolved"])
    print(f"Included primary models: {', '.join(primary_models) if primary_models else 'None'}")
    print(f"Wrote tables to {written['tables_dir']}")
    print(f"Wrote figures to {written['figures_dir']}")
    print(f"Wrote run summary to {written['run_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
