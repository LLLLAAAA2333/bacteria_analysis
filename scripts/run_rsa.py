"""CLI entry point for biochemical RSA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.model_space import resolve_direct_global_profile_inputs, resolve_model_inputs
from bacteria_analysis.rsa import load_geometry_pooled_neural_rdms, run_biochemical_rsa
from bacteria_analysis.rsa_outputs import write_rsa_outputs
from bacteria_analysis.rsa_aggregated_responses import load_aggregated_response_context_inputs

DEFAULT_GEOMETRY_ROOT = Path("results/geometry")
LEGACY_STAGE2_ROOT = Path("results/stage2_geometry")
DEFAULT_MATRIX_PATH = Path("data/matrix.xlsx")
DEFAULT_MODEL_INPUT_ROOT = Path("data/model_space")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run biochemical RSA directly from preprocess outputs and matrix.xlsx by default, "
            "with curated model-space and geometry fallback modes available for compatibility."
        )
    )
    parser.add_argument(
        "--geometry-root",
        dest="geometry_root",
        default=str(DEFAULT_GEOMETRY_ROOT),
        help="Geometry output root containing pooled RDM parquet files for legacy curated fallback runs.",
    )
    parser.add_argument("--stage2-root", dest="geometry_root", help=argparse.SUPPRESS)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH), help="Path to the metabolite matrix workbook.")
    parser.add_argument(
        "--model-input-root",
        default=None,
        help="Optional curated model-space directory; omit this to run the default direct RSA path.",
    )
    parser.add_argument(
        "--preprocess-root",
        default=None,
        help="Preprocessing output root for the primary direct RSA workflow and preprocess-backed curated runs.",
    )
    parser.add_argument(
        "--response-aggregation",
        choices=("mean", "median"),
        default="mean",
        help="Aggregation used to build grouped neural responses when --preprocess-root is provided.",
    )
    parser.add_argument("--prototype-aggregation", dest="response_aggregation", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="results", help="Base directory for RSA outputs.")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of stimulus-label permutations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutation and robustness summaries.")
    return parser.parse_args(argv)


def resolve_geometry_root(geometry_root: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    if geometry_root:
        explicit = Path(geometry_root)
        if explicit.is_absolute() or explicit not in {DEFAULT_GEOMETRY_ROOT, LEGACY_STAGE2_ROOT}:
            return explicit

    candidates = _resolve_repo_candidates((DEFAULT_GEOMETRY_ROOT, LEGACY_STAGE2_ROOT), root_dir=root_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_matrix_path(matrix_path: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    return _resolve_repo_path(matrix_path, default_relative=DEFAULT_MATRIX_PATH, root_dir=root_dir)


def resolve_model_input_root(model_input_root: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    return _resolve_repo_path(model_input_root, default_relative=DEFAULT_MODEL_INPUT_ROOT, root_dir=root_dir)


def resolve_preprocess_root(preprocess_root: str | None, *, root_dir: Path = ROOT_DIR) -> Path | None:
    if preprocess_root is None:
        return None
    return _resolve_repo_path(preprocess_root, default_relative=Path(preprocess_root), root_dir=root_dir)


def _resolve_repo_path(explicit_path: str | None, *, default_relative: Path, root_dir: Path) -> Path:
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.is_absolute() or explicit != default_relative:
            return explicit

    candidates = _resolve_repo_candidates((default_relative,), root_dir=root_dir)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_repo_candidates(default_relatives: tuple[Path, ...], *, root_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for default_relative in default_relatives:
        candidates.append(root_dir / default_relative)
        if root_dir.parent.name == ".worktrees":
            candidates.append(root_dir.parents[1] / default_relative)
    return candidates


def _included_ranked_models(model_registry: pd.DataFrame) -> list[str]:
    excluded_mask = model_registry.get("excluded_from_primary_ranking", False)
    if not isinstance(excluded_mask, pd.Series):
        excluded_mask = pd.Series(bool(excluded_mask), index=model_registry.index, dtype=bool)
    else:
        excluded_mask = excluded_mask.fillna(False).astype(bool)

    primary_mask = model_registry["model_tier"].astype(str).str.strip().str.lower().eq("primary") & ~excluded_mask
    return model_registry.loc[primary_mask, "model_id"].astype(str).tolist()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    raw_model_input_root = args.model_input_root
    matrix_path = resolve_matrix_path(args.matrix)
    preprocess_root = resolve_preprocess_root(args.preprocess_root)
    run_kwargs: dict[str, object] = {
        "permutations": args.permutations,
        "seed": args.seed,
    }

    if preprocess_root is not None and raw_model_input_root is None:
        resolved_inputs = resolve_direct_global_profile_inputs(
            preprocess_root=preprocess_root,
            matrix_path=matrix_path,
        )
        run_kwargs["aggregated_response_inputs"] = load_aggregated_response_context_inputs(
            preprocess_root,
            view_names=("response_window", "full_trajectory"),
        )
        run_kwargs["response_aggregation"] = args.response_aggregation
    elif preprocess_root is not None and raw_model_input_root is not None:
        model_input_root = resolve_model_input_root(raw_model_input_root)
        resolved_inputs = resolve_model_inputs(model_input_root, matrix_path)
        run_kwargs["aggregated_response_inputs"] = load_aggregated_response_context_inputs(
            preprocess_root,
            view_names=("response_window", "full_trajectory"),
        )
        run_kwargs["response_aggregation"] = args.response_aggregation
    elif preprocess_root is None and raw_model_input_root is not None:
        model_input_root = resolve_model_input_root(raw_model_input_root)
        resolved_inputs = resolve_model_inputs(model_input_root, matrix_path)
        geometry_root = resolve_geometry_root(args.geometry_root)
        run_kwargs["neural_matrices"] = load_geometry_pooled_neural_rdms(geometry_root)
    else:
        raise ValueError("Direct RSA requires --preprocess-root when --model-input-root is omitted")

    core_outputs = run_biochemical_rsa(resolved_inputs, **run_kwargs)
    rsa_output_root = Path(args.output_root) / "rsa"
    written = write_rsa_outputs(core_outputs, rsa_output_root)

    ranked_models = _included_ranked_models(core_outputs["model_registry_resolved"])
    print(f"Included ranked models: {', '.join(ranked_models) if ranked_models else 'None'}")
    print(f"Wrote tables to {written['tables_dir']}")
    print(f"Wrote figures to {written['figures_dir']}")
    print(f"Wrote run summary to {written['run_summary_json']}")
    return 0


DEFAULT_STAGE2_ROOT = LEGACY_STAGE2_ROOT
resolve_stage2_root = resolve_geometry_root


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke tests
        print(f"Biochemical RSA failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
