"""CLI entry point for Stage 2 geometry analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.geometry import parse_stage2_views, run_geometry_pipeline
from bacteria_analysis.geometry_outputs import write_stage2_outputs

DEFAULT_INPUT_ROOT = Path("data/processed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 neural geometry from Stage 0 trial outputs.")
    parser.add_argument("--input-root", help="Root directory containing Stage 0 trial_level outputs.")
    parser.add_argument("--output-root", default="results", help="Base directory for Stage 2 outputs.")
    parser.add_argument("--views", default="response_window,full_trajectory", help="Comma-separated Stage 2 views.")
    return parser.parse_args(argv)


def resolve_input_root(input_root: str | None, *, root_dir: Path = ROOT_DIR) -> Path:
    if input_root:
        return Path(input_root)

    candidates = [root_dir / DEFAULT_INPUT_ROOT]
    if root_dir.parent.name == ".worktrees":
        candidates.append(root_dir.parents[1] / DEFAULT_INPUT_ROOT)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        input_root = resolve_input_root(args.input_root)
        included_views = parse_stage2_views(args.views)
        core_outputs = run_geometry_pipeline(input_root, view_names=included_views)
        stage2_output_root = Path(args.output_root) / "stage2_geometry"
        written = write_stage2_outputs(core_outputs, stage2_output_root)
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke tests
        print(f"Stage 2 geometry failed: {exc}", file=sys.stderr)
        return 1

    print(f"Included views: {', '.join(included_views)}")
    print(f"Wrote tables to {written['tables_dir']}")
    print(f"Wrote figures to {written['figures_dir']}")
    print(f"Wrote run summary to {written['run_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
