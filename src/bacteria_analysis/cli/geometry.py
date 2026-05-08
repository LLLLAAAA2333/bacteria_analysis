"""CLI entry point for geometry analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bacteria_analysis.geometry import parse_geometry_views, run_geometry_pipeline
from bacteria_analysis.geometry_outputs import write_geometry_outputs

DEFAULT_INPUT_ROOT = Path("data/processed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural geometry from trial-level outputs.")
    parser.add_argument("--input-root", help="Root directory containing trial-level outputs.")
    parser.add_argument("--output-root", default="results", help="Base directory for geometry outputs.")
    parser.add_argument("--views", default="response_window,full_trajectory", help="Comma-separated geometry views.")
    return parser.parse_args(argv)


def resolve_input_root(input_root: str | None, *, root_dir: Path | None = None) -> Path:
    if input_root:
        return Path(input_root)

    resolved_root = root_dir or Path.cwd()
    candidates = [resolved_root / DEFAULT_INPUT_ROOT]
    if resolved_root.parent.name == ".worktrees":
        candidates.append(resolved_root.parents[1] / DEFAULT_INPUT_ROOT)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        input_root = resolve_input_root(args.input_root)
        included_views = parse_geometry_views(args.views)
        core_outputs = run_geometry_pipeline(input_root, view_names=included_views)
        geometry_output_root = Path(args.output_root) / "geometry"
        written = write_geometry_outputs(core_outputs, geometry_output_root)
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke tests
        print(f"Geometry analysis failed: {exc}", file=sys.stderr)
        return 1

    print(f"Included views: {', '.join(included_views)}")
    print(f"Wrote tables to {written['tables_dir']}")
    print(f"Wrote figures to {written['figures_dir']}")
    print(f"Wrote run summary to {written['run_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
