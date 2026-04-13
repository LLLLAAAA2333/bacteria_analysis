"""CLI entry point for Stage 3 model-space auto-seeding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.model_space_seed import build_model_space

DEFAULT_MATRIX_PATH = Path("data/matrix.xlsx")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provisional Stage 3 model-space CSV inputs.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument("--preprocess-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--identity-evidence-cache", default=None)
    parser.add_argument("--taxonomy-enrichment-cache", default=None)
    parser.add_argument("--cache-version", default="manual-cache-v1")
    parser.add_argument("--refresh-pubchem-cache", action="store_true")
    return parser.parse_args(argv)


def resolve_repo_path(path_value: str | None, *, default_relative: Path | None = None) -> Path:
    if path_value:
        explicit = Path(path_value)
        if explicit.is_absolute():
            return explicit
        candidates = [ROOT_DIR / explicit]
    elif default_relative is not None:
        candidates = [ROOT_DIR / default_relative]
    else:
        raise ValueError("path_value or default_relative must be provided")

    if ROOT_DIR.parent.name == ".worktrees":
        extra_candidates: list[Path] = []
        for candidate in candidates:
            relative_candidate = candidate.relative_to(ROOT_DIR)
            extra_candidates.append(ROOT_DIR.parents[1] / relative_candidate)
        candidates.extend(extra_candidates)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        matrix_path = resolve_repo_path(args.matrix, default_relative=DEFAULT_MATRIX_PATH)
        preprocess_root = resolve_repo_path(args.preprocess_root)
        registry_path = resolve_repo_path(args.registry)
        output_root = resolve_repo_path(args.output_root)
        identity_evidence_path = None
        if args.identity_evidence_cache:
            identity_evidence_path = resolve_repo_path(args.identity_evidence_cache)
        taxonomy_enrichment_path = None
        if args.taxonomy_enrichment_cache:
            taxonomy_enrichment_path = resolve_repo_path(args.taxonomy_enrichment_cache)
        written = build_model_space(
            matrix_path=matrix_path,
            preprocess_root=preprocess_root,
            registry_path=registry_path,
            output_root=output_root,
            identity_evidence_path=identity_evidence_path,
            taxonomy_enrichment_path=taxonomy_enrichment_path,
            cache_version=args.cache_version,
            refresh_pubchem_cache=args.refresh_pubchem_cache,
        )
    except Exception as exc:  # pragma: no cover - exercised in smoke tests
        print(f"Stage 3 model-space build failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote provisional model-space inputs to {written['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
