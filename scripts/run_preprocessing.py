"""CLI entry point for preprocessing raw neuron segment parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.io import (
    ensure_output_dirs,
    read_parquet,
    write_json,
    write_markdown_report,
    write_parquet,
    write_tensor_npz,
)
from bacteria_analysis.preprocessing import run_preprocessing_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing on a neuron segment parquet file.")
    parser.add_argument("--input", required=True, help="Path to the raw input parquet file.")
    parser.add_argument("--output-root", required=True, help="Directory for preprocessing outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_root = Path(args.output_root)

    raw_df = read_parquet(input_path)
    outputs = run_preprocessing_pipeline(raw_df)
    paths = ensure_output_dirs(output_root)

    clean_path = write_parquet(outputs["clean_df"], paths["clean_dir"] / "neuron_segments_clean.parquet")
    metadata_path = write_parquet(outputs["metadata"], paths["trial_level_dir"] / "trial_metadata.parquet")
    wide_path = write_parquet(outputs["wide"], paths["trial_level_dir"] / "trial_wide_baseline_centered.parquet")
    tensor_path = write_tensor_npz(
        paths["trial_level_dir"] / "trial_tensor_baseline_centered.npz",
        outputs["tensor"],
        outputs["metadata"]["trial_id"].tolist(),
        outputs["metadata"]["stimulus"].tolist(),
        outputs["metadata"]["stim_name"].tolist(),
    )
    report_json_path = write_json(outputs["report"], paths["qc_dir"] / "preprocessing_report.json")
    report_md_path = write_markdown_report(outputs["report"], paths["qc_dir"] / "preprocessing_report.md")

    print(f"Loaded {len(raw_df)} rows from {input_path}")
    print(f"Wrote clean table to {clean_path}")
    print(f"Wrote trial metadata to {metadata_path}")
    print(f"Wrote wide table to {wide_path}")
    print(f"Wrote tensor to {tensor_path}")
    print(f"Wrote QC report to {report_json_path} and {report_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
