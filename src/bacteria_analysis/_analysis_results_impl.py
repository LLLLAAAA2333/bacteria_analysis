"""Small result containers and explicit save helpers for analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

HASH_SIZE_LIMIT_BYTES = 50 * 1024 * 1024


@dataclass
class AnalysisResult:
    analysis_id: str
    parameters: dict[str, object]
    summary: dict[str, object] | pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    rdms: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Figure | Callable[[Path | None], Figure | None]] = field(default_factory=dict)
    audit: dict[str, object | pd.DataFrame] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    debug_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def save_analysis_result(
    result: AnalysisResult,
    output_root: str | Path,
    *,
    include_debug: bool = False,
    include_audit: bool = False,
) -> dict[str, Path]:
    """Write explicit final artifacts for an in-memory analysis result."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {"output_root": root}

    written["summary_json"] = _write_json(root / "summary.json", _summary_payload(result.summary))
    written["summary_md"] = _write_text(root / "summary.md", _summary_markdown(result))
    written["parameters_json"] = _write_json(root / "parameters.json", result.parameters)
    if result.diagnostics:
        written["diagnostics_json"] = _write_json(root / "diagnostics.json", result.diagnostics)

    written.update(_write_dataframes(result.tables, root / "tables", prefix="tables"))
    written.update(_write_dataframes(result.rdms, root / "rdms", prefix="rdms", include_index=True))
    written.update(_write_figures(result.figures, root / "figures"))
    if include_audit:
        written.update(_write_audit(result, root / "audit"))

    if include_debug and result.debug_tables:
        written.update(_write_dataframes(result.debug_tables, root / "debug", prefix="debug"))

    return written


def _write_dataframes(
    tables: dict[str, pd.DataFrame],
    directory: Path,
    *,
    prefix: str,
    include_index: bool = False,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    if not tables:
        return written
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = directory / f"{_safe_name(name)}.csv"
        table.to_csv(path, index=include_index)
        written[f"{prefix}.{name}"] = path
    return written


def _write_figures(figures: dict[str, Figure | Callable[[Path | None], Figure | None]], directory: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    if not figures:
        return written
    directory.mkdir(parents=True, exist_ok=True)
    for name, figure_or_writer in figures.items():
        safe_name = _safe_name(name)
        path = directory / (safe_name if safe_name.lower().endswith(".png") else f"{safe_name}.png")
        if isinstance(figure_or_writer, Figure):
            figure_or_writer.savefig(path, dpi=150, bbox_inches="tight")
        else:
            figure_or_writer(path)
        written[f"figures.{name}"] = path
    return written


def _write_audit(result: AnalysisResult, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, value in result.audit.items():
        safe_name = _safe_name(name)
        if isinstance(value, pd.DataFrame):
            path = directory / f"{safe_name}.csv"
            value.to_csv(path, index=False)
        else:
            path = directory / f"{safe_name}.json"
            _write_json(path, value)
        written[f"audit.{name}"] = path

    manifest_path = directory / "source_manifest.json"
    _write_json(manifest_path, _build_source_manifest(result))
    written["audit.source_manifest"] = manifest_path
    return written


def _build_source_manifest(result: AnalysisResult) -> dict[str, object]:
    flat_parameters = _flatten_mapping(result.parameters)
    return {
        "analysis_id": result.analysis_id,
        "parameters": _json_safe(result.parameters),
        "source_paths": [_source_path_entry(key, value) for key, value in _source_path_items(flat_parameters)],
        "seeds": _matching_parameters(flat_parameters, ("seed",)),
        "permutation_counts": _matching_parameters(flat_parameters, ("permutation", "permutations")),
        "resampling_counts": _matching_parameters(flat_parameters, ("resample", "resamples", "subset_count")),
        "git_commit": _git_commit(),
        "package_version": _package_version(),
    }


def _source_path_items(flat_parameters: dict[str, object]) -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for key, value in flat_parameters.items():
        key_lower = key.lower()
        if key_lower.endswith("path") or key_lower.endswith("paths") or key_lower.endswith("_root"):
            if isinstance(value, (list, tuple)):
                items.extend((f"{key}[{index}]", item) for index, item in enumerate(value))
            else:
                items.append((key, value))
    return items


def _source_path_entry(parameter: str, value: object) -> dict[str, object]:
    path = Path(str(value))
    entry: dict[str, object] = {
        "parameter": parameter,
        "path": str(path),
        "exists": path.exists(),
    }
    if path.is_file():
        size = path.stat().st_size
        entry["size_bytes"] = int(size)
        if size <= HASH_SIZE_LIMIT_BYTES:
            entry["sha256"] = _sha256(path)
    return entry


def _matching_parameters(flat_parameters: dict[str, object], tokens: tuple[str, ...]) -> dict[str, object]:
    matched = {
        key: value
        for key, value in flat_parameters.items()
        if any(token in key.lower() for token in tokens)
    }
    return _json_safe(matched)


def _flatten_mapping(mapping: dict[str, object], prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _summary_payload(summary: dict[str, object] | pd.DataFrame) -> object:
    if isinstance(summary, pd.DataFrame):
        return {
            "columns": summary.columns.astype(str).tolist(),
            "records": summary.to_dict(orient="records"),
        }
    return summary


def _summary_markdown(result: AnalysisResult) -> str:
    lines = [f"# {result.analysis_id}", ""]
    if isinstance(result.summary, pd.DataFrame):
        lines.append(_dataframe_markdown(result.summary))
    else:
        lines.extend(["| key | value |", "| --- | --- |"])
        for key, value in result.summary.items():
            lines.append(f"| {key} | {_markdown_value(value)} |")
    lines.append("")
    return "\n".join(lines)


def _dataframe_markdown(frame: pd.DataFrame) -> str:
    columns = frame.columns.astype(str).tolist()
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in frame.to_dict(orient="records"):
        rows.append("| " + " | ".join(_markdown_value(record[column]) for column in frame.columns) + " |")
    return "\n".join(rows)


def _markdown_value(value: object) -> str:
    safe = _json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, ensure_ascii=False, sort_keys=True)
    return str(safe)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_list()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _safe_name(name: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return safe.strip("._") or "item"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("bacteria-analysis")
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "AnalysisResult",
    "save_analysis_result",
]
