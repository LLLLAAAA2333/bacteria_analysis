"""Execute in fresh kernels and save local output copies; preparation is opt-in."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "00": "00_preparation.ipynb",
    "01": "01_response_atlas.ipynb",
    "02": "02_repeatability.ipynb",
    "03": "03_global_chemical_neural.ipynb",
    "04": "04_local_chemical_states.ipynb",
}


def execute(selected=None):
    """Save successful runs under reports/ without writing outputs into source notebooks."""
    records = []
    for number in selected or ("01", "02", "03", "04"):
        path = ROOT / "poster/notebooks" / NOTEBOOKS[number]
        print(f"Running {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        manager = KernelManager(kernel_name="python3")
        # Use the Python that launched this script, not a host-specific kernel argv.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        started = time.monotonic()
        client = NotebookClient(notebook, km=manager, timeout=240,
                                resources={"metadata": {"path": str(path.parent)}})
        client.execute(cleanup_kc=True)
        nbformat.validate(notebook)
        output_path = ROOT / "reports/notebooks/poster" / path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_path)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        records.append({
            "notebook": str(path.relative_to(ROOT)),
            "executed_copy": str(output_path.relative_to(ROOT)),
            "code_cells": len(code_cells),
            "execution_counts": [cell.execution_count for cell in code_cells],
            "seconds": round(time.monotonic() - started, 2),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "executed_copy_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "status": "passed",
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"Passed {path.name}: {len(code_cells)} code cells", flush=True)
    record_path = ROOT / "reports/poster/notebook_execution.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    merged = {}
    for record in previous.get("notebooks", []):
        record.setdefault("executed_at_utc", previous.get("executed_at_utc"))
        merged[record["notebook"]] = record
    merged.update({record["notebook"]: record for record in records})
    record_path.write_text(json.dumps({
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "last_run_notebooks": [record["notebook"] for record in records],
        "notebooks": [merged[name] for name in sorted(merged)],
    }, ensure_ascii=False, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", nargs="*", help="Notebook numbers: 00 01 02 03 04; defaults to 01–04")
    args = parser.parse_args()
    if any(number not in NOTEBOOKS for number in args.numbers):
        parser.error("Choose notebook numbers from 00, 01, 02, 03, 04")
    execute(list(dict.fromkeys(args.numbers)) or None)
