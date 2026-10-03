"""Index retained exploratory code and its current poster reading entry."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
ROUTES = [
    ("reports/exploration_response_profiles_individual_snr_20261002/", "00;01;02;03", "preparation.py;atlas.py;repeatability.py", "Preparation defaults/table schemas and parity references; old chemical/HMDS uses legacy chemistry"),
    ("reports/exploration_response_profiles_20261001/code/response_representation.py", "00", "preparation.py", "Pure SNR, weighted SVD and aggregation extracted; no runtime legacy import"),
    ("reports/exploration_chemical_pattern_direct_report_20261003/", "03;04", "global_comparison.py;local_states.py", "Current fresh162 input; full prediction experiments retained at source"),
    ("reports/exploration_genus_within_between_20261003/", "03", "global_comparison.py", "Fresh162 distance background and balanced genus summaries"),
    ("reports/exploration_genus_patterns_independent_20261003/", "03", "", "Independent chemical and neural genus patterns displayed as background"),
    ("reports/poster_local_chemical_neural_20261003/", "04", "local_states.py", "Accepted figure_data plus cached validation; tables also include failed Bact extension"),
    ("reports/exploration_bacteroides_adf_ash_chemical_20261003/", "04", "local_states.py", "Previous fixed-contrast selection and validation"),
    ("reports/exploration_bacteroides_local_model_20261003/chemical/", "04", "chemical_axes.py", "Pure chemical training/transform algorithm extracted"),
    ("reports/exploration_bacteroides_local_model_20261003/", "04", "", "Historical local PC1 exploration; not the accepted main figure"),
    ("reports/exploration_bacteroides_neural_reliability_20261003/", "04", "", "Historical neural reliability and subspace checks"),
    ("reports/response_structure", "01", "atlas.py", "Original response-model exploration retained; not refitted by poster"),
    ("notebook/response_structure", "01", "atlas.py", "Original fitting/display pipeline retained; saved arrays used by poster"),
    ("notebook/sample_comparison_poster", "03", "", "Historical seven-neuron pair illustration only"),
    ("reports/sample_comparison", "03", "", "Historical pair illustration and backups"),
    ("reports/exploration_response_profiles_", "01;02", "", "Earlier SNR definitions/representations retained as history"),
]


def build_index():
    rows = []
    with (ROOT / "analysis/explorations/source_manifest.csv").open(newline="") as stream:
        historical_paths = {row["current_path"]: row["original_path"] for row in csv.DictReader(stream)}
    for folder in ("notebooks", "src/bacteria_analysis", "analysis/explorations"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix not in {".py", ".ipynb"} or ".ipynb_checkpoints" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            original = historical_paths.get(relative, relative.replace("src/bacteria_analysis/", "notebook/").replace("notebooks/", "notebook/"))
            route = next((entry for entry in ROUTES if original.startswith(entry[0])), None)
            rows.append({
                "retained_source": relative,
                "original_source": original,
                "figure_group": route[1] if route else "historical",
                "current_modules_under_poster_analysis": route[2] if route else "",
                "relationship": route[3] if route else "Earlier exploration; retained at source, not executed by poster notebooks",
                "maintenance": ("Historical snapshot; port required before execution" if relative.startswith("analysis/explorations/")
                                else "Maintained analysis source; poster-specific changes belong in poster/"),
            })
    target = ROOT / "poster/docs/script_inventory.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(f"Indexed {build_index()} retained source files")
