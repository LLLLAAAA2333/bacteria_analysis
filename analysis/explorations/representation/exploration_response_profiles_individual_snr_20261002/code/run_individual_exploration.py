"""Rebuild the individual-SNR exploration with a preselected lower cutoff."""
from pathlib import Path
import hashlib
import json

from individual_representation import load_inputs, save_full_representation
from individual_comparisons import run_comparisons, run_bootstrap_chord
from individual_plots import plot_exploration
from individual_hmds_full import run_full_hmds
from refresh_individual_figures import refresh_individual_figures

PRIMARY_THRESHOLD = .5


def main(output_dir=None):
    repo = Path(__file__).resolve().parents[3]
    out = Path(output_dir).resolve() if output_dir is not None else Path(__file__).resolve().parents[1]
    if any((out / name).exists() for name in ("hmds", "hmds3d", "hmds_full106")):
        raise FileExistsError("Choose a new output_dir; the checked HMDS is preserved")
    progress = lambda value: print(value, flush=True)
    progress("Loading reviewed individual mean curves")
    data = load_inputs(repo / "reports")
    full = save_full_representation(data, out, primary=PRIMARY_THRESHOLD)
    comparisons = run_comparisons(
        data, full["primary"], out, repo / "reports", repeats=100, seed=20261001,
        primary_threshold=PRIMARY_THRESHOLD,
        fits={f"snr_{label}": fit for label, fit in full["results"].items() if label != "unfiltered"},
        progress=progress)
    (out / "comparison_results.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    plots = plot_exploration(data, full["results"], out, PRIMARY_THRESHOLD)
    progress("Individual-SNR 0.5 figures saved; preparing new HMDS uncertainty")
    bootstrap = run_bootstrap_chord(data, full["primary"], out, draws=1000, seed=20261001,
                                    primary_threshold=PRIMARY_THRESHOLD, progress=progress)
    (out / "bootstrap_results.json").write_text(json.dumps(bootstrap, indent=2)+"\n")
    hmds = run_full_hmds(out, repo, progress=progress)
    if not all(hmds.get(d) and hmds[d]["neural"]["converged"] for d in ("2d", "3d")):
        raise RuntimeError("Full-sample HMDS did not pass both dimension checks; diagnostics are preserved")
    refresh_individual_figures(out)
    plots = json.loads((out / "figures/plot_parameters.json").read_text())
    hashes = {**data["metadata"]["input_sha256"], **data["metadata"]["original_notebook_sha256"]}
    unchanged = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("An input or original notebook changed during this analysis")
    verification = dict(inputs_and_original_notebooks_unchanged=unchanged,
                        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(Path(__file__).parent.glob("*.py"))},
                        primary_threshold=PRIMARY_THRESHOLD, plot_parameters=plots, hmds=hmds)
    (out / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    progress(json.dumps({"output": str(out), "loao": comparisons["loao"],
                         "hmds": {d: hmds[d]["neural"]["status"] for d in ("2d", "3d")}}, indent=2))


if __name__ == "__main__":
    main()
