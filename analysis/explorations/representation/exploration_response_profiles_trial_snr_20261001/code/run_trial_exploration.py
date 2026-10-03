"""Run the trial-SNR revision in its own directory; never execute notebooks."""
from pathlib import Path
import hashlib
import json

from trial_representation import load_inputs, save_full_representation
from trial_comparisons import run_comparisons, run_bootstrap_chord
from trial_plots import plot_exploration, plot_hmds
from trial_hmds import run_hmds


def main(output_dir=None):
    repo = Path(__file__).resolve().parents[3]
    reports = repo / "reports"
    out = Path(output_dir).resolve() if output_dir is not None else Path(__file__).resolve().parents[1]
    if (out / "hmds").exists():
        raise FileExistsError("Choose a new output_dir for a full rerun; the checked HMDS is preserved")
    progress = lambda value: print(value, flush=True)
    progress("Checking trial means, cache provenance and original Figure 4 compatibility")
    data = load_inputs(reports)
    full = save_full_representation(data, out)
    progress("Trial-SNR representations fitted; comparing independent animal halves")
    comparisons = run_comparisons(data, full["primary"], out, reports, repeats=100, seed=20261001,
                                  fits={f"snr_{label}": fit for label, fit in full["results"].items()
                                        if label != "unfiltered"}, progress=progress)
    (out / "comparison_results.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    plots = plot_exploration(data, full["results"], out)
    progress("Profile, explicit filtering heatmap, density histograms and RDM figures saved")
    bootstrap = run_bootstrap_chord(data, full["primary"], out, draws=1000, seed=20261001, progress=progress)
    (out / "bootstrap_results.json").write_text(json.dumps(bootstrap, indent=2)+"\n")
    progress("Fitting neural HMDS with the original notebook method")
    hmds = run_hmds(out, repo, progress=progress)
    if (out / "hmds/neural_coordinates.csv").exists():
        plot_hmds(out)
    hashes = {**data["metadata"]["input_sha256"], **data["metadata"]["original_notebook_sha256"]}
    unchanged = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("Input or original notebook changed during the exploration")
    verification = dict(inputs_and_original_notebooks_unchanged=unchanged,
                        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(Path(__file__).parent.glob("*.py"))},
                        trial_mean_verification=data["metadata"], plot_parameters=plots,
                        hmds=hmds)
    (out / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    progress(json.dumps({"output": str(out), "loao": comparisons["loao"], "hmds": hmds}, indent=2))


if __name__ == "__main__":
    main()
