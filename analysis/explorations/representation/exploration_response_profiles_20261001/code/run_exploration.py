"""Reproduce this independent exploration without running or editing notebooks."""
from pathlib import Path
import hashlib
import json

from response_representation import load_inputs, save_full_representation
from response_comparisons import run_comparisons
from response_plots import plot_exploration


def main():
    out = Path(__file__).resolve().parents[1]
    reports = out.parent
    print("Validating cached curves and original Figure 4 provenance", flush=True)
    data = load_inputs(reports)
    full = save_full_representation(data, out)
    print("Fitted full-data representations at four SNR cutoffs and no gate", flush=True)
    comparisons = run_comparisons(
        data, full["primary"], out, reports, repeats=100, seed=20261001,
        fits={f"snr_{label}": fit for label, fit in full["results"].items()
              if label != "unfiltered"},
        progress=lambda message: print(message, flush=True),
    )
    (out / "comparison_results.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    plots = plot_exploration(data, full["results"], out)
    hashes = {**data["metadata"]["input_sha256"],
              **data["metadata"]["original_notebook_sha256"]}
    unchanged = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
                 for name, value in hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("An input or original notebook changed during the exploration")
    verification = {
        "inputs_and_original_notebooks_unchanged": unchanged,
        "cache_bin_max_abs_error": data["metadata"]["cache_bin_max_abs_error"],
        "old_template_max_abs_error": data["metadata"]["old_template_max_abs_error"],
        "old_coefficient_max_abs_error": data["metadata"]["old_coefficient_max_abs_error"],
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((out / "code").glob("*.py"))},
        "plots": plots,
    }
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"output": str(out), "figures": plots["figures"],
                      "loao": comparisons["loao"],
                      "common_rdm_pairs": comparisons["rdm"]["n_common_all_representation_pairs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
