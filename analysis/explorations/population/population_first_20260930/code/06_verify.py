"""Bounded delivery checks; does not refit scientific models or alter inputs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import runpy
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
T, L = OUT / "tables", OUT / "logs"


def record(path, base):
    return dict(path=str(path.relative_to(base)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    checks = {}
    baseline = json.loads((L / "alignment_inputs.json").read_text())
    for item in baseline:
        assert record(ROOT / item["path"], ROOT) == item, item["path"]
    old = json.loads((ROOT / "reports/atlas_extension_20260930/logs/input_manifest.json").read_text())
    old = {v["path"]: v for v in old}
    for name in ["trial_curves.parquet", "taxonomy.csv",
                 "chemical_reference_groups.csv", "chemical_legacy_logfc.csv"]:
        path = ROOT / "reports/exploration_20260929/tables" / name
        item = record(path, ROOT)
        assert item == old[item["path"]], str(path)
        baseline.append(item)
    (L / "input_manifest.json").write_text(json.dumps(baseline, indent=2))
    checks["input_hashes_unchanged"] = len(baseline)

    scripts = [OUT / "run_analysis.py", *sorted((OUT / "code").glob("*.py"))]
    for path in scripts:
        compile(path.read_text(), str(path), "exec")
    checks["scripts_compile"] = len(scripts)
    animal = pd.read_parquet(T / "aligned_neural_animal_5bins.parquet")
    chemical = pd.read_csv(T / "aligned_chemical_log2fc_paired.csv", index_col=0)
    assert animal.shape == (607, 65) and chemical.shape == (106, 380)
    assert np.isfinite(chemical).all().all()
    assert set(animal.index.get_level_values("sample_id")) == set(chemical.index)
    checks["aligned_dimensions"] = dict(animal=list(animal.shape), chemical=list(chemical.shape))
    for name in ["population_structure", "neural_value"]:
        assert json.loads((L / f"{name}_verification.json").read_text())["status"] == "passed"

    # Reaggregate predictions, instead of trusting prewritten summary tables.
    pred = pd.read_csv(T / "population_chemistry_predictions.csv")
    assert np.isfinite(pred[["observed", "predicted", "weight", "train_cell_scale"]]).all().all()
    assert not pred.duplicated(["model", "sample_id", "date", "worm_key", "neuron_class", "bin"]).any()
    weights = pred.groupby(["model", "sample_id", "neuron_class", "bin"]).weight.sum()
    assert np.allclose(weights, 1, rtol=0, atol=1e-12)
    functions = runpy.run_path(str(OUT / "code/03_population_chemistry.py"))
    scores, means = functions["summarize"](pred)
    stored = pd.read_csv(T / "population_chemistry_scores.csv")
    error = float(np.max(np.abs(scores.r2 - stored.r2)))
    assert error < 1e-10
    deletion = functions["deletion_scores"](means)
    saved = pd.read_csv(T / "population_chemistry_deletion.csv")
    assert np.allclose(deletion.r2, saved.r2, rtol=0, atol=1e-10)
    risks = pd.read_csv(T / "population_chemistry_inner_risks.csv")
    chosen = pd.read_csv(T / "population_chemistry_choices.csv")
    for row in chosen.itertuples(index=False):
        d = risks[(risks.outer_block == row.outer_block) & (risks.model == row.model)]
        assert d.groupby("penalty").scaled_mse.mean().idxmin() == row.penalty
    checks["population_chemistry"] = dict(rows=len(pred), summary_max_error=error,
                                        deletion_renormalization_matches=True,
                                        all_45_inner_choices_match=True)

    # Verify three independent fold audits; dates remain audit metadata.
    for prefix, left, right in [
        ("population_chemistry", "train_strains", "test_strains"),
        ("shared_task", "train_ids", "test_ids"),
        ("latent_chemistry", "train_ids", "test_ids")]:
        audit = pd.read_csv(T / f"{prefix}_fold_audit.csv", dtype=str)
        assert len(audit) == 81
        for _, row in audit.iterrows():
            assert not set(row[left].split(";")) & set(row[right].split(";"))
            if prefix == "population_chemistry":
                assert not set(row.train_animals.split(";")) & set(row.test_animals.split(";"))
            elif prefix == "latent_chemistry":
                assert not set(row.train_blocks.split(";")) & set(row.test_blocks.split(";"))
            else:
                assert row.shared_strains == "0" and row.shared_animals == "0"
        checks[f"{prefix}_disjoint_folds"] = len(audit)

    shared = pd.read_csv(T / "shared_task_predictions.csv")
    summary = pd.read_csv(T / "shared_task_summary.csv").set_index("model")
    directory = None
    for model, d in shared.groupby("model"):
        keys = set(map(tuple, d[["sample_id", "date"]].to_numpy()))
        if directory is None:
            directory = keys
        assert keys == directory
        assert np.allclose(d.groupby("sample_id").strain_weight.sum(), 1)
        recalls = [np.average(g.correct, weights=g.strain_weight) for _, g in d.groupby("genus")]
        assert np.isclose(np.mean(recalls), summary.loc[model, "macro_recall"], atol=1e-12)
    checks["shared_task"] = dict(identical_test_directory=True, scores_recomputed=True)

    latent = pd.read_csv(T / "latent_chemistry_predictions.csv")
    stored = pd.read_csv(T / "latent_chemistry_summary.csv").set_index(["model", "rank"])
    assert np.allclose(latent.groupby(["model", "rank", "axis", "sample_id"]).weight.sum(), 1)
    for key, d in latent.groupby(["model", "rank"]):
        r2 = 1 - np.sum(d.weight * (d.observed - d.predicted)**2) / np.sum(d.weight * d.observed**2)
        assert np.isclose(r2, stored.loc[key, "r2"], atol=1e-12)
    checks["latent_chemistry_scores_recomputed"] = True

    reviews = ["independent_population_review.md", "independent_neural_value_review.md",
               "independent_review.md", "independent_latent_review.md"]
    for name in reviews:
        assert (L / name).stat().st_size > 100
    checks["independent_reviews"] = reviews
    pending = {L / "verification.json", L / "output_manifest.json"}
    nlinks = 0
    for path in OUT.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            linked = path.parent / target.split("#")[0]
            assert linked.exists() or linked in pending, f"{path.name}: {target}"
            nlinks += 1
    checks["document_links_resolve"] = nlinks
    for name in ["population_structure_held_responses", "neural_value_paired_population",
                 "shared_task_taxonomy"]:
        for extension in ["png", "pdf"]:
            assert (OUT / f"figures/{name}.{extension}").stat().st_size > 10000
    result = dict(status="passed", checks=checks, python=sys.version, platform=platform.platform(),
                  versions={name: importlib.metadata.version(name) for name in
                            ["numpy", "pandas", "scipy", "pyarrow", "openpyxl", "matplotlib", "scikit-learn"]},
                  execution="Analyses 00–05 executed stepwise. This check ran on final outputs; sequential wrapper not replayed.",
                  limitations="Numerical and code review are not independent biological validation.")
    (L / "verification.json").write_text(json.dumps(result, indent=2))
    manifest = [record(path, OUT) for path in sorted(OUT.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
                and path.suffix != ".log" and path.name not in
                ["output_manifest.json", "run_status.json", ".DS_Store"]]
    (L / "output_manifest.json").write_text(json.dumps(manifest, indent=2))
    assert all(record(OUT / item["path"], OUT) == item for item in manifest)
    print(json.dumps(dict(status="passed", checks=checks, manifested_outputs=len(manifest)), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (L / "verification.json").write_text(json.dumps(dict(status="failed", error=repr(error)), indent=2))
        raise
