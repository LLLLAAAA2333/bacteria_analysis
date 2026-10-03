"""Reproducible checks and manifests for this bounded research round.

Reads original inputs and previous outputs; writes only this round's logs.
These are computational checks, not independent biological validation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
T, L = OUT / "tables", OUT / "logs"
SEED = 2026093003


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(paths, base):
    return [dict(path=str(p.relative_to(base)), bytes=p.stat().st_size,
                 sha256=digest(p)) for p in sorted(set(paths))]


def read(name, **kwargs):
    return pd.read_csv(T / name, **kwargs)


def main():
    checks = {}
    prior = ROOT / "reports/population_first_20260930"
    original = json.loads((prior / "logs/input_manifest.json").read_text())
    for row in original:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    checks["unchanged_original_inputs"] = len(original)
    for directory in [prior, ROOT / "reports/chemical_neighborhood_focus_20260930"]:
        rows = json.loads((directory / "logs/output_manifest.json").read_text())
        for row in rows:
            assert digest(directory / row["path"]) == row["sha256"], row["path"]
        checks[f"unchanged_{directory.name}_outputs"] = len(rows)

    source = prior / "tables"
    actual_inputs = [source / f for f in [
        "aligned_neural_animal_5bins.parquet", "neural_value_pairs.csv",
        "population_structure_loadings.csv", "aligned_chemical_log2fc_paired.csv",
        "aligned_chemical_report_observed_paired.parquet", "aligned_chemical_metadata.csv"]]
    trial_path = ROOT / "reports/exploration_20260929/tables/trial_curves.parquet"
    actual_inputs.append(trial_path)
    (L / "input_manifest.json").write_text(json.dumps(dict(
        direct_inputs=manifest(actual_inputs, ROOT),
        provenance_inputs=original), indent=2) + "\n")

    for name in ["pair_signal_verification.json", "pair_context_verification.json",
                 "neuron_patterns.json", "figure_verification.json"]:
        result = json.loads((L / name).read_text())
        assert result["status"] in ["success", "passed"], name
        for row in result.get("inputs", []):
            assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    checks["successful_analysis_logs"] = 4

    pairs = read("pair_signal_summary.csv", dtype={"date": str})
    pc = read("pair_signal_percell.csv")
    rows = read("poster_neighborhood_rows.csv", dtype={"date": str})
    selected = pairs[pairs.nearest_either.eq(1) & pairs.group_n_strains.ge(3)]
    assert len(pairs) == 147 and len(rows) == 28
    assert set(rows.pair_id) == set(selected.pair_id)
    assert rows.energy_fixed.is_monotonic_decreasing
    assert rows.n_cells.value_counts().to_dict() == {13: 26, 10: 2}
    assert len(set(rows.strain_a) | set(rows.strain_b)) == 40
    assert int(rows.loo_fixed_min.gt(0).sum()) == 19
    assert rows.chemical_rms_log2fc.between(1.65, 3.35).all()
    assert rows.loo_fixed_min.ge(-.3).all() and rows.loo_fixed_max.le(3.2).all()
    assert pairs.chemical_rms_log2fc.between(1.62, 3.69).all()
    checks.update(n_pairs=147, primary_pairs=28, primary_complete13=26,
                  primary_10cell=2, primary_strains=40,
                  positive_after_every_animal_deletion=19,
                  plotted_data_within_axis_limits=True)

    differences = read("pair_signal_animal_differences.csv", dtype={"date": str})
    subset = differences[differences.pair_id.isin(rows.pair_id)]
    checks["primary_animals"] = len(subset[["date", "worm_key"]].drop_duplicates())
    assert checks["primary_animals"] == 38
    # Independently express each score using explicit DISTINCT-animal dot
    # products, instead of the sum-of-squares formula used in 01.
    lookup = pc.set_index(["pair_id", "neuron"])
    errors = []
    for key, group in differences.groupby(["pair_id", "neuron"]):
        p = lookup.loc[key]
        if not p.eligible:
            continue
        d = group.pivot(index="worm_key", columns="bin", values="difference_fixed").dropna().to_numpy()
        assert d.shape == (p.n_animals, 5)
        terms = [np.dot(d[i], d[j]) / 5 for i in range(len(d)) for j in range(len(d)) if i != j]
        errors.append(abs(np.mean(terms) - p.energy_fixed))
    checks["cell_energy_explicit_product_max_abs_error"] = float(max(errors))
    assert max(errors) < 1e-10
    for mode in ["fixed", "raw"]:
        summed = pc.groupby("pair_id")[f"contribution_{mode}"].sum()
        expected = pairs.set_index("pair_id")[f"energy_{mode}"].reindex(summed.index)
        assert np.allclose(summed, expected, atol=1e-12, rtol=0)
    plotted = read("poster_cell_contributions.csv", index_col=0)
    assert plotted.shape == (28, 13) and plotted.isna().sum().sum() == 6
    assert np.allclose(plotted.sum(axis=1), rows.set_index("pair_id").loc[plotted.index].energy_fixed)
    assert (pc.energy_fixed < 0).any() and (plotted < 0).any().any()
    deletion = read("pair_signal_animal_deletion.csv")
    for mode in ["fixed", "raw", "excluded"]:
        d = deletion.groupby("pair_id")[f"energy_{mode}"].agg(["min", "max"])
        base = pairs.set_index("pair_id").loc[d.index]
        for direction in ["min", "max"]:
            assert np.allclose(d[direction], base[f"loo_{mode}_{direction}"], atol=1e-12, rtol=0)
    checks["contributions_negative_missing_and_deletion_ranges"] = "passed"

    transfer = read("neuron_patterns_transfer_cells.csv", dtype={"date": str})
    folds = read("neuron_patterns_transfer_folds.csv", dtype={"date": str})
    keys = ["pair_id", "date", "held_worm", "train_mode", "test_mode"]
    assert not folds.duplicated(keys).any()
    assert transfer.groupby(keys).is_selected.sum().eq(1).all()
    for r in transfer.itertuples():
        assert r.held_worm not in r.train_animals.split("|")
        assert r.n_train_animals == len(r.train_animals.split("|"))
    chosen = transfer[transfer.is_selected].set_index(keys)
    joined = chosen.join(folds.set_index(keys), rsuffix="_fold", validate="one_to_one")
    assert joined.neuron.eq(joined.selected_cell).all()
    assert np.allclose(joined.held_alignment, joined.selected_held_alignment)

    # Six seeded, independent reconstructions from the nearest-to-raw trial
    # table: three per presentation direction. No import of 01/03 functions.
    trial = pd.read_parquet(trial_path).reset_index()
    trial["date"] = trial.date.astype(str)
    rng = np.random.default_rng(SEED)
    indices = []
    for mode in ["first", "later"]:
        available = transfer.index[transfer.train_mode.eq(mode)]
        indices.extend(rng.choice(available, size=3, replace=False).tolist())
    reconstruction_errors = []
    pair_lookup = pairs.set_index("pair_id")
    for idx in indices:
        r = transfer.loc[idx]
        p = pair_lookup.loc[r.pair_id]
        q = trial[trial.date.eq(r.date) & trial.neuron_class.eq(r.neuron) &
                  trial.sample_id.isin([p.strain_a, p.strain_b])].sort_values("segment_index")
        per_animal = {}
        for (animal, strain), g in q.groupby(["worm_key", "sample_id"]):
            binned = np.array([g[[str(k) for k in range(j*5, (j+1)*5)]].mean(axis=1).to_numpy()
                               for j in range(5)]).T
            per_animal[(animal, strain, "first")] = binned[0]
            if len(g) > 1:
                per_animal[(animal, strain, "later")] = binned[1:].mean(axis=0)
        def delta(animal, mode):
            return per_animal[(animal, p.strain_a, mode)] - per_animal[(animal, p.strain_b, mode)]
        train = np.array([delta(a, r.train_mode) for a in r.train_animals.split("|")])
        target = delta(r.held_worm, r.test_mode)
        terms = [np.dot(train[i], train[j])/5 for i in range(len(train))
                 for j in range(len(train)) if i != j]
        training = np.mean(terms) / r.scale**2
        alignment = np.mean(train.mean(axis=0)*target) / r.scale**2
        reconstruction_errors.extend([abs(training-r.train_energy), abs(alignment-r.held_alignment)])
    assert max(reconstruction_errors) < 1e-10
    checks.update(transfer_folds=len(folds), transfer_cell_entries=len(transfer),
                  no_held_animal_in_cell_selection=True,
                  trial_reconstruction_seed=SEED, reconstructed_trial_entries=6,
                  trial_reconstruction_max_abs_error=float(max(reconstruction_errors)))

    species = read("poster_species_group_effects.csv")
    assert len(species) == 7 and species.estimate.lt(0).sum() == 5
    assert np.isclose(species.estimate.mean(), -.468879, atol=1e-6)
    assert species.estimate.between(-1.05, .2).all()
    for path in (OUT / "code").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    checks["compiled_scripts"] = len(list((OUT / "code").glob("*.py")))
    for name in ["poster_neighborhood_atlas", "poster_chemical_neural_landscape",
                 "poster_species_tendency", "poster_neighborhood_atlas_raw"]:
        for ext in ["png", "pdf", "svg"]:
            assert (OUT / "figures" / f"{name}.{ext}").stat().st_size > 1000
    checks["nonempty_figure_exports"] = 12
    assert (L / "independent_review.md").exists()
    pending = {L / name for name in ["environment.json", "verification.json", "output_manifest.json"]}
    n_links = 0
    for path in OUT.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if link.startswith(("https://", "http://", "#")):
                continue
            target = path.parent / link.split("#")[0]
            assert target.exists() or target in pending, (path.name, link)
            n_links += 1
    checks["local_document_links"] = n_links

    environment = dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), seed_for_audit=SEED,
        analysis_randomization="None; deterministic descriptive analysis",
        packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow"]})
    (L / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    result = dict(status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
                  checks=checks, interpretation="Computational verification; no independent biological validation")
    (L / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    output_files = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    and p.name != "output_manifest.json" and p.suffix != ".log"]
    (L / "output_manifest.json").write_text(json.dumps(manifest(output_files, OUT), indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Recorded {len(output_files)} output hashes; streaming .log files and the manifest itself are excluded.")


if __name__ == "__main__":
    main()
