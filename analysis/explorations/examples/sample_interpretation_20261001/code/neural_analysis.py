"""Limited, animal-paired interpretation of five previously selected samples.

Run from the repository with:
    .pixi/envs/default/bin/python reports/sample_interpretation_20261001/code/neural_analysis.py

Reuses verified animal responses and existing whole-animal-held-out templates;
does not fit a model, change the baseline, normalize a curve, or run a Notebook.
All means use animals, not trials, as replicates. NaN remains missing.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
TABLES, LOGS = OUT / "tables", OUT / "logs"
SOURCE = ROOT / "reports/response_structure_20260930"
CURVE_FILE = ROOT / "reports/exploration_20260929/tables/animal_curves.parquet"
SAMPLES = {"A021": "20260601", "A022": "20260601", "A023": "20260601",
           "A007": "20260520", "A010": "20260520"}
GROUPS = {"main": ["A021", "A022", "A023"], "contrast": ["A007", "A010"]}
PAIRS = [("A021", "A022"), ("A021", "A023"), ("A022", "A023"), ("A007", "A010")]
CELLS = ["ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
         "ASEL", "ASER", "AWCON", "AWCOFF"]
STAGES = {"0-10s": (0, 10), "10-25s": (10, 25), "25-40s": (25, 40),
          "0-25s": (0, 25), "0-40s": (0, 40)}
KEYS = ["sample_id", "date", "animal_id", "neuron_class"]


def fingerprint(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(table, name):
    if "neuron_class" in table:
        table = table.copy()
        table["cell_order"] = table.neuron_class.map({cell: i for i, cell in enumerate(CELLS)})
        sort_keys = [k for k in ["pair", "sample_id", "date", "animal_id", "window",
                                 "cell_order", "start_s", "time_start", "time_s", "metric"] if k in table]
        table = table.sort_values(sort_keys, kind="stable")
    table.to_csv(TABLES / ("neural_" + name + ".csv"), index=False)
    return table


def describe(values):
    """Descriptive sign and delete-one sensitivity; no inferential interval."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if not n:
        return dict(n=0, mean=np.nan, sd=np.nan, sem=np.nan, median=np.nan,
                    minimum=np.nan, maximum=np.nan, n_positive=0, n_negative=0,
                    n_zero=0, loo_mean_min=np.nan, loo_mean_max=np.nan)
    loo = (y.sum() - y) / (n - 1) if n > 1 else np.array([np.nan])
    return dict(n=n, mean=float(y.mean()), sd=float(y.std(ddof=1)) if n > 1 else np.nan,
                sem=float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                median=float(np.median(y)), minimum=float(y.min()), maximum=float(y.max()),
                n_positive=int((y > 0).sum()), n_negative=int((y < 0).sum()),
                n_zero=int((y == 0).sum()),
                loo_mean_min=float(loo.min()), loo_mean_max=float(loo.max()))


def summarize(table, keys, value):
    rows = []
    for key, group in table.groupby(keys, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, key), **describe(group[value])))
    return pd.DataFrame(rows)


def stages_from_windows(table, keys, value):
    rows = []
    for key, group in table.groupby(keys, sort=False, dropna=False):
        for stage, (start, stop) in STAGES.items():
            part = group[group.time_start.ge(start) & group.time_end.le(stop)]
            n_expected = (stop - start) // 5
            n_finite = int(np.isfinite(part[value]).sum())
            # Requiring full stage coverage prevents variable windows from
            # quietly changing the estimand. The selected inputs have no
            # partially missing observed cells.
            mean = part[value].mean() if n_finite == n_expected else np.nan
            rows.append(dict(zip(keys, key), stage=stage, start_s=start, end_s=stop,
                             n_finite_bins=n_finite, n_expected_bins=n_expected,
                             **{value: mean}))
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    inputs = [SOURCE / "data/observations.parquet", SOURCE / "tables/templates.csv",
              SOURCE / "tables/folds.csv", SOURCE / "data/preparation_metadata.json", CURVE_FILE,
              ROOT / "reports/population_first_20260930/tables/aligned_neural_animal_5bins.parquet"]
    before = [fingerprint(p) for p in inputs]
    obs = pd.read_parquet(inputs[0])
    obs = obs.loc[obs.sample_id.isin(SAMPLES)].copy()
    for column in ["date", "block", "animal_id", "worm_key"]:
        obs[column] = obs[column].astype(str)
    assert obs.date.eq(obs.sample_id.map(SAMPLES)).all()
    assert not obs.duplicated(KEYS + ["bin_index"]).any()
    assert set(obs.neuron_class) == set(CELLS)
    assert set(obs.n_volumes) == {0, 5}
    save(obs, "observations")

    coverage = []
    for (sample, date, cell), part in obs.groupby(["sample_id", "date", "neuron_class"]):
        animal_n = part.groupby("animal_id").response.count()
        assert animal_n.isin([0, 8]).all()
        coverage.append(dict(sample_id=sample, date=date, neuron_class=cell,
                             n_recorded_animals=len(animal_n), n_observed_animals=int((animal_n == 8).sum()),
                             observed_animals=";".join(animal_n[animal_n.eq(8)].index),
                             missing_animals=";".join(animal_n[animal_n.eq(0)].index)))
    coverage = save(pd.DataFrame(coverage), "coverage")
    group_coverage = []
    for group, samples in GROUPS.items():
        animal_sets = [set(obs.loc[obs.sample_id.eq(s), "animal_id"]) for s in samples]
        all_common = set.intersection(*animal_sets)
        for cell in CELLS:
            cell_sets = [set(obs.loc[obs.sample_id.eq(s) & obs.neuron_class.eq(cell)
                                     & obs.response.notna(), "animal_id"]) for s in samples]
            common = set.intersection(*cell_sets)
            # All plotted sample means and pair differences are on identical
            # per-cell animal support in these selected groups.
            assert all(animals == common for animals in cell_sets)
            group_coverage.append(dict(group=group, sample_ids=";".join(samples), neuron_class=cell,
                                       n_common_recorded=len(all_common), n_common_observed=len(common),
                                       all_samples_have_identical_observed_animals=True,
                                       common_observed_animals=";".join(sorted(common))))
    save(pd.DataFrame(group_coverage), "group_coverage")

    # Relative physical time, one-second source sampling; no interpolation.
    curves = pd.read_parquet(CURVE_FILE).reset_index()
    curves = curves[curves.sample_id.isin(SAMPLES)].copy()
    curves.date = curves.date.astype(str)
    curves["animal_id"] = curves.date + "|" + curves.worm_key.astype(str)
    assert curves.date.eq(curves.sample_id.map(SAMPLES)).all()
    curve_long = curves.melt(id_vars=KEYS + ["worm_key"],
                            value_vars=[str(t) for t in range(-5, 40)],
                            var_name="time_s", value_name="response")
    curve_long.time_s = curve_long.time_s.astype(int)
    curve_long["source_time_point"] = curve_long.time_s + 5
    save(curve_long.sort_values(KEYS + ["time_s"]), "curves")
    save(summarize(curve_long, ["sample_id", "date", "neuron_class", "time_s"], "response"), "curve_summary")
    curve_bins = curve_long[curve_long.time_s.ge(0)].copy()
    curve_bins["bin_index"] = curve_bins.time_s // 5
    curve_bins = curve_bins.groupby(KEYS + ["bin_index"], as_index=False).response.mean()
    checked = obs.merge(curve_bins, on=KEYS + ["bin_index"], how="left", suffixes=("", "_curve"), validate="one_to_one")
    assert np.array_equal(np.isfinite(checked.response), np.isfinite(checked.response_curve))
    curve_error = float(np.nanmax(abs(checked.response - checked.response_curve)))
    assert curve_error < 1e-12
    save(summarize(obs, ["sample_id", "date", "neuron_class", "bin_index", "time_start", "time_end"], "response"), "window_summary")
    animal_stages = save(stages_from_windows(obs, KEYS, "response"), "animal_stages")
    save(summarize(animal_stages, ["sample_id", "date", "neuron_class", "stage", "start_s", "end_s"], "response"), "stage_summary")

    # Independent reuse check against the established five-bin population table.
    old = pd.read_parquet(inputs[-1]).reset_index()
    old = old[old.sample_id.isin(SAMPLES)].copy()
    old.date = old.date.astype(str)
    old["animal_id"] = old.date + "|" + old.worm_key.astype(str)
    old = old.set_index(["sample_id", "date", "animal_id"])
    differences = []
    for row in obs[obs.bin_index.lt(5)].itertuples():
        col = f"{row.neuron_class}__{row.time_start:02d}_{row.time_end:02d}"
        val = old.loc[(row.sample_id, row.date, row.animal_id), col]
        assert np.isfinite(row.response) == np.isfinite(val)
        if np.isfinite(val):
            differences.append(abs(val - row.response))
    assert max(differences) < 1e-12

    paired = []
    for first, second in PAIRS:
        keys = ["date", "animal_id", "neuron_class", "bin_index", "time_start", "time_end"]
        pair = obs[obs.sample_id.eq(first)][keys + ["response"]].merge(
            obs[obs.sample_id.eq(second)][keys + ["response"]], on=keys,
            suffixes=("_first", "_second"), validate="one_to_one")
        pair.insert(0, "pair", second + "-" + first)
        pair.insert(1, "sample_first", first)
        pair.insert(2, "sample_second", second)
        pair["difference"] = pair.response_second - pair.response_first
        paired.append(pair)
    paired = save(pd.concat(paired, ignore_index=True), "paired_windows")
    pair_keys = ["pair", "sample_first", "sample_second", "date", "animal_id", "neuron_class"]
    paired_stages = save(stages_from_windows(paired, pair_keys, "difference"), "paired_stages")
    save(summarize(paired, [k for k in pair_keys if k != "animal_id"] + ["bin_index", "time_start", "time_end"], "difference"), "pair_window_summary")
    save(summarize(paired_stages, [k for k in pair_keys if k != "animal_id"] + ["stage", "start_s", "end_s"], "difference"), "pair_stage_summary")

    templates = pd.read_csv(inputs[1])
    templates = templates[templates.fit_type.eq("loao") & templates.fold.isin(obs.animal_id)].copy()
    save(templates, "oof_templates")
    folds = pd.read_csv(inputs[2])
    train_sets = {(r.window, r.heldout_animal): set(json.loads(r.training_animals)) for r in folds.itertuples()}
    for (_, held), train in train_sets.items():
        assert held not in train
    template_map = {(window, animal, cell): part.sort_values("bin_index").template.to_numpy()
                    for (window, animal, cell), part in templates.groupby(["window", "fold", "cell"])}
    projection_rows, projection_windows, projection_stages = [], [], []
    identity_errors, orthogonality_errors, template_errors = [], [], []
    for (pair, first, second, date, cell), part in paired.groupby(
            ["pair", "sample_first", "sample_second", "date", "neuron_class"]):
        all_differences = part.pivot(index="animal_id", columns="bin_index", values="difference")
        for window, n_bins in [("0-25s", 5), ("0-40s", 8)]:
            complete = all_differences.loc[:, range(n_bins)].dropna()
            for animal, y in complete.iterrows():
                train_animals = complete.index.difference([animal])
                if len(train_animals) == 0:
                    continue
                assert set(train_animals).issubset(train_sets[(window, animal)])
                t = template_map[(window, animal, cell)]
                assert len(t) == n_bins and np.isfinite(t).all()
                norm = float(np.mean(t * t))
                template_errors.append(abs(norm - 1))
                assert abs(norm - 1) < 1e-12
                assert t[np.argmax(abs(t))] > 0
                d = y.to_numpy()
                other = complete.loc[train_animals].mean().to_numpy()
                alpha = float(np.dot(d, t) / np.dot(t, t))
                alpha_other = float(np.dot(other, t) / np.dot(t, t))
                parallel, parallel_other = alpha * t, alpha_other * t
                residual, residual_other = d - parallel, other - parallel_other
                along = float(np.mean(parallel * parallel_other))
                outside = float(np.mean(residual * residual_other))
                total = float(np.mean(d * other))
                identity_errors.append(abs(along + outside - total))
                orthogonality_errors += [abs(float(np.mean(residual * t))), abs(float(np.mean(residual_other * t)))]
                keys = dict(pair=pair, sample_first=first, sample_second=second, date=date,
                            neuron_class=cell, window=window, animal_id=animal)
                projection_rows.append(dict(**keys, n_paired_animals=len(complete), n_other_paired_animals=len(train_animals),
                                            template_mean_square=norm, alpha=alpha, other_mean_alpha=alpha_other,
                                            along_agreement=along, residual_agreement=outside, total_agreement=total,
                                            difference_rms=float(np.sqrt(np.mean(d*d))),
                                            residual_rms=float(np.sqrt(np.mean(residual*residual))),
                                            other_residual_rms=float(np.sqrt(np.mean(residual_other*residual_other)))))
                for k in range(n_bins):
                    projection_windows.append(dict(**keys, bin_index=k, time_start=5*k, time_end=5*k+5,
                                                   template=t[k], difference=d[k], parallel=parallel[k], residual=residual[k],
                                                   other_mean_difference=other[k], other_parallel=parallel_other[k],
                                                   other_residual=residual_other[k]))
                for stage, (start, stop) in STAGES.items():
                    if stop > n_bins * 5:
                        continue
                    ix = slice(start // 5, stop // 5)
                    projection_stages.append(dict(**keys, stage=stage, start_s=start, end_s=stop,
                                                  residual_stage_mean=float(np.mean(residual[ix])),
                                                  other_residual_stage_mean=float(np.mean(residual_other[ix])),
                                                  residual_stage_agreement=float(np.mean(residual[ix] * residual_other[ix]))))
    projection = save(pd.DataFrame(projection_rows), "oof_projection_animals")
    save(pd.DataFrame(projection_windows), "oof_projection_windows")
    projection_stages = save(pd.DataFrame(projection_stages), "oof_projection_stages")
    projection_keys = ["pair", "sample_first", "sample_second", "date", "neuron_class", "window"]
    summaries = []
    for metric in ["alpha", "along_agreement", "residual_agreement", "total_agreement"]:
        s = summarize(projection, projection_keys, metric)
        s["metric"] = metric
        summaries.append(s)
    save(pd.concat(summaries, ignore_index=True), "oof_projection_summary")
    stage_summaries = []
    for metric in ["residual_stage_mean", "residual_stage_agreement"]:
        s = summarize(projection_stages, projection_keys + ["stage", "start_s", "end_s"], metric)
        s["metric"] = metric
        stage_summaries.append(s)
    save(pd.concat(stage_summaries, ignore_index=True), "oof_projection_stage_summary")

    # Focused identities detect swapped pair direction, template mismatch,
    # incorrect projection scaling and corrupted missingness, independently
    # of whether any scientific example looks compelling.
    triad = paired[paired.pair.ne("A010-A007")].pivot(index=["animal_id", "neuron_class", "bin_index"], columns="pair", values="difference")
    triangle_error = float(np.nanmax(abs(triad["A023-A021"] - triad["A022-A021"] - triad["A023-A022"])))
    assert triangle_error < 1e-12
    assert max(identity_errors) < 1e-12
    assert max(orthogonality_errors) < 1e-12
    # Independent direct slice reproduces a stage mean without groupby.
    selected = obs[(obs.sample_id == "A022") & (obs.animal_id == "20260601|w1") & (obs.neuron_class == "AWCON")]
    direct_stage = selected[selected.bin_index.isin([2, 3, 4])].response.to_numpy().mean()
    staged = animal_stages[(animal_stages.sample_id == "A022") & (animal_stages.animal_id == "20260601|w1")
                           & (animal_stages.neuron_class == "AWCON") & (animal_stages.stage == "10-25s")].response.iloc[0]
    assert abs(direct_stage - staged) < 1e-12
    after = [fingerprint(p) for p in inputs]
    assert before == after
    checks = dict(selected_samples=SAMPLES, n_animal_conditions=int(obs[KEYS[:3]].drop_duplicates().shape[0]),
                  n_animals=int(obs.animal_id.nunique()), n_observation_rows=len(obs),
                  n_finite_observation_rows=int(obs.response.notna().sum()),
                  n_curve_rows=len(curve_long), n_paired_window_rows=len(paired),
                  n_projection_animal_rows=len(projection), curve_bin_max_abs_error=curve_error,
                  existing_population_bin_max_abs_error=max(differences), triad_difference_identity_max_error=triangle_error,
                  projection_additivity_max_error=max(identity_errors), projection_orthogonality_max_error=max(orthogonality_errors),
                  template_rms_squared_max_error=max(template_errors), independent_stage_mean_check=True,
                  all_heldout_animals_excluded_from_template_training=True,
                  within_group_sample_and_pair_animal_support_identical_for_every_cell=True,
                  complete_stage_and_projection_vectors_required=True, input_files_unchanged=True)
    (LOGS / "neural_verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    methods = dict(
        scope="Five selected samples only; no new model fitting or whole-Notebook execution",
        animal_unit="date|worm_key; animal means across trials reused unchanged",
        sample_dates=SAMPLES, pairs_second_minus_first=PAIRS, neuron_classes=CELLS,
        primary_window="[0,25) s", support_window="[0,40) s", bin_seconds=5, stages=STAGES,
        units="delta_F_over_F0; no additional baseline subtraction or curve scaling",
        time="seconds relative to stimulus onset; source time_point = time_s + 5; stimulus [0,10) s",
        missingness="NaN preserved; phase and projection require all constituent bins; selected observed cells have complete time coverage",
        summary="Unweighted animal means, sample SD, descriptive SEM, signed counts and delete-one means; no p-values or confirmatory intervals",
        templates="Existing global leave-one-animal-out templates; same window and cell; all records of test animal excluded from original fit",
        template_scale="mean(t**2)=1; largest absolute template entry is positive; template may be biphasic",
        projection="For each held-out animal delta = second - first; alpha = dot(delta,t)/dot(t,t); residual = delta - alpha*t. Alpha uses the test animal's observed difference; this is descriptive projection onto an independently estimated template, not prediction of the held-out response. The oof prefix refers to the template origin only.",
        reproducibility="Project other paired animals' mean delta using the same held-out-animal template. Average parallel_test*parallel_other and residual_test*residual_other across bins. The two sum to average delta_test*delta_other. Positive residual agreement indicates same-direction template-external variation across animals; negative is retained.",
        agreement_units="(delta_F_over_F0)^2; neither residual RMS nor agreement is explained biological variance",
        uncertainty="These examples, templates and selection are from the same dataset. Overlapping training sets and post hoc cell selection do not create independent validation. Score delete-one means remove scores only, not refit every nested training set.",
        limits=["Strain effects remain confounded with the fixed stimulus sequence in this acquisition protocol.",
                "Calcium response magnitudes across cell types are not comparable firing rates.",
                "Phase mean differences describe calcium trajectories, not precise firing latency or receptor mechanism.",
                "A positive pair difference does not mean either sample is above baseline.",
                "A007/A010 is a previously unstable comparison, not a validated negative control."],
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    (LOGS / "neural_methods.json").write_text(json.dumps(methods, indent=2) + "\n")
    (LOGS / "neural_input_manifest.json").write_text(json.dumps(before + [fingerprint(Path(__file__))], indent=2) + "\n")
    outputs = sorted(TABLES.glob("neural_*.csv")) + [LOGS / "neural_verification.json", LOGS / "neural_methods.json"]
    (LOGS / "neural_output_manifest.json").write_text(json.dumps([fingerprint(p) for p in outputs], indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
