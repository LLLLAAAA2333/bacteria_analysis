"""Chemistry-defined pairs projected onto the exact Figure 4 0–40 s templates.

Notebook entry: analyze_amplitude_neighborhoods(reports_dir, output_dir).
This reads reviewed caches, not raw data, and does not refit any template.
Animal pairing, the eight bins, and original delta-F/F0 units are preserved.
Main-pair selection is a separate explicit call to select_main_pairs().
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd


CELLS = ("AWCON", "ASK", "ADF", "ASJ", "AWA", "AWB", "ASH")
BENCHMARK = "20260601_A022_A023"
MIN_ANIMALS = 3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cross_animal(values):
    """Distinct-animal products, averaged over features; retain negative values."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return np.nan
    if not np.isfinite(x).all():
        raise ValueError("cross_animal requires complete finite observations")
    return float(np.mean((x.sum(axis=0)**2 - (x*x).sum(axis=0)) / (len(x)*(len(x)-1))))


def project_complete_curves(curves, template):
    """Return amplitudes for complete rows; all-missing stays NaN, partial stops."""
    y, h = np.asarray(curves, float), np.asarray(template, float)
    if y.ndim != 2 or y.shape[1] != 8 or h.shape != (8,):
        raise ValueError("Expected animal × 8-bin curves and an 8-bin template")
    if not np.isfinite(h).all() or not np.isclose(np.mean(h*h), 1, rtol=1e-10):
        raise ValueError("Expected a finite unit-RMS template")
    if np.isinf(y).any():
        raise ValueError("Infinite curves are invalid")
    counts = np.isfinite(y).sum(axis=1)
    if not np.isin(counts, [0, 8]).all():
        raise ValueError("Partial-bin support requires a new, explicit analysis rule")
    a = np.full(len(y), np.nan)
    a[counts == 8] = (y[counts == 8] @ h) / np.dot(h, h)
    return a


def paired_statistics(a, b, template):
    """Same animals, same cell: paired curves and descriptive amplitude summaries."""
    a, b, h = np.asarray(a, float), np.asarray(b, float), np.asarray(template, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Pair inputs must have identical finite animal × bin support")
    aa, ab = project_complete_curves(a, h), project_complete_curves(b, h)
    d = aa - ab
    curve = a - b
    residual = curve - d[:, None]*h
    n = len(d)
    mean = float(d.mean()) if n else np.nan
    sem = float(d.std(ddof=1)/np.sqrt(n)) if n >= 2 else np.nan
    mean_curve_sq = float(np.mean(curve.mean(axis=0)**2)) if n else np.nan
    mean_residual_sq = float(np.mean(residual.mean(axis=0)**2)) if n else np.nan
    eligible = n >= MIN_ANIMALS
    energy = cross_animal(d) if eligible else np.nan
    curve_energy = cross_animal(curve) if eligible else np.nan
    residual_energy = cross_animal(residual) if eligible else np.nan
    deletions = []
    if eligible:
        for i in range(n):
            keep = np.arange(n) != i
            deletions.append(dict(deleted_index=i, mean_delta=float(d[keep].mean()),
                                  energy=cross_animal(d[keep]),
                                  curve_energy=cross_animal(curve[keep]),
                                  residual_energy=cross_animal(residual[keep])))
    loo = pd.DataFrame(deletions)
    result = dict(n_animals=n, eligible=eligible, mean_delta=mean,
                  abs_mean_delta=abs(mean), sem_delta=sem,
                  mean_amp_a=float(aa.mean()) if n else np.nan,
                  mean_amp_b=float(ab.mean()) if n else np.nan,
                  n_delta_positive=int((d > 0).sum()), n_delta_negative=int((d < 0).sum()),
                  energy=energy, curve_energy=curve_energy, residual_energy=residual_energy,
                  mean_curve_rms=float(np.sqrt(mean_curve_sq)),
                  mean_residual_rms=float(np.sqrt(mean_residual_sq)),
                  model_residual_fraction=mean_residual_sq/mean_curve_sq if mean_curve_sq > 0 else np.nan,
                  decomposition_error=abs(mean_curve_sq - mean**2 - mean_residual_sq),
                  energy_decomposition_error=abs(curve_energy-energy-residual_energy),
                  energy_identity_error=abs(energy-(mean**2-sem**2)),
                  loo_energy_min=float(loo.energy.min()) if eligible else np.nan,
                  loo_energy_max=float(loo.energy.max()) if eligible else np.nan,
                  loo_mean_min=float(loo.mean_delta.min()) if eligible else np.nan,
                  loo_mean_max=float(loo.mean_delta.max()) if eligible else np.nan)
    return result, aa, ab, d, curve, residual, deletions


def analyze_amplitude_neighborhoods(reports_dir, output_dir):
    """Recompute all 147 audited pairs for seven cells; do not select a main set."""
    from .response_structure_display import validate_evidence

    reports, out = Path(reports_dir), Path(output_dir)
    source = reports / "representation/response_structure_20260930"
    fig4_path = reports / "representation/response_process_draft_20261001/plot_parameters.json"
    fig4 = json.loads(fig4_path.read_text())
    if tuple(fig4["cells"]) != CELLS or fig4["window_seconds"] != [0, 40]:
        raise ValueError("Figure 4 panel or analysis window changed")
    inputs = [fig4_path]
    for name, expected in fig4["source_sha256"].items():
        path = source / name
        if sha256(path) != expected:
            raise ValueError(f"Source no longer matches Figure 4: {name}")
        inputs.append(path)
    validate_evidence(source)
    audit_path = reports / "chemical_neural/chemical_neighborhood_focus_20260930/tables/chemical_neighbor_audit_pairs.csv"
    pairs = pd.read_csv(audit_path, dtype={"date": str})
    if len(pairs) != 147 or pairs.pair_id.duplicated().any() or not (pairs.strain_a < pairs.strain_b).all():
        raise ValueError("Expected 147 unique, lexicographically oriented audited pairs")
    inputs.append(audit_path)
    obs = pd.read_parquet(source / "data/observations.parquet")
    obs = obs.rename(columns={"sample_id": "strain", "neuron_class": "cell"})
    obs = obs.loc[obs.cell.isin(CELLS)].copy()
    obs["block"] = obs.block.astype(str)
    keys = ["strain", "block", "cell", "animal_id"]
    if obs.duplicated(keys + ["bin_index"]).any() or not obs.bin_index.isin(range(8)).all():
        raise ValueError("Duplicate or out-of-window observations")
    wide = obs.pivot(index=keys, columns="bin_index", values="response").reindex(columns=range(8))
    templates = pd.read_csv(source / "tables/templates.csv").query('window == "0-40s" and fit_type == "full"')
    templates = templates.loc[templates.cell.isin(CELLS)]
    h = templates.pivot(index="cell", columns="bin_index", values="template").reindex(index=CELLS, columns=range(8))
    if len(templates) != len(CELLS)*8:
        raise ValueError("Unexpected template support")
    amplitude_rows = []
    for cell in CELLS:
        ht = h.loc[cell].to_numpy()
        if ht[np.argmax(np.abs(ht))] <= 0:
            raise ValueError("Template sign convention changed")
        y = wide.xs(cell, level="cell", drop_level=False)
        z = y.index.to_frame(index=False)
        z["amplitude"] = project_complete_curves(y.to_numpy(), ht)
        z["complete_curve"] = z.amplitude.notna()
        amplitude_rows.append(z)
    amplitudes = pd.concat(amplitude_rows, ignore_index=True)
    projected = amplitudes.groupby(["strain", "block", "cell"]).amplitude.mean()
    coefficients = pd.read_csv(source / "tables/coefficients.csv", dtype={"block": str})
    coefficients = coefficients.query('window == "0-40s" and fit_type == "full"')
    coefficients = coefficients.loc[coefficients.cell.isin(CELLS)].set_index(["strain", "block", "cell"]).coefficient
    projected, coefficients = projected.sort_index(), coefficients.sort_index()
    if not projected.index.equals(coefficients.index) or not np.allclose(projected, coefficients, rtol=1e-10, atol=1e-12):
        raise ValueError("Animal amplitudes do not reproduce the saved Figure 4 coefficients")
    percell, animal_rows, deletion_rows, curve_rows = [], [], [], []
    for p in pairs.itertuples(index=False):
        for cell in CELLS:
            a = wide.loc[(p.strain_a, p.date, cell)].dropna()
            b = wide.loc[(p.strain_b, p.date, cell)].dropna()
            common = a.index.intersection(b.index).sort_values()
            s, aa, ab, d, curves, residuals, deletions = paired_statistics(
                a.loc[common].to_numpy(), b.loc[common].to_numpy(), h.loc[cell].to_numpy())
            s.update(pair_id=p.pair_id, cell=cell,
                     saved_coefficient_delta=float(coefficients.loc[(p.strain_a, p.date, cell)]-coefficients.loc[(p.strain_b, p.date, cell)]),
                     n_animals_a=len(a), n_animals_b=len(b), identical_animal_support=a.index.equals(b.index))
            s["paired_vs_saved_delta"] = s["mean_delta"] - s["saved_coefficient_delta"]
            percell.append(s)
            for i, animal in enumerate(common):
                animal_rows.append(dict(pair_id=p.pair_id, cell=cell, animal_id=animal,
                                        amplitude_a=aa[i], amplitude_b=ab[i], delta=d[i]))
                for j in range(8):
                    curve_rows.append(dict(pair_id=p.pair_id, cell=cell, animal_id=animal,
                                           bin_index=j, difference=curves[i, j], residual=residuals[i, j]))
            for row in deletions:
                animal = common[row.pop("deleted_index")]
                deletion_rows.append(dict(pair_id=p.pair_id, cell=cell, deleted_animal=animal, **row))
    pc, animals, deletions = pd.DataFrame(percell), pd.DataFrame(animal_rows), pd.DataFrame(deletion_rows)
    common_cells = [c for c in CELLS if pc.loc[pc.cell.eq(c), "eligible"].all()]
    if not common_cells:
        raise ValueError("No fixed cell panel meets n>=3 for all pairs")
    summary, pair_deletions = [], []
    for p in pairs.itertuples(index=False):
        q = pc.loc[pc.pair_id.eq(p.pair_id)].set_index("cell")
        row = dict(p._asdict(), selected_main=False, is_benchmark=p.pair_id == BENCHMARK,
                   n_eligible_cells=int(q.eligible.sum()),
                   energy_all7=float(q.energy.mean()) if q.eligible.all() else np.nan,
                   energy_common=float(q.loc[common_cells].energy.mean()),
                   amplitude_rms_common=float(np.sqrt(np.mean(q.loc[common_cells].mean_delta**2))),
                   n_positive_common=int(q.loc[common_cells].energy.gt(0).sum()),
                   n_deletion_positive_common=int(q.loc[common_cells].loo_energy_min.gt(0).sum()))
        z = animals.loc[animals.pair_id.eq(p.pair_id) & animals.cell.isin(common_cells)]
        full = z.pivot(index="animal_id", columns="cell", values="delta").reindex(columns=common_cells)
        complete = full.dropna()
        row["n_complete_common_animals"] = len(complete)
        row["energy_complete_animals"] = cross_animal(complete.to_numpy()) if len(complete) >= MIN_ANIMALS else np.nan
        row["n_animals_any_common"] = len(full)
        for animal in full.index:
            energy = np.mean([cross_animal(full.drop(index=animal)[c].dropna().to_numpy()) for c in common_cells])
            pair_deletions.append(dict(pair_id=p.pair_id, deleted_animal=animal, energy_common=energy))
        pdz = pair_deletions[-len(full):]
        row["loo_energy_common_min"] = min(x["energy_common"] for x in pdz)
        row["loo_energy_common_max"] = max(x["energy_common"] for x in pdz)
        positive = q.loc[common_cells].energy.clip(lower=0)
        row["largest_positive_cell_common"] = str(positive.idxmax()) if positive.sum() > 0 else ""
        row["largest_positive_share_common"] = float(positive.max()/positive.sum()) if positive.sum() > 0 else np.nan
        row["largest_abs_delta_cell_common"] = str(q.loc[common_cells].abs_mean_delta.idxmax())
        summary.append(row)
    summary = pd.DataFrame(summary)
    # Chemical verification uses the existing report mask, with no imputation.
    chem_dir = reports / "population/population_first_20260930/tables"
    fc_path, mask_path = chem_dir / "aligned_chemical_log2fc_all.csv", chem_dir / "aligned_chemical_report_observed_all.parquet"
    fc, mask = pd.read_csv(fc_path, index_col=0), pd.read_parquet(mask_path)
    inputs.extend([fc_path, mask_path])
    chemical_errors = []
    for p in pairs.itertuples(index=False):
        joint = mask.loc[p.strain_a] & mask.loc[p.strain_b]
        rms = float(np.sqrt(np.mean((fc.loc[p.strain_a, joint]-fc.loc[p.strain_b, joint])**2)))
        if int(joint.sum()) != p.n_joint_reported:
            raise ValueError("Joint-report feature coverage changed")
        chemical_errors.append(abs(rms-p.joint_rms_log2fc))
    verification = dict(status="passed", n_pairs=len(summary), n_cells=len(CELLS),
                        n_condition_coefficients=len(projected), n_animal_condition_cells=len(amplitudes),
                        max_coefficient_error=float(np.max(np.abs(projected-coefficients))),
                        max_chemical_rms_error=max(chemical_errors),
                        common_cells=common_cells,
                        n_pair_cells_below_min=int((~pc.eligible).sum()),
                        n_pair_cells_unequal_animal_support=int((~pc.identical_animal_support).sum()),
                        max_paired_vs_saved_delta=float(pc.paired_vs_saved_delta.abs().max()))
    for name in ["decomposition_error", "energy_decomposition_error", "energy_identity_error"]:
        verification["max_"+name] = float(pc[name].max())
    if any(v > 1e-10 for k, v in verification.items() if k.startswith("max_") and k != "max_paired_vs_saved_delta"):
        raise ValueError(f"Numerical audit failed: {verification}")
    table = out / "tables"
    table.mkdir(parents=True, exist_ok=True)
    for name, frame in [("animal_amplitudes", amplitudes), ("pair_cell_summary", pc),
                        ("pair_animal_amplitudes", animals), ("pair_cell_animal_deletion", deletions),
                        ("pair_animal_curve_differences", pd.DataFrame(curve_rows)),
                        ("pair_animal_deletion", pd.DataFrame(pair_deletions)), ("pair_summary", summary)]:
        frame.to_csv(table / f"{name}.csv", index=False)
    templates.to_csv(table / "shared_templates.csv", index=False)
    parameters = dict(window_seconds=[0, 40], bin_seconds=5, cells=list(CELLS),
                      common_cells=common_cells, min_animals=MIN_ANIMALS,
                      pair_orientation="lexicographic strain_a minus strain_b",
                      main_value="absolute mean paired animal template-amplitude difference, original delta_F_over_F0 units",
                      amplitude="dot(animal_curve, fixed_full_template) / dot(template, template)",
                      energy="mean_delta^2 - sample_variance(delta)/n; negative values retained",
                      uncertainty="paired animal SEM and delete-one-animal influence ranges; neither CI nor significance",
                      residual="paired 8-bin curve minus paired amplitude difference times the same template",
                      template_limit="full-data templates, descriptive projections; no held-out or unbiased claim",
                      coverage="complete eight-bin curves; n>=3 for main pair-cell results; no missing fill",
                      normalization="none beyond the saved template unit-RMS normalization",
                      refit=False, main_selection=None,
                      input_sha256={str(p.resolve()): sha256(p) for p in inputs},
                      code_sha256=sha256(__file__))
    (out / "analysis_parameters.json").write_text(json.dumps(parameters, indent=2)+"\n")
    (out / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    return summary, pc, verification


def select_main_pairs(output_dir, reports_dir, selection):
    """Select chemistry only: benchmark rule or within-group joint-report nearest."""
    out, reports = Path(output_dir), Path(reports_dir)
    p = pd.read_csv(out / "tables/pair_summary.csv", dtype={"date": str})
    baseline = p.loc[p.pair_id.eq(BENCHMARK)].iloc[0]
    if selection == "figure5_benchmark":
        selected = p.joint_rms_log2fc.le(baseline.joint_rms_log2fc)
        selected &= p.joint_median_abs_log2fc_difference.le(baseline.joint_median_abs_log2fc_difference)
        for name in ["joint_fraction_abs_difference_le_1", "joint_fraction_abs_difference_le_2", "joint_profile_pearson"]:
            selected &= p[name].ge(baseline[name])
        prior_path = reports / "examples/figure5_chemistry_first_20261001/chemical_shortlist_before_neural_review.csv"
        prior = pd.read_csv(prior_path)
        if set(p.loc[selected, "pair_id"]) != set(prior.pair_id) | {BENCHMARK}:
            raise ValueError("Figure 5 chemical selection is not reproduced")
        rule = "Joint-report RMS and median <= A022/A023; fractions within 1/2 log2FC and profile r >= benchmark. Five prior chemistry-only candidates plus the benchmark. Not chemical equivalence."
    elif selection == "joint_nearest":
        selected = pd.Series(False, index=p.index)
        for _, group in p.groupby(["date", "genus", "reference_group"]):
            for strain in sorted(set(group.strain_a) | set(group.strain_b)):
                candidate = group.loc[group.strain_a.eq(strain) | group.strain_b.eq(strain)]
                selected.loc[candidate.index] |= np.isclose(candidate.joint_rms_log2fc, candidate.joint_rms_log2fc.min(), rtol=0, atol=1e-12)
        rule = "Either strain selects the other by minimum joint-report RMS within its audited date/genus/reference group; all ties retained. Two-strain groups allowed and identified."
    else:
        raise ValueError("Choose figure5_benchmark or joint_nearest explicitly")
    p["selected_main"] = selected
    p.to_csv(out / "tables/pair_summary.csv", index=False)
    p.loc[selected].sort_values(["joint_rms_log2fc", "pair_id"]).to_csv(out / "tables/main_pairs.csv", index=False)
    # A common-feature sensitivity does not change the already fixed selection.
    cdir = reports / "population/population_first_20260930/tables"
    fc = pd.read_csv(cdir / "aligned_chemical_log2fc_all.csv", index_col=0)
    mask = pd.read_parquet(cdir / "aligned_chemical_report_observed_all.parquet")
    strains = sorted(set(p.loc[selected, "strain_a"]) | set(p.loc[selected, "strain_b"]))
    shared = mask.loc[strains].all(axis=0)
    sensitivity = []
    for row in p.loc[selected].itertuples(index=False):
        rms = np.sqrt(np.mean((fc.loc[row.strain_a, shared]-fc.loc[row.strain_b, shared])**2)) if shared.any() else np.nan
        sensitivity.append(dict(pair_id=row.pair_id, n_common_features=int(shared.sum()),
                                joint_rms_log2fc=row.joint_rms_log2fc, common_feature_rms=rms))
    pd.DataFrame(sensitivity).to_csv(out / "tables/common_chemical_features_sensitivity.csv", index=False)
    params = json.loads((out / "analysis_parameters.json").read_text())
    params["main_selection"] = dict(name=selection, rule=rule, n_pairs=int(selected.sum()),
                                    selected_before_neural_pattern_review=True,
                                    benchmark=BENCHMARK if selection == "figure5_benchmark" else None)
    if selection == "figure5_benchmark":
        params["input_sha256"][str(prior_path.resolve())] = sha256(prior_path)
    params["code_sha256"] = sha256(__file__)
    (out / "analysis_parameters.json").write_text(json.dumps(params, indent=2)+"\n")
    return p.loc[selected]


def summarize_main_patterns(output_dir):
    """Descriptive recurrence tables, without clustering, p values or new selection."""
    out = Path(output_dir)
    table = out / "tables"
    params = json.loads((out / "analysis_parameters.json").read_text())
    if params["main_selection"] is None:
        raise ValueError("Fix the chemistry-only main set before inspecting patterns")
    pairs = pd.read_csv(table / "main_pairs.csv", dtype={"date": str})
    pc = pd.read_csv(table / "pair_cell_summary.csv")
    pc = pc.loc[pc.pair_id.isin(pairs.pair_id)].copy()
    animals = pd.read_csv(table / "pair_animal_amplitudes.csv")
    extra = []
    for row in pc.itertuples(index=False):
        a = animals.loc[animals.pair_id.eq(row.pair_id) & animals.cell.eq(row.cell)]
        n = len(a)
        opposite = row.mean_amp_a * row.mean_amp_b < 0
        stable_opposite = False
        if row.eligible:
            loo_a = (a.amplitude_a.sum()-a.amplitude_a)/(n-1)
            loo_b = (a.amplitude_b.sum()-a.amplitude_b)/(n-1)
            stable_opposite = bool(opposite and (loo_a*loo_b < 0).all())
        extra.append(dict(pair_id=row.pair_id, cell=row.cell,
                          sem_amp_a=float(a.amplitude_a.sem()), sem_amp_b=float(a.amplitude_b.sem()),
                          n_amp_a_positive=int(a.amplitude_a.gt(0).sum()),
                          n_amp_b_positive=int(a.amplitude_b.gt(0).sum()),
                          opposite_mean_amplitudes=opposite,
                          opposite_after_each_deletion=stable_opposite,
                          energy_positive_after_each_deletion=bool(row.eligible and row.loo_energy_min > 0)))
    pc = pc.merge(pd.DataFrame(extra), on=["pair_id", "cell"], validate="one_to_one")
    pc.to_csv(table / "main_pair_cell_patterns.csv", index=False)
    rows = []
    for cell in CELLS:
        q = pc.loc[pc.cell.eq(cell) & pc.eligible]
        supported = q.loc[q.energy_positive_after_each_deletion]
        rows.append(dict(cell=cell, n_eligible_pairs=len(q),
                         median_abs_delta=float(q.abs_mean_delta.median()),
                         min_abs_delta=float(q.abs_mean_delta.min()), max_abs_delta=float(q.abs_mean_delta.max()),
                         n_positive_energy=int(q.energy.gt(0).sum()),
                         n_energy_positive_after_each_deletion=len(supported),
                         n_opposite_mean_amplitudes=int(q.opposite_mean_amplitudes.sum()),
                         n_opposite_after_each_deletion=int(q.opposite_after_each_deletion.sum()),
                         supported_pair_ids=";".join(supported.pair_id)))
    pd.DataFrame(rows).to_csv(table / "main_cell_patterns.csv", index=False)
    combinations_rows = []
    for c1, c2 in combinations(params["common_cells"], 2):
        q1 = pc.loc[pc.cell.eq(c1)].set_index("pair_id")
        q2 = pc.loc[pc.cell.eq(c2)].set_index("pair_id").reindex(q1.index)
        supported = q1.energy_positive_after_each_deletion & q2.energy_positive_after_each_deletion
        ids = q1.index[supported]
        group_info = pairs.loc[pairs.pair_id.isin(ids)]
        products = q1.loc[ids].mean_delta*q2.loc[ids].mean_delta
        combinations_rows.append(dict(cell_a=c1, cell_b=c2, n_pairs_both_deletion_positive=int(supported.sum()),
                                      n_same_coefficient_direction=int(products.gt(0).sum()),
                                      n_opposite_coefficient_direction=int(products.lt(0).sum()),
                                      n_dates=group_info.date.nunique(),
                                      n_comparison_groups=len(group_info[["date", "genus", "reference_group"]].drop_duplicates()),
                                      pair_ids=";".join(ids)))
    pd.DataFrame(combinations_rows).to_csv(table / "main_cell_combinations.csv", index=False)
    params["pattern_summary"] = {
        "scope": "All selected pairs; continuous differences and descriptive recurrence only",
        "influence_marker": "U remains >0 after deleting any one paired animal; not a significance threshold",
        "opposite_amplitudes": "Opposite mean template coefficients, checked again after each animal deletion; not whole-curve reversal",
        "coefficient_direction": "A-minus-B signs relative to cell-specific templates; not excitation/inhibition",
        "combinations": "All 15 pairs of six commonly covered cells; shared animals/pairs/groups are not independent replications",
        "scale_limit": "Original calcium units; largest entries depend on cell response scales",
    }
    params["code_sha256"] = sha256(__file__)
    (out / "analysis_parameters.json").write_text(json.dumps(params, indent=2)+"\n")
    return pc
