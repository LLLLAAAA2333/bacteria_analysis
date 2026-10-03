"""Notebook-callable chemical–neural exploration from the original workbook.

No old fold changes, chemical denominators, group mappings, panels or candidates
are read. The already defined 13-cell neural template coefficients are retained.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split


PARAMETERS = {
    "seed": 20261003, "holdout_fraction": 1 / 3,
    "split": "one unstratified random strain split; no chemical or taxonomic grouping",
    "source_sheet": "metabolism_raw_data.xlsx / all",
    "unit_source": "same workbook / A_vs_B / Name and unit columns",
    "qc_rsd": "sample SD (ddof=1) / mean of observed QC-1 through QC-39",
    "maximum_qc_rsd": 0.30, "minimum_qc_observations": 2,
    "chemical_completeness": "finite and positive original report values in all participating strains; no imputation",
    "chemical_transform": "log2(reported ng/mL divided by 1 ng/mL); no pseudocount",
    "neural_representation": "existing 13 signed template coefficients; recompute per-strain L2 normalization",
    "recording_date_gate": False,
    "chemical_candidates": "discovery chemistry only, average linkage of 1 - absolute Spearman correlation",
    "linkage_cut": 0.30, "minimum_annotations": 3, "minimum_mass_column_families": 3,
    "minimum_pc1_variance_fraction": 0.50,
    "axis": "PC1 after discovery mean/SD scaling; weight annotations by 1/sqrt(same Mass-column family count)",
    "orientation": "largest absolute effective loading positive; discovery score SD one",
    "model": "single intercept and one chemical score predicting all 13 unit coefficients",
    "baseline": "discovery mean 13-dimensional neural vector",
    "selection": "best pooled 5-fold discovery full-vector SSE improvement over fold-training neural means",
    "inner_folds": "fixed outer-discovery chemical memberships; scales, PC1 and neural regression refitted per fold",
    "evaluation": "one frozen candidate; fixed neural direction correlation, full-vector error, slope cosine",
    "secondary_descriptions": ["heldout within-genus association", "leave-one-genus-out association",
        "selected chemistry covariation", "association with whole-panel mean log concentration",
        "pre-gate same-template neural sensitivity"],
    "limitations": [
        "same previously explored 106-strain dataset; this is internal exploratory train/holdout checking, not a new independent validation experiment",
        "neural templates and SNR representation are existing atlas estimates; chemical completeness uses all current strains' availability",
        "no verified experimental nuisance groups supplied; culture, medium, sample level and measurement effects may remain",
        "chemical report material is independently cultured and is not the measured neural stimulus aliquot",
        "genus and recording dates are retained for description, not used to form chemical modules or the primary regression",
    ],
}


def save_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        return value
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2) + "\n")


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1e-12 else np.nan


def source_inputs(root):
    root = Path(root)
    neural_dir = root / "reports/exploration_response_profiles_individual_snr_20261002/tables"
    paths = [root / "data/metabolism_raw_data.xlsx", root / "data/GM300_bacteria_species_summary.xlsx",
             neural_dir / "strain_coefficients.csv", neural_dir / "condition_metrics.csv"]
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    coefficients = pd.read_csv(paths[2], index_col="strain").sort_index()
    assert coefficients.index.is_unique and coefficients.shape == (106, 13)
    assert np.isfinite(coefficients.to_numpy()).all()
    norms = np.linalg.norm(coefficients, axis=1)
    assert (norms > 0).all()
    neural = coefficients.div(norms, axis=0)
    source = pd.read_excel(paths[0], sheet_name="all")
    source["name"] = source["name"].str.strip()
    source = source.set_index("name", verify_integrity=True)
    values = source.loc[:, coefficients.index].apply(pd.to_numeric, errors="raise").T
    values.index.name, values.columns.name = "strain", "metabolite"
    qc_cols = [f"QC-{i}" for i in range(1, 40)]
    qc = source[qc_cols].apply(pd.to_numeric, errors="raise")
    rsd = qc.std(axis=1, ddof=1) / qc.mean(axis=1)
    audit = pd.DataFrame({"n_observed_strains": values.notna().sum(),
                          "all_positive_finite": np.isfinite(values).all() & values.gt(0).all(),
                          "qc_n": qc.notna().sum(axis=1), "qc_rsd_recomputed": rsd,
                          "qc_rsd_reported": source.QCRSD})
    audit["selected"] = (audit.all_positive_finite & audit.qc_n.ge(PARAMETERS["minimum_qc_observations"])
                         & np.isfinite(rsd) & rsd.between(0, PARAMETERS["maximum_qc_rsd"]))
    assert np.allclose(rsd, source.QCRSD, atol=1e-12, rtol=1e-10, equal_nan=True)
    names = list(audit.index[audit.selected])
    assert len(names) >= 3
    raw = values[names]
    chemical = np.log2(raw)
    fields = ["Mass", "RT", "column", "ChineseName", "KEGG", "HMDB", "SuperClass", "Class", "SubClass", "DirectParent"]
    metadata = source.loc[names, fields].copy().rename_axis("metabolite")
    units = pd.read_excel(paths[0], sheet_name="A_vs_B", usecols=["Name", "unit"])
    units["Name"] = units.Name.str.strip()
    units = units.set_index("Name", verify_integrity=True)
    metadata["unit"] = units.loc[names, "unit"].to_numpy()
    assert metadata.unit.eq("ng/mL").all()
    metadata["family"] = [f"{mass} | {column}" if pd.notna(mass) else str(name)
                          for name, mass, column in zip(metadata.index, metadata.Mass, metadata.column)]
    metadata["qc_rsd_recomputed"] = rsd.loc[names]
    taxonomy = pd.read_excel(paths[1], sheet_name="Axxx_species_mapping").set_index("AID", verify_integrity=True)
    context = taxonomy.loc[coefficients.index, ["genus_clean", "species_clean", "QC_flag"]].rename(
        columns={"genus_clean": "genus", "species_clean": "species", "QC_flag": "taxonomy_note"})
    assert context[["genus", "species"]].notna().all().all()
    context.index.name = "strain"
    conditions = pd.read_csv(paths[3])
    assert not conditions.duplicated(["strain", "block", "cell"]).any()
    rebuilt_coefficients = conditions.groupby(["strain", "cell"]).coefficient.mean().unstack("cell").loc[coefficients.index, coefficients.columns]
    assert np.allclose(rebuilt_coefficients, coefficients, atol=1e-12, rtol=1e-10)
    context["dates"] = conditions.groupby("strain").block.apply(lambda v: ";".join(sorted(set(v.astype(str)))))
    pre_gate = conditions.groupby(["strain", "cell"]).raw_coefficient.mean().unstack("cell").loc[coefficients.index, coefficients.columns]
    pre_gate_norm = np.linalg.norm(pre_gate, axis=1)
    pre_gate = pre_gate.div(np.where(pre_gate_norm > 1e-12, pre_gate_norm, np.nan), axis=0)
    return chemical, neural, context, metadata, audit, values, pre_gate, manifest


def fit_axis(x, metadata):
    mean, sd = x.mean(), x.std(ddof=1)
    assert sd.gt(1e-12).all()
    family = metadata.loc[x.columns, "family"]
    weight = 1 / np.sqrt(family.map(family.value_counts()).to_numpy(float))
    weighted_z = ((x - mean) / sd).to_numpy() * weight
    _, singular, vt = np.linalg.svd(weighted_z, full_matrices=False)
    effective = vt[0] * weight
    if effective[np.argmax(np.abs(effective))] < 0:
        effective = -effective
    score = ((x - mean) / sd) @ effective
    effective /= score.std(ddof=1)
    return {"members": list(x.columns), "mean": mean.to_dict(), "sd": sd.to_dict(),
            "effective_weight": dict(zip(x.columns, effective)),
            "pc1_variance_fraction": float(singular[0] ** 2 / np.sum(singular ** 2))}


def apply_axis(axis, x):
    z = (x[axis["members"]] - pd.Series(axis["mean"])) / pd.Series(axis["sd"])
    return (z @ pd.Series(axis["effective_weight"])).rename("chemical_score"), z


def fit_neural(score, y):
    centered_score = score - score.mean()
    denominator = float(centered_score @ centered_score)
    assert denominator > 1e-12
    beta = (y - y.mean()).T @ centered_score / denominator
    intercept = y.mean() - beta * score.mean()
    return beta, intercept


def prediction(score, beta, intercept):
    return np.outer(score, beta) + np.asarray(intercept)


def chemical_modules(x, metadata):
    correlations = x.corr(method="spearman")
    distances = np.clip(1 - np.abs(correlations.to_numpy()), 0, 1)
    np.fill_diagonal(distances, 0)
    labels = fcluster(linkage(squareform(distances, checks=False), method="average"),
                      t=PARAMETERS["linkage_cut"], criterion="distance")
    modules = []
    for label in sorted(set(labels)):
        members = sorted(x.columns[labels == label])
        families = metadata.loc[members, "family"].nunique()
        if len(members) < PARAMETERS["minimum_annotations"] or families < PARAMETERS["minimum_mass_column_families"]:
            continue
        axis = fit_axis(x[members], metadata)
        if axis["pc1_variance_fraction"] < PARAMETERS["minimum_pc1_variance_fraction"]:
            continue
        rho = correlations.loc[members, members].to_numpy()[np.triu_indices(len(members), 1)]
        modules.append({"members": members, "n_members": len(members), "n_families": int(families),
                        "median_abs_pair_rho": float(np.median(np.abs(rho))),
                        "minimum_abs_pair_rho": float(np.min(np.abs(rho))),
                        "pc1_variance_fraction": axis["pc1_variance_fraction"]})
    modules.sort(key=lambda m: (-m["n_members"], tuple(m["members"])))
    for i, module in enumerate(modules, 1):
        module["module_id"] = f"M{i:02d}"
    return modules, correlations


def cv_score(members, x, y, metadata, folds):
    rows = []
    for fold, (train, test) in enumerate(folds, 1):
        xt, xv, yt, yv = x.iloc[train][members], x.iloc[test][members], y.iloc[train], y.iloc[test]
        axis = fit_axis(xt, metadata)
        st, _ = apply_axis(axis, xt)
        sv, _ = apply_axis(axis, xv)
        beta, intercept = fit_neural(st, yt)
        pred = prediction(sv, beta, intercept)
        sse = float(np.sum((yv.to_numpy() - pred) ** 2))
        base = float(np.sum((yv.to_numpy() - yt.mean().to_numpy()) ** 2))
        rows.append({"fold": fold, "n": len(test), "model_sse": sse, "mean_baseline_sse": base})
    return 1 - sum(r["model_sse"] for r in rows) / sum(r["mean_baseline_sse"] for r in rows), rows


def run_discovery(root, out):
    out = Path(out)
    assert not (out / "frozen_candidate.json").exists(), "Candidate already frozen."
    (out / "tables").mkdir(parents=True, exist_ok=True)
    save_json(out / "parameters.json", PARAMETERS)
    x, y, context, meta, audit, raw_all, pre_gate, manifest = source_inputs(root)
    save_json(out / "source_manifest.json", manifest)
    raw_all.to_csv(out / "tables/chemical_report_all_380.csv")
    audit.rename_axis("metabolite").to_csv(out / "tables/fresh_feature_audit.csv")
    meta.to_csv(out / "tables/fresh_feature_metadata.csv")
    x.to_csv(out / "tables/fresh_chemical_log2.csv")
    y.to_csv(out / "tables/neural_unit_coefficients.csv")
    pre_gate.to_csv(out / "tables/neural_pre_gate_unit_coefficients.csv")
    context.to_csv(out / "tables/sample_context.csv")
    train, test = train_test_split(np.arange(len(x)), test_size=PARAMETERS["holdout_fraction"], random_state=PARAMETERS["seed"])
    discovery, holdout = sorted(x.index[train]), sorted(x.index[test])
    split = context.copy()
    split["split"] = np.where(split.index.isin(discovery), "discovery", "holdout")
    split.to_csv(out / "tables/strain_split.csv")
    split.groupby(["split", "genus"]).size().rename("n").to_csv(out / "tables/genus_coverage.csv")
    xd, yd = x.loc[discovery], y.loc[discovery]
    modules, correlations = chemical_modules(xd, meta)
    save_json(out / "chemical_modules.json", modules)
    correlations.to_csv(out / "tables/discovery_chemical_correlations.csv")
    assert modules, "No chemical modules pass the frozen eligibility rules."
    folds = list(KFold(n_splits=5, shuffle=True, random_state=PARAMETERS["seed"] + 1).split(xd))
    rows, fold_rows = [], []
    for module in modules:
        score, details = cv_score(module["members"], xd, yd, meta, folds)
        rows.append({k: v for k, v in module.items() if k != "members"} | {"cv_vector_improvement": score})
        fold_rows.extend([{**row, "module_id": module["module_id"]} for row in details])
    ranking = pd.DataFrame(rows).sort_values(["cv_vector_improvement", "module_id"], ascending=[False, True])
    ranking.to_csv(out / "tables/discovery_candidates.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "tables/discovery_cv_folds.csv", index=False)
    selected = next(m for m in modules if m["module_id"] == ranking.iloc[0].module_id)
    axis = fit_axis(xd[selected["members"]], meta)
    score, _ = apply_axis(axis, xd)
    beta, intercept = fit_neural(score, yd)
    assert np.isfinite(beta).all() and np.linalg.norm(beta) > 1e-12
    frozen = {"module_id": selected["module_id"], "axis": axis, "neural_beta": beta.to_dict(),
              "neural_intercept": intercept.to_dict(), "neural_direction": (beta / np.linalg.norm(beta)).to_dict(),
              "discovery_ids": discovery, "holdout_ids": holdout,
              "cv_vector_improvement": float(ranking.iloc[0].cv_vector_improvement), "source_manifest": manifest}
    save_json(out / "frozen_candidate.json", frozen)
    counts = {"raw_features": len(audit), "complete_positive_features": int(audit.all_positive_finite.sum()),
              "retained_features": len(x.columns), "n_strains": len(x), "n_discovery": len(discovery), "n_holdout": len(holdout),
              "n_candidates": len(modules), "selected": selected, "cv_vector_improvement": frozen["cv_vector_improvement"]}
    save_json(out / "discovery_summary.json", counts)
    return counts


def residual_against_level(values, level):
    design = np.column_stack([np.ones(len(level)), np.asarray(level)])
    array = np.asarray(values)
    return array - design @ np.linalg.lstsq(design, array, rcond=None)[0]


def run_holdout(root, out):
    out = Path(out)
    assert not (out / "results.json").exists(), "This frozen candidate has already been evaluated."
    frozen_bytes = (out / "frozen_candidate.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    x, y, context, meta, audit, raw_all, pre_gate, manifest = source_inputs(root)
    assert manifest == frozen["source_manifest"]
    train, test = frozen["discovery_ids"], frozen["holdout_ids"]
    assert not set(train) & set(test) and set(train) | set(test) == set(x.index)
    score, z = apply_axis(frozen["axis"], x)
    beta, intercept, direction = [pd.Series(frozen[name]).loc[y.columns] for name in ["neural_beta", "neural_intercept", "neural_direction"]]
    test_beta, _ = fit_neural(score.loc[test], y.loc[test])
    pred = prediction(score, beta, intercept)
    projection = y @ direction
    positions = x.index.get_indexer(test)
    sse = float(np.sum((y.loc[test].to_numpy() - pred[positions]) ** 2))
    baseline_sse = float(np.sum((y.loc[test].to_numpy() - y.loc[train].mean().to_numpy()) ** 2))
    samples = context.copy()
    samples["split"] = np.where(samples.index.isin(train), "discovery", "holdout")
    samples["chemical_score"], samples["neural_projection"] = score, projection
    samples["predicted_projection"] = pred @ direction
    samples["panel_mean_log2"] = x.mean(axis=1)
    samples.to_csv(out / "tables/sample_scores.csv")
    z.to_csv(out / "tables/selected_chemical_standardized.csv")
    pd.DataFrame({"discovery": beta, "holdout": test_beta}).rename_axis("cell").to_csv(out / "tables/neural_slopes.csv")
    members = meta.loc[frozen["axis"]["members"]].copy()
    members["effective_weight"] = pd.Series(frozen["axis"]["effective_weight"])
    members["orientation"] = np.sign(members.effective_weight)
    for name, ids in [("discovery", train), ("holdout", test)]:
        members[name + "_rho_to_score"] = [spearmanr(z.loc[ids, member], score.loc[ids]).statistic for member in members.index]
    members.rename_axis("metabolite").to_csv(out / "tables/selected_members.csv")
    chemistry_correlations = {}
    signs = np.sign(pd.Series(frozen["axis"]["effective_weight"]).loc[members.index].to_numpy())
    for name, ids in [("discovery", train), ("holdout", test)]:
        matrix = x.loc[ids, members.index].corr(method="spearman")
        matrix.to_csv(out / f"tables/selected_chemistry_{name}_correlations.csv")
        aligned = matrix.to_numpy() * np.outer(signs, signs)
        upper = np.triu_indices(len(members), 1)
        chemistry_correlations[name] = {"median_signed_aligned_rho": float(np.median(aligned[upper])),
                                        "minimum_signed_aligned_rho": float(np.min(aligned[upper]))}
    genus = context.loc[test, "genus"]
    eligible = genus.index[genus.map(genus.value_counts()).ge(2)]
    gx = score.loc[eligible] - score.loc[eligible].groupby(genus.loc[eligible]).transform("mean")
    gy = projection.loc[eligible] - projection.loc[eligible].groupby(genus.loc[eligible]).transform("mean")
    genus_rows = []
    for label, ids in genus.groupby(genus).groups.items():
        if len(ids) < 2:
            continue
        if np.std(score.loc[ids]) <= 1e-12:
            local_beta = pd.Series(np.nan, index=y.columns)
        else:
            local_beta, _ = fit_neural(score.loc[ids], y.loc[ids])
        genus_rows.append({"genus": label, "n": len(ids), "chemical_range": float(score.loc[ids].max() - score.loc[ids].min()),
                           "projection_slope": float(local_beta @ direction), "slope_cosine": cosine(beta, local_beta)})
    pd.DataFrame(genus_rows).to_csv(out / "tables/holdout_genus_summary.csv", index=False)
    leave_genus = [{"excluded_genus": label, "n": int(genus.ne(label).sum()),
                    "correlation": corr(score.loc[genus.index[genus.ne(label)]], projection.loc[genus.index[genus.ne(label)]])}
                   for label in sorted(genus.unique())]
    pd.DataFrame(leave_genus).to_csv(out / "tables/leave_one_genus_out.csv", index=False)
    pg_summary = {"status": "not computable", "chemical_axis_fixed": True, "neural_direction_refit_on_discovery_only": True}
    if np.isfinite(pre_gate.to_numpy()).all():
        pg_beta, pg_intercept = fit_neural(score.loc[train], pre_gate.loc[train])
        pg_test_beta, _ = fit_neural(score.loc[test], pre_gate.loc[test])
        if np.linalg.norm(pg_beta) > 1e-12:
            pg_direction = pg_beta / np.linalg.norm(pg_beta)
            pg_sse = np.sum((pre_gate.loc[test].to_numpy() - prediction(score.loc[test], pg_beta, pg_intercept)) ** 2)
            pg_baseline = np.sum((pre_gate.loc[test].to_numpy() - pre_gate.loc[train].mean().to_numpy()) ** 2)
            pg_summary.update({"status": "computed", "holdout_r": corr(score.loc[test], pre_gate.loc[test] @ pg_direction),
                               "holdout_vector_improvement": 1 - pg_sse / pg_baseline if pg_baseline > 1e-12 else np.nan,
                               "holdout_slope_cosine": cosine(pg_beta, pg_test_beta)})
    panel = x.mean(axis=1)
    result = {
        "selected_module": frozen["module_id"], "n_discovery": len(train), "n_holdout": len(test),
        "n_members": len(members), "n_families": int(members.family.nunique()),
        "discovery_cv_vector_improvement": frozen["cv_vector_improvement"],
        "discovery_r": corr(score.loc[train], projection.loc[train]),
        "primary_holdout_r": corr(score.loc[test], projection.loc[test]),
        "holdout_vector_improvement": 1 - sse / baseline_sse,
        "holdout_model_sse": sse, "holdout_mean_baseline_sse": baseline_sse,
        "holdout_slope_cosine": cosine(beta, test_beta),
        "holdout_projection_slope_ratio": float(test_beta @ direction / np.linalg.norm(beta)),
        "chemical_covariation": chemistry_correlations,
        "within_genus_r": corr(gx, gy), "within_genus_n": len(eligible),
        "within_genus_genera": int(genus.loc[eligible].nunique()),
        "positive_holdout_genera": int(sum(row["projection_slope"] > 0 for row in genus_rows)),
        "within_genus_residual_df": len(eligible) - int(genus.loc[eligible].nunique()),
        "holdout_chemical_score_vs_panel_level_r": corr(score.loc[test], panel.loc[test]),
        "holdout_r_after_panel_level_residualization": corr(residual_against_level(score.loc[test], panel.loc[test]),
                                                             residual_against_level(projection.loc[test], panel.loc[test])),
        "pre_gate_sensitivity": pg_summary,
        "frozen_candidate_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "limitations": PARAMETERS["limitations"],
    }
    assert (out / "frozen_candidate.json").read_bytes() == frozen_bytes
    save_json(out / "results.json", result)
    return result
