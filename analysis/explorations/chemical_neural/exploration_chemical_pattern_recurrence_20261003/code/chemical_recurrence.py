"""One frozen discovery/holdout search for chemical–population co-change.

Import run_discovery(root, out), then run_holdout(root, out) in a notebook.
The second call evaluates only the candidate frozen by the first call.
Inputs are saved strain summaries; neural template estimation is not refitted.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split


PARAMETERS = {
    "seed": 20261003,
    "holdout_fraction": 1 / 3,
    "split": "one fixed strain split, stratified by reference; no resplitting",
    "chemical_panel": "fixed complete162 log2(original report + 1)",
    "neural_target": "13 signed template coefficients divided by strain L2 norm",
    "date_gate": False,
    "module_discovery": "discovery-only within-reference chemical residuals",
    "linkage": "average",
    "distance": "1 - absolute Spearman correlation",
    "linkage_cut": 0.30,
    "minimum_members": 3,
    "minimum_mass_column_families": 3,
    "minimum_pc1_variance_fraction": 0.50,
    "chemical_axis": "PC1 of discovery-scaled within-reference log values",
    "family_weight": "1/sqrt(number of annotations in the same Mass-column family in the module)",
    "family_caution": "a redundancy sensitivity grouping, not proof of identical molecules",
    "axis_orientation": "largest absolute effective loading is positive",
    "axis_scale": "discovery within-reference score sample SD = 1",
    "selection": "largest 5-fold discovery CV 13-vector SSE improvement over reference-only means",
    "inner_cv": "module membership fixed by outer discovery chemistry; fold scales, axes and neural slopes refitted",
    "holdout_primary": "reference-centered correlation of frozen chemical score and frozen neural direction projection",
    "permutations": 9999,
    "permutation_scheme": "shuffle holdout chemical scores within reference, one-sided positive association",
    "secondary": ["full-vector out-of-sample SSE improvement", "13-vector slope cosine", "within genus-reference association", "reference-specific and leave-one-reference-out association", "pre-gate and separately fitted unfiltered neural sensitivity"],
    "limitations": [
        "internal strain holdout for the chemical-neural mapping only; saved neural templates/SNR use the existing atlas",
        "complete162 eligibility was defined previously using all strains' chemical availability/QC",
        "strains can share recording animals; permutation is an exploratory conditional label benchmark, not independent experimental replication",
        "reference labels are not verified experimental batches",
        "chemical measurements are from independently cultured material, not measured stimulus aliquots",
        "no target-guided retry, split change, threshold tuning or second heldout candidate",
    ],
}


def write_json(path, data):
    def native(value):
        if isinstance(value, dict):
            return {str(k): native(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [native(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        return value
    Path(path).write_text(json.dumps(native(data), indent=2, ensure_ascii=False) + "\n")


def load_inputs(root):
    source = Path(root) / "reports/exploration_matched_pair_context_20261003/tables"
    names = ["chemical_log2_report_plus1.csv", "neural_unit_coefficients.csv",
             "neural_coefficients.csv", "sample_context.csv", "feature_metadata.csv"]
    manifest = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    x, y, coefficients, context = [pd.read_csv(source / name, index_col="strain").sort_index()
                                  for name in names[:4]]
    meta = pd.read_csv(source / names[4], index_col="metabolite").loc[x.columns]
    for frame in [x, y, coefficients, context]:
        assert frame.index.is_unique and frame.index.equals(x.index)
    assert x.shape == (106, 162) and y.shape == (106, 13)
    assert np.isfinite(x.to_numpy()).all() and np.isfinite(y.to_numpy()).all()
    assert np.allclose(np.linalg.norm(y, axis=1), 1)
    assert np.allclose(coefficients.div(np.linalg.norm(coefficients, axis=1), axis=0), y)
    meta["family"] = [str(mass) + " | " + str(column) if pd.notna(mass) else str(name)
                      for name, mass, column in zip(meta.index, meta.Mass, meta.column)]
    return x, y, context, meta, manifest


def centered(frame, groups):
    return frame - frame.groupby(groups).transform("mean")


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else np.nan


def fit_axis(x, groups, meta):
    means = x.groupby(groups).mean()
    residual = centered(x, groups)
    scale = residual.std(ddof=1)
    assert scale.gt(1e-12).all()
    families = meta.loc[x.columns, "family"]
    weight = 1 / np.sqrt(families.map(families.value_counts()).to_numpy(float))
    z = residual.div(scale).to_numpy() * weight
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    loading = vt[0]
    effective = loading * weight
    if effective[np.argmax(np.abs(effective))] < 0:
        loading, effective = -loading, -effective
    score_sd = np.std(z @ loading, ddof=1)
    return {
        "members": list(x.columns), "reference_means": means.to_dict(orient="index"),
        "feature_sd": scale.to_dict(), "effective_weight": dict(zip(x.columns, effective / score_sd)),
        "pc1_variance_fraction": float(singular[0] ** 2 / np.sum(singular ** 2)),
        "score_sd_before_normalizing": float(score_sd),
    }


def apply_axis(axis, x, groups):
    members = axis["members"]
    means = pd.DataFrame.from_dict(axis["reference_means"], orient="index").loc[groups, members]
    means.index = x.index
    z = (x[members] - means).div(pd.Series(axis["feature_sd"]))
    score = z @ pd.Series(axis["effective_weight"])
    return score.rename("chemical_score"), z


def fit_neural(score, y, groups):
    sx, sy = centered(score, groups), centered(y, groups)
    beta = sy.T @ sx / float(sx @ sx)
    intercept = (y - np.outer(score, beta)).groupby(groups).mean()
    return beta, intercept


def predict_neural(score, beta, intercept, groups):
    baseline = intercept.loc[groups].to_numpy()
    return baseline + np.outer(score, beta)


def discover_modules(x, groups, meta):
    residual = centered(x, groups)
    corr = residual.corr(method="spearman")
    distance = np.clip(1 - np.abs(corr.to_numpy()), 0, 1)
    np.fill_diagonal(distance, 0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"),
                      t=PARAMETERS["linkage_cut"], criterion="distance")
    modules = []
    for label in sorted(set(labels)):
        members = sorted(x.columns[labels == label])
        n_families = meta.loc[members, "family"].nunique()
        if len(members) < PARAMETERS["minimum_members"] or n_families < PARAMETERS["minimum_mass_column_families"]:
            continue
        axis = fit_axis(x[members], groups, meta)
        if axis["pc1_variance_fraction"] < PARAMETERS["minimum_pc1_variance_fraction"]:
            continue
        pair_corr = corr.loc[members, members].to_numpy()[np.triu_indices(len(members), 1)]
        modules.append({"members": members, "n_members": len(members), "n_families": n_families,
                        "pc1_variance_fraction": axis["pc1_variance_fraction"],
                        "median_abs_pair_rho": float(np.median(np.abs(pair_corr))),
                        "minimum_abs_pair_rho": float(np.min(np.abs(pair_corr)))})
    modules.sort(key=lambda m: (-len(m["members"]), tuple(m["members"])))
    for i, module in enumerate(modules, 1):
        module["module_id"] = f"M{i:02d}"
    return modules, corr


def cv_candidate(members, x, y, groups, meta, folds):
    squared_error, baseline_error, fold_rows = 0., 0., []
    for fold, (train, test) in enumerate(folds, 1):
        xt, xv, yt, yv = x.iloc[train][members], x.iloc[test][members], y.iloc[train], y.iloc[test]
        gt, gv = groups.iloc[train], groups.iloc[test]
        axis = fit_axis(xt, gt, meta)
        st, _ = apply_axis(axis, xt, gt)
        sv, _ = apply_axis(axis, xv, gv)
        beta, intercept = fit_neural(st, yt, gt)
        pred = predict_neural(sv, beta, intercept, gv)
        base = yt.groupby(gt).mean().loc[gv].to_numpy()
        sse, base_sse = float(np.sum((yv.to_numpy() - pred) ** 2)), float(np.sum((yv.to_numpy() - base) ** 2))
        squared_error += sse
        baseline_error += base_sse
        fold_rows.append({"fold": fold, "n_test": len(test), "model_sse": sse,
                          "reference_sse": base_sse, "improvement": 1 - sse / base_sse})
    return 1 - squared_error / baseline_error, fold_rows


def run_discovery(root, out):
    out = Path(out)
    assert not (out / "frozen_candidate.json").exists(), "Discovery is frozen; use a new directory for a separately declared analysis."
    (out / "tables").mkdir(parents=True, exist_ok=True)
    write_json(out / "parameters.json", PARAMETERS)
    x, y, context, meta, manifest = load_inputs(root)
    train, test = train_test_split(np.arange(len(x)), test_size=PARAMETERS["holdout_fraction"],
                                   random_state=PARAMETERS["seed"], stratify=context.reference)
    discovery, holdout = sorted(x.index[train]), sorted(x.index[test])
    split = context.copy()
    split["split"] = np.where(split.index.isin(discovery), "discovery", "holdout")
    split.to_csv(out / "tables/strain_split.csv")
    split.groupby(["split", "reference", "genus"]).size().rename("n").to_csv(out / "tables/split_coverage.csv")
    write_json(out / "source_manifest.json", manifest)
    xd, yd, groups = x.loc[discovery], y.loc[discovery], context.loc[discovery, "reference"]
    modules, corr = discover_modules(xd, groups, meta)
    corr.to_csv(out / "tables/discovery_chemical_correlations.csv")
    write_json(out / "chemical_modules.json", modules)
    if not modules:
        write_json(out / "results.json", {"status": "no eligible chemical modules", "n_discovery": len(discovery), "n_holdout": len(holdout)})
        return {"status": "no eligible chemical modules"}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=PARAMETERS["seed"] + 1).split(xd, groups))
    audit, cv_rows = [], []
    for module in modules:
        gain, rows = cv_candidate(module["members"], xd, yd, groups, meta, folds)
        audit.append({k: v for k, v in module.items() if k != "members"} | {"cv_vector_improvement": gain})
        cv_rows.extend([{"module_id": module["module_id"], **row} for row in rows])
    audit = pd.DataFrame(audit).sort_values(["cv_vector_improvement", "module_id"], ascending=[False, True])
    audit.to_csv(out / "tables/discovery_candidates.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "tables/discovery_cv_folds.csv", index=False)
    selected = next(m for m in modules if m["module_id"] == audit.iloc[0].module_id)
    axis = fit_axis(xd[selected["members"]], groups, meta)
    score, _ = apply_axis(axis, xd, groups)
    beta, intercept = fit_neural(score, yd, groups)
    assert np.isfinite(beta).all() and np.linalg.norm(beta) > 1e-12, "No finite nonzero neural direction to freeze."
    frozen = {"module_id": selected["module_id"], "axis": axis,
              "neural_beta": beta.to_dict(), "neural_intercept": intercept.to_dict(orient="index"),
              "neural_direction": (beta / np.linalg.norm(beta)).to_dict(),
              "discovery_ids": discovery, "holdout_ids": holdout,
              "cv_vector_improvement": float(audit.iloc[0].cv_vector_improvement), "source_manifest": manifest}
    write_json(out / "frozen_candidate.json", frozen)
    return {"status": "frozen", "n_discovery": len(discovery), "n_holdout": len(holdout),
            "n_candidates": len(modules), "selected": selected, "cv_vector_improvement": frozen["cv_vector_improvement"]}


def permuted_association(score, projection, groups, seed, n_perm):
    sx, sy = centered(score, groups), centered(projection, groups)
    observed = correlation(sx, sy)
    arrays = [np.flatnonzero(np.asarray(groups) == group) for group in pd.unique(groups)]
    rng = np.random.default_rng(seed)
    x, y = sx.to_numpy(), sy.to_numpy()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if not np.isfinite(observed) or denominator <= 1e-12:
        return np.nan, np.nan, np.full(n_perm, np.nan)
    exceed = 0
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = x.copy()
        for index in arrays:
            shuffled[index] = rng.permutation(x[index])
        value = float(shuffled @ y / denominator)
        null[i] = value
        exceed += value >= observed
    return observed, (exceed + 1) / (n_perm + 1), null


def neural_sensitivity(root, frozen, x, context, y_columns):
    path = Path(root) / "reports/exploration_response_profiles_individual_snr_20261002/tables"
    primary = pd.read_csv(path / "condition_metrics.csv")
    sensitivity = pd.read_csv(path / "sensitivity_condition_metrics.csv")
    frames = {
        "pre_gate_same_template": primary.groupby(["strain", "cell"]).raw_coefficient.mean().unstack("cell"),
        "unfiltered_refitted_template": sensitivity.loc[sensitivity.threshold.astype(str).eq("unfiltered")].groupby(["strain", "cell"]).coefficient.mean().unstack("cell"),
    }
    train, test = frozen["discovery_ids"], frozen["holdout_ids"]
    scores, _ = apply_axis(frozen["axis"], x, context.reference)
    rows = []
    for name, coefficients in frames.items():
        coefficients = coefficients.loc[x.index, y_columns]
        assert np.isfinite(coefficients.to_numpy()).all()
        unit = coefficients.div(np.linalg.norm(coefficients, axis=1), axis=0)
        beta, intercept = fit_neural(scores.loc[train], unit.loc[train], context.loc[train, "reference"])
        direction = beta / np.linalg.norm(beta)
        test_beta, _ = fit_neural(scores.loc[test], unit.loc[test], context.loc[test, "reference"])
        prediction = predict_neural(scores.loc[test], beta, intercept, context.loc[test, "reference"])
        base = unit.loc[train].groupby(context.loc[train, "reference"]).mean().loc[context.loc[test, "reference"]].to_numpy()
        gain = 1 - np.sum((unit.loc[test].to_numpy() - prediction) ** 2) / np.sum((unit.loc[test].to_numpy() - base) ** 2)
        r = correlation(centered(scores.loc[test], context.loc[test, "reference"]),
                        centered(unit.loc[test] @ direction, context.loc[test, "reference"]))
        rows.append({"representation": name, "holdout_r": r, "holdout_vector_improvement": gain,
                     "holdout_slope_cosine": cosine(beta, test_beta), "chemical_candidate_refit_or_reselected": False,
                     "neural_direction_refit_on_discovery_only": True})
    return pd.DataFrame(rows)


def run_holdout(root, out):
    out = Path(out)
    assert not (out / "results.json").exists(), "Holdout already evaluated; no adaptive reruns."
    frozen_path = out / "frozen_candidate.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    x, y, context, meta, manifest = load_inputs(root)
    assert manifest == frozen["source_manifest"]
    sensitivity_source = Path(root) / "reports/exploration_response_profiles_individual_snr_20261002/tables"
    write_json(out / "sensitivity_source_manifest.json", {
        name: hashlib.sha256((sensitivity_source / name).read_bytes()).hexdigest()
        for name in ["condition_metrics.csv", "sensitivity_condition_metrics.csv"]})
    train, test = frozen["discovery_ids"], frozen["holdout_ids"]
    assert not set(train) & set(test) and set(train) | set(test) == set(x.index)
    score, z = apply_axis(frozen["axis"], x, context.reference)
    beta = pd.Series(frozen["neural_beta"]).loc[y.columns]
    direction = pd.Series(frozen["neural_direction"]).loc[y.columns]
    intercept = pd.DataFrame.from_dict(frozen["neural_intercept"], orient="index").loc[:, y.columns]
    projection = (y @ direction).rename("neural_projection")
    gt = context.loc[test, "reference"]
    test_beta, _ = fit_neural(score.loc[test], y.loc[test], gt)
    prediction = predict_neural(score, beta, intercept, context.reference)
    baseline = y.loc[train].groupby(context.loc[train, "reference"]).mean().loc[gt].to_numpy()
    test_positions = x.index.get_indexer(test)
    model_sse = float(np.sum((y.loc[test].to_numpy() - prediction[test_positions]) ** 2))
    baseline_sse = float(np.sum((y.loc[test].to_numpy() - baseline) ** 2))
    r, p, null = permuted_association(score.loc[test], projection.loc[test], gt,
                                     PARAMETERS["seed"] + 2, PARAMETERS["permutations"])
    cell_groups = context.loc[test, "reference"] + " | " + context.loc[test, "genus"]
    counts = cell_groups.value_counts()
    eligible_ids = cell_groups.index[cell_groups.map(counts).ge(2)]
    within_r = correlation(centered(score.loc[eligible_ids], cell_groups.loc[eligible_ids]),
                           centered(projection.loc[eligible_ids], cell_groups.loc[eligible_ids]))
    reference_rows = []
    for reference, ids in context.loc[test].groupby("reference").groups.items():
        local_beta, _ = fit_neural(score.loc[ids], y.loc[ids], context.loc[ids, "reference"])
        local_x = score.loc[ids] - score.loc[ids].mean()
        local_y = projection.loc[ids] - projection.loc[ids].mean()
        reference_rows.append({"reference": reference, "n": len(ids),
                               "correlation": correlation(local_x, local_y),
                               "projection_slope": float(local_x @ local_y / (local_x @ local_x)),
                               "neural_slope_cosine": cosine(beta, local_beta)})
    reference_rows = pd.DataFrame(reference_rows)
    genus_rows = []
    for cell, ids in cell_groups.groupby(cell_groups).groups.items():
        if len(ids) < 2:
            continue
        local_x = score.loc[ids] - score.loc[ids].mean()
        local_y = projection.loc[ids] - projection.loc[ids].mean()
        local_beta, _ = fit_neural(score.loc[ids], y.loc[ids], cell_groups.loc[ids])
        genus_rows.append({"cell": cell, "reference": context.loc[ids[0], "reference"],
                           "genus": context.loc[ids[0], "genus"], "n": len(ids),
                           "chemical_range": float(score.loc[ids].max() - score.loc[ids].min()),
                           "projection_slope": float(local_x @ local_y / (local_x @ local_x)),
                           "neural_slope_cosine": cosine(beta, local_beta)})
    leave_ref = []
    for ref in sorted(gt.unique()):
        ids = gt.index[gt.ne(ref)]
        leave_ref.append({"excluded_reference": ref, "n": len(ids),
                          "correlation": correlation(centered(score.loc[ids], gt.loc[ids]), centered(projection.loc[ids], gt.loc[ids]))})
    members = meta.loc[frozen["axis"]["members"]].copy()
    members["effective_weight"] = pd.Series(frozen["axis"]["effective_weight"])
    members["orientation"] = np.sign(members.effective_weight)
    for label, ids in [("discovery", train), ("holdout", test)]:
        zz = centered(z.loc[ids], context.loc[ids, "reference"])
        ss = centered(score.loc[ids], context.loc[ids, "reference"])
        members[label + "_rho_to_score"] = [correlation(rankdata(zz[m]), rankdata(ss)) for m in members.index]
    members.rename_axis("metabolite").to_csv(out / "tables/selected_members.csv")
    z.to_csv(out / "tables/selected_chemical_standardized.csv")
    y.to_csv(out / "tables/neural_unit_coefficients.csv")
    pd.DataFrame({"discovery": beta, "holdout": test_beta}).rename_axis("cell").to_csv(out / "tables/neural_slopes.csv")
    samples = context.copy()
    samples["split"] = np.where(samples.index.isin(train), "discovery", "holdout")
    samples["chemical_score"] = score
    samples["neural_projection"] = projection
    samples["predicted_projection"] = prediction @ direction
    grouping = samples["split"] + " | " + samples.reference
    samples["chemical_within"] = centered(score, grouping)
    samples["projection_within"] = centered(projection, grouping)
    samples.to_csv(out / "tables/sample_scores.csv")
    reference_rows.to_csv(out / "tables/reference_summary.csv", index=False)
    pd.DataFrame(genus_rows).to_csv(out / "tables/holdout_genus_reference_summary.csv", index=False)
    pd.DataFrame(leave_ref).to_csv(out / "tables/leave_one_reference_out.csv", index=False)
    pd.DataFrame({"null_r": null}).to_csv(out / "tables/holdout_permutation_null.csv", index=False)
    sensitivity = neural_sensitivity(root, frozen, x, context, y.columns)
    sensitivity.to_csv(out / "tables/neural_sensitivity.csv", index=False)
    member_corr_train = centered(z.loc[train], context.loc[train, "reference"]).corr(method="spearman")
    member_corr_test = centered(z.loc[test], context.loc[test, "reference"]).corr(method="spearman")
    member_corr_train.to_csv(out / "tables/selected_chemistry_discovery_correlations.csv")
    member_corr_test.to_csv(out / "tables/selected_chemistry_holdout_correlations.csv")
    n_members = len(members)
    upper = np.triu_indices(n_members, 1)
    result = {
        "status": "single frozen candidate evaluated; no reranking",
        "selected_module": frozen["module_id"], "n_discovery": len(train), "n_holdout": len(test),
        "n_members": n_members, "n_families": int(members.family.nunique()),
        "discovery_cv_vector_improvement": frozen["cv_vector_improvement"],
        "primary_holdout_r": r, "primary_permutation_p": p,
        "holdout_vector_improvement": 1 - model_sse / baseline_sse,
        "holdout_model_sse": model_sse, "holdout_reference_sse": baseline_sse,
        "holdout_slope_cosine": cosine(beta, test_beta),
        "holdout_projection_slope_ratio": float(test_beta @ direction / np.linalg.norm(beta)),
        "within_genus_reference_r": within_r,
        "within_genus_reference_n": len(eligible_ids),
        "within_genus_reference_cells": int(counts.ge(2).sum()),
        "within_genus_reference_genera": int(context.loc[eligible_ids, "genus"].nunique()),
        "within_genus_reference_residual_df": int(len(eligible_ids) - counts.ge(2).sum()),
        "discovery_chemical_median_abs_rho": float(np.median(np.abs(member_corr_train.to_numpy()[upper]))),
        "holdout_chemical_median_abs_rho": float(np.median(np.abs(member_corr_test.to_numpy()[upper]))),
        "positive_holdout_references": int(reference_rows.projection_slope.gt(0).sum()),
        "positive_holdout_genus_reference_cells": int(sum(row["projection_slope"] > 0 for row in genus_rows)),
        "frozen_candidate_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "neural_sensitivity": sensitivity.to_dict(orient="records"),
        "limitations": PARAMETERS["limitations"],
    }
    assert frozen_path.read_bytes() == frozen_bytes
    write_json(out / "results.json", result)
    return result


def posthoc_strain_influence(samples):
    """Descriptive influence check added after seeing A118 in the heatmap.

    Neither the candidate nor any model is refitted or reselected.
    """
    holdout = samples.loc[samples.split.eq("holdout")]
    rows = []
    for omitted in holdout.index:
        remaining = holdout.drop(index=omitted)
        r = correlation(centered(remaining.chemical_score, remaining.reference),
                        centered(remaining.neural_projection, remaining.reference))
        rows.append({"omitted_strain": omitted, "n": len(remaining), "within_reference_r": r})
    return pd.DataFrame(rows)
