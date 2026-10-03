"""Notebook-callable one-chemical-axis model; all selection is refit in each fold.

Inputs: saved log2(ng/mL) chemical matrix and unit-length 13-neuron matrix.
Outputs: new report tables/JSON only. No original data or notebook is changed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parents[1]
SOURCE = REPO / "reports/exploration_chemical_pattern_direct_report_20261003/tables"
spec = importlib.util.spec_from_file_location(
    "local_chemical_axes", REPORT / "chemical/code/local_chemical_axes.py"
)
chem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chem)
ZERO = 1e-12


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) <= ZERO or np.std(y) <= ZERO:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fit_neural(y):
    """Center only; orient every PCA loading using its own largest entry."""
    mean = y.mean(axis=0)
    _, singular, loading = np.linalg.svd(y.to_numpy() - mean.to_numpy(), full_matrices=False)
    signs = np.sign(loading[np.arange(len(loading)), np.argmax(np.abs(loading), axis=1)])
    loading *= signs[:, None]
    return mean, loading, singular ** 2 / np.sum(singular ** 2)


def fit_model(x, y, feature_metadata):
    """Fit only the supplied aligned training strains, including selection."""
    if not x.index.equals(y.index):
        raise ValueError("Chemical and neural training rows must be identically ordered")
    mean, loadings, evr = fit_neural(y)
    loading = pd.Series(loadings[0], index=y.columns)
    target = (y - mean) @ loading
    axes = chem.fit_axes(x, feature_metadata)
    candidates = []
    for name, score in axes["train_scores"].items():
        r = correlation(score, target)
        candidates.append({"axis": name, "pearson_r": r, "r_squared": r ** 2,
                           "n_members": len(axes["module_members"][name]),
                           "members": json.dumps(sorted(axes["module_members"][name]))})
    eligible = [v for v in candidates if np.isfinite(v["r_squared"])]
    eligible.sort(key=lambda v: (-v["r_squared"], tuple(json.loads(v["members"]))))
    selected = eligible[0]["axis"] if eligible else None
    intercept, slope = 0.0, 0.0
    if selected is not None:
        score = axes["train_scores"][selected].to_numpy()
        intercept, slope = np.linalg.lstsq(
            np.column_stack([np.ones(len(score)), score]), target.to_numpy(), rcond=None
        )[0]
    return {"mean": mean, "loading": loading, "evr": evr, "target": target,
            "chemical": axes, "candidates": pd.DataFrame(candidates),
            "selected": selected, "intercept": float(intercept), "slope": float(slope)}


def predict_model(x, fitted):
    axes = chem.transform_axes(x, fitted["chemical"])
    if fitted["selected"] is None:
        score = pd.Series(0.0, index=x.index)
    else:
        score = axes[fitted["selected"]]
    component = fitted["intercept"] + fitted["slope"] * score
    pred = pd.DataFrame(
        fitted["mean"].to_numpy()[None, :] + np.outer(component, fitted["loading"]),
        index=x.index, columns=fitted["mean"].index,
    )
    return pred, component, score


def fitted_record(fitted):
    c = fitted["chemical"]
    return {"training_strains": c["train_ids"], "neural_mean": fitted["mean"].to_dict(),
            "neural_pc1_loading": fitted["loading"].to_dict(),
            "neural_variance_ratios": fitted["evr"].tolist(),
            "selected_axis": fitted["selected"], "intercept": fitted["intercept"],
            "slope": fitted["slope"], "chemical_means": c["means"].to_dict(),
            "chemical_sample_sd": c["scales"].to_dict(),
            "excluded_features": c["excluded_features"],
            "all_candidate_members": c["module_members"],
            "all_candidate_weights": {k: v.to_dict() for k, v in c["score_weights"].items()},
            "candidate_correlations": fitted["candidates"].to_dict("records"),
            "chemical_parameters": c["parameters"]}


def run_cross_validation(x, y, metadata, feature_metadata, full_fit):
    rows, folds, vectors, audits = [], [], [], []
    full_members = set(full_fit["chemical"]["module_members"].get(full_fit["selected"], []))
    schemes = {
        "leave_one_strain_out": [(i, [i]) for i in x.index],
        "leave_one_recorded_species_out": list(metadata.groupby("species", sort=True).groups.items()),
    }
    for scheme, groups in schemes.items():
        for omitted, test_ids in groups:
            test_ids = list(test_ids)
            train_ids = x.index[~x.index.isin(test_ids)]
            fitted = fit_model(x.loc[train_ids], y.loc[train_ids], feature_metadata)
            pred, pred_t, test_z = predict_model(x.loc[test_ids], fitted)
            true_t = (y.loc[test_ids] - fitted["mean"]) @ fitted["loading"]
            baseline = y.loc[test_ids] - fitted["mean"]
            error = y.loc[test_ids] - pred
            selected_members = set(fitted["chemical"]["module_members"].get(fitted["selected"], []))
            union = selected_members | full_members
            fold = {"scheme": scheme, "omitted": omitted, "train_n": len(train_ids),
                    "test_n": len(test_ids), "selected_axis_fold_local": fitted["selected"],
                    "n_candidates": len(fitted["candidates"]),
                    "selected_members": json.dumps(sorted(selected_members)),
                    "selected_members_jaccard_full": len(selected_members & full_members) / len(union) if union else 1.0,
                    "pc1_absolute_cosine_full": abs(float(fitted["loading"] @ full_fit["loading"])),
                    "training_pc1_variance_ratio": fitted["evr"][0],
                    "intercept": fitted["intercept"], "slope": fitted["slope"],
                    "train_ids": ";".join(train_ids), "test_ids": ";".join(test_ids)}
            folds.append(fold)
            audits.append({"scheme": scheme, "omitted": omitted, "test_strains": test_ids,
                           **fitted_record(fitted)})
            for strain in test_ids:
                rows.append({"scheme": scheme, "strain": strain, "omitted": omitted,
                             "species": metadata.loc[strain, "species"],
                             "dates": metadata.loc[strain, "dates"],
                             "chemical_axis_score_fold_local": test_z.loc[strain],
                             "neural_pc1_observed_fold_local": true_t.loc[strain],
                             "neural_pc1_predicted_fold_local": pred_t.loc[strain],
                             "vector_sse_model": float((error.loc[strain] ** 2).sum()),
                             "vector_sse_training_mean": float((baseline.loc[strain] ** 2).sum()),
                             "pc1_squared_error_model": float((true_t.loc[strain] - pred_t.loc[strain]) ** 2),
                             "pc1_squared_error_zero": float(true_t.loc[strain] ** 2)})
                for neuron in y.columns:
                    vectors.append({"scheme": scheme, "strain": strain, "neuron": neuron,
                                    "observed": y.loc[strain, neuron],
                                    "predicted": pred.loc[strain, neuron],
                                    "baseline_training_mean": fitted["mean"][neuron]})
    rows = pd.DataFrame(rows)
    summaries = []
    for scheme, part in rows.groupby("scheme", sort=False):
        sums = part[["vector_sse_model", "vector_sse_training_mean",
                     "pc1_squared_error_model", "pc1_squared_error_zero"]].sum()
        summaries.append({"scheme": scheme, "n_predictions": len(part), **sums.to_dict(),
                          "vector_error_improvement": 1 - sums.vector_sse_model / sums.vector_sse_training_mean,
                          "fold_pc1_error_improvement": 1 - sums.pc1_squared_error_model / sums.pc1_squared_error_zero,
                          "n_strains_better_than_mean": int((part.vector_sse_model < part.vector_sse_training_mean).sum())})
    return rows, pd.DataFrame(folds), pd.DataFrame(vectors), pd.DataFrame(summaries), audits


def influence_and_metadata(scores, metadata):
    """Fixed-full-fit axes only. Residual correlations are descriptive."""
    rows = []
    groups = [("strain", i, [i]) for i in scores.index]
    groups += [("recorded_species", k, list(v)) for k, v in metadata.groupby("species").groups.items()]
    for kind, omitted, ids in groups:
        remaining = scores.drop(ids)
        rows.append({"omission_type": kind, "omitted": omitted, "n_remaining": len(remaining),
                     "fixed_axes_pearson_r": correlation(remaining.chemical_axis_score, remaining.neural_PC1)})
    adjustments = []
    for name, column in [("recorded_species", "species"), ("exact_recorded_date_set", "dates")]:
        counts = metadata[column].value_counts()
        repeated_ids = metadata.index[metadata[column].map(counts) >= 2]
        part = scores.loc[repeated_ids, ["chemical_axis_score", "neural_PC1"]]
        labels = metadata.loc[repeated_ids, column]
        residual = part - part.groupby(labels).transform("mean")
        residual = residual.add_prefix(name + "_demeaned_")
        scores = scores.join(residual)
        adjustments.append({"grouping": name, "n_strains_in_repeated_groups": len(part),
                            "n_repeated_groups": labels.nunique(),
                            "residual_dimension": len(part) - labels.nunique(),
                            "fixed_axes_residual_pearson_r": correlation(residual.iloc[:, 0], residual.iloc[:, 1])})
    return pd.DataFrame(rows), pd.DataFrame(adjustments), scores


def run_analysis(report=REPORT):
    """Run the authorized 29-strain analysis and save into the new report only."""
    report = Path(report)
    out = report / "tables"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Analysis tables exist; review them before choosing a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / f for f in ("sample_context.csv", "fresh_chemical_log2.csv",
                                 "neural_unit_coefficients.csv", "fresh_feature_metadata.csv")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata = pd.read_csv(paths[0], index_col="strain", keep_default_na=False)
    metadata = metadata.loc[metadata.genus.eq("Bacteroides")].sort_index()
    x = pd.read_csv(paths[1], index_col=0).loc[metadata.index]
    y = pd.read_csv(paths[2], index_col=0).loc[metadata.index]
    features = pd.read_csv(paths[3], keep_default_na=False)
    assert x.shape == (29, 162) and y.shape == (29, 13)
    assert np.isfinite(x.to_numpy()).all() and np.isfinite(y.to_numpy()).all()
    assert np.allclose(np.linalg.norm(y, axis=1), 1)
    fitted = fit_model(x, y, features)
    pred, pred_t, z = predict_model(x, fitted)
    candidates = fitted["candidates"].copy()
    candidates["selected_full_data"] = candidates.axis.eq(fitted["selected"])
    candidates.sort_values("r_squared", ascending=False).to_csv(out / "full_candidate_associations.csv", index=False)
    scores = metadata.copy()
    scores["chemical_axis_score"] = z
    scores["neural_PC1"] = fitted["target"]
    scores["neural_PC1_predicted_apparent"] = pred_t
    scores["neural_PC1_residual_apparent"] = fitted["target"] - pred_t
    influence, adjustment, scores = influence_and_metadata(scores, metadata)
    scores.to_csv(out / "strain_model_scores.csv", index_label="strain")
    influence.to_csv(out / "fixed_axis_influence.csv", index=False)
    adjustment.to_csv(out / "fixed_axis_metadata_diagnostics.csv", index=False)
    pred.to_csv(out / "full_apparent_predicted_neural_vectors.csv", index_label="strain")
    rows, folds, vectors, summary, audit = run_cross_validation(x, y, metadata, features, fitted)
    rows.to_csv(out / "heldout_strain_errors.csv", index=False)
    folds.to_csv(out / "heldout_fold_selections.csv", index=False)
    vectors.to_csv(out / "heldout_neural_vectors.csv", index=False)
    summary.to_csv(out / "heldout_pooled_performance.csv", index=False)
    (out / "heldout_fitted_parameters.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    (out / "full_fitted_parameters.json").write_text(json.dumps(fitted_record(fitted), indent=2, allow_nan=False))
    members = fitted["chemical"]["module_members"].get(fitted["selected"], [])
    selected = features.set_index("metabolite").loc[members].copy()
    selected["training_log2_mean"] = fitted["chemical"]["means"].loc[members]
    selected["training_log2_sample_sd"] = fitted["chemical"]["scales"].loc[members]
    selected["score_weight"] = fitted["chemical"]["score_weights"].get(fitted["selected"], pd.Series(dtype=float))
    selected.to_csv(out / "selected_chemical_axis_members.csv", index_label="metabolite")
    baseline_sse = float(((y - fitted["mean"]) ** 2).to_numpy().sum())
    apparent_sse = float(((y - pred) ** 2).to_numpy().sum())
    overview = {"n_strains": len(x), "n_recorded_species": metadata.species.nunique(),
                "n_chemical_candidates": len(candidates), "selected_axis": fitted["selected"],
                "selected_members": members, "intercept": fitted["intercept"], "slope": fitted["slope"],
                "full_selected_pearson_r": correlation(z, fitted["target"]),
                "neural_pc1_variance_ratio": float(fitted["evr"][0]),
                "neural_pc2_variance_ratio": float(fitted["evr"][1]),
                "full_apparent_vector_error_improvement": 1 - apparent_sse / baseline_sse,
                "full_baseline_vector_sse": baseline_sse, "full_apparent_vector_sse": apparent_sse,
                "cv_performance": summary.to_dict("records"),
                "metadata_diagnostics": adjustment.to_dict("records")}
    (out / "model_summary.json").write_text(json.dumps(overview, indent=2, allow_nan=False))
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (report / "source_manifest.json").write_text(json.dumps(hashes, indent=2))
    return overview
