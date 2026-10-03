"""Bounded linear, interaction and additive-curvature comparison for a fixed score.

Use ``run_comparison(report, output)`` in a notebook. The historical k=5 and
chemical target stay fixed; each CV fit fold repeats neuron ranking and learns
all scalers/expansions from its own training rows. No raw data are modified.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _rank_neurons(x, y, names, k=5):
    """Rank marginal absolute Pearson r, breaking exact ties by input order."""
    xc, yc = x - x.mean(axis=0), y - y.mean()
    # Range detects exact constant columns even if the floating-point mean has
    # a rounding residual, which otherwise creates a spurious nonzero variance.
    constant = np.ptp(x, axis=0) == 0
    xc[:, constant] = 0
    xss, yss = np.square(xc).sum(axis=0), float(np.square(yc).sum())
    if np.count_nonzero(~constant) < k:
        raise ValueError(f"Fewer than {k} variable training inputs; fixed protocol cannot continue.")
    rho = np.divide(xc.T @ yc, np.sqrt(xss * yss),
                    out=np.zeros(x.shape[1]), where=(xss * yss) > 0)
    order = sorted(range(len(names)), key=lambda j: (bool(constant[j]), -abs(rho[j]), j))
    rows = [{"rank": rank, "neuron": names[j], "original_column": j,
             "pearson_r": float(rho[j]), "absolute_pearson_r": float(abs(rho[j])),
             "zero_variance": bool(constant[j])}
            for rank, j in enumerate(order, 1)]
    return np.asarray(order), rows


def _fit_model(x, y, family, alpha):
    steps = [("input_scaler", StandardScaler())]
    if family == "interaction":
        steps += [("expansion", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                  ("expanded_scaler", StandardScaler())]
    elif family == "curvature":
        steps += [("expansion", SplineTransformer(n_knots=3, degree=2, knots="uniform",
                                                 extrapolation="linear", include_bias=False)),
                  ("expanded_scaler", StandardScaler())]
    elif family != "linear":
        raise ValueError(f"Unknown family: {family}")
    model = Pipeline(steps + [("ridge", Ridge(alpha=alpha))])
    model.fit(x, y)
    return model


def _scaler_state(scaler):
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "variance": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_),
            "with_mean": True, "with_std": True}


def _freeze_model(model, family, selected, names, alpha):
    steps, ridge = model.named_steps, model.named_steps["ridge"]
    specification = {
        "family": family, "neurons": [names[j] for j in selected],
        "original_column_indices": selected.tolist(), "selected_alpha": float(alpha),
        "input_scaler": _scaler_state(steps["input_scaler"]),
        "n_transformed_features": int(ridge.n_features_in_),
        "ridge_coefficients": ridge.coef_.tolist(), "ridge_intercept": float(ridge.intercept_),
        "ridge_fit_intercept": True,
    }
    if family != "linear":
        specification["expanded_scaler"] = _scaler_state(steps["expanded_scaler"])
        expansion = steps["expansion"]
        specification["expanded_feature_names"] = expansion.get_feature_names_out(
            specification["neurons"]).tolist()
        if family == "interaction":
            specification["expansion"] = {
                "class": "PolynomialFeatures", "degree": 2, "interaction_only": True,
                "include_bias": False, "powers": expansion.powers_.tolist(),
            }
        else:
            specification["expansion"] = {
                "class": "SplineTransformer", "n_knots": 3, "degree": 2,
                "knots": "uniform", "include_bias": False, "extrapolation": "linear",
                "basis_order": "Input order; discard the last basis per input",
                "linear_extrapolation": "Outside each training boundary use its basis value plus boundary derivative times distance",
                "bsplines": [{"t": b.t.tolist(), "c": b.c.tolist(), "k": int(b.k),
                               "axis": int(b.axis), "extrapolate": bool(b.extrapolate)}
                              for b in expansion.bsplines_],
            }
    return specification


def _frozen_features(x_selected, specification):
    """Rebuild features from JSON using NumPy/SciPy, not sklearn transformers."""
    scaler = specification["input_scaler"]
    z = (x_selected - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    family = specification["family"]
    if family == "linear":
        return z
    expansion = specification["expansion"]
    if family == "interaction":
        powers = np.asarray(expansion["powers"])
        z = np.prod(z[:, None, :] ** powers[None, :, :], axis=2)
    else:
        blocks = []
        for j, saved in enumerate(expansion["bsplines"]):
            spline = BSpline(np.asarray(saved["t"]), np.asarray(saved["c"]), saved["k"],
                             extrapolate=saved["extrapolate"], axis=saved["axis"])
            left, right = spline.t[spline.k], spline.t[-spline.k - 1]
            x = z[:, j]
            values = spline(np.clip(x, left, right))
            below, above = x < left, x > right
            values[below] = spline(left) + (x[below] - left)[:, None] * spline(left, nu=1)
            values[above] = spline(right) + (x[above] - right)[:, None] * spline(right, nu=1)
            blocks.append(values[:, :-1])
        z = np.concatenate(blocks, axis=1)
    scaler = specification["expanded_scaler"]
    return (z - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])


def run_comparison(report: Path, output: Path) -> dict:
    """Fit the pre-specified 18 alpha/family candidates and evaluate three models.

    Outputs are CSV/JSON audit files only. Holdout metrics do not choose a model.
    Bootstrap intervals condition on the previously selected target, historical
    k, split, final input selection and fitted models. Existing output is refused.
    """
    report, output = Path(report), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing results are preserved.")
    paths = [report / "frozen_candidate.json", report / "tables/sample_scores.csv",
             report / "tables/neural_unit_coefficients.csv",
             report / "reduced_input_comparison_20261003/results.json",
             report / "reduced_input_comparison_20261003/heldout_predictions.csv",
             report / "reduced_input_comparison_20261003/frozen_models.json"]
    hashes = {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen = json.loads(paths[0].read_text())
    scores, neural = [pd.read_csv(p, index_col=0) for p in paths[1:3]]
    previous_result = json.loads(paths[3].read_text())
    train, held = frozen["discovery_ids"], frozen["holdout_ids"]
    names, k = neural.columns.tolist(), 5
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    families = ["linear", "interaction", "curvature"]
    assert previous_result["selected_k"] == k
    assert len(train) == 70 and len(held) == 36 and not set(train).intersection(held)
    assert neural.shape == (106, 13) and scores.shape[0] == 106
    assert not neural.index.has_duplicates and not scores.index.has_duplicates
    assert set(train + held) == set(neural.index) == set(scores.index)
    assert np.isfinite(neural.to_numpy()).all() and np.isfinite(scores.chemical_score).all()
    assert scores.loc[train, "split"].eq("discovery").all() and scores.loc[held, "split"].eq("holdout").all()
    xtrain, ytrain = neural.loc[train].to_numpy(), scores.loc[train, "chemical_score"].to_numpy()
    limitations = [
        "The chemical module, score and k=5 were selected in previous analyses. This CV tunes only current family-specific alpha values and is not an unbiased evaluation of the whole workflow.",
        "The same 36 held-out strains have been examined repeatedly; this is internal exploration, not independent validation. Holdout performance does not select the training_chosen_family.",
        "Two exploratory extension-versus-linear comparisons are reported; no formal significance or mechanism claim is made.",
        "Paired fixed-prediction bootstrap treats strains as exchangeable and omits training/selection uncertainty and shared genus/animal dependence.",
        "Marginal Pearson screening may discard variables informative only through interactions. These restricted models cannot rule out neural nonlinearity in general.",
        "Feature expansion also changes the structure of the Ridge penalty; a performance change alone does not establish a nonlinear mechanism.",
        "The curvature model is additive quadratic B-spline basis expansion with Ridge shrinkage, not a GAM with an explicit roughness penalty. The interaction model includes main effects and pairwise products, without squared terms.",
        "Selected inputs retain coefficients normalized by the original 13-neuron L2 norm; this does not show that measuring only five neurons is sufficient or that omitted neurons are biologically useless.",
        "Random strain splitting permits genera shared across training and holdout; unseen-genus transfer remains untested.",
        "Chemical and neural measurements come from independently cultured materials. Statistical interaction is not evidence of a neuronal mechanism or causal effect.",
    ]
    protocol = {
        "source_sha256": hashes, "sklearn_version": sklearn.__version__,
        "training_ids": train, "heldout_ids": held, "target": "Frozen chemical_score only",
        "inputs": names, "selected_k": k, "k_origin": "Previous training-selected reduced model; not retuned",
        "selector": "Absolute Pearson correlation on each training fold; zero variance last; stable original-column ties; require >=5 variable inputs",
        "families": {
            "linear": "Select5 -> StandardScaler -> Ridge",
            "interaction": "Select5 -> StandardScaler -> PolynomialFeatures(degree=2,interaction_only=True,include_bias=False) -> StandardScaler -> Ridge; 15 features",
            "curvature": "Select5 -> StandardScaler -> SplineTransformer(n_knots=3,degree=2,knots=uniform,extrapolation=linear,include_bias=False) -> StandardScaler -> Ridge; 15 features",
        },
        "ridge_alpha_grid": alphas, "cv": {"method": "KFold", "n_splits": 5, "shuffle": True,
            "seed": 20261004, "selection": "Each family minimizes pooled validation SSE; exact ties select smaller alpha",
            "preprocessing": "Selector, input scaler, expansion and expanded scaler fitted on the current fold's fit rows only"},
        "family_selection": "Lowest training CV SSE; exact ties prefer linear, interaction, curvature in that order",
        "holdout_evaluation": "Freeze all three training-winner models before predicting the 36 strains; no further candidates or heldout winner selection",
        "neural_normalization": "Keep original 13-dimensional unit coefficients; no renormalization after selection",
        "baseline": "The 70-strain training mean chemical score",
        "bootstrap": {"replicates": 5000, "seed": 20261005, "paired": True,
                      "unit": "heldout strain", "interval": "Conditional 2.5th and 97.5th percentiles"},
        "limitations": limitations,
    }
    output.mkdir(parents=True)
    _write_json(output / "protocol.json", protocol)

    folds = list(KFold(n_splits=5, shuffle=True, random_state=20261004).split(xtrain))
    fold_selected, ranking_rows, membership_rows = [], [], []
    for fold, (fit, validate) in enumerate(folds, 1):
        order, rows = _rank_neurons(xtrain[fit], ytrain[fit], names, k)
        fold_selected.append(order[:k])
        ranking_rows += [{"fold": fold, **row} for row in rows]
        fit_set = set(fit)
        membership_rows += [{"fold": fold, "strain": strain,
                             "role": "fit" if j in fit_set else "validate"}
                            for j, strain in enumerate(train)]
    cv_rows, audit_rows = [], []
    for family in families:
        for alpha in alphas:
            for fold, ((fit, validate), selected) in enumerate(zip(folds, fold_selected), 1):
                xfit, xvalidate = xtrain[fit][:, selected], xtrain[validate][:, selected]
                model = _fit_model(xfit, ytrain[fit], family, alpha)
                np.testing.assert_allclose(model.named_steps["input_scaler"].mean_, xfit.mean(axis=0), atol=1e-12)
                assert int(model.named_steps["input_scaler"].n_samples_seen_) == len(fit)
                dimension = int(model.named_steps["ridge"].n_features_in_)
                assert dimension == (5 if family == "linear" else 15)
                if family != "linear":
                    steps = model.named_steps
                    fitted_expansion = steps["expansion"].transform(steps["input_scaler"].transform(xfit))
                    np.testing.assert_allclose(steps["expanded_scaler"].mean_, fitted_expansion.mean(axis=0), atol=1e-12)
                    assert int(steps["expanded_scaler"].n_samples_seen_) == len(fit)
                    if family == "interaction":
                        assert np.max(steps["expansion"].powers_) == 1
                        assert (steps["expansion"].powers_.sum(axis=1) == 1).sum() == 5
                        assert (steps["expansion"].powers_.sum(axis=1) == 2).sum() == 10
                    else:
                        zfit = steps["input_scaler"].transform(xfit)
                        for j, spline in enumerate(steps["expansion"].bsplines_):
                            np.testing.assert_allclose([spline.t[spline.k], spline.t[-spline.k-1]],
                                                       [zfit[:, j].min(), zfit[:, j].max()], atol=1e-12)
                prediction = model.predict(xvalidate)
                cv_rows.append({"family": family, "alpha": alpha, "fold": fold,
                                "n_validation": len(validate), "n_transformed_features": dimension,
                                "validation_sse": float(np.square(prediction - ytrain[validate]).sum()),
                                "selected_neurons": "|".join(names[j] for j in selected)})
                audit_rows.append({"family": family, "alpha": alpha, "fold": fold,
                                   "n_fit": len(fit), "n_transformed_features": dimension,
                                   "input_scaler_fit_rows_verified": True,
                                   "expansion_scaler_and_knots_fit_rows_verified": True})
    cv = pd.DataFrame(cv_rows)
    pooled = cv.groupby(["family", "alpha"], as_index=False).validation_sse.sum()
    choices = {}
    for family in families:
        best = pooled.loc[pooled.family.eq(family)].sort_values(["validation_sse", "alpha"], kind="stable").iloc[0]
        choices[family] = {"selected_alpha": float(best.alpha), "pooled_training_cv_sse": float(best.validation_sse)}
    training_chosen_family = min(families, key=lambda family: (choices[family]["pooled_training_cv_sse"], families.index(family)))
    order, final_ranking = _rank_neurons(xtrain, ytrain, names, k)
    selected = order[:k]
    selected_names = [names[j] for j in selected]
    assert selected_names == previous_result["selected_neurons"] == ["AWCON", "ADF", "ASH", "AWB", "ASER"]
    models = {family: _fit_model(xtrain[:, selected], ytrain, family, choices[family]["selected_alpha"])
              for family in families}
    frozen_models = {
        "source_sha256": hashes, "training_ids": train, "heldout_ids": held,
        "target": "chemical_score", "training_target_mean": float(ytrain.mean()),
        "selected_k": k, "selected_neurons": selected_names, "training_chosen_family": training_chosen_family,
        "selector": protocol["selector"], "sklearn_version": sklearn.__version__,
        "models": {family: {**_freeze_model(models[family], family, selected, names, choices[family]["selected_alpha"]),
                            "pooled_training_cv_sse": choices[family]["pooled_training_cv_sse"]}
                   for family in families},
    }
    _write_json(output / "frozen_models.json", frozen_models)
    cv.to_csv(output / "training_cv.csv", index=False)
    pooled.to_csv(output / "training_cv_pooled.csv", index=False)
    pd.DataFrame(final_ranking).to_csv(output / "training_neuron_ranking.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(output / "training_fold_neuron_rankings.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(output / "training_fold_membership.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output / "training_preprocessing_checks.csv", index=False)

    # Evaluate exactly the three frozen training-winner models, without adaptation.
    xheld, observed = neural.loc[held].to_numpy()[:, selected], scores.loc[held, "chemical_score"].to_numpy()
    predictions = {family: models[family].predict(xheld) for family in families}
    old_frozen = json.loads(paths[5].read_text())
    old_predictions = pd.read_csv(paths[4], index_col=0).loc[held]
    assert train == old_frozen["training_ids"] and held == old_frozen["heldout_ids"]
    assert choices["linear"]["selected_alpha"] == previous_result["reduced_alpha"] == 10.0
    np.testing.assert_allclose(predictions["linear"], old_predictions.reduced, atol=1e-12, rtol=0)
    np.testing.assert_allclose(observed, old_predictions.observed, atol=1e-12, rtol=0)
    baseline = float(ytrain.mean())
    np.testing.assert_allclose(baseline, old_predictions.baseline, atol=1e-12, rtol=0)

    # Independently rebuild saved feature transforms, including boundary extension,
    # then fit a NumPy closed-form Ridge solution to verify the final predictions.
    independent_checks = {}
    for family in families:
        saved, model = frozen_models["models"][family], models[family]
        ztrain = _frozen_features(xtrain[:, selected], saved)
        zheld = _frozen_features(xheld, saved)
        np.testing.assert_allclose(ztrain, model[:-1].transform(xtrain[:, selected]), atol=1e-12, rtol=0)
        np.testing.assert_allclose(zheld, model[:-1].transform(xheld), atol=1e-12, rtol=0)
        synthetic = np.vstack([xtrain[:, selected].min(axis=0) - .5,
                               xtrain[:, selected].max(axis=0) + .5])
        np.testing.assert_allclose(_frozen_features(synthetic, saved), model[:-1].transform(synthetic), atol=1e-12, rtol=0)
        frozen_prediction = zheld @ np.asarray(saved["ridge_coefficients"]) + saved["ridge_intercept"]
        np.testing.assert_allclose(frozen_prediction, predictions[family], atol=1e-12, rtol=0)
        mean = ztrain.mean(axis=0)
        zc = ztrain - mean
        beta = np.linalg.solve(zc.T @ zc + saved["selected_alpha"] * np.eye(ztrain.shape[1]),
                               zc.T @ (ytrain - ytrain.mean()))
        independent_prediction = (zheld - mean) @ beta + ytrain.mean()
        np.testing.assert_allclose(independent_prediction, predictions[family], atol=1e-12, rtol=0)
        independent_checks[family] = {
            "n_transformed_features": int(ztrain.shape[1]),
            "frozen_transform_reconstructed_train_and_holdout": True,
            "synthetic_out_of_range_transform_reconstructed": True,
            "frozen_prediction_max_abs_difference": float(np.max(np.abs(frozen_prediction - predictions[family]))),
            "independent_numpy_prediction_max_abs_difference": float(np.max(np.abs(independent_prediction - predictions[family]))),
        }
    range_rows = []
    for j, name in enumerate(selected_names):
        lower, upper = float(xtrain[:, selected[j]].min()), float(xtrain[:, selected[j]].max())
        below, above = xheld[:, j] < lower, xheld[:, j] > upper
        range_rows.append({"neuron": name, "training_min": lower, "training_max": upper,
                           "n_heldout_below": int(below.sum()), "n_heldout_above": int(above.sum()),
                           "heldout_outside_ids": "|".join(np.asarray(held)[below | above])})
    pd.DataFrame(range_rows).to_csv(output / "input_range_audit.csv", index=False)

    errors = {family: np.square(prediction - observed) for family, prediction in predictions.items()}
    base_errors, base_sse = np.square(baseline - observed), float(np.square(baseline - observed).sum())
    resamples = np.random.default_rng(20261005).integers(0, len(held), size=(5000, len(held)))
    bootstrap_base = base_errors[resamples].sum(axis=1)
    bootstrap_sse = {family: error[resamples].sum(axis=1) for family, error in errors.items()}
    metrics = {}
    for family, error in errors.items():
        sse = float(error.sum())
        metrics[family] = {
            **choices[family], "n_transformed_features": frozen_models["models"][family]["n_transformed_features"],
            "sse": sse, "rmse": float(np.sqrt(sse / len(held))),
            "pearson_r": float(np.corrcoef(observed, predictions[family])[0, 1]),
            "relative_error_reduction": 1 - sse / base_sse,
            "conditional_bootstrap_95pct_interval": np.quantile(1 - bootstrap_sse[family] / bootstrap_base, [.025, .975]).tolist(),
        }
    comparisons = {}
    for family in ("interaction", "curvature"):
        difference = bootstrap_sse[family] - bootstrap_sse["linear"]
        comparisons[family] = {
            "relative_error_reduction_vs_linear": 1 - metrics[family]["sse"] / metrics["linear"]["sse"],
            "conditional_bootstrap_95pct_interval": np.quantile(1 - bootstrap_sse[family] / bootstrap_sse["linear"], [.025, .975]).tolist(),
            "sse_difference_vs_linear": metrics[family]["sse"] - metrics["linear"]["sse"],
            "sse_difference_conditional_bootstrap_95pct_interval": np.quantile(difference, [.025, .975]).tolist(),
            "improvement_percentage_point_difference": 100 * (metrics[family]["relative_error_reduction"] - metrics["linear"]["relative_error_reduction"]),
            "improvement_percentage_point_difference_conditional_bootstrap_95pct_interval": np.quantile(-100 * difference / bootstrap_base, [.025, .975]).tolist(),
            "n_strains_lower_error": int(np.sum(errors[family] < errors["linear"])),
            "n_strains_equal_error": int(np.sum(errors[family] == errors["linear"])),
        }
    rows = []
    for i, strain in enumerate(held):
        row = {"strain": strain, "genus": scores.loc[strain, "genus"], "observed": observed[i], "baseline": baseline}
        row.update({family: predictions[family][i] for family in families})
        row["error_baseline"] = base_errors[i]
        row.update({"error_" + family: errors[family][i] for family in families})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "heldout_predictions.csv", index=False)
    assert hashes == {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    verification = {
        "source_files_unchanged": True, "linear_alpha_reproduced": True,
        "linear_prediction_max_abs_difference": float(np.max(np.abs(predictions["linear"] - old_predictions.reduced))),
        "absolute_prediction_tolerance": 1e-12,
        "training_cv_candidates": 18, "training_cv_fold_fits": 90, "evaluated_holdout_models": 3,
        "protocol_saved_before_training": True, "models_saved_before_holdout_evaluation": True,
        "fold_input_ranking_and_preprocessing_fitted_only_on_training_rows": True,
        "independent_numpy_and_saved_transform_checks": independent_checks,
    }
    result = {
        "n_train": len(train), "n_holdout": len(held), "selected_k": k, "selected_neurons": selected_names,
        "training_chosen_family": training_chosen_family, "training_mean_baseline": baseline,
        "baseline_sse": base_sse, "baseline_rmse": float(np.sqrt(base_sse / len(held))),
        "models": metrics, "comparisons": comparisons, "input_range_audit": range_rows,
        "limitations": limitations, "verification": verification,
    }
    _write_json(output / "verification.json", verification)
    _write_json(output / "results.json", result)
    return result
