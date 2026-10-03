"""One bounded comparison of reduced neural inputs for the frozen chemical score.

Call ``run_comparison(report, output)`` from a notebook. The only candidates are
all 13 existing unit coefficients and training-selected subsets of 3 or 5. All
selection and input scaling are fitted within each training CV fold. This does
not redo upstream chemical-module selection or the 13-dimensional normalization.
"""

from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _rank_neurons(x, y, names):
    """Absolute training Pearson correlation; constant inputs last, stable ties."""
    xc, yc = x - x.mean(axis=0), y - y.mean()
    xss, yss = np.square(xc).sum(axis=0), float(np.square(yc).sum())
    constant = xss == 0
    rho = np.divide(xc.T @ yc, np.sqrt(xss * yss),
                    out=np.zeros(x.shape[1]), where=(xss * yss) > 0)
    order = sorted(range(len(names)), key=lambda j: (bool(constant[j]), -abs(rho[j]), j))
    rows = [{"rank": rank, "neuron": names[j], "original_column": j,
             "pearson_r": float(rho[j]), "absolute_pearson_r": float(abs(rho[j])),
             "zero_variance": bool(constant[j])}
            for rank, j in enumerate(order, 1)]
    return np.asarray(order), rows


def _fit_model(x, y, selected, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x[:, selected], y)
    return model


def _freeze_model(model, selected, names, alpha):
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
    return {"neurons": [names[j] for j in selected], "original_column_indices": selected.tolist(),
            "alpha": float(alpha), "input_means": scaler.mean_.tolist(),
            "input_scales": scaler.scale_.tolist(),
            "coefficients_standardized_inputs": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_)}


def run_comparison(report: Path, output: Path) -> dict:
    """Write protocol, fit two training-selected models, then compare 36 strains.

    Inputs are the already frozen 70/36 split, 106 x 13 unit coefficients and
    one 106-vector chemical score. Saved scores retain their original discovery
    scaling. Subsetting never recalculates the neural L2 norm. Bootstrap intervals
    condition on the fixed target, models, split and upstream selection decisions.
    Existing output directories are never overwritten.
    """
    report, output = Path(report), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing results are preserved.")
    paths = [report / "frozen_candidate.json", report / "tables/sample_scores.csv",
             report / "tables/neural_unit_coefficients.csv",
             report / "reverse_prediction_20261003/heldout_predictions.csv",
             report / "reverse_prediction_20261003/frozen_reverse_model.json"]
    hashes = {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen = json.loads(paths[0].read_text())
    scores, neural = [pd.read_csv(p, index_col=0) for p in paths[1:3]]
    train, held = frozen["discovery_ids"], frozen["holdout_ids"]
    names = neural.columns.tolist()
    alphas, ks = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], [3, 5]
    assert len(train) == 70 and len(held) == 36 and not set(train).intersection(held)
    assert neural.shape == (106, 13) and scores.shape[0] == 106
    assert not neural.index.has_duplicates and not scores.index.has_duplicates
    assert set(train + held) == set(neural.index) == set(scores.index)
    assert np.isfinite(neural.to_numpy()).all() and np.isfinite(scores.chemical_score).all()
    assert scores.loc[train, "split"].eq("discovery").all()
    assert scores.loc[held, "split"].eq("holdout").all()
    xtrain = neural.loc[train].to_numpy()
    ytrain = scores.loc[train, "chemical_score"].to_numpy()
    limitations = [
        "The chemical module was already selected using the 70 training strains' neural data; the score was defined in those strains. CV tunes only the present input-selection and Ridge settings, not the whole pipeline.",
        "The same 36 held-out strains have previously been examined; this is an internal exploratory comparison, not independent validation.",
        "Paired bootstrap holds targets, models, training and selection fixed. It treats strains as exchangeable and omits training/selection uncertainty and shared genus/animal dependence.",
        "Selected inputs retain the existing coefficients normalized by the L2 norm of all 13 neurons. This is not evidence that measuring only the selected neurons is sufficient.",
        "Excluded neurons are not established to be biologically useless; feature ranking is conditional on this target, representation and dataset.",
        "The five overlapping training folds describe subset stability, not five independent validations.",
        "Random strain splitting permits genera shared between training and holdout; transfer to unseen genera is untested.",
        "Chemistry was measured in independently cultured material, not the neural-stimulus aliquot; prediction does not establish causation.",
    ]
    protocol = {
        "source_sha256": hashes, "training_ids": train, "heldout_ids": held,
        "target": "Previously frozen chemical_score only; no new target fitting",
        "inputs": names, "reduced_k_grid": ks, "ridge_alpha_grid": alphas,
        "selector": "Descending absolute training Pearson correlation; zero-variance inputs last; ties preserve original input-column order",
        "model": "StandardScaler + Ridge with fitted intercept; selected coefficients keep their original 13-dimensional L2 normalization",
        "cv": {"method": "KFold", "n_splits": 5, "shuffle": True, "seed": 20261004,
               "preprocessing": "Refit neuron ranking and StandardScaler using each fold's training data only",
               "selection": "Minimum pooled validation SSE; all13 tunes alpha; reduced jointly tunes k and alpha",
               "exact_ties": "Prefer smaller k, then smaller alpha"},
        "holdout_evaluation": "Evaluate only the one training-selected reduced model and the all13 model after freezing both; do not evaluate other candidates",
        "baseline": "Fixed mean chemical_score across the 70 training strains",
        "bootstrap": {"replicates": 5000, "seed": 20261005, "unit": "held-out strain",
                      "paired": True, "interval": "2.5th and 97.5th percentiles; conditional only"},
        "stability": "Selected k only: sets in the five training folds, selection counts and mean pairwise Jaccard",
        "limitations": limitations,
    }
    output.mkdir(parents=True)
    # Persist the bounded search before fitting or evaluating any candidate.
    _write_json(output / "protocol.json", protocol)

    folds = list(KFold(n_splits=5, shuffle=True, random_state=20261004).split(xtrain))
    fold_orders, fold_ranking_rows, membership_rows = [], [], []
    for fold, (fit, validate) in enumerate(folds, 1):
        order, ranking_rows = _rank_neurons(xtrain[fit], ytrain[fit], names)
        fold_orders.append(order)
        fold_ranking_rows.extend([{"fold": fold, **row} for row in ranking_rows])
        membership_rows.extend({"fold": fold, "strain": train[j],
                                "role": "fit" if j in set(fit) else "validate"}
                               for j in range(len(train)))
    cv_rows = []
    for kind, k in [("all13", 13)] + [("reduced", k) for k in ks]:
        for alpha in alphas:
            for fold, ((fit, validate), order) in enumerate(zip(folds, fold_orders), 1):
                selected = np.arange(13) if kind == "all13" else order[:k]
                model = _fit_model(xtrain[fit], ytrain[fit], selected, alpha)
                prediction = model.predict(xtrain[validate][:, selected])
                cv_rows.append({"model": kind, "k": k, "alpha": alpha, "fold": fold,
                                "n_validation": len(validate),
                                "validation_sse": float(np.square(prediction - ytrain[validate]).sum()),
                                "selected_neurons": "|".join(names[j] for j in selected)})
    cv = pd.DataFrame(cv_rows)
    pooled = cv.groupby(["model", "k", "alpha"], as_index=False).validation_sse.sum()
    choices = {}
    for kind in ("all13", "reduced"):
        best = pooled.loc[pooled.model.eq(kind)].sort_values(
            ["validation_sse", "k", "alpha"], kind="stable").iloc[0]
        choices[kind] = {"k": int(best.k), "alpha": float(best.alpha),
                         "pooled_training_cv_sse": float(best.validation_sse)}
    order, final_ranking = _rank_neurons(xtrain, ytrain, names)
    k = choices["reduced"]["k"]
    selected = {"all13": np.arange(13), "reduced": order[:k]}
    models = {kind: _fit_model(xtrain, ytrain, selected[kind], choices[kind]["alpha"])
              for kind in choices}
    frozen_models = {
        "target": "chemical_score", "training_ids": train, "heldout_ids": held,
        "training_target_mean": float(ytrain.mean()), "source_sha256": hashes,
        "models": {kind: {**_freeze_model(models[kind], selected[kind], names, choices[kind]["alpha"]),
                           "pooled_training_cv_sse": choices[kind]["pooled_training_cv_sse"]}
                   for kind in choices},
    }
    # Freeze fitted models and all training choices before using held-out targets.
    _write_json(output / "frozen_models.json", frozen_models)
    cv.to_csv(output / "training_cv.csv", index=False)
    pooled.to_csv(output / "training_cv_pooled.csv", index=False)
    pd.DataFrame(final_ranking).to_csv(output / "training_neuron_ranking.csv", index=False)
    pd.DataFrame(fold_ranking_rows).to_csv(output / "training_fold_neuron_rankings.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(output / "training_fold_membership.csv", index=False)

    fold_sets = [[names[j] for j in fold_order[:k]] for fold_order in fold_orders]
    frequencies = {name: sum(name in fold_set for fold_set in fold_sets) for name in names}
    pairwise_jaccard = [len(set(a) & set(b)) / len(set(a) | set(b))
                        for a, b in combinations(fold_sets, 2)]
    stability = {"selected_k": k, "fold_sets": fold_sets, "frequencies": frequencies,
                 "mean_pairwise_jaccard": float(np.mean(pairwise_jaccard)),
                 "pairwise_jaccard": pairwise_jaccard,
                 "interpretation": "Overlapping-fold subset stability; not independent replication"}
    stability_rows = [{"neuron": name, "n_folds_selected": frequencies[name],
                       "fraction_folds_selected": frequencies[name] / len(folds),
                       "selected_on_all70": name in frozen_models["models"]["reduced"]["neurons"]}
                      for name in names]
    pd.DataFrame(stability_rows).to_csv(output / "selection_stability.csv", index=False)

    # First and only evaluation of the two selected models on held-out targets.
    xheld = neural.loc[held].to_numpy()
    observed = scores.loc[held, "chemical_score"].to_numpy()
    predictions = {kind: models[kind].predict(xheld[:, selected[kind]]) for kind in models}
    for kind, specification in frozen_models["models"].items():
        reconstructed = ((xheld[:, selected[kind]] - np.asarray(specification["input_means"]))
                         / np.asarray(specification["input_scales"])) @ np.asarray(
                             specification["coefficients_standardized_inputs"]) + specification["intercept"]
        np.testing.assert_allclose(predictions[kind], reconstructed, atol=1e-12, rtol=0)
    old_model = json.loads(paths[4].read_text())
    old_predictions = pd.read_csv(paths[3])
    old_primary = old_predictions.loc[old_predictions.target.eq("chemical_score")].set_index("strain").loc[held]
    assert choices["all13"]["alpha"] == old_model["selected_alpha"] == 100.0
    assert train == old_model["training_ids"] and held == old_model["heldout_ids"]
    assert names == old_model["inputs"]
    np.testing.assert_allclose(predictions["all13"], old_primary.predicted, atol=1e-12, rtol=0)
    np.testing.assert_allclose(observed, old_primary.observed, atol=1e-12, rtol=0)
    baseline = float(ytrain.mean())
    np.testing.assert_allclose(baseline, old_primary.training_mean_baseline, atol=1e-12, rtol=0)
    errors = {kind: np.square(prediction - observed) for kind, prediction in predictions.items()}
    base_errors = np.square(baseline - observed)
    base_sse = float(base_errors.sum())
    resamples = np.random.default_rng(20261005).integers(0, len(held), size=(5000, len(held)))
    bootstrap_base_sse = base_errors[resamples].sum(axis=1)
    bootstrap_sse = {kind: error[resamples].sum(axis=1) for kind, error in errors.items()}
    metrics = {}
    for kind, error in errors.items():
        sse = float(error.sum())
        metrics[kind] = {
            "sse": sse, "rmse": float(np.sqrt(sse / len(held))),
            "pearson_r": float(np.corrcoef(observed, predictions[kind])[0, 1]),
            "relative_error_reduction": 1 - sse / base_sse,
            "conditional_bootstrap_95pct_interval": np.quantile(
                1 - bootstrap_sse[kind] / bootstrap_base_sse, [.025, .975]).tolist(),
        }
    direct_gain = 1 - metrics["reduced"]["sse"] / metrics["all13"]["sse"]
    direct_bootstrap = 1 - bootstrap_sse["reduced"] / bootstrap_sse["all13"]
    sse_difference_bootstrap = bootstrap_sse["reduced"] - bootstrap_sse["all13"]
    improvement_difference = 100 * (metrics["reduced"]["relative_error_reduction"]
                                  - metrics["all13"]["relative_error_reduction"])
    comparison = {
        "relative_error_reduction_vs_all13": direct_gain,
        "conditional_bootstrap_95pct_interval": np.quantile(direct_bootstrap, [.025, .975]).tolist(),
        "improvement_percentage_point_difference": improvement_difference,
        "improvement_percentage_point_difference_conditional_bootstrap_95pct_interval": np.quantile(
            -100 * sse_difference_bootstrap / bootstrap_base_sse, [.025, .975]).tolist(),
        "sse_difference_reduced_minus_all13": metrics["reduced"]["sse"] - metrics["all13"]["sse"],
        "sse_difference_conditional_bootstrap_95pct_interval": np.quantile(
            sse_difference_bootstrap, [.025, .975]).tolist(),
        "n_strains_lower_error": int(np.sum(errors["reduced"] < errors["all13"])),
        "n_strains_equal_error": int(np.sum(errors["reduced"] == errors["all13"])),
    }
    rows = [{"strain": strain, "genus": scores.loc[strain, "genus"],
             "observed": observed[i], "baseline": baseline,
             "all13": predictions["all13"][i], "reduced": predictions["reduced"][i],
             "error_baseline": base_errors[i], "error_all13": errors["all13"][i],
             "error_reduced": errors["reduced"][i]}
            for i, strain in enumerate(held)]
    pd.DataFrame(rows).to_csv(output / "heldout_predictions.csv", index=False)
    assert hashes == {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    verification = {
        "source_files_unchanged": True, "all13_alpha_reproduced": True,
        "all13_prediction_max_abs_difference": float(np.max(np.abs(predictions["all13"] - old_primary.predicted))),
        "all13_prediction_tolerance": 1e-12,
        "frozen_model_predictions_reconstructed": True,
        "n_evaluated_holdout_models": 2, "n_training_cv_candidates": int(pooled.shape[0]),
        "protocol_saved_before_training": True, "models_saved_before_holdout_evaluation": True,
    }
    result = {
        "n_train": len(train), "n_holdout": len(held), "selected_k": k,
        "selected_neurons": frozen_models["models"]["reduced"]["neurons"],
        "all13_alpha": choices["all13"]["alpha"], "reduced_alpha": choices["reduced"]["alpha"],
        "training_mean_baseline": baseline, "baseline_sse": base_sse,
        "baseline_rmse": float(np.sqrt(base_sse / len(held))),
        "models": metrics, "comparison": comparison, "selection_stability": stability,
        "verification": verification, "limitations": limitations,
    }
    _write_json(output / "verification.json", verification)
    _write_json(output / "results.json", result)
    return result
