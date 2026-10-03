"""Bounded reverse prediction of the fixed chemical pattern from all 13 neurons."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_check(report: Path, output: Path) -> dict:
    """Keep the selected module/split fixed; tune only Ridge alpha within training.

    All four outputs share the same alpha: the frozen common chemical score is
    primary and its three component log-concentration z scores are descriptive.
    The score is a fixed linear combination of those components, so this does not
    add an independent fourth chemical target. No neural feature is removed.
    """
    report, output = Path(report), Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; existing checks are preserved.")
    files = [report / "frozen_candidate.json", report / "tables/sample_scores.csv",
             report / "tables/neural_unit_coefficients.csv",
             report / "tables/selected_chemical_standardized.csv"]
    hashes = {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    frozen = json.loads(files[0].read_text())
    scores, neural, chemical = [pd.read_csv(p, index_col=0) for p in files[1:]]
    train, held = frozen["discovery_ids"], frozen["holdout_ids"]
    members = frozen["axis"]["members"]
    targets = ["chemical_score"] + members
    weights = np.array([frozen["axis"]["effective_weight"][m] for m in members])
    y = pd.concat([scores.chemical_score, chemical[members]], axis=1)
    assert len(train) == 70 and len(held) == 36 and not set(train).intersection(held)
    assert neural.shape == (106, 13) and y.shape == (106, 4)
    assert set(train + held) == set(neural.index) == set(y.index)
    assert np.isfinite(neural).all().all() and np.isfinite(y).all().all()
    np.testing.assert_allclose(y[members].to_numpy() @ weights, y.chemical_score, atol=1e-12)
    xtrain, ytrain = neural.loc[train].to_numpy(), y.loc[train].to_numpy()
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=20261004).split(xtrain))
    rows = []
    for alpha in alphas:
        for fold, (fit, validate) in enumerate(folds, 1):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(xtrain[fit], ytrain[fit])
            prediction = model.predict(xtrain[validate])
            rows.append({"alpha": alpha, "fold": fold, "n_validation": len(validate),
                         "primary_score_sse": float(np.square(prediction[:, 0] - ytrain[validate, 0]).sum())})
    cv = pd.DataFrame(rows)
    ranking = cv.groupby("alpha").primary_score_sse.sum().sort_values(kind="stable")
    selected_alpha = float(ranking.index[0])
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
    model.fit(xtrain, ytrain)
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
    output.mkdir(parents=True)
    cv.to_csv(output / "training_alpha_cv.csv", index=False)
    parameters = {
        "source_sha256": hashes, "model": "StandardScaler + multioutput Ridge with intercept",
        "inputs": neural.columns.tolist(), "outputs": targets, "primary_output": "chemical_score",
        "training_ids": train, "heldout_ids": held, "alpha_grid": alphas,
        "alpha_selection": "Minimum pooled 5-fold training validation SSE for the fixed chemical score",
        "cv_seed": 20261004, "selected_alpha": selected_alpha,
        "input_means": scaler.mean_.tolist(), "input_scales": scaler.scale_.tolist(),
        "coefficients_standardized_inputs": ridge.coef_.tolist(), "intercepts": ridge.intercept_.tolist(),
        "training_target_means": ytrain.mean(axis=0).tolist(),
        "chemical_target_definition": "Previously frozen discovery means, SDs and PC1 weights; no target refit",
        "secondary_target_tuning": "Same alpha as primary score; no separate search",
        "bootstrap": "5000 paired strain resamples of fixed held-out predictions; conditional intervals only",
        "bootstrap_seed": 20261005,
    }
    # Record the training-only decisions before evaluating held-out predictions.
    (output / "frozen_reverse_model.json").write_text(json.dumps(parameters, indent=2) + "\n")
    predictions = model.predict(neural.loc[held].to_numpy())
    observed = y.loc[held].to_numpy()
    np.testing.assert_allclose(predictions[:, 1:] @ weights, predictions[:, 0], atol=1e-12)
    baseline = ytrain.mean(axis=0)
    errors = np.square(predictions - observed)
    baseline_errors = np.square(baseline - observed)
    rng = np.random.default_rng(20261005)
    resamples = rng.integers(0, len(held), size=(5000, len(held)))
    bootstrap_gains = 1 - errors[resamples].sum(axis=1) / baseline_errors[resamples].sum(axis=1)
    metrics, prediction_rows = {}, []
    for j, target in enumerate(targets):
        sse, base_sse = float(errors[:, j].sum()), float(baseline_errors[:, j].sum())
        metrics[target] = {
            "n": len(held), "model_sse": sse, "training_mean_baseline_sse": base_sse,
            "relative_error_reduction": 1 - sse / base_sse,
            "conditional_bootstrap_95pct_interval": np.quantile(bootstrap_gains[:, j], [0.025, 0.975]).tolist(),
            "rmse": float(np.sqrt(sse / len(held))),
            "baseline_rmse": float(np.sqrt(base_sse / len(held))),
            "pearson_r": float(np.corrcoef(observed[:, j], predictions[:, j])[0, 1]),
            "r2_against_heldout_mean": float(1 - sse / np.square(observed[:, j] - observed[:, j].mean()).sum()),
        }
        for i, strain in enumerate(held):
            prediction_rows.append({"strain": strain, "genus": scores.loc[strain, "genus"],
                                    "target": target, "observed": observed[i, j],
                                    "predicted": predictions[i, j], "training_mean_baseline": baseline[j],
                                    "squared_error": errors[i, j], "baseline_squared_error": baseline_errors[i, j]})
    joint_sse, joint_base = float(errors[:, 1:].sum()), float(baseline_errors[:, 1:].sum())
    result = {
        "direction": "All 13 neural unit coefficients -> fixed chemical pattern and its 3 members",
        "n_train": len(train), "n_holdout": len(held), "selected_alpha": selected_alpha,
        "metrics": metrics,
        "three_component_joint_error_reduction": 1 - joint_sse / joint_base,
        "limitations": [
            "Fixed module was selected using these training strains in the forward task; CV tunes alpha only",
            "Previously explored held-out data; internal exploratory reverse check, not independent validation",
            "Same cached neural representation and full-panel completeness screening are reused",
            "Random strain split permits genera shared between training and holdout; unseen-genus transfer is untested",
            "Bootstrap conditions on the chosen targets, fitted model and split, assumes resampled strains exchangeable; it excludes selection/training uncertainty and shared-animal/genus dependence",
            "Prediction concerns log-concentration patterns in independently cultured material, not stimulus concentrations or causal effects",
            "Forward and reverse error reductions have different targets and denominators and cannot rank the two tasks directly",
        ],
    }
    pd.DataFrame(prediction_rows).to_csv(output / "heldout_predictions.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(output / "heldout_metrics.csv", index_label="target")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    make_prediction_figure(output)
    assert hashes == {str(p.relative_to(report)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return result


def make_prediction_figure(output: Path) -> list[Path]:
    """Render the saved prediction check without refitting or recomputing intervals."""
    output = Path(output)
    metrics = json.loads((output / "results.json").read_text())["metrics"]
    saved = pd.read_csv(output / "heldout_predictions.csv")
    primary = saved.loc[saved.target.eq("chemical_score")]
    observed = primary.observed.to_numpy()[:, None]
    predictions = primary.predicted.to_numpy()[:, None]
    baseline = primary.training_mean_baseline.to_numpy()

    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(6.4, 6.3))
        limits = [float(min(observed[:, 0].min(), predictions[:, 0].min()) - 0.2),
                  float(max(observed[:, 0].max(), predictions[:, 0].max()) + 0.2)]
        ax.plot(limits, limits, linestyle="--", linewidth=1.2, color="#7C858B", label="Perfect prediction")
        ax.axhline(baseline[0], color="#BCC3C7", linewidth=1, label="Training-mean prediction")
        ax.scatter(observed[:, 0], predictions[:, 0], s=42, color="#356878", alpha=0.83, edgecolors="white", linewidths=0.5)
        ax.set(xlim=limits, ylim=limits, aspect="equal", xlabel="Measured chemical score",
               ylabel="Predicted from all 13 neurons")
        ax.set_title("Predicting the chemical pattern", loc="left", fontsize=16, pad=34)
        ax.text(0, 1.045, "36 strains · internal held-out check", transform=ax.transAxes, color="#5C6870", fontsize=11)
        gain = metrics["chemical_score"]["relative_error_reduction"]
        ax.text(0.04, 0.95, f"Error reduction: {100 * gain:.1f}%", transform=ax.transAxes, va="top", fontsize=12)
        lower, upper = metrics["chemical_score"]["conditional_bootstrap_95pct_interval"]
        ax.text(0.04, 0.901, f"Conditional 95% interval: {100 * lower:.1f}% to {100 * upper:.1f}%",
                transform=ax.transAxes, va="top", fontsize=9, color="#5C6870")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"reverse_prediction.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    (output / "figure_caption.txt").write_text(
        "All 36 held-out strains, with no example selection. The fixed target is the common "
        "score of Glucaric acid, Lumichrome and Vitamin B1. A Ridge linear model uses all 13 "
        "existing neural unit coefficients; feature standardization and alpha selection "
        "use training data only. Dashed diagonal indicates exact prediction; the horizontal "
        "line predicts the training mean chemical score for every strain. Error reduction "
        "uses the training-mean baseline. This is a reverse exploratory check in previously "
        "seen data, not a mathematical inversion of the forward model or causal evidence. "
        "Chemical targets and the neural representation were previously defined. The "
        "conditional 95% interval uses 5000 paired held-out strain resamples, treating "
        "strains as exchangeable and holding model, split and target selection fixed; "
        "it excludes training/selection uncertainty and shared-animal/genus dependence. "
        "See results.json for the limits of this internal check.\n"
    )
    return [output / name for name in ("reverse_prediction.png", "reverse_prediction.svg", "figure_caption.txt")]
