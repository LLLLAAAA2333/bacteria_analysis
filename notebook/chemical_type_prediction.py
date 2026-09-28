"""Date-held-out chemical classification of provisional neural temporal types.

Repeated strains are purged from training; all chemical transforms and model
selection use training data only. Unresolved/rare types remain descriptive.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


MODELS = ("majority", "genus", "chemistry", "genus+chemistry")
METRIC_COLUMNS = ["date", "model", "n_strains", "balanced_accuracy", "macro_f1"]
AUDIT_COLUMNS = ["date", "train_strains", "test_strains", "train_types", "test_types",
                 "unseen_test_types", "unseen_test_genera", "status"]
TUNING_COLUMNS = ["outer_date", "model", "C", "inner_date", "balanced_accuracy", "status"]
SUMMARY_COLUMNS = ["model", "n_strains", "n_dates", "balanced_accuracy", "macro_f1"]


def purged_date_split(data, date):
    test = data.loc[data.date.eq(date)].copy()
    train = data.loc[~data.date.eq(date) & ~data.sample_id.isin(test.sample_id)].copy()
    return train, test


def strain_weights(data):
    """One total unit per strain, even when it has multiple date records."""
    return 1.0 / data.sample_id.map(data.sample_id.value_counts()).to_numpy()


def feature_arrays(train, test, chemicals, model, seed, return_names=False):
    x_train, x_test, names = [], [], []
    chemical_scale = 1.0
    if "genus" in model:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x_train.append(encoder.fit_transform(train[["genus"]]))
        x_test.append(encoder.transform(test[["genus"]]))
        names.extend(encoder.get_feature_names_out(["genus"]))
    if "chemistry" in model:
        reference = chemicals.loc[train.sample_id.unique()]
        # Numerical feature filtering is training-only and independent of labels.
        columns = reference.columns[reference.nunique().gt(1)]
        if len(columns):
            transformer = QuantileTransformer(n_quantiles=min(100, len(reference)),
                                               output_distribution="uniform", subsample=None,
                                               random_state=seed)
            transformer.fit(reference[columns])
            # Chemical block dimension scaling prevents 380 columns from getting
            # greater total scale merely by outnumbering the genus indicators.
            scale = np.sqrt(len(columns))
            chemical_scale = scale
            x_train.append((2 * transformer.transform(chemicals.loc[train.sample_id, columns]) - 1) / scale)
            x_test.append((2 * transformer.transform(chemicals.loc[test.sample_id, columns]) - 1) / scale)
            names.extend(columns)
    if not x_train:
        x_train, x_test, names = [np.zeros((len(train), 1))], [np.zeros((len(test), 1))], ["intercept_placeholder"]
    arrays = (np.concatenate(x_train, axis=1), np.concatenate(x_test, axis=1))
    return (*arrays, names, chemical_scale) if return_names else arrays


def predict_fold(train, test, chemicals, model, C, seed, return_weights=False):
    weights = strain_weights(train)
    totals = pd.Series(weights).groupby(train.response_type.to_numpy()).sum()
    if model == "majority":
        prediction = np.repeat(totals.idxmax(), len(test))
        return (prediction, []) if return_weights else prediction
    if len(totals) < 2:
        raise ValueError("Training set has fewer than two response types")
    x_train, x_test, names, scale = feature_arrays(train, test, chemicals, model, seed, return_names=True)
    balance = weights.sum() / (len(totals) * train.response_type.map(totals).to_numpy())
    fit = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fit.fit(x_train, train.response_type, sample_weight=weights * balance)
    prediction = fit.predict(x_test)
    if not return_weights:
        return prediction
    coefficients = fit.coef_
    if len(fit.classes_) == 2:
        coefficients = np.vstack([-coefficients[0] / 2, coefficients[0] / 2])
    weights = [dict(response_type=label, feature_id=feature, rank_range_logit_weight=float(2 * value / scale))
               for label, vector in zip(fit.classes_, coefficients)
               for feature, value in zip(names, vector) if feature in chemicals.columns]
    return prediction, weights


def scores(data, prediction):
    labels = sorted(data.response_type.unique())
    weights = strain_weights(data)
    return dict(balanced_accuracy=recall_score(data.response_type, prediction, labels=labels,
                                              average="macro", sample_weight=weights, zero_division=0),
                macro_f1=f1_score(data.response_type, prediction, labels=labels, average="macro",
                                  sample_weight=weights, zero_division=0))


def choose_C(outer_train, chemicals, model, settings):
    rows = []
    inner_folds = []
    for date in sorted(outer_train.date.unique()):
        train, test = purged_date_split(outer_train, date)
        train = train.loc[train.prediction_eligible].copy()
        test = test.loc[test.prediction_eligible].copy()
        if len(test) and train.response_type.nunique() >= 2:
            inner_folds.append((date, train, test))
    if len(inner_folds) < 2:
        return 1.0, "fixed C=1: fewer than two usable inner date folds", rows
    for C in settings["C_grid"]:
        for date, train, test in inner_folds:
            try:
                prediction = predict_fold(train, test, chemicals, model, C, settings["seed"])
                value = scores(test, prediction)["balanced_accuracy"]
                rows.append(dict(C=C, inner_date=date, balanced_accuracy=value, status="ok"))
            except (ValueError, FloatingPointError, ConvergenceWarning) as error:
                rows.append(dict(C=C, inner_date=date, balanced_accuracy=np.nan, status=str(error)))
    table = pd.DataFrame(rows)
    valid = table.groupby("C").balanced_accuracy.count().eq(len(inner_folds))
    means = table.groupby("C").balanced_accuracy.mean().loc[valid]
    if means.empty:
        raise ValueError("No C value fitted successfully in all inner date folds")
    # Ascending C breaks exact ties in favor of stronger regularization.
    selected = means.sort_index().idxmax()
    return float(selected), "inner date CV; equal weight per held-out date", rows


def evaluate_types(date_labels, taxonomy, chemicals, settings):
    data = date_labels.merge(taxonomy[["sample_id", "genus"]], on="sample_id", how="left", validate="many_to_one")
    data["date"] = data.date.astype(str)
    data["has_chemistry"] = data.sample_id.isin(chemicals.index)
    data["has_genus"] = data.genus.notna()
    data["resolved"] = data.response_type.ne("unresolved")
    paired = data.loc[data.has_chemistry & data.has_genus & data.resolved].copy()
    support = paired.groupby("response_type").agg(n_strains=("sample_id", "nunique"),
                                                  n_dates=("date", "nunique"), n_rows=("date", "size"))
    support["eligible"] = support.n_strains.ge(settings["min_class_strains"]) & support.n_dates.ge(settings["min_class_dates"])
    classes = sorted(support.index[support.eligible].tolist())
    data["prediction_eligible"] = data.has_chemistry & data.has_genus & data.response_type.isin(classes)
    empty_predictions = pd.DataFrame(columns=["sample_id", "date", "response_type", "direction", "genus",
                                              "prediction", "model", "C", "selection"])
    result = dict(support=support, matching=data, predictions=empty_predictions,
                  fold_metrics=pd.DataFrame(columns=METRIC_COLUMNS), fold_audit=pd.DataFrame(columns=AUDIT_COLUMNS),
                  tuning=pd.DataFrame(columns=TUNING_COLUMNS), summary=pd.DataFrame(columns=SUMMARY_COLUMNS),
                  weights=pd.DataFrame(columns=["date", "model", "C", "response_type", "feature_id", "rank_range_logit_weight"]),
                  classes=classes, status="")
    if len(classes) < 2:
        result["status"] = "Skipped: fewer than two types meet the fixed strain/date coverage requirements."
        return result
    predictions, metrics, audits, tuning, coefficient_rows = [], [], [], [], []
    for date in sorted(data.date.unique()):
        # Purge all strains observed on the held-out date, including ambiguous
        # observations, before restricting the supervised cohort.
        train_all, test_all = purged_date_split(data, date)
        train = train_all.loc[train_all.prediction_eligible].copy()
        test = test_all.loc[test_all.prediction_eligible].copy()
        audit = dict(date=date, train_strains=train.sample_id.nunique(), test_strains=test.sample_id.nunique(),
                     train_types=train.response_type.nunique(), test_types=test.response_type.nunique(),
                     unseen_test_types="|".join(sorted(set(test.response_type) - set(train.response_type))),
                     unseen_test_genera="|".join(sorted(set(test.genus) - set(train.genus))),
                     status="ok")
        if test.empty or train.response_type.nunique() < 2:
            audit["status"] = "skipped: no eligible test rows or fewer than two training types"
            audits.append(audit)
            continue
        fold_predictions, fold_weights = [], []
        print(f"Chemical type CV: held-out date {date}, {train.sample_id.nunique()} train / {test.sample_id.nunique()} test strains", flush=True)
        try:
            for model in MODELS:
                if model == "majority":
                    C, selection, inner = np.nan, "training majority", []
                else:
                    C, selection, inner = choose_C(train_all, chemicals, model, settings)
                tuning.extend(dict(outer_date=date, model=model, **row) for row in inner)
                prediction, weights = predict_fold(train, test, chemicals, model, C, settings["seed"], return_weights=True)
                fold_weights.extend(dict(date=date, model=model, C=C, **weight) for weight in weights)
                columns = ["sample_id", "date", "response_type", "direction", "genus"]
                columns += [name for name in ["confidence", "agreement", "signal_fraction", "quality_flags"] if name in test]
                rows = test[columns].copy()
                rows["prediction"], rows["model"], rows["C"] = prediction, model, C
                rows["selection"] = selection
                fold_predictions.append(rows)
            # Compare models on exactly the same successful outer folds.
            for rows in fold_predictions:
                predictions.append(rows)
                metrics.append(dict(date=date, model=rows.model.iloc[0], n_strains=test.sample_id.nunique(),
                                    **scores(test, rows.prediction)))
            coefficient_rows.extend(fold_weights)
        except (ValueError, FloatingPointError, ConvergenceWarning) as error:
            audit["status"] = "all models excluded for this fold: " + str(error)
        audits.append(audit)
    result.update(fold_audit=pd.DataFrame(audits, columns=AUDIT_COLUMNS),
                  tuning=pd.DataFrame(tuning, columns=TUNING_COLUMNS),
                  fold_metrics=pd.DataFrame(metrics, columns=METRIC_COLUMNS))
    if not predictions:
        result["status"] = "Skipped: no common successful outer date folds."
        return result
    result["predictions"] = pd.concat(predictions, ignore_index=True)
    result["weights"] = pd.DataFrame(coefficient_rows, columns=result["weights"].columns)
    summaries = []
    for model, rows in result["predictions"].groupby("model", sort=False):
        summaries.append(dict(model=model, n_strains=rows.sample_id.nunique(), n_dates=rows.date.nunique(),
                              **scores(rows, rows.prediction)))
    result["summary"] = pd.DataFrame(summaries)
    result["status"] = "Completed; descriptive held-out metrics for provisional dynamic labels; rare/unavailable types excluded."
    return result


def plot_predictions(result, folder):
    if result["predictions"].empty:
        return
    labels = result["classes"]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4), layout="constrained")
    for ax, model in zip(axes, MODELS):
        rows = result["predictions"].loc[result["predictions"].model.eq(model)]
        matrix = confusion_matrix(rows.response_type, rows.prediction, labels=labels,
                                  sample_weight=strain_weights(rows), normalize="true")
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] > .5 else "black")
        ax.set(title=model, xticks=range(len(labels)), yticks=range(len(labels)),
               xticklabels=labels, yticklabels=labels, xlabel="Predicted type", ylabel="Observed type")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Held-out dates, repeated strains purged | provisional types; row-normalized confusion")
    fig.savefig(folder / "chemical_type_confusion.png", dpi=150)
    plt.show()

    table = result["fold_metrics"].pivot(index="date", columns="model", values="balanced_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for model in MODELS:
        axes[0].plot(range(len(table)), table[model], marker="o", markersize=4, label=model)
    axes[0].set(xticks=range(len(table)), xticklabels=table.index, ylim=(0, 1),
                ylabel="Macro recall in held-out date", title="Same held-out rows for every model")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].legend(fontsize=8)
    improvement = table["genus+chemistry"] - table["genus"]
    axes[1].bar(range(len(table)), improvement, color=np.where(improvement >= 0, "#438777", "#B67C64"))
    axes[1].axhline(0, color="black", lw=.6)
    axes[1].set(xticks=range(len(table)), xticklabels=table.index,
                ylabel="Change in macro recall", title="Adding chemistry to the genus model")
    axes[1].tick_params(axis="x", rotation=60)
    fig.savefig(folder / "chemical_type_date_performance.png", dpi=150)
    plt.show()


def summarize_weights(result, features):
    """Correlated-feature model weights are exploratory, not independent chemical effects."""
    rows = result["weights"]
    if rows.empty:
        return pd.DataFrame(columns=["model", "response_type", "feature_id", "metabolite"])
    records = []
    for (model, kind, feature), group in rows.groupby(["model", "response_type", "feature_id"]):
        values = group.rank_range_logit_weight.to_numpy()
        median = float(np.median(values))
        records.append(dict(model=model, response_type=kind, feature_id=feature,
                            median_weight=median, median_abs_weight=float(np.median(np.abs(values))),
                            sign_agreement=float(np.mean(np.sign(values) == np.sign(median))),
                            n_folds=group.date.nunique()))
    return pd.DataFrame(records).merge(features, on="feature_id", how="left", validate="many_to_one")


def plot_chemical_patterns(weight_summary, strain_labels, chemicals, features, folder, n_features=15):
    """Post-selection descriptive heatmap; not independent validation of ranked features."""
    if weight_summary.empty:
        return []
    rank = weight_summary.loc[weight_summary.model.eq("genus+chemistry")].sort_values("median_abs_weight", ascending=False)
    rank = rank.drop_duplicates("feature_id")
    if "block" in rank:
        rank["display_block"] = rank.block.fillna(rank.feature_id)
        rank = rank.drop_duplicates("display_block")
    ids = rank.head(n_features).feature_id.tolist()
    labels = strain_labels.loc[strain_labels.response_type.ne("unresolved") & strain_labels.sample_id.isin(chemicals.index)].copy()
    labels["phenotype"] = labels.response_type + ":" + labels.direction
    # Full-panel percentile ranks are for visualization only; predictions above
    # use transformers fitted inside each training fold.
    display_values = chemicals.loc[labels.sample_id, ids].rank(pct=True)
    display_values["phenotype"] = labels.set_index("sample_id").phenotype
    means = display_values.groupby("phenotype").mean()
    counts = labels.phenotype.value_counts()
    fig, ax = plt.subplots(figsize=(13, max(4, .7 * len(means))), layout="constrained")
    im = ax.imshow(means, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    names = features.set_index("feature_id").metabolite
    ax.set(xticks=range(len(ids)), xticklabels=[names[x] for x in ids],
           yticks=range(len(means)), yticklabels=[f"{label} (n={counts[label]})" for label in means.index])
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    fig.colorbar(im, ax=ax, label="Mean chemical percentile")
    ax.set_title("Chemical patterns by provisional shape/direction\nFeatures selected by cross-fold model weights; correlated molecules are not independent effects", fontsize=10)
    fig.savefig(folder / "chemical_type_patterns.png", dpi=150)
    plt.show()
    return ids
