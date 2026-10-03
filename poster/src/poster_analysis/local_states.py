"""Reusable calculations and figures for local chemical–neural states (FIG04).

Functions accept DataFrames with strain indices, annotation columns or the saved
13 unit-neural columns. Full-data selection is descriptive. ``fit_one_fold``
demonstrates train-only fitting; this module does not launch a cross-validation
sweep. The old Bacteroides contrast and new Bifidobacterium population objective
remain distinct. File reads are explicit and no historical scripts are imported.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .chemical_axes import fit_axes, transform_axes
from .constants import NEURONS

GENERA = ["Bacteroides", "Bifidobacterium"]
GROUPS = ["Low", "Mid", "High"]
GROUP_COLORS = {"Low": "#347FA3", "Mid": "#9CA5A6", "High": "#C66B36"}
DISPLAY_TABLES = ["strain_scores", "neural_slopes", "selected_members",
                  "group_neural_means", "selected_chemical_z"]


def load_local_inputs(chemistry_dir, local_report_dir, fixed_contrast_dir):
    """Read indexed source tables only; directory arguments make provenance explicit."""
    chem = Path(chemistry_dir) / "tables"
    local, fixed = Path(local_report_dir), Path(fixed_contrast_dir)
    paths = {
        "log2": chem / "fresh_chemical_log2.csv",
        "metadata": chem / "fresh_feature_metadata.csv",
        "unit": chem / "neural_unit_coefficients.csv",
        "context": chem / "sample_context.csv",
        "population_performance": local / "tables/heldout_performance.csv",
        "contrast_performance": fixed / "model/tables/pooled_performance.csv",
        "contrast_predictions": fixed / "model/tables/heldout_predictions.csv",
        "bact_members": fixed / "model/tables/selected_full_state_members.csv",
        "bif_members": local / "tables/selected_members.csv",
        "population_fold_selection": local / "tables/fold_selections.csv",
        "population_predictions": local / "tables/heldout_predictions.csv",
    }
    inputs = {"log2": pd.read_csv(paths["log2"], index_col="strain"),
              "metadata": pd.read_csv(paths["metadata"], index_col="metabolite"),
              "unit": pd.read_csv(paths["unit"], index_col="strain"),
              "context": pd.read_csv(paths["context"], index_col="strain", dtype={"dates": str})}
    for key in set(paths) - set(inputs):
        inputs[key] = pd.read_csv(paths[key])
    inputs["reference_display"] = {}
    for name in DISPLAY_TABLES:
        key = f"reference_{name}"
        paths[key] = local / f"figure_data/{name}.csv"
        inputs["reference_display"][name] = pd.read_csv(paths[key])
    inputs["source_paths"] = paths
    for frame in [inputs["log2"], inputs["unit"], inputs["context"]]:
        if not frame.index.is_unique:
            raise ValueError("Source strain indices must be unique")
    if list(inputs["unit"].columns) != list(NEURONS):
        raise ValueError("Unexpected neural coordinate order")
    return inputs


def standardize_chemistry(log_frame, means=None, sds=None):
    """Return z, means and sample SDs. Supply training scales for heldout rows.

    Input is log2(c / (1 ng/mL)); no +1 or imputation. Constants get z=NaN and
    are excluded by ``fit_axes``. Chemical-state members must have positive SD.
    """
    means = log_frame.mean() if means is None else pd.Series(means)
    sds = log_frame.std(ddof=1) if sds is None else pd.Series(sds)
    return (log_frame - means) / sds.where(sds > 1e-12), means, sds


def family_weights(member_metadata):
    """Equal mass-column-family weight, then equal annotation weight per family."""
    family = member_metadata["family"]
    counts = family.value_counts()
    return pd.Series([1 / (len(counts) * counts[f]) for f in family],
                     index=member_metadata.index, name="weight")


def score_state(log_frame, members, means, sds, weights):
    """Transform rows with fixed training statistics and selected member weights."""
    members = list(members)
    means, sds, weights = map(pd.Series, [means, sds, weights])
    if (sds.loc[members] <= 1e-12).any():
        raise ValueError("State contains constant training features")
    z, _, _ = standardize_chemistry(log_frame.loc[:, members], means.loc[members], sds.loc[members])
    return (z @ weights.loc[members]).rename("chemical_score")


def fit_ols(score, responses):
    """Unscaled coordinate-wise OLS with one shared predictor and an intercept.

    Returns coefficients and training diagnostics indexed by response name.
    This accepts any named response DataFrame, including a single contrast.
    """
    if isinstance(responses, pd.Series):
        responses = responses.to_frame()
    score = score.reindex(responses.index) if isinstance(score, pd.Series) else pd.Series(score, index=responses.index)
    x, y = score.to_numpy(float), responses.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS inputs must be finite and aligned")
    dx, dy = x - x.mean(), y - y.mean(axis=0)
    xx = float(dx @ dx)
    slope = dx @ dy / xx if xx > 1e-24 else np.zeros(y.shape[1])
    intercept = y.mean(axis=0) - slope * x.mean()
    prediction = intercept + x[:, None] * slope
    mean_sse, sse = np.sum(dy ** 2, axis=0), np.sum((y - prediction) ** 2, axis=0)
    denominator = np.sqrt(xx * mean_sse)
    r = np.divide(dx @ dy, denominator, out=np.zeros(y.shape[1]), where=denominator > 1e-24)
    iqr = float(np.quantile(x, .75) - np.quantile(x, .25))
    return pd.DataFrame({"intercept": intercept, "slope": slope,
                         "genus_mean": y.mean(axis=0), "sse_model": sse, "sse_mean": mean_sse,
                         "pearson_r": r, "score_iqr": iqr, "iqr_effect": slope * iqr},
                        index=pd.Index(responses.columns, name="neuron"))


def select_state(log_frame, metadata, unit, *, objective="all13"):
    """Discover chemical groups and select one using a stated neural objective.

    ``all13`` minimizes summed SSE over unscaled unit coordinates.
    ``adf_minus_ash`` maximizes contrast r² (the historical anchor rule).
    Exact ties use sorted member names. There is no component/grid search.
    """
    if not log_frame.index.equals(unit.index):
        raise ValueError("Chemical and neural indices must agree")
    if objective not in ["all13", "adf_minus_ash"]:
        raise ValueError("Choose all13 or adf_minus_ash")
    targets = unit if objective == "all13" else (unit.ADF - unit.ASH).rename("ADF_minus_ASH").to_frame()
    axes = fit_axes(log_frame, metadata)
    rows, models = [], {}
    for state, members in axes["module_members"].items():
        model = fit_ols(axes["train_scores"][state], targets)
        models[state] = model
        total, null = float(model.sse_model.sum()), float(model.sse_mean.sum())
        rows.append({"state_id": state, "n_annotations": len(members),
                     "n_families": int(metadata.loc[members, "family"].nunique()),
                     "sse_model_total": total, "sse_mean_total": null,
                     "apparent_sse_improvement": 1 - total / null if null > 0 else 0,
                     "selection_value": total if objective == "all13" else -float(model.pearson_r.iloc[0] ** 2),
                     "members_json": json.dumps(members)})
    ranked = sorted(rows, key=lambda row: (row["selection_value"], tuple(sorted(json.loads(row["members_json"])))))
    selected = ranked[0]["state_id"] if ranked else None
    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates["selected"] = candidates.state_id.eq(selected)
    score = axes["train_scores"][selected] if selected else pd.Series(0., index=log_frame.index)
    return {"axes": axes, "state_id": selected, "objective": objective, "candidates": candidates,
            "score": score.rename("chemical_score"), "model": fit_ols(score, unit),
            "selection_model": models.get(selected, fit_ols(score, targets)),
            "mean_fallback": selected is None}


def fit_one_fold(log_frame, metadata, unit, train_ids, test_ids, *, objective="all13"):
    """Explicit train-only chemical scales, groups, selection and response fitting."""
    train_ids, test_ids = list(train_ids), list(test_ids)
    if set(train_ids) & set(test_ids):
        raise ValueError("Training and test rows overlap")
    fitted = select_state(log_frame.loc[train_ids], metadata, unit.loc[train_ids], objective=objective)
    transformed = transform_axes(log_frame.loc[test_ids], fitted["axes"])
    state, model = fitted["state_id"], fitted["model"]
    score = transformed[state] if state else pd.Series(0., index=test_ids)
    predictions = pd.DataFrame(np.array(model.intercept) + score.to_numpy()[:, None] * np.array(model.slope),
                               index=test_ids, columns=unit.columns)
    baseline = pd.DataFrame(np.tile(model.genus_mean, (len(test_ids), 1)), index=test_ids, columns=unit.columns)
    return {"fitted": fitted, "score": score, "predictions": predictions, "baseline": baseline,
            "observed": unit.loc[test_ids], "train_ids": train_ids, "test_ids": test_ids}


def rank_thirds(scores):
    """Deterministic equal-count groups: score first, strain ID breaks ties."""
    scores = pd.Series(scores)
    order = sorted(scores.index, key=lambda strain: (float(scores.loc[strain]), str(strain)))
    result = pd.DataFrame(index=pd.Index(scores.index, name="strain"))
    result["chemical_score"] = scores
    result["rank_group"] = ""
    for group, ids in zip(GROUPS, np.array_split(np.array(order), 3)):
        result.loc[ids, "rank_group"] = group
    result["chemical_rank"] = [order.index(strain) + 1 for strain in result.index]
    return result


def observed_group_means(unit, groups):
    """Observed group means, centered on the observed genus mean; no row z-score."""
    rows = []
    for group in GROUPS:
        ids = groups.index[groups.rank_group.eq(group)]
        for neuron in unit:
            mean, overall = float(unit.loc[ids, neuron].mean()), float(unit[neuron].mean())
            rows.append({"rank_group": group, "neuron": neuron, "n": len(ids),
                         "observed_mean": mean, "genus_mean": overall, "centered_mean": mean - overall})
    return pd.DataFrame(rows)


def state_display(log_frame, metadata, unit, context, fitted, genus):
    """Build compact plot data from one explicit fitted state, without refitting."""
    state, axes, model = fitted["state_id"], fitted["axes"], fitted["model"]
    if state is None:
        raise ValueError("A mean-only fallback cannot define a display state")
    members = axes["module_members"][state]
    groups = rank_thirds(fitted["score"])
    scores = context.loc[log_frame.index].copy().join(groups)
    scores["genus"], scores["state_id"] = genus, state
    scores["taxonomy_flag"] = scores.taxonomy_note.fillna("").astype(str).str.strip().ne("")
    for neuron in unit:
        scores[f"unit_{neuron}"] = unit[neuron]
    scores["unit_ADF_minus_ASH"] = unit.ADF - unit.ASH
    direction = model.slope.to_numpy()
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm else direction * 0
    scores["fitted_direction_projection"] = (unit - unit.mean()).to_numpy() @ direction
    is_contrast = fitted["objective"] == "adf_minus_ash"
    scores["plot_response"] = scores.unit_ADF_minus_ASH if is_contrast else scores.fitted_direction_projection
    scores["readout"] = "Fixed ADF-ASH contrast" if is_contrast else "Projection onto fitted 13-neuron direction"
    selected = metadata.loc[members].copy()
    selected["weight"] = axes["score_weights"][state]
    selected["training_log2_mean"] = axes["means"].loc[members]
    selected["training_log2_sample_sd"] = axes["scales"].loc[members]
    selected["state_id"] = state
    z, _, _ = standardize_chemistry(log_frame.loc[:, members], axes["means"].loc[members], axes["scales"].loc[members])
    chem = z.rename_axis(index="strain", columns="metabolite").stack().rename("z").reset_index()
    chem["rank_group"] = chem.strain.map(groups.rank_group)
    frames = {"strain_scores": scores.reset_index(), "neural_slopes": model.reset_index(),
              "selected_members": selected.reset_index(), "group_neural_means": observed_group_means(unit, groups),
              "selected_chemical_z": chem}
    for frame in frames.values():
        frame["genus"] = genus
    return frames


def combine_display_tables(states):
    return {name: pd.concat([state[name] for state in states], ignore_index=True) for name in DISPLAY_TABLES}


def compare_display_tables(computed, reference, atol=1e-12):
    """Numerical parity on every plotting value; permit additional useful columns."""
    specifications = {
        "strain_scores": (["genus", "strain"], ["chemical_score", "plot_response", "chemical_rank"] + [f"unit_{n}" for n in NEURONS]),
        "neural_slopes": (["genus", "neuron"], ["slope", "genus_mean", "score_iqr", "iqr_effect"]),
        "selected_members": (["genus", "metabolite"], ["weight", "training_log2_mean", "training_log2_sample_sd"]),
        "group_neural_means": (["genus", "rank_group", "neuron"], ["observed_mean", "genus_mean", "centered_mean", "n"]),
        "selected_chemical_z": (["genus", "strain", "metabolite"], ["z"]),
    }
    rows = []
    for name, (keys, cols) in specifications.items():
        actual = computed[name].set_index(keys).sort_index()
        expected = reference[name].set_index(keys).sort_index()
        pd.testing.assert_index_equal(actual.index, expected.index)
        diff = np.abs(actual[cols].to_numpy(float) - expected[cols].to_numpy(float))
        np.testing.assert_allclose(actual[cols], expected[cols], atol=atol, rtol=1e-12)
        if "rank_group" in actual.columns:
            np.testing.assert_array_equal(actual.rank_group, expected.rank_group)
        rows.append({"table": name, "n_rows": len(actual), "max_abs_error": float(diff.max())})
    return pd.DataFrame(rows)


def plot_main(tables):
    """Return the final FIG04 figure from observed values and disclosed fitted axis."""
    limit = float(np.ceil(tables["group_neural_means"].centered_mean.abs().max() / .05) * .05)
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("Local chemical states and neural response patterns", y=.973, fontsize=18, weight="bold")
    positions = [(.105, .54, .35, .26), (.60, .54, .35, .26)]
    heat_positions = [(.105, .153, .35, .285), (.60, .153, .35, .285)]
    for k, genus in enumerate(GENERA):
        s = tables["strain_scores"].query("genus == @genus")
        members = tables["selected_members"].query("genus == @genus")
        cx = positions[k][0] + positions[k][2] / 2
        fig.text(cx, .917, genus, ha="center", fontsize=16, fontstyle="italic")
        fig.text(cx, .885, f"{len(s)} strains · {s.species.nunique()} recorded species", ha="center", fontsize=11, color="#4C5357")
        fig.text(cx, .855, f"{len(members)} co-varying chemical annotations", ha="center", fontsize=11, color="#4C5357")
        examples = "Includes p-Cresol, N-acetylleucine and ADP" if k == 0 else "Includes succinic acid, choline and vitamin B1"
        fig.text(cx, .826, examples, ha="center", fontsize=9.5, color="#60686C")
        ax = fig.add_axes(positions[k])
        for group in GROUPS:
            part = s[s.rank_group == group]
            ax.scatter(part.chemical_score, part.plot_response, color=GROUP_COLORS[group], s=44,
                       edgecolor="white", linewidth=.6, zorder=3)
        x, y = s.chemical_score.to_numpy(), s.plot_response.to_numpy()
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color="#51595D", lw=1.25)
        ax.axhline(0, color="#D6D9DB", lw=.6)
        ax.set(xlabel="Chemical-state score", ylabel="ADF − ASH (unit coefficients)" if k == 0 else "Projection onto fitted\n13-neuron direction")
        ax.text(-.22, 1.02, "AB"[k], transform=ax.transAxes, fontsize=16, weight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#ECEEF0", lw=.55)
        ax.margins(x=.09, y=.12)
        ax = fig.add_axes(heat_positions[k])
        g = tables["group_neural_means"].query("genus == @genus")
        matrix = g.pivot(index="neuron", columns="rank_group", values="centered_mean").loc[list(NEURONS), GROUPS]
        ax.pcolormesh(np.arange(4) - .5, np.arange(14) - .5, matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(xlim=(-.5, 2.5), ylim=(12.5, -.5))
        counts = s.rank_group.value_counts()
        ax.set_yticks(range(13), NEURONS)
        ax.set_xticks(range(3), [f"{group}\n(n = {counts[group]})" for group in GROUPS])
        for tick, group in zip(ax.get_xticklabels(), GROUPS):
            tick.set_color(GROUP_COLORS[group])
        ax.tick_params(length=0, pad=5)
        ax.set_title("Observed neural deviations", pad=11, fontsize=12)
        ax.set_xlabel("Chemical-state thirds", labelpad=8)
        ax.text(-.22, 1.02, "CD"[k], transform=ax.transAxes, fontsize=16, weight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 13, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", length=0)
    cax = fig.add_axes([.365, .055, .30, .013])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r"), cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
    cb.set_label("Observed unit coefficient − genus mean", fontsize=10, labelpad=4)
    cb.solids.set_rasterized(False)
    cb.outline.set_visible(False)
    return fig


def plot_individual_support(tables, genus):
    """Every strain, selected chemical annotation and neural coordinate, ordered by score."""
    s = tables["strain_scores"].query("genus == @genus").sort_values(["chemical_score", "strain"])
    chem = tables["selected_chemical_z"].query("genus == @genus")
    members = tables["selected_members"].query("genus == @genus").metabolite.tolist()
    z = chem.pivot(index="metabolite", columns="strain", values="z").loc[members, s.strain]
    neural = s.set_index("strain")[[f"unit_{n}" for n in NEURONS]].T
    neural = neural.sub(neural.mean(axis=1), axis=0)
    fig = plt.figure(figsize=(13.8, max(8.6, 4.8 + .19 * len(members))))
    gs = fig.add_gridspec(3, 1, left=.32, right=.895, bottom=.18, top=.90, height_ratios=[1.5, len(members), 13], hspace=.16)
    top = fig.add_subplot(gs[0])
    top.plot(range(len(s)), s.chemical_score, color="#3A4247", lw=1)
    top.scatter(range(len(s)), s.chemical_score, c=s.rank_group.map(GROUP_COLORS), s=20)
    top.set(xlim=(-.5, len(s) - .5), ylabel="Score", xticks=[])
    score_limits = [float(s.chemical_score.min()), float(s.chemical_score.max())]
    top.set_yticks(score_limits, [f"{value:.1f}" for value in score_limits], fontsize=9)
    top.margins(y=.35)
    top.spines["bottom"].set_visible(False)
    for slot, values, names, label in [(1, z.to_numpy(), members, "Chemical log2 concentration (z)"),
                                       (2, neural.to_numpy(), NEURONS, "Unit coefficient − genus mean")]:
        ax = fig.add_subplot(gs[slot])
        vmax = np.ceil(np.max(np.abs(values)) * 10) / 10
        im = ax.pcolormesh(np.arange(values.shape[1] + 1) - .5, np.arange(values.shape[0] + 1) - .5,
                          values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set(xlim=(-.5, values.shape[1] - .5), ylim=(values.shape[0] - .5, -.5))
        visible_names = [name.replace("（", "(").replace("）", ")") for name in names]
        ax.set_yticks(range(len(names)), visible_names, fontsize=8 if slot == 1 else 9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        boundary = 0
        for group in GROUPS[:-1]:
            boundary += (s.rank_group == group).sum()
            ax.axvline(boundary - .5, color="#3B4146", lw=1.2)
        if slot == 2:
            labels = [f'{row.strain}  {row.species.split(" ", 1)[-1]}' for row in s.itertuples()]
            ax.set_xticks(range(len(s)), labels, rotation=65, ha="right", fontsize=8)
            ax.set_xlabel("Every strain, ordered by chemical-state score")
        else:
            ax.set_xticks([])
        box = ax.get_position()
        cax = fig.add_axes([.913, box.y0 + box.height * .1, .01, box.height * .8])
        cb = fig.colorbar(im, cax=cax)
        cb.solids.set_rasterized(False)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)
    fig.suptitle(f"{genus}: individual support", fontsize=16, weight="bold", y=.968)
    fig.text(.625, .932, f"All {len(s)} strains · all selected chemical annotations · all 13 neurons", ha="center", fontsize=11)
    return fig


def plot_member_covariance(z, title="Selected chemical annotations"):
    """Chemical-only Pearson covariance structure (shown as correlation)."""
    correlation = z.corr()
    fig, ax = plt.subplots(figsize=(8.2, 7.4), layout="constrained")
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [f"{i + 1:02d}" for i in range(len(z.columns))]
    ax.set_xticks(range(len(short)), short, rotation=90, fontsize=8)
    ax.set_yticks(range(len(short)), [f"{n}  {m}" for n, m in zip(short, z.columns)], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=.7, label="Pearson r")
    return fig


def plot_score_contributions(z_row, weights, strain):
    """One deterministic example: each weighted z value sums to the score."""
    contribution = (z_row * weights).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, .23 * len(contribution))), layout="constrained")
    ax.barh(contribution.index, contribution, color=np.where(contribution >= 0, GROUP_COLORS["High"], GROUP_COLORS["Low"]))
    ax.axvline(0, color="#53595D", lw=.6)
    ax.set(xlabel="Weighted standardized contribution", title=f"Chemical-state score contributions: {strain}")
    ax.tick_params(axis="y", labelsize=8)
    return fig


def plot_neural_effects(slopes):
    """Observed-state fitted changes across one chemical-score IQR; descriptive."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, layout="constrained")
    for ax, genus in zip(axes, GENERA):
        values = slopes[slopes.genus.eq(genus)].set_index("neuron").loc[list(NEURONS), "iqr_effect"]
        ax.barh(list(NEURONS), values, color=np.where(values >= 0, GROUP_COLORS["High"], GROUP_COLORS["Low"]))
        ax.invert_yaxis()
        ax.axvline(0, color="#53595D", lw=.6)
        ax.set(title=genus, xlabel="Fitted unit-coordinate change over score IQR")
    return fig


def plot_cached_performance(performance, genus="Bifidobacterium"):
    """Saved internal leaveout results; positive values mean lower pooled SSE."""
    block = performance[performance.genus.eq(genus)]
    order = list(NEURONS) + ["all13"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for offset, scheme, color, label in [(-.18, "leave_one_strain_out", "#347FA3", "Strain leaveout"),
                                         (.18, "leave_one_recorded_species_out", "#C66B36", "Recorded-species leaveout")]:
        v = block[block.scheme.eq(scheme)].set_index("neuron").loc[order, "error_improvement"]
        ax.barh(np.arange(len(order)) + offset, 100 * v, height=.34, color=color, label=label)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.axvline(0, color="#53595D", lw=.7)
    ax.set(xlabel="Pooled SSE improvement over training mean (%)", title=f"{genus}: cached internal checks")
    ax.legend(frameon=False, loc="lower right")
    return fig
