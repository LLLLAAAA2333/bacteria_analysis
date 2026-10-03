"""Small data/plot helpers for cell48; inference uses statsmodels in neural_chemical_mixed.py."""
from pathlib import Path
import hashlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_inputs(root, animal_responses, neuron, phase, chemical_profiles=None,
                   reliable_metabolites=None, group_directory=None):
    """Align strain IDs once; keep the same observations in the two model families."""
    root = Path(root)
    responses = animal_responses.loc[animal_responses.neuron_class.eq(neuron)].copy()
    if responses.empty or phase not in set(responses.phase):
        raise ValueError("Requested neuron/phase is absent from cell47")
    keys = ["sample_id", "date", "worm_key", "phase"]
    if responses.duplicated(keys).any():
        raise ValueError("Duplicate animal/strain/phase responses")
    for key in ("sample_id", "date", "worm_key"):
        responses[key] = responses[key].astype(str)
    responses["animal_id"] = responses.date + "/" + responses.worm_key
    neural_ids = pd.Index(sorted(responses.sample_id.unique()), name="sample_id")
    source = "in-memory chemical_profiles (already log2FC)"
    if chemical_profiles is None:
        fc = pd.read_excel(root / "data/matrix.xlsx", index_col=0)
        fc.index = fc.index.astype(str).str.strip()
        fc.columns = fc.columns.astype(str).str.strip()
        fc = fc.loc[neural_ids.intersection(fc.index)].apply(pd.to_numeric, errors="coerce")
        requested = fc.columns if reliable_metabolites is None else pd.Index(reliable_metabolites)
        unknown = requested.difference(fc.columns)
        if len(unknown):
            raise ValueError(f"Unknown reliable-metabolite names: {unknown.tolist()}")
        valid = (np.isfinite(fc) & fc.gt(0)).all(axis=0)
        chemical_profiles = np.log2(fc.loc[:, requested.intersection(fc.columns[valid], sort=False)])
        source = "matrix.xlsx: log2 once, positive finite FC across neural-paired strains"
    chemistry = chemical_profiles.copy()
    chemistry.index = chemistry.index.astype(str).str.strip()
    chemistry.columns = chemistry.columns.astype(str).str.strip()
    if not chemistry.index.is_unique or not chemistry.columns.is_unique or chemistry.empty:
        raise ValueError("Chemical sample IDs/features must be unique and nonempty")
    chemistry = chemistry.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(chemistry.to_numpy()).all():
        raise ValueError("chemical_profiles contains nonfinite values; resolve upstream feature selection")
    taxonomy = pd.read_excel(root / "data/GM300_bacteria_species_summary.xlsx", sheet_name="Axxx_species_mapping")
    taxonomy = taxonomy.rename(columns={"AID": "sample_id", "genus_clean": "genus"})
    taxonomy["sample_id"] = taxonomy.sample_id.astype(str).str.strip()
    taxonomy["genus"] = taxonomy.genus.astype("string").str.strip().replace("", pd.NA)
    if taxonomy.sample_id.duplicated().any():
        raise ValueError("Taxonomy has duplicate strain IDs")
    responses = responses.merge(taxonomy[["sample_id", "genus", "QC_flag"]], on="sample_id", how="left", validate="many_to_one")
    audit = pd.DataFrame(index=neural_ids)
    audit["has_chemistry"] = audit.index.isin(chemistry.index)
    audit["has_genus"] = audit.index.isin(taxonomy.loc[taxonomy.genus.notna(), "sample_id"])
    selected = responses.loc[responses.phase.eq(phase) & np.isfinite(responses.response)]
    audit["has_target_response"] = audit.index.isin(selected.sample_id)
    audit["included"] = audit.all(axis=1)
    audit["reason"] = audit.apply(lambda row: "; ".join(
        name for name in ("has_chemistry", "has_genus", "has_target_response") if not row[name]
    ), axis=1)
    ids = audit.index[audit.included]
    responses = responses.loc[responses.sample_id.isin(ids)].copy()
    selected = responses.loc[responses.phase.eq(phase) & np.isfinite(responses.response)].copy()
    windows = selected[["window_start_s", "window_stop_s"]].drop_duplicates()
    if len(windows) != 1:
        raise ValueError("Screening phase must have one fixed time window")
    selected = selected.sort_values(["sample_id", "date", "worm_key"]).reset_index(drop=True)
    selected["observation_id"] = np.arange(len(selected))
    chemistry = chemistry.loc[ids]
    features = pd.DataFrame({"feature_id": [f"F{i:04d}" for i in range(len(chemistry.columns))],
                             "metabolite": chemistry.columns})
    features["chemical_qc"] = ("unverified: numerical validity only" if reliable_metabolites is None
                                 else "user-specified reliable-metabolite list")
    features["block"] = pd.NA
    features["paired_check_status"] = "not available"
    if group_directory is not None:
        directory = Path(group_directory)
        members = pd.read_csv(directory / "feature_blocks_and_weights.csv")
        checks = pd.read_csv(directory / "paired_group_checks.csv")
        summary = pd.read_csv(directory / "group_summary.csv")
        group_ids = pd.read_csv(directory / "chemical_block_rdm_paired.csv", index_col=0).index
        if set(neural_ids) != set(group_ids):
            raise ValueError("Frozen chemical-group paired panel differs from cell47; choose the matching artifact")
        if members.metabolite.duplicated().any():
            raise ValueError("Frozen chemical blocks contain duplicate feature assignments")
        annotation = members[["metabolite", "block"]].merge(checks, on="block", validate="many_to_one")
        annotation = annotation.merge(summary[["block", "n_features"]], on="block", validate="many_to_one")
        member_lists = members.groupby("block").metabolite.agg(lambda names: " | ".join(names))
        annotation["block_members"] = annotation.block.map(member_lists)
        features = features.drop(columns=["block", "paired_check_status"]).merge(
            annotation, on="metabolite", how="left", validate="one_to_one")
    return selected, responses, chemistry, features, audit, source


def run_models(directory):
    from .neural_chemical_mixed import screen_chemicals
    return screen_chemicals(directory)


def file_provenance(path):
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}


def plot_results(results, responses, chemistry, features, residuals, animal_input, neuron,
                 phase, directory, n_examples=3):
    """Model effects and raw data are separate; raw scatter has no unadjusted OLS line."""
    directory = Path(directory)
    ordered = results.loc[results.q_wald.notna()].sort_values(["q_wald", "p_wald", "feature_id"], kind="stable")
    if ordered.empty:
        print("No estimable Wald results; inspect model_results.csv and the matching audit.")
        return []
    # Display selection only; it neither changes the FDR family nor validates candidates.
    examples = ordered.drop_duplicates("feature_id").copy()
    examples["display_group"] = examples.block.fillna(examples.feature_id)
    examples = examples.drop_duplicates("display_group").head(n_examples)
    top_ids = ordered.drop_duplicates("feature_id").head(12).feature_id.tolist()
    labels = features.set_index("feature_id").metabolite
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.4 * len(top_ids) + 1)), sharey=True, layout="constrained")
    for ax, model in zip(axes, ["overall", "within_genus"]):
        rows = results.loc[results.model.eq(model)].set_index("feature_id").reindex(top_ids)
        for y, (_, row) in enumerate(rows.iterrows()):
            if np.isfinite(row.beta) and np.isfinite(row.ci_low):
                color = "#26734D" if row.candidate else "#A46B36"
                ax.errorbar(row.beta, y, xerr=[[row.beta - row.ci_low], [row.ci_high - row.beta]],
                            fmt="o", color=color, markersize=4, capsize=2)
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set(title=model.replace("_", " "), xlabel=r"Slope: $\Delta F/F_0$ per log$_2$FC")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(top_ids)), labels=[textwrap.fill(labels[x], 34) for x in top_ids], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Wald 95% intervals (unadjusted, selected after screening)\nGreen: BY hit passing model flags; brown: review / no BY hit", fontsize=11)
    fig.savefig(directory / "effects.png", dpi=150)
    plt.show()

    dates = sorted(responses.date.unique())
    colors = {date: plt.get_cmap("tab10")(i % 10) for i, date in enumerate(dates)}
    phases = [phase] + (["post"] if phase != "post" and "post" in set(responses.phase) else [])
    fig, axes = plt.subplots(len(examples), len(phases), figsize=(13, 3.8 * len(examples)),
                             squeeze=False, layout="constrained")
    for i, example in enumerate(examples.itertuples()):
        for j, current_phase in enumerate(phases):
            ax = axes[i, j]
            d = responses.loc[responses.phase.eq(current_phase)].copy()
            d["chemical"] = d.sample_id.map(chemistry[example.metabolite])
            for date, rows in d.groupby("date"):
                ax.scatter(rows.chemical, rows.response, s=9, alpha=0.3, color=colors[date], linewidths=0, label=date)
            per_date = d.groupby(["sample_id", "date"])[["chemical", "response"]].mean()
            means = per_date.groupby("sample_id").mean()
            ax.scatter(means.chemical, means.response, s=22, facecolors="none", edgecolors="black", linewidths=0.6)
            for label_index, strain in enumerate(means.response.abs().nlargest(3).index):
                row = means.loc[strain]
                ax.annotate(strain, (row.chemical, row.response), xytext=(4, [8, 22, -18][label_index]),
                            textcoords="offset points", fontsize=7,
                            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.4})
            window = d[["window_start_s", "window_stop_s"]].iloc[0]
            ax.axhline(0, color="#AAAAAA", lw=0.6)
            ax.set(title=f"{neuron} [{window.iloc[0]:g}, {window.iloc[1]:g}) s"
                         + (" — screened" if current_phase == phase else " — descriptive only"),
                   xlabel=textwrap.fill(example.metabolite, 55) + " (log2FC)", ylabel=r"Animal mean $\Delta F/F_0$")
            ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=5, fontsize=8,
               title="Colored dots: animals by date; open circles: date-equal strain means")
    fig.suptitle("Exploratory examples: lowest BY q across distinct chemical blocks\nRaw scatter is not the adjusted-model fit; QC and model flags remain applicable", fontsize=11)
    fig.savefig(directory / "candidate_scatter.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(len(examples), 2, figsize=(11, 3.3 * len(examples)), squeeze=False, layout="constrained")
    for i, example in enumerate(examples.itertuples()):
        for ax, model in zip(axes[i], ["overall", "within_genus"]):
            d = residuals.loc[residuals.feature_id.eq(example.feature_id) & residuals.model.eq(model)]
            ax.scatter(d.fitted, d.standardized_residual, s=8, alpha=0.4, color="#446784")
            ax.axhline(0, color="#777777", lw=0.6)
            status = results.loc[results.feature_id.eq(example.feature_id) & results.model.eq(model), "status"].iloc[0]
            ax.set(title=f"{textwrap.shorten(example.metabolite, width=40)}\n{model}: {status}",
                   xlabel="Conditional fitted response", ylabel="Residual / fitted residual SD")
    fig.suptitle("Residual patterns: Wald inference assumes the specified covariance model", fontsize=11)
    fig.savefig(directory / "candidate_residuals.png", dpi=150)
    plt.show()
    return examples.feature_id.tolist()
