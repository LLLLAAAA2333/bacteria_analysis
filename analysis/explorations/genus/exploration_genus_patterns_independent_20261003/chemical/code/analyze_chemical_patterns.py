"""Chemical-only descriptive genus patterns; callable from an existing Notebook.

Inputs are 106 strains x 162 complete log2 concentrations; multi-strain analysis
uses 90 strains x 162 annotations and 13 equal-weight genus reference centers.
No neural data are read. See ../protocol.md for the fixed analysis decisions.
"""
from pathlib import Path
import hashlib
import json
import platform
import shutil

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOL = 1e-12
CUT_DISTANCE = 0.5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_rows(frame, metric="euclidean"):
    if len(frame) < 2:
        return frame.index.tolist()
    tree = linkage(pdist(frame.to_numpy(), metric=metric), method="average", optimal_ordering=True)
    return frame.index[leaves_list(tree)].tolist()


def weighted_corr(values, weights):
    """Population weighted Pearson correlation across observations (rows)."""
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    centered = values - (weights[:, None] * values).sum(axis=0)
    covariance = (centered * weights[:, None]).T @ centered
    denominator = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > TOL)


def distributions(values, genus):
    """One record per genus/column, preserving full denominator and zero counts."""
    records = []
    for name in sorted(genus.unique()):
        block = values.loc[genus == name]
        for feature in values:
            v = block[feature].to_numpy()
            records.append(dict(genus=name, feature=feature, n=len(v), mean=v.mean(), sd=v.std(ddof=1),
                                minimum=v.min(), q25=np.quantile(v, .25), median=np.median(v),
                                q75=np.quantile(v, .75), maximum=v.max(), n_positive=int((v > TOL).sum()),
                                n_negative=int((v < -TOL).sum()), n_zero=int((abs(v) <= TOL).sum())))
    return pd.DataFrame(records)


def discover_modules(centers_z, metadata):
    corr = np.corrcoef(centers_z.to_numpy().T)
    distance = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
    labels = fcluster(tree, t=CUT_DISTANCE, criterion="distance")
    groups = []
    ungrouped = []
    for label in sorted(set(labels)):
        members = centers_z.columns[labels == label].tolist()
        if len(members) >= 3 and metadata.loc[members, "family"].nunique() >= 3:
            groups.append(members)
        else:
            ungrouped.extend(members)
    groups.sort(key=lambda members: (-len(members), min(members)))
    modules = {f"C{i:02d}": members for i, members in enumerate(groups, 1)}
    feature_order = centers_z.columns[leaves_list(tree)].tolist()
    return modules, ungrouped, feature_order, corr


def module_weights(members, metadata):
    families = metadata.loc[members, "family"]
    counts = families.value_counts()
    return pd.Series([1 / (len(counts) * counts[f]) for f in families], index=members)


def save_heatmap(data, path, title, color_label, rowlabels=None, vlim=None, figsize=None, cmap="RdBu_r"):
    if vlim is None:
        vlim = float(np.ceil(np.nanmax(np.abs(data.to_numpy())) * 2) / 2)
    if figsize is None:
        figsize = (12.2, max(4.5, 2.9 + .42 * len(data)))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.to_numpy(), aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_xticks(np.arange(data.shape[1]), data.columns, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(np.arange(data.shape[0]), rowlabels or data.index, fontsize=10)
    ax.set_title(title, loc="left", fontsize=14, pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=.024, pad=.025, label=color_label)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return vlim


def make_figures(output, centers_z, scores, module_centers, modules, metadata, context, genus_order,
                 module_order, feature_order, consistency, member_table):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.titlesize": 13})
    figs = output / "figures"
    counts = context.genus.value_counts()
    genus_labels = {g: f"{g} (n={counts[g]})" for g in genus_order}
    display = module_centers.loc[genus_order, module_order].T
    display.columns = [genus_labels[g] for g in genus_order]
    labels = [f"{m}  |  {len(modules[m])} annotations / {metadata.loc[modules[m], 'family'].nunique()} families" for m in module_order]
    vlim = save_heatmap(display, figs / "01_chemical_module_centers", "Chemical combinations across genera",
                       "Family-weighted standardized mean", rowlabels=labels)
    strain_order = [s for g in genus_order for s in sorted(context.index[context.genus == g])]
    sample_scores = scores.loc[strain_order, module_order].T
    fig, ax = plt.subplots(figsize=(15.5, max(4.8, .43 * len(module_order) + 2.5)))
    im = ax.imshow(sample_scores, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
    boundaries = np.cumsum([counts[g] for g in genus_order])
    midpoints = (np.r_[0, boundaries[:-1]] + boundaries - 1) / 2
    ax.set_xticks(midpoints, [genus_labels[g] for g in genus_order], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(module_order)), module_order)
    for b in boundaries[:-1]:
        ax.axvline(b - .5, color="white", lw=1.5)
    ax.set_title("Chemical combinations in individual strains", loc="left", pad=14)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=.019, pad=.018, label="Family-weighted standardized score")
    fig.tight_layout()
    fig.savefig(figs / "02_chemical_modules_all_strains.png", dpi=180)
    fig.savefig(figs / "02_chemical_modules_all_strains.svg")
    plt.close(fig)

    fractions = consistency.pivot(index="module", columns="genus", values="strain_direction_fraction").loc[module_order, genus_order]
    fig, ax = plt.subplots(figsize=(12.2, max(4.5, .43 * len(module_order) + 2.8)))
    im = ax.imshow(fractions, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(genus_order)), [genus_labels[g] for g in genus_order], rotation=55, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(module_order)), module_order)
    ax.set_title("Strains matching their genus center's direction", loc="left", pad=14)
    for row in range(len(module_order)):
        for col in range(len(genus_order)):
            value = fractions.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if value >= .7 else "black")
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=.024, pad=.025, label="Fraction of all strains in genus")
    fig.tight_layout()
    fig.savefig(figs / "03_module_strain_direction.png", dpi=180)
    fig.savefig(figs / "03_module_strain_direction.svg")
    plt.close(fig)

    # Exhaustive inspection resource: all features, including those not assigned to modules.
    mapping = member_table.set_index("metabolite").module.to_dict()
    chunks = np.array_split(feature_order, 3)
    max_abs = float(np.ceil(np.abs(centers_z.to_numpy()).max() * 2) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(28, 20), constrained_layout=True)
    for ax, chunk in zip(axes, chunks):
        frame = centers_z.loc[genus_order, list(chunk)].T
        im = ax.imshow(frame, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, interpolation="nearest")
        labels = [f"{f} [{mapping.get(f, 'U')}]".replace("（", "(").replace("）", ")") for f in chunk]
        ax.set_yticks(range(len(chunk)), labels, fontsize=6.8)
        ax.set_xticks(range(len(genus_order)), genus_order, rotation=70, ha="right", fontsize=7.8)
        ax.tick_params(length=0)
    fig.suptitle("All 162 chemical annotations: genus-centered standardized means", fontsize=16)
    fig.colorbar(im, ax=axes, fraction=.011, pad=.01, label="Standardized genus mean; U = ungrouped")
    fig.savefig(figs / "04_all_162_chemical_centers.png", dpi=140)
    fig.savefig(figs / "04_all_162_chemical_centers.svg")
    plt.close(fig)
    return strain_order, vlim


def run_analysis(repo_root, out=None):
    """Run this chemical analysis into a NEW output folder; refuse existing results.

    `out` is an optional path. Existing stored results are viewable with the
    parent report's code/notebook_cells.py and must not be silently overwritten.
    """
    repo = Path(repo_root).resolve()
    source = repo / "reports/exploration_chemical_pattern_direct_report_20261003/tables"
    canonical = repo / "reports/exploration_genus_patterns_independent_20261003/chemical"
    output = Path(out).resolve() if out is not None else canonical
    if (output / "manifest.json").exists() or any((output / "tables").glob("*.csv")):
        raise FileExistsError(f"Refusing to overwrite existing chemical results: {output}; choose a new out directory.")
    for name in ["tables", "figures"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    if output != canonical:
        shutil.copyfile(canonical / "protocol.md", output / "protocol.md")
    tables = output / "tables"
    paths = {name: source / name for name in ["fresh_chemical_log2.csv", "fresh_feature_metadata.csv", "sample_context.csv"]}
    all_log = pd.read_csv(paths["fresh_chemical_log2.csv"], index_col="strain")
    all_context = pd.read_csv(paths["sample_context.csv"], index_col="strain")
    metadata = pd.read_csv(paths["fresh_feature_metadata.csv"]).set_index("metabolite")
    assert all_log.index.is_unique and all_log.columns.is_unique and metadata.index.is_unique and all_context.index.is_unique
    assert set(all_log.index) == set(all_context.index)
    assert set(all_log.columns) == set(metadata.index)
    assert all_log.shape == (106, 162) and np.isfinite(all_log.to_numpy()).all()
    metadata = metadata.loc[all_log.columns]
    assert metadata.family.notna().all()
    all_context = all_context.loc[all_log.index]
    counts = all_context.genus.value_counts().sort_index()
    coverage = counts.rename("n").to_frame()
    coverage["included_main"] = coverage.n >= 2
    coverage.to_csv(tables / "genus_coverage.csv")
    keep = all_context.genus.isin(counts.index[counts >= 2])
    context = all_context.loc[keep].copy()
    log = all_log.loc[context.index].copy()
    genus = context.genus
    assert log.shape == (90, 162) and genus.nunique() == 13
    centers = log.groupby(genus).mean()
    reference = centers.mean(axis=0)
    scale = centers.std(axis=0, ddof=1)
    valid = scale > TOL
    retained = scale.index[valid]
    strain_z = (log[retained] - reference[retained]) / scale[retained]
    centers_z = (centers[retained] - reference[retained]) / scale[retained]
    centers_delta = centers - reference
    modules, ungrouped, feature_order, genus_corr = discover_modules(centers_z, metadata)
    assert modules, "Fixed clustering produced no eligible modules; preserve results and redesign visualization only."
    weights = {m: module_weights(members, metadata) for m, members in modules.items()}
    scores = pd.DataFrame({m: strain_z[members] @ weights[m] for m, members in modules.items()})
    scores.index.name = "strain"
    module_centers = scores.groupby(genus).mean()
    genus_order = order_rows(centers_z)
    module_order = order_rows(module_centers.T, "correlation")

    # Save original scales before compact representations.
    context.to_csv(tables / "sample_context_main.csv")
    log.to_csv(tables / "strain_log2_90x162.csv")
    (log - reference).to_csv(tables / "strain_log2_difference_from_equal_genus_reference.csv")
    strain_z.to_csv(tables / "strain_standardized_90x162.csv")
    centers.to_csv(tables / "genus_mean_log2_13x162.csv")
    centers.T.to_csv(tables / "feature_genus_mean_log2_162x13.csv")
    centers_delta.T.to_csv(tables / "feature_genus_log2_difference_162x13.csv")
    centers_z.T.to_csv(tables / "feature_genus_standardized_162x13.csv")
    reference_info = pd.DataFrame({"equal_genus_reference_log2": reference, "between_genus_sd_log2": scale,
                                   "genus_mean_range_log2": centers.max() - centers.min(), "included_standardized": valid})
    reference_info.index.name = "metabolite"
    reference_info.join(metadata).to_csv(tables / "feature_reference_scale_metadata.csv")
    distributions(log, genus).to_csv(tables / "genus_feature_raw_log2_distribution.csv", index=False)
    distributions(strain_z, genus).to_csv(tables / "genus_feature_standardized_distribution.csv", index=False)
    scores.to_csv(tables / "strain_module_scores.csv")
    module_centers.to_csv(tables / "genus_module_centers.csv")
    distributions(scores, genus).rename(columns={"feature": "module"}).to_csv(tables / "genus_module_distribution.csv", index=False)

    member_rows = []
    for m, members in modules.items():
        for feature in members:
            member_rows.append({"module": m, "metabolite": feature, "family": metadata.loc[feature, "family"],
                                "score_weight": weights[m][feature], "SuperClass": metadata.loc[feature, "SuperClass"],
                                "Class": metadata.loc[feature, "Class"], "SubClass": metadata.loc[feature, "SubClass"]})
    members_df = pd.DataFrame(member_rows)
    members_df.to_csv(tables / "module_members.csv", index=False)
    representative_rows = []
    for m, members in modules.items():
        profile = module_centers[m].to_numpy()
        ranked = sorted([(f, np.corrcoef(centers_z[f].to_numpy(), profile)[0, 1]) for f in members],
                        key=lambda item: (-item[1], item[0]))
        for rank, (feature, correlation) in enumerate(ranked, 1):
            representative_rows.append(dict(module=m, rank=rank, metabolite=feature,
                                            correlation_with_module_genus_profile=correlation,
                                            representative_top3=rank <= 3))
    pd.DataFrame(representative_rows).to_csv(tables / "module_representative_annotations.csv", index=False)
    metadata.loc[ungrouped].to_csv(tables / "ungrouped_feature_metadata.csv")

    consistency_rows, loo_rows = [], []
    for g in centers.index:
        strain_ids = context.index[genus == g]
        for m, members in modules.items():
            values = scores.loc[strain_ids, m].to_numpy()
            center = module_centers.loc[g, m]
            sign = 0 if abs(center) <= TOL else int(np.sign(center))
            matching = sign * values > TOL
            member_matching = sign * centers_z.loc[g, members].to_numpy() > TOL
            loo = (values.sum() - values) / (len(values) - 1)
            # Changing one genus center also changes the 13-genus reference.
            loo_reference_reestimated = loo - (loo - center) / len(centers)
            loo_same = sign != 0 and bool(np.all(sign * loo_reference_reestimated > TOL))
            strain_fraction = matching.mean() if sign else np.nan
            member_fraction = np.sum(weights[m].to_numpy() * member_matching) if sign else np.nan
            supported = bool(sign and strain_fraction >= .75 and member_fraction >= .75 and loo_same)
            consistency_rows.append(dict(genus=g, module=m, n=len(values), center=center, direction=sign,
                                         n_strains_same_direction=int(matching.sum()) if sign else 0,
                                         strain_direction_fraction=strain_fraction,
                                         member_direction_fraction_unweighted=member_matching.mean() if sign else np.nan,
                                         member_direction_fraction_family_weighted=member_fraction,
                                         loo_min=loo_reference_reestimated.min(), loo_max=loo_reference_reestimated.max(),
                                         loo_all_same_direction=loo_same, direction_supported=supported,
                                         direction_supported_and_abs_score_ge_half=supported and abs(center) >= .5))
            for strain, value in zip(strain_ids, loo_reference_reestimated):
                loo_rows.append(dict(genus=g, module=m, omitted_strain=strain, loo_center=value))
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(tables / "module_direction_consistency.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(tables / "module_leave_one_strain_out.csv", index=False)

    obs_weights = genus.map(lambda g: 1 / (len(centers) * counts[g])).to_numpy()
    residual = log[retained] - centers.loc[genus, retained].set_axis(log.index)
    individual_corr = weighted_corr(log[retained].to_numpy(), obs_weights)
    within_corr = weighted_corr(residual.to_numpy(), obs_weights)
    pair_rows = []
    index = {f: i for i, f in enumerate(retained)}
    for m, members in modules.items():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                f1, f2 = members[a], members[b]
                i, j = index[f1], index[f2]
                pair_rows.append(dict(module=m, feature_1=f1, feature_2=f2,
                                      cross_family=metadata.loc[f1, "family"] != metadata.loc[f2, "family"],
                                      between_genus_r=genus_corr[i, j],
                                      individual_equal_genus_r=individual_corr[i, j], within_genus_equal_genus_r=within_corr[i, j]))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(tables / "module_pair_correlations.csv", index=False)
    summaries = []
    for m, members in modules.items():
        base = dict(module=m, n_annotations=len(members), n_families=metadata.loc[members, "family"].nunique(),
                    log2_genus_mean_range_median=reference_info.loc[members, "genus_mean_range_log2"].median(),
                    log2_between_genus_sd_median=reference_info.loc[members, "between_genus_sd_log2"].median())
        for scope, pp in [("all_pairs", pairs[pairs.module == m]), ("cross_family_pairs", pairs[(pairs.module == m) & pairs.cross_family])]:
            record = {**base, "pair_scope": scope, "n_pairs": len(pp)}
            for col in ["between_genus_r", "individual_equal_genus_r", "within_genus_equal_genus_r"]:
                vals = pp[col].dropna()
                record.update({col + "_median": vals.median(), col + "_q25": vals.quantile(.25),
                               col + "_q75": vals.quantile(.75), col + "_minimum": vals.min(),
                               col + "_fraction_positive": (vals > 0).mean()})
            summaries.append(record)
    pd.DataFrame(summaries).to_csv(tables / "module_summary.csv", index=False)
    composition = members_df.groupby(["module", "SuperClass"], dropna=False).agg(n_annotations=("metabolite", "size"), score_weight=("score_weight", "sum")).reset_index()
    composition.to_csv(tables / "module_superclass_composition.csv", index=False)
    within_var = strain_z.groupby(genus).var(ddof=1).mean()
    amplitude = pd.DataFrame({"between_genus_sd_log2": scale[retained], "equal_genus_rms_within_sd_log2": np.sqrt(log[retained].groupby(genus).var(ddof=1).mean()),
                              "within_variance_on_z_scale": within_var})
    amplitude.to_csv(tables / "between_and_within_feature_amplitude.csv")
    strain_order, main_vlim = make_figures(output, centers_z, scores, module_centers, modules, metadata, context,
                                          genus_order, module_order, feature_order, consistency, members_df)
    orders = dict(genus_order=genus_order, module_order=module_order, feature_order=feature_order,
                  strain_order=strain_order, modules=modules, ungrouped=ungrouped)
    (output / "orders.json").write_text(json.dumps(orders, indent=2) + "\n")
    manifest = dict(inputs={k: {"path": str(v), "sha256": sha256(v)} for k, v in paths.items()},
                    protocol_sha256=sha256(output / "protocol.md"), code_sha256=sha256(Path(__file__)),
                    dimensions={"all_strains": len(all_log), "main_strains": len(log), "features": log.shape[1],
                                "multi_strain_genera": len(centers), "modules": len(modules), "ungrouped_features": len(ungrouped)},
                    parameters={"correlation": "Pearson", "linkage": "average", "module_cut_distance": CUT_DISTANCE,
                                "min_annotations": 3, "min_families": 3, "numerical_zero": TOL,
                                "direction_fraction_cut": .75, "module_score": "equal family then equal member", "scale_ddof": 1,
                                "main_heatmap_symmetric_limit": main_vlim},
                    versions={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                    non_neural_input_assertion="Only the three listed chemical/context files are read as data.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "figures/01_chemical_module_centers_caption.txt").write_text(
        "All modules from fixed average-linkage clustering of 13 genus chemical centers (Pearson distance cut 0.5; >=3 annotations and >=3 Mass-column families). "
        "Scores average standardized log2 concentration within families, then equally across families. Reference and scale use 13 equally weighted genus centers; colors are dimensionless and do not show concentration or log-fold change. "
        "Genus order uses all 162 chemical profiles; no neural values influence grouping or ordering. Means do not imply all strains agree; see individual-strain and direction-fraction figures. All 162 annotations, including ungrouped entries, remain available in figure 04 and full tables. "
        "Module discovery and description use the same chemical data; findings are exploratory.\n")
    (output / "figures/02_chemical_modules_all_strains_caption.txt").write_text(
        "Every one of 90 strains is one column, grouped by genus in chemical-only order and ordered alphabetically by strain ID within genus. "
        "Module scores use the same reference, scales, family weights and symmetric color limits as figure 01; individual values exceeding the main figure limits saturate. "
        "All exact values and strain order are exported. Uneven widths reflect observed strain counts, not genus importance in the reference.\n")
    (output / "figures/03_module_strain_direction_caption.txt").write_text(
        "Each cell is the fraction of all strains in the genus whose module score has the same sign as that genus's mean score relative to the equal-genus reference. "
        "Zero strain scores stay in the denominator and count as neither sign. This is descriptive agreement, not uncertainty or independent validation. Small n can give deceptively complete agreement.\n")
    (output / "figures/04_all_162_chemical_centers_caption.txt").write_text(
        "Exhaustive inspection heatmap of all 162 report annotations; bracketed labels indicate module, with U denoting ungrouped annotations. "
        "Feature and genus order use chemical data only. Colors are standardized genus means; raw log2 centers/differences and feature SDs are separately exported because normalization can magnify small differences.\n")
    print(json.dumps(manifest["dimensions"]))
    return output


if __name__ == "__main__":
    run_analysis(Path(__file__).resolve().parents[4])
