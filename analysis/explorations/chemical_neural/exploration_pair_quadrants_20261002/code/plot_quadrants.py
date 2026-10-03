"""Plot the saved four-corner exploration tables; do not recompute distances.

Call ``plot_all(output_directory)`` from a Notebook, or run this file to plot
the tables in its parent output directory. Every plotted pair is identified
in ``tables/atlas_pair_order.csv``. Figure text is deliberately descriptive.
"""

from pathlib import Path
import json
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


CATEGORIES = ("Cnear_Nnear", "Cnear_Nfar", "Cfar_Nnear", "Cfar_Nfar")
COLORS = ("#277DA8", "#C66A27", "#9664AF", "#398866")
LABELS = (
    "Chemical near / neural near",
    "Chemical near / neural far",
    "Chemical far / neural near",
    "Chemical far / neural far",
)
SHORT_LABELS = ("Near / near", "Near / far", "Far / near", "Far / far")


def _save(fig, directory, stem):
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _matrix(table, index, values, order):
    return table.pivot(index=index, columns="category", values=values).reindex(
        index=order, columns=CATEGORIES
    ).to_numpy(float)


def _draw_pair_map(pairs, summary, thresholds, figures, tail_fraction):
    fig = plt.figure(figsize=(11.4, 5.1))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1), wspace=0.17)
    ax, support = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    other = ~pairs.category.isin(CATEGORIES)
    ax.scatter(pairs.loc[other, "chemical"], pairs.loc[other, "neural"],
               s=7, color="#D7DADF", alpha=0.45, linewidths=0, rasterized=True)
    for category, color in zip(CATEGORIES, COLORS):
        subset = pairs.loc[pairs.category.eq(category)]
        ax.scatter(subset.chemical, subset.neural, s=10, color=color,
                   alpha=0.58, linewidths=0, rasterized=True)
    for key in ("chemical_near", "chemical_far"):
        ax.axvline(thresholds[key], color="#686E76", lw=0.75, ls=(0, (3, 3)))
    for key in ("neural_near", "neural_far"):
        ax.axhline(thresholds[key], color="#686E76", lw=0.75, ls=(0, (3, 3)))
    ax.set(xlabel="Chemical distance (RMS log₂FC difference)",
           ylabel="Neural distance (1 − cosine similarity)")
    ax.set_title("All sample pairs", loc="left", fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)
    support.set_axis_off()
    support.set(xlim=(0, 1), ylim=(0, 1))
    support.text(0, 1.025, "Four distance subsets", transform=support.transAxes,
                 fontsize=12, fontweight="bold", va="bottom")
    support.text(0.02, 0.90, "Chemical / neural", fontsize=9, color="#4D5259")
    for xpos, header in ((0.56, "Pairs"), (0.73, "Samples"), (0.93, "Same\nreference")):
        support.text(xpos, 0.90, header, fontsize=9, ha="center", va="bottom", color="#4D5259")
    summary = summary.set_index("category")
    for i, (category, label, color) in enumerate(zip(CATEGORIES, SHORT_LABELS, COLORS)):
        y = 0.76 - i * 0.17
        record = summary.loc[category]
        support.scatter([0.025], [y], s=47, color=color, clip_on=False)
        support.text(0.075, y, label, fontsize=10, va="center")
        support.text(0.56, y, f"{record.n_pairs:,.0f}", fontsize=10, ha="center", va="center")
        support.text(0.73, y, f"{record.n_samples:.0f}", fontsize=10, ha="center", va="center")
        value = record.get("same_reference_fraction", np.nan)
        support.text(0.93, y, f"{100 * value:.1f}%" if np.isfinite(value) else "—",
                     fontsize=10, ha="center", va="center")
    support.plot([0.015, 0.99], [0.85, 0.85], lw=0.6, color="#BFC4CB")
    tail_percent = f"{100 * tail_fraction:g}%"
    support.text(0.02, 0.085, f"Near: bottom {tail_percent}    Far: top {tail_percent}",
                 fontsize=9, color="#4D5259")
    return _save(fig, figures, "01_pair_map")


def _draw_characteristics(classes, features, selected, figures):
    class_order = (classes[["superclass", "n_features"]].drop_duplicates()
                   .sort_values(["n_features", "superclass"], ascending=[False, True]))
    class_names = class_order.superclass.tolist()
    feature_names = selected.sort_values("selection_rank").feature.tolist()
    class_values = _matrix(classes, "superclass", "mean_relative_contribution", class_names)
    feature_values = _matrix(features, "feature", "mean_relative_contribution", feature_names)
    coverage = _matrix(features, "feature", "both_reported_fraction", feature_names)
    show_coverage = not np.allclose(coverage, 1, equal_nan=False)
    all_positive = np.concatenate([class_values.ravel(), feature_values.ravel()])
    all_positive = all_positive[np.isfinite(all_positive) & (all_positive > 0)]
    lower = min(-1, np.floor(np.log2(all_positive.min())))
    upper = max(1, np.ceil(np.log2(all_positive.max())))
    norm = TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#EBEDF0")
    fig = plt.figure(figsize=(15.0 if show_coverage else 12.0, 7.0))
    # Explicit label columns prevent long compound names from colliding with
    # either heatmap. All contribution matrices use exactly the same mapping.
    grid = fig.add_gridspec(1, 5 if show_coverage else 4,
                           width_ratios=[3.0, 2.4, 3.2, 2.4] + ([2.4] if show_coverage else []),
                           wspace=0.09, left=0.025, right=0.98, top=0.91, bottom=0.24)
    label_class, ax_class = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    label_feature, ax_feature = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])
    images = []
    for axis, values in ((ax_class, class_values), (ax_feature, feature_values)):
        log_values = np.log2(np.maximum(values, 2.0 ** lower))
        images.append(axis.imshow(log_values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"))
        axis.set_xticks(range(4), SHORT_LABELS, fontsize=9, rotation=38, ha="right")
        axis.set_xlabel("Chemical / neural distance", fontsize=9, labelpad=7)
        axis.tick_params(axis="both", length=0, pad=6)
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
        for row in range(values.shape[0]):
            for col in range(4):
                value = values[row, col]
                if not np.isfinite(value):
                    continue
                rgba = cmap(norm(log_values[row, col]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(col, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=8, color="white" if luminance < 0.5 else "#23272B")
    for axis, values in ((label_class, class_values), (label_feature, feature_values)):
        axis.set(xlim=(0, 1), ylim=(len(values) - 0.5, -0.5))
        axis.set_axis_off()
    for row, record in enumerate(class_order.itertuples()):
        label = textwrap.fill(str(record.superclass).strip(), 33)
        label_class.text(0.98, row, f"{label} ({record.n_features})", fontsize=9,
                         ha="right", va="center", linespacing=1.0)
    for row, name in enumerate(feature_names):
        label_feature.text(0.98, row, textwrap.fill(name.strip(), 31), fontsize=9,
                           ha="right", va="center", linespacing=1.0)
    ax_class.set_title(f"All {class_order.n_features.sum()} features\nChemical classes", loc="left", fontsize=12, fontweight="bold", pad=16)
    ax_feature.set_title("Complete-report features\nHighlighted compounds", loc="left", fontsize=12, fontweight="bold", pad=16)
    cbar_axis = fig.add_axes([0.355, 0.075, 0.43, 0.022])
    bar = fig.colorbar(images[0], cax=cbar_axis, orientation="horizontal")
    ticks = np.unique(np.concatenate([np.arange(lower, 1, 2), np.arange(0, upper + 1)]))
    bar.set_ticks(ticks, labels=[f"{2.0 ** value:.3g}" for value in ticks])
    bar.ax.tick_params(length=3, labelsize=8)
    bar.set_label("Relative chemical contribution (uniform-feature baseline = 1)", fontsize=10, labelpad=6)
    if show_coverage:
        axis = fig.add_subplot(grid[4])
        coverage_image = axis.imshow(coverage * 100, cmap="Greys", vmin=0, vmax=100,
                                     aspect="auto", interpolation="nearest")
        axis.set_xticks(range(4), SHORT_LABELS, fontsize=8, rotation=38, ha="right")
        axis.set_yticks([])
        axis.set_title("Both reported", loc="left", fontsize=12, fontweight="bold", pad=16)
        axis.tick_params(length=0, pad=6)
        axis.spines[:].set_visible(False)
        coverage_bar = fig.colorbar(coverage_image, ax=axis, orientation="horizontal",
                                   fraction=0.045, pad=0.17)
        coverage_bar.set_label("Pair coverage (%)", fontsize=9)
    return _save(fig, figures, "02_chemical_characteristics"), bool(show_coverage)


def _draw_atlas(pairs, selected, arrays, figures, tables):
    feature_names = selected.sort_values("selection_rank").feature.tolist()
    all_features = arrays["features"].astype(str).tolist()
    feature_positions = [all_features.index(name) for name in feature_names]
    cell_names = arrays["cells"].astype(str).tolist()
    chemical = arrays["abs_chemical_difference"][:, feature_positions]
    neural = arrays["neural_contribution"].astype(float)
    neural_total = neural.sum(axis=1, keepdims=True)
    neural_share = np.divide(neural, neural_total, out=np.full_like(neural, np.nan),
                             where=neural_total > 0)
    blocks = [pairs.loc[pairs.category.eq(category)].sort_values(
        ["chemical", "neural", "strain_a", "strain_b"], kind="stable") for category in CATEGORIES]
    selected_rows = np.concatenate([block.index.to_numpy() for block in blocks])
    chemical_norm = Normalize(vmin=0, vmax=float(np.nanmax(chemical[selected_rows])))
    neural_norm = Normalize(vmin=0, vmax=float(np.nanmax(neural_share[selected_rows])))
    height_ratios = np.array([max(len(block), 120) for block in blocks], float)
    height = max(13, 4.0 + height_ratios.sum() / 110)
    fig = plt.figure(figsize=(13.0, height))
    grid = fig.add_gridspec(4, 2, width_ratios=(len(feature_names), len(cell_names)),
                           height_ratios=height_ratios, hspace=0.13, wspace=0.10,
                           left=0.075, right=0.97, top=0.94, bottom=0.145)
    pair_order = []
    for section, (block, label, color) in enumerate(zip(blocks, LABELS, COLORS)):
        axes = [fig.add_subplot(grid[section, column]) for column in range(2)]
        rows = block.index.to_numpy()
        first_image = axes[0].imshow(chemical[rows], cmap="magma_r", norm=chemical_norm,
                                    aspect="auto", interpolation="nearest", rasterized=True)
        second_image = axes[1].imshow(neural_share[rows], cmap="viridis", norm=neural_norm,
                                     aspect="auto", interpolation="nearest", rasterized=True)
        axes[0].set_title(f"{label}   ·   {len(block):,} pairs", loc="left", color=color,
                          fontsize=11, fontweight="bold", pad=9)
        for axis, labels in zip(axes, (feature_names, cell_names)):
            axis.set_yticks([0, len(block) - 1], ["1", f"{len(block):,}"])
            axis.tick_params(axis="y", labelsize=8, length=2)
            axis.set_xticks(range(len(labels)))
            axis.tick_params(axis="x", length=0)
            axis.spines[:].set_visible(False)
            if section == 3:
                axis.set_xticklabels([label.strip() for label in labels], rotation=58,
                                     ha="right", fontsize=8, rotation_mode="anchor")
            else:
                axis.set_xticklabels([])
        axes[0].set_ylabel("Pair row", fontsize=9)
        axes[1].set_yticklabels([])
        for display_row, record in enumerate(block.itertuples(), start=1):
            pair_order.append({"category": record.category, "row_in_category": display_row,
                               "pair_id": record.pair_id, "strain_a": record.strain_a,
                               "strain_b": record.strain_b, "chemical": record.chemical,
                               "neural": record.neural})
    fig.text(0.075, 0.976, "All pairs in the four distance subsets", fontsize=14,
              fontweight="bold", va="top")
    fig.text(0.075, 0.953, "Chemical feature differences", fontsize=11, va="bottom")
    fig.text(0.571, 0.953, "Neural distance contributions", fontsize=11, va="bottom")
    chemical_bar_axis = fig.add_axes([0.11, 0.025, 0.34, 0.012])
    neural_bar_axis = fig.add_axes([0.61, 0.025, 0.28, 0.012])
    chemical_bar = fig.colorbar(first_image, cax=chemical_bar_axis, orientation="horizontal")
    neural_bar = fig.colorbar(second_image, cax=neural_bar_axis, orientation="horizontal")
    chemical_bar.set_label("Absolute log₂FC difference", fontsize=10, labelpad=5)
    neural_bar.set_label("Share of pair's neural distance", fontsize=10, labelpad=5)
    for bar in (chemical_bar, neural_bar):
        bar.ax.tick_params(labelsize=9, length=3)
    pd.DataFrame(pair_order).to_csv(tables / "atlas_pair_order.csv", index=False)
    return _save(fig, figures, "03_all_selected_pairs")


def plot_all(out: Path):
    """Render the saved descriptive summaries and complete selected-pair atlas."""
    out = Path(out)
    tables = out / "tables"
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    params = json.loads((out / "parameters.json").read_text())
    pairs = pd.read_csv(tables / "pair_catalogue.csv")
    summary = pd.read_csv(tables / "category_summary.csv")
    classes = pd.read_csv(tables / "class_summary.csv")
    features = pd.read_csv(tables / "feature_summary.csv")
    selected = pd.read_csv(tables / "selected_features.csv")
    arrays = np.load(tables / "pair_arrays.npz")
    if "pair_ids" in arrays:
        assert np.array_equal(pairs.pair_id.astype(str), arrays["pair_ids"].astype(str)), "Array pair order differs"
    paths = []
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.edgecolor": "#454B53",
                         "axes.labelcolor": "#23272B", "text.color": "#23272B",
                         "xtick.color": "#454B53", "ytick.color": "#454B53"}):
        paths.extend(_draw_pair_map(pairs, summary, params["primary_thresholds"], figures,
                                    params["tail_fraction"]))
        new_paths, coverage_shown = _draw_characteristics(classes, features, selected, figures)
        paths.extend(new_paths)
        paths.extend(_draw_atlas(pairs, selected, arrays, figures, tables))
    coverage_sentence = ("Coverage is shown separately as the fraction of pairs with both raw values reported."
                         if coverage_shown else "All highlighted compounds are reported for both members of every pair; the constant 100% coverage panel is omitted.")
    quality_sentence = ""
    quality_path = tables / "chemical_distance_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        missing_share = quality.loc[quality.feature_scope.eq("one_sided_report_missing"), "mean_distance_share_pct"]
        complete_share = quality.loc[quality.feature_scope.eq("complete_162"), "mean_distance_share_pct"]
        quality_sentence = (
            f" In the all-feature chemical distance, one-sided missing raw reports account for "
            f"{missing_share.min():.1f}–{missing_share.max():.1f}% of squared distance on average within "
            f"categories. The complete-report 162-feature set accounts for {complete_share.min():.1f}–"
            f"{complete_share.max():.1f}%; highlighted compounds are a subset of this set. "
            "The all-feature and complete-report panels therefore cover different evidence. "
            "See the separate context-check figure (04)."
        )
    captions = f"""01_pair_map
All {len(pairs):,} unordered pairs among {params['n_samples']} matched samples. Chemical distance is the RMS difference of {params['n_features']} existing log2FC features; neural distance is 1 minus cosine similarity of the {params['n_cells']} signed template coefficients after the existing individual-SNR >= 0.5 filter. Near and far are the bottom and top {100 * params['tail_fraction']:g}% of the marginal all-pair distance distributions. Gray pairs are outside the four corner subsets. Samples are reused across pairs, so pair counts are not independent replicates. 'Same reference' is the percentage of pairs sharing the existing chemical reference group: chemical-far subsets are almost entirely cross-reference pairs, making reference-group structure a material limitation of their interpretation.

02_chemical_characteristics
Descriptive candidate summary for Figure 5, with four distance-defined pair groups. Each compound's squared log2FC difference is divided by that pair's mean squared difference across all {params['n_features']} features, then averaged equally over pairs; 1 is the uniform-feature baseline. Chemical classes average this quantity across their features, with class feature counts in parentheses. Both panels use the same log2 color mapping, ratio-labelled colorbar, and white baseline at 1; numeric values are untransformed ratios. The highlighted compounds are selected from the pre-existing complete-report/QC RSD <= 0.30 list, using the union of the top three contributions per category and the top four absolute log2 neural-far/near contribution contrasts within each chemical-distance band. Selection and description use the same data. {coverage_sentence}{quality_sentence} Feature contributions describe the existing chemical distance, not response direction, enrichment, association significance, or neural causation. Reference groups and repeated sample endpoints limit biological interpretation; endpoint-balanced and 20%/30% threshold sensitivity tables are supplied separately.

03_all_selected_pairs
Inspection atlas containing every pair in the four corner subsets. Within each category, rows are ordered by increasing chemical distance, neural distance, and sample identifiers; the exact mapping is saved in tables/atlas_pair_order.csv. Columns show all highlighted compounds and all {params['n_cells']} neural cells. Left: absolute log2FC differences, with a common scale across groups. Right: each cell's nonnegative contribution 0.5*(unit_vector_a-unit_vector_b)^2 divided by the pair's neural distance; shares sum to 1 for a nonzero distance. These are relative shares, so comparable color does not imply comparable total neural distance. A zero neural distance would have undefined shares and is shown as missing. No selected-pair rows are omitted or averaged. See the separate all-pair catalogue and full feature-summary tables for identifiers and the complete chemical feature set. This atlas is a data-inspection resource, not a presentation figure.
"""
    (figures / "plot_captions.txt").write_text(captions)
    return paths


if __name__ == "__main__":
    for output_path in plot_all(Path(__file__).resolve().parents[1]):
        print(output_path)
