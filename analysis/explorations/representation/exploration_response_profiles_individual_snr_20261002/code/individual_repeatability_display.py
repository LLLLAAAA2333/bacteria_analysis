"""Notebook-style repeatability display from saved split-half similarities.

This is a display-only change. Bin edges, values, separate density
normalizations and mean markers match the preceding tall histogram.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform


STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": "#132C4E", "axes.labelcolor": "#132C4E",
    "axes.edgecolor": "#132C4E", "xtick.color": "#132C4E", "ytick.color": "#132C4E",
    "axes.linewidth": .8, "axes.grid": False,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "figure.dpi": 110, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
}
BLUE, RED = "#5386CB", "#D95D67"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _histogram_summary(values, bins):
    valid = np.asarray(values)[np.isfinite(values)]
    if not len(valid):
        raise ValueError("Each similarity distribution requires a finite value")
    counts, _ = np.histogram(valid, bins=bins)
    density, _ = np.histogram(valid, bins=bins, density=True)
    if counts.sum() != len(valid):
        raise ValueError("Histogram edges would exclude a finite similarity")
    if not np.isclose(np.sum(density * np.diff(bins)), 1.):
        raise ValueError("Separate histogram density failed to integrate to one")
    return valid, dict(n=int(len(valid)), mean=float(valid.mean()),
                       minimum=float(valid.min()), maximum=float(valid.max()),
                       bin_counts=counts.tolist(), density=density.tolist())


def plot_repeatability(output_dir):
    """Restore Notebook 02 Panel B geometry without recomputing similarities."""
    out = Path(output_dir).resolve()
    tables, figures = out / "tables", out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    split_path = tables / "split_filtered_cosine.csv"
    raw_path = tables / "rdm_raw.csv"
    row_path = tables / "figure_row_order.csv"
    representation_path = out / "representation_parameters.json"
    source_paths = [split_path, raw_path, row_path, representation_path]
    source_hashes = {str(path): _hash(path) for path in source_paths}
    representation = json.loads(representation_path.read_text())
    split = pd.read_csv(split_path, index_col=0)
    ids = split.index.astype(str)
    split = split.loc[ids, ids]
    values = split.to_numpy(float)
    if not ids.is_unique or not np.allclose(values, values.T, equal_nan=True, atol=1e-12):
        raise ValueError("Saved split-half similarities must have unique IDs and be symmetric")

    # Recreate the existing raw-profile hierarchy only for display ordering.
    raw = pd.read_csv(raw_path, index_col=0).loc[ids, ids].to_numpy(float)
    if not np.isfinite(raw).all() or not np.allclose(raw, raw.T, atol=1e-12):
        raise ValueError("The saved raw-profile hierarchy requires a complete symmetric RDM")
    raw = np.clip((raw + raw.T) / 2, 0, 2)
    np.fill_diagonal(raw, 0.)
    tree = linkage(squareform(raw, checks=True), method="average")
    order = leaves_list(tree)
    ordered_ids = ids.take(order).tolist()
    saved_order = pd.read_csv(row_path)
    id_column = "strain" if "strain" in saved_order else "sample_id"
    if saved_order[id_column].tolist() != ordered_ids:
        raise ValueError("Dendrogram leaves differ from the saved profile order")

    # Exact previous edges: retain the full cosine range and every finite tail.
    bins = np.arange(-1., 1.0001, .05)
    same, same_summary = _histogram_summary(np.diag(values), bins)
    other, other_summary = _histogram_summary(values[np.triu_indices(len(ids), 1)], bins)
    archive = figures / "previous_tall_repeatability"
    archived = []
    for suffix in ("png", "svg"):
        previous = figures / f"02_repeatability_distribution.{suffix}"
        target = archive / previous.name
        if previous.exists() and not target.exists():
            archive.mkdir(exist_ok=True)
            shutil.copy2(previous, target)
        if target.exists():
            archived.append(str(target))

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.8, 6.4))
        fig.text(.018, .985, "B", fontsize=21, weight="bold", va="top")
        fig.text(.07, .978, "Cross-half response-pattern similarity",
                 fontsize=14, weight="bold", va="top")
        ax = fig.add_axes([.105, .17, .32, .64])
        top = fig.add_axes([.105, .82, .32, .075])
        left = fig.add_axes([.04, .17, .055, .64])
        tree_style = dict(no_labels=True, color_threshold=.7 * tree[:, 2].max(),
                          above_threshold_color="#687F99")
        for tree_axis, orientation in [(top, "top"), (left, "left")]:
            drawn = dendrogram(tree, ax=tree_axis, orientation=orientation, **tree_style)
            if not np.array_equal(drawn["leaves"], order):
                raise ValueError("Rendered tree leaves differ from the displayed matrix")
            tree_axis.axis("off")
            for collection in tree_axis.collections:
                collection.set_linewidth(.7)
        top.set_xlim(0, 10 * len(ids))
        left.set_ylim(10 * len(ids), 0)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#DCDCDC")
        image = ax.imshow(values[np.ix_(order, order)], cmap=cmap,
                          norm=TwoSlopeNorm(vmin=-.5, vcenter=0, vmax=1),
                          origin="upper", aspect="equal", interpolation="nearest", rasterized=True)
        ax.set(xticks=[], yticks=[], xlabel="Bacterial stimuli (profile order)")
        ax.xaxis.labelpad = 10
        left.text(-.12, .5, f"Bacterial stimuli (n = {len(ids)})", rotation=90,
                  ha="right", va="center", transform=left.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes([.45, .28, .013, .42])
        cax.set_title("Cosine\nsimilarity", fontsize=9, pad=12)
        finite = values[np.isfinite(values)]
        low, high = finite.min() < -.5, finite.max() > 1
        extend = "both" if low and high else "min" if low else "max" if high else "neither"
        colorbar = fig.colorbar(image, cax=cax, ticks=[-.5, 0, .5, 1], extend=extend)
        colorbar.outline.set_visible(False)

        histogram = fig.add_axes([.60, .28, .365, .42])
        histogram.set_title("Distribution of similarities", fontsize=12, pad=18)
        plotted_checks = []
        for group, color, label, expected in [
            (other, BLUE, "Different stimuli", other_summary),
            (same, RED, "Same stimulus", same_summary),
        ]:
            density, actual_edges, _ = histogram.hist(
                group, bins=bins, density=True, color=color, alpha=.32,
                edgecolor=color, linewidth=.5,
                label=f"{label} (mean = {group.mean():.2f})")
            histogram.hist(group, bins=bins, density=True, histtype="step", color=color, linewidth=1.1)
            histogram.axvline(group.mean(), color=color, linestyle=(0, (4, 3)), linewidth=1.3)
            exact = np.array_equal(actual_edges, bins) and np.array_equal(density, expected["density"])
            if not exact:
                raise ValueError("Plotted histogram changed the saved-value density")
            plotted_checks.append(exact)
        histogram.set(xlim=(-1, 1), ylim=(0, None), xlabel="Cosine similarity", ylabel="Density")
        histogram.xaxis.set_major_locator(MaxNLocator(5))
        histogram.yaxis.set_major_locator(MaxNLocator(4))
        histogram.spines[["top", "right"]].set_visible(False)
        handles, labels = histogram.get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(.595, .91),
                   frameon=False, fontsize=9)
        files = []
        for suffix in ("png", "svg"):
            path = figures / f"02_repeatability_distribution.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            files.append(str(path))
        plt.close(fig)

    if any(_hash(path) != source_hashes[str(path)] for path in source_paths):
        raise ValueError("A saved scientific input changed during the display refresh")
    reference = out.parent.parent / "notebook/02_reproducibility_inspection.ipynb"
    parameters = dict(
        display_only=True, refit=False, source_sha256=source_hashes,
        reference_notebook=str(reference), reference_sha256=_hash(reference),
        reference_source_cells=[27, 30], code_sha256=_hash(__file__),
        primary_threshold=representation["primary_threshold"], n_samples=len(ids),
        figure_size_inches=[12.8, 6.4], histogram_size_inches=[4.672, 2.688], dpi=300,
        histogram_axes=[.60, .28, .365, .42], histogram_xlim=[-1, 1], histogram_ymin=0,
        histogram_bin_width=.05, histogram_edges=bins.tolist(),
        histogram_density="Each finite group normalized separately to unit area",
        colors=dict(same=RED, different=BLUE), alpha=.32,
        row_order=ordered_ids, row_order_rule="Saved raw-profile RDM average-linkage tree; same leaves as the profile",
        matrix_color_limits=[-.5, 0, 1], matrix_missing_color="#DCDCDC",
        matrix_saturated_entries=int((values < -.5).sum() + (values > 1).sum()),
        same=same_summary, different=other_summary,
        undefined_different_pairs=int(len(ids) * (len(ids) - 1) // 2 - len(other)),
        checks=dict(input_hashes_unchanged=True, all_finite_values_in_histograms=True,
                    plotted_densities_match_saved_values=all(plotted_checks),
                    histogram_edges_match_previous_tall_display=True,
                    mean_markers_match_saved_values=True, row_order_matches_saved=True),
        smoothing=False, axis_breaks=False, previous_figures=archived, files=files,
    )
    parameters_path = figures / "repeatability_display_parameters.json"
    parameters_path.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    caption_path = figures / "repeatability_caption.txt"
    caption_path.write_text(
        f"Saved individual-SNR ≥ {representation['primary_threshold']:g} cross-half similarities, "
        "displayed in Notebook 02 Panel B's original short, wide layout. The same saved numerical "
        "matrix underlies the heatmap and histograms; no split, gate, template or similarity was "
        "recomputed. Each histogram contains one mean over valid splits per sample or unique "
        "sample pair, with separate unit-area normalization and unchanged 0.05-wide bins. "
        f"There are {len(same)} same-stimulus values and {len(other):,} different-stimulus values; "
        f"{parameters['undefined_different_pairs']} pairs have no defined similarity and remain excluded. "
        "All finite values, including negative tails, are shown on −1 to 1 without smoothing, "
        "an axis break, or selective truncation. The heatmap keeps the original −0.5 to 1 color "
        "range; values below −0.5 saturate in color but remain unchanged in the histograms. "
        "Gray denotes undefined similarities. Both dendrograms use the saved raw-profile hierarchy "
        "and match the separate profile order. Gates and templates were previously fitted "
        "independently in each whole-animal half within date; each half required at least two "
        "animals per condition × cell. Comparisons use the full eight-bin 0–40 s reconstruction "
        "and at least four shared cells. Repeated partitions and overlapping pairs are not "
        "independent biological replicates.\n",
        encoding="utf-8")
    return dict(files=files, parameters=str(parameters_path), caption=str(caption_path),
                same_mean=same_summary["mean"], different_mean=other_summary["mean"],
                checks=parameters["checks"])
