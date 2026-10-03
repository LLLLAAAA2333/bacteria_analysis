"""Read and visualize saved whole-animal split results without recomputing splits."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

COLORS = {"same": "#D95D67", "different": "#5386CB"}


def load_repeatability_data(report):
    """Return saved similarity/support matrices and matched validation tables."""
    tables = Path(report) / "tables"
    split = pd.read_csv(tables / "split_filtered_cosine.csv", index_col=0)
    split = split.loc[split.index, split.index]
    values = split.to_numpy(float)
    if not split.index.is_unique or not np.allclose(values, values.T, equal_nan=True, atol=1e-12):
        raise ValueError("Split matrix must have unique IDs and symmetric similarities")
    support = pd.read_csv(tables / "split_filtered_valid_splits.csv", index_col=0).loc[split.index, split.index]
    if not np.array_equal(np.isfinite(values), support.to_numpy() > 0):
        raise ValueError("Saved split support does not identify finite similarities")
    return dict(split=split, support=support,
                sample_order=pd.read_csv(tables / "figure_row_order.csv")["strain"].tolist(),
                control_pairs=pd.read_csv(tables / "split_control_pairs.csv"),
                loao_errors=pd.read_csv(tables / "loao_errors.csv"),
                loao_cells=pd.read_csv(tables / "loao_cell_summary.csv").set_index("cell"))


def extract_distributions(similarity):
    """One saved split-average per stimulus or unique pair; omit undefined values."""
    values = np.asarray(similarity, float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Similarity must be square")
    if not np.allclose(values, values.T, equal_nan=True, atol=1e-12):
        raise ValueError("Unique-pair extraction requires symmetric similarities")
    same = np.diag(values)
    different = values[np.triu_indices(len(values), 1)]
    return dict(same=same[np.isfinite(same)], different=different[np.isfinite(different)],
                undefined_same=int((~np.isfinite(same)).sum()),
                undefined_different=int((~np.isfinite(different)).sum()))


def histogram_summary(values, bins):
    """Finite count, mean, density; reject bins that silently discard finite tails."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite values")
    counts, _ = np.histogram(values, bins=bins)
    if counts.sum() != len(values):
        raise ValueError("Histogram bins exclude finite observations")
    density = counts / (len(values) * np.diff(bins))
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                counts=counts, density=density)


def plot_repeatability(same, different, bins=None):
    """Poster panel: two separately normalized empirical distributions, full tails."""
    bins = np.arange(-1., 1.0001, .05) if bins is None else np.asarray(bins)
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    for values, key, label in [(different, "different", "Different stimuli"), (same, "same", "Same stimulus")]:
        summary = histogram_summary(values, bins)
        ax.stairs(summary["density"], bins, fill=True, alpha=.28, color=COLORS[key])
        ax.stairs(summary["density"], bins, lw=1.4, color=COLORS[key],
                  label=f"{label} (n = {summary['n']:,}; mean = {summary['mean']:.2f})")
        ax.axvline(summary["mean"], color=COLORS[key], ls=(0, (4, 3)), lw=1.4)
    ax.set(xlim=(-1, 1), ylim=(0, None), xlabel="Cross-half cosine similarity", ylabel="Density")
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, loc="upper left")
    return fig


def plot_similarity_matrix(similarity, sample_order):
    """Supporting view; preserve undefined pairs in gray and saved atlas row order."""
    values = similarity.loc[sample_order, sample_order].to_numpy(float)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DCDCDC")
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-.5, vcenter=0, vmax=1),
                      interpolation="nearest", rasterized=True)
    ax.set(xticks=[], yticks=[], xlabel="Bacterial stimuli (fixed profile order)",
           ylabel="Bacterial stimuli (fixed profile order)")
    extend = "min" if np.nanmin(values) < -.5 else "neither"
    fig.colorbar(image, ax=ax, shrink=.8, extend=extend).set_label("Cross-half cosine similarity")
    return fig


def aggregate_heldout_errors(errors):
    """Equal held-out animals → available blocks → strains within each neuron."""
    columns = ["mse_filtered", "mse_unfiltered", "mse_zero"]
    scored = errors.loc[errors["scored"].eq(True)]
    conditions = scored.groupby(["sample_id", "block", "cell"])[columns].mean()
    strains = conditions.groupby(["sample_id", "cell"])[columns].mean()
    return strains.groupby("cell")[columns].mean()
