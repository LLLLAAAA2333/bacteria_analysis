"""Visualize metabolite correlation clustering and module results.

Generates:
  1. Clean dendrogram with module colour bands
  2. Module-level correlation heatmap (192 x 192, not 380 x 380)
  3. Module size distribution
  4. Per-module abundance traces (selected modules)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DIST_THRESHOLD = 8.0   # Euclidean distance on log2(1+FC) — tune after sweep
OUTPUT_DIR = Path("results/metabolite_modules")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_palette(n: int) -> list[str]:
    """Generate n distinguishable, non-grey colours."""
    base = (list(mcolors.TABLEAU_COLORS.values())
            + list(mcolors.CSS4_COLORS.values()))
    filtered = [c for c in base
                if 0.12 < np.mean(mcolors.to_rgb(c)) < 0.88]
    return (filtered * (1 + n // len(filtered)))[:n]


def _short(name: str, max_len: int = 30) -> str:
    return name[:max_len - 1] + "…" if len(name) > max_len else name


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # ---- 1. Load data ----
    met_raw = pd.read_excel("data/matrix.xlsx")
    met_raw = met_raw.rename(columns={"Unnamed: 0": "aid"})
    met_raw["aid"] = met_raw["aid"].astype(str)
    met_cols = [c for c in met_raw.columns if c != "aid"]
    met_vals = met_raw[met_cols].values.astype(float)
    met_log = np.log2(1 + met_vals)
    met_df = pd.DataFrame(met_log, columns=met_cols, index=met_raw["aid"])

    n_samples, n_mets = met_df.shape

    # ---- 2. Clustering (Euclidean distance on log2(1+FC)) ----
    dist_condensed = pdist(met_df.values.T, metric="euclidean")
    Z = linkage(dist_condensed, method="ward")
    leaf_order = leaves_list(Z)
    labels = fcluster(Z, DIST_THRESHOLD, criterion="distance")

    module_ids = sorted(set(labels))
    n_modules = len(module_ids)
    module_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)

    # ---- 3. Module eigengenes (PC1 per module) ----
    from sklearn.decomposition import PCA

    module_eigengenes = {}  # mod_id -> (n_samples,) array
    module_members = {}     # mod_id -> list of metabolite names

    for mod_id in module_ids:
        mask = labels == mod_id
        members = [met_cols[i] for i in range(n_mets) if mask[i]]
        module_members[mod_id] = members
        data = met_df[members].values
        if len(members) == 1:
            eig = data[:, 0].copy()
        else:
            eig = PCA(n_components=1).fit_transform(data)[:, 0]
            if np.corrcoef(eig, data.mean(axis=1))[0, 1] < 0:
                eig = -eig
        module_eigengenes[mod_id] = eig

    print(f"{n_mets} metabolites -> {n_modules} modules (Euclidean distance < {DIST_THRESHOLD:.1f})")

    # ---- Plot ----
    plot_dendrogram_with_modules(Z, labels, leaf_order, met_cols, n_modules)
    plot_module_correlation_heatmap(module_eigengenes, module_ids, module_sizes)
    plot_module_size_distribution(module_sizes)
    plot_selected_module_traces(met_df, module_members, module_eigengenes, module_sizes)

    print(f"Done -> {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Figure 1: Dendrogram with module colour strip
# ---------------------------------------------------------------------------

def plot_dendrogram_with_modules(Z, labels, leaf_order, met_names, n_modules):
    palette = _build_palette(n_modules)
    unique_mods = sorted(set(labels))
    mod_to_color = {m: palette[i % len(palette)] for i, m in enumerate(unique_mods)}
    ordered_labels = labels[leaf_order]

    fig = plt.figure(figsize=(34, 12))

    # -- main dendrogram --
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.02)
    ax_dendro = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_dendro)

    dd = dendrogram(
        Z, ax=ax_dendro,
        labels=[_short(met_names[i], 35) for i in leaf_order],
        leaf_font_size=2.8,
        color_threshold=0,
        above_threshold_color="#cccccc",
        link_color_func=lambda _: "#cccccc",
    )
    ax_dendro.set_ylabel("Ward merge distance (Euclidean)", fontsize=10)
    ax_dendro.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_dendro.set_title(
        f"Metabolite clustering  ({len(met_names)} metabolites, {n_modules} modules)\n"
        f"Ward linkage on Euclidean distance (log2(1+FC)), cut at {DIST_THRESHOLD}",
        fontsize=11,
    )

    # -- module colour strip --
    for i in range(len(leaf_order)):
        mod = ordered_labels[i]
        rect = Rectangle((i - 0.4, 0), 0.8, 1,
                         facecolor=mod_to_color[mod], edgecolor="none",
                         alpha=0.9)
        ax_strip.add_patch(rect)

    ax_strip.set_xlim(-0.5, len(leaf_order) - 0.5)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_yticks([])
    ax_strip.axis("off")

    fig.savefig(OUTPUT_DIR / "dendrogram_modules.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> dendrogram_modules.png")


# ---------------------------------------------------------------------------
# Figure 2: Module-level correlation heatmap
# ---------------------------------------------------------------------------

def plot_module_correlation_heatmap(module_eigengenes, module_ids, module_sizes):
    """Correlation between module eigengenes, clustered hierarchically."""
    mod_order = list(module_sizes.index)  # largest first
    eig_matrix = np.column_stack([module_eigengenes[m] for m in mod_order])

    # Module-module correlation
    mod_corr = np.corrcoef(eig_matrix.T)
    # Re-cluster modules
    mod_dist = 1 - mod_corr
    np.fill_diagonal(mod_dist, 0)
    mod_dist = (mod_dist + mod_dist.T) / 2
    Z_mod = linkage(squareform(mod_dist), method="ward")
    mod_leaf = leaves_list(Z_mod)
    mod_corr_ordered = mod_corr[mod_leaf][:, mod_leaf]

    n = len(mod_order)
    fig, ax = plt.subplots(figsize=(max(18, n * 0.14), max(14, n * 0.12)))

    im = ax.imshow(mod_corr_ordered, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="none")

    # Label selected large modules on axes
    tick_pos = []
    tick_labels = []
    for i, idx in enumerate(mod_leaf):
        mod = mod_order[idx]
        sz = module_sizes[mod]
        if sz >= 3:  # only label modules with 3+ members
            tick_pos.append(i)
            # Pick representative metabolite as label
            tick_labels.append(f"mod_{mod:03d} (n={sz})")

    # Subsample if too many
    if len(tick_pos) > 40:
        step = len(tick_pos) // 35
        tick_pos = tick_pos[::step]
        tick_labels = tick_labels[::step]

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=90, ha="center")
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=5)

    ax.set_title(f"Module-module Pearson correlation ({n} modules)\n"
                 "Ordered by hierarchical clustering of module eigengenes",
                 fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.78, label="Pearson r")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "module_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  -> module_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3: Module size distribution
# ---------------------------------------------------------------------------

def plot_module_size_distribution(module_sizes):
    sizes = module_sizes.values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- histogram --
    ax = axes[0]
    bins = np.arange(0.5, sizes.max() + 1.5, 1)
    ax.hist(sizes, bins=bins, color="#4a5568", edgecolor="white", alpha=0.85)
    ax.axvline(sizes[sizes >= 3].mean(), color="#e74c3c", ls="--", lw=1.5,
               label=f"mean (multi): {sizes[sizes >= 3].mean():.1f}")
    ax.axvline(2, color="#3498db", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Module size (number of metabolites)")
    ax.set_ylabel("Count")
    ax.set_title(f"Module size distribution ({len(sizes)} modules)")
    ax.legend(fontsize=8)

    # -- pie: singletons vs multi vs large --
    ax = axes[1]
    singletons = (sizes == 1).sum()
    pairs = (sizes == 2).sum()
    multi = (sizes >= 3).sum()
    ax.pie([singletons, pairs, multi],
           labels=[f"n=1: {singletons}", f"n=2: {pairs}", f"n>=3: {multi}"],
           colors=["#bdc3c7", "#f39c12", "#2ecc71"],
           autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 10})
    ax.set_title("Module size breakdown", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "module_size_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  -> module_size_distribution.png")


# ---------------------------------------------------------------------------
# Figure 4: Selected module abundance traces
# ---------------------------------------------------------------------------

def plot_selected_module_traces(met_df, module_members, module_eigengenes,
                                 module_sizes):
    """Show raw abundance profiles for top-6 largest modules."""
    top_mods = list(module_sizes.head(6).index)
    n_mods = len(top_mods)

    fig, axes = plt.subplots(n_mods, 1, figsize=(16, 2.5 * n_mods), sharex=True)
    if n_mods == 1:
        axes = [axes]

    aid_order = met_df.index.tolist()  # A001..A299

    for ax, mod_id in zip(axes, top_mods):
        members = module_members[mod_id]
        n_mem = len(members)
        eig = module_eigengenes[mod_id]

        # Plot individual metabolite traces (thin, grey)
        for met in members:
            ax.plot(range(len(aid_order)), met_df[met].values,
                    alpha=0.25, lw=0.6, color="#95a5a6")

        # Plot eigengene on top (thick, color)
        ax.plot(range(len(aid_order)), eig, color="#e74c3c", lw=2.2,
                label=f"PC1 ({n_mem} metabolites)")

        # Short member list as title
        short_members = ", ".join(_short(m, 25) for m in members[:4])
        if n_mem > 4:
            short_members += f", ... (+{n_mem - 4})"
        ax.set_ylabel("log2(1+FC)")
        ax.set_title(f"module_{mod_id:03d}: {short_members}", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)

    ax.set_xlabel("Sample index (A001–A299)")
    fig.suptitle("Top modules — abundance profiles (thin: individual metabolites, thick: module PC1)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "module_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> module_traces.png")


if __name__ == "__main__":
    main()
