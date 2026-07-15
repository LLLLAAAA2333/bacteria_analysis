"""Metabolite module reduction via correlation-based hierarchical clustering.

Groups the 380 metabolites into modules of highly-correlated compounds,
then represents each module by its PC1 (module eigengene).  Reduces the
380-column metabolite matrix to a compact module matrix for downstream
analyses.

Similar in spirit to PCA, but modules stay interpretable — each module
is a set of co-varying metabolites you can inspect directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

CUT_THRESHOLDS = [2, 4, 6, 8, 10, 12, 14]  # Euclidean distance on log2(1+FC)
OUTPUT_DIR = Path("results/metabolite_modules")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# populated at runtime
_BEST_THRESHOLD: float = 0.30


def _module_palette(n: int) -> list[str]:
    import matplotlib.colors as mcolors
    base = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    filtered = [c for c in base
                if np.mean(mcolors.to_rgb(c)) < 0.85
                and np.mean(mcolors.to_rgb(c)) > 0.15]
    return (filtered * (1 + n // len(filtered)))[:n]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    global _BEST_THRESHOLD

    # ---- 1. Load & transform ----
    met_raw = pd.read_excel("data/matrix.xlsx")
    met_raw = met_raw.rename(columns={"Unnamed: 0": "aid"})
    met_raw["aid"] = met_raw["aid"].astype(str)

    met_cols = [c for c in met_raw.columns if c != "aid"]
    met_vals = met_raw[met_cols].values.astype(float)
    met_log = np.log2(1 + met_vals)
    met_df = pd.DataFrame(met_log, columns=met_cols, index=met_raw["aid"])
    n_samples, n_mets = met_df.shape
    print(f"Metabolite matrix: {n_samples} samples × {n_mets} metabolites")

    # ---- 2. Euclidean distance on log2(1+FC) → Ward clustering ----
    dist_condensed = pdist(met_df.values.T, metric="euclidean")
    dist_mat = squareform(dist_condensed)

    Z = linkage(dist_condensed, method="ward")
    leaf_order = leaves_list(Z)

    # ---- 3. Threshold sweep ----
    print("\nThreshold sweep (Ward, Euclidean distance on log2(1+FC)):")
    print(f"  {'thr':>6s}  {'n_mod':>6s}  {'mean_sz':>8s}  {'singleton':>10s}  {'max_sz':>7s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*7}")

    best = None
    for thresh in CUT_THRESHOLDS:
        lbl = fcluster(Z, thresh, criterion="distance")
        n_mod = len(set(lbl))
        sizes = pd.Series(lbl).value_counts()
        singletons = (sizes == 1).sum()
        print(f"  {thresh:6.1f}  {n_mod:6d}  {sizes.mean():8.1f}  {singletons:10d}  {sizes.max():7d}")
        if 15 <= n_mod <= 120 and singletons / n_mod < 0.3:
            if best is None or abs(n_mod - 40) < abs(best[1] - 40):
                best = (thresh, n_mod)

    _BEST_THRESHOLD = best[0] if best else 8.0
    print(f"\n-> Selected threshold = {_BEST_THRESHOLD:.1f} "
          f"(Euclidean distance on log2(1+FC))")

    # ---- 3b. PCA: how many PCs to capture 80%/90%/95% variance? ----
    pca_full = PCA()
    pca_full.fit(met_df.values)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    for target in [0.80, 0.90, 0.95]:
        n_pc = np.searchsorted(cumvar, target) + 1
        print(f"  PCA: {n_pc} PCs capture {target:.0%} variance (of 380 metabolites)")

    # ---- 4. Assign modules ----
    labels = fcluster(Z, _BEST_THRESHOLD, criterion="distance")
    n_modules = len(set(labels))
    module_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)

    # ---- 5. Module PC1 reduction ----
    module_pcs: dict[int, np.ndarray] = {}
    module_members: dict[int, list[str]] = {}
    module_explained: dict[int, float] = {}

    for mod_id in sorted(set(labels)):
        mask = labels == mod_id
        member_names = [met_cols[i] for i in range(n_mets) if mask[i]]
        member_data = met_df[member_names].values

        module_members[mod_id] = member_names

        if len(member_names) == 1:
            pc1 = member_data[:, 0].copy()
            var_explained = 1.0
        else:
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(member_data)[:, 0]
            var_explained = pca.explained_variance_ratio_[0]

        # Orient PC1 positively with module mean
        if np.corrcoef(pc1, member_data.mean(axis=1))[0, 1] < 0:
            pc1 = -pc1

        module_pcs[mod_id] = pc1
        module_explained[mod_id] = var_explained

    # ---- 6. Build & save reduced matrix ----
    mod_order = sorted(module_pcs.keys(), key=lambda m: -module_sizes[m])
    reduced = pd.DataFrame(
        {f"module_{m:03d}": module_pcs[m] for m in mod_order},
        index=met_df.index,
    )
    reduced.to_parquet(OUTPUT_DIR / "metabolite_modules.parquet")
    print(f"\nReduced matrix (modules): {reduced.shape}  ->  {OUTPUT_DIR / 'metabolite_modules.parquet'}")

    # Also save PCA scores (top 80 PCs capture 90%+ variance)
    n_pca_keep = np.searchsorted(cumvar, 0.90) + 1
    pca_scores = pca_full.transform(met_df.values)[:, :n_pca_keep]
    pca_df = pd.DataFrame(
        pca_scores,
        index=met_df.index,
        columns=[f"PC{i+1}" for i in range(n_pca_keep)],
    )
    pca_df.to_parquet(OUTPUT_DIR / "metabolite_pca.parquet")
    print(f"Reduced matrix (PCA):   (299, {n_pca_keep})  ->  {OUTPUT_DIR / 'metabolite_pca.parquet'}")

    # Module membership CSV
    rows = []
    for mod_id in sorted(module_members.keys()):
        for met in module_members[mod_id]:
            rows.append({
                "module": f"module_{mod_id:03d}",
                "size": len(module_members[mod_id]),
                "pc1_var": f"{module_explained[mod_id]:.3f}",
                "metabolite": met,
            })
    memb = pd.DataFrame(rows)
    memb.to_csv(OUTPUT_DIR / "module_membership.csv", index=False)
    print(f"Membership CSV  →  {OUTPUT_DIR / 'module_membership.csv'}")

    # ---- 7. Summary ----
    print(f"\n=== {n_modules} modules ===")
    for mod_id in mod_order[:35]:
        members = module_members[mod_id]
        short = ", ".join(m[:35] for m in members[:5])
        if len(members) > 5:
            short += f"  …(+{len(members)-5})"
        print(f"  module_{mod_id:03d}  n={len(members):3d}  "
              f"var={module_explained[mod_id]:.2f}  {short}")
    if n_modules > 35:
        print(f"  … and {n_modules - 35} more modules")

    # ---- 8. Plots ----
    plot_dendrogram(Z, labels, leaf_order, met_cols, module_members)
    corr_for_viz = np.corrcoef(met_df.values.T)
    plot_correlation_heatmap(corr_for_viz, leaf_order, labels, met_cols)

    print(f"\nDone → {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_dendrogram(Z, labels, leaf_order, met_names, module_members):
    global _BEST_THRESHOLD

    # ---- truncated overview ----
    fig, ax = plt.subplots(figsize=(22, 7))
    dendrogram(
        Z, ax=ax,
        truncate_mode="lastp", p=min(60, len(module_members) * 2),
        leaf_font_size=7,
        color_threshold=0,
        above_threshold_color="#888888",
        link_color_func=lambda _: "#888888",
    )
    ax.set_title(
        f"Metabolite clustering (Ward, Euclidean distance on log2(1+FC))\n"
        f"{len(met_names)} metabolites -> {len(module_members)} modules "
        f"(distance < {_BEST_THRESHOLD:.1f})",
        fontsize=10,
    )
    ax.set_ylabel("Ward merge distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)

    # ---- full dendrogram with module colour strip ----
    n_mods = len(module_members)
    palette = _module_palette(n_mods)
    unique_mods = sorted(module_members.keys())
    mod_to_color = {m: palette[i % len(palette)] for i, m in enumerate(unique_mods)}
    ordered_labels = labels[leaf_order]

    fig, ax = plt.subplots(figsize=(32, 10))
    dd = dendrogram(
        Z, ax=ax,
        labels=[met_names[i][:40] for i in leaf_order],
        leaf_font_size=3.2,
        color_threshold=0,
        above_threshold_color="#cccccc",
        link_color_func=lambda _: "#cccccc",
    )
    ymax = ax.get_ylim()[1]
    for i in range(len(leaf_order)):
        mod = ordered_labels[i]
        rect = Rectangle((i - 0.4, -0.06 * ymax), 0.8, 0.03 * ymax,
                         facecolor=mod_to_color[mod], edgecolor="none",
                         clip_on=False, alpha=0.85)
        ax.add_patch(rect)

    ax.set_title(
        f"Full dendrogram with module colours  "
        f"({len(met_names)} metabolites, {n_mods} modules)",
        fontsize=9,
    )
    ax.set_ylabel("Ward merge distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dendrogram_full.png", dpi=100)
    plt.close(fig)


def plot_correlation_heatmap(corr_mat, leaf_order, labels, met_names):
    """Correlation matrix, subsampled if needed."""
    n = len(leaf_order)
    if n > 200:
        step = n // 150
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    corr_sub = corr_mat[leaf_order][:, leaf_order][idx][:, idx]

    fig, ax = plt.subplots(figsize=(max(14, len(idx) * 0.16), 11))
    im = ax.imshow(corr_sub, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Metabolite-metabolite Pearson correlation (reordered by clustering)\n"
        f"{len(met_names)} metabolites, {len(set(labels))} modules",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
