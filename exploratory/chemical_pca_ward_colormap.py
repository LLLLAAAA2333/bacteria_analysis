"""Chemical PCA → Euclidean distance → Ward clustering → leaf-order colormap.

Pipeline:
  log2(FC) → z-score → SVD PCA(10) → Euclidean distance → Ward linkage
  → leaf-order colormap (from plot_chemical_hmds.py) → assign AID colors

Outputs (results/chemical_pca_ward/):
  - pca_variance_explained.png
  - chemical_pca_ward_dendrogram.png
  - aid_colormap.csv            (AID → hex color)
  - pca_scores.csv              (106 strains × 10 PCs)
  - chemical_pca_euclidean_rdm.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    linkage,
    leaves_list,
    dendrogram,
    cophenet,
)
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import (
    read_metabolite_matrix,
    enrich_neural_dataframe,
    _canonicalize_metabolite_name,
)

OUTPUT_DIR = Path("results/chemical_pca_ward")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP_NAME = "turbo"
N_PCS = 10

# ── colormap (from plot_chemical_hmds.py) ────────────────────────────────


def leaf_order_colormap(Z, n, cmap_name="turbo"):
    """Assign colors to leaves based on cophenetic gaps in dendrogram order.

    Large gaps between consecutive leaves → sharp color transitions.
    Small gaps → smooth transitions.  Uses arcsinh to stretch gaps nonlinearly.
    """
    order = leaves_list(Z)
    coph = squareform(cophenet(Z))
    gaps = np.array([coph[order[i], order[i + 1]] for i in range(n - 1)])
    med = float(np.median(gaps)) or 1.0
    cumdist = np.concatenate([[0.0], np.cumsum(np.arcsinh(gaps / med))])
    normed = (cumdist - cumdist.min()) / (cumdist.max() - cumdist.min() + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(normed)  # RGBA in leaf order
    # Reorder back to original input order
    colors_orig = np.zeros((n, 4))
    for leaf_pos, orig_idx in enumerate(order):
        colors_orig[orig_idx] = colors[leaf_pos]
    return colors_orig, order, normed


# ── data loading & preprocessing ─────────────────────────────────────────


def raw_missing_rates(rm, sids):
    sc = [s for s in sids.astype(str) if s in rm.columns]
    return rm.loc[:, sc].isna().mean(axis=1)


def load_and_prepare():
    """QC filter → log2 → z-score → PCA. Returns (pca_scores, explained_variances, metabolite_names)."""
    # Get 106bac strain AIDs
    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = (
        enriched.groupby("stimulus")[["species", "genus", "aid"]]
        .first()
        .drop_duplicates(subset="aid")
        .set_index("aid")
    )
    neural_aids = set(stim_info.index)

    # QC filter
    fc = read_metabolite_matrix("data/data_fc_missingto1.xlsx")
    raw_meta = pd.read_excel(
        "data/metabolism_raw_data.xlsx", sheet_name="all", engine="openpyxl"
    )
    annotations = raw_meta.copy()
    annotations["_feature_name"] = annotations["name"].map(_canonicalize_metabolite_name)
    annotations["_qcrsd"] = pd.to_numeric(annotations["QCRSD"], errors="coerce")
    annotations["_raw_missing_rate"] = raw_missing_rates(annotations, fc.index)
    available = set(fc.columns.astype(str))
    retained = annotations.loc[
        annotations["_feature_name"].isin(available)
        & annotations["_qcrsd"].le(0.2)
        & annotations["_raw_missing_rate"].le(0.5)
    ].drop_duplicates("_feature_name", keep="first")
    retained_features = retained["_feature_name"].astype(str).tolist()

    # Match to neural strains
    neural_samples = sorted([s for s in fc.index if s in neural_aids])
    print(f"{len(neural_samples)} strains, {len(retained_features)} metabolites after QC")

    chemical = fc.loc[neural_samples, retained_features].apply(pd.to_numeric, errors="coerce")
    chemical = chemical.where(chemical > 0)
    chemical_log2 = np.log2(chemical)

    # z-score
    means = chemical_log2.mean(axis=0)
    stds = chemical_log2.std(axis=0, ddof=0)
    valid = stds[np.isfinite(stds) & (stds > 0)].index
    chemical_z = (chemical_log2.loc[:, valid] - means[valid]) / stds[valid]
    metabolite_names = list(chemical_z.columns)
    print(f"{len(valid)} metabolites after z-score (non-constant)")

    # Standard PCA (SVD)
    X = chemical_z.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    eigenvalues = (S ** 2) / max(X.shape[0] - 1, 1)
    explained = eigenvalues / eigenvalues.sum()

    pcs = X @ Vt[:N_PCS].T
    pcs_df = pd.DataFrame(
        pcs,
        index=neural_samples,
        columns=[f"PC{i+1}" for i in range(N_PCS)],
    )

    # Merge with genus/species info
    stim_info_reindexed = stim_info.reindex(neural_samples)
    pcs_df["genus"] = stim_info_reindexed["genus"].values
    pcs_df["species"] = stim_info_reindexed["species"].values

    return pcs_df, explained, metabolite_names, Vt, stim_info


# ── plots ────────────────────────────────────────────────────────────────


def plot_variance_explained(
    explained: np.ndarray, output_path: Path
) -> None:
    """Bar + cumulative line plot for PCA variance explained."""
    x = np.arange(1, N_PCS + 1)
    cumulative = np.cumsum(explained[:N_PCS]) * 100

    fig, ax1 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    bars = ax1.bar(
        x, explained[:N_PCS] * 100,
        color="#64748B", edgecolor="white", linewidth=0.8,
        label="Individual",
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance explained (%)", color="#64748B")
    ax1.tick_params(axis="y", labelcolor="#64748B")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(
        x, cumulative,
        color="#C2410C", marker="o", linewidth=2, markersize=7,
        label="Cumulative",
    )
    ax2.set_ylabel("Cumulative variance (%)", color="#C2410C")
    ax2.tick_params(axis="y", labelcolor="#C2410C")

    # Annotate bars
    for bar, pct in zip(bars, explained[:N_PCS] * 100):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.5,
        )
    # Annotate cumulative
    for i, cum in enumerate(cumulative):
        ax2.annotate(
            f"{cum:.1f}%", (x[i], cum),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color="#C2410C",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title(
        f"Chemical PCA variance explained  —  {N_PCS} PCs, 106 strains",
        fontsize=12,
    )
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_dendrogram_with_colormap(
    Z, labels, colors_hex, output_path: Path,
) -> None:
    """Dendrogram with leaf labels colored by the leaf-order colormap."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.22), 5.5), constrained_layout=True)

    dendrogram(
        Z, ax=ax, labels=labels, leaf_font_size=7,
        color_threshold=0, above_threshold_color="#2c3e50",
        link_color_func=lambda k: "#2c3e50",
    )

    # Color leaf labels by our colormap
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        try:
            idx = list(labels).index(label_text)
            tick_label.set_color(colors_hex[idx])
        except (ValueError, IndexError):
            pass

    ax.set_title(
        f"Chemical PCA Ward dendrogram  —  Euclidean distance, {N_PCS} PCs, 106 strains",
        fontsize=11,
    )
    ax.set_ylabel("Ward merge cost")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────


def main():
    pcs_df, explained, met_names, Vt, stim_info = load_and_prepare()
    aids = pcs_df.index.tolist()
    n = len(aids)
    print(f"Total strains: {n}")

    # ── 1. Variance explained plot ─────────────────────────────────────

    plot_variance_explained(
        explained,
        OUTPUT_DIR / "pca_variance_explained.png",
    )

    # Print summary
    cumulative = np.cumsum(explained[:N_PCS]) * 100
    print("\nPCA variance explained:")
    for i in range(N_PCS):
        print(f"  PC{i+1:2d}: {explained[i]*100:5.1f}%  (cumulative {cumulative[i]:.1f}%)")

    # ── 2. Euclidean distance on PC scores ──────────────────────────────

    pc_values = pcs_df.iloc[:, :N_PCS].to_numpy(dtype=float)
    euclidean_condensed = pdist(pc_values, metric="euclidean")
    euclidean_matrix = squareform(euclidean_condensed)
    rdm_df = pd.DataFrame(euclidean_matrix, index=aids, columns=aids)
    rdm_df.to_csv(OUTPUT_DIR / "chemical_pca_euclidean_rdm.csv")
    print(f"\nSaved: chemical_pca_euclidean_rdm.csv ({n}×{n})")

    # ── 3. Ward hierarchical clustering ─────────────────────────────────

    Z = linkage(euclidean_condensed, method="ward")
    colors_rgba, order, normed = leaf_order_colormap(Z, n, cmap_name=CMAP_NAME)

    # Convert RGBA → hex
    colors_hex = [
        "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for r, g, b, _ in colors_rgba
    ]

    # ── 4. Dendrogram ──────────────────────────────────────────────────

    # Labels = AID + genus
    leaf_labels = [
        f"{aid}  [{stim_info.loc[aid, 'genus']}]" if aid in stim_info.index else aid
        for aid in aids
    ]
    plot_dendrogram_with_colormap(
        Z, leaf_labels, colors_hex,
        OUTPUT_DIR / "chemical_pca_ward_dendrogram.png",
    )

    # ── 5. Save colormap CSV ───────────────────────────────────────────

    colormap_df = pd.DataFrame(
        {
            "aid": aids,
            "color_hex": colors_hex,
            "leaf_order_position": [list(order).index(i) for i in range(n)],
            "color_normed": [normed[list(order).index(i)] for i in range(n)],
            "genus": [stim_info.loc[aid, "genus"] if aid in stim_info.index else "" for aid in aids],
            "species": [stim_info.loc[aid, "species"] if aid in stim_info.index else "" for aid in aids],
        }
    )
    colormap_df.to_csv(OUTPUT_DIR / "aid_colormap.csv", index=False)
    print(f"Saved: aid_colormap.csv")

    # ── 6. Save PC scores ──────────────────────────────────────────────

    pcs_df.to_csv(OUTPUT_DIR / "pca_scores.csv")
    print(f"Saved: pca_scores.csv")

    # ── 7. Color swatch preview ────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(max(10, n * 0.25), 2.5), constrained_layout=True)
    # Swatches in leaf order
    ordered_colors = [colors_hex[leaf_pos] for leaf_pos in order]
    ordered_labels = [leaf_labels[leaf_pos] for leaf_pos in order]
    for i, (color, label) in enumerate(zip(ordered_colors, ordered_labels)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5))
        ax.text(i + 0.5, -0.35, label, ha="center", va="top", fontsize=5.5,
                rotation=90)
    ax.set_xlim(0, n)
    ax.set_ylim(-2.5, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Chemical PCA Ward leaf-order colormap  —  {CMAP_NAME}, 106 strains",
        fontsize=11,
    )
    fig.savefig(
        OUTPUT_DIR / "colormap_swatches.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Saved: colormap_swatches.png")

    print(f"\nAll outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
