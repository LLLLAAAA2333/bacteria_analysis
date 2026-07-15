"""16S phylogeny → cophenetic distance → Ward colormap → chemical & neural HMDS.

Pipeline:
  1. Parse 16S Newick tree → pairwise cophenetic distance matrix
  2. Ward hierarchical clustering → leaf-order colormap (same method as
     chemical_pca_ward_colormap.py)
  3. Apply phylogeny colormap to chemical PCA HMDS (3D)
  4. Apply phylogeny colormap to neural correlation HMDS (3D)

Outputs (results/phylogeny_hmds/):
  - phylogeny_colormap.csv
  - phylogeny_dendrogram.png
  - chemical_hmds_3d_phylogeny.png + .html
  - neural_hmds_3d_phylogeny.png + .html
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
    linkage, leaves_list, dendrogram, cophenet,
)
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "exploratory"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe
from compare_86bac_chord_hmds import (
    normalize_to_max_two,
    scipy_hyperbolic_mds,
    lorentz_to_poincare,
    recenter_poincare,
    preservation_metrics,
    upper_triangle,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
TREE_PATH = Path("data/16S.aln.trim.fa.treefile")
CHEMICAL_RDM_DIR = Path("results/chemical_pca_ward")
OUTPUT_DIR = Path("results/phylogeny_hmds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP_NAME = "turbo"
HMDS_DIM = 2
HMDS_STARTS = 8
HMDS_MAXITER = 900
HMDS_SEED = 42

# ── all 13 neurons (for neural HMDS) ──
ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)
_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}


# ======================================================================
# 1. Phylogenetic tree → distance matrix (via Bio.Phylo)
# ======================================================================

def load_tree_distance() -> tuple[np.ndarray, list[str]]:
    """Load 16S Newick tree → pairwise cophenetic distance matrix.

    Uses Bio.Phylo to parse the tree and compute path distances
    between all terminal (leaf) nodes.
    """
    from Bio import Phylo

    tree = Phylo.read(str(TREE_PATH), "newick")

    # Collect terminal clades (leaves)
    terminals = tree.get_terminals()
    leaf_names = [t.name for t in terminals]
    n = len(leaf_names)

    # Compute pairwise distances via tree.distance()
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(terminals[i], terminals[j]) or 0.0
            dist[i, j] = dist[j, i] = d

    print(f"Tree: {n} leaves, "
          f"distance range [{dist.min():.4f}, {dist.max():.4f}]")
    return dist, leaf_names


# ======================================================================
# 2. Colormap (exactly from chemical_pca_ward_colormap.py)
# ======================================================================

def leaf_order_colormap(Z, n, cmap_name="turbo"):
    """Assign colors to leaves based on cophenetic gaps in dendrogram order."""
    order = leaves_list(Z)
    coph = squareform(cophenet(Z))
    gaps = np.array([coph[order[i], order[i + 1]] for i in range(n - 1)])
    med = float(np.median(gaps)) or 1.0
    cumdist = np.concatenate([[0.0], np.cumsum(np.arcsinh(gaps / med))])
    normed = (cumdist - cumdist.min()) / (cumdist.max() - cumdist.min() + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(normed)
    colors_orig = np.zeros((n, 4))
    for leaf_pos, orig_idx in enumerate(order):
        colors_orig[orig_idx] = colors[leaf_pos]
    return colors_orig, order, normed


# ======================================================================
# 3. Neural prototype builder (for neural HMDS)
# ======================================================================

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _build_prototype_matrix(raw: pd.DataFrame, neuron: str) -> pd.DataFrame:
    raw_names = _neuron_raw_names(neuron)
    subset = raw[raw["neuron"].isin(raw_names)].copy()
    subset["trial_id"] = (
        pd.to_datetime(subset["date"]).dt.strftime("%Y%m%d")
        + "__" + subset["worm_key"].astype(str)
        + "__" + subset["segment_index"].astype(str)
    )
    trial_avg = (
        subset.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"]
        .mean().reset_index()
    )
    proto = (
        trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"]
        .median().reset_index()
    )
    return proto.pivot(index="stimulus", columns="time_point",
                       values="delta_F_over_F0")


# ======================================================================
# 4. HMDS builders
# ======================================================================

def load_chemical_rdm() -> tuple[np.ndarray, list[str]]:
    """Load precomputed PCA Euclidean RDM."""
    rdm = pd.read_csv(CHEMICAL_RDM_DIR / "chemical_pca_euclidean_rdm.csv", index_col=0)
    colormap = pd.read_csv(CHEMICAL_RDM_DIR / "aid_colormap.csv", index_col="aid")
    aids = sorted(set(rdm.index) & set(colormap.index))
    dist = rdm.loc[aids, aids].to_numpy(dtype=float)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    return dist, aids


def build_neural_distance(mats: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    """Build Pearson correlation distance from concatenated neuron prototypes."""
    common = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
    blocks = []
    for neuron in ALL_NEURONS:
        mat = mats[neuron].loc[common]
        mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
        mat_z.columns = [f"{neuron}_t{t}" for t in mat_z.columns]
        blocks.append(mat_z.values)
    feature_matrix = np.column_stack(blocks)
    r = np.corrcoef(feature_matrix)
    dist = np.clip(1 - r, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    return dist, common


def embed_hdms(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run 3D hyperbolic MDS."""
    dist_norm = normalize_to_max_two(dist)
    n_params = dist.shape[0] * HMDS_DIM
    lorentz, emb, lam, _ = scipy_hyperbolic_mds(
        dist_norm, dim=HMDS_DIM, starts=HMDS_STARTS,
        maxiter=HMDS_MAXITER, seed=HMDS_SEED,
    )
    poincare = recenter_poincare(lorentz_to_poincare(lorentz))
    predicted = emb / lam
    metrics = preservation_metrics(dist_norm, emb, predicted, n_params=n_params)
    return poincare, dist_norm, predicted, metrics


# ======================================================================
# 5. Plotting
# ======================================================================

def plot_colormap_dendrogram(
    Z, leaf_labels, colors_hex, output_path: Path,
):
    """Dendrogram with leaf labels colored by phylogeny colormap."""
    n = len(leaf_labels)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.22), 5.5), constrained_layout=True)
    dendrogram(
        Z, ax=ax, labels=leaf_labels, leaf_font_size=6,
        color_threshold=0, above_threshold_color="#2c3e50",
        link_color_func=lambda k: "#2c3e50",
    )
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        try:
            idx = list(leaf_labels).index(label_text)
            tick_label.set_color(colors_hex[idx])
        except (ValueError, IndexError):
            pass
    ax.set_title(
        f"16S Phylogeny Ward dendrogram  —  {n} strains",
        fontsize=11,
    )
    ax.set_ylabel("Ward merge cost")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_2d_hmds(
    coords: np.ndarray,
    labels: list[str],
    colors_hex: list[str],
    metrics: dict,
    title_label: str,
    output_path: Path,
):
    """2D Poincaré disk with phylogeny colormap."""
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

    # Unit circle boundary
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax.add_artist(circle)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors_hex, s=60, alpha=0.90,
        edgecolors="white", linewidths=0.6,
    )
    for i, label in enumerate(labels):
        ax.annotate(
            str(label), (coords[i, 0], coords[i, 1]),
            fontsize=4, alpha=0.55, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 3),
        )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Poincaré 1", fontsize=9)
    ax.set_ylabel("Poincaré 2", fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.4, alpha=0.5)
    ax.set_title(
        f"{title_label} HMDS 2D — 16S Phylogeny colormap\n"
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_2d_hmds_html(
    coords: np.ndarray,
    labels: list[str],
    colors_hex: list[str],
    metrics: dict,
    title_label: str,
    output_path: Path,
):
    """Interactive 2D Poincaré disk via plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers+text",
        marker=dict(
            size=8, color=colors_hex, opacity=0.92,
            line=dict(color="white", width=0.5),
        ),
        text=[str(l) for l in labels],
        textposition="top center",
        textfont=dict(size=7, color="#333333"),
        hovertext=[f"<b>{l}</b>" for l in labels],
        hoverinfo="text",
        showlegend=False,
    ))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines",
        line=dict(color="#6b7280", width=0.8),
        hoverinfo="skip", showlegend=False,
    ))

    rho = metrics['distance_spearman']
    stress = metrics['normalized_raw_stress']
    fig.update_layout(
        title=dict(
            text=f"{title_label} HMDS 2D — 16S Phylogeny colormap<br>"
                 f"<sup>ρ={rho:.3f}  stress={stress:.3f}</sup>",
            font=dict(size=14),
        ),
        xaxis=dict(range=[-1.08, 1.08], constrain="domain", title="Poincaré 1"),
        yaxis=dict(range=[-1.08, 1.08], scaleanchor="x", scaleratio=1,
                   title="Poincaré 2"),
        width=950, height=850,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"  → {output_path}")


# ======================================================================
# main
# ======================================================================

def main():
    # ── 1. Parse 16S tree → distance matrix ──
    print("=" * 60)
    print("Step 1: Parsing 16S phylogenetic tree...")
    print("=" * 60)
    tree_dist, tree_leaves = load_tree_distance()
    n_tree = len(tree_leaves)

    # ── 2. Load chemical RDM → get 106 AIDs ──
    chem_dist, chem_aids = load_chemical_rdm()
    print(f"Chemical RDM: {len(chem_aids)} strains")

    # ── 3. Intersect tree leaves with chemical AIDs ──
    common_aids = sorted(set(tree_leaves) & set(chem_aids))
    print(f"Intersection (tree ∩ chemical): {len(common_aids)} strains")
    if len(common_aids) < 10:
        print("ERROR: Too few common strains!")
        return

    # Subset tree distance to 106 experimental AIDs
    tree_idx = {name: i for i, name in enumerate(tree_leaves)}
    common_idx = [tree_idx[a] for a in common_aids]
    tree_dist_sub = tree_dist[np.ix_(common_idx, common_idx)]
    print(f"Phylogeny distance range (106 strains): "
          f"[{tree_dist_sub.min():.4f}, {tree_dist_sub.max():.4f}]")

    # ── 4. Ward clustering on 106 experimental strains → colormap ──
    print("\n" + "=" * 60)
    print("Step 2: Ward clustering on 106 experimental strains → colormap...")
    print("=" * 60)
    Z = linkage(squareform(tree_dist_sub), method="ward")
    colors_rgba, order, normed = leaf_order_colormap(Z, len(common_aids), cmap_name=CMAP_NAME)
    colors_hex = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in colors_rgba
    ]

    # Save colormap
    colormap_df = pd.DataFrame({
        "aid": common_aids,
        "color_hex": colors_hex,
        "leaf_order_position": [list(order).index(i) for i in range(len(common_aids))],
        "color_normed": [normed[list(order).index(i)] for i in range(len(common_aids))],
    })
    colormap_df.to_csv(OUTPUT_DIR / "phylogeny_colormap.csv", index=False)
    print(f"  → phylogeny_colormap.csv ({len(common_aids)} strains)")

    # Add genus/species info for dendrogram labels
    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    stim_info = stim_info.drop_duplicates(subset="aid").set_index("aid")
    leaf_labels = [
        f"{a} [{stim_info.loc[a, 'genus']}]" if a in stim_info.index else a
        for a in common_aids
    ]

    plot_colormap_dendrogram(
        Z, leaf_labels, colors_hex,
        OUTPUT_DIR / "phylogeny_dendrogram.png",
    )

    # Colormap swatches
    fig, ax = plt.subplots(figsize=(max(10, len(common_aids) * 0.22), 2.5),
                           constrained_layout=True)
    ordered_colors = [colors_hex[leaf_pos] for leaf_pos in order]
    ordered_labels = [leaf_labels[leaf_pos] for leaf_pos in order]
    for i, (color, label) in enumerate(zip(ordered_colors, ordered_labels)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color,
                                    edgecolor="white", linewidth=0.5))
        ax.text(i + 0.5, -0.35, label, ha="center", va="top", fontsize=5,
                rotation=90)
    ax.set_xlim(0, len(common_aids))
    ax.set_ylim(-2.5, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"16S Phylogeny Ward leaf-order colormap — {len(common_aids)} strains",
                 fontsize=11)
    fig.savefig(OUTPUT_DIR / "phylogeny_colormap_swatches.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → phylogeny_colormap_swatches.png")

    # ── 5. Build AID → color map ──
    aid_to_color = dict(zip(common_aids, colors_hex))

    # ── 6. Chemical HMDS ──
    print("\n" + "=" * 60)
    print("Step 3: Chemical HMDS with phylogeny colors...")
    print("=" * 60)
    # Subset chemical dist to common AIDs
    chem_idx = {a: i for i, a in enumerate(chem_aids)}
    chem_common_idx = [chem_idx[a] for a in common_aids]
    chem_dist_sub = chem_dist[np.ix_(chem_common_idx, chem_common_idx)]

    chem_coords, chem_dn, chem_pred, chem_metrics = embed_hdms(chem_dist_sub)
    chem_colors = [aid_to_color[a] for a in common_aids]
    print(f"  ρ={chem_metrics['distance_spearman']:.3f}  "
          f"stress={chem_metrics['normalized_raw_stress']:.3f}")

    plot_2d_hmds(
        chem_coords, common_aids, chem_colors, chem_metrics,
        "Chemical PCA",
        OUTPUT_DIR / "chemical_hmds_2d_phylogeny.png",
    )
    plot_2d_hmds_html(
        chem_coords, common_aids, chem_colors, chem_metrics,
        "Chemical PCA",
        OUTPUT_DIR / "chemical_hmds_2d_phylogeny.html",
    )

    # Chemical Shepard
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    orig_c = upper_triangle(chem_dn)
    pred_c = upper_triangle(chem_pred)
    limit_c = float(max(orig_c.max(), pred_c.max()) * 1.04)
    ax.scatter(orig_c, pred_c, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit_c], [0, limit_c], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit_c); ax.set_ylim(0, limit_c)
    ax.set_aspect("equal")
    ax.set_xlabel("Input Euclidean distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"Chemical Shepard\nρ={chem_metrics['distance_spearman']:.3f}  "
        f"stress={chem_metrics['normalized_raw_stress']:.3f}", fontsize=10)
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(OUTPUT_DIR / "chemical_hmds_2d_phylogeny_shepard.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → shepard diagram")

    # ── 7. Neural HMDS ──
    print("\n" + "=" * 60)
    print("Step 4: Neural HMDS with phylogeny colors...")
    print("=" * 60)

    # Build neural prototypes
    mats: dict[str, pd.DataFrame] = {}
    for neuron in ALL_NEURONS:
        mats[neuron] = _build_prototype_matrix(raw, neuron)

    neural_dist, neural_stimuli = build_neural_distance(mats)
    # Map stimulus → AID
    stim_info2 = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    stim_to_aid_map = {}
    for stim, row in stim_info2.iterrows():
        stim_to_aid_map[stim] = row["aid"]

    # Map neural stimuli to AIDs, keep only those in common_aids
    neural_aids_list = []
    neural_idx_keep = []
    for i, stim in enumerate(neural_stimuli):
        aid = stim_to_aid_map.get(stim)
        if aid is not None and aid in aid_to_color:
            neural_aids_list.append(aid)
            neural_idx_keep.append(i)

    neural_dist_sub = neural_dist[np.ix_(neural_idx_keep, neural_idx_keep)]
    print(f"Neural: {len(neural_idx_keep)} strains matched to phylogeny AIDs")

    neural_coords, neural_dn, neural_pred, neural_metrics = embed_hdms(neural_dist_sub)
    neural_colors = [aid_to_color[a] for a in neural_aids_list]
    print(f"  ρ={neural_metrics['distance_spearman']:.3f}  "
          f"stress={neural_metrics['normalized_raw_stress']:.3f}")

    plot_2d_hmds(
        neural_coords, neural_aids_list, neural_colors, neural_metrics,
        "Neural",
        OUTPUT_DIR / "neural_hmds_2d_phylogeny.png",
    )
    plot_2d_hmds_html(
        neural_coords, neural_aids_list, neural_colors, neural_metrics,
        "Neural",
        OUTPUT_DIR / "neural_hmds_2d_phylogeny.html",
    )

    # Neural Shepard
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    orig_n = upper_triangle(neural_dn)
    pred_n = upper_triangle(neural_pred)
    limit_n = float(max(orig_n.max(), pred_n.max()) * 1.04)
    ax.scatter(orig_n, pred_n, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit_n], [0, limit_n], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit_n); ax.set_ylim(0, limit_n)
    ax.set_aspect("equal")
    ax.set_xlabel("Input correlation distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"Neural Shepard\nρ={neural_metrics['distance_spearman']:.3f}  "
        f"stress={neural_metrics['normalized_raw_stress']:.3f}", fontsize=10)
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(OUTPUT_DIR / "neural_hmds_2d_phylogeny_shepard.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → shepard diagram")

    print(f"\nDone → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
