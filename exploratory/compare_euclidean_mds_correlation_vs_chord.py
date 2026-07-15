"""Compare Euclidean MDS on correlation distance vs chord distance — 106bac FT.

Pipeline:
  1. Build full-trajectory neural prototypes → Pearson correlation RDM
  2. Correlation distance (1-r) → Euclidean MDS (2D + 3D)
  3. Chord distance sqrt(2(1-r)) → normalize → Euclidean MDS (2D + 3D)
  4. All plots use 16S phylogeny Ward leaf-order colormap.
"""

from __future__ import annotations

import sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = str(PROJECT_ROOT / "src")
EXPL = str(PROJECT_ROOT / "exploratory")
for p in (SRC, EXPL):
    if p not in sys.path:
        sys.path.insert(0, p)

from bacteria_analysis.features.neural import build_trial_feature_matrix, neural_feature_columns
from bacteria_analysis.analyses.hmds import (
    chord_from_linear, clean_distance_matrix, normalize_to_max_two,
    upper_triangle, preservation_metrics, leaf_order_colormap,
)
from compare_neural_rdm_variants_hmds_106bac_phylo import (
    _aggregate_features, _sample_id_from_stim_name,
)

OUTPUT_DIR = Path("results/euclidean_mds_correlation_vs_chord")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260714
MDS_KWARGS = dict(metric=True, dissimilarity="precomputed", random_state=SEED,
                   n_init=8, max_iter=1000, normalized_stress=False)

# ---------------------------------------------------------------------------
# 1. Build phylogeny colormap
# ---------------------------------------------------------------------------
from Bio import Phylo
from scipy.cluster.hierarchy import linkage, leaves_list, cophenet

tree = Phylo.read("data/16S.aln.trim.fa.treefile", "newick")
terminals = tree.get_terminals()
leaf_names = [str(t.name) for t in terminals]
n_tree = len(leaf_names)
tree_dist = np.zeros((n_tree, n_tree))
for i in range(n_tree):
    for j in range(i + 1, n_tree):
        d = tree.distance(terminals[i], terminals[j]) or 0.0
        tree_dist[i, j] = tree_dist[j, i] = d
Z = linkage(squareform(tree_dist), method="ward")
colors_rgba, order, _normed = leaf_order_colormap(Z, n_tree)
colors_hex = ["#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))
              for r, g, b, _ in colors_rgba]
aid_to_color = dict(zip(leaf_names, colors_hex))
print(f"Phylogeny colormap: {len(aid_to_color)} leaves")

# ---------------------------------------------------------------------------
# 2. Build 106bac full-trajectory correlation RDM
# ---------------------------------------------------------------------------
raw = pd.read_parquet("data/106bac.parquet")
tf = build_trial_feature_matrix(raw, view="full_trajectory", merge_lr=True)
cols = neural_feature_columns(tf)
protos = _aggregate_features(tf, group_columns=["stim_name"], feature_columns=cols)
protos["sample_id"] = protos["stim_name"].apply(_sample_id_from_stim_name)
print(f"Prototypes: {len(protos)} stimuli, {len(cols)} features")

arr = protos[cols].to_numpy(float)
n = arr.shape[0]
labels = protos["sample_id"].tolist()

# Pearson correlation RDM (1 - r)
corr_rdm = np.full((n, n), np.nan)
np.fill_diagonal(corr_rdm, 0.0)
for i in range(n):
    for j in range(i + 1, n):
        v = np.isfinite(arr[i]) & np.isfinite(arr[j])
        if v.sum() < 2: continue
        a, b = arr[i, v], arr[j, v]
        if np.std(a) == 0 or np.std(b) == 0: continue
        corr_rdm[i, j] = corr_rdm[j, i] = float(np.clip(1 - np.corrcoef(a, b)[0, 1], 0, 2))

# Filter to phylogeny AIDs
common_aids = sorted(set(labels) & set(aid_to_color.keys()))
aid_idx = [labels.index(a) for a in common_aids]
corr_rdm_sub = corr_rdm[np.ix_(aid_idx, aid_idx)]
phylo_colors = [aid_to_color[a] for a in common_aids]
print(f"Intersection with phylogeny: {len(common_aids)} AIDs")

# Chord distance
chord_dist = chord_from_linear(corr_rdm_sub)
chord_norm = normalize_to_max_two(chord_dist)

# Rank-transformed correlation distance
from scipy.stats import rankdata
orig_pairs = upper_triangle(corr_rdm_sub)
ranks = rankdata(orig_pairs)
rank_dist = np.zeros_like(corr_rdm_sub)
tri_i, tri_j = np.triu_indices(corr_rdm_sub.shape[0], k=1)
for k, (i, j) in enumerate(zip(tri_i, tri_j)):
    rank_dist[i, j] = rank_dist[j, i] = ranks[k]
rank_dist_norm = rank_dist / ranks.max() * 2.0  # scale to [0, 2]

# ---------------------------------------------------------------------------
# 3. Euclidean MDS
# ---------------------------------------------------------------------------
def run_mds(dist_matrix, dim, label):
    mds = MDS(n_components=dim, **MDS_KWARGS)
    coords = mds.fit_transform(dist_matrix)
    emb = squareform(pdist(coords, metric="euclidean"))
    n_params = dist_matrix.shape[0] * dim + 1 - dim * (dim - 1) / 2
    metrics = preservation_metrics(dist_matrix, emb, emb, n_params=n_params)
    print(f"  {label:35s} {dim}D: stress={metrics['normalized_raw_stress']:.4f}  "
          f"rho={metrics['distance_spearman']:.4f}")
    return coords, metrics

results = {}
for dist_input, dist_label, key in [
    (corr_rdm_sub,  "Correlation distance (1-r)", "correlation"),
    (chord_norm,    "Chord distance  sqrt(2(1-r)) normalized", "chord"),
    (rank_dist_norm,"Rank-transformed correlation  [0,2]", "rank"),
]:
    for dim in [2, 3]:
        coords, m = run_mds(dist_input, dim, dist_label)
        results[(key, dim)] = (coords, m, dist_input)

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
def plot_2d(coords, aids, colors, metrics, title, path):
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=60, alpha=0.90,
               edgecolors="white", linewidths=0.6)
    for i, a in enumerate(aids):
        ax.annotate(str(a), (coords[i, 0], coords[i, 1]),
                    fontsize=4, alpha=0.55, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3))
    ax.set_aspect("equal")
    ax.set_xlabel("MDS 1", fontsize=9)
    ax.set_ylabel("MDS 2", fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.4, alpha=0.5)
    rho = metrics["distance_spearman"]
    stress = metrics["normalized_raw_stress"]
    ax.set_title(f"{title}\nrho={rho:.3f}  stress={stress:.3f}", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

def plot_3d_html(coords, aids, colors, metrics, title, path):
    import plotly.graph_objects as go
    radii = np.linalg.norm(coords, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        marker=dict(size=5, color=colors, opacity=0.90,
                    line=dict(color="white", width=0.4)),
        text=[str(a) for a in aids],
        textposition="top center",
        textfont=dict(size=7, color="#444444"),
        hovertext=[f"<b>{a}</b><br>r={radii[i]:.3f}" for i, a in enumerate(aids)],
        hoverinfo="text", showlegend=False,
    ))
    rho = metrics["distance_spearman"]
    stress = metrics["normalized_raw_stress"]
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>rho={rho:.3f}  stress={stress:.3f}</sup>",
                    font=dict(size=14)),
        scene=dict(aspectmode="cube"),
        width=900, height=850,
    )
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  -> {path}")

for dist_key, dist_label in [
    ("correlation", "Correlation (1-r) + Euclidean MDS"),
    ("chord",        "Chord + Euclidean MDS"),
]:
    for dim in [2, 3]:
        coords, m, _ = results[(dist_key, dim)]
        tag = f"{dist_key}_{dim}d"
        if dim == 2:
            plot_2d(coords, common_aids, phylo_colors, m,
                    f"106bac FT  {dist_label}  {dim}D",
                    OUTPUT_DIR / f"euclidean_mds_{tag}.png")
        else:
            plot_3d_html(coords, common_aids, phylo_colors, m,
                         f"106bac FT  {dist_label}  {dim}D",
                         OUTPUT_DIR / f"euclidean_mds_{tag}.html")

# ---------------------------------------------------------------------------
# 5. Shepard diagrams
# ---------------------------------------------------------------------------
def plot_shepard(original_dist, embedded_dist, metrics, title, path):
    orig = upper_triangle(original_dist)
    emb = upper_triangle(embedded_dist)
    limit = float(max(orig.max(), emb.max()) * 1.04)
    rho = metrics["distance_spearman"]
    stress = metrics["normalized_raw_stress"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    # --- left: Shepard scatter ---
    ax = axes[0]
    ax.scatter(orig, emb, s=6, color="#4a5568", alpha=0.20, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input distance", fontsize=9)
    ax.set_ylabel("Embedded distance", fontsize=9)
    ax.set_title(f"Shepard\nrho={rho:.3f}  stress={stress:.3f}", fontsize=10)
    ax.grid(True, color="#e5e7eb", lw=0.4)

    # --- right: distance distribution ---
    ax = axes[1]
    bins = np.linspace(0, limit, 50)
    ax.hist(orig, bins=bins, alpha=0.55, color="#2563EB", label="Input", density=True)
    ax.hist(emb, bins=bins, alpha=0.45, color="#DC2626", label="Embedded", density=True)
    ax.set_xlabel("Distance", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Distance distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, color="#e5e7eb", lw=0.4)

    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

for dist_key, dist_label in [("correlation", "Correlation (1-r) + Euclidean MDS"),
                               ("chord", "Chord + Euclidean MDS"),
                               ("rank", "Rank-transformed + Euclidean MDS")]:
    for dim in [2, 3]:
        coords, m, dist_input = results[(dist_key, dim)]
        emb_dist = squareform(pdist(coords, metric="euclidean"))
        tag = f"{dist_key}_{dim}d"
        plot_shepard(dist_input, emb_dist, m,
                     f"106bac FT  {dist_label}  {dim}D",
                     OUTPUT_DIR / f"shepard_{tag}.png")

# ---------------------------------------------------------------------------
# 6. Diagnostic: distance distribution stats
# ---------------------------------------------------------------------------
print(f"\n--- Distance distribution diagnostics ---")
for dist_key, dist_label in [("correlation", "Correlation"), ("chord", "Chord"), ("rank", "Rank")]:
    _, _, dist_input = results[(dist_key, 2)]
    pairs = upper_triangle(dist_input)
    print(f"\n{dist_label}:")
    print(f"  range:  [{pairs.min():.4f}, {pairs.max():.4f}]")
    print(f"  mean={pairs.mean():.4f}  median={np.median(pairs):.4f}  std={pairs.std():.4f}")
    print(f"  q01={np.quantile(pairs,0.01):.4f}  q05={np.quantile(pairs,0.05):.4f}")
    print(f"  q95={np.quantile(pairs,0.95):.4f}  q99={np.quantile(pairs,0.99):.4f}")
    # Coefficient of variation
    print(f"  CV (std/mean) = {pairs.std()/pairs.mean():.3f}")
    # Fraction of pairs in central 50%
    q25, q75 = np.quantile(pairs, [0.25, 0.75])
    print(f"  IQR / range = {(q75-q25)/(pairs.max()-pairs.min()):.3f}")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print(f"\nDone -> {OUTPUT_DIR}")
print(f"\n{'Method':40s} {'Dim':>3s}  {'stress':>7s}  {'rho':>7s}")
for dist_key, dist_label in [("correlation", "Correlation"), ("chord", "Chord"), ("rank", "Rank")]:
    for dim in [2, 3]:
        _, m, _ = results[(dist_key, dim)]
        print(f"{dist_label:40s} {dim:>3d}  {m['normalized_raw_stress']:7.4f}  {m['distance_spearman']:7.4f}")
