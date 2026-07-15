"""Chemical distance embeddings: 3 pipelines x {HMDS, MDS} x {2D, 3D}.

  log2 FC only          -> Euclidean -> max2
  log2 + z-score        -> Euclidean -> max2
  log2 + z-score + PCA10 -> Euclidean -> max2

Output: comparison grids (2D + 3D), plus individual 3D Poincare balls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bacteria_analysis._data_loaders import enrich_neural_dataframe
from compare_86bac_chord_hmds import (
    normalize_to_max_two, scipy_hyperbolic_mds,
    lorentz_to_poincare, recenter_poincare, preservation_metrics,
    upper_triangle, euclidean_metric_mds,
)

OUTPUT_DIR = Path("results/chemical_hmds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PCA = 10; STARTS = 8; MAXITER = 900; SEED = 42


# ======================================================================
# Colormap
# ======================================================================

def leaf_order_colormap(Z, n, cmap_name="turbo"):
    from scipy.cluster.hierarchy import cophenet
    order = leaves_list(Z)
    coph = squareform(cophenet(Z))
    gaps = np.array([coph[order[i], order[i + 1]] for i in range(n - 1)])
    med = np.median(gaps) or 1.0
    cumdist = np.concatenate([[0.0], np.cumsum(np.arcsinh(gaps / med))])
    normed = (cumdist - cumdist.min()) / (cumdist.max() - cumdist.min() + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(normed)
    colors_orig = np.zeros((n, 4))
    for leaf_pos, orig_idx in enumerate(order):
        colors_orig[orig_idx] = colors[leaf_pos]
    return colors_orig, order, normed


# ======================================================================
# Distance builders
# ======================================================================

def build_distance_log2(met_log2):
    return normalize_to_max_two(squareform(pdist(met_log2, "euclidean")))


def build_distance_log2_zscore(met_log2):
    met_z = (met_log2 - met_log2.mean(axis=0)) / met_log2.std(axis=0)
    return normalize_to_max_two(squareform(pdist(met_z, "euclidean")))


def build_distance_log2_zscore_pca(met_log2):
    met_z = (met_log2 - met_log2.mean(axis=0)) / met_log2.std(axis=0)
    scores = PCA(n_components=N_PCA).fit_transform(met_z)
    return normalize_to_max_two(squareform(pdist(scores, "euclidean")))


# ======================================================================
# Embedding runner
# ======================================================================

def embed(dist, dim, use_hmds):
    if use_hmds:
        lorentz, emb, lam, _ = scipy_hyperbolic_mds(
            dist, dim=dim, starts=STARTS, maxiter=MAXITER, seed=SEED)
        coords = recenter_poincare(lorentz_to_poincare(lorentz))
        predicted = emb / lam
    else:
        coords, emb = euclidean_metric_mds(dist, dim=dim, seed=SEED, n_init=4)
        predicted = emb
    metrics = preservation_metrics(dist, emb, predicted,
                                   n_params=dist.shape[0] * dim)
    return coords, predicted, metrics


# ======================================================================
# Comparison grid
# ======================================================================

def plot_comparison_grid(results, colors_hex, aids, dim, output_path):
    """Rows=embedding/Shepard, Cols=methods.

    For 3D HMDS: embedding row uses 3d projection.
    For others: 2D scatter.
    """
    n_cols = len(results)
    fig = plt.figure(figsize=(5*n_cols + 2, 10))

    # Create axes individually
    emb_axes = []
    shp_axes = []
    for col_idx, res in enumerate(results):
        is_hmds = res["hmds"]
        if dim == 3 and is_hmds:
            emb_axes.append(fig.add_subplot(2, n_cols, col_idx + 1, projection="3d"))
        else:
            emb_axes.append(fig.add_subplot(2, n_cols, col_idx + 1))
        shp_axes.append(fig.add_subplot(2, n_cols, n_cols + col_idx + 1))

    for col_idx, res in enumerate(results):
        ax_emb = emb_axes[col_idx]
        ax_shp = shp_axes[col_idx]

        coords = res["coords"]
        dist_in = res["input_distance"]
        dist_pred = res["predicted_distance"]
        metrics = res["metrics"]
        is_hmds = res["hmds"]

        if is_hmds and dim == 2:
            circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False,
                                alpha=0.5, lw=1)
            ax_emb.add_artist(circle)
            ax_emb.set_xlim(-1.05, 1.05); ax_emb.set_ylim(-1.05, 1.05)
            ax_emb.set_aspect("equal")

        if is_hmds:
            pref = "Poincare"
        else:
            pref = "MDS"

        if dim == 3 and is_hmds:
            ax_emb.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=colors_hex, s=24, alpha=0.88,
                           edgecolors="white", linewidths=0.25)
            ax_emb.set_xlim(-1.02, 1.02); ax_emb.set_ylim(-1.02, 1.02)
            ax_emb.set_zlim(-1.02, 1.02)
            ax_emb.set_xlabel(f"{pref} 1", fontsize=7)
            ax_emb.set_ylabel(f"{pref} 2", fontsize=7)
            ax_emb.set_zlabel(f"{pref} 3", fontsize=7)
        else:
            ax_emb.scatter(coords[:, 0], coords[:, 1],
                           c=colors_hex, s=32, alpha=0.88,
                           edgecolors="white", linewidths=0.3)
            ax_emb.set_xlabel(f"{pref} 1", fontsize=8)
            ax_emb.set_ylabel(f"{pref} 2", fontsize=8)
            if dim == 3 and not is_hmds:
                ax_emb.set_aspect("equal")

        ax_emb.set_title(res["title"], fontsize=9, fontweight="bold")
        ax_emb.tick_params(labelsize=6)

        # Shepard
        orig_pairs = upper_triangle(dist_in)
        pred_pairs = upper_triangle(dist_pred)
        limit = float(max(orig_pairs.max(), pred_pairs.max()) * 1.04)
        ax_shp.scatter(orig_pairs, pred_pairs, s=3, color="#4a5568", alpha=0.16,
                       linewidths=0)
        ax_shp.plot([0, limit], [0, limit], "--", color="#111827", lw=0.7)
        ax_shp.set_xlim(0, limit); ax_shp.set_ylim(0, limit)
        ax_shp.set_aspect("equal")
        ax_shp.set_xlabel("Input distance", fontsize=7)
        ax_shp.set_ylabel("Predicted", fontsize=7)
        ax_shp.set_title(f"rho={metrics['distance_spearman']:.3f}  "
                         f"stress={metrics['normalized_raw_stress']:.3f}",
                         fontsize=8)
        ax_shp.tick_params(labelsize=6)
        ax_shp.grid(True, color="#e5e7eb", lw=0.3)

    fig.suptitle(f"Chemical distance embeddings — {dim}D, 106 strains",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {output_path}")


# ======================================================================
# individual 3D Poincare balls
# ======================================================================

def plot_3d_ball(coords, colors_hex, aids, title_str, metrics, output_path):
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    u, v = np.mgrid[0:2*np.pi:32j, 0:np.pi:16j]
    ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                      color="#6b7280", linewidth=0.4, alpha=0.18,
                      rstride=2, cstride=2)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors_hex, s=42, alpha=0.9,
               edgecolors="white", linewidths=0.4)
    for i, aid in enumerate(aids):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2],
                str(aid), fontsize=3.2, alpha=0.5, ha="center")
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_zlim(-1.02, 1.02)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"{title_str}\nrho={metrics['distance_spearman']:.3f}  "
                 f"stress={metrics['normalized_raw_stress']:.3f}", fontsize=10)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {output_path}")


# ======================================================================
# main
# ======================================================================

def main():
    met_df = pd.read_parquet("results/metabolite_modules/metabolites_reduced.parquet")
    met_log2 = np.log2(met_df.values.astype(float))
    all_aids = list(met_df.index)

    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    stim_info = stim_info.drop_duplicates(subset="aid").set_index("aid")
    neural_aids = set(stim_info.index)

    mask = [a in neural_aids for a in all_aids]
    aids = [a for a, m in zip(all_aids, mask) if m]
    met_log2_106 = met_log2[mask]
    n = len(aids)

    Z = linkage(pdist(met_log2_106, "euclidean"), "ward")
    colors_rgba, _, _ = leaf_order_colormap(Z, n)
    colors_hex = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
                  for r, g, b, _ in colors_rgba]

    dists = {
        "log2 FC":               build_distance_log2(met_log2_106),
        "log2 + z-score":        build_distance_log2_zscore(met_log2_106),
        "log2 + z-score + PCA10": build_distance_log2_zscore_pca(met_log2_106),
    }

    # ---- 2D + 3D comparison grids ----
    for dim in [2, 3]:
        results = []
        for label, dist in dists.items():
            for use_hmds in [True, False]:
                emb_label = "HMDS" if use_hmds else "MDS"
                print(f"[{dim}D] {label} + {emb_label}...", end=" ", flush=True)
                coords, predicted, metrics = embed(dist, dim, use_hmds)
                print(f"rho={metrics['distance_spearman']:.3f}  "
                      f"stress={metrics['normalized_raw_stress']:.3f}")
                results.append({
                    "coords": coords, "input_distance": dist,
                    "predicted_distance": predicted, "metrics": metrics,
                    "hmds": use_hmds,
                    "title": f"{label}\n{emb_label}",
                })
        plot_comparison_grid(results, colors_hex, aids, dim,
                             OUTPUT_DIR / f"comparison_grid_{dim}d.png")

    # ---- individual 3D balls for best method per pipeline ----
    for label, dist in dists.items():
        print(f"[3D ball] {label} + HMDS...", end=" ", flush=True)
        coords, predicted, metrics = embed(dist, 3, True)
        print(f"rho={metrics['distance_spearman']:.3f}")
        slug = label.replace(" ", "_").replace("+", "").replace("__", "_")
        plot_3d_ball(coords, colors_hex, aids, label, metrics,
                     OUTPUT_DIR / f"poincare_ball_{slug}.png")

    # Save best coordinates
    best_dist = dists["log2 FC"]
    best_coords, _, best_m = embed(best_dist, 3, True)
    pd.DataFrame({
        "AID": aids,
        "poincare1": best_coords[:, 0],
        "poincare2": best_coords[:, 1],
        "poincare3": best_coords[:, 2],
        "color_hex": colors_hex,
    }).to_csv(OUTPUT_DIR / "poincare_coordinates.csv", index=False)
    print(f"-> {OUTPUT_DIR / 'poincare_coordinates.csv'}")


if __name__ == "__main__":
    main()
