"""106bac neural chord HMDS, colored by chemical PCA Ward leaf-order colormap.

Pipeline:
  neural trace prototypes → Pearson correlation distance (1−r) → chord distance
  → normalize_to_max_two → HMDS (2D + 3D) → color by chemical colormap

Matches plot_86bac_hmds_by_gram_status.py style for the 3D Poincare ball.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis.features.neural import build_trial_feature_matrix, neural_feature_columns, build_stimulus_prototypes
from bacteria_analysis._data_loaders import enrich_neural_dataframe
from compare_86bac_chord_hmds import (
    chord_from_linear,
    normalize_to_max_two,
    upper_triangle,
    preservation_metrics,
    lorentz_to_poincare,
    recenter_poincare,
    scipy_hyperbolic_mds,
)

OUTPUT_DIR = Path("results/chemical_pca_ward")
NEURON = "merged"  # use all L/R merged neurons
CMAP_NAME = "turbo"
STARTS = 8
MAXITER = 900
SEED = 42


# ── helpers ────────────────────────────────────────────────────────────

def add_poincare_wireframe(axis) -> None:
    u, v = np.mgrid[0:2 * np.pi:32j, 0:np.pi:16j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    axis.plot_wireframe(
        x, y, z, color="#4b5563", linewidth=0.85, alpha=0.40,
        rstride=3, cstride=3,
    )


def set_poincare_3d_axis(axis) -> None:
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-1.02, 1.02)
    axis.set_zlim(-1.02, 1.02)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.set_zlabel("Poincare 3", fontsize=9)
    axis.tick_params(labelsize=8)


def plot_shepard(dist_input, dist_predicted, metrics, output_path):
    orig = upper_triangle(dist_input)
    pred = upper_triangle(dist_predicted)
    limit = float(max(orig.max(), pred.max()) * 1.04)
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.scatter(orig, pred, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input chord distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"ρ={metrics['distance_spearman']:.3f}  r={metrics['distance_pearson']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}", fontsize=10,
    )
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load colormap ───────────────────────────────────────────────
    colormap = pd.read_csv(OUTPUT_DIR / "aid_colormap.csv")
    color_by_aid = dict(zip(colormap["aid"], colormap["color_hex"]))
    print(f"Loaded colormap: {len(color_by_aid)} AIDs")

    # ── 2. Build neural prototypes ─────────────────────────────────────
    raw = pd.read_parquet("data/106bac.parquet")
    features = build_trial_feature_matrix(raw, view="full_trajectory", merge_lr=True)
    prototypes = build_stimulus_prototypes(features, aggregation="median")
    feature_cols = neural_feature_columns(prototypes)

    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    prototypes["aid"] = prototypes["stimulus"].map(stim_info["aid"])

    # Keep only strains that have chemical colormap
    prototypes = prototypes[prototypes["aid"].isin(color_by_aid.keys())].copy()
    aids = prototypes["aid"].tolist()
    print(f"Neural prototypes: {len(prototypes)} strains × {len(feature_cols)} features")

    # ── 3. Pearson correlation distance → chord ────────────────────────
    values = prototypes.loc[:, feature_cols].to_numpy(dtype=float)
    # z-score each row (Pearson is invariant to this but keeps data well-behaved)
    row_means = np.nanmean(values, axis=1, keepdims=True)
    row_stds = np.nanstd(values, axis=1, ddof=0, keepdims=True)
    values_z = np.where(row_stds > 0, (values - row_means) / row_stds, 0.0)

    r_mat = np.corrcoef(values_z)
    pearson_dist = np.clip(1.0 - r_mat, 0.0, 2.0)
    np.fill_diagonal(pearson_dist, 0.0)
    pearson_dist = (pearson_dist + pearson_dist.T) / 2.0

    chord_dist = chord_from_linear(pearson_dist)
    chord_norm = normalize_to_max_two(chord_dist)
    print(f"Chord distance matrix: {chord_norm.shape}")

    # ── 4. HMDS 2D + 3D ────────────────────────────────────────────────
    colors_hex = [color_by_aid[aid] for aid in aids]
    n = len(aids)

    for dim in [2, 3]:
        print(f"\n--- {dim}D HMDS ---")
        lorentz, embedded, lam, _ = scipy_hyperbolic_mds(
            chord_norm, dim=dim, starts=STARTS, maxiter=MAXITER, seed=SEED,
        )
        poincare_coords = recenter_poincare(lorentz_to_poincare(lorentz))
        predicted = embedded / lam
        metrics = preservation_metrics(chord_norm, embedded, predicted, n_params=n * dim)
        print(f"  ρ={metrics['distance_spearman']:.3f}  stress={metrics['normalized_raw_stress']:.3f}")

        # Store for later use
        if dim == 2:
            poincare_2d = poincare_coords
            metrics_2d = metrics
        else:
            poincare_3d = poincare_coords
            predicted_3d = predicted
            chord_norm_ref = chord_norm  # keep reference

        # Shepard
        plot_shepard(
            chord_norm, predicted, metrics,
            OUTPUT_DIR / f"neural_chord_hmds_{dim}d_shepard.png",
        )

        # 2D disk
        if dim == 2:
            fig, ax = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
            circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
            ax.add_artist(circle)
            ax.scatter(poincare_2d[:, 0], poincare_2d[:, 1], c=colors_hex, s=60, alpha=0.90,
                       edgecolors="white", linewidths=0.6)
            for i, aid in enumerate(aids):
                ax.annotate(aid, (poincare_2d[i, 0], poincare_2d[i, 1]),
                            fontsize=4, alpha=0.5, ha="center", va="bottom",
                            textcoords="offset points", xytext=(0, 3))
            ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Poincare 1", fontsize=9); ax.set_ylabel("Poincare 2", fontsize=9)
            ax.grid(True, color="#e5e7eb", linewidth=0.6)
            ax.set_title(
                f"106bac Neural chord HMDS 2D  —  chemical colormap\n"
                f"ρ={metrics_2d['distance_spearman']:.3f}  stress={metrics_2d['normalized_raw_stress']:.3f}",
                fontsize=10,
            )
            fig.savefig(OUTPUT_DIR / "neural_chord_hmds_2d_poincare.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        # 3D ball
        if dim == 3:
            fig = plt.figure(figsize=(10, 9), constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            add_poincare_wireframe(ax)
            ax.scatter(poincare_3d[:, 0], poincare_3d[:, 1], poincare_3d[:, 2],
                       c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
            set_poincare_3d_axis(ax)
            ax.set_title(
                f"106bac Neural chord HMDS 3D  —  chemical colormap\n"
                f"ρ={metrics['distance_spearman']:.3f}  stress={metrics['normalized_raw_stress']:.3f}",
                fontsize=10,
            )
            fig.savefig(OUTPUT_DIR / "neural_chord_hmds_3d_poincare_ball.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ── 5. 3D comparison grid: neural × 3 views + chemical × 3 views ───

    # Reload chemical HMDS coords
    chem_coords = pd.read_csv(OUTPUT_DIR / "hmds_coordinates.csv")
    # Ensure same AID order
    chem_by_aid = chem_coords.set_index("aid")
    neural_aids_set = set(aids)
    shared = sorted(neural_aids_set & set(chem_by_aid.index))
    poincare_chem = chem_by_aid.loc[shared, ["hmds3d_p1", "hmds3d_p2", "hmds3d_p3"]].to_numpy(float)

    # Rebuild neural 3D for shared order
    shared_idx = [aids.index(a) for a in shared]
    poincare_neur_shared = poincare_3d[shared_idx]

    colors_shared = [color_by_aid[a] for a in shared]

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)

    views = [
        ("Neural 3D", poincare_neur_shared, None),
        ("Neural top", poincare_neur_shared, {"elev": 90, "azim": 0}),
        ("Neural side", poincare_neur_shared, {"elev": 10, "azim": 90}),
        ("Chemical 3D", poincare_chem, None),
        ("Chemical top", poincare_chem, {"elev": 90, "azim": 0}),
        ("Chemical side", poincare_chem, {"elev": 10, "azim": 90}),
    ]
    for i, (title, coords, view_kwargs) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        add_poincare_wireframe(ax)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors_shared, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
        if view_kwargs:
            ax.view_init(**view_kwargs)
        set_poincare_3d_axis(ax)
        ax.set_title(title, fontsize=10)

    fig.suptitle("106bac Neural chord HMDS vs Chemical PCA HMDS  —  chemical colormap",
                 fontsize=13, y=1.01)
    fig.savefig(OUTPUT_DIR / "neural_vs_chemical_hmds_3d_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 6. Save neural HMDS coordinates ────────────────────────────────
    pd.DataFrame({
        "aid": aids,
        "hmds2d_p1": poincare_2d[:, 0],
        "hmds2d_p2": poincare_2d[:, 1],
        "hmds3d_p1": poincare_3d[:, 0],
        "hmds3d_p2": poincare_3d[:, 1],
        "hmds3d_p3": poincare_3d[:, 2],
        "color_hex": colors_hex,
    }).to_csv(OUTPUT_DIR / "neural_hmds_coordinates.csv", index=False)
    print(f"\nSaved: neural_hmds_coordinates.csv")
    print(f"All outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
