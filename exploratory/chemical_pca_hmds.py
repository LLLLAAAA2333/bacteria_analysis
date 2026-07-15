"""HMDS embedding (2D + 3D) of chemical PCA Euclidean distance, colored by Ward leaf-order colormap.

Reads: results/chemical_pca_ward/chemical_pca_euclidean_rdm.csv
       results/chemical_pca_ward/aid_colormap.csv

Outputs: results/chemical_pca_ward/
  - euc_hmds_2d_poincare.png
  - euc_hmds_2d_shepard.png
  - euc_hmds_3d_poincare.png
  - euc_hmds_3d_shepard.png
  - euc_hmds_3d_poincare_ball.png  (3D Poincare ball with wireframe)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "exploratory"))

from compare_86bac_chord_hmds import (
    normalize_to_max_two,
    upper_triangle,
    euclidean_metric_mds,
    preservation_metrics,
    lorentz_to_poincare,
    recenter_poincare,
    scipy_hyperbolic_mds,
)

INPUT_DIR = Path("results/chemical_pca_ward")
OUTPUT_DIR = INPUT_DIR  # Same directory

STARTS = 8
MAXITER = 900
SEED = 42


# ── load ─────────────────────────────────────────────────────────────────


def load_data():
    """Load Euclidean distance matrix and colormap."""
    rdm = pd.read_csv(INPUT_DIR / "chemical_pca_euclidean_rdm.csv", index_col=0)
    colormap = pd.read_csv(INPUT_DIR / "aid_colormap.csv", index_col="aid")

    # Ensure consistent order
    aids = sorted(set(rdm.index) & set(colormap.index))
    dist_matrix = rdm.loc[aids, aids].to_numpy(dtype=float)

    # Clean
    dist_clean = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_clean, 0.0)
    dist_clean = np.maximum(dist_clean, 0.0)

    colors_hex = colormap.loc[aids, "color_hex"].tolist()
    return dist_clean, aids, colors_hex


# ── plots ────────────────────────────────────────────────────────────────


def plot_2d_poincare(
    coords: np.ndarray,
    colors_hex: list[str],
    aids: list[str],
    metrics: dict,
    output_path: Path,
):
    """2D Poincare disk scatter."""
    fig, ax = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax.add_artist(circle)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors_hex, s=60, alpha=0.90,
        edgecolors="white", linewidths=0.6,
    )
    for i, aid in enumerate(aids):
        ax.annotate(
            aid, (coords[i, 0], coords[i, 1]),
            fontsize=4, alpha=0.55, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 3),
        )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Poincaré 1", fontsize=9)
    ax.set_ylabel("Poincaré 2", fontsize=9)
    ax.set_title(
        f"Chemical PCA HMDS 2D\n"
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_shepard(
    dist_input: np.ndarray,
    dist_predicted: np.ndarray,
    metrics: dict,
    output_path: Path,
):
    """Shepard diagram: input vs predicted distances."""
    orig = upper_triangle(dist_input)
    pred = upper_triangle(dist_predicted)
    limit = float(max(orig.max(), pred.max()) * 1.04)

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.scatter(orig, pred, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input Euclidean distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"Shepard diagram\n"
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"r={metrics['distance_pearson']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def add_poincare_wireframe(axis) -> None:
    """Draw unit-sphere wireframe matching plot_hmds_by_chemical_distance.py style."""
    u, v = np.mgrid[0 : 2 * np.pi : 32j, 0 : np.pi : 16j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    axis.plot_wireframe(
        x, y, z,
        color="#4b5563",
        linewidth=0.85,
        alpha=0.40,
        rstride=3,
        cstride=3,
    )


def set_poincare_3d_axis(axis) -> None:
    """Standard 3D Poincare ball axis limits and labels."""
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-1.02, 1.02)
    axis.set_zlim(-1.02, 1.02)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.set_zlabel("Poincare 3", fontsize=9)
    axis.tick_params(labelsize=8)


def plot_3d_poincare(
    coords: np.ndarray,
    colors_hex: list[str],
    aids: list[str],
    metrics: dict,
    output_path: Path,
):
    """3D Poincare ball with wireframe sphere, matching reference style."""
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    add_poincare_wireframe(ax)

    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=colors_hex, s=48, alpha=0.92,
        edgecolor="white", linewidth=0.45,
    )
    for i, aid in enumerate(aids[:20]):  # label subset to avoid clutter
        ax.text(
            coords[i, 0], coords[i, 1], coords[i, 2],
            aid, fontsize=4, alpha=0.5, ha="center",
        )

    set_poincare_3d_axis(ax)
    ax.set_title(
        f"Chemical PCA HMDS 3D Poincaré ball\n"
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_all_views(
    coords_2d: np.ndarray,
    coords_3d: np.ndarray,
    colors_hex: list[str],
    aids: list[str],
    metrics_2d: dict,
    metrics_3d: dict,
    output_path: Path,
):
    """2×3 comparison: 2D Poincare, 3D Poincare, 2D Shepard, 3D Shepard, + 2 more 3D views."""
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)

    # 2D Poincare
    ax1 = fig.add_subplot(2, 3, 1)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax1.add_artist(circle)
    ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors_hex, s=44, alpha=0.92,
                edgecolor="white", linewidth=0.45)
    ax1.set_xlim(-1.03, 1.03); ax1.set_ylim(-1.03, 1.03)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(f"2D HMDS  ρ={metrics_2d['distance_spearman']:.3f}", fontsize=9)
    ax1.set_xlabel("Poincare 1", fontsize=9); ax1.set_ylabel("Poincare 2", fontsize=9)
    ax1.grid(True, color="#e5e7eb", linewidth=0.6)

    # 3D Poincare
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    add_poincare_wireframe(ax2)
    ax2.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
    set_poincare_3d_axis(ax2)
    ax2.set_title(f"3D HMDS  ρ={metrics_3d['distance_spearman']:.3f}", fontsize=9)

    # 3D alternative view 1 (top-down)
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    add_poincare_wireframe(ax3)
    ax3.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
    ax3.view_init(elev=90, azim=0)
    set_poincare_3d_axis(ax3)
    ax3.set_title(f"3D top view\nstress={metrics_3d['normalized_raw_stress']:.3f}", fontsize=9)

    # 2D Shepard
    ax4 = fig.add_subplot(2, 3, 4)
    _shepard_subplot(ax4, metrics_2d)

    # 3D Shepard
    ax5 = fig.add_subplot(2, 3, 5)
    _shepard_subplot(ax5, metrics_3d)

    # 3D alternative view 2 (side)
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    add_poincare_wireframe(ax6)
    ax6.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
    ax6.view_init(elev=10, azim=90)
    set_poincare_3d_axis(ax6)
    ax6.set_title("3D side view", fontsize=9)

    fig.suptitle("Chemical PCA Euclidean distance — HMDS embedding with Ward leaf-order colormap",
                 fontsize=13, y=1.01)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _shepard_subplot(ax, metrics):
    """Placeholder Shepard metrics display within a comparison grid subplot."""
    ax.axis("off")
    lines = [
        f"ρ = {metrics['distance_spearman']:.4f}",
        f"r = {metrics['distance_pearson']:.4f}",
        f"stress = {metrics['normalized_raw_stress']:.3f}",
        f"MAE = {metrics['mean_absolute_residual']:.4f}",
        f"BIC = {metrics['bic_global_variance']:.1f}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.5, 0.8 - i * 0.15, line, ha="center", va="center",
                fontsize=10, transform=ax.transAxes, fontfamily="monospace")
    ax.set_title("Fidelity metrics", fontsize=9)


# ── main ─────────────────────────────────────────────────────────────────


def main():
    dist_clean, aids, colors_hex = load_data()
    print(f"Loaded: {len(aids)} strains, distance matrix {dist_clean.shape}")

    # Normalize to max 2.0 (standard HMDS prep)
    dist_norm = normalize_to_max_two(dist_clean)
    n_params = dist_clean.shape[0] * 2  # for 2D

    # ── 2D HMDS ────────────────────────────────────────────────────────

    print("\n--- 2D HMDS ---")
    lorentz_2d, embedded_2d, lam_2d, _ = scipy_hyperbolic_mds(
        dist_norm, dim=2, starts=STARTS, maxiter=MAXITER, seed=SEED,
    )
    poincare_2d = recenter_poincare(lorentz_to_poincare(lorentz_2d))
    predicted_2d = embedded_2d / lam_2d
    metrics_2d = preservation_metrics(dist_norm, embedded_2d, predicted_2d, n_params=n_params)
    print(f"  ρ={metrics_2d['distance_spearman']:.3f}  stress={metrics_2d['normalized_raw_stress']:.3f}")

    plot_2d_poincare(
        poincare_2d, colors_hex, aids, metrics_2d,
        OUTPUT_DIR / "euc_hmds_2d_poincare.png",
    )
    plot_shepard(
        dist_norm, predicted_2d, metrics_2d,
        OUTPUT_DIR / "euc_hmds_2d_shepard.png",
    )

    # ── 3D HMDS ────────────────────────────────────────────────────────

    print("\n--- 3D HMDS ---")
    lorentz_3d, embedded_3d, lam_3d, _ = scipy_hyperbolic_mds(
        dist_norm, dim=3, starts=STARTS, maxiter=MAXITER, seed=SEED,
    )
    poincare_3d = recenter_poincare(lorentz_to_poincare(lorentz_3d))
    predicted_3d = embedded_3d / lam_3d
    n_params_3d = dist_clean.shape[0] * 3
    metrics_3d = preservation_metrics(dist_norm, embedded_3d, predicted_3d, n_params=n_params_3d)
    print(f"  ρ={metrics_3d['distance_spearman']:.3f}  stress={metrics_3d['normalized_raw_stress']:.3f}")

    plot_3d_poincare(
        poincare_3d, colors_hex, aids, metrics_3d,
        OUTPUT_DIR / "euc_hmds_3d_poincare_ball.png",
    )
    plot_shepard(
        dist_norm, predicted_3d, metrics_3d,
        OUTPUT_DIR / "euc_hmds_3d_shepard.png",
    )

    # ── Comparison grid ────────────────────────────────────────────────

    plot_all_views(
        poincare_2d, poincare_3d, colors_hex, aids,
        metrics_2d, metrics_3d,
        OUTPUT_DIR / "euc_hmds_comparison_grid.png",
    )

    # ── Save coordinates ────────────────────────────────────────────────

    pd.DataFrame({
        "aid": aids,
        "hmds2d_p1": poincare_2d[:, 0],
        "hmds2d_p2": poincare_2d[:, 1],
        "hmds3d_p1": poincare_3d[:, 0],
        "hmds3d_p2": poincare_3d[:, 1],
        "hmds3d_p3": poincare_3d[:, 2],
        "color_hex": colors_hex,
    }).to_csv(OUTPUT_DIR / "hmds_coordinates.csv", index=False)
    print(f"\nSaved: hmds_coordinates.csv")

    print(f"\nAll outputs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
