"""Single-neuron chord HMDS (2D + 3D) for ASH, AWA, AWCON — chemical colormap.

Matches inspect_neural_dimensionality.py's per-neuron Pearson distance → chord → HMDS,
with coloring from chemical PCA Ward leaf-order colormap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

OUTPUT_DIR = Path("results/chemical_pca_ward/single_neuron_hmds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEURONS = ["ASH", "AWA", "AWCON"]
STARTS = 8
MAXITER = 900
SEED = 42

_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}
_NON_BILATERAL = ("ASEL", "ASER", "AWCOFF", "AWCON")


def _neuron_raw_names(neuron: str) -> list[str]:
    merged = _LR_MERGE.get(neuron)
    return list(merged) if merged is not None else [neuron]


def build_neuron_prototypes(raw, neuron: str):
    """Stimulus × time median prototypes for a single neuron (L/R merged if applicable)."""
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
    mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")
    return mat


def pearson_distance(mat: pd.DataFrame) -> np.ndarray:
    """Pearson correlation distance (1−r), symmetric, diagonal=0."""
    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    r = np.corrcoef(mat_z.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    return (d + d.T) / 2


def add_poincare_wireframe(axis) -> None:
    u, v = np.mgrid[0:2 * np.pi:32j, 0:np.pi:16j]
    x = np.cos(u) * np.sin(v); y = np.sin(u) * np.sin(v); z = np.cos(v)
    axis.plot_wireframe(x, y, z, color="#4b5563", linewidth=0.85, alpha=0.40,
                        rstride=3, cstride=3)


def set_poincare_3d_axis(axis) -> None:
    axis.set_xlim(-1.02, 1.02); axis.set_ylim(-1.02, 1.02); axis.set_zlim(-1.02, 1.02)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.set_zlabel("Poincare 3", fontsize=9)
    axis.tick_params(labelsize=8)


def plot_3d_ball(poincare, colors_hex, aids, neuron, metrics, output_path):
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    add_poincare_wireframe(ax)
    ax.scatter(poincare[:, 0], poincare[:, 1], poincare[:, 2],
               c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
    for i, aid in enumerate(aids[:20]):
        ax.text(poincare[i, 0], poincare[i, 1], poincare[i, 2],
                aid, fontsize=4, alpha=0.5, ha="center")
    set_poincare_3d_axis(ax)
    ax.set_title(
        f"{neuron} chord HMDS 3D  —  chemical colormap\n"
        f"ρ={metrics['distance_spearman']:.3f}  stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2d_disk(poincare, colors_hex, aids, neuron, metrics, output_path):
    fig, ax = plt.subplots(figsize=(8, 7.5), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax.add_artist(circle)
    ax.scatter(poincare[:, 0], poincare[:, 1], c=colors_hex, s=52, alpha=0.90,
               edgecolor="white", linewidth=0.45)
    for i, aid in enumerate(aids):
        ax.annotate(aid, (poincare[i, 0], poincare[i, 1]),
                    fontsize=4, alpha=0.5, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3))
    ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Poincare 1", fontsize=9); ax.set_ylabel("Poincare 2", fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.set_title(
        f"{neuron} chord HMDS 2D  —  chemical colormap\n"
        f"ρ={metrics['distance_spearman']:.3f}  stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_shepard(dist_input, dist_predicted, metrics, neuron, dim, output_path):
    orig = upper_triangle(dist_input)
    pred = upper_triangle(dist_predicted)
    limit = float(max(orig.max(), pred.max()) * 1.04)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.scatter(orig, pred, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input chord distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"{neuron} {dim}D Shepard  ρ={metrics['distance_spearman']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}", fontsize=9)
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────

def main():
    # Load colormap
    colormap = pd.read_csv("results/chemical_pca_ward/aid_colormap.csv")
    color_by_aid = dict(zip(colormap["aid"], colormap["color_hex"]))
    print(f"Colormap: {len(color_by_aid)} AIDs")

    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    summary_rows = []

    for neuron in NEURONS:
        print(f"\n{'='*60}")
        print(f"  {neuron}")
        print(f"{'='*60}")

        # Build prototypes & filter to strains with chemical colormap
        mat = build_neuron_prototypes(raw, neuron)
        mat.index = mat.index.map(
            lambda s: stim_info.loc[s, "aid"] if s in stim_info.index else s
        )
        mat = mat.loc[[a for a in mat.index if a in color_by_aid]]
        aids = mat.index.tolist()
        print(f"  {len(aids)} strains × {mat.shape[1]} timepoints")

        # Pearson distance → chord → normalize
        pearson_d = pearson_distance(mat)
        chord_d = chord_from_linear(pearson_d)
        chord_norm = normalize_to_max_two(chord_d)

        colors_hex = [color_by_aid[a] for a in aids]
        n = len(aids)

        all_metrics = {}

        for dim in [2, 3]:
            print(f"  {dim}D HMDS...", end=" ", flush=True)
            lorentz, embedded, lam, _ = scipy_hyperbolic_mds(
                chord_norm, dim=dim, starts=STARTS, maxiter=MAXITER, seed=SEED,
            )
            poincare = recenter_poincare(lorentz_to_poincare(lorentz))
            predicted = embedded / lam
            metrics = preservation_metrics(chord_norm, embedded, predicted, n_params=n * dim)
            print(f"ρ={metrics['distance_spearman']:.3f}  stress={metrics['normalized_raw_stress']:.3f}")
            all_metrics[dim] = metrics

            prefix = OUTPUT_DIR / f"{neuron}_chord_hmds_{dim}d"

            if dim == 2:
                plot_2d_disk(poincare, colors_hex, aids, neuron, metrics, prefix.with_suffix(".png"))
            else:
                plot_3d_ball(poincare, colors_hex, aids, neuron, metrics, prefix.with_suffix(".png"))

            plot_shepard(chord_norm, predicted, metrics, neuron, dim,
                        OUTPUT_DIR / f"{neuron}_chord_hmds_{dim}d_shepard.png")

        summary_rows.append({
            "neuron": neuron,
            "n_strains": n,
            "n_timepoints": int(mat.shape[1]),
            "hmds_2d_rho": all_metrics[2]["distance_spearman"],
            "hmds_2d_stress": all_metrics[2]["normalized_raw_stress"],
            "hmds_3d_rho": all_metrics[3]["distance_spearman"],
            "hmds_3d_stress": all_metrics[3]["normalized_raw_stress"],
        })

    # ── Comparison figure: 2×3 grid (3 neurons × 2 views each) ────────
    # Regenerate all 3D coordinates for shared plotting
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    for col_idx, neuron in enumerate(NEURONS):
        mat = build_neuron_prototypes(raw, neuron)
        mat.index = mat.index.map(
            lambda s: stim_info.loc[s, "aid"] if s in stim_info.index else s
        )
        mat = mat.loc[[a for a in mat.index if a in color_by_aid]]
        aids = mat.index.tolist()
        pearson_d = pearson_distance(mat)
        chord_d = chord_from_linear(pearson_d)
        chord_norm = normalize_to_max_two(chord_d)
        colors_hex = [color_by_aid[a] for a in aids]

        lorentz_3d, embedded_3d, lam_3d, _ = scipy_hyperbolic_mds(
            chord_norm, dim=3, starts=STARTS, maxiter=MAXITER, seed=SEED,
        )
        poincare_3d = recenter_poincare(lorentz_to_poincare(lorentz_3d))

        # Row 1: 3D view
        ax1 = fig.add_subplot(2, 3, col_idx + 1, projection="3d")
        add_poincare_wireframe(ax1)
        ax1.scatter(poincare_3d[:, 0], poincare_3d[:, 1], poincare_3d[:, 2],
                    c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
        set_poincare_3d_axis(ax1)
        predicted_3d = embedded_3d / lam_3d
        m = preservation_metrics(chord_norm, embedded_3d, predicted_3d, n_params=len(aids)*3)
        ax1.set_title(f"{neuron} 3D  ρ={m['distance_spearman']:.3f}  stress={m['normalized_raw_stress']:.3f}",
                      fontsize=10)

        # Row 2: top-down view
        ax2 = fig.add_subplot(2, 3, col_idx + 4, projection="3d")
        add_poincare_wireframe(ax2)
        ax2.scatter(poincare_3d[:, 0], poincare_3d[:, 1], poincare_3d[:, 2],
                    c=colors_hex, s=48, alpha=0.92, edgecolor="white", linewidth=0.45)
        ax2.view_init(elev=90, azim=0)
        set_poincare_3d_axis(ax2)
        ax2.set_title(f"{neuron} top-down", fontsize=10)

    fig.suptitle("Single-neuron chord HMDS 3D  —  chemical Ward colormap",
                 fontsize=13, y=1.01)
    fig.savefig(OUTPUT_DIR / "all_neurons_hmds_3d_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Print summary ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for _, row in summary_df.iterrows():
        print(f"  {row['neuron']:8s}  n={int(row['n_strains']):3d}  "
              f"2D ρ={row['hmds_2d_rho']:.3f} s={row['hmds_2d_stress']:.3f}  "
              f"3D ρ={row['hmds_3d_rho']:.3f} s={row['hmds_3d_stress']:.3f}")
    summary_df.to_csv(OUTPUT_DIR / "hmds_summary.csv", index=False)
    print(f"\nSaved → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
