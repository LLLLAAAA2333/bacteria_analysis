"""Inspect response patterns of single neurons — overlay, hierarchical clustering, dendrogram.

Currently focused on ASH (L/R merged) across 86 stimuli from the 86bac dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe

NEURON = "AWCOFF"
K =1  # number of Ward clusters — check dendrogram to choose
OUTPUT_DIR = Path(f"results/neural_dimensionality/{NEURON}")
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# L/R merge map from neural.py — same merge pairs, same order
_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"),
    "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"),
    "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"),
    "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"),
    "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}
# Non-bilateral neurons keep their single-channel identity
_NON_BILATERAL = ("ASEL", "ASER", "AWCOFF", "AWCON")


def _neuron_raw_names(neuron: str) -> list[str]:
    """Return the raw column names that compose *neuron*.

    ``"ASH"`` → ``["ASHL", "ASHR"]``, ``"ADFL"`` → ``["ADFL"]``,
    ``"ASEL"`` → ``["ASEL"]``.
    """
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _prepare_neuron_prototypes(
    raw: pd.DataFrame, neuron: str
) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """Build stimulus × time median prototypes for a neuron.

    L/R pairs (e.g. ``"ASH"``) are merged by averaging across the two
    sides within each trial.  Raw names (e.g. ``"ADFL"``) and
    non-bilateral neurons (``"ASEL"``) are used unmerged.

    Returns ``(mat, timepoints, stim_info)``.
    """
    raw_names = _neuron_raw_names(neuron)
    subset = raw[raw["neuron"].isin(raw_names)].copy()
    subset["trial_id"] = (
        pd.to_datetime(subset["date"]).dt.strftime("%Y%m%d")
        + "__"
        + subset["worm_key"].astype(str)
        + "__"
        + subset["segment_index"].astype(str)
    )

    # groupby-mean handles both single-neuron (pass-through) and L/R merge
    trial_avg = (
        subset.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"]
        .mean()
        .reset_index()
    )
    proto = (
        trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"]
        .median()
        .reset_index()
    )

    mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")
    timepoints = mat.columns.astype(int).tolist()

    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    return mat, timepoints, stim_info


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

def plot_overlay(mat: pd.DataFrame, timepoints: list[int], *,
                 neuron: str = "ASH", output_dir: Path):
    """Raw and z-scored overlay of all stimulus trajectories for a neuron."""

    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, ylabel, title, color in [
        (axes[0], mat,      "ΔF/F₀",  f"{neuron} — dfof",       "#2c3e50"),
        (axes[1], mat_z,    "z-score", f"{neuron} — z-scored",       "#8e44ad"),
    ]:
        for stim in data.index:
            ax.plot(timepoints, data.loc[stim].values, alpha=0.35, lw=1, color=color)
        ax.axvspan(5, 15, alpha=0.12, color="#e74c3c", zorder=-1)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.4)
        ax.set_xlabel("Time(s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_dir / f"{neuron}_overlay.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# cluster overlay
# ---------------------------------------------------------------------------

def plot_cluster_overlays(
    mat: pd.DataFrame, *, neuron: str = "ASH", k: int = 2, output_dir: Path
):
    """Z-scored trajectory overlay + mean for each Ward cluster.

    One subplot per cluster.  Useful to inspect what temporal shape
    each cluster actually represents.
    """
    from scipy.cluster.hierarchy import fcluster

    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    d = _pearson_distance(mat)
    Z = linkage(squareform(d), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    timepoints = mat.columns.astype(int).tolist()

    n_cols = min(k, 3)
    n_rows = (k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if k == 1:
        axes = np.array([axes])
    axes = axes.flat

    for cl in range(1, k + 1):
        ax = axes[cl - 1]
        mask = labels == cl
        cluster_data = mat_z.values[mask]
        n_members = mask.sum()

        for row in cluster_data:
            ax.plot(timepoints, row, alpha=0.30, lw=0.7, color="#2c3e50")
        mean = cluster_data.mean(axis=0)
        ax.plot(timepoints, mean, color="#e74c3c", lw=2, label=f"mean (n={n_members})")
        ax.axvspan(5, 15, alpha=0.2, color="#e74c3c")
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.35)
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("z-score")
        ax.set_title(f"{neuron} — cluster {cl}  (n={n_members})")
        ax.legend(fontsize=8)

    # hide unused subplots
    for idx in range(k, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / f"{neuron}_cluster_overlays_k{k}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram(
    mat: pd.DataFrame, stim_info: pd.DataFrame, *,
    neuron: str = "ASH", fig_width: float = 12, output_dir: Path,
):
    """Hierarchical clustering dendrogram with AID leaf labels.

    Uses Pearson correlation distance (1−r) and Ward linkage.  Colours
    are suppressed so that the two-cluster split is immediately visible
    from the branch-gap alone.
    """

    # z-score (Pearson is invariant to this, but keeps data well-behaved)
    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)

    r_mat = np.corrcoef(mat_z.values)
    dist = np.clip(1 - r_mat, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2

    Z = linkage(squareform(dist), method="ward")

    aid_labels = [stim_info.loc[s, "aid"] for s in mat.index]

    fig, ax = plt.subplots(figsize=(fig_width, 5))
    dendrogram(
        Z,
        ax=ax,
        labels=aid_labels,
        leaf_font_size=6,
        color_threshold=0,
        above_threshold_color="#2c3e50",
        link_color_func=lambda k: "#2c3e50",
    )

    ax.set_title(f"{neuron} hierarchical clustering  (Pearson distance, Ward linkage)", fontsize=12)
    ax.set_ylabel("Ward merge cost")
    fig.tight_layout()
    fig.savefig(output_dir / f"{neuron}_dendrogram_aid.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# per-genus heatmaps
# ---------------------------------------------------------------------------

def plot_per_genus_heatmaps(mat: pd.DataFrame, stim_info: pd.DataFrame, *, neuron: str = "ASH"):
    """One heatmap per genus (rows: stimuli, columns: time)."""

    from matplotlib.colors import TwoSlopeNorm

    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=1.5)
    gen_dir = OUTPUT_DIR / f"{neuron}_by_genus"
    gen_dir.mkdir(parents=True, exist_ok=True)

    for genus, group in stim_info.groupby("genus"):
        members = group.index.tolist()
        n = len(members)
        data = mat.loc[members].values

        # intra-genus clustering
        if n >= 2:
            r = np.corrcoef(data)
            d = np.clip(1 - r, 0, None)
            np.fill_diagonal(d, 0)
            d = (d + d.T) / 2
            Z_ = linkage(squareform(d), method="ward")
            from scipy.cluster.hierarchy import leaves_list
            order = leaves_list(Z_)
            members = [members[i] for i in order]
            data = mat.loc[members].values

        row_labels = [stim_info.loc[s, "species"] for s in members]
        fig_h = max(3, n * 0.40)

        fig, axes = plt.subplots(1, 2, figsize=(12, fig_h),
                                 gridspec_kw={"width_ratios": [0.85, 0.15]})

        # heatmap
        ax = axes[0]
        ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm,
                  extent=[-0.5, mat.shape[1] - 0.5, n - 0.5, -0.5])
        ax.axvspan(4.5, 14.5, alpha=0.2, color="#e74c3c")
        ax.set_yticks(range(n))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Time(s)")
        ax.set_title(f"{genus} (n={n})")
        for i in range(1, n):
            ax.axhline(i - 0.5, color="white", lw=0.5)

        # mean ± SD
        ax2 = axes[1]
        tp = list(range(mat.shape[1]))
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax2.fill_betweenx(tp, mean - std, mean + std, alpha=0.2, color="#2c3e50")
        ax2.plot(mean, tp, color="#2c3e50", lw=2)
        ax2.axhline(0, color="black", lw=0.3, ls="--")
        ax2.axhspan(-0.5, 4.5, alpha=0.08, color="gray")
        ax2.axhspan(4.5, 14.5, alpha=0.05, color="#e74c3c")
        ax2.set_ylim(tp[-1] + 0.5, tp[0] - 0.5)
        ax2.set_xlabel("ΔF/F₀", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_title("mean ± SD", fontsize=9)

        fig.tight_layout()
        safe = genus.replace(" ", "_").replace("/", "_")
        fig.savefig(gen_dir / f"{safe}.png", dpi=150)
        plt.close(fig)

    # singletons panel
    singles = stim_info.groupby("genus").filter(lambda g: len(g) <= 1)
    if len(singles) > 0:
        members = singles.index.tolist()
        data = mat.loc[members].values
        row_labels = [
            f'{stim_info.loc[s, "genus"]}: {stim_info.loc[s, "species"]}'
            for s in members
        ]
        fig_h = max(2, len(members) * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm,
                  extent=[-0.5, mat.shape[1] - 0.5, len(members) - 0.5, -0.5])
        ax.axvspan(-0.5, 4.5, alpha=0.08, color="gray")
        ax.axvspan(4.5, 14.5, alpha=0.05, color="#e74c3c")
        ax.set_yticks(range(len(members)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xlabel("Time(s)")
        ax.set_title(f"Singletons (n={len(members)})")
        fig.tight_layout()
        fig.savefig(gen_dir / "_singletons.png", dpi=150)
        plt.close(fig)

    print(f"Per-genus heatmaps → {gen_dir}")


# ---------------------------------------------------------------------------
# 2D embedding (HMDS / Poincaré disk)
# ---------------------------------------------------------------------------

def _pearson_distance(mat: pd.DataFrame) -> np.ndarray:
    """Pearson correlation distance (1−r), forced symmetric."""
    r = np.corrcoef(mat.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    return (d + d.T) / 2


_CLUSTER_PALETTE = np.array([
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e",
])


def plot_hmds_embedding(
    mat: pd.DataFrame,
    stim_info: pd.DataFrame,
    *,
    neuron: str = "ASH",
    k: int = 2,
    output_dir: Path,
):
    """Embed stimuli on a Poincaré disk via hyperbolic MDS.

    Uses chord distance.  Colours by *k*-class Ward labels (plain
    ``cluster N`` — no biological meaning is assumed).  Use the
    dendrogram to pick an appropriate *k*.
    """
    from compare_86bac_chord_hmds import (
        chord_from_linear,
        normalize_to_max_two,
        scipy_hyperbolic_mds,
        lorentz_to_poincare,
        recenter_poincare,
        preservation_metrics,
    )
    from scipy.cluster.hierarchy import fcluster

    pearson = _pearson_distance(mat)
    chord = chord_from_linear(pearson)
    chord_norm = normalize_to_max_two(chord)

    coords_lorentz, embedded_hyp, lambda_val, _meta = scipy_hyperbolic_mds(
        chord_norm, dim=2, starts=8, maxiter=900, seed=42,
    )
    poincare = recenter_poincare(lorentz_to_poincare(coords_lorentz))

    # Ward k-class labels
    Z = linkage(squareform(_pearson_distance(mat)), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    colors = _CLUSTER_PALETTE[(labels - 1) % len(_CLUSTER_PALETTE)]

    predicted = embedded_hyp / lambda_val
    metrics = preservation_metrics(chord_norm, embedded_hyp, predicted,
                                   n_params=chord_norm.shape[0] * 2)

    fig, (ax_disk, ax_shep) = plt.subplots(1, 2, figsize=(15, 7))

    # -- Poincaré disk --
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax_disk.add_artist(circle)

    if k == 1:
        ax_disk.scatter(
            poincare[:, 0], poincare[:, 1],
            c="#4a5568", edgecolors="white", linewidths=0.5, s=60, alpha=0.88,
        )
    else:
        for cl in range(1, k + 1):
            mask = labels == cl
            ax_disk.scatter(
                poincare[mask, 0], poincare[mask, 1],
                c=colors[mask], s=60, alpha=0.88, edgecolors="white", linewidths=0.5,
                label=f"cluster {cl} (n={mask.sum()})",
            )
        ax_disk.legend(fontsize=9, loc="upper right")

    aids = [stim_info.loc[s, "aid"] for s in mat.index]
    for i, aid in enumerate(aids):
        ax_disk.annotate(aid, (poincare[i, 0], poincare[i, 1]),
                         fontsize=4.5, alpha=0.6, ha="center", va="bottom",
                         textcoords="offset points", xytext=(0, 3))

    ax_disk.set_xlim(-1.05, 1.05)
    ax_disk.set_ylim(-1.05, 1.05)
    ax_disk.set_aspect("equal")
    ax_disk.set_xlabel("Poincaré 1")
    ax_disk.set_ylabel("Poincaré 2")
    ax_disk.set_title(f"{neuron} — HMDS Poincaré disk")

    # -- Shepard diagram --
    from compare_86bac_chord_hmds import upper_triangle
    orig_pairs = upper_triangle(chord_norm)
    pred_pairs = upper_triangle(predicted)
    limit = float(max(orig_pairs.max(), pred_pairs.max()) * 1.04)

    ax_shep.scatter(orig_pairs, pred_pairs, s=8, color="#4a5568", alpha=0.20, linewidths=0)
    ax_shep.plot([0, limit], [0, limit], "--", color="#111827", lw=1)
    ax_shep.set_xlim(0, limit)
    ax_shep.set_ylim(0, limit)
    ax_shep.set_aspect("equal")
    ax_shep.set_xlabel("Input chord distance")
    ax_shep.set_ylabel("Embedded predicted distance")
    ax_shep.set_title(
        f"Shepard diagram\n"
        f"rho={metrics['distance_spearman']:.3f}, "
        f"r={metrics['distance_pearson']:.3f}, "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=11,
    )
    ax_shep.grid(True, color="#e5e7eb", lw=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / f"{neuron}_hmds_2d.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)


def main():
    raw = pd.read_parquet("data/106bac.parquet")
    # for neuron in ALL_NEURONS:
    neuron = NEURON
    output_dir = Path(f"results/neural_dimensionality/{neuron}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mat, timepoints, stim_info = _prepare_neuron_prototypes(raw, neuron=neuron)
    print(f"{neuron}: {mat.shape[0]} stimuli × {mat.shape[1]} timepoints")

    plot_overlay(mat, timepoints, neuron=neuron, output_dir=output_dir)
    plot_dendrogram(mat, stim_info, neuron=neuron, fig_width=10, output_dir=output_dir)
    plot_cluster_overlays(mat, neuron=neuron, k=K, output_dir=output_dir)
    plot_hmds_embedding(mat, stim_info, neuron=neuron, k=K, output_dir=output_dir)

    print(f"  → {output_dir}")


if __name__ == "__main__":
    main()
