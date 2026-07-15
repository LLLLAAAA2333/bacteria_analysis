"""Heatmap of Ward cluster labels across all 13 neurons.

Each cell shows the cluster ID (1/2/3) a stimulus belongs to for a given
neuron, based on z-scored time-course Ward clustering.  This gives a compact
"response phenotype" overview: which stimuli drive similar patterns and which
neurons share response structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

# k per neuron — determined by dendrogram inspection
K_MAP: dict[str, int] = {
    "ADF": 2, "ADL": 2, "ASEL": 2, "ASER": 1,
    "ASG": 1, "ASH": 2, "ASI": 2, "ASJ": 3,
    "ASK": 2, "AWA": 3, "AWB": 2, "AWCOFF": 1, "AWCON": 2,
}

# k=1 neurons excluded from heatmap (no cluster structure to show)
_K1_NEURONS = {n for n, k in K_MAP.items() if k == 1}

OUTPUT_DIR = Path("results/neural_dimensionality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}
_NON_BILATERAL = ("ASEL", "ASER", "AWCOFF", "AWCON")

# Only neurons with k > 1 (meaningful cluster structure)
NEURON_ORDER = [
    "ADF", "ADL", "ASEL", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCON",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _build_prototype(raw: pd.DataFrame, neuron: str) -> tuple[pd.DataFrame, list[int]]:
    raw_names = _neuron_raw_names(neuron)
    subset = raw[raw["neuron"].isin(raw_names)].copy()
    subset["trial_id"] = (
        pd.to_datetime(subset["date"]).dt.strftime("%Y%m%d")
        + "__"
        + subset["worm_key"].astype(str)
        + "__"
        + subset["segment_index"].astype(str)
    )
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
    return mat, timepoints


def _pearson_distance(mat: pd.DataFrame) -> np.ndarray:
    r = np.corrcoef(mat.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    return (d + d.T) / 2


def _ward_labels(mat: pd.DataFrame, k: int) -> pd.Series:
    """Return Ward cluster labels (1..k) for each stimulus (row of *mat*)."""
    if k <= 1:
        return pd.Series(1, index=mat.index, dtype=int)

    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    d = _pearson_distance(mat_z)
    Z = linkage(squareform(d), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    return pd.Series(labels, index=mat.index, dtype=int)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    raw = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    # ---- compute cluster labels for every neuron ----
    all_labels: dict[str, pd.Series] = {}
    all_stimuli: set[str] | None = None

    for neuron in K_MAP:
        mat, _ = _build_prototype(raw, neuron)
        k = K_MAP[neuron]
        labels = _ward_labels(mat, k)
        all_labels[neuron] = labels
        if all_stimuli is None:
            all_stimuli = set(labels.index)
        else:
            all_stimuli &= set(labels.index)

    common_stim = sorted(all_stimuli)
    print(f"Common stimuli across all neurons: {len(common_stim)}")

    # ---- build label matrix (stimulus × neuron) ----
    all_neurons = list(K_MAP.keys())
    label_matrix_full = pd.DataFrame(index=common_stim, columns=all_neurons, dtype=int)
    for neuron in all_neurons:
        label_matrix_full[neuron] = all_labels[neuron].reindex(common_stim)

    # ---- filter to non-k=1 neurons for heatmap ----
    label_matrix = label_matrix_full[NEURON_ORDER]

    # ---- sort rows by label similarity ----
    # Use the label matrix itself for row clustering (Hamming-like)
    row_order = _sort_by_label_pattern(label_matrix)

    # ---- prepare AID labels ----
    aid_labels = [str(stim_info.loc[s, "aid"]) for s in row_order]

    # ---- plot ----
    plot_label_heatmap(label_matrix.loc[row_order], aid_labels)

    # ---- print summary ----
    print("\nCluster label distribution per neuron:")
    for neuron in all_neurons:
        k = K_MAP[neuron]
        counts = label_matrix_full[neuron].value_counts().sort_index()
        parts = ", ".join(f"c{c}: {counts.get(c, 0)}" for c in range(1, k + 1))
        tag = "  [excluded]" if k == 1 else ""
        print(f"  {neuron} (k={k}): {parts}{tag}")

    print(f"\nDone → {OUTPUT_DIR}")


def _sort_by_label_pattern(label_matrix: pd.DataFrame) -> list[str]:
    """Sort stimuli by similarity of their cluster-label vectors.

    Uses a simple approach: treat labels as categorical, compute Hamming
    distance, then hierarchical clustering for row order.
    """
    n = len(label_matrix)
    if n <= 2:
        return label_matrix.index.tolist()

    # One-hot encode labels → treat as continuous for correlation distance
    encoded = pd.get_dummies(label_matrix.astype(str)).astype(float)
    r = np.corrcoef(encoded.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    Z = linkage(squareform(d), method="ward")
    order_idx = leaves_list(Z)
    return label_matrix.index[order_idx].tolist()


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def plot_label_heatmap(label_matrix: pd.DataFrame, aid_labels: list[str]):
    """Heatmap transposed: neurons as rows, stimuli as columns.

    Each cell is a cluster label (1/2/3), colored categorically.
    k=1 neurons are excluded (no structure to show).
    """

    neurons = label_matrix.columns.tolist()
    n_stim, n_neuron = label_matrix.shape

    # ---- transpose: neurons = rows, stimuli = columns ----
    data = label_matrix.values.T.astype(float)  # (n_neuron × n_stim)

    # ---- categorical colormap ----
    # 1 = cool blue, 2 = warm red, 3 = green
    cmap = matplotlib.colors.ListedColormap(["#3498db", "#e74c3c", "#2ecc71"])
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(20, n_stim * 0.22), max(5, n_neuron * 0.55)))

    ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
              extent=[-0.5, n_stim - 0.5, n_neuron - 0.5, -0.5])

    # ---- cell text ----
    for i in range(n_neuron):
        for j in range(n_stim):
            val = int(data[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=6, fontweight="bold",
                    color="white" if val in (1, 2) else "#1a5c2e")

    # ---- axes ----
    ax.set_yticks(range(n_neuron))
    ytick_labels = [f"{n}  (k={K_MAP[n]})" for n in neurons]
    ax.set_yticklabels(ytick_labels, fontsize=9)

    ax.set_xticks(range(n_stim))
    ax.set_xticklabels(aid_labels, fontsize=4.5, rotation=90)
    ax.xaxis.tick_top()

    ax.set_ylim(n_neuron - 0.5, -0.5)

    # ---- separator lines ----
    for i in range(1, n_neuron):
        ax.axhline(i - 0.5, color="white", lw=1.2)
    for j in range(1, n_stim):
        ax.axvline(j - 0.5, color="white", lw=0.3)

    # ---- legend ----
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="cluster 1"),
        Patch(facecolor="#e74c3c", label="cluster 2"),
        Patch(facecolor="#2ecc71", label="cluster 3"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower left",
              bbox_to_anchor=(1.01, 0), ncol=1, frameon=False,
              title="Cluster", title_fontsize=8)

    ax.set_title("Ward cluster labels: neuron (rows) × stimulus (columns)\n"
                 "(z-scored time-course, Pearson distance, Ward linkage)",
                 fontsize=11, pad=22)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_label_heatmap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUTPUT_DIR / 'cluster_label_heatmap.png'}")


if __name__ == "__main__":
    main()
