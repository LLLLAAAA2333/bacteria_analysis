"""Cross-neuron cluster co-occurrence analysis.

Explores whether the C. elegans nervous system exhibits a "joint discrete state
space" — combinations of (neuron, cluster_state) that frequently co-occur
across stimuli.

Produces 4 visualizations:
  1. Neuron×Neuron co-clustering heatmap (k>1 neurons only)
  2. Stimulus × Cluster-state matrix (all 13 neurons, per-neuron optimal k)
  3. Conditional transition coupling matrix (asymmetric, P(j=sb | i=sa))
  4. Cluster combination frequency (unique global states sorted by prevalence)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)

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

# Per-neuron optimal k (from manual inspection of dendrograms)
K_MAP: dict[str, int] = {
    "ASG": 1,
    "AWCOFF": 1,
    "ASER": 1,
    "AWA": 2,   # ← override: test k=2 instead of 3
}
# All others default to k=2

FILTERED_NEURONS = tuple(n for n in ALL_NEURONS if K_MAP.get(n, 2) > 1)
# 10 neurons: ADF, ADL, ASEL, ASH, ASI, ASJ, ASK, AWA, AWB, AWCON

OUTPUT_DIR = Path("results/correlation_analysis/awa_k2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Neuron identity palette (13 distinct colors)
NEURON_PALETTE = np.array([
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324",
])

# Discrete cluster label colours
CLUSTER_COLORS = {1: "#4E79A7", 2: "#E15759", 3: "#F28E2B"}


# ---------------------------------------------------------------------------
# shared data pipeline
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    """Return the raw column names for *neuron* (resolves L/R merge pairs)."""
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _build_prototype_matrix(
    raw: pd.DataFrame, neuron: str
) -> pd.DataFrame:
    """Build stimulus × timepoint median prototype matrix for a single neuron."""
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
    return proto.pivot(index="stimulus", columns="time_point",
                       values="delta_F_over_F0")


def _cluster_labels(
    mat: pd.DataFrame, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ward-cluster *mat* rows into *k* groups via Pearson distance.

    Returns ``(labels, Z)`` where *labels* are 1-indexed cluster IDs.
    """
    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    r = np.corrcoef(mat_z.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    Z = linkage(squareform(d), method="ward")
    labels = fcluster(Z, k, criterion="maxclust") if k > 1 else np.ones(mat.shape[0], dtype=int)
    return labels, Z


def _build_all_prototypes_and_labels(
    raw: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],       # mats: neuron → stimulus×timepoint DataFrame
    pd.DataFrame,                   # labels_k2: stimulus × neuron (all k=2)
    pd.DataFrame,                   # labels_kmapped: stimulus × neuron (per-neuron k)
    pd.DataFrame,                   # stim_info
]:
    """One-stop pipeline: prototype matrices + cluster labels for all 13 neurons."""
    mats: dict[str, pd.DataFrame] = {}
    for neuron in ALL_NEURONS:
        mats[neuron] = _build_prototype_matrix(raw, neuron)

    # common stimulus intersection
    common_stimuli = sorted(set.intersection(
        *[set(m.index) for m in mats.values()]
    ))
    n_stim = len(common_stimuli)
    print(f"Common stimuli: {n_stim}")

    # build label matrices
    labels_k2 = pd.DataFrame(index=common_stimuli, columns=list(ALL_NEURONS), dtype=int)
    labels_kmapped = pd.DataFrame(index=common_stimuli, columns=list(ALL_NEURONS), dtype=int)

    for neuron in ALL_NEURONS:
        mat = mats[neuron].loc[common_stimuli]
        k_opt = K_MAP.get(neuron, 2)
        lbl_k2, _ = _cluster_labels(mat, k=2)
        lbl_km, _ = _cluster_labels(mat, k=k_opt)
        labels_k2[neuron] = lbl_k2
        labels_kmapped[neuron] = lbl_km

    # stimulus metadata
    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    return mats, labels_k2, labels_kmapped, stim_info


# ---------------------------------------------------------------------------
# 1. Neuron × Neuron Co-clustering Heatmap  (k>1 only)
# ---------------------------------------------------------------------------

def plot_coclustering_heatmap(
    labels_k2: pd.DataFrame,
    output_dir: Path,
):
    """Symmetric co-clustering probability heatmaps for neurons with k>1."""
    filtered = labels_k2[list(FILTERED_NEURONS)]
    arr = filtered.values  # (n_stim, 10), values {1, 2}
    n_stim, n_neuron = arr.shape
    neuron_list = list(FILTERED_NEURONS)

    is_c1 = (arr == 1).astype(float)
    is_c2 = (arr == 2).astype(float)

    P_both_c1 = (is_c1.T @ is_c1) / n_stim   # P(i=c1 AND j=c1)
    P_same = (is_c1.T @ is_c1 + is_c2.T @ is_c2) / n_stim  # P(same label)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

    for ax, data, title, cmap in [
        (axes[0], P_both_c1, "P(both in cluster 1)", "YlOrRd"),
        (axes[1], P_same,    "P(same cluster label)",  "RdYlGn"),
    ]:
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(n_neuron):
            for j in range(n_neuron):
                val = data[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9 if i != j else 8,
                        color="white" if val > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

        ax.set_xticks(range(n_neuron))
        ax.set_xticklabels(neuron_list, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_neuron))
        ax.set_yticklabels(neuron_list, fontsize=9)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle("Neuron × Neuron co-clustering probabilities  (k>1 neurons, k=2 forced)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "coclustering_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → coclustering_heatmap.png")


# ---------------------------------------------------------------------------
# 2. Stimulus × Cluster-state Matrix
# ---------------------------------------------------------------------------

def plot_label_matrix_heatmap(
    labels_kmapped: pd.DataFrame,
    stim_info: pd.DataFrame,
    output_dir: Path,
):
    """Heatmap: rows=neurons, cols=stimuli, color=cluster label (per-neuron optimal k)."""
    # labels_kmapped: (n_stim, n_neuron), we transpose to (n_neuron, n_stim)
    data = labels_kmapped.T  # rows=neurons, cols=stimuli
    n_neuron, n_stim = data.shape

    # --- row ordering: cluster stimuli by their label vectors ---
    # use k=2 labels for consistent clustering (kmapped has varying scales)
    # one-hot → Pearson distance → Ward
    encoded = pd.get_dummies(labels_kmapped.astype(str)).astype(float)
    r = np.corrcoef(encoded.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    Z_stim = linkage(squareform(d), method="ward")
    stim_order = leaves_list(Z_stim)

    data_ordered = data.iloc[:, stim_order]
    stim_names_ordered = [labels_kmapped.index[i] for i in stim_order]
    aid_labels = [stim_info.loc[s, "aid"] for s in stim_names_ordered]

    # --- discrete colormap ---
    max_k = max(K_MAP.get(n, 2) for n in ALL_NEURONS)
    cluster_cmap = mcolors.ListedColormap(
        [CLUSTER_COLORS[i] for i in range(1, max_k + 1)]
    )
    bounds = np.arange(0.5, max_k + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, max_k)

    # --- draw ---
    fig_h = max(6, n_neuron * 0.55)
    fig_w = max(18, n_stim * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(data_ordered.values, aspect="auto", cmap=cluster_cmap, norm=norm)

    # neuron labels with k suffix
    y_labels = [f"{n} (k={K_MAP.get(n, 2)})" for n in data.index]
    ax.set_yticks(range(n_neuron))
    ax.set_yticklabels(y_labels, fontsize=9)

    # stimulus AID labels (every Nth)
    step = max(1, n_stim // 80)
    tick_pos = list(range(0, n_stim, step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([aid_labels[i] for i in tick_pos],
                       rotation=90, fontsize=5.5, ha="center", va="top")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # subtle separator lines between neurons
    for i in range(1, n_neuron):
        ax.axhline(i - 0.5, color="white", lw=1.2)

    ax.set_title(f"Stimulus × Cluster-state Matrix  ({n_neuron} neurons × {n_stim} stimuli)",
                 fontsize=12, pad=15)
    ax.set_xlabel("stimulus (AID)", fontsize=9)

    # colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cluster_cmap),
        ax=ax, shrink=0.3, aspect=20, pad=0.01,
    )
    cbar.set_ticks(range(1, max_k + 1))
    cbar.set_label("cluster label", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "stimulus_cluster_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → stimulus_cluster_matrix.png")


# ---------------------------------------------------------------------------
# 3. Conditional Transition Coupling Matrix  (asymmetric)
# ---------------------------------------------------------------------------

def plot_transition_coupling_matrix(
    labels_k2: pd.DataFrame,
    output_dir: Path,
):
    """Asymmetric heatmap: P(neuron_j=state_b | neuron_i=state_a)."""
    arr = labels_k2.values  # (n_stim, 13), values {1, 2}
    n_stim, n_neuron = arr.shape
    n_pairs = n_neuron * 2  # 26

    pair_labels: list[str] = []
    for n in ALL_NEURONS:
        pair_labels.append(f"{n}_c1")
        pair_labels.append(f"{n}_c2")

    cond_prob = np.zeros((n_pairs, n_pairs))

    for ni in range(n_neuron):
        for si, state_val in enumerate([1, 2]):
            row_idx = ni * 2 + si
            row_mask = arr[:, ni] == state_val
            n_cond = row_mask.sum()
            if n_cond == 0:
                continue
            for nj in range(n_neuron):
                for sj, col_val in enumerate([1, 2]):
                    col_idx = nj * 2 + sj
                    joint = (row_mask & (arr[:, nj] == col_val)).sum()
                    cond_prob[row_idx, col_idx] = joint / n_cond

    # verify: for each source (ni,si) and target neuron nj,
    # P(nj=c1|src) + P(nj=c2|src) should be 1.0
    row_ok = True
    for ni in range(n_neuron):
        for si in range(2):
            row_idx = ni * 2 + si
            if cond_prob[row_idx].sum() == 0:  # empty condition (k=1 neuron c2)
                continue
            for nj in range(n_neuron):
                pair_sum = cond_prob[row_idx, nj * 2] + cond_prob[row_idx, nj * 2 + 1]
                if abs(pair_sum - 1.0) > 1e-9:
                    print(f"  WARNING: row ({ALL_NEURONS[ni]}_c{si+1}), "
                          f"target {ALL_NEURONS[nj]}: sum={pair_sum:.6f}")
                    row_ok = False
    if row_ok:
        print("  All per-neuron conditionals sum to 1.0  OK")

    # --- draw ---
    fig, ax = plt.subplots(figsize=(17, 14))

    im = ax.imshow(cond_prob, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # annotate strong couplings only
    for i in range(n_pairs):
        for j in range(n_pairs):
            v = cond_prob[i, j]
            if v > 0.70 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color="#1a1a2e")

    # neuron block separators
    for k in range(1, n_neuron):
        ax.axhline(k * 2 - 0.5, color="white", lw=1.5)
        ax.axvline(k * 2 - 0.5, color="white", lw=1.5)

    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=6.5)
    ax.set_title(
        f"Conditional Transition Matrix  "
        f"P(col | row)  ({n_pairs}×{n_pairs}, k=2 forced)",
        fontsize=12,
    )
    ax.set_xlabel("target state  P(neuron_j=state_b | neuron_i=state_a)", fontsize=9)
    ax.set_ylabel("source state", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.72, label="conditional probability")
    fig.tight_layout()
    fig.savefig(output_dir / "transition_coupling_matrix.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  → transition_coupling_matrix.png")


# ---------------------------------------------------------------------------
# 4. Cluster Combination Frequency  (bipartite: combinations ↔ neurons)
# ---------------------------------------------------------------------------


def plot_combination_frequency(
    labels_kmapped: pd.DataFrame,
    output_dir: Path,
):
    """Bipartite view of discrete neural states (k>1 neurons only).

    Top    — frequency bars: one bar per unique cluster combination.
    Bottom — transposed combination matrix: rows=neurons, cols=combinations,
             colour = cluster label.
    """
    # --- filter to k>1 neurons ---
    filtered = labels_kmapped[list(FILTERED_NEURONS)]
    k_max = max(K_MAP.get(n, 2) for n in FILTERED_NEURONS)

    # --- build unique combinations and their frequencies ---
    label_strs = filtered.astype(str).agg("|".join, axis=1)
    combo_counts = label_strs.value_counts()
    n_combos = len(combo_counts)

    combo_labels_list = [
        list(map(int, s.split("|"))) for s in combo_counts.index
    ]
    combo_df = pd.DataFrame(
        combo_labels_list,
        columns=list(FILTERED_NEURONS),
        index=combo_counts.index,
    )
    combo_df.insert(0, "n_stimuli", combo_counts.values)

    print(f"\n  Unique cluster combinations (k>1): {n_combos}")
    print(f"  Most common: n={combo_counts.iloc[0]} "
          f"({combo_counts.iloc[0] / len(labels_kmapped) * 100:.1f}% of stimuli)")
    if n_combos > 1:
        top3 = combo_counts.head(3)
        print(f"  Top 3 frequencies: {dict(top3)}")

    # --- colormap ---
    cluster_colors = [CLUSTER_COLORS[i] for i in range(1, k_max + 1)]
    cluster_cmap = mcolors.ListedColormap(cluster_colors)
    bounds = np.arange(0.5, k_max + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, k_max)

    # --- draw: top=bars, bottom=matrix ---
    n_neurons = len(FILTERED_NEURONS)
    bar_height_ratio = 0.12
    fig_w = max(16, n_combos * 0.22)
    fig_h = max(7, n_neurons * 0.6 + 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        2, 1, height_ratios=[bar_height_ratio, 1 - bar_height_ratio],
        hspace=0.02, left=0.08, right=0.88, top=0.93, bottom=0.08,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    # --- top panel: frequency bars ---
    freqs = combo_df["n_stimuli"].values
    x_positions = np.arange(n_combos)
    bars = ax_bar.bar(x_positions, freqs, width=0.7, color="#2c3e50", alpha=0.85)
    for bar, val in zip(bars, freqs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(val), ha="center", fontsize=6.5, fontweight="bold",
                    color="#2c3e50")

    ax_bar.set_ylabel("n", fontsize=8, rotation=0, labelpad=12)
    ax_bar.set_title(f"Cluster Combination Frequency  ({n_combos} unique patterns)",
                     fontsize=11, loc="left", pad=6)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_xlim(-0.6, n_combos - 0.4)

    # --- bottom panel: transposed matrix (neurons × combinations) ---
    mat_data = combo_df.drop(columns="n_stimuli").values.T  # (n_neurons, n_combos)
    ax_mat.imshow(mat_data, aspect="auto", cmap=cluster_cmap, norm=norm)

    neuron_labels = [f"{n} (k={K_MAP.get(n, 2)})" for n in FILTERED_NEURONS]
    ax_mat.set_yticks(range(n_neurons))
    ax_mat.set_yticklabels(neuron_labels, fontsize=8)
    ax_mat.set_ylabel("neuron", fontsize=9)

    ax_mat.set_xticks([])
    ax_mat.set_xlabel(f"{n_combos} unique cluster combinations  →", fontsize=9)

    # horizontal separators between neurons
    for y in range(1, n_neurons):
        ax_mat.axhline(y - 0.5, color="white", lw=0.8)

    # vertical separators between combinations (thin)
    for x in range(1, n_combos):
        ax_mat.axvline(x - 0.5, color="white", lw=0.2, alpha=0.4)

    # --- colorbar ---
    cbar_ax = fig.add_axes([0.90, 0.10, 0.012, 0.14])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cluster_cmap),
        cax=cbar_ax,
    )
    cbar.set_ticks(range(1, k_max + 1))
    cbar.set_ticklabels([f"c{i}" for i in range(1, k_max + 1)])

    fig.savefig(output_dir / "combination_frequency.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  → combination_frequency.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading 106bac data...")
    raw = pd.read_parquet("data/106bac.parquet")

    print("Building prototypes and cluster labels for all 13 neurons...")
    mats, labels_k2, labels_kmapped, stim_info = \
        _build_all_prototypes_and_labels(raw)

    # --- label distribution summary ---
    print("\nCluster label distributions:")
    for neuron in ALL_NEURONS:
        k_opt = K_MAP.get(neuron, 2)
        counts = labels_kmapped[neuron].value_counts().sort_index()
        count_str = ", ".join(f"c{k}: {counts.get(k, 0)}" for k in range(1, k_opt + 1))
        print(f"  {neuron:8s} (k={k_opt}): {count_str}")

    output_dir = OUTPUT_DIR

    print("\n[1/4] Neuron × Neuron Co-clustering Heatmap (k>1 only)")
    plot_coclustering_heatmap(labels_k2, output_dir)

    print("\n[2/4] Stimulus × Cluster-state Matrix")
    plot_label_matrix_heatmap(labels_kmapped, stim_info, output_dir)

    print("\n[3/4] Conditional Transition Coupling Matrix")
    plot_transition_coupling_matrix(labels_k2, output_dir)

    print("\n[4/4] Cluster Combination Frequency")
    plot_combination_frequency(labels_kmapped, output_dir)

    print(f"\nDone → {output_dir}")


if __name__ == "__main__":
    main()
