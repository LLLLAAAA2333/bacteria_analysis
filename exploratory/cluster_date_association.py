"""Check whether Ward-cluster membership per neuron is associated with sample date.

Key question: after Ward hierarchical clustering of stimulus response patterns,
do the resulting clusters segregate by measurement date?

For each neuron type we:
1. Build stimulus × time prototype matrix (median across trials)
2. Run Ward hierarchical clustering (Pearson distance) at k=2 (k=3 for AWA)
3. Permutation test: shuffle cluster labels 10k times, compute MI as test statistic
4. Also bin dates into early/mid/late periods for a cleaner 2×3 view
5. Visualise cluster × date distributions
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}

ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)

K_MAP = {n: 2 for n in ALL_NEURONS}
K_MAP["AWA"] = 3
K_MAP["ASG"] = 1       # ≈single class per prior inspection
K_MAP["AWCOFF"] = 1    # ≈single class per prior inspection
K_MAP["ASER"] = 1      # ≈single class per prior inspection

N_PERM = 10_000
SEED = 42

OUTPUT_DIR = Path("results/cluster_date_association")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _build_prototypes_and_dates(
    raw: pd.DataFrame, neuron: str,
) -> tuple[pd.DataFrame, pd.Series]:
    raw_names = _neuron_raw_names(neuron)
    subset = raw[raw["neuron"].isin(raw_names)].copy()
    subset["trial_id"] = (
        subset["date"].astype(str) + "__"
        + subset["worm_key"].astype(str) + "__"
        + subset["segment_index"].astype(str)
    )
    trial_avg = (
        subset.groupby(["trial_id", "stimulus", "date", "time_point"])["delta_F_over_F0"]
        .mean().reset_index()
    )
    proto = (
        trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"]
        .median().reset_index()
    )
    mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")
    date_series = (
        trial_avg.groupby("stimulus")["date"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iat[0])
    ).reindex(mat.index)
    return mat, date_series


def _pearson_distance(mat: pd.DataFrame) -> np.ndarray:
    r = np.corrcoef(mat.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    return (d + d.T) / 2


def _permutation_test(labels: np.ndarray, dates: np.ndarray,
                      n_perm: int = N_PERM, seed: int = SEED) -> tuple[float, float, np.ndarray]:
    """Permutation test: MI(cluster_labels, dates) against null distribution."""
    observed_mi = mutual_info_score(dates, labels)
    rng = np.random.default_rng(seed)
    null_mi = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(labels)
        null_mi[i] = mutual_info_score(dates, shuffled)
    p_value = (np.sum(null_mi >= observed_mi) + 1) / (n_perm + 1)
    return observed_mi, p_value, null_mi


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyse_one_neuron(raw: pd.DataFrame, neuron: str, k: int) -> dict:
    mat, dates = _build_prototypes_and_dates(raw, neuron)
    if mat.shape[0] < k:
        return {"neuron": neuron, "k": k, "n_stimuli": mat.shape[0],
                "p_perm": None, "mi": None, "contingency": None,
                "clusters": None, "dates": None}

    dist = _pearson_distance(mat)
    Z = linkage(squareform(dist), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")

    # Permutation test with MI
    date_codes = pd.Categorical(dates).codes
    mi, p_perm, null_mi = _permutation_test(labels, date_codes)

    # Also bin dates into 3 periods: early, mid, late
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    # Split into thirds
    cut1 = unique_dates[n_dates // 3]
    cut2 = unique_dates[2 * n_dates // 3]
    def _bin_date(d):
        if d <= cut1:
            return f"early\n(≤{cut1})"
        elif d <= cut2:
            return f"mid\n({cut1}–{cut2})"
        else:
            return f"late\n(>{cut2})"
    date_bins = dates.apply(_bin_date)

    table_full = pd.crosstab(labels, dates)
    table_binned = pd.crosstab(labels, date_bins)

    # MI with binned dates
    bin_codes = pd.Categorical(date_bins).codes
    mi_binned, p_binned, _ = _permutation_test(labels, bin_codes)

    return {
        "neuron": neuron, "k": k, "n_stimuli": mat.shape[0],
        "mi": mi, "p_perm": p_perm,
        "mi_binned": mi_binned, "p_binned": p_binned,
        "contingency_full": table_full,
        "contingency_binned": table_binned,
        "clusters": pd.Series(labels, index=mat.index),
        "dates": dates, "date_bins": date_bins,
        "null_mi": null_mi,
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_all(results: list[dict]):
    valid = [r for r in results if r["clusters"] is not None]
    neuron_order = [n for n in ALL_NEURONS if n in [r["neuron"] for r in valid]]

    # --- Figure 1: binned contingency heatmaps ---
    n = len(valid)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2),
                             squeeze=False)

    for idx, r in enumerate(valid):
        ax = axes[idx // n_cols, idx % n_cols]
        table = r["contingency_binned"]
        # Row-normalize: proportion of each cluster in each date bin
        table_norm = table.div(table.sum(axis=1), axis=0)
        im = ax.imshow(table_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = table.values[i, j]
                norm_val = table_norm.values[i, j]
                ax.text(j, i, str(val), ha="center", va="center", fontsize=9,
                        color="white" if norm_val > 0.5 else "black",
                        fontweight="bold")

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, fontsize=7)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"cl {cl}" for cl in table.index], fontsize=8)
        p_str = f"p={r['p_binned']:.4f}"
        ax.set_title(f"{r['neuron']}  {p_str}", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Cluster × Date-period contingency (row-normalised)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "binned_contingency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: null distributions + observed MI ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2),
                             squeeze=False)

    for idx, r in enumerate(valid):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.hist(r["null_mi"], bins=50, color="#b0bec5", edgecolor="#78909c", alpha=0.8)
        ax.axvline(r["mi"], color="#e74c3c", lw=2, ls="--",
                   label=f"obs MI={r['mi']:.3f}\np={r['p_perm']:.4f}")
        ax.set_xlabel("MI")
        ax.set_ylabel("freq")
        ax.set_title(r["neuron"], fontsize=10)
        ax.legend(fontsize=7)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Permutation null: MI(cluster, date)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "permutation_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: per-neuron timeline swarm ---
    # For each neuron, show cluster × date as points
    rows_data = []
    for r in valid:
        for stim in r["clusters"].index:
            rows_data.append({
                "neuron": r["neuron"],
                "date": r["dates"][stim],
                "cluster": r["clusters"][stim],
            })
    df = pd.DataFrame(rows_data)
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    all_dates_sorted = sorted(df["date_dt"].unique())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2),
                             squeeze=False)

    for idx, r in enumerate(valid):
        ax = axes[idx // n_cols, idx % n_cols]
        neuron = r["neuron"]
        sub = df[df["neuron"] == neuron]

        for cl in sorted(sub["cluster"].unique()):
            cl_sub = sub[sub["cluster"] == cl]
            # Count per date and plot as line
            date_counts = cl_sub.groupby("date_dt").size().reindex(all_dates_sorted, fill_value=0)
            ax.plot(all_dates_sorted, date_counts.values, 'o-', lw=1.5, ms=5,
                    label=f"cl {cl} (n={len(cl_sub)})", color=f"C{int(cl)}")

        ax.set_title(neuron, fontsize=10, fontweight="bold")
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        if idx % n_cols == 0:
            ax.set_ylabel("count", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Stimulus count per cluster over measurement dates", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "timeline_by_cluster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: summary bar chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    neuron_labels = [r["neuron"] for r in valid]
    p_values = [r["p_binned"] for r in valid]
    mi_values = [r["mi"] for r in valid]
    colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in p_values]

    y_pos = range(len(neuron_labels))
    ax1.barh(y_pos, [-np.log10(max(p, 1e-10)) for p in p_values],
             color=colors, edgecolor="white", height=0.7)
    ax1.axvline(-np.log10(0.05), color="#e74c3c", ls="--", lw=1, alpha=0.5,
                label="α=0.05")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(neuron_labels)
    ax1.set_xlabel("-log10(p)")
    ax1.set_title("Date association significance (binned dates)")
    ax1.legend(fontsize=8)
    ax1.invert_yaxis()

    ax2.barh(y_pos, mi_values, color=colors, edgecolor="white", height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(neuron_labels)
    ax2.set_xlabel("Mutual Information (bits)")
    ax2.set_title("MI(cluster, date)")
    ax2.invert_yaxis()

    fig.suptitle("Cluster–date association summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    raw = pd.read_parquet("data/106bac.parquet")

    unique_dates = sorted(raw["date"].unique())
    print(f"{len(unique_dates)} unique dates: {unique_dates}")
    print(f"{raw['stimulus'].nunique()} unique stimuli\n")

    results = []
    for neuron in ALL_NEURONS:
        k = K_MAP[neuron]
        r = analyse_one_neuron(raw, neuron, k)
        results.append(r)

        p_str = f"p_perm={r['p_perm']:.4f}" if r['p_perm'] is not None else "p_perm=N/A"
        p_bin_str = f"p_binned={r['p_binned']:.4f}" if r['p_binned'] is not None else ""
        mi_str = f"MI={r['mi']:.3f}" if r['mi'] is not None else ""
        print(f"{r['neuron']:8s}  k={k}  n={r['n_stimuli']:3d}  {mi_str}  {p_str}  {p_bin_str}")
        if r["contingency_binned"] is not None:
            print(r["contingency_binned"].to_string(), "\n")

    print("--- Plotting ---")
    plot_all(results)

    # Save CSV
    rows = []
    for r in results:
        if r["clusters"] is None:
            continue
        for stim in r["clusters"].index:
            rows.append({
                "neuron": r["neuron"], "stimulus": stim,
                "date": r["dates"][stim], "date_bin": r["date_bins"][stim],
                "cluster": r["clusters"][stim],
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cluster_date_assignments.csv", index=False)
    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
