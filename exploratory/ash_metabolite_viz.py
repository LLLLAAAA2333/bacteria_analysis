"""Neuron response patterns vs metabolites: box plots grouped by Ward cluster labels.

Uses the same Ward clustering as inspect_neural_dimensionality.py — labels come
from z-scored time-course Pearson distance + Ward linkage, cut at k per the
user's dendrogram inspection.

log2(fold change), no z-score.  "Crosses zero" bonus: one cluster > 0,
another < 0 (relative to reference).

One clean figure: panel of top metabolite box plots.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

NEURON = "AWA"
K = 3
MET_FILE = "results/metabolite_modules/metabolites_reduced.parquet"
N_TOP = 20
N_PERM = 10_000

OUTPUT_DIR = Path(f"results/{NEURON}_metabolite_viz_filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def _build_prototype(raw, neuron):
    raw_names = _neuron_raw_names(neuron)
    subset = raw[raw["neuron"].isin(raw_names)].copy()
    subset["trial_id"] = (
        pd.to_datetime(subset["date"]).dt.strftime("%Y%m%d")
        + "__" + subset["worm_key"].astype(str)
        + "__" + subset["segment_index"].astype(str)
    )
    trial_avg = subset.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"].mean().reset_index()
    proto = trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"].median().reset_index()
    mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")
    return mat


def _ward_labels(mat: pd.DataFrame, k: int) -> pd.Series:
    """Ward cluster labels (1..k), same method as inspect_neural_dimensionality."""
    if k <= 1:
        return pd.Series(1, index=mat.index, dtype=int)
    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    r = np.corrcoef(mat_z.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    Z = linkage(squareform(d), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")
    return pd.Series(labels, index=mat.index, dtype=int)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # ---- 1. Neuron proto + Ward labels ----
    raw_neural = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw_neural)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    mat = _build_prototype(raw_neural, NEURON)
    labels = _ward_labels(mat, K)
    stim_to_aid = {s: stim_info.loc[s, "aid"] for s in mat.index}

    # ---- 2. Cluster summary ----
    cluster_aids: dict[int, set[str]] = {}
    for cl in range(1, K + 1):
        stimuli = labels[labels == cl].index
        aids = {stim_to_aid[s] for s in stimuli}
        cluster_aids[cl] = aids
        print(f"  Cluster {cl}: {len(aids)} stimuli")

    # ---- 3. Load metabolites (log2 FC, no z-score) ----
    if MET_FILE.endswith(".parquet"):
        met_df = pd.read_parquet(MET_FILE)
    else:
        met_df = pd.read_excel(MET_FILE)
        id_col = met_df.columns[0]
        met_df = met_df.rename(columns={id_col: "aid"})
        met_df["aid"] = met_df["aid"].astype(str)
        met_df = met_df.set_index("aid")
    met_cols = list(met_df.columns)
    met_log2 = np.log2(met_df.values.astype(float))
    aid_to_idx = {aid: i for i, aid in enumerate(met_df.index)}

    # ---- 4. Per-cluster data ----
    cluster_data: dict[int, np.ndarray] = {}
    cluster_n: dict[int, int] = {}
    for cl in range(1, K + 1):
        idx = [aid_to_idx[a] for a in cluster_aids[cl] if a in aid_to_idx]
        cluster_data[cl] = met_log2[idx]
        cluster_n[cl] = len(idx)

    # ---- 5. Select best cluster pair (largest absolute separation) ----
    # For k>2: find the pair of clusters with largest mean difference
    best_pair = None
    best_diff = -1
    for c1 in range(1, K + 1):
        for c2 in range(c1 + 1, K + 1):
            d1 = cluster_data[c1]
            d2 = cluster_data[c2]
            diff = np.abs(d1.mean(axis=0) - d2.mean(axis=0)).max()
            if diff > best_diff:
                best_diff = diff
                best_pair = (c1, c2)

    c_a, c_b = best_pair
    print(f"\nComparing cluster {c_a} (n={cluster_n[c_a]}) vs cluster {c_b} (n={cluster_n[c_b]})")

    data_a = cluster_data[c_a]
    data_b = cluster_data[c_b]

    # ---- 6. Observed SNR + permutation test ----
    n_mets = len(met_cols)
    merged = np.vstack([data_a, data_b])           # (n_total, n_mets)
    n_total = merged.shape[0]
    n_a = data_a.shape[0]
    true_labels = np.array([0] * n_a + [1] * (n_total - n_a))

    # Observed statistics
    mask_a = true_labels == 0
    mask_b = true_labels == 1
    obs_med_a = np.median(merged[mask_a], axis=0)
    obs_med_b = np.median(merged[mask_b], axis=0)
    obs_std_a = np.std(merged[mask_a], axis=0)
    obs_std_b = np.std(merged[mask_b], axis=0)
    obs_snr = np.abs(obs_med_a - obs_med_b) / (obs_std_a + obs_std_b + 1e-10)
    obs_cross = (obs_med_a > 0) & (obs_med_b < 0) | (obs_med_a < 0) & (obs_med_b > 0)

    # Permutation: shuffle labels, recompute SNR
    perm_snr_max = np.zeros((N_PERM, n_mets))
    rng = np.random.default_rng(42)

    print(f"  Running {N_PERM} permutations...", end="", flush=True)
    for p in range(N_PERM):
        shuf = rng.permutation(true_labels)
        m0 = shuf == 0
        m1 = shuf == 1
        med0 = np.median(merged[m0], axis=0)
        med1 = np.median(merged[m1], axis=0)
        std0 = np.std(merged[m0], axis=0)
        std1 = np.std(merged[m1], axis=0)
        snr_perm = np.abs(med0 - med1) / (std0 + std1 + 1e-10)
        perm_snr_max[p] = snr_perm
    print(" done.")

    # Permutation p-value: fraction of permuted SNR >= observed SNR
    perm_p = np.mean(perm_snr_max >= obs_snr[np.newaxis, :], axis=0)

    # ---- 7. Build results ----
    results = []
    for j, col in enumerate(met_cols):
        results.append({
            "metabolite": col,
            f"median_c{c_a}": obs_med_a[j],
            f"median_c{c_b}": obs_med_b[j],
            f"std_c{c_a}": obs_std_a[j],
            f"std_c{c_b}": obs_std_b[j],
            "snr": obs_snr[j],
            "abs_diff": abs(obs_med_a[j] - obs_med_b[j]),
            "p_perm": perm_p[j],
            "crosses_zero": obs_cross[j],
        })

    res_df = pd.DataFrame(results)
    res_df["score"] = res_df["crosses_zero"].astype(float) * 100 + res_df["snr"]
    res_df = res_df.sort_values("score", ascending=False)

    # Filter to permutation-significant only
    res_sig = res_df[res_df["p_perm"] <= 0.05]
    n_show = min(N_TOP, len(res_sig))
    print(f"Significant (permutation p <= 0.05): {len(res_sig)}/{len(res_df)}, showing top {n_show}")

    # ---- 8. Print ----
    c_a_str, c_b_str = f"c{c_a}", f"c{c_b}"
    print(f"\nTop metabolites by SNR (permutation test, {N_PERM} shuffles):")
    hdr = f"  {'metabolite':45s}  {c_a_str:>7s}  {c_b_str:>7s}  {'SNR':>6s}  {'p_perm':>8s}"
    print(hdr)
    for _, row in res_sig.head(n_show).iterrows():
        name = row["metabolite"][:43]
        print(f"  {name:45s}  {row[f'median_c{c_a}']:7.2f}  "
              f"{row[f'median_c{c_b}']:7.2f}  {row['snr']:6.3f}  {row['p_perm']:8.4f}")

    # Show all clusters' medians for top hits (useful when K > 2)
    if K > 2 and n_show > 0:
        print(f"\n  All-cluster medians for top {n_show}:")
        cl_headers = "".join(f"  {'c' + str(cl):>8s}" for cl in range(1, K + 1))
        print(f"  {'metabolite':45s}{cl_headers}")
        for _, row in res_sig.head(n_show).iterrows():
            name = row["metabolite"][:43]
            meds = []
            for cl in range(1, K + 1):
                j = met_cols.index(row["metabolite"])
                meds.append(np.median(cluster_data[cl][:, j]))
            med_str = "".join(f"  {m:8.2f}" for m in meds)
            print(f"  {name:45s}{med_str}")

    # ---- 8. Plot ----
    if n_show > 0:
        plot_top_boxplots(res_sig.head(n_show), met_log2, met_cols,
                          cluster_data, cluster_n, c_a, c_b)
    else:
        print("  No significant metabolites to plot.")

    print(f"\nDone -> {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# one figure: N_TOP box plots
# ---------------------------------------------------------------------------

def plot_top_boxplots(top_df, met_log2, met_cols,
                      cluster_data, cluster_n, c_a, c_b):
    n = len(top_df)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.8, n_rows * 2.5))
    axes = np.atleast_1d(axes).flat

    cl_ids = sorted(cluster_data.keys())  # all clusters [1, 2, 3, ...]
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    colors = {cl: palette[i % len(palette)] for i, cl in enumerate(cl_ids)}

    for i, (_, row) in enumerate(top_df.iterrows()):
        ax = axes[i]
        met_name = row["metabolite"]
        j = met_cols.index(met_name)

        # Gather data for all clusters
        all_vals = []
        all_meds = []
        all_stds = []
        for cl in cl_ids:
            vals = cluster_data[cl][:, j]
            all_vals.append(vals)
            all_meds.append(np.median(vals))
            all_stds.append(np.std(vals))

        positions = list(range(len(cl_ids)))
        bp = ax.boxplot(all_vals,
                         positions=positions, widths=0.5,
                         patch_artist=True,
                         medianprops={"color": "black", "lw": 1.5},
                         flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})

        for k_box, cl in enumerate(cl_ids):
            bp["boxes"][k_box].set_facecolor(colors[cl])
            bp["boxes"][k_box].set_alpha(0.75)

        # Jittered points
        rng = np.random.default_rng(42 + i)
        for k_pt, cl in enumerate(cl_ids):
            for val in all_vals[k_pt]:
                ax.scatter(k_pt + rng.uniform(-0.08, 0.08), val,
                           c=colors[cl], s=10, alpha=0.30, linewidths=0)

        # p-value + SNR
        p_str = f"p_perm={row['p_perm']:.4f}"
        snr_str = f"SNR={row['snr']:.2f}"
        ax.annotate(f"{p_str}  {snr_str}", xy=(0.5, 0.92), xycoords="axes fraction",
                    ha="center", fontsize=6.5, color="#555555")

        # Title: metabolite + all-cluster medians
        short_name = met_name[:28] + ".." if len(met_name) > 28 else met_name
        med_parts = "  ".join(f"c{cl}={all_meds[k]:+.1f}" for k, cl in enumerate(cl_ids))
        ax.set_title(f"{short_name}\n{med_parts}", fontsize=6.8, linespacing=1.3)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"c{cl}\n(n={cluster_n[cl]})" for cl in cl_ids], fontsize=7)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.3)

        if i % n_cols == 0:
            ax.set_ylabel("log2(FC)", fontsize=8)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    # Highlight the compared pair in title
    comp_str = " vs ".join(f"c{cl}" for cl in [c_a, c_b])
    fig.suptitle(f"{NEURON} (k={K} Ward): top {n} metabolites\n"
                 f"SNR based on {comp_str}, permutation test ({N_PERM} shuffles), dashed = reference (0)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_boxplots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> top_boxplots.png")


if __name__ == "__main__":
    main()
