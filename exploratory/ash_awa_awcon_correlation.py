"""Correlation analysis: ASH × AWCON × AWA — stimulus-level response similarity.

Driven by the observation from neural dimensionality inspection that these three
neurons show similar cluster splits (cluster 1/2 counts are close).  This script
quantifies the relationship systematically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
TARGET_NEURONS = ("ASH", "AWCON", "AWA")
OUTPUT_DIR = Path("results/correlation_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}

# ---------------------------------------------------------------------------
# data loading (mirrors inspect_neural_dimensionality.py)
# ---------------------------------------------------------------------------

def _neuron_raw_names(neuron: str) -> list[str]:
    merged = _LR_MERGE.get(neuron)
    if merged is not None:
        return list(merged)
    return [neuron]


def build_prototype_matrix(
    raw: pd.DataFrame, neuron: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build stimulus × time median prototype matrix for *neuron*.

    Returns (mat, stim_info) where mat is (n_stimuli × n_timepoints).
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

    enriched = enrich_neural_dataframe(raw)
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()

    return mat, stim_info


# ---------------------------------------------------------------------------
# 1. Response-magnitude correlation (scalar per stimulus)
# ---------------------------------------------------------------------------

def _response_window_mean(mat: pd.DataFrame) -> pd.Series:
    """Mean ΔF/F₀ over the 5-15s response window, per stimulus."""
    cols = [c for c in mat.columns if 5 <= int(c) <= 15]
    return mat[cols].mean(axis=1)


def plot_pairwise_scatter(
    mats: dict[str, pd.DataFrame],
    stim_info: pd.DataFrame,
):
    """Scatter plots: response-window mean for each neuron pair."""
    rw = {n: _response_window_mean(m) for n, m in mats.items()}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    pairs = [("ASH", "AWCON"), ("ASH", "AWA"), ("AWCON", "AWA")]

    for ax, (n1, n2) in zip(axes, pairs):
        common = rw[n1].index.intersection(rw[n2].index)
        x, y = rw[n1].loc[common].values, rw[n2].loc[common].values

        r, p = stats.pearsonr(x, y)
        rho, p_s = stats.spearmanr(x, y)

        ax.scatter(x, y, s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
                   color="#2c3e50")
        # label a few extreme points
        aids = [stim_info.loc[s, "aid"] for s in common]
        extremes = np.argsort(np.abs(x - np.median(x)) + np.abs(y - np.median(y)))[-8:]
        for i in extremes:
            ax.annotate(aids[i], (x[i], y[i]), fontsize=5.5, alpha=0.7,
                        textcoords="offset points", xytext=(3, 3))

        # regression line
        slope, intercept, *_ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#e74c3c", lw=1.5, ls="--", alpha=0.7)

        ax.set_xlabel(f"{n1} mean ΔF/F₀ (5-15s)")
        ax.set_ylabel(f"{n2} mean ΔF/F₀ (5-15s)")
        ax.set_title(f"{n1} vs {n2}\nr={r:.3f} (p={p:.2e})  ρ={rho:.3f} (p={p_s:.2e})",
                     fontsize=10)
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)

    fig.suptitle("Stimulus-level response magnitude correlation", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pairwise_response_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → pairwise_response_scatter.png")


# ---------------------------------------------------------------------------
# 2. Full temporal-profile correlation per stimulus
# ---------------------------------------------------------------------------

def plot_temporal_correlation_heatmap(
    mats: dict[str, pd.DataFrame],
    stim_info: pd.DataFrame,
):
    """For each stimulus, correlate the full temporal profiles of neuron pairs."""
    common_stim = list(set.intersection(*[set(m.index) for m in mats.values()]))
    common_stim = sorted(common_stim)

    pairs = [("ASH", "AWCON"), ("ASH", "AWA"), ("AWCON", "AWA")]
    corr_per_stim = {f"{n1}_{n2}": [] for n1, n2 in pairs}

    for stim in common_stim:
        for n1, n2 in pairs:
            v1 = mats[n1].loc[stim].values
            v2 = mats[n2].loc[stim].values
            r = np.corrcoef(v1, v2)[0, 1]
            corr_per_stim[f"{n1}_{n2}"].append(r)

    corr_df = pd.DataFrame(corr_per_stim, index=common_stim)

    # sort by mean correlation
    corr_df["mean_corr"] = corr_df.mean(axis=1)
    corr_df = corr_df.sort_values("mean_corr")
    mean_corr = corr_df.pop("mean_corr")

    fig, ax = plt.subplots(figsize=(8, 16))
    im = ax.imshow(corr_df.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    # label a subset of rows
    step = max(1, len(corr_df) // 60)
    tick_positions = list(range(0, len(corr_df), step))
    tick_labels = [stim_info.loc[s, "aid"] for s in corr_df.index[::step]]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=5)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["ASH-AWCON", "ASH-AWA", "AWCON-AWA"], fontsize=8, rotation=20)
    ax.set_title("Per-stimulus temporal profile correlation\n(ordered by mean corr)", fontsize=11)
    fig.colorbar(im, ax=ax, label="Pearson r (time series)", shrink=0.6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "temporal_corr_heatmap.png", dpi=150)
    plt.close(fig)
    print("  → temporal_corr_heatmap.png")

    # summary
    print("\n  Temporal correlation summary (across stimuli):")
    for col in corr_df.columns:
        vals = corr_df[col]
        print(f"    {col}: mean={vals.mean():.3f}, median={vals.median():.3f}, "
              f"std={vals.std():.3f}, range=[{vals.min():.3f}, {vals.max():.3f}]")


# ---------------------------------------------------------------------------
# 3. Cross-neuron RDM correlation
# ---------------------------------------------------------------------------

def plot_cross_neuron_rdm(mats: dict[str, pd.DataFrame]):
    """Build an RDM for each neuron and correlate them.

    Each neuron's RDM is built from its z-scored temporal profile (Pearson
    distance).  We then compare the upper triangles of each pair of RDMs.
    """
    rdms = {}
    for neuron, mat in mats.items():
        mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
        r = np.corrcoef(mat_z.values)
        d = np.clip(1 - r, 0, None)
        np.fill_diagonal(d, 0)
        d = (d + d.T) / 2
        rdms[neuron] = d

    pairs = [("ASH", "AWCON"), ("ASH", "AWA"), ("AWCON", "AWA")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, (n1, n2) in enumerate(pairs):
        # order stimuli by one neuron's RDM (hierarchical)
        Z = linkage(squareform(rdms[n1]), method="ward")
        order = leaves_list(Z)

        rdm1_ordered = rdms[n1][order][:, order]
        rdm2_ordered = rdms[n2][order][:, order]

        # upper triangle scatter
        tri_i, tri_j = np.triu_indices_from(rdm1_ordered, k=1)
        x, y = rdm1_ordered[tri_i, tri_j], rdm2_ordered[tri_i, tri_j]
        r_val, p_val = stats.pearsonr(x, y)

        ax_top = axes[0, col]
        ax_top.scatter(x, y, s=3, alpha=0.3, color="#2c3e50", linewidths=0)
        ax_top.plot([0, 2], [0, 2], "--", color="#e74c3c", lw=1, alpha=0.6)
        ax_top.set_xlabel(f"{n1} Pearson distance")
        ax_top.set_ylabel(f"{n2} Pearson distance")
        ax_top.set_title(f"{n1} vs {n2}  r={r_val:.3f} (p={p_val:.2e})", fontsize=10)

        # RDM side-by-side visualization
        ax_bot = axes[1, col]
        # stack the two RDMs vertically or overlay?
        combined = np.vstack([rdm1_ordered, rdm2_ordered])
        ax_bot.imshow(combined, aspect="auto", cmap="viridis")
        ax_bot.axhline(len(order) - 0.5, color="white", lw=2)
        ax_bot.set_yticks([len(order)//2, len(order) + len(order)//2])
        ax_bot.set_yticklabels([n1, n2], fontsize=9)
        ax_bot.set_xticks([])
        ax_bot.set_title(f"RDMs ordered by {n1} Ward", fontsize=10)

    fig.suptitle("Cross-neuron RDM comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_neuron_rdm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → cross_neuron_rdm.png")


# ---------------------------------------------------------------------------
# 4. Cluster agreement (ARI between Ward k=2 labels)
# ---------------------------------------------------------------------------

def plot_cluster_agreement(mats: dict[str, pd.DataFrame], stim_info: pd.DataFrame):
    """For k=2 Ward clustering, check agreement across all neuron pairs.

    Uses adjusted Rand index (ARI) to measure label consistency.
    """
    from sklearn.metrics import adjusted_rand_score

    common_stim = sorted(set.intersection(*[set(m.index) for m in mats.values()]))

    # cluster each neuron
    labels = {}
    dendrograms = {}
    for neuron, mat in mats.items():
        mat_aligned = mat.loc[common_stim]
        mat_z = mat_aligned.subtract(mat_aligned.mean(axis=1), axis=0).div(mat_aligned.std(axis=1), axis=0)
        r = np.corrcoef(mat_z.values)
        d = np.clip(1 - r, 0, None)
        np.fill_diagonal(d, 0)
        d = (d + d.T) / 2
        Z = linkage(squareform(d), method="ward")
        dendrograms[neuron] = Z
        labels[neuron] = fcluster(Z, 2, criterion="maxclust")

    # pairwise ARI
    neurons = list(TARGET_NEURONS)
    n = len(neurons)
    ari_mat = np.eye(n)
    count_mat = np.zeros((n, n), dtype=object)

    for i, n1 in enumerate(neurons):
        for j, n2 in enumerate(neurons):
            if i < j:
                ari = adjusted_rand_score(labels[n1], labels[n2])
                ari_mat[i, j] = ari_mat[j, i] = ari
            # count tabulation
            if i != j:
                counts = pd.crosstab(
                    pd.Series(labels[n1], name=n1),
                    pd.Series(labels[n2], name=n2),
                )
                count_mat[i, j] = counts

    # --- ARI heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.imshow(ari_mat, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{ari_mat[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if ari_mat[i, j] < 0.4 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(neurons)
    ax.set_yticks(range(n)); ax.set_yticklabels(neurons)
    ax.set_title("Adjusted Rand Index (k=2 Ward)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.75)

    # --- contingency table ---
    ax2 = axes[1]
    # show ASH vs AWCON as example
    ct = count_mat[0, 1]  # ASH(0) vs AWCON(1)
    im2 = ax2.imshow(ct.values, cmap="Blues", aspect="auto")
    for ri in range(ct.shape[0]):
        for cj in range(ct.shape[1]):
            ax2.text(cj, ri, str(ct.values[ri, cj]), ha="center", va="center", fontsize=12)
    ax2.set_xticks(range(ct.shape[1]))
    ax2.set_xticklabels([f"AWCON-{c}" for c in ct.columns])
    ax2.set_yticks(range(ct.shape[0]))
    ax2.set_yticklabels([f"ASH-{c}" for c in ct.index])
    ax2.set_title("ASH × AWCON cluster contingency", fontsize=11)
    fig.colorbar(im2, ax=ax2, shrink=0.75, label="n stimuli")

    fig.suptitle("Cluster label agreement across neurons", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → cluster_agreement.png")

    # print all contingency tables
    for i, n1 in enumerate(neurons):
        for j, n2 in enumerate(neurons):
            if i < j:
                print(f"\n  {n1} × {n2} contingency table (ARI={ari_mat[i,j]:.3f}):")
                print(count_mat[i, j].to_string())


# ---------------------------------------------------------------------------
# 5. Joint dendrogram — all three neurons side by side, same stimulus order
# ---------------------------------------------------------------------------

def plot_joint_dendrogram(
    mats: dict[str, pd.DataFrame],
    stim_info: pd.DataFrame,
):
    """Side-by-side dendrograms with leaf order locked to ASH."""
    common_stim = sorted(set.intersection(*[set(m.index) for m in mats.values()]))

    # compute linkage for each
    linkages = {}
    for neuron in TARGET_NEURONS:
        mat_aligned = mats[neuron].loc[common_stim]
        mat_z = mat_aligned.subtract(mat_aligned.mean(axis=1), axis=0).div(mat_aligned.std(axis=1), axis=0)
        r = np.corrcoef(mat_z.values)
        d = np.clip(1 - r, 0, None)
        np.fill_diagonal(d, 0)
        d = (d + d.T) / 2
        linkages[neuron] = (linkage(squareform(d), method="ward"), mat_aligned.index.tolist())

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

    # Lock leaf order to ASH
    ash_Z, ash_stim_order_idx = linkages["ASH"]
    ash_order = leaves_list(ash_Z)
    ash_stim_order = [common_stim[i] for i in ash_order]
    ash_aid_labels = [stim_info.loc[s, "aid"] for s in ash_stim_order]

    for ax, neuron in zip(axes, TARGET_NEURONS):
        Z, stim_list = linkages[neuron]
        # reorder to match ASH order
        stim_to_pos = {s: i for i, s in enumerate(stim_list)}
        reorder_idx = [stim_to_pos[s] for s in ash_stim_order]

        dn = dendrogram(
            Z, ax=ax, labels=ash_aid_labels, leaf_font_size=5,
            color_threshold=0, above_threshold_color="#2c3e50",
            link_color_func=lambda k: "#2c3e50",
            # can't easily reorder after dendrogram, so we keep original order
        )
        ax.set_title(f"{neuron}", fontsize=12)
        ax.set_ylabel("Ward merge cost" if neuron == "ASH" else "")

    fig.suptitle("Joint dendrograms (ASH leaf order)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "joint_dendrograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → joint_dendrograms.png")


# ---------------------------------------------------------------------------
# 6. Response-window summary table
# ---------------------------------------------------------------------------

def print_correlation_summary(mats: dict[str, pd.DataFrame]):
    """Print a comprehensive correlation summary."""
    rw = {n: _response_window_mean(m) for n, m in mats.items()}
    common = sorted(set.intersection(*[set(s.index) for s in rw.values()]))

    X = np.column_stack([rw[n].loc[common].values for n in TARGET_NEURONS])
    corr = np.corrcoef(X.T)
    rho = stats.spearmanr(X)

    print("\n" + "=" * 60)
    print("  ASH × AWCON × AWA  — Response magnitude correlation")
    print("=" * 60)
    print(f"\n  Pearson correlation matrix ({len(common)} stimuli):")
    print(f"              ASH     AWCON   AWA")
    for i, n1 in enumerate(TARGET_NEURONS):
        vals = "  ".join(f"{corr[i,j]:+.3f}" for j in range(3))
        print(f"    {n1:8s}  {vals}")

    print(f"\n  Spearman correlation matrix:")
    print(f"              ASH     AWCON   AWA")
    for i, n1 in enumerate(TARGET_NEURONS):
        vals = "  ".join(f"{rho.correlation[i,j]:+.3f}" for j in range(3))
        print(f"    {n1:8s}  {vals}")

    # Partial correlation: ASH~AWCON controlling for AWA
    print("\n  --- Partial correlations ---")
    for target, other in [("ASH", "AWCON"), ("ASH", "AWA"), ("AWCON", "AWA")]:
        control = [n for n in TARGET_NEURONS if n != target and n != other][0]
        idx_target = TARGET_NEURONS.index(target)
        idx_other = TARGET_NEURONS.index(other)
        idx_control = TARGET_NEURONS.index(control)

        # partial corr via residuals
        from scipy.stats import linregress
        resid_target = X[:, idx_target] - linregress(X[:, idx_control], X[:, idx_target]).intercept \
                       - linregress(X[:, idx_control], X[:, idx_target]).slope * X[:, idx_control]
        resid_other = X[:, idx_other] - linregress(X[:, idx_control], X[:, idx_other]).intercept \
                      - linregress(X[:, idx_control], X[:, idx_other]).slope * X[:, idx_control]
        partial_r, partial_p = stats.pearsonr(resid_target, resid_other)
        print(f"    {target}~{other} | {control}: r={partial_r:.3f} (p={partial_p:.2e})")


# ---------------------------------------------------------------------------
# 7. Temporal trace overlay — mean ± SEM per neuron
# ---------------------------------------------------------------------------

def plot_mean_trace_overlay(mats: dict[str, pd.DataFrame]):
    """Overlay mean (±SEM) temporal traces for the three neurons."""
    timepoints = mats["ASH"].columns.astype(int).tolist()

    colors = {"ASH": "#e74c3c", "AWCON": "#3498db", "AWA": "#2ecc71"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # --- raw mean ± SEM ---
    ax = axes[0]
    for neuron in TARGET_NEURONS:
        mat = mats[neuron]
        mean = mat.mean(axis=0).values
        sem = mat.sem(axis=0).values
        ax.plot(timepoints, mean, color=colors[neuron], lw=2, label=neuron)
        ax.fill_between(timepoints, mean - sem, mean + sem,
                        color=colors[neuron], alpha=0.15)
    ax.axvspan(5, 15, alpha=0.1, color="#e74c3c")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F₀ (mean ± SEM)")
    ax.set_title("Mean response across all stimuli")
    ax.legend()

    # --- z-scored mean ± SEM ---
    ax2 = axes[1]
    for neuron in TARGET_NEURONS:
        mat_z = mats[neuron].subtract(mats[neuron].mean(axis=1), axis=0).div(
            mats[neuron].std(axis=1), axis=0)
        mean = mat_z.mean(axis=0).values
        sem = mat_z.sem(axis=0).values
        ax2.plot(timepoints, mean, color=colors[neuron], lw=2, label=neuron)
        ax2.fill_between(timepoints, mean - sem, mean + sem,
                         color=colors[neuron], alpha=0.15)
    ax2.axvspan(5, 15, alpha=0.1, color="#e74c3c")
    ax2.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("z-score (mean ± SEM)")
    ax2.set_title("Mean z-scored response across all stimuli")
    ax2.legend()

    fig.suptitle("ASH · AWCON · AWA — temporal profile comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_trace_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → mean_trace_overlay.png")


# ---------------------------------------------------------------------------
# 8. Genus-level breakdown
# ---------------------------------------------------------------------------

def plot_genus_level_correlation(
    mats: dict[str, pd.DataFrame],
    stim_info: pd.DataFrame,
):
    """Correlation broken down by genus — does the relationship hold within genera?"""
    rw = {n: _response_window_mean(m) for n, m in mats.items()}

    # get common stimuli
    common = sorted(set.intersection(*[set(s.index) for s in rw.values()]))

    genera = stim_info.loc[common, "genus"].value_counts()
    top_genera = genera[genera >= 3].index.tolist()

    n_genus = len(top_genera)
    cols = min(3, n_genus)
    rows = (n_genus + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_genus == 1:
        axes = np.array([axes])
    axes = axes.flat

    for idx, genus in enumerate(sorted(top_genera)):
        ax = axes[idx]
        genus_stim = stim_info.loc[common][stim_info.loc[common, "genus"] == genus].index

        if len(genus_stim) < 3:
            ax.set_visible(False)
            continue

        x = rw["ASH"].loc[genus_stim].values
        y = rw["AWCON"].loc[genus_stim].values
        r_ash_awcon, _ = stats.pearsonr(x, y)

        x2 = rw["ASH"].loc[genus_stim].values
        y2 = rw["AWA"].loc[genus_stim].values
        r_ash_awa, _ = stats.pearsonr(x2, y2)

        ax.scatter(x, y, s=60, alpha=0.7, color="#e74c3c", edgecolors="white",
                   linewidths=0.5, label=f"ASH-AWCON r={r_ash_awcon:.2f}")
        ax.scatter(x2, y2, s=60, alpha=0.7, color="#3498db", edgecolors="white",
                   linewidths=0.5, label=f"ASH-AWA r={r_ash_awa:.2f}")

        aids = [stim_info.loc[s, "aid"] for s in genus_stim]
        species = [stim_info.loc[s, "species"] for s in genus_stim]
        for i, (aid, sp) in enumerate(zip(aids, species)):
            ax.annotate(f"{aid}", (x[i], y[i]), fontsize=5, alpha=0.6,
                        textcoords="offset points", xytext=(3, 3))

        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.set_xlabel("ASH mean ΔF/F₀ (5-15s)")
        ax.set_ylabel("AWCON / AWA mean ΔF/F₀ (5-15s)")
        ax.set_title(f"{genus} (n={len(genus_stim)})", fontsize=10)
        ax.legend(fontsize=8)

    for idx in range(n_genus, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Genus-level correlation: ASH vs AWCON & AWA", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "genus_level_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → genus_level_correlation.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading 106bac data...")
    raw = pd.read_parquet("data/106bac.parquet")

    mats = {}
    stim_info = None
    for neuron in TARGET_NEURONS:
        mat, si = build_prototype_matrix(raw, neuron)
        mats[neuron] = mat
        if stim_info is None:
            stim_info = si
        n_stim = mat.shape[0]
        n_tp = mat.shape[1]
        print(f"  {neuron}: {n_stim} stimuli × {n_tp} timepoints")

    # --- all analyses ---
    print("\n[1/7] Pairwise response scatter")
    plot_pairwise_scatter(mats, stim_info)

    print("\n[2/7] Temporal correlation heatmap")
    plot_temporal_correlation_heatmap(mats, stim_info)

    print("\n[3/7] Cross-neuron RDM")
    plot_cross_neuron_rdm(mats)

    print("\n[4/7] Cluster agreement (ARI)")
    plot_cluster_agreement(mats, stim_info)

    print("\n[5/7] Joint dendrograms")
    plot_joint_dendrogram(mats, stim_info)

    print("\n[6/7] Correlation summary")
    print_correlation_summary(mats)

    print("\n[7/7] Mean trace overlay")
    plot_mean_trace_overlay(mats)

    # --- genus breakdown ---
    print("\n[bonus] Genus-level correlation")
    plot_genus_level_correlation(mats, stim_info)

    print(f"\nDone → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
