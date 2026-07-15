"""Compare ASH neural response clusters against chemical PC scores.

Tests whether the two Ward clusters of ASH (identified via neural trace shape)
can be separated by chemical metabolite profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis._data_loaders import (
    read_metabolite_matrix,
    enrich_neural_dataframe,
    _canonicalize_metabolite_name,
)

OUTPUT_DIR = Path("results/ash_cluster_chemical_pcs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────


def raw_missing_rates(raw_metadata, sample_ids):
    sample_columns = [s for s in sample_ids.astype(str) if s in raw_metadata.columns]
    return raw_metadata.loc[:, sample_columns].isna().mean(axis=1)


def build_ash_clusters(raw_parquet_path):
    """Return Series: AID → ASH cluster label (1 or 2)."""
    raw = pd.read_parquet(raw_parquet_path)
    enriched = enrich_neural_dataframe(raw)

    ash_raw = raw[raw["neuron"].isin(["ASHL", "ASHR"])].copy()
    ash_raw["trial_id"] = (
        pd.to_datetime(ash_raw["date"]).dt.strftime("%Y%m%d")
        + "__"
        + ash_raw["worm_key"].astype(str)
        + "__"
        + ash_raw["segment_index"].astype(str)
    )
    trial_avg = (
        ash_raw.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"]
        .mean()
        .reset_index()
    )
    proto = (
        trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"]
        .median()
        .reset_index()
    )
    ash_mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")

    ash_z = ash_mat.subtract(ash_mat.mean(axis=1), axis=0).div(ash_mat.std(axis=1), axis=0)
    r_mat = np.corrcoef(ash_z.values)
    ash_dist = np.clip(1 - r_mat, 0, None)
    np.fill_diagonal(ash_dist, 0)
    ash_dist = (ash_dist + ash_dist.T) / 2

    ash_clusters = fcluster(
        linkage(squareform(ash_dist), method="ward"), 2, criterion="maxclust"
    )

    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    labels = pd.DataFrame(
        {"stimulus": ash_mat.index.tolist(), "ash_cluster": ash_clusters}
    )
    labels["aid"] = labels["stimulus"].map(stim_info["aid"])
    return labels.set_index("aid")["ash_cluster"]


def build_chemical_pcs(fc_path, raw_meta_path, retained_features):
    """Return (pcs_df, metabolite_names, Vt)."""
    fc = read_metabolite_matrix(fc_path)
    raw_meta = pd.read_excel(raw_meta_path, sheet_name="all", engine="openpyxl")

    annotations = raw_meta.copy()
    annotations["_feature_name"] = annotations["name"].map(_canonicalize_metabolite_name)
    annotations["_qcrsd"] = pd.to_numeric(annotations["QCRSD"], errors="coerce")
    annotations["_raw_missing_rate"] = raw_missing_rates(annotations, fc.index)
    available = set(fc.columns.astype(str))
    retained = annotations.loc[
        annotations["_feature_name"].isin(available)
        & annotations["_qcrsd"].le(0.2)
        & annotations["_raw_missing_rate"].le(0.5)
    ].drop_duplicates("_feature_name", keep="first")
    retained_features = retained["_feature_name"].astype(str).tolist()

    neural_samples = sorted([s for s in fc.index if s in ash_aids])
    chemical = fc.loc[neural_samples, retained_features].apply(pd.to_numeric, errors="coerce")
    chemical = chemical.where(chemical > 0)
    chemical_log2 = np.log2(chemical)

    means = chemical_log2.mean(axis=0)
    stds = chemical_log2.std(axis=0, ddof=0)
    valid = stds[np.isfinite(stds) & (stds > 0)].index
    chemical_z = (chemical_log2.loc[:, valid] - means[valid]) / stds[valid]
    metabolite_names = list(chemical_z.columns)

    X = chemical_z.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    pcs = X @ Vt[:10].T
    pcs_df = pd.DataFrame(
        pcs, index=neural_samples, columns=[f"PC{i + 1}" for i in range(10)]
    )
    return pcs_df, metabolite_names, Vt, chemical_z, neural_samples


# ── main ─────────────────────────────────────────────────────────────────


ash_clusters = build_ash_clusters("data/106bac.parquet")
ash_aids = set(ash_clusters.index)
print(f"ASH: {len(ash_aids)} strains, "
      f"cluster1 n={(ash_clusters == 1).sum()}, "
      f"cluster2 n={(ash_clusters == 2).sum()}")

pcs_df, met_names, Vt, chem_z, shared_samples = build_chemical_pcs(
    "data/data_fc_missingto1.xlsx",
    "data/metabolism_raw_data.xlsx",
    None,  # computed internally
)

# Align
shared = sorted(ash_aids & set(shared_samples))
cluster_s = ash_clusters.loc[shared]
y = (cluster_s == 2).astype(int)

# ── 1. Per-PC Mann–Whitney ──────────────────────────────────────────────

print("\n===== ASH cluster vs chemical PCs =====")
pc_pvals = {}
for pc in pcs_df.columns:
    c1 = pcs_df.loc[cluster_s[cluster_s == 1].index, pc]
    c2 = pcs_df.loc[cluster_s[cluster_s == 2].index, pc]
    u, p = mannwhitneyu(c1, c2, alternative="two-sided")
    pc_pvals[pc] = p
    sig = ""
    if p < 0.05:
        sig = "*"
    if p < 0.01:
        sig = "**"
    if p < 0.001:
        sig = "***"
    print(f"  {pc}:  c1={c1.mean():+.3f}  c2={c2.mean():+.3f}  p={p:.4f} {sig}")

# ── 2. Top loadings for significant PCs ─────────────────────────────────

for pc_idx, pc_label in [(1, "PC2"), (4, "PC5"), (5, "PC6")]:
    loadings = Vt[pc_idx]
    top = np.argsort(-np.abs(loadings))[:15]
    print(f"\n{pc_label} top metabolites:")
    for idx in top:
        sign = "+" if loadings[idx] > 0 else "-"
        print(f"  {sign} {met_names[idx]:45s}  ({loadings[idx]:.4f})")

# ── 3. Logistic regression ──────────────────────────────────────────────

X_all = pcs_df.loc[shared].values
lr = LogisticRegressionCV(
    Cs=10, cv=5, penalty="l2", solver="liblinear", random_state=42, scoring="roc_auc"
)
lr.fit(X_all, y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = cross_val_score(lr, X_all, y, cv=skf, scoring="roc_auc")
print(f"\n===== Logistic Regression (10 PCs → ASH cluster) =====")
print(f"  5-fold CV AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
print(f"  Coefs: {dict(zip(pcs_df.columns, lr.coef_[0]))}")

# Just PC2+PC5+PC6
X_sig = pcs_df.loc[shared, ["PC2", "PC5", "PC6"]].values
lr_sig = LogisticRegressionCV(
    Cs=10, cv=5, penalty="l2", solver="liblinear", random_state=42, scoring="roc_auc"
)
lr_sig.fit(X_sig, y)
aucs_sig = cross_val_score(lr_sig, X_sig, y, cv=skf, scoring="roc_auc")
print(f"\n===== Logistic Regression (PC2+5+6 only) =====")
print(f"  5-fold CV AUC: {aucs_sig.mean():.3f} ± {aucs_sig.std():.3f}")
print(f"  Coefs: PC2={lr_sig.coef_[0][0]:.4f}, PC5={lr_sig.coef_[0][1]:.4f}, PC6={lr_sig.coef_[0][2]:.4f}")

# ── 4. Top individual metabolites ───────────────────────────────────────

print("\n===== Top individual metabolites (MW, uncorrected) =====")
c1_mask = cluster_s == 1
c2_mask = cluster_s == 2
results = []
for i, met in enumerate(met_names):
    v1 = chem_z.iloc[:, i].loc[shared][c1_mask]
    v2 = chem_z.iloc[:, i].loc[shared][c2_mask]
    u, p = mannwhitneyu(v1, v2, alternative="two-sided")
    results.append({"metabolite": met, "p": p, "delta": v2.mean() - v1.mean()})
res_df = pd.DataFrame(results).sort_values("p")
n_tests = len(res_df)
for _, row in res_df.head(15).iterrows():
    bonf = min(row["p"] * n_tests, 1.0)
    sig = ""
    if bonf < 0.05:
        sig = "*"
    if bonf < 0.01:
        sig = "**"
    if bonf < 0.001:
        sig = "***"
    print(f"  {row['metabolite']:45s}  p={row['p']:.6f}  bonf={bonf:.4f}  Δ={row['delta']:.3f} {sig}")

# ── 5. Scatter plot ─────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
plot_pairs = [
    (0, 0, "PC2", "PC5"),
    (0, 1, "PC2", "PC6"),
    (0, 2, "PC5", "PC6"),
    (1, 0, "PC1", "PC2"),
    (1, 1, "PC1", "PC5"),
    (1, 2, "PC4", "PC7"),
]
colors = np.where(cluster_s == 1, "#e74c3c", "#3498db")
for row, col, pcx, pcy in plot_pairs:
    ax = axes[row, col]
    x_vals = pcs_df.loc[shared, pcx]
    y_vals = pcs_df.loc[shared, pcy]
    ax.scatter(
        x_vals[cluster_s == 1], y_vals[cluster_s == 1],
        c="#e74c3c", s=48, alpha=0.75, edgecolors="white", linewidths=0.5,
        label=f"ASH cluster 1 (n={(cluster_s==1).sum()})",
    )
    ax.scatter(
        x_vals[cluster_s == 2], y_vals[cluster_s == 2],
        c="#3498db", s=48, alpha=0.75, edgecolors="white", linewidths=0.5,
        label=f"ASH cluster 2 (n={(cluster_s==2).sum()})",
    )
    ax.set_xlabel(pcx)
    ax.set_ylabel(pcy)
    ax.set_title(f"{pcx} vs {pcy}")
    if row == 0 and col == 0:
        ax.legend(fontsize=7, loc="upper right")

fig.suptitle("ASH neural response clusters in chemical PC space", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ash_clusters_chemical_pcs.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / 'ash_clusters_chemical_pcs.png'}")

# ── 6. ASH trajectory overlay by cluster ────────────────────────────────

raw = pd.read_parquet("data/106bac.parquet")
ash_raw = raw[raw["neuron"].isin(["ASHL", "ASHR"])].copy()
ash_raw["trial_id"] = (
    pd.to_datetime(ash_raw["date"]).dt.strftime("%Y%m%d")
    + "__"
    + ash_raw["worm_key"].astype(str)
    + "__"
    + ash_raw["segment_index"].astype(str)
)
trial_avg = (
    ash_raw.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"]
    .mean()
    .reset_index()
)
proto = (
    trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"]
    .median()
    .reset_index()
)
ash_mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")
ash_z = ash_mat.subtract(ash_mat.mean(axis=1), axis=0).div(ash_mat.std(axis=1), axis=0)
timepoints = ash_mat.columns.astype(int).tolist()

stim_info = enrich_neural_dataframe(raw)
aid_map = stim_info.groupby("stimulus")["aid"].first()
ash_mat.index = ash_mat.index.map(aid_map)
ash_z.index = ash_z.index.map(aid_map)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cl, color, label in [(1, "#e74c3c", "cluster 1"), (2, "#3498db", "cluster 2")]:
    aids_in_cl = cluster_s[cluster_s == cl].index.tolist()
    for aid in aids_in_cl:
        if aid in ash_z.index:
            axes[0].plot(timepoints, ash_z.loc[aid].values, alpha=0.25, lw=0.7, color=color)
    mean_traj = ash_z.loc[[a for a in aids_in_cl if a in ash_z.index]].mean(axis=0)
    axes[0].plot(timepoints, mean_traj, color=color, lw=2.5, label=label)
    # raw dfof
    for aid in aids_in_cl:
        if aid in ash_mat.index:
            axes[1].plot(timepoints, ash_mat.loc[aid].values, alpha=0.25, lw=0.7, color=color)
    mean_raw = ash_mat.loc[[a for a in aids_in_cl if a in ash_mat.index]].mean(axis=0)
    axes[1].plot(timepoints, mean_raw, color=color, lw=2.5, label=label)

for ax in axes:
    ax.axvspan(5, 15, alpha=0.1, color="#e74c3c")
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_title("ASH z-scored by cluster")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("z-score")
axes[1].set_title("ASH ΔF/F₀ by cluster")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("ΔF/F₀")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ash_trajectories_by_cluster.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTPUT_DIR / 'ash_trajectories_by_cluster.png'}")
