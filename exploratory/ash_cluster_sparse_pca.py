"""ASH cluster separation: standard PCA vs SparsePCA (α=1.0)."""

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
from scipy.stats import mannwhitneyu
from sklearn.decomposition import SparsePCA
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

# ── helpers ────────────────────────────────────────────────────────────

def raw_missing_rates(rm, sids):
    sc = [s for s in sids.astype(str) if s in rm.columns]
    return rm.loc[:, sc].isna().mean(axis=1)


def build_ash_clusters(raw_path):
    raw = pd.read_parquet(raw_path)
    enriched = enrich_neural_dataframe(raw)

    ash = raw[raw["neuron"].isin(["ASHL", "ASHR"])].copy()
    ash["trial_id"] = (
        pd.to_datetime(ash["date"]).dt.strftime("%Y%m%d")
        + "__" + ash["worm_key"].astype(str)
        + "__" + ash["segment_index"].astype(str)
    )
    trial_avg = ash.groupby(["trial_id", "stimulus", "time_point"])["delta_F_over_F0"].mean().reset_index()
    proto = trial_avg.groupby(["stimulus", "time_point"])["delta_F_over_F0"].median().reset_index()
    mat = proto.pivot(index="stimulus", columns="time_point", values="delta_F_over_F0")

    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    r_mat = np.corrcoef(mat_z.values)
    dist = np.clip(1 - r_mat, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2

    clusters = fcluster(linkage(squareform(dist), method="ward"), 2, criterion="maxclust")
    stim_info = enriched.groupby("stimulus")[["species", "genus", "aid"]].first()
    labels = pd.DataFrame({"stimulus": mat.index.tolist(), "ash_cluster": clusters})
    labels["aid"] = labels["stimulus"].map(stim_info["aid"])
    return labels.set_index("aid")["ash_cluster"]


def prepare_chemical_data(fc_path, raw_meta_path):
    """QC-filter, log2, z-score. Returns (chem_z, metabolite_names, ash_aids)."""
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
    return chemical_z, list(chemical_z.columns), neural_samples


def evaluate_cluster_separation(scores_df, cluster_s, label):
    """Run Mann-Whitney per component + logistic regression. Print summary."""
    shared = sorted(set(scores_df.index) & set(cluster_s.index))
    y = (cluster_s.loc[shared] == 2).astype(int)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Per-component MW
    for col in scores_df.columns:
        c1 = scores_df.loc[cluster_s[cluster_s == 1].index.intersection(shared), col]
        c2 = scores_df.loc[cluster_s[cluster_s == 2].index.intersection(shared), col]
        u, p = mannwhitneyu(c1, c2, alternative="two-sided")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*" if p < 0.05 else "")
        nz = "—"
        print(f"  {col}:  c1={c1.mean():+.3f}  c2={c2.mean():+.3f}  p={p:.4f} {sig}")

    # Logistic regression
    X = scores_df.loc[shared].values
    lr = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l2", solver="liblinear",
        random_state=42, scoring="roc_auc",
    )
    lr.fit(X, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(lr, X, y, cv=skf, scoring="roc_auc")
    print(f"  CV AUC ({len(scores_df.columns)} components): {aucs.mean():.3f} ± {aucs.std():.3f}")

    # Top-3 components only
    pvals = []
    for col in scores_df.columns:
        c1 = scores_df.loc[cluster_s[cluster_s == 1].index.intersection(shared), col]
        c2 = scores_df.loc[cluster_s[cluster_s == 2].index.intersection(shared), col]
        _, p = mannwhitneyu(c1, c2, alternative="two-sided")
        pvals.append((col, p))
    top3 = [col for col, _ in sorted(pvals, key=lambda x: x[1])[:3]]

    X3 = scores_df.loc[shared, top3].values
    lr3 = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l2", solver="liblinear",
        random_state=42, scoring="roc_auc",
    )
    lr3.fit(X3, y)
    aucs3 = cross_val_score(lr3, X3, y, cv=skf, scoring="roc_auc")
    coef_str = ", ".join([f"{c}={lr3.coef_[0][i]:.4f}" for i, c in enumerate(top3)])
    print(f"  CV AUC ({' + '.join(top3)}): {aucs3.mean():.3f} ± {aucs3.std():.3f}")
    print(f"  Coefs: {coef_str}")
    return shared, y


# ── main ────────────────────────────────────────────────────────────────

ash_clusters = build_ash_clusters("data/106bac.parquet")
ash_aids = set(ash_clusters.index)
print(f"ASH: {len(ash_aids)} strains, "
      f"c1 n={int((ash_clusters == 1).sum())}, c2 n={int((ash_clusters == 2).sum())}")

chem_z, met_names, shared_samples = prepare_chemical_data(
    "data/data_fc_missingto1.xlsx",
    "data/metabolism_raw_data.xlsx",
)
X_centered = chem_z.to_numpy(dtype=float)
X_centered = X_centered - X_centered.mean(axis=0)

# ── Standard PCA ────────────────────────────────────────────────────────

_, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
pcs_std = X_centered @ Vt[:10].T
pcs_std_df = pd.DataFrame(pcs_std, index=shared_samples,
                           columns=[f"PC{i+1}" for i in range(10)])

evaluate_cluster_separation(pcs_std_df, ash_clusters, "Standard PCA (SVD)")

# ── SparsePCA α=1.0 ─────────────────────────────────────────────────────

spca = SparsePCA(n_components=10, alpha=1.0, max_iter=1000, random_state=42)
scores_sparse = spca.fit_transform(X_centered)
spca_comps = spca.components_  # (10, 256)
scores_sparse_df = pd.DataFrame(scores_sparse, index=shared_samples,
                                 columns=[f"sPC{i+1}" for i in range(10)])

# Sparsity stats
for i in range(10):
    nz = int(np.sum(np.abs(spca_comps[i]) > 1e-6))
    top_idx = np.argsort(-np.abs(spca_comps[i]))[:3]
    top_mets = [met_names[j] for j in top_idx]
    print(f"  sPC{i+1}: nonzeros={nz:3d}  top: {top_mets}")

evaluate_cluster_separation(scores_sparse_df, ash_clusters, "SparsePCA (α=1.0)")

# ── Scatter comparison ─────────────────────────────────────────────────

shared = sorted(ash_aids & set(shared_samples))
cluster_s = ash_clusters.loc[shared]

fig, axes = plt.subplots(2, 2, figsize=(13, 12))

# Standard PCA: PC2 vs PC5
for ax, xcol, ycol, title in [
    (axes[0, 0], "PC2", "PC5", "Standard PCA: PC2 vs PC5"),
    (axes[0, 1], "PC5", "PC6", "Standard PCA: PC5 vs PC6"),
    (axes[1, 0], "sPC2", "sPC5", "SparsePCA: sPC2 vs sPC5"),
    (axes[1, 1], "sPC5", "sPC6", "SparsePCA: sPC5 vs sPC6"),
]:
    x = pcs_std_df if "sP" not in xcol else scores_sparse_df
    x_vals = x.loc[shared, xcol]
    y_vals = x.loc[shared, ycol]
    for cl, color, label in [(1, "#e74c3c", "cluster 1"), (2, "#3498db", "cluster 2")]:
        mask = cluster_s == cl
        ax.scatter(x_vals[mask], y_vals[mask], c=color, s=48, alpha=0.75,
                   edgecolors="white", linewidths=0.5, label=label)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(fontsize=7)

fig.suptitle("ASH clusters in chemical PC space: Standard vs SparsePCA", fontsize=13)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ash_clusters_std_vs_sparse_pca.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── AUC comparison bar ──────────────────────────────────────────────────

results = [
    ("Std PCA\n10 comps", 0.848),
    ("Std PCA\nPC2+5+6", 0.859),
]

# Compute for sparse
X_sp = scores_sparse_df.loc[shared].values
y = (cluster_s == 2).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", solver="liblinear", random_state=42, scoring="roc_auc")
lr.fit(X_sp, y)
auc_sp_full = cross_val_score(lr, X_sp, y, cv=skf, scoring="roc_auc")
results.append(("SparsePCA\n10 comps", float(auc_sp_full.mean())))

# Top 3 sparse
sparse_pvals = []
for col in scores_sparse_df.columns:
    c1 = scores_sparse_df.loc[cluster_s[cluster_s == 1].index.intersection(shared), col]
    c2 = scores_sparse_df.loc[cluster_s[cluster_s == 2].index.intersection(shared), col]
    _, p = mannwhitneyu(c1, c2, alternative="two-sided")
    sparse_pvals.append((col, p))
top3_sparse = [col for col, _ in sorted(sparse_pvals, key=lambda x: x[1])[:3]]
X_sp3 = scores_sparse_df.loc[shared, top3_sparse].values
lr3 = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", solver="liblinear", random_state=42, scoring="roc_auc")
lr3.fit(X_sp3, y)
auc_sp3 = cross_val_score(lr3, X_sp3, y, cv=skf, scoring="roc_auc")
results.append((f"SparsePCA\n{' + '.join(top3_sparse)}", float(auc_sp3.mean())))

# Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [r[0] for r in results]
aucs = [r[1] for r in results]
colors = ["#64748B", "#64748B", "#e67e22", "#e67e22"]
bars = ax.bar(range(len(labels)), aucs, color=colors, edgecolor="white", linewidth=1)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.03,
            f"{auc:.3f}", ha="center", va="top", fontsize=11, fontweight="bold", color="white")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("5-fold CV AUC")
ax.set_title("ASH cluster separation: Standard PCA vs SparsePCA")
ax.set_ylim(0.75, 0.95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ash_clusters_auc_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved to {OUTPUT_DIR}/")
