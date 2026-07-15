"""Diagnose chemical distance metric: curse of dimensionality, feature correlation,
and Mahalanobis distance for 106bac RSA."""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from bacteria_analysis._analysis_dataset_impl import AnalysisDataset
from bacteria_analysis._data_loaders import build_stimulus_sample_map, read_metabolite_matrix
from bacteria_analysis.features.chemical import (
    _feature_annotations,
    _numeric_matrix,
    _retained_features,
    _stimulus_matrix,
    _transform_matrix,
)
from bacteria_analysis.analyses.rdm.builders import build_neural_rdm
from bacteria_analysis.analyses.rdm.core import align_square_rdms

# ---- shared setup ----
neural_raw = pd.read_parquet("data/106bac.parquet")
neural_raw["date"] = neural_raw["date"].fillna("").astype(str).str.strip()
matrix = read_metabolite_matrix("data/matrix.xlsx")
metadata = pd.read_excel("data/metabolism_raw_data.xlsx", sheet_name="all", engine="openpyxl")
stimulus_sample_map = build_stimulus_sample_map(neural_raw, matrix_sample_ids=matrix.index)
dates = tuple(sorted(neural_raw["date"].dropna().astype(str).unique()))
ds = AnalysisDataset(
    neural=neural_raw.reset_index(drop=True),
    matrix=matrix,
    metadata=metadata.reset_index(drop=True),
    stimulus_sample_map=stimulus_sample_map,
    included_dates=dates,
    excluded_dates=(),
    parameters={},
)

m = _numeric_matrix(matrix)
ann = _feature_annotations(metadata.reset_index(drop=True), require_name=False)
retained = _retained_features(m.columns.astype(str).tolist(), ann, qc_threshold=0.2)
stim_mat = _stimulus_matrix(ds, m.loc[:, retained])
log2_mat = _transform_matrix(stim_mat, transform="log2")
X = log2_mat.to_numpy(dtype=float)  # 106 x 261
labels = log2_mat.index.astype(str).tolist()

print(f"Data: {X.shape[0]} points x {X.shape[1]} dimensions")

# ========================================================================
# 1. CURSE OF DIMENSIONALITY: distance concentration
# ========================================================================
print(f"\n{'='*65}")
print("1. DISTANCE CONCENTRATION (dimension curse check)")
print(f"{'='*65}")

d_full = pdist(X, metric="euclidean")
d_sq = squareform(d_full)
n = X.shape[0]

# Nearest / farthest neighbor ratio per point
nn_ratios = []
for i in range(n):
    d_i = d_sq[i].copy()
    d_i[i] = np.inf
    d_i = d_i[np.isfinite(d_i)]
    if len(d_i) > 1:
        nn_ratios.append(float(np.min(d_i)) / float(np.max(d_i)))

print(f"  Nearest/farthest neighbor ratio (0=good contrast, 1=all same):")
print(f"    mean   = {np.mean(nn_ratios):.4f}")
print(f"    median = {np.median(nn_ratios):.4f}")
print(f"    min    = {np.min(nn_ratios):.4f}")

cv_real = float(np.std(d_full) / np.mean(d_full))
print(f"  Pairwise distance CV: {cv_real:.4f}")

# Compare with random Gaussian data
rng = np.random.default_rng(42)
d_rand = pdist(rng.normal(size=(106, 261)), metric="euclidean")
cv_rand = float(np.std(d_rand) / np.mean(d_rand))
print(f"  Random Gaussian (106 x 261) CV: {cv_rand:.4f}")

# Distance distribution summary
print(f"  Distance percentiles:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"    {p}%: {float(np.percentile(d_full, p)):.2f}")
dyn_range = float(np.max(d_full)) / float(np.min(d_full[d_full > 0]))
print(f"  Dynamic range (max / min_nonzero): {dyn_range:.1f}x")

# ========================================================================
# 2. FEATURE CORRELATION STRUCTURE
# ========================================================================
print(f"\n{'='*65}")
print("2. METABOLITE CORRELATION STRUCTURE")
print(f"{'='*65}")

corr = np.corrcoef(X.T)
iu = np.triu_indices(261, k=1)
offdiag_abs = np.abs(corr[iu])
n_pairs = len(offdiag_abs)

print(f"  Total metabolite pairs: {n_pairs:,}")
for thresh in [0.5, 0.7, 0.9, 0.95]:
    pct = float(np.mean(offdiag_abs > thresh))
    print(f"    |r| > {thresh:.2f}: {pct:.2%} ({int(pct * n_pairs)} pairs)")

# Effective dimensionality of the correlation matrix
eigvals = eigh(corr, eigvals_only=True)
eigvals = eigvals[::-1]
cumvar = np.cumsum(eigvals) / np.sum(eigvals)
print(f"\n  Correlation matrix effective dimensionality:")
for th in [0.5, 0.8, 0.9, 0.95, 0.99]:
    n_comp = int(np.searchsorted(cumvar, th) + 1)
    print(f"    {th*100:.0f}% variance: {n_comp} PCs")

# ========================================================================
# 3. RSA vs NUMBER OF PCs (gradual)
# ========================================================================
print(f"\n{'='*65}")
print("3. RSA vs NUMBER OF PCs (where does the signal live?)")
print(f"{'='*65}")

# Neural RDM
neural_result = build_neural_rdm(
    ds, view="response_window", aggregation="median", merge_lr=True, distance="correlation"
)
n_rdm = neural_result.matrix


def build_rdm_from_features(arr):
    n_pts = arr.shape[0]
    dist = np.full((n_pts, n_pts), np.nan)
    np.fill_diagonal(dist, 0.0)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            valid = np.isfinite(arr[i]) & np.isfinite(arr[j])
            if valid.sum() >= 1:
                d = float(np.linalg.norm(arr[i, valid] - arr[j, valid]))
                dist[i, j] = dist[j, i] = d
    return pd.DataFrame(dist, index=labels, columns=labels)


def rsa_score(c_rdm):
    an, ac = align_square_rdms(n_rdm, c_rdm)
    nn = len(an)
    iu_idx = np.triu_indices(nn, k=1)
    nv, cv = an.to_numpy(float)[iu_idx], ac.to_numpy(float)[iu_idx]
    mask = np.isfinite(nv) & np.isfinite(cv)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(rankdata(nv[mask]), rankdata(cv[mask]))[0, 1])


# Z-score + PCA
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
U, s, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
pca_scores = U * s  # (106, 261)

max_pc = min(105, 261)  # rank limit
pc_list = [1, 2, 3, 5, 10, 15, 20, 30, 50, 80, 100, max_pc]
print(f"  {'PCs':>5}  {'cum var%':>8}  {'RSA':>8}  {'delta':>8}")
print(f"  {'-'*35}")
prev_r = None
for npc in pc_list:
    rdm = build_rdm_from_features(pca_scores[:, :npc])
    r = rsa_score(rdm)
    cv = float(np.cumsum(s**2)[npc - 1] / np.sum(s**2) * 100)
    delta = f"{r - prev_r:+.4f}" if prev_r is not None else ""
    print(f"  {npc:>5}  {cv:>8.1f}  {r:>8.4f}  {delta:>8}")
    prev_r = r

# ========================================================================
# 4. MAHALANOBIS DISTANCE
# ========================================================================
print(f"\n{'='*65}")
print("4. MAHALANOBIS DISTANCE (decorrelates dimensions)")
print(f"{'='*65}")

cov_emp = np.cov(Z.T, ddof=1)
eigvals_cov = eigh(cov_emp, eigvals_only=True)
cond_raw = float(eigvals_cov[-1] / eigvals_cov[0]) if eigvals_cov[0] > 0 else np.inf
print(f"  Covariance condition number (raw): {cond_raw:.1e}")
print(f"  Min eigenvalue: {float(eigvals_cov[0]):.2e}")

# Ridge-regularized Mahalanobis
target = np.mean(np.diag(cov_emp)) * np.eye(cov_emp.shape[0])

for shrinkage in [0.1, 0.3, 0.5, 0.7, 0.9]:
    cov_reg = (1 - shrinkage) * cov_emp + shrinkage * target
    cov_inv = np.linalg.inv(cov_reg + 1e-8 * np.eye(cov_reg.shape[0]))

    dist_m = np.full((n, n), np.nan)
    np.fill_diagonal(dist_m, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            diff = Z[i] - Z[j]
            valid = np.isfinite(diff)
            if valid.sum() < 2:
                continue
            diff_v = diff[valid]
            cov_sub_inv = cov_inv[np.ix_(valid, valid)]
            d = float(np.sqrt(diff_v @ cov_sub_inv @ diff_v))
            dist_m[i, j] = dist_m[j, i] = d

    rdm_m = pd.DataFrame(dist_m, index=labels, columns=labels)
    r_val = rsa_score(rdm_m)
    # Also check distance concentration for Mahalanobis
    d_maha = dist_m[np.triu_indices(n, k=1)]
    d_maha = d_maha[np.isfinite(d_maha)]
    cv_maha = float(np.std(d_maha) / np.mean(d_maha)) if len(d_maha) > 0 else np.nan
    print(f"  shrinkage={shrinkage:.1f}: RSA={r_val:.4f}, dist CV={cv_maha:.4f}")

# ========================================================================
# 5. ALTERNATIVE: Euclidean in PC space that whitens (Mahalanobis ~ PC-weighted)
# ========================================================================
print(f"\n{'='*65}")
print("5. PCA-WHITENED Euclidean (= Mahalanobis in full space)")
print(f"{'='*65}")

# PCA whitening: divide each PC by sqrt(eigenvalue) → spherical
# This is equivalent to Mahalanobis distance (up to regularization)
# Do this with varying numbers of PCs
for npc in [10, 30, 50, 100, max_pc]:
    # Whitened scores
    s_sub = s[:npc]
    whitened = U[:, :npc]  # already unit variance per PC direction
    rdm_w = build_rdm_from_features(whitened)
    r_w = rsa_score(rdm_w)
    print(f"  PCA({npc}) whitened: RSA={r_w:.4f}")
