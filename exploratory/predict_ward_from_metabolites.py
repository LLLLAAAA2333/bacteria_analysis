"""Can metabolite profiles predict Ward cluster labels?

PCA (20 PCs) + L2-logistic regression with LOOCV.
Permutation test for significance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

N_PCS = 20          # 80% variance
N_PERM = 500        # permutation test shuffles

NEURONS = [
    ("ASH", 2), ("AWA", 2), ("AWCON", 2),
]

_LR_MERGE = {
    "ADF": ("ADFL", "ADFR"), "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"), "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"), "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"), "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}


def _build_prototype(raw, neuron):
    merged = _LR_MERGE.get(neuron)
    raw_names = list(merged) if merged else [neuron]
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

def main():
    raw_neural = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw_neural)
    stim_info = enriched.groupby("stimulus")["aid"].first()
    stim_to_aid = stim_info.to_dict()

    met_df = pd.read_parquet("results/metabolite_modules/metabolites_reduced.parquet")
    met_log2 = np.log2(met_df.values.astype(float))
    met_cols = list(met_df.columns)
    aid_to_idx = {aid: i for i, aid in enumerate(met_df.index)}

    for neuron, k in NEURONS:
        print(f"\n{'='*60}")
        print(f"  {neuron}  (k={k})")
        print(f"{'='*60}")

        mat = _build_prototype(raw_neural, neuron)
        labels = _ward_labels(mat, k)

        valid_stim = [s for s in mat.index if stim_to_aid[s] in aid_to_idx]
        X = np.array([met_log2[aid_to_idx[stim_to_aid[s]]] for s in valid_stim])
        y = labels.loc[valid_stim].values

        n = len(y)
        count_a = (y == 1).sum()
        count_b = (y == 2).sum()
        baseline = max(count_a, count_b) / n
        print(f"  n={n}  (c1={count_a}, c2={count_b}), chance baseline={baseline:.1%}")

        # ---- PCA + standardise ----
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=N_PCS)
        X_pca = pca.fit_transform(X_scaled)
        print(f"  PCA: {X.shape[1]} metabolites -> {N_PCS} PCs  "
              f"({pca.explained_variance_ratio_.sum():.0%} variance)")

        # ---- LOOCV with L2-logistic ----
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        loo = LeaveOneOut()
        correct = 0
        for train_idx, test_idx in loo.split(X_pca):
            lr.fit(X_pca[train_idx], y[train_idx])
            if lr.predict(X_pca[test_idx])[0] == y[test_idx[0]]:
                correct += 1
        acc = correct / n
        print(f"  LOOCV accuracy: {acc:.3f} ({correct}/{n})")

        # ---- Permutation test ----
        rng = np.random.default_rng(42)
        perm_acc = np.zeros(N_PERM)
        for p in range(N_PERM):
            y_shuf = rng.permutation(y)
            corr_p = 0
            for train_idx, test_idx in loo.split(X_pca):
                lr.fit(X_pca[train_idx], y_shuf[train_idx])
                if lr.predict(X_pca[test_idx])[0] == y_shuf[test_idx[0]]:
                    corr_p += 1
            perm_acc[p] = corr_p / n

        p_perm = np.mean(perm_acc >= acc)
        print(f"  Permutation ({N_PERM} shuffles): null mean={perm_acc.mean():.3f} "
              f"+/-{perm_acc.std():.3f}, p={p_perm:.4f}")

        if acc > baseline and p_perm <= 0.05:
            print(f"  >>> ABOVE CHANCE <<<")
            # Show top PC loadings
            for pc_i in range(min(3, N_PCS)):
                top_idx = np.argsort(-np.abs(pca.components_[pc_i]))[:5]
                top_mets = [met_cols[i][:40] for i in top_idx]
                top_w = [pca.components_[pc_i][i] for i in top_idx]
                print(f"  PC{pc_i+1} top loadings: " +
                      ", ".join(f"{m} ({w:+.3f})" for m, w in zip(top_mets, top_w)))
        else:
            print(f"  --- not significant ---")


if __name__ == "__main__":
    main()
