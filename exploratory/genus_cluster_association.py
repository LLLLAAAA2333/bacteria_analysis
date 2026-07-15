"""Is Ward cluster membership associated with bacterial genus/species?

For each neuron: Fisher exact test per genus — is a genus overrepresented
in one Ward cluster?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from bacteria_analysis._data_loaders import enrich_neural_dataframe

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

NEURONS = [("ASH", 2), ("AWA", 2), ("AWCON", 2)]

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


def _ward_labels(mat, k):
    if k <= 1:
        return pd.Series(1, index=mat.index, dtype=int)
    mat_z = mat.subtract(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)
    r = np.corrcoef(mat_z.values)
    d = np.clip(1 - r, 0, None)
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    Z = linkage(squareform(d), method="ward")
    return pd.Series(fcluster(Z, k, criterion="maxclust"), index=mat.index, dtype=int)


# ---------------------------------------------------------------------------

def main():
    raw_neural = pd.read_parquet("data/106bac.parquet")
    enriched = enrich_neural_dataframe(raw_neural)
    stim_info = enriched.groupby("stimulus")[["species", "genus"]].first()

    for neuron, k in NEURONS:
        print(f"\n{'='*65}")
        print(f"  {neuron}  (k={k})")
        print(f"{'='*65}")

        mat = _build_prototype(raw_neural, neuron)
        labels = _ward_labels(mat, k)

        # Build genus × cluster contingency
        df = pd.DataFrame({
            "stimulus": mat.index,
            "cluster": labels.values,
            "genus": [stim_info.loc[s, "genus"] for s in mat.index],
            "species": [stim_info.loc[s, "species"] for s in mat.index],
        })

        # ---- Per-genus Fisher test ----
        genera = df["genus"].value_counts()
        n_total = len(df)

        print(f"\n  {'genus':20s}  {'n':>4s}  {'c1':>4s}  {'c2':>4s}  "
              f"{'%c1':>7s}  {'OR':>7s}  {'p':>8s}")
        print(f"  {'-'*20}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}")

        significant = []
        for genus_name in genera.index:
            sub = df[df["genus"] == genus_name]
            n_genus = len(sub)
            if n_genus < 2:
                continue

            c1_count = (sub["cluster"] == 1).sum()
            c2_count = (sub["cluster"] == 2).sum()

            # 2×2 table: genus vs rest, c1 vs c2
            c1_other = (df["cluster"] == 1).sum() - c1_count
            c2_other = (df["cluster"] == 2).sum() - c2_count
            table = np.array([[c1_count, c2_count],
                              [c1_other, c2_other]])

            try:
                odds_ratio, p_val = fisher_exact(table)
            except ValueError:
                p_val = 1.0
                odds_ratio = np.nan

            pct_c1 = c1_count / n_genus * 100
            line = (f"  {genus_name:20s}  {n_genus:4d}  {c1_count:4d}  "
                    f"{c2_count:4d}  {pct_c1:6.1f}%  "
                    f"{odds_ratio:7.2f}  {p_val:8.4f}")
            print(line)

            if p_val <= 0.05:
                significant.append((genus_name, p_val, odds_ratio, n_genus, pct_c1))

        # ---- Summary ----
        if significant:
            print(f"\n  Significant genera (p <= 0.05, uncorrected):")
            for g, p, or_, n, pct in sorted(significant, key=lambda x: x[1]):
                direction = "> c2" if or_ > 1 else "> c1"
                print(f"    {g}: n={n}, {pct:.0f}% in c1, OR={or_:.2f}, p={p:.4f} "
                      f"({direction})")
        else:
            print(f"\n  No genus significantly associated with cluster")


if __name__ == "__main__":
    main()
