"""Deduplicate metabolites by correlation: keep one representative per redundant group.

Approach:
  1. Load raw fold-change matrix (no log transform).
  2. Z-score each metabolite column.
  3. Compute |Pearson r| between all metabolite pairs.
  4. Hierarchical clustering on 1 - |r| (anti-correlated = also redundant).
  5. Cut at a correlation threshold to form groups.
  6. Per group: keep the metabolite with highest mean |r| to its group members.
  7. Output the reduced raw-FC matrix + group membership.

Result: a set of "independent" metabolites where no pair has |r| > threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

# Correlation thresholds to sweep (within-group |r| >= this)
R_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

OUTPUT_DIR = Path("results/metabolite_modules")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # ---- 1. Load raw fold-change ----
    met_raw = pd.read_excel("data/matrix.xlsx")
    met_raw = met_raw.rename(columns={"Unnamed: 0": "aid"})
    met_raw["aid"] = met_raw["aid"].astype(str)

    met_cols = [c for c in met_raw.columns if c != "aid"]
    met_raw_fc = met_raw[met_cols].values.astype(float)        # (299, 380)
    met_df = pd.DataFrame(met_raw_fc, columns=met_cols, index=met_raw["aid"])
    n_samples, n_mets = met_df.shape
    print(f"Raw FC matrix: {n_samples} samples x {n_mets} metabolites")

    # ---- 2. Z-score ----
    met_z = (met_raw_fc - met_raw_fc.mean(axis=0)) / met_raw_fc.std(axis=0)

    # ---- 3. |r| matrix → distance → clustering ----
    r_mat = np.corrcoef(met_z.T)               # (380, 380)
    abs_r_mat = np.abs(r_mat)                   # anti-correlated = also redundant
    dist_mat = 1 - abs_r_mat
    np.fill_diagonal(dist_mat, 0)

    dist_mat = (dist_mat + dist_mat.T) / 2
    np.fill_diagonal(dist_mat, 0)
    Z = linkage(squareform(dist_mat), method="ward")

    # ---- 4. Sweep thresholds ----
    print(f"\n{'r_thr':>8s}  {'n_groups':>9s}  {'removed':>8s}  "
          f"{'singleton':>10s}  {'max_grp':>8s}  {'kept':>6s}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}")

    for r_thr in R_THRESHOLDS:
        dist_thr = 1 - r_thr
        lbl = fcluster(Z, dist_thr, criterion="distance")
        n_groups = len(set(lbl))
        sizes = pd.Series(lbl).value_counts()
        singletons = (sizes == 1).sum()
        removed = n_mets - n_groups
        print(f"  {r_thr:8.2f}  {n_groups:9d}  {removed:8d}  "
              f"{singletons:10d}  {sizes.max():8d}  {n_mets - removed:6d}")

    # ---- 5. Pick a threshold, select representatives ----
    # Default: r >= 0.85 (remove extreme redundancy while keeping most structure)
    chosen_r = 0.85
    print(f"\nUsing r >= {chosen_r} for deduplication")

    dist_thr = 1 - chosen_r
    labels = fcluster(Z, dist_thr, criterion="distance")
    n_groups = len(set(labels))

    # For each group, pick the most central metabolite (highest mean |r| to group)
    rep_map: dict[int, str] = {}       # group_id -> representative metabolite name
    group_members: dict[int, list[str]] = {}
    removed_cols: list[str] = []

    for grp in sorted(set(labels)):
        idx = np.where(labels == grp)[0]
        member_names = [met_cols[i] for i in idx]
        group_members[grp] = member_names

        if len(idx) == 1:
            rep_map[grp] = member_names[0]
        else:
            # Subset of abs_r for this group
            sub_r = abs_r_mat[np.ix_(idx, idx)]
            mean_r = sub_r.mean(axis=0)  # mean |r| to other group members
            best_local = np.argmax(mean_r)
            rep_map[grp] = member_names[best_local]
            for j, name in enumerate(member_names):
                if j != best_local:
                    removed_cols.append(name)

    kept_cols = list(rep_map.values())
    print(f"  Kept: {len(kept_cols)}  |  Removed: {len(removed_cols)}")

    # ---- 6. Build reduced matrix (raw FC, no log, no z-score) ----
    reduced = met_df[kept_cols].copy()
    reduced.to_parquet(OUTPUT_DIR / "metabolites_reduced.parquet")
    print(f"  -> {OUTPUT_DIR / 'metabolites_reduced.parquet'}")

    # Group membership
    rows = []
    for grp in sorted(group_members.keys()):
        rep = rep_map[grp]
        for met in group_members[grp]:
            rows.append({
                "group": grp,
                "size": len(group_members[grp]),
                "representative": rep,
                "status": "kept" if met == rep else "removed",
                "metabolite": met,
            })
    memb = pd.DataFrame(rows)
    memb.to_csv(OUTPUT_DIR / "metabolite_groups.csv", index=False)
    print(f"  -> {OUTPUT_DIR / 'metabolite_groups.csv'}")

    # ---- 7. Summary of groups with 3+ members ----
    print(f"\nGroups with 3+ members:")
    for grp in sorted(group_members.keys()):
        members = group_members[grp]
        if len(members) >= 3:
            rep = rep_map[grp]
            others = [m for m in members if m != rep]
            print(f"  group {grp} (n={len(members)}): rep = {rep[:50]}")
            for o in others:
                print(f"      - {o[:55]}")

    print(f"\nDone -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
