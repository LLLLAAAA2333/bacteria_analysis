"""Review existing chemical comparisons before examining neural effects.

Uses A022/A023 as a descriptive benchmark, not an equivalence threshold.
The shortlist is fixed by chemistry alone; no refit or new chemical processing.
Run from the repository root using the existing project Python environment.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
audit = pd.read_csv(REPORTS / "chemical_neighborhood_focus_20260930/tables/chemical_neighbor_audit_pairs.csv")
baseline = audit.loc[audit.pair_id.eq("20260601_A022_A023")].iloc[0]
shortlist = audit.loc[audit.joint_rms_log2fc.lt(baseline.joint_rms_log2fc)].sort_values("joint_rms_log2fc").copy()
assert set(shortlist.pair_id) == {"20260311_A231_A232", "20260520_A001_A002", "20260520_A005_A006",
                                  "20260311_A235_A236", "20260311_A237_A241"}
# Verify that lower RMS has not traded off the other available closeness summaries.
assert shortlist.joint_median_abs_log2fc_difference.le(baseline.joint_median_abs_log2fc_difference).all()
for name in ["joint_fraction_abs_difference_le_1", "joint_fraction_abs_difference_le_2", "joint_profile_pearson"]:
    assert shortlist[name].ge(baseline[name]).all()
shortlist.to_csv(OUT / "chemical_shortlist_before_neural_review.csv", index=False)
shown = pd.concat([baseline.to_frame().T, shortlist], ignore_index=True)
chemical = REPORTS / "population_first_20260930/tables"
fc = pd.read_csv(chemical / "aligned_chemical_log2fc_all.csv", index_col=0)
mask = pd.read_parquet(chemical / "aligned_chemical_report_observed_all.parquet")
raw = pd.read_csv(chemical / "aligned_chemical_report_values_all.csv", index_col=0)
strains = list(set(shown.strain_a) | set(shown.strain_b))
assert mask.loc[strains].equals(raw.loc[strains].notna())
values = []
for row in shown.itertuples():
    joint = mask.loc[row.strain_a] & mask.loc[row.strain_b]
    a, b = fc.loc[row.strain_a, joint], fc.loc[row.strain_b, joint]
    assert joint.sum() == row.n_joint_reported
    assert np.isclose(np.sqrt(np.mean((a-b)**2)), row.joint_rms_log2fc, atol=1e-12)
    values.append(pd.DataFrame({"pair_id": row.pair_id, "feature": a.index,
                                "a_log2fc": a.to_numpy(), "b_log2fc": b.to_numpy()}))
plotted = pd.concat(values, ignore_index=True)
plotted.to_csv(OUT / "chemical_review_scatter_values.csv", index=False)
low = np.floor(plotted[["a_log2fc", "b_log2fc"]].min().min()/5)*5
high = np.ceil(plotted[["a_log2fc", "b_log2fc"]].max().max()/5)*5
style = {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#AAB7C2", "text.color": "#20374D", "axes.labelcolor": "#20374D",
         "xtick.color": "#71808F", "ytick.color": "#71808F"}
with plt.rc_context(style):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.8))
    fig.subplots_adjust(left=.06, right=.985, bottom=.075, top=.89, wspace=.25, hspace=.41)
    fig.suptitle("Chemistry-first candidate review", x=.06, y=.975, ha="left", fontsize=19)
    fig.text(.06, .932, "Identical axes  ·  Jointly reported features  ·  A022/A023 retained as the benchmark", color="#71808F")
    for ax, row in zip(axes.ravel(), shown.itertuples()):
        d = plotted.loc[plotted.pair_id.eq(row.pair_id)]
        ax.plot([low, high], [low, high], color="#A7B4BF", lw=1)
        ax.scatter(d.a_log2fc, d.b_log2fc, s=12, color="#536F86", alpha=.65, edgecolors="white", linewidths=.25)
        ax.set(xlim=(low, high), ylim=(low, high), aspect="equal",
               xlabel=row.strain_a + " log₂FC", ylabel=row.strain_b + " log₂FC")
        ax.set_title(f"{row.strain_a} / {row.strain_b}  ·  RMS {row.joint_rms_log2fc:.3f}", fontsize=11, pad=9)
        ax.xaxis.label.set_color("#16839C")
        ax.yaxis.label.set_color("#CB7539")
    fig.savefig(OUT / "chemical_shortlist_review.png", dpi=180, facecolor="white")
    plt.close(fig)
(OUT / "selection_parameters.json").write_text(json.dumps({
    "benchmark": str(baseline.pair_id),
    "input_scope": "147 already-audited same-date, same-genus, same-reference pairs",
    "shortlist_rule": "Joint-report RMS below A022/A023; then verify median, fractions within 1 and 2 log2FC, and profile r are no worse. No neural outcome used in this shortlist.",
    "chemical_equivalence_threshold": None,
    "pair_specific_feature_masks": True,
    "two_strain_groups": "Allowed on direct chemical closeness evidence; automatic nearest-neighbor status supplies no additional evidence.",
    "scientific_limits": "Separate cultures, not stimulus aliquots; no chemical biological-replicate uncertainty; common-reference profile r is descriptive, not independent feature replication.",
    "plot_axis_limits": [float(low), float(high)],
    "n_shortlisted": len(shortlist),
}, indent=2) + "\n")
print(shortlist[["pair_id", "n_joint_reported", "joint_rms_log2fc", "joint_fraction_abs_difference_le_1"]].to_string(index=False))
