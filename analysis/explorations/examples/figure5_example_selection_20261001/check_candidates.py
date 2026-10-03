"""Focused figure-5 example check from saved outputs; no refit or raw-data changes.

Candidate shortlist comes from the existing 28 chemical-neighbor pairs.
The PNG is a selection diagnostic, not the replacement poster figure.
Run from the repository root with the existing project Python environment.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent
MODEL = REPORTS / "response_structure_20260930"
CELLS = ["AWCON", "ASK", "ADF", "ASJ", "AWA", "AWB", "ASH"]
PAIRS = [("A040", "A041", ["ADF", "ASH", "AWA"]),
         ("A044", "A045", ["ADF", "ASH", "AWB"])]
COLORS = ["#16839C", "#CB7539"]

coeff = pd.read_csv(MODEL / "tables/coefficients.csv", dtype={"block": str})
coeff = coeff.loc[coeff.window.eq("0-40s") & coeff.fit_type.eq("full")]
templates = pd.read_csv(MODEL / "tables/templates.csv")
templates = templates.loc[templates.window.eq("0-40s") & templates.fit_type.eq("full")]

# All rankings below summarize existing scores, not a new neural analysis.
table_root = REPORTS / "poster_neighborhoods_20260930/tables"
neighbors = pd.read_csv(table_root / "poster_neighborhood_rows.csv", dtype={"date": str})
all_pairs = pd.read_csv(table_root / "pair_signal_summary.csv")
context = pd.read_csv(table_root / "pair_context.csv")
for metric in ["energy_raw", "energy_fixed"]:
    neighbors[metric + "_rank_of_28"] = neighbors[metric].rank(ascending=False, method="min")
    ranks = all_pairs.set_index("pair_id")[metric].rank(ascending=False, method="min")
    neighbors[metric + "_rank_of_147"] = neighbors.pair_id.map(ranks)
neighbors = neighbors.merge(context[["pair_id", "joint_reported_n", "joint_reported_rms",
                                     "sequence_mean_abs_gap"]], on="pair_id", validate="one_to_one")
cosines = []
for row in neighbors.itertuples():
    matrix = coeff.loc[coeff.block.eq(row.date) & coeff.strain.isin([row.strain_a, row.strain_b])
                       & coeff.cell.isin(CELLS)].pivot(index="strain", columns="cell", values="coefficient")
    a, b = matrix.reindex(index=[row.strain_a, row.strain_b], columns=CELLS).to_numpy()
    assert np.isfinite([a, b]).all()
    cosines.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
neighbors["coefficient_cosine_7cells_0_40s"] = cosines
neighbors.to_csv(OUT / "existing_28_candidates.csv", index=False)

strains = [s for a, b, _ in PAIRS for s in [a, b]]
curves = pd.read_parquet(REPORTS / "exploration_20260929/tables/animal_curves.parquet",
                         filters=[("sample_id", "in", strains), ("date", "==", "20260429"),
                                  ("neuron_class", "in", ["ADF", "ASH", "AWA", "AWB"])]).reset_index()
keys = ["sample_id", "date", "worm_key", "neuron_class"]
assert not curves.duplicated(keys).any()
long = curves.melt(id_vars=keys, value_vars=[str(i) for i in range(40)],
                   var_name="time_s", value_name="response")
long["time_s"] = long.time_s.astype(int)
long["bin_index"] = long.time_s // 5
bins = long.groupby(keys + ["bin_index"]).response.mean().dropna()
cached = pd.read_parquet(MODEL / "data/observations.parquet", filters=[
    ("sample_id", "in", strains), ("block", "==", "20260429"),
    ("neuron_class", "in", ["ADF", "ASH", "AWA", "AWB"]), ("bin_index", "<", 8)])
cached["date"] = cached.date.astype(str)
expected = cached.set_index(keys + ["bin_index"]).response.dropna()
assert bins.index.equals(expected.sort_index().index)
assert np.allclose(bins, expected.sort_index(), atol=1e-12, rtol=1e-12)
summary = long.groupby(["sample_id", "neuron_class", "time_s"]).response.agg(
    mean="mean", sd="std", n="count").reset_index()
summary["sem"] = summary.sd / np.sqrt(summary.n)
summary.to_csv(OUT / "diagnostic_curve_summary.csv", index=False)

style = {"font.family": "DejaVu Sans", "font.size": 11,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#AAB7C2", "axes.labelcolor": "#20374D",
         "text.color": "#20374D", "xtick.color": "#71808F", "ytick.color": "#71808F"}
with plt.rc_context(style):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    fig.subplots_adjust(left=.08, right=.975, bottom=.09, top=.84, hspace=.67, wspace=.32)
    fig.suptitle("Candidate check: measured curves and saved reconstruction", fontsize=18, x=.08, ha="left", y=.98)
    fig.text(.08, .92, "1-s animal mean ± SEM  ·  Dashed: 5-s template × amplitude  ·  Cell-specific y scales",
             fontsize=10, color="#71808F")
    for row, (a, b, cells) in enumerate(PAIRS):
        fig.text(.08, .875 - row * .45, f"{a} / {b}", fontsize=14, weight="bold")
        for col, cell in enumerate(cells):
            ax = axes[row, col]
            ax.axvspan(0, 10, color="#E7ECF1", alpha=.75, lw=0)
            ax.axhline(0, color="#AAB7C2", lw=.7)
            for strain, color in zip([a, b], COLORS):
                d = summary.loc[summary.sample_id.eq(strain) & summary.neuron_class.eq(cell)].sort_values("time_s")
                assert d.n.nunique() == 1 and d.n.iloc[0] >= 3
                n = int(d.n.iloc[0])
                t = templates.loc[templates.cell.eq(cell)].sort_values("bin_index")
                value = coeff.loc[coeff.strain.eq(strain) & coeff.block.eq("20260429") & coeff.cell.eq(cell), "coefficient"].item()
                ax.fill_between(d.time_s, d["mean"] - d["sem"], d["mean"] + d["sem"], color=color, alpha=.17, lw=0)
                ax.plot(d.time_s, d["mean"], color=color, lw=2.1)
                ax.plot(t.time_s, t.template * value, color=color, lw=1.2, ls=(0, (4, 3)))
            ax.set_title(f"{cell}  ·  n = {n}", fontsize=12, pad=9)
            ax.set(xlim=(0, 40), xticks=[0, 10, 25, 40], xlabel="Time (s)", ylabel="ΔF/F₀")
        axes[row, 2].legend(handles=[Line2D([0], [0], color=c, lw=2, label=s)
                                    for s, c in zip([a, b], COLORS)],
                            loc="upper right", frameon=False, fontsize=9)
    fig.savefig(OUT / "candidate_curve_check.png", dpi=180, facecolor="white")
    plt.close(fig)
print("Saved candidate score table and focused curve diagnostic; cached bins match the 1-s curves.")
