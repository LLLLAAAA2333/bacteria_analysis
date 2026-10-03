"""Focused A231/A232 diagnostic after chemistry-only shortlisting.

No new fitting. Raw 1-s animal means/SEM and the saved 5-s model are shown.
This check does not replace the current poster PDF.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL = REPORTS / "response_structure_20260930"
STRAINS = ["A231", "A232"]
CELLS = ["AWB", "ASH", "AWA"]
BLOCK = "20260311"
COLORS = ["#16839C", "#CB7539"]
wide = pd.read_parquet(REPORTS / "exploration_20260929/tables/animal_curves.parquet", filters=[
    ("sample_id", "in", STRAINS), ("date", "==", BLOCK), ("neuron_class", "in", CELLS)]).reset_index()
keys = ["sample_id", "date", "worm_key", "neuron_class"]
assert not wide.duplicated(keys).any()
curves = wide.melt(id_vars=keys, value_vars=[str(i) for i in range(40)], var_name="time_s", value_name="response")
curves["time_s"] = curves.time_s.astype(int)
curves["bin_index"] = curves.time_s // 5
assert curves.groupby(keys).response.count().isin([0, 40]).all()
cached = pd.read_parquet(MODEL / "data/observations.parquet", filters=[
    ("sample_id", "in", STRAINS), ("block", "==", BLOCK), ("neuron_class", "in", CELLS), ("bin_index", "<", 8)])
cached["date"] = cached.date.astype(str)
expected = cached.set_index(keys + ["bin_index"]).response.dropna().sort_index()
actual = curves.groupby(keys + ["bin_index"]).response.mean().dropna().sort_index()
assert actual.index.equals(expected.index)
assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
summary = curves.groupby(["sample_id", "neuron_class", "time_s"]).response.agg(mean="mean", sd="std", n="count").reset_index()
summary["sem"] = summary.sd / np.sqrt(summary.n)
summary.to_csv(OUT / "a231_a232_curve_summary.csv", index=False)
coeff = pd.read_csv(MODEL / "tables/coefficients.csv", dtype={"block": str})
coeff = coeff.loc[coeff.window.eq("0-40s") & coeff.fit_type.eq("full") & coeff.block.eq(BLOCK) & coeff.strain.isin(STRAINS)]
coeff.to_csv(OUT / "a231_a232_saved_amplitudes.csv", index=False)
templates = pd.read_csv(MODEL / "tables/templates.csv")
templates = templates.loc[templates.window.eq("0-40s") & templates.fit_type.eq("full")]
chem = pd.read_csv(OUT / "chemical_review_scatter_values.csv")
chem = chem.loc[chem.pair_id.eq("20260311_A231_A232")]
style = {"font.family": "DejaVu Sans", "font.size": 10,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#AAB7C2", "text.color": "#20374D", "axes.labelcolor": "#20374D",
         "xtick.color": "#71808F", "ytick.color": "#71808F"}
with plt.rc_context(style):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.5))
    fig.subplots_adjust(left=.055, right=.985, bottom=.18, top=.72, wspace=.39)
    fig.suptitle("A231 / A232 candidate check", x=.055, y=.955, ha="left", fontsize=19)
    fig.text(.055, .86, "Joint-report chemistry  ·  1-s animal mean ± SEM  ·  Dashed: saved template × amplitude", color="#71808F")
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2.5, label=s) for s, c in zip(STRAINS, COLORS)],
               loc="upper right", bbox_to_anchor=(.985, .96), ncol=2, frameon=False, fontsize=11)
    ax = axes[0]
    ax.plot([-10, 25], [-10, 25], color="#A7B4BF", lw=1)
    ax.scatter(chem.a_log2fc, chem.b_log2fc, s=13, color="#536F86", alpha=.65, edgecolors="white", linewidths=.25)
    ax.set(xlim=(-10, 25), ylim=(-10, 25), aspect="equal", xlabel="A231 log₂FC", ylabel="A232 log₂FC", title="Reference chemistry  ·  326 features")
    ax.xaxis.label.set_color(COLORS[0]); ax.yaxis.label.set_color(COLORS[1])
    for ax, cell in zip(axes[1:], CELLS):
        ax.axvspan(0, 10, color="#E7ECF1", alpha=.75, lw=0)
        ax.axhline(0, color="#AAB7C2", lw=.7)
        for strain, color in zip(STRAINS, COLORS):
            d = summary.loc[summary.sample_id.eq(strain) & summary.neuron_class.eq(cell)].sort_values("time_s")
            assert d.n.nunique() == 1 and d.n.iloc[0] >= 3
            n = int(d.n.iloc[0])
            t = templates.loc[templates.cell.eq(cell)].sort_values("bin_index")
            a = coeff.loc[coeff.strain.eq(strain) & coeff.cell.eq(cell), "coefficient"].item()
            meanbins = actual.xs((strain, BLOCK), level=("sample_id", "date")).groupby(["neuron_class", "bin_index"]).mean().loc[cell]
            assert np.isclose(np.mean(meanbins.to_numpy() * t.template.to_numpy()), a, atol=1e-12)
            ax.fill_between(d.time_s, d["mean"]-d["sem"], d["mean"]+d["sem"], color=color, alpha=.17, lw=0)
            ax.plot(t.time_s, t.template*a, color=color, lw=1.2, ls=(0, (4, 3)), alpha=.75)
            ax.plot(d.time_s, d["mean"], color=color, lw=2.1)
        ax.set(title=f"{cell}  ·  n = {n}", xlim=(0, 40), xticks=[0, 10, 40], xlabel="Time (s)", ylabel="ΔF/F₀")
    fig.text(.985, .035, "Cell-specific y scales; full-data model reconstruction", ha="right", color="#71808F", fontsize=9)
    fig.savefig(OUT / "a231_a232_candidate_check.png", dpi=180, facecolor="white")
    plt.close(fig)
print("A231/A232 diagnostic saved; selected 1-s curves match cached bins and saved amplitude projections.")
