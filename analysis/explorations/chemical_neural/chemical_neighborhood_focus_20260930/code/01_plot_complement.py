"""Direct chemical-profile / paired-calcium evidence; no fitted classifier.

Reuse the two examples selected by full-panel chemical distance in the prior
round. All 380 log2FCs and all 13 neurons remain visible. Missing report values
are marked, not silently removed or replaced in the primary definition.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT / "reports/population_first_20260930/tables"
CELLS = ["ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH",
         "ASJ", "ASEL", "ASER", "AWCON", "AWCOFF"]
COLORS = ["#227c9d", "#c06c35"]


def fingerprint(path):
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    for directory in ["tables", "figures", "logs"]:
        (OUT / directory).mkdir(exist_ok=True)
    names = ["aligned_chemical_log2fc_paired.csv",
             "aligned_chemical_report_observed_paired.parquet",
             "neural_value_pairs.csv", "neural_value_example_curves.csv",
             "neural_value_example_paired_differences.csv"]
    manifest = [fingerprint(SOURCE / name) for name in names]
    path = OUT / "logs/figure_inputs.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Source files changed"
    else:
        path.write_text(json.dumps(manifest, indent=2))
    chemical = pd.read_csv(SOURCE / names[0], index_col=0)
    observed = pd.read_parquet(SOURCE / names[1])
    pairs = pd.read_csv(SOURCE / names[2])
    curves = pd.read_csv(SOURCE / names[3])
    differences = pd.read_csv(SOURCE / names[4])
    # Selection depends only on chemistry: closest pair and closest same-species
    # pair are the same rank-1/rank-2 examples specified before response review.
    chosen = [pairs.sort_values("chemical_rank").iloc[0],
              pairs[pairs.same_species.eq(1)].sort_values("chemical_rank").iloc[0]]
    assert [int(p.chemical_rank) for p in chosen] == [1, 2]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    chemical_rows, evidence = [], []
    diff_bound = float(differences.response_difference_mean_0_25.abs().max()) * 1.1
    shared_min = float(chemical.loc[["A007", "A010", "A022", "A023"]].min().min()) - .8
    shared_max = float(chemical.loc[["A007", "A010", "A022", "A023"]].max().max()) + .8

    for pair in chosen:
        a, b = pair.strain_a, pair.strain_b
        x, y = chemical.loc[a], chemical.loc[b]
        joint = observed.loc[a] & observed.loc[b]
        one_missing = observed.loc[a] ^ observed.loc[b]
        both_missing = ~(observed.loc[a] | observed.loc[b])
        squared = (x - y)**2
        rms = float(np.sqrt(squared.mean()))
        joint_rms = float(np.sqrt(squared[joint].mean()))
        assert np.isclose(rms, pair.chemical_rms_log2fc, atol=1e-12)
        for feature in chemical.columns:
            chemical_rows.append(dict(pair_id=pair.pair_id, strain_a=a, strain_b=b,
                feature=feature, log2fc_a=x[feature], log2fc_b=y[feature],
                report_observed_a=bool(observed.loc[a, feature]),
                report_observed_b=bool(observed.loc[b, feature])))
        evidence.append(dict(pair_id=pair.pair_id, chemical_rank=int(pair.chemical_rank),
            group_n_strains=int(pair.group_n_strains), rms_all_380=rms,
            n_jointly_reported=int(joint.sum()), rms_jointly_reported=joint_rms,
            n_one_missing=int(one_missing.sum()), n_both_missing=int(both_missing.sum()),
            one_missing_fraction_squared_distance=float(squared[one_missing].sum()/squared.sum())))
        d = differences[differences.pair_id.eq(pair.pair_id)].copy()
        c = curves[curves.pair_id.eq(pair.pair_id)].copy()
        # Independently recover every animal-neuron paired difference from the
        # near-original curve export; averaging time is display-only.
        means = c[c.time.ge(0) & c.time.lt(25)].groupby(
            ["worm_key", "neuron_class", "sample_id"]).response.mean().unstack("sample_id")
        expected = (means[a] - means[b]).dropna()
        actual = d.set_index(["worm_key", "neuron_class"]).response_difference_mean_0_25
        assert np.allclose(actual, expected.reindex(actual.index), atol=1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.8),
                                 gridspec_kw={"width_ratios": [1, 1.23]},
                                 layout="constrained")
        ax = axes[0]
        ax.scatter(x[joint], y[joint], s=17, color="#566c78", alpha=.70,
                   linewidth=0, label=f"Reported in both strains (n={joint.sum()})")
        ax.scatter(x[~joint], y[~joint], s=27, marker="x", color="#c5875b",
                   alpha=.78, linewidth=.9, label=f"Report missing in ≥1 strain (n={(~joint).sum()})")
        ax.plot([shared_min, shared_max], [shared_min, shared_max],
                color=".55", ls="--", lw=.9, zorder=-1)
        ax.set(xlim=(shared_min, shared_max), ylim=(shared_min, shared_max),
               xlabel=f"{a}: chemical log₂FC", ylabel=f"{b}: chemical log₂FC",
               title="Chemical profiles: one point per feature\n"
                     f"RMS: all 380 = {rms:.2f}; jointly reported = {joint_rms:.2f}\n"
                     f"Chemical rank {int(pair.chemical_rank)} / {len(pairs)} comparable pairs")
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax = axes[1]
        labels = []
        for row, cell in enumerate(CELLS):
            cell_data = d[d.neuron_class.eq(cell)].sort_values("worm_key")
            values = cell_data.response_difference_mean_0_25.to_numpy()
            positions = row + np.linspace(-.15, .15, len(values))
            ax.scatter(values, positions, s=31, color=COLORS[0], alpha=.80,
                       edgecolor="white", linewidth=.5, zorder=3)
            ax.plot([values.mean(), values.mean()], [row-.17, row+.17],
                    color="black", lw=2, zorder=4)
            labels.append(f"{cell}  (n={len(values)})")
        ax.axvline(0, color=".6", lw=.9)
        ax.set_yticks(range(13), labels)
        ax.set(ylim=(12.6, -.6), xlim=(-diff_bound, diff_bound),
               xlabel=f"Paired response difference: {a} − {b}\n"
                      "Mean calcium ΔF/F₀ over 0–25 s",
               title="Neural responses: one point per animal")
        ax.grid(axis="y", color=".93", linewidth=.6)
        species = (str(pair.species_a).strip() if pair.same_species else
                   f"{str(pair.species_a).strip()} / {str(pair.species_b).strip()}")
        fig.suptitle(f"What neural measurements add for chemical neighbors: {a} / {b}\n"
                     f"{species}", fontsize=13)
        fig.supxlabel("Chemistry: medium-relative FC; crosses inherit upstream missing-value processing.\n"
                      "Calcium: trials averaged first; black bars = animal means; the same animals contribute across neurons.",
                      fontsize=8)
        stem = OUT / f"figures/chemical_neighbor_{a}_{b}"
        for extension in ["png", "pdf"]:
            fig.savefig(stem.with_suffix("." + extension), dpi=220)
        plt.close(fig)

        # Supporting time courses, stripped of prediction/accuracy annotations.
        fig, axes = plt.subplots(4, 4, figsize=(11.6, 9.0), sharex=True, layout="constrained")
        for i, cell in enumerate(CELLS):
            ax = axes.flat[i]
            for strain, color in zip([a, b], COLORS):
                z = c[c.neuron_class.eq(cell) & c.sample_id.eq(strain)]
                for _, g in z.groupby("worm_key"):
                    ax.plot(g.time, g.response, color=color, alpha=.22, lw=.65)
                z = z.groupby("time").response.mean()
                ax.plot(z.index, z.values, color=color, lw=1.8)
            n = d[d.neuron_class.eq(cell)].worm_key.nunique()
            ax.axhline(0, color=".7", lw=.6)
            ax.axvspan(0, 10, color=".9", zorder=-1)
            ax.set(title=f"{cell} (n={n})", xlim=(-5, 24), xticks=[0, 10, 20])
            if i % 4 == 0:
                ax.set_ylabel("Calcium ΔF/F₀")
            if i >= 9:
                ax.set_xlabel("Time from stimulus onset (s)")
                ax.tick_params(labelbottom=True)
        for i in [13, 14, 15]:
            axes.flat[i].axis("off")
        axes.flat[13].legend(handles=[Line2D([0], [0], color=k, lw=2, label=v)
                                     for k, v in zip(COLORS, [a, b])],
                             loc="upper left", frameon=False)
        axes.flat[14].text(0, .95, "Thin curves: individual animals\nThick curves: animal means\n"
                          "Gray interval: stimulus present\nEach neuron has its own y scale",
                          va="top", fontsize=9)
        fig.suptitle(f"Measured population calcium responses: {a} / {b}\n"
                     "All 13 neurons; no neuron selected by the response difference", fontsize=13)
        stem = OUT / f"figures/response_curves_{a}_{b}"
        for extension in ["png", "pdf"]:
            fig.savefig(stem.with_suffix("." + extension), dpi=180)
        plt.close(fig)
    pd.DataFrame(chemical_rows).to_csv(OUT / "tables/example_chemical_points.csv", index=False)
    pd.DataFrame(evidence).to_csv(OUT / "tables/example_chemical_context.csv", index=False)
    differences.to_csv(OUT / "tables/example_paired_response_points.csv", index=False)
    curves.to_csv(OUT / "tables/example_response_curves.csv", index=False)
    differences.groupby(["pair_id", "neuron_class"]).response_difference_mean_0_25.agg(
        n="size", mean="mean", median="median", minimum="min", maximum="max",
        n_positive=lambda v: (v > 0).sum()).to_csv(OUT / "tables/example_response_summary.csv")
    status = dict(status="passed", selected_pairs=[p.pair_id for p in chosen],
                  selection="Closest pair and closest same-species pair, using the original full 380-feature FC metric",
                  raw_curve_to_display_differences="All 147 animal-neuron points independently recomputed within 1e-12",
                  source_inputs_unchanged=manifest == [fingerprint(SOURCE / name) for name in names],
                  chemical_units="log2 fold-change relative to medium; not exposure concentration",
                  neural_units="Delta F/F0; trial means then [0,25) time mean, same-animal strain difference",
                  randomness="None; plotting offsets deterministic", python=sys.version,
                  platform=platform.platform(),
                  versions={name: importlib.metadata.version(name)
                            for name in ["numpy", "pandas", "matplotlib", "pyarrow"]})
    (OUT / "logs/figure_verification.json").write_text(json.dumps(status, indent=2))
    print(pd.DataFrame(evidence).to_string(index=False))


if __name__ == "__main__":
    main()
