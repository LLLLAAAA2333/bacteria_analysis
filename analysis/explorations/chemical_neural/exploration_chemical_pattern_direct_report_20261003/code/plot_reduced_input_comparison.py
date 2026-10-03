"""Display the frozen all-neuron versus training-selected subset comparison."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def make_figure(output: Path) -> list[Path]:
    """Plot every held-out strain on identical axes; no fitting or selection."""
    output = Path(output)
    sources = [output / "results.json", output / "heldout_predictions.csv"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    results = json.loads(sources[0].read_text())
    data = pd.read_csv(sources[1])
    assert len(data) == 36 and data.strain.nunique() == 36
    observed = data.observed.to_numpy()
    shown = data[["observed", "all13", "reduced"]].to_numpy()
    assert np.isfinite(shown).all()
    limits = [float(shown.min() - 0.2), float(shown.max() + 0.2)]
    selected = results["selected_neurons"]
    blue, amber = "#356878", "#B67951"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.975, top=0.79, bottom=0.20, wspace=0.17)
        titles = ["All 13 neurons", f"{results['selected_k']} selected neurons"]
        for ax, key, title, color in zip(axes, ("all13", "reduced"), titles, (blue, amber)):
            values = data[key].to_numpy()
            metrics = results["models"][key]
            ax.plot(limits, limits, color="#7C858B", linestyle="--", linewidth=1.2, zorder=1)
            ax.axhline(float(data.baseline.iloc[0]), color="#BCC3C7", linewidth=1, zorder=1)
            ax.scatter(observed, values, s=34, color=color, alpha=0.83,
                       edgecolors="white", linewidths=0.45, zorder=2)
            ax.set(xlim=limits, ylim=limits, aspect="equal", xlabel="Measured chemical score")
            ax.set_title(title, loc="left", color=color, fontsize=15, pad=12)
            gain = metrics["relative_error_reduction"] * 100
            ax.text(0.045, 0.95, f"Error reduction: {gain:.1f}%", transform=ax.transAxes,
                    va="top", fontsize=11)
            lower, upper = metrics["conditional_bootstrap_95pct_interval"]
            ax.text(0.045, 0.894, f"Conditional 95% interval: {lower * 100:.1f}% to {upper * 100:.1f}%",
                    transform=ax.transAxes, va="top", fontsize=9, color="#5C6870")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Predicted chemical score")
        fig.suptitle("Does using fewer neurons improve prediction?", y=0.97, fontsize=19)
        fig.text(0.5, 0.902, "Same 36 strains · same chemical target · internal exploratory check",
                 ha="center", fontsize=11, color="#5C6870")
        legend = [Line2D([0], [0], linestyle="--", color="#7C858B", label="Perfect prediction"),
                  Line2D([0], [0], color="#BCC3C7", label="Training-mean prediction")]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.071),
                   ncol=2, frameon=False, fontsize=10)
        fig.text(0.5, 0.035, "Training-selected subset: " + ", ".join(selected),
                 ha="center", fontsize=10, color="#53616A")
        comparison = results["comparison"]
        direct_gain = comparison["relative_error_reduction_vs_all13"] * 100
        low, high = [v * 100 for v in comparison["conditional_bootstrap_95pct_interval"]]
        fig.text(0.5, 0.002,
                 f"Gain over all 13: {direct_gain:.1f}% lower error · paired conditional 95% interval: {low:.1f}% to {high:.1f}%",
                 ha="center", fontsize=10, color="#53616A")
        outputs = []
        for suffix in (".png", ".svg"):
            path = output / ("reduced_input_comparison" + suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    caption = (
        "Observed and predicted fixed chemical scores for the same 36 held-out strains. "
        "The left model uses all 13 existing neural unit coefficients. The right uses the "
        f"training-selected {results['selected_k']}-neuron subset ({', '.join(selected)}). "
        "Both are Ridge regressions with fold-training feature standardization. The "
        "reduced procedure considered only 3 or 5 neurons, ranked by absolute Pearson "
        "association with the fixed target using each fold's training data. Subset size "
        "and Ridge alpha were selected by pooled five-fold training validation error. "
        "The full model separately selected its alpha from the same fixed grid and folds. "
        "Only the winning reduced procedure was evaluated on held-out data. Each plotted "
        "point is one strain; every strain is included. Both panels have identical "
        "horizontal and vertical scales. The dashed diagonal is exact prediction; the "
        "horizontal line predicts the 70-training-strain chemical mean for every strain.\n\n"
        "Panel error reductions are relative to that same training-mean baseline, not "
        "percent accuracy. Conditional 95% intervals use 5000 paired resamples of the "
        "36 strains with predictions fixed. The direct paired comparison between models "
        "is reported in results.json; comparing interval overlap is not the test of model "
        "difference. These intervals omit target-selection and training uncertainty and "
        "shared-animal/genus dependence. This is a check on previously explored data, "
        "not new independent validation. The chemical target, strain split and existing "
        "neural representation remain fixed. Selected coordinates still use the original "
        "13-neuron normalization; this does not establish that measuring only the selected "
        "neurons would suffice or that excluded neurons lack biological information. "
        "Fold-selection frequencies describe sensitivity within overlapping training "
        "subsamples, not biological importance or independent replication. This one "
        "marginal-correlation screening procedure is not an exhaustive subset search.\n"
    )
    path = output / "figure_caption.txt"
    path.write_text(caption, encoding="utf-8")
    outputs.append(path)
    path = output / "display_parameters.json"
    path.write_text(json.dumps({"source_sha256": hashes, "axis_limits": limits,
                               "same_axes": True, "n_strains_per_panel": len(data),
                               "refitted": False, "subset_selected_for_display": False}, indent=2) + "\n")
    outputs.append(path)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return outputs
