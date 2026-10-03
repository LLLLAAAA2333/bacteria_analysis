"""Display the pre-specified linear, interaction and additive-curve check."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def make_figures(output: Path) -> list[Path]:
    """Render saved results only: equal-axis predictions and paired comparisons."""
    output = Path(output)
    sources = [output / "results.json", output / "heldout_predictions.csv"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = json.loads(sources[0].read_text())
    data = pd.read_csv(sources[1])
    assert len(data) == 36 and data.strain.nunique() == 36
    kinds = ["linear", "interaction", "curvature"]
    titles = ["Linear", "Pairwise interactions", "Additive curves"]
    colors = ["#356878", "#B67951", "#7D6596"]
    values = data[["observed"] + kinds].to_numpy()
    assert np.isfinite(values).all()
    limits = [float(values.min() - 0.2), float(values.max() + 0.2)]
    outputs = []

    def save(fig, stem):
        for suffix in (".png", ".svg"):
            path = output / (stem + suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.79, bottom=0.235, wspace=0.13)
        for ax, kind, title, color in zip(axes, kinds, titles, colors):
            metric = result["models"][kind]
            ax.plot(limits, limits, linestyle="--", color="#7C858B", linewidth=1.2)
            ax.axhline(float(data.baseline.iloc[0]), color="#BCC3C7", linewidth=1)
            ax.scatter(data.observed, data[kind], color=color, s=31, alpha=0.83,
                       edgecolors="white", linewidths=0.45, zorder=3)
            ax.set(xlim=limits, ylim=limits, aspect="equal", xlabel="Measured chemical score")
            ax.set_title(title, loc="left", fontsize=14, pad=12, color=color)
            gain = metric["relative_error_reduction"] * 100
            ax.text(0.045, 0.955, f"Error reduction: {gain:.1f}%", transform=ax.transAxes,
                    fontsize=10.5, va="top")
            ax.text(0.045, 0.895, f"r = {metric['pearson_r']:.2f}", transform=ax.transAxes,
                    fontsize=10, va="top", color="#5C6870")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Predicted chemical score")
        fig.suptitle("Do interactions or curves improve prediction?", y=0.975, fontsize=20)
        fig.text(0.5, 0.909, "Same 36 strains · same chemical score · five selected neural inputs",
                 ha="center", fontsize=11, color="#5C6870")
        legend = [Line2D([0], [0], linestyle="--", color="#7C858B", label="Perfect prediction"),
                  Line2D([0], [0], color="#BCC3C7", label="Training-mean prediction")]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.117),
                   frameon=False, ncol=2, fontsize=10)
        for y, kind, title in zip((0.082, 0.045), kinds[1:], titles[1:]):
            direct = result["comparisons"][kind]
            gain = 100 * direct["relative_error_reduction_vs_linear"]
            lower, upper = [100 * v for v in direct["conditional_bootstrap_95pct_interval"]]
            fig.text(0.5, y,
                     f"{title} vs linear: {gain:+.1f}% error reduction · conditional 95% interval: {lower:.1f}% to {upper:.1f}%",
                     ha="center", fontsize=10, color="#53616A")
        fig.text(0.5, 0.007, "Previously explored data · all transformations and tuning use training data only",
                 ha="center", fontsize=9, color="#68757B")
        save(fig, "nonlinear_prediction_comparison")

        # Direct paired error changes answer whether an extension adds information.
        fig, ax = plt.subplots(figsize=(9, 4.1))
        ax.axvline(0, color="#939CA1", linewidth=1.2, zorder=1)
        endpoints = [0.0]
        for y, kind, color in zip((1, 0), kinds[1:], colors[1:]):
            direct = result["comparisons"][kind]
            gain = 100 * direct["relative_error_reduction_vs_linear"]
            lower, upper = [100 * v for v in direct["conditional_bootstrap_95pct_interval"]]
            # Draw percentile intervals directly: they need not enclose the point estimate.
            ax.hlines(y, lower, upper, color=color, linewidth=2.5, zorder=2)
            ax.scatter([gain], [y], color=color, s=70, zorder=3)
            ax.text(gain, y + 0.19, f"{gain:+.1f}%", color=color, ha="center", fontsize=12)
            endpoints.extend([lower, upper, gain])
        span = max(max(endpoints) - min(endpoints), 10)
        ax.set_xlim(min(endpoints) - span * 0.08, max(endpoints) + span * 0.08)
        ax.set_ylim(-0.6, 1.7)
        ax.set_yticks([1, 0], titles[1:])
        ax.set_xlabel("Prediction error reduction vs linear (%)", labelpad=11)
        ax.set_title("Additional predictive value", loc="left", fontsize=17, pad=28)
        ax.text(0, 1.035, "Same 36 strains · paired conditional 95% intervals",
                transform=ax.transAxes, fontsize=10, color="#5C6870")
        ax.tick_params(axis="y", length=0, pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#A9B1B6")
        fig.tight_layout()
        save(fig, "nonlinear_paired_error_comparison")

    caption = (
        "Pre-specified comparison of five-input linear Ridge regression, a pairwise-interaction "
        "extension and additive spline Ridge regression for the fixed chemical score. Each "
        "training fold reselects five neurons using absolute marginal Pearson correlation. "
        "All input scaling, feature transformations, spline knots and output-feature scaling "
        "are fitted on the fold-training strains. The interaction model retains five main "
        "effects and ten pairwise products, without squared terms. The curve model uses "
        "univariate quadratic B-splines with three uniformly spaced training-range knots "
        "per input, no cross-neuron interactions and linear boundary extrapolation. It is "
        "a low-complexity additive spline model with coefficient shrinkage, not a "
        "roughness-penalized GAM. Each family selects one Ridge alpha from the same "
        "six-value grid using five-fold training validation error. All three fitted "
        "models are frozen before this held-out evaluation. The training-preferred "
        f"family is {result['training_chosen_family']}; no family is selected by held-out error.\n\n"
        "The prediction panels include all 36 strains on identical axes. The dashed line "
        "is exact prediction, and the horizontal line predicts the training mean chemical "
        "score for every strain. Panel error reduction uses this shared mean baseline. "
        "The paired comparison instead measures 1 - extension SSE / linear SSE, so a "
        "positive value indicates lower error than the linear model. Intervals are "
        "percentile intervals from 5000 paired resamples of the held-out strains with "
        "all predictions fixed. They omit training/selection uncertainty and shared "
        "genus/animal dependence; two exploratory comparisons do not establish a formal "
        "confirmatory significance result.\n\n"
        "The same held-out data have already been explored. The chemical module, score, "
        "neural representation and input count were previously chosen. Training CV here "
        "only tunes the current models; it is not an unbiased evaluation of the entire "
        "discovery pipeline. The selected unit coefficients retain the normalization "
        "based on all 13 neurons. Marginal screening can omit variables with pure "
        "interaction information, so a negative result does not rule out nonlinear "
        "information in the full representation. Expanding features also changes the "
        "Ridge penalty geometry; predictive gains do not demonstrate a neural mechanism.\n"
    )
    path = output / "figure_caption.txt"
    path.write_text(caption, encoding="utf-8")
    outputs.append(path)
    path = output / "display_parameters.json"
    path.write_text(json.dumps({"source_sha256": hashes, "prediction_axis_limits": limits,
                               "same_prediction_axes": True, "n_strains_per_panel": len(data),
                               "refitted": False, "neuron_order": result["selected_neurons"]}, indent=2) + "\n")
    outputs.append(path)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return outputs
