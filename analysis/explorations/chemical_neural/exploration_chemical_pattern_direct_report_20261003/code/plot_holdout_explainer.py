"""Explain the saved train/holdout prediction; do not select or fit a model."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


def make_figure(out: Path) -> list[Path]:
    """Draw the data flow and verify the full-vector errors from frozen parameters."""
    out = Path(out)
    paths = [out / "frozen_candidate.json", out / "results.json",
             out / "tables/sample_scores.csv", out / "tables/neural_unit_coefficients.csv"]
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frozen, results = [json.loads(p.read_text()) for p in paths[:2]]
    scores = pd.read_csv(paths[2], index_col="strain")
    neural = pd.read_csv(paths[3], index_col=0)
    train, held = frozen["discovery_ids"], frozen["holdout_ids"]
    assert len(train) == 70 and len(held) == 36 and not set(train).intersection(held)
    assert len(neural.columns) == 13 and set(train + held) == set(neural.index)
    beta = pd.Series(frozen["neural_beta"]).reindex(neural.columns).to_numpy()
    intercept = pd.Series(frozen["neural_intercept"]).reindex(neural.columns).to_numpy()
    predicted = scores.loc[held, "chemical_score"].to_numpy()[:, None] * beta + intercept
    observed = neural.loc[held].to_numpy()
    baseline = neural.loc[train].mean().to_numpy()
    model_sse = float(np.square(predicted - observed).sum())
    baseline_sse = float(np.square(baseline - observed).sum())
    improvement = 1 - model_sse / baseline_sse
    np.testing.assert_allclose(
        [model_sse, baseline_sse, improvement],
        [results["holdout_model_sse"], results["holdout_mean_baseline_sse"],
         results["holdout_vector_improvement"]], rtol=1e-12, atol=1e-14,
    )

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(12, 8.8), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        blue, amber, ink = "#356878", "#B67951", "#233B45"

        def box(x, y, w, h, text, face, edge, fontsize=12):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.012",
                facecolor=face, edgecolor=edge, linewidth=1.3,
            ))
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fontsize, color=ink, linespacing=1.65)

        def arrow(start, end, color=blue, style="solid"):
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=15,
                linewidth=1.6, color=color, linestyle=style,
            ))

        ax.text(0.5, 0.963, "How chemistry predicts the neural profile",
                ha="center", fontsize=21, color=ink)
        ax.text(0.055, 0.891, "1  Learn from 70 strains", color=blue,
                fontsize=14, fontweight="bold")
        box(0.055, 0.742, 0.275, 0.11, "Chemical measurements\n+ 13-neuron profiles", "#ECF3F5", blue)
        box(0.418, 0.742, 0.295, 0.11, "Select chemical pattern\nFit 13-neuron mapping", "#ECF3F5", blue)
        box(0.801, 0.742, 0.144, 0.11, "Freeze\nscore + mapping", "#E2EEF1", blue, 11)
        arrow((0.34, 0.797), (0.408, 0.797))
        arrow((0.723, 0.797), (0.791, 0.797))

        ax.text(0.055, 0.666, "2  Predict 36 held-out strains", color=blue,
                fontsize=14, fontweight="bold")
        box(0.055, 0.514, 0.275, 0.11, "Chemical measurements\nonly", "#ECF3F5", blue)
        box(0.418, 0.514, 0.295, 0.11, "Apply the fixed rule\nŷ = a + b × chemical score", "#E2EEF1", blue)
        box(0.801, 0.514, 0.144, 0.11, "13 predicted\ncoefficients", "#ECF3F5", blue, 11)
        arrow((0.34, 0.569), (0.408, 0.569))
        arrow((0.723, 0.569), (0.791, 0.569))
        # The frozen rule, not held-out neural measurements, feeds prediction.
        ax.plot([0.873, 0.873, 0.566], [0.734, 0.697, 0.697],
                color=blue, linewidth=1.3, linestyle=(0, (3, 3)))
        arrow((0.566, 0.697), (0.566, 0.634), style="dashed")

        ax.text(0.055, 0.438, "3  Compare with measurements", color=amber,
                fontsize=14, fontweight="bold")
        box(0.055, 0.286, 0.275, 0.11, "Measured neural coefficients\nSame 36 held-out strains", "#FAF0E9", amber)
        box(0.418, 0.286, 0.295, 0.11, "Compare all 36 × 13 values\nSum squared prediction errors", "#F5F4F1", "#8D9699", 11.5)
        arrow((0.34, 0.341), (0.408, 0.341), color=amber)
        ax.plot([0.873, 0.873], [0.506, 0.341], color=blue, linewidth=1.6)
        arrow((0.873, 0.341), (0.723, 0.341))

        err_ax = fig.add_axes([0.275, 0.098, 0.44, 0.108])
        values = [baseline_sse, model_sse]
        err_ax.barh([1, 0], values, color=["#C2C9CC", blue], height=0.43)
        err_ax.set_yticks([1, 0], ["70-strain mean", "Chemical prediction"], fontsize=11)
        err_ax.set_xlim(0, 18.5)
        err_ax.set_ylim(-0.55, 1.55)
        err_ax.set_xticks([0, 5, 10, 15])
        err_ax.set_xlabel("Squared prediction error · lower is better", fontsize=10, labelpad=7)
        err_ax.tick_params(axis="y", length=0, pad=10)
        err_ax.tick_params(axis="x", length=3, colors="#5C6870", labelsize=9)
        err_ax.spines[["top", "right", "left"]].set_visible(False)
        err_ax.spines["bottom"].set_color("#ADB5B9")
        for y, value in zip([1, 0], values):
            err_ax.text(value + 0.2, y, f"{value:.2f}", va="center", fontsize=10, color=ink)
        ax.text(0.83, 0.159, f"{100 * improvement:.1f}%", fontsize=25,
                fontweight="bold", ha="center", color=blue)
        ax.text(0.83, 0.122, "lower error", fontsize=12, ha="center", color=ink)
        ax.text(0.5, 0.023, "Internal split of previously explored data; existing neural templates reused.",
                ha="center", fontsize=10, color="#5C6870")

        figures = out / "figures"
        figures.mkdir(exist_ok=True)
        outputs = []
        for suffix in (".png", ".svg"):
            path = figures / ("06_holdout_prediction_explained" + suffix)
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            outputs.append(path)
        plt.close(fig)

    caption = (
        "Held-out prediction in the existing direct-report exploration. In the current "
        "70/36 split, chemical candidate selection and fitting of the chemical score and "
        "13-coefficient regression use the 70 discovery strains. The selected annotations "
        "are Glucaric acid, Lumichrome and Vitamin B1. The score uses discovery log-concentration "
        "means, SDs and fixed weights; these are not re-estimated on held-out strains. "
        "For each of the 36 held-out strains, chemistry supplies a score s and the saved "
        "model predicts the full vector as y_hat = a + b*s. Here a and b each contain 13 "
        "fixed coefficients. Predictions are not normalized to unit length again. Held-out "
        "neural measurements are used to assess predictions, not to refit this mapping.\n\n"
        "Both errors use the same 36 strains and all 13 observed unit template coefficients. "
        "The baseline predicts every held-out strain with the same 13-vector: the mean "
        "neural profile across the 70 discovery strains. It is a neural prediction baseline, "
        "not an old chemical reference group or fold-change denominator. Model SSE = "
        f"{model_sse:.12f}; baseline SSE = {baseline_sse:.12f}. Relative improvement is "
        f"1 - model SSE / baseline SSE = {100 * improvement:.6f}%. This is error reduction, "
        "not accuracy, a percentage of correctly predicted strains, or an independent "
        "variance-explained estimate. Bar axes start at zero. No neuron or held-out strain "
        "is selected for this calculation.\n\n"
        "This explains a frozen-model internal exploratory check on previously explored "
        "data, not a new independent validation experiment. Neural templates and SNR "
        "representations were reused; chemical feature completeness used the full panel. "
        "The whole 106-strain grouped-mean presentation does not enter this prediction. "
        "This figure replays the saved prediction arithmetic to verify existing errors; "
        "it does not fit or select a new model.\n"
    )
    path = figures / "holdout_prediction_explained_caption.txt"
    path.write_text(caption, encoding="utf-8")
    outputs.append(path)
    audit = {"source_sha256": hashes, "n_training": len(train), "n_held_out": len(held),
             "n_neurons": len(neural.columns), "prediction_equation": "a + b * chemical_score",
             "prediction_renormalized": False, "refitted": False,
             "model_sse": model_sse, "training_mean_baseline_sse": baseline_sse,
             "relative_error_reduction": improvement, "checked_against_saved_results": True}
    path = figures / "holdout_prediction_explained_parameters.json"
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    outputs.append(path)
    assert hashes == {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return outputs
