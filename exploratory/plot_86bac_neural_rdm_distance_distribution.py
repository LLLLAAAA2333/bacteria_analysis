from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
NEURAL_RDM = "neural_shape_rdm__active_scaled_flattened_correlation.csv"


def read_distances(root: Path) -> np.ndarray:
    rdm = pd.read_csv(root / "tables" / NEURAL_RDM, index_col=0)
    values = rdm.to_numpy(float)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    return values[np.triu_indices_from(values, k=1)]


def summarize(distances: np.ndarray) -> dict[str, object]:
    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return {
        "n_pairs": int(distances.size),
        "mean_distance": float(distances.mean()),
        "std_distance": float(distances.std(ddof=1)),
        "distance_quantiles": {
            str(q): float(np.quantile(distances, q))
            for q in quantiles
        },
        "correlation_quantiles": {
            str(q): float(np.quantile(1.0 - distances, q))
            for q in quantiles
        },
        "fraction_distance_at_or_below": {
            str(threshold): float(np.mean(distances <= threshold))
            for threshold in thresholds
        },
        "fraction_negative_correlation": float(np.mean(distances > 1.0)),
    }


def plot_distribution(distances: np.ndarray, output_path: Path) -> None:
    sorted_distances = np.sort(distances)
    ecdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].hist(distances, bins=45, color="#4a5568", alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[0].axvline(np.median(distances), color="#dc2626", linewidth=1.6, label=f"median={np.median(distances):.3f}")
    axes[0].axvline(np.mean(distances), color="#2563eb", linewidth=1.4, linestyle="--", label=f"mean={np.mean(distances):.3f}")
    axes[0].set_xlabel("Neural correlation distance (1 - r)", fontsize=9)
    axes[0].set_ylabel("Number of sample pairs", fontsize=9)
    axes[0].set_title("Distance distribution", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(sorted_distances, ecdf, color="#111827", linewidth=1.8)
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        axes[1].axvline(threshold, color="#9ca3af", linewidth=0.8, linestyle=":")
        axes[1].text(
            threshold,
            0.04,
            f"{np.mean(distances <= threshold):.0%}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#4b5563",
        )
    axes[1].set_xlabel("Neural correlation distance (1 - r)", fontsize=9)
    axes[1].set_ylabel("Cumulative fraction", fontsize=9)
    axes[1].set_title("ECDF", fontsize=10)
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.grid(True, color="#e5e7eb", linewidth=0.6)
        axis.tick_params(labelsize=8)
    figure.suptitle("86bac neural RDM distance distribution", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run(root: Path) -> dict[str, object]:
    distances = read_distances(root)
    summary = summarize(distances)
    output_dir = root / "tables" / "mds_diagnostics"
    figures_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "neural_rdm_distance_distribution_summary.json"
    figure_path = figures_dir / "neural_rdm_distance_distribution.png"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_distribution(distances, figure_path)
    return {
        "summary": summary,
        "outputs": {
            "summary": str(summary_path),
            "figure": str(figure_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the 86bac neural RDM correlation-distance distribution.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))


if __name__ == "__main__":
    main()
