from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
AUDIT_TABLE = "chemical_feature_type_audit/matrix_chemical_feature_type_audit.csv"


def load_mass_table(root: Path) -> pd.DataFrame:
    table = pd.read_csv(root / "tables" / AUDIT_TABLE)
    table["mass_first"] = pd.to_numeric(table["mass_first"], errors="coerce")
    table = table[np.isfinite(table["mass_first"])].copy()
    if table.empty:
        raise ValueError("No finite mass_first values found")
    return table


def summarize(table: pd.DataFrame) -> dict[str, object]:
    mass = table["mass_first"].to_numpy(float)
    thresholds = [300, 500, 700]
    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return {
        "n_features": int(len(table)),
        "mean_mass": float(np.mean(mass)),
        "std_mass": float(np.std(mass, ddof=1)),
        "mass_quantiles": {str(q): float(np.quantile(mass, q)) for q in quantiles},
        "threshold_counts": {
            f"gt_{threshold}": int(np.sum(mass > threshold))
            for threshold in thresholds
        },
        "threshold_fractions": {
            f"gt_{threshold}": float(np.mean(mass > threshold))
            for threshold in thresholds
        },
    }


def plot_distribution(table: pd.DataFrame, summary: dict[str, object], output_path: Path) -> None:
    mass = table["mass_first"].to_numpy(float)
    sorted_mass = np.sort(mass)
    ecdf = np.arange(1, len(sorted_mass) + 1) / len(sorted_mass)
    threshold_colors = {300: "#2563eb", 500: "#dc2626", 700: "#7c3aed"}

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    axes[0].hist(mass, bins=42, color="#4a5568", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[0].axvline(np.median(mass), color="#111827", linewidth=1.5, label=f"median={np.median(mass):.0f} Da")
    for threshold, color in threshold_colors.items():
        count = int(np.sum(mass > threshold))
        axes[0].axvline(threshold, color=color, linestyle="--", linewidth=1.2, label=f">{threshold} Da: {count}")
    axes[0].set_xlabel("Mass from annotation field (Da)", fontsize=9)
    axes[0].set_ylabel("Number of features", fontsize=9)
    axes[0].set_title("Mass distribution", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(sorted_mass, ecdf, color="#111827", linewidth=1.8)
    for threshold, color in threshold_colors.items():
        fraction_le = float(np.mean(mass <= threshold))
        axes[1].axvline(threshold, color=color, linestyle="--", linewidth=1.0)
        axes[1].text(
            threshold,
            0.04,
            f"<={threshold}: {fraction_le:.0%}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color=color,
        )
    axes[1].set_xlabel("Mass from annotation field (Da)", fontsize=9)
    axes[1].set_ylabel("Cumulative fraction", fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("ECDF", fontsize=10)

    for axis in axes:
        axis.grid(True, color="#e5e7eb", linewidth=0.6)
        axis.tick_params(labelsize=8)
    figure.suptitle(
        "matrix.xlsx chemical feature mass distribution\n"
        f"n={summary['n_features']}; mean={summary['mean_mass']:.1f} Da; "
        f"median={summary['mass_quantiles']['0.5']:.1f} Da",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run(root: Path) -> dict[str, object]:
    table = load_mass_table(root)
    summary = summarize(table)
    output_dir = root / "tables" / "chemical_feature_type_audit"
    figure_path = root / "figures" / "matrix_chemical_mass_distribution.png"
    summary_path = output_dir / "matrix_chemical_mass_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_distribution(table, summary, figure_path)
    return {
        "summary": summary,
        "outputs": {
            "figure": str(figure_path),
            "summary": str(summary_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mass distribution for matrix.xlsx chemical features.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))


if __name__ == "__main__":
    main()
