"""Plot class-level neural-vs-chemical RDM shuffle figures for the 76-bacteria batch."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_neural_chemical_rdm_foundation as foundation
from bacteria_analysis.features.chemical import build_chemical_class_feature_matrices


DEFAULT_NEURAL_PATH = Path("data/76bac_20260311to20260429.parquet")
DEFAULT_MATRIX_PATH = Path("data/data_fc_missingto1_filtered.xlsx")
DEFAULT_CHEMICAL_SUMMARY_PATH = Path("data/data_fc_missingto1_filtered_summary.xlsx")
DEFAULT_METADATA_PATH = Path("data/metabolism_raw_data.xlsx")
DEFAULT_OUTPUT_ROOT = Path("results/76bac_20260311to20260429/rdm_qc20_zscore_class/rw")
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_MIN_FEATURES = 3
DEFAULT_TOP_N = 12
DEFAULT_SEED = 8600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-path", type=Path, default=DEFAULT_NEURAL_PATH)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--chemical-summary-path", type=Path, default=DEFAULT_CHEMICAL_SUMMARY_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--qc-threshold", type=float, default=0.2)
    parser.add_argument("--min-features", type=int, default=DEFAULT_MIN_FEATURES)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        neural_path=args.neural_path,
        matrix_path=args.matrix_path,
        chemical_summary_path=args.chemical_summary_path,
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        qc_threshold=float(args.qc_threshold),
        min_features=int(args.min_features),
        n_permutations=int(args.permutations),
        top_n=int(args.top_n),
        seed=int(args.seed),
    )


def run(
    *,
    neural_path: Path,
    matrix_path: Path,
    chemical_summary_path: Path,
    metadata_path: Path,
    output_root: Path,
    qc_threshold: float,
    min_features: int,
    n_permutations: int,
    top_n: int,
    seed: int,
) -> None:
    if min_features < 1:
        raise ValueError("min-features must be >= 1")
    if n_permutations < 1:
        raise ValueError("permutations must be >= 1")

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    dataset = foundation.build_dataset(neural_path, matrix_path, chemical_summary_path, metadata_path)
    neural = foundation.build_neural_rdm(
        dataset,
        view="response_window",
        aggregation="median",
        distance="correlation",
    ).matrix
    class_features = build_chemical_class_feature_matrices(
        dataset,
        taxonomy_level="Class",
        qc_threshold=qc_threshold,
        min_features=min_features,
        transform="log2",
    )
    if not class_features:
        raise ValueError("no Class groups passed the requested QC/min-feature filters")

    summaries, null_by_class = compare_classes(
        neural=neural,
        class_features=class_features,
        n_permutations=n_permutations,
        seed=seed,
    )
    plot_class_comparison(
        summaries=summaries,
        output_path=figures / "class_rsa_shuffle_comparison.png",
    )
    plot_top_class_distributions(
        summaries=summaries,
        null_by_class=null_by_class,
        top_n=top_n,
        n_permutations=n_permutations,
        output_path=figures / "top_class_shuffle_distributions.png",
    )


def compare_classes(
    *,
    neural: pd.DataFrame,
    class_features: dict[str, object],
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    null_by_class: dict[str, np.ndarray] = {}
    for class_index, (class_name, feature_result) in enumerate(sorted(class_features.items())):
        standardized = foundation.zscore_feature_columns(feature_result.matrix)
        chemical = foundation.euclidean_rdm_from_feature_matrix(standardized)
        labels = foundation.shared_labels(neural, chemical)
        pair_values = foundation.build_pair_values(neural, chemical, labels)
        observed = foundation.spearman(pair_values["neural_distance"], pair_values["chemical_distance"])
        null_values = foundation.label_shuffle_null(
            neural,
            chemical,
            labels,
            n_permutations,
            seed + class_index,
        )
        finite = null_values[np.isfinite(null_values)]
        rows.append(
            {
                "class_name": class_name,
                "feature_count": int(standardized.shape[1]),
                "observed_rsa": observed,
                "null_q95": foundation.nan_quantile(finite, 0.95),
                "null_q99": foundation.nan_quantile(finite, 0.99),
                "null_median": foundation.nan_quantile(finite, 0.50),
                "p_one_sided_ge": foundation.empirical_p(observed, finite),
            }
        )
        null_by_class[class_name] = finite

    summaries = pd.DataFrame(rows)
    return summaries.sort_values("observed_rsa", ascending=False).reset_index(drop=True), null_by_class


def plot_class_comparison(*, summaries: pd.DataFrame, output_path: Path) -> None:
    ordered = summaries.sort_values("observed_rsa", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    labels = [
        f"{wrap_label(name, width=34)}  ({count})"
        for name, count in zip(ordered["class_name"], ordered["feature_count"], strict=True)
    ]
    observed = ordered["observed_rsa"].to_numpy(float)
    q95 = ordered["null_q95"].to_numpy(float)
    q99 = ordered["null_q99"].to_numpy(float)
    passed_q99 = observed > q99

    fig_height = max(5.2, 0.34 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(9.6, fig_height), constrained_layout=True)
    ax.hlines(y, q95, q99, color="#9A9A9A", linewidth=2.0, alpha=0.95, label="shuffle q95-q99")
    ax.scatter(q95, y, s=18, color="#B8B8B8", zorder=3)
    ax.scatter(q99, y, s=20, color="#5F5F5F", zorder=3, label="shuffle q99")
    ax.scatter(
        observed,
        y,
        s=42,
        color=np.where(passed_q99, "#F5A623", "#FFFFFF"),
        edgecolor="#F5A623",
        linewidth=1.2,
        zorder=4,
        label="observed RSA",
    )
    ax.axvline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("RSA vs neural RDM")
    ax.set_title("Class chemical RDM RSA with label-shuffle thresholds", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    x_values = np.concatenate([observed[np.isfinite(observed)], q99[np.isfinite(q99)], q95[np.isfinite(q95)]])
    if x_values.size:
        pad = max(0.025, 0.08 * (float(np.max(x_values)) - float(np.min(x_values))))
        ax.set_xlim(float(np.min(x_values)) - pad, float(np.max(x_values)) + pad)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_top_class_distributions(
    *,
    summaries: pd.DataFrame,
    null_by_class: dict[str, np.ndarray],
    top_n: int,
    n_permutations: int,
    output_path: Path,
) -> None:
    top = summaries.head(max(1, min(top_n, len(summaries)))).copy()
    n_rows = len(top)
    all_values = [top["observed_rsa"].to_numpy(float)]
    all_values.extend(null_by_class[str(name)] for name in top["class_name"])
    finite_values = np.concatenate([values[np.isfinite(values)] for values in all_values if len(values)])
    x_min, x_max = (-0.25, 0.30)
    if finite_values.size:
        x_min = min(x_min, float(np.quantile(finite_values, 0.001)) - 0.02)
        x_max = max(x_max, float(np.quantile(finite_values, 0.999)) + 0.02)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8.8, max(4.2, 1.04 * n_rows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, top.itertuples(index=False), strict=True):
        class_name = str(row.class_name)
        null_values = null_by_class[class_name]
        weights = np.ones_like(null_values) / len(null_values) if len(null_values) else None
        ax.hist(
            null_values,
            bins=60,
            weights=weights,
            color="#7E4CC2",
            edgecolor="#7E4CC2",
            linewidth=0.2,
            alpha=0.9,
        )
        ax.axvline(float(row.observed_rsa), color="#F5A623", linewidth=1.6)
        ax.axvline(float(row.null_q99), color="#666666", linestyle="--", linewidth=1.1)
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        label = f"{class_name}  n={int(row.feature_count)}  RSA={float(row.observed_rsa):.3f}  p={float(row.p_one_sided_ge):.4f}"
        ax.text(0.01, 0.82, label, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#777777")
        ax.spines["bottom"].set_linewidth(0.7)

    axes[-1].set_xlabel("shuffle RSA")
    fig.supylabel("fraction", fontsize=9)
    fig.suptitle(
        f"Top Class label-shuffle null distributions ({n_permutations:,} shuffles each)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def wrap_label(value: object, *, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


if __name__ == "__main__":
    main()
