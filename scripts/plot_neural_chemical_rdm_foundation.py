"""Plot the primary neural-vs-chemical RDM comparison with a label-shuffle null."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


BASE = Path("results/202604_without_20260331")
DEFAULT_NEURAL_RDM = BASE / "neural_median_peak_review_lr_merge" / "neural_rdm__response_window__median_flattened.parquet"
DEFAULT_CHEMICAL_RDM = (
    BASE
    / "backup"
    / "20260422_result_refactor_unused"
    / "global_profile_method_grid_20260422"
    / "tables"
    / "model_rdm__fold_change_matrix__log2_euclidean__qcrsd20.parquet"
)
DEFAULT_ORDER = (
    BASE
    / "joint_consensus_rdm_review"
    / "tables"
    / "joint_order__merged_median__global_qc20_log2_euclidean.csv"
)
DEFAULT_OUT = BASE / "date_controlled_rsa_review"
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_SUBSET_COUNT = 200
DEFAULT_SUBSET_FRACTION = 0.8
DEFAULT_SUBSET_PERMUTATIONS = 500
DEFAULT_SEED = 20260422
MODEL_ID = "fold_change_matrix__log2_euclidean__qcrsd20"


def main() -> None:
    args = parse_args()
    run(
        neural_rdm_path=args.neural_rdm,
        chemical_rdm_path=args.chemical_rdm,
        order_path=args.order,
        output_root=args.output_root,
        n_permutations=args.permutations,
        subset_count=args.subset_count,
        subset_fraction=args.subset_fraction,
        subset_permutations=args.subset_permutations,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-rdm", type=Path, default=DEFAULT_NEURAL_RDM)
    parser.add_argument("--chemical-rdm", type=Path, default=DEFAULT_CHEMICAL_RDM)
    parser.add_argument("--order", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--subset-count", type=int, default=DEFAULT_SUBSET_COUNT)
    parser.add_argument("--subset-fraction", type=float, default=DEFAULT_SUBSET_FRACTION)
    parser.add_argument("--subset-permutations", type=int, default=DEFAULT_SUBSET_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def run(
    *,
    neural_rdm_path: Path,
    chemical_rdm_path: Path,
    order_path: Path,
    output_root: Path,
    n_permutations: int,
    subset_count: int,
    subset_fraction: float,
    subset_permutations: int,
    seed: int,
) -> None:
    if n_permutations < 1:
        raise ValueError("permutations must be >= 1")
    if subset_count < 1:
        raise ValueError("subset-count must be >= 1")
    if not 0 < subset_fraction <= 1:
        raise ValueError("subset-fraction must be in (0, 1]")
    if subset_permutations < 1:
        raise ValueError("subset-permutations must be >= 1")

    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    neural = load_rdm(neural_rdm_path)
    chemical = load_rdm(chemical_rdm_path)
    labels = shared_labels(neural, chemical)
    label_meta = load_label_metadata(order_path, labels)
    display_order = neural_cluster_order(neural, labels)
    order_meta = build_order_meta(display_order, label_meta)
    pair_values = build_pair_values(neural, chemical, labels)
    observed = spearman(pair_values["neural_distance"], pair_values["chemical_distance"])
    null_values = label_shuffle_null(neural, chemical, labels, n_permutations, seed)
    summary = summarize_null(observed, null_values, n_permutations, seed, len(labels), len(pair_values))
    subset_summary, subset_null, subset_overall = random_subset_permutation(
        neural=neural,
        chemical=chemical,
        labels=labels,
        subset_count=subset_count,
        subset_fraction=subset_fraction,
        subset_permutations=subset_permutations,
        seed=seed + 1,
    )

    prefix = f"neural_chemical_rdm_foundation__response_window_label_shuffle_{n_permutations // 1000}k"
    pair_values.to_csv(tables / f"{prefix}_pair_values.csv", index=False)
    order_meta.to_csv(tables / f"{prefix}_stimulus_order.csv", index=False)
    pd.DataFrame({"iteration": np.arange(n_permutations), "rsa_similarity": null_values}).to_csv(
        tables / f"{prefix}_null_values.csv",
        index=False,
    )
    pd.DataFrame([summary]).to_csv(tables / f"{prefix}_summary.csv", index=False)
    subset_prefix = (
        f"neural_chemical_rdm_foundation__response_window_random_subsets_"
        f"{subset_count}x{subset_permutations}_frac{int(round(subset_fraction * 100)):02d}"
    )
    subset_summary.to_csv(tables / f"{subset_prefix}_subset_summary.csv", index=False)
    subset_null.to_csv(tables / f"{subset_prefix}_null_values.csv", index=False)
    pd.DataFrame([subset_overall]).to_csv(tables / f"{subset_prefix}_summary.csv", index=False)

    legacy_combined_path = figures / f"{prefix}.png"
    if legacy_combined_path.exists():
        legacy_combined_path.unlink()
    heatmap_path = figures / f"{prefix}_rdms.png"
    distribution_path = figures / f"{prefix}_distribution.png"
    distribution_fraction_path = figures / f"{prefix}_distribution_fraction.png"
    subset_distribution_path = figures / f"{subset_prefix}_distribution_fraction.png"
    plot_heatmap_figure(
        neural=neural,
        chemical=chemical,
        display_order=display_order,
        order_meta=order_meta,
        summary=summary,
        output_path=heatmap_path,
    )
    plot_distribution_figure(
        null_values=null_values,
        summary=summary,
        n_permutations=n_permutations,
        output_path=distribution_path,
        y_mode="count",
    )
    plot_distribution_figure(
        null_values=null_values,
        summary=summary,
        n_permutations=n_permutations,
        output_path=distribution_fraction_path,
        y_mode="fraction",
    )
    plot_random_subset_distribution(
        subset_summary=subset_summary,
        subset_null=subset_null,
        subset_overall=subset_overall,
        output_path=subset_distribution_path,
    )
    write_run_summary(
        output_root=output_root,
        neural_rdm_path=neural_rdm_path,
        chemical_rdm_path=chemical_rdm_path,
        order_path=order_path,
        summary=summary,
        subset_overall=subset_overall,
        figure_paths=[heatmap_path, distribution_path, distribution_fraction_path],
        subset_figure_path=subset_distribution_path,
        prefix=prefix,
        subset_prefix=subset_prefix,
    )


def load_rdm(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "stimulus_row" not in frame.columns:
        raise ValueError(f"{path} is missing stimulus_row")
    matrix = frame.set_index("stimulus_row")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


def shared_labels(neural: pd.DataFrame, chemical: pd.DataFrame) -> list[str]:
    labels = [label for label in neural.index.astype(str) if label in chemical.index and label in chemical.columns]
    if len(labels) < 3:
        raise ValueError(f"Need at least 3 shared stimuli, found {len(labels)}")
    missing_columns = [label for label in labels if label not in neural.columns]
    if missing_columns:
        raise ValueError(f"Neural RDM is missing columns for: {missing_columns}")
    return labels


def load_label_metadata(order_path: Path, labels: list[str]) -> pd.DataFrame:
    label_set = set(labels)
    if order_path.exists():
        order = pd.read_csv(order_path)
        if "view" in order.columns:
            order = order.loc[order["view"].astype(str).eq("response_window")]
        order = order.loc[order["stimulus"].astype(str).isin(label_set)].copy()
        meta = pd.DataFrame({"stimulus": labels})
        meta = meta.merge(order.drop_duplicates("stimulus"), on="stimulus", how="left")
    else:
        meta = pd.DataFrame({"stimulus": labels})
    return meta


def neural_cluster_order(neural: pd.DataFrame, labels: list[str]) -> list[str]:
    matrix = neural.loc[labels, labels].to_numpy(float)
    if len(labels) < 3:
        return labels
    if not np.isfinite(matrix).all():
        fill_value = float(np.nanmedian(matrix[np.isfinite(matrix)]))
        matrix = np.where(np.isfinite(matrix), matrix, fill_value)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    tree = linkage(squareform(matrix, checks=False), method="average", optimal_ordering=True)
    return [labels[index] for index in leaves_list(tree)]


def build_order_meta(display_order: list[str], label_meta: pd.DataFrame) -> pd.DataFrame:
    order_meta = pd.DataFrame({"stimulus": display_order})
    order_meta = order_meta.merge(label_meta.drop_duplicates("stimulus"), on="stimulus", how="left")
    order_meta.insert(0, "display_position", np.arange(1, len(display_order) + 1))
    order_meta["display_label"] = order_meta.apply(stimulus_display_label, axis=1)
    return order_meta


def stimulus_display_label(row: pd.Series) -> str:
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    stim_name = row.get("stim_name")
    if isinstance(stim_name, str) and stim_name:
        return stim_name.split()[0]
    return str(row["stimulus"])


def build_pair_values(neural: pd.DataFrame, chemical: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for left, right in combinations(labels, 2):
        rows.append(
            {
                "stimulus_left": left,
                "stimulus_right": right,
                "neural_distance": float(neural.loc[left, right]),
                "chemical_distance": float(chemical.loc[left, right]),
            }
        )
    return pd.DataFrame(rows)


def label_shuffle_null(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    labels: list[str],
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return label_shuffle_null_with_rng(neural, chemical, labels, n_permutations, rng)


def label_shuffle_null_with_rng(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    labels: list[str],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    neural_values = neural.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
    chemical_matrix = chemical.loc[labels, labels].to_numpy(float)
    null_values = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        permutation = rng.permutation(len(labels))
        permuted = chemical_matrix[np.ix_(permutation, permutation)][upper_i, upper_j]
        null_values[iteration] = spearman(neural_values, permuted)
    return null_values


def random_subset_permutation(
    *,
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    labels: list[str],
    subset_count: int,
    subset_fraction: float,
    subset_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    subset_size = max(3, min(len(labels), int(round(len(labels) * subset_fraction))))
    subset_rows = []
    null_frames = []
    for subset_index in range(subset_count):
        selected_indices = np.sort(rng.choice(len(labels), size=subset_size, replace=False))
        subset_labels = [labels[index] for index in selected_indices]
        pair_values = build_pair_values(neural, chemical, subset_labels)
        observed = spearman(pair_values["neural_distance"], pair_values["chemical_distance"])
        null_values = label_shuffle_null_with_rng(neural, chemical, subset_labels, subset_permutations, rng)
        finite = null_values[np.isfinite(null_values)]
        subset_rows.append(
            {
                "subset_index": subset_index,
                "n_stimuli": subset_size,
                "n_pairs": int(len(pair_values)),
                "stimuli": ";".join(subset_labels),
                "observed_rsa": observed,
                "null_mean": nan_stat(np.mean, finite),
                "null_q95": nan_quantile(finite, 0.95),
                "null_q99": nan_quantile(finite, 0.99),
                "null_max": float(np.max(finite)) if finite.size else np.nan,
                "n_ge_observed": int(np.sum(finite >= observed)) if np.isfinite(observed) else 0,
                "p_one_sided_ge": empirical_p(observed, finite),
                "observed_percentile": float(np.mean(finite <= observed) * 100.0) if finite.size else np.nan,
            }
        )
        null_frames.append(
            pd.DataFrame(
                {
                    "subset_index": subset_index,
                    "iteration": np.arange(subset_permutations),
                    "rsa_similarity": null_values,
                }
            )
        )

    subset_summary = pd.DataFrame(subset_rows)
    subset_null = pd.concat(null_frames, ignore_index=True)
    observed_values = subset_summary["observed_rsa"].to_numpy(float)
    percentile_values = subset_summary["observed_percentile"].to_numpy(float)
    p_values = subset_summary["p_one_sided_ge"].to_numpy(float)
    pooled_null = subset_null["rsa_similarity"].to_numpy(float)
    pooled_null = pooled_null[np.isfinite(pooled_null)]
    subset_overall = {
        "subset_scheme": "uniform random stimulus subsets without date stratification",
        "subset_count": subset_count,
        "subset_fraction": subset_fraction,
        "subset_size": subset_size,
        "subset_permutations": subset_permutations,
        "total_null_values": int(len(subset_null)),
        "seed": seed,
        "observed_rsa_mean": nan_stat(np.mean, observed_values[np.isfinite(observed_values)]),
        "observed_rsa_median": nan_quantile(observed_values[np.isfinite(observed_values)], 0.5),
        "observed_rsa_q25": nan_quantile(observed_values[np.isfinite(observed_values)], 0.25),
        "observed_rsa_q75": nan_quantile(observed_values[np.isfinite(observed_values)], 0.75),
        "pooled_null_mean": nan_stat(np.mean, pooled_null),
        "pooled_null_q95": nan_quantile(pooled_null, 0.95),
        "pooled_null_q99": nan_quantile(pooled_null, 0.99),
        "pooled_null_max": float(np.max(pooled_null)) if pooled_null.size else np.nan,
        "median_observed_percentile": nan_quantile(percentile_values[np.isfinite(percentile_values)], 0.5),
        "fraction_subsets_observed_gt_own_null_q95": fraction(
            subset_summary["observed_rsa"].to_numpy(float) > subset_summary["null_q95"].to_numpy(float)
        ),
        "fraction_subsets_observed_gt_own_null_q99": fraction(
            subset_summary["observed_rsa"].to_numpy(float) > subset_summary["null_q99"].to_numpy(float)
        ),
        "fraction_subsets_p_le_0_05": fraction(p_values <= 0.05),
        "fraction_subsets_p_le_0_01": fraction(p_values <= 0.01),
    }
    return subset_summary, subset_null, subset_overall


def summarize_null(
    observed: float,
    null_values: np.ndarray,
    n_permutations: int,
    seed: int,
    n_stimuli: int,
    n_pairs: int,
) -> dict[str, object]:
    finite = null_values[np.isfinite(null_values)]
    n_ge = int(np.sum(finite >= observed)) if np.isfinite(observed) else 0
    return {
        "model_id": MODEL_ID,
        "neural_rdm": "response_window_median_flattened",
        "chemical_transform": "log2",
        "chemical_distance": "euclidean",
        "chemical_feature_filter": "RSD<=0.2",
        "permutation_scheme": "chemical RDM stimulus labels shuffled; neural RDM fixed",
        "n_stimuli": n_stimuli,
        "n_pairs": n_pairs,
        "n_permutations": n_permutations,
        "seed": seed,
        "observed_rsa": observed,
        "null_mean": nan_stat(np.mean, finite),
        "null_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
        "null_q025": nan_quantile(finite, 0.025),
        "null_q50": nan_quantile(finite, 0.50),
        "null_q95": nan_quantile(finite, 0.95),
        "null_q975": nan_quantile(finite, 0.975),
        "null_q99": nan_quantile(finite, 0.99),
        "null_q999": nan_quantile(finite, 0.999),
        "null_max": float(np.max(finite)) if finite.size else np.nan,
        "n_ge_observed": n_ge,
        "p_one_sided_ge": empirical_p(observed, finite),
        "observed_percentile": float(np.mean(finite <= observed) * 100.0) if finite.size else np.nan,
    }


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return np.nan
    return pearson(avg_rank(left[mask]), avg_rank(right[mask]))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or right.size < 3:
        return np.nan
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if denom == 0:
        return np.nan
    return float(np.sum(left * right) / denom)


def avg_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, copy=False).rank(method="average").to_numpy(float)


def empirical_p(observed: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        return np.nan
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def nan_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else np.nan


def nan_stat(func: object, values: np.ndarray) -> float:
    return float(func(values)) if len(values) else np.nan


def fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    return float(np.mean(mask)) if mask.size else np.nan


def plot_heatmap_figure(
    *,
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    display_order: list[str],
    order_meta: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
) -> None:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#FFFFFF")
    labels = order_meta["display_label"].astype(str).tolist()
    tick_positions = list(range(len(labels)))

    fig = plt.figure(figsize=(13.4, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.04, 1.0, 0.04])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    colorbar_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    neural_display = mask_diagonal(neural.loc[display_order, display_order])
    chemical_display = mask_diagonal(chemical.loc[display_order, display_order])

    for ax, colorbar_axis, matrix, title, colorbar_label in [
        (axes[0], colorbar_axes[0], neural_display, "Neural RDM\nresponse window", "correlation distance"),
        (
            axes[1],
            colorbar_axes[1],
            chemical_display,
            f"Chemical RDM\nlog2 Euclidean, RSD <= 0.2\nRSA={float(summary['observed_rsa']):.3f}",
            "log2 Euclidean distance",
        ),
    ]:
        finite = matrix.to_numpy(float)
        finite = finite[np.isfinite(finite)]
        image = ax.imshow(
            matrix.to_numpy(float),
            cmap=cmap,
            vmin=float(np.min(finite)) if finite.size else None,
            vmax=float(np.max(finite)) if finite.size else None,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=4.8)
        ax.set_yticklabels(labels, fontsize=4.8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Neural vs Chemical RDM\nordered by response-window neural clustering",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.copy()
    values = display.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


def rank_normalized_display(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy(float)
    finite = np.isfinite(values)
    finite &= ~np.eye(len(matrix), dtype=bool)
    display = np.full(values.shape, np.nan, dtype=float)
    if np.any(finite):
        ranks = avg_rank(values[finite])
        display[finite] = (ranks - 1.0) / max(1.0, len(ranks) - 1.0)
    return pd.DataFrame(display, index=matrix.index, columns=matrix.columns)


def plot_distribution_figure(
    *,
    null_values: np.ndarray,
    summary: dict[str, object],
    n_permutations: int,
    output_path: Path,
    y_mode: str,
) -> None:
    if y_mode not in {"count", "fraction"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    finite = null_values[np.isfinite(null_values)]
    observed = float(summary["observed_rsa"])
    q99 = float(summary["null_q99"])
    weights = np.ones_like(finite) / finite.size if y_mode == "fraction" and finite.size else None

    ax.hist(
        finite,
        bins=80,
        weights=weights,
        color="#7E4CC2",
        edgecolor="#7E4CC2",
        linewidth=0.2,
        alpha=0.95,
    )
    ax.axvline(observed, color="#F5A623", linewidth=1.8)
    ax.axvline(q99, color="#6E6E6E", linestyle="--", linewidth=1.4, zorder=4)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xlabel("shuffle RSA")
    ax.set_ylabel("fraction" if y_mode == "fraction" else "permutations")
    ax.set_title(
        "Response-window label-shuffle null distribution\n"
        f"{n_permutations:,} stimulus-label permutations; orange = observed RSA; gray dashed = null q99",
        fontsize=10,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color("#666666")
        ax.spines[spine_name].set_linewidth(0.8)

    text = (
        f"obs={observed:.2f}\n"
        f"p={float(summary['p_one_sided_ge']):.4f}\n"
        f"max={float(summary['null_max']):.2f}"
    )
    ax.text(
        0.98,
        0.92,
        text,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_random_subset_distribution(
    *,
    subset_summary: pd.DataFrame,
    subset_null: pd.DataFrame,
    subset_overall: dict[str, object],
    output_path: Path,
) -> None:
    null_values = subset_null["rsa_similarity"].to_numpy(float)
    null_values = null_values[np.isfinite(null_values)]
    observed_values = subset_summary["observed_rsa"].to_numpy(float)
    observed_values = observed_values[np.isfinite(observed_values)]
    percentiles = subset_summary["observed_percentile"].to_numpy(float)
    percentiles = percentiles[np.isfinite(percentiles)]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    bins = np.linspace(-0.3, 0.3, 81)
    null_weights = np.ones_like(null_values) / null_values.size if null_values.size else None
    observed_weights = np.ones_like(observed_values) / observed_values.size if observed_values.size else None

    axes[0].hist(
        null_values,
        bins=bins,
        weights=null_weights,
        color="#7E4CC2",
        edgecolor="#7E4CC2",
        linewidth=0.2,
        alpha=0.92,
        label="subset label-shuffle null",
    )
    axes[0].hist(
        observed_values,
        bins=bins,
        weights=observed_weights,
        histtype="step",
        color="#F5A623",
        linewidth=2.0,
        label="subset observed RSA",
    )
    axes[0].axvline(float(subset_overall["pooled_null_q99"]), color="#6E6E6E", linestyle="--", linewidth=1.4)
    axes[0].axvline(float(subset_overall["observed_rsa_median"]), color="#F5A623", linewidth=1.5)
    axes[0].set_xlim(-0.3, 0.3)
    axes[0].set_xlabel("RSA")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("Random-subset null and observed RSA", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    percentile_min = 90.0 if percentiles.size and float(np.min(percentiles)) >= 90.0 else 0.0
    percentile_bins = np.linspace(percentile_min, 100.0, 41)
    percentile_weights = np.ones_like(percentiles) / percentiles.size if percentiles.size else None
    axes[1].hist(
        percentiles,
        bins=percentile_bins,
        weights=percentile_weights,
        color="#F5A623",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
    )
    axes[1].axvline(95.0, color="#6E6E6E", linestyle="--", linewidth=1.4)
    axes[1].axvline(99.0, color="#6E6E6E", linestyle=":", linewidth=1.4)
    axes[1].set_xlim(percentile_min, 100.0)
    axes[1].set_xlabel("observed percentile within own subset null")
    axes[1].set_ylabel("fraction of subsets")
    axes[1].set_title("Observed RSA percentile by subset", fontsize=10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine_name in ["left", "bottom"]:
            ax.spines[spine_name].set_color("#666666")
            ax.spines[spine_name].set_linewidth(0.8)

    text = (
        f"subsets={int(subset_overall['subset_count'])}\n"
        f"subset size={int(subset_overall['subset_size'])}\n"
        f"shuffles/subset={int(subset_overall['subset_permutations'])}\n"
        f"obs>q95={float(subset_overall['fraction_subsets_observed_gt_own_null_q95']):.2f}\n"
        f"obs>q99={float(subset_overall['fraction_subsets_observed_gt_own_null_q99']):.2f}"
    )
    axes[1].text(0.04, 0.96, text, ha="left", va="top", transform=axes[1].transAxes, fontsize=8)
    fig.suptitle("Random stimulus-subset permutation stability", fontsize=12)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def write_run_summary(
    *,
    output_root: Path,
    neural_rdm_path: Path,
    chemical_rdm_path: Path,
    order_path: Path,
    summary: dict[str, object],
    subset_overall: dict[str, object],
    figure_paths: list[Path],
    subset_figure_path: Path,
    prefix: str,
    subset_prefix: str,
) -> None:
    lines = [
        "# Neural vs Chemical RDM Foundation",
        "",
        f"- Neural RDM: `{neural_rdm_path.as_posix()}`.",
        f"- Chemical RDM: `{chemical_rdm_path.as_posix()}`.",
        "- Chemical model: log2 transformed fold-change profile, Euclidean distance, RSD <= 0.2 feature filter.",
        "- Null model: chemical RDM stimulus labels shuffled; neural RDM fixed.",
        f"- Stimulus labels from: `{order_path.as_posix()}`.",
        "- Heatmap order: response-window neural hierarchical clustering.",
        "",
        "## Summary",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "observed_rsa": summary["observed_rsa"],
                        "null_mean": summary["null_mean"],
                        "null_q95": summary["null_q95"],
                        "null_q99": summary["null_q99"],
                        "null_max": summary["null_max"],
                        "n_ge_observed": summary["n_ge_observed"],
                        "p_one_sided_ge": summary["p_one_sided_ge"],
                    }
                ]
            )
        ),
        "",
        "## Random Subset Stability",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "subset_count": subset_overall["subset_count"],
                        "subset_size": subset_overall["subset_size"],
                        "subset_permutations": subset_overall["subset_permutations"],
                        "observed_rsa_median": subset_overall["observed_rsa_median"],
                        "pooled_null_q99": subset_overall["pooled_null_q99"],
                        "median_observed_percentile": subset_overall["median_observed_percentile"],
                        "fraction_obs_gt_own_q95": subset_overall["fraction_subsets_observed_gt_own_null_q95"],
                        "fraction_obs_gt_own_q99": subset_overall["fraction_subsets_observed_gt_own_null_q99"],
                    }
                ]
            )
        ),
        "",
        "## Output Files",
        "",
        *[f"- `{path.relative_to(output_root).as_posix()}`" for path in figure_paths],
        f"- `{subset_figure_path.relative_to(output_root).as_posix()}`",
        f"- `tables/{prefix}_summary.csv`",
        f"- `tables/{prefix}_null_values.csv`",
        f"- `tables/{prefix}_pair_values.csv`",
        f"- `tables/{prefix}_stimulus_order.csv`",
        f"- `tables/{subset_prefix}_summary.csv`",
        f"- `tables/{subset_prefix}_subset_summary.csv`",
        f"- `tables/{subset_prefix}_null_values.csv`",
    ]
    (output_root / "neural_chemical_rdm_foundation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns]
    for _, row in frame.iterrows():
        rows.append([format_cell(row[column]) for column in columns])
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(columns))]
    rendered = ["| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"]
    rendered.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows[1:]:
        rendered.append("| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |")
    return "\n".join(rendered)


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
