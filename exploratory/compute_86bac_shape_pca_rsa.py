from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bacteria_analysis.features.neural import build_trial_feature_matrix, neural_feature_columns  # noqa: E402
from bacteria_analysis._data_loaders import read_metabolite_matrix  # noqa: E402
from bacteria_analysis._data_loaders import _canonicalize_metabolite_name  # noqa: E402


MERGED_NEURONS = (
    "ADF",
    "ADL",
    "ASEL",
    "ASER",
    "ASG",
    "ASH",
    "ASI",
    "ASJ",
    "ASK",
    "AWA",
    "AWB",
    "AWCOFF",
    "AWCON",
)

DEFAULT_OUTPUT_DIR = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"


def sample_id_from_stim_name(stim_name: object) -> str:
    parts = str(stim_name).strip().split()
    return parts[0] if parts else ""


def sample_number(sample_id: object) -> int:
    match = re.search(r"(\d+)", str(sample_id))
    return int(match.group(1)) if match else -1


def feature_window_columns(columns: list[str], *, window_start: int, window_stop: int) -> list[str]:
    wanted_times = {f"t{time_point:02d}" for time_point in range(window_start, window_stop)}
    return [column for column in columns if column.rsplit("__", 1)[-1] in wanted_times]


def aggregate_neural_features(
    features: pd.DataFrame,
    *,
    group_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group in features.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prototype = np.nanmedian(values, axis=0)
        row = dict(zip(group_columns, group_key, strict=True))
        row["n_trials"] = int(group["trial_id"].nunique())
        row.update(dict(zip(feature_columns, prototype, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_active_scales(
    date_stim_prototypes: pd.DataFrame,
    *,
    feature_columns: list[str],
    active_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for neuron in MERGED_NEURONS:
        neuron_columns = [column for column in feature_columns if column.startswith(f"{neuron}__")]
        values = date_stim_prototypes.loc[:, neuron_columns].to_numpy(dtype=float, copy=False).ravel()
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            scale = 1.0
            n_active = 0
            method = "fallback_no_finite_values"
        else:
            active_values = finite_values[np.abs(finite_values) >= active_threshold]
            n_active = int(active_values.size)
            if active_values.size:
                scale = float(np.mean(np.abs(active_values)))
                method = "mean_abs_active_frames"
            else:
                scale = 1.0
                method = "silent_neuron_unit_scale"
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
            method = "fallback_unit_scale"
        rows.append(
            {
                "neuron": neuron,
                "active_scale": scale,
                "n_active_frames_for_scale": n_active,
                "scaling_method": method,
            }
        )
    return pd.DataFrame(rows)


def active_scale_prototypes(
    prototypes: pd.DataFrame,
    *,
    feature_columns: list[str],
    scales: pd.DataFrame,
) -> pd.DataFrame:
    scale_by_neuron = scales.set_index("neuron")["active_scale"].to_dict()
    scaled = prototypes.copy()
    for column in feature_columns:
        neuron = column.split("__", 1)[0]
        scaled[column] = scaled[column].astype(float) / float(scale_by_neuron[neuron])
    return scaled


def build_correlation_rdm(values: pd.DataFrame, *, label_column: str, feature_columns: list[str]) -> pd.DataFrame:
    labels = values[label_column].astype(str).tolist()
    array = values.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
    distances = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)
    valid_counts = np.zeros((len(labels), len(labels)), dtype=int)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            valid = np.isfinite(array[i]) & np.isfinite(array[j])
            valid_counts[i, j] = valid_counts[j, i] = int(valid.sum())
            if valid.sum() < 2:
                continue
            left = array[i, valid]
            right = array[j, valid]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            distance = float(np.clip(1.0 - np.corrcoef(left, right)[0, 1], 0.0, 2.0))
            distances[i, j] = distances[j, i] = distance

    rdm = pd.DataFrame(distances, index=labels, columns=labels)
    rdm.index.name = label_column
    return rdm


def build_euclidean_rdm(values: pd.DataFrame, *, label_column: str, feature_columns: list[str]) -> pd.DataFrame:
    labels = values[label_column].astype(str).tolist()
    array = values.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
    distances = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            valid = np.isfinite(array[i]) & np.isfinite(array[j])
            if valid.sum() < 1:
                continue
            distance = float(np.linalg.norm(array[i, valid] - array[j, valid]))
            distances[i, j] = distances[j, i] = distance
    rdm = pd.DataFrame(distances, index=labels, columns=labels)
    rdm.index.name = label_column
    return rdm


def raw_missing_rates(raw_metadata: pd.DataFrame, sample_ids: pd.Index) -> pd.Series:
    sample_columns = [sample_id for sample_id in sample_ids.astype(str) if sample_id in raw_metadata.columns]
    if not sample_columns:
        raise ValueError("raw metabolite workbook has no matching sample columns")
    return raw_metadata.loc[:, sample_columns].isna().mean(axis=1)


def retained_chemical_features(
    matrix: pd.DataFrame,
    raw_metadata: pd.DataFrame,
    *,
    qcrsd_threshold: float,
    missing_rate_threshold: float,
) -> pd.DataFrame:
    annotations = raw_metadata.copy()
    annotations["_feature_name"] = annotations["name"].map(_canonicalize_metabolite_name)
    annotations["_qcrsd"] = pd.to_numeric(annotations["QCRSD"], errors="coerce")
    annotations["_raw_missing_rate"] = raw_missing_rates(annotations, matrix.index)

    available = set(matrix.columns.astype(str))
    retained = annotations.loc[
        annotations["_feature_name"].isin(available)
        & annotations["_qcrsd"].le(qcrsd_threshold)
        & annotations["_raw_missing_rate"].le(missing_rate_threshold)
    ].copy()
    retained = retained.drop_duplicates("_feature_name", keep="first")
    return retained


def zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    valid_columns = stds[np.isfinite(stds) & (stds > 0)].index
    return (matrix.loc[:, valid_columns] - means.loc[valid_columns]) / stds.loc[valid_columns]


def run_pca(matrix: pd.DataFrame, *, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = matrix.to_numpy(dtype=float, copy=True)
    x = x - x.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    max_components = min(n_components, x.shape[0] - 1, x.shape[1], len(singular_values))
    scores = x @ vt[:max_components].T
    loadings = vt[:max_components].T
    eigenvalues = (singular_values**2) / max(x.shape[0] - 1, 1)
    explained = eigenvalues / eigenvalues.sum()

    score_columns = [f"PC{i}" for i in range(1, max_components + 1)]
    scores_df = pd.DataFrame(scores[:, :max_components], index=matrix.index, columns=score_columns)
    loadings_df = pd.DataFrame(loadings[:, :max_components], index=matrix.columns, columns=score_columns)
    explained_df = pd.DataFrame(
        {
            "pc": score_columns,
            "explained_variance_percent": explained[:max_components] * 100,
            "cumulative_variance_percent": np.cumsum(explained[:max_components]) * 100,
        }
    )
    return scores_df, loadings_df, explained_df


def upper_triangle_pair_table(neural: pd.DataFrame, chemical: pd.DataFrame) -> pd.DataFrame:
    labels = [label for label in neural.index.astype(str).tolist() if label in set(chemical.index.astype(str))]
    neural_aligned = neural.loc[labels, labels]
    chemical_aligned = chemical.loc[labels, labels]
    rows = []
    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            right = labels[j]
            rows.append(
                {
                    "sample_left": left,
                    "sample_right": right,
                    "neural_shape_distance": float(neural_aligned.iloc[i, j]),
                    "chemical_pca_distance": float(chemical_aligned.iloc[i, j]),
                }
            )
    return pd.DataFrame(rows), neural_aligned, chemical_aligned


def finite_pair_vectors(pair_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clean = pair_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["neural_shape_distance", "chemical_pca_distance"]
    )
    return (
        clean["neural_shape_distance"].to_numpy(dtype=float),
        clean["chemical_pca_distance"].to_numpy(dtype=float),
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x, method="average"), rankdata(y, method="average"))


def label_shuffle_null(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return label_shuffle_null_with_rng(
        neural,
        chemical,
        n_permutations=n_permutations,
        rng=rng,
    )


def label_shuffle_null_with_rng(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = neural.index.astype(str).tolist()
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    neural_array = neural.to_numpy(dtype=float)
    chemical_array = chemical.to_numpy(dtype=float)
    neural_vector = neural_array[iu]
    neural_rank = rankdata(neural_vector, method="average")
    null = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        perm = rng.permutation(n)
        permuted_chemical_vector = chemical_array[np.ix_(perm, perm)][iu]
        mask = np.isfinite(neural_rank) & np.isfinite(permuted_chemical_vector)
        null[iteration] = pearson(
            neural_rank[mask],
            rankdata(permuted_chemical_vector[mask], method="average"),
        )
    return null


def random_subset_label_shuffle_stability(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    *,
    n_subsets: int,
    subset_fraction: float,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not 0 < subset_fraction <= 1:
        raise ValueError("subset_fraction must be greater than 0 and at most 1")

    labels = np.asarray(neural.index.astype(str).tolist(), dtype=object)
    subset_size = max(3, min(len(labels), int(np.floor(len(labels) * subset_fraction))))
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for subset_index in range(n_subsets):
        sampled = set(rng.choice(labels, size=subset_size, replace=False))
        subset_labels = [label for label in labels if label in sampled]
        subset_pair_table, subset_neural, subset_chemical = upper_triangle_pair_table(
            neural.loc[subset_labels, subset_labels],
            chemical.loc[subset_labels, subset_labels],
        )
        neural_vector, chemical_vector = finite_pair_vectors(subset_pair_table)
        observed = spearman(neural_vector, chemical_vector)
        null = label_shuffle_null_with_rng(
            subset_neural,
            subset_chemical,
            n_permutations=n_permutations,
            rng=rng,
        )
        finite_null = null[np.isfinite(null)]
        q95 = float(np.quantile(finite_null, 0.95)) if finite_null.size else np.nan
        q99 = float(np.quantile(finite_null, 0.99)) if finite_null.size else np.nan
        p_greater = empirical_p_greater(observed, finite_null)
        percentile = float(np.mean(finite_null <= observed) * 100) if finite_null.size else np.nan

        summary_rows.append(
            {
                "subset_index": int(subset_index),
                "n_samples": int(len(subset_labels)),
                "n_pairs": int(len(neural_vector)),
                "observed_rsa": float(observed),
                "null_q95": q95,
                "null_q99": q99,
                "observed_percentile": percentile,
                "p_one_sided_ge": p_greater,
            }
        )
        null_rows.extend(
            {
                "subset_index": int(subset_index),
                "iteration": int(iteration),
                "rsa_similarity": float(value),
            }
            for iteration, value in enumerate(null)
        )

    subset_summary = pd.DataFrame(summary_rows)
    subset_null = pd.DataFrame(null_rows)
    null_values = subset_null["rsa_similarity"].to_numpy(dtype=float) if len(subset_null) else np.array([], dtype=float)
    observed_values = (
        subset_summary["observed_rsa"].to_numpy(dtype=float) if len(subset_summary) else np.array([], dtype=float)
    )
    subset_overall = {
        "subset_count": int(n_subsets),
        "subset_size": int(subset_size),
        "subset_fraction": float(subset_fraction),
        "subset_permutations": int(n_permutations),
        "observed_rsa_mean": float(np.nanmean(observed_values)) if observed_values.size else np.nan,
        "observed_rsa_median": float(np.nanmedian(observed_values)) if observed_values.size else np.nan,
        "pooled_null_q95": float(np.nanquantile(null_values, 0.95)) if null_values.size else np.nan,
        "pooled_null_q99": float(np.nanquantile(null_values, 0.99)) if null_values.size else np.nan,
        "fraction_subsets_observed_gt_own_null_q95": float(
            np.mean(subset_summary["observed_rsa"].to_numpy(float) > subset_summary["null_q95"].to_numpy(float))
        )
        if len(subset_summary)
        else np.nan,
        "fraction_subsets_observed_gt_own_null_q99": float(
            np.mean(subset_summary["observed_rsa"].to_numpy(float) > subset_summary["null_q99"].to_numpy(float))
        )
        if len(subset_summary)
        else np.nan,
    }
    return subset_summary, subset_null, subset_overall


def empirical_p_greater(observed: float, null: np.ndarray) -> float:
    finite = null[np.isfinite(null)]
    if finite.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.sum(finite >= observed) + 1) / (finite.size + 1))


def bootstrap_pair_resample(
    pair_table: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    clean = pair_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["neural_shape_distance", "chemical_pca_distance"]
    )
    neural = clean["neural_shape_distance"].to_numpy(dtype=float)
    chemical = clean["chemical_pca_distance"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for iteration in range(n_resamples):
        indices = rng.integers(0, len(clean), size=len(clean))
        rows.append(
            {
                "iteration": iteration,
                "n_pairs": int(len(indices)),
                "rsa_spearman": spearman(neural[indices], chemical[indices]),
                "resample_type": "pair_bootstrap",
            }
        )
    return pd.DataFrame(rows)


def stimulus_subset_resample(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    *,
    n_resamples: int,
    subset_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 < subset_fraction <= 1:
        raise ValueError("subset_fraction must be greater than 0 and at most 1")
    labels = np.asarray(neural.index.astype(str).tolist(), dtype=object)
    subset_size = max(3, min(len(labels), int(np.floor(len(labels) * subset_fraction))))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for iteration in range(n_resamples):
        sampled = set(rng.choice(labels, size=subset_size, replace=False))
        subset = [label for label in labels if label in sampled]
        subset_pair_table, _, _ = upper_triangle_pair_table(
            neural.loc[subset, subset],
            chemical.loc[subset, subset],
        )
        x, y = finite_pair_vectors(subset_pair_table)
        rows.append(
            {
                "iteration": iteration,
                "n_samples": int(len(subset)),
                "n_pairs": int(len(x)),
                "rsa_spearman": spearman(x, y),
                "resample_type": f"stimulus_subset_{subset_fraction:g}",
            }
        )
    return pd.DataFrame(rows)


def neural_cluster_order(rdm: pd.DataFrame) -> list[str]:
    labels = rdm.index.astype(str).tolist()
    if len(labels) < 3:
        return labels

    matrix = rdm.loc[labels, labels].to_numpy(dtype=float, copy=True)
    off_diagonal = matrix[~np.eye(len(labels), dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    fill_value = float(np.max(finite)) if finite.size else 1.0
    matrix = np.where(np.isfinite(matrix), matrix, fill_value)
    np.fill_diagonal(matrix, 0.0)
    condensed = squareform(matrix, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return [labels[index] for index in order]


def mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.apply(pd.to_numeric, errors="coerce").copy()
    values = display.to_numpy(dtype=float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


def plot_foundation_rdm_pair(
    *,
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    display_order: list[str],
    observed_rsa: float,
    chemical_pcs: int,
    output_path: Path,
) -> None:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#FFFFFF")
    n_labels = len(display_order)
    label_fontsize = max(3.4, min(4.8, 240.0 / max(n_labels, 1)))
    figure_width = max(13.4, 0.20 * n_labels + 3.0)
    figure_height = max(5.7, 0.085 * n_labels + 1.6)

    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.04, 1.0, 0.04])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    colorbar_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    tick_positions = list(range(n_labels))
    neural_display = mask_diagonal(neural.loc[display_order, display_order])
    chemical_display = mask_diagonal(chemical.loc[display_order, display_order])

    panels = [
        (
            axes[0],
            colorbar_axes[0],
            neural_display,
            "Neural RDM\nactive-scaled trace shape",
            "correlation distance",
        ),
        (
            axes[1],
            colorbar_axes[1],
            chemical_display,
            f"Chemical RDM\nQC20 log2 z-score PCA{chemical_pcs}\nRSA={observed_rsa:.3f}",
            "Euclidean distance in PC space",
        ),
    ]
    for ax, colorbar_axis, matrix, title, colorbar_label in panels:
        finite = matrix.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            cmap=cmap,
            vmin=float(np.min(finite)) if finite.size else None,
            vmax=float(np.max(finite)) if finite.size else None,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(display_order, rotation=90, fontsize=label_fontsize)
        ax.set_yticklabels(display_order, fontsize=label_fontsize)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.suptitle("Neural vs Chemical RDM\nordered by neural-shape clustering", fontsize=12)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_foundation_distribution(
    *,
    null_values: np.ndarray,
    observed_rsa: float,
    p_greater: float,
    n_permutations: int,
    output_path: Path,
    y_mode: str,
) -> None:
    if y_mode not in {"count", "fraction"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")

    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    q99 = float(np.quantile(finite, 0.99)) if finite.size else np.nan
    null_max = float(np.max(finite)) if finite.size else np.nan
    weights = np.ones_like(finite) / finite.size if y_mode == "fraction" and finite.size else None

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.hist(
        finite,
        bins=80,
        weights=weights,
        color="#7E4CC2",
        edgecolor="#7E4CC2",
        linewidth=0.2,
        alpha=0.95,
    )
    ax.axvline(observed_rsa, color="#F5A623", linewidth=1.8)
    ax.axvline(q99, color="#6E6E6E", linestyle="--", linewidth=1.4, zorder=4)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xlabel("shuffle RSA")
    ax.set_ylabel("fraction" if y_mode == "fraction" else "permutations")
    ax.set_title(
        "86bac label-shuffle null distribution\n"
        f"{n_permutations:,} stimulus-label permutations; orange = observed RSA; gray dashed = null q99",
        fontsize=10,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color("#666666")
        ax.spines[spine_name].set_linewidth(0.8)
    ax.text(
        0.98,
        0.92,
        f"obs={observed_rsa:.2f}\np={p_greater:.4f}\nmax={null_max:.2f}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_foundation_random_subset_distribution(
    *,
    subset_summary: pd.DataFrame,
    subset_null: pd.DataFrame,
    subset_overall: dict[str, object],
    output_path: Path,
) -> None:
    null_values = subset_null["rsa_similarity"].to_numpy(dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    observed_values = subset_summary["observed_rsa"].to_numpy(dtype=float)
    observed_values = observed_values[np.isfinite(observed_values)]
    percentiles = subset_summary["observed_percentile"].to_numpy(dtype=float)
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

    axes[1].text(
        0.04,
        0.96,
        "subsets={subset_count}\nsubset size={subset_size}\nshuffles/subset={subset_permutations}\n"
        "obs>q95={q95:.2f}\nobs>q99={q99:.2f}".format(
            subset_count=int(subset_overall["subset_count"]),
            subset_size=int(subset_overall["subset_size"]),
            subset_permutations=int(subset_overall["subset_permutations"]),
            q95=float(subset_overall["fraction_subsets_observed_gt_own_null_q95"]),
            q99=float(subset_overall["fraction_subsets_observed_gt_own_null_q99"]),
        ),
        ha="left",
        va="top",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    fig.suptitle("Random stimulus-subset permutation stability", fontsize=12)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, output_path: Path, *, title: str, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, interpolation="nearest")
    labels = matrix.index.astype(str).tolist()
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rsa_scatter(pair_table: pd.DataFrame, output_path: Path, *, spearman_r: float) -> None:
    clean = pair_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["neural_shape_distance", "chemical_pca_distance"]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    ax.scatter(
        clean["chemical_pca_distance"],
        clean["neural_shape_distance"],
        s=10,
        alpha=0.45,
        color="#2563EB",
        edgecolors="none",
    )
    ax.set_xlabel("Chemical PCA distance")
    ax.set_ylabel("Neural shape distance")
    ax.set_title(f"RSA paired distances (Spearman r = {spearman_r:.3f})")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pca_variance(explained: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    x = np.arange(len(explained)) + 1
    ax.bar(x, explained["explained_variance_percent"], color="#64748B")
    ax.plot(x, explained["cumulative_variance_percent"], color="#C2410C", marker="o", linewidth=1.5)
    ax.set_xlabel("Chemical PC")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Chemical PCA variance")
    ax.set_xticks(x)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_distribution(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    observed: float | None = None,
    color: str = "#64748B",
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    ax.hist(finite, bins=45, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
    if observed is not None and np.isfinite(observed):
        ax.axvline(observed, color="#C2410C", linewidth=2.0, label=f"Observed = {observed:.3f}")
        ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_resample_distributions(
    pair_bootstrap: pd.DataFrame,
    stimulus_subset: pd.DataFrame,
    output_path: Path,
    *,
    observed: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    configs = [
        (axes[0], pair_bootstrap, "Pair bootstrap"),
        (axes[1], stimulus_subset, "Stimulus subset"),
    ]
    for ax, frame, title in configs:
        values = frame["rsa_spearman"].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=45, color="#2563EB", alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.axvline(observed, color="#C2410C", linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Spearman RSA")
        ax.set_ylabel("Count")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 86bac neural shape RDM, chemical PCA RDM, and RSA.")
    parser.add_argument("--neural-parquet", type=Path, default=Path("data") / "86bac.parquet")
    parser.add_argument("--fold-change-matrix", type=Path, default=Path("data") / "data_fc_missingto1.xlsx")
    parser.add_argument("--raw-metabolites", type=Path, default=Path("data") / "metabolism_raw_data.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-start", type=int, default=5)
    parser.add_argument("--window-stop", type=int, default=25)
    parser.add_argument("--active-threshold", type=float, default=0.2)
    parser.add_argument("--qcrsd-threshold", type=float, default=0.2)
    parser.add_argument("--missing-rate-threshold", type=float, default=0.5)
    parser.add_argument("--chemical-pcs", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--subset-fraction", type=float, default=0.8)
    parser.add_argument("--subset-permutation-subsets", type=int, default=200)
    parser.add_argument("--subset-permutation-shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_neural = pd.read_parquet(args.neural_parquet)
    trial_features = build_trial_feature_matrix(raw_neural, view="full_trajectory", merge_lr=True)
    all_feature_columns = neural_feature_columns(trial_features)
    window_columns = feature_window_columns(
        all_feature_columns,
        window_start=args.window_start,
        window_stop=args.window_stop,
    )

    date_stim = aggregate_neural_features(
        trial_features,
        group_columns=["date", "stim_name"],
        feature_columns=window_columns,
    )
    active_scales = compute_active_scales(
        date_stim,
        feature_columns=window_columns,
        active_threshold=args.active_threshold,
    )
    active_scales.to_csv(tables_dir / "neural_active_scales.csv", index=False)

    sample_prototypes = aggregate_neural_features(
        trial_features,
        group_columns=["stim_name"],
        feature_columns=window_columns,
    )
    sample_prototypes["sample_id"] = sample_prototypes["stim_name"].map(sample_id_from_stim_name)
    sample_prototypes = sample_prototypes.sort_values("sample_id", key=lambda s: s.map(sample_number)).reset_index(drop=True)
    scaled_neural = active_scale_prototypes(
        sample_prototypes,
        feature_columns=window_columns,
        scales=active_scales,
    )
    scaled_neural.to_csv(tables_dir / "active_scaled_flattened_neural_prototypes.csv", index=False)

    neural_rdm = build_correlation_rdm(scaled_neural, label_column="sample_id", feature_columns=window_columns)
    neural_rdm.to_csv(tables_dir / "neural_shape_rdm__active_scaled_flattened_correlation.csv")

    fold_change = read_metabolite_matrix(args.fold_change_matrix)
    raw_metadata = pd.read_excel(args.raw_metabolites, sheet_name="all", engine="openpyxl")
    retained = retained_chemical_features(
        fold_change,
        raw_metadata,
        qcrsd_threshold=args.qcrsd_threshold,
        missing_rate_threshold=args.missing_rate_threshold,
    )
    retained.to_csv(tables_dir / "retained_chemical_features.csv", index=False)

    shared_samples = [sample for sample in scaled_neural["sample_id"].astype(str).tolist() if sample in fold_change.index]
    if len(shared_samples) < 3:
        raise ValueError("fewer than 3 shared neural/chemical samples")

    retained_features = retained["_feature_name"].astype(str).tolist()
    chemical = fold_change.loc[shared_samples, retained_features].apply(pd.to_numeric, errors="coerce")
    if (chemical <= 0).to_numpy().any():
        bad = chemical.columns[(chemical <= 0).any(axis=0)].tolist()
        raise ValueError(f"log2 fold-change requires positive retained values: {bad[:10]}")
    chemical_log2 = np.log2(chemical)
    chemical_z = zscore_columns(chemical_log2)
    chemical_z.to_csv(tables_dir / "chemical_log2_zscore_matrix.csv")

    chemical_scores, chemical_loadings, chemical_explained = run_pca(
        chemical_z,
        n_components=args.chemical_pcs,
    )
    chemical_scores_out = chemical_scores.copy()
    chemical_scores_out.insert(0, "sample_id", chemical_scores_out.index)
    chemical_scores_out.to_csv(tables_dir / "chemical_pca_scores.csv", index=False)
    chemical_loadings.to_csv(tables_dir / "chemical_pca_loadings.csv")
    chemical_explained.to_csv(tables_dir / "chemical_pca_explained_variance.csv", index=False)

    chemical_rdm_input = chemical_scores.copy()
    chemical_rdm_input.insert(0, "sample_id", chemical_rdm_input.index)
    chemical_rdm = build_euclidean_rdm(
        chemical_rdm_input,
        label_column="sample_id",
        feature_columns=chemical_scores.columns.tolist(),
    )
    chemical_rdm.to_csv(tables_dir / "chemical_rdm__qc20_missing50_log2_zscore_pca10_euclidean.csv")

    pair_table, aligned_neural, aligned_chemical = upper_triangle_pair_table(neural_rdm, chemical_rdm)
    pair_table.to_csv(tables_dir / "paired_rdm_distances.csv", index=False)
    neural_vector, chemical_vector = finite_pair_vectors(pair_table)
    observed_spearman = spearman(neural_vector, chemical_vector)
    observed_pearson = pearson(neural_vector, chemical_vector)
    null = label_shuffle_null(
        aligned_neural,
        aligned_chemical,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    pd.DataFrame({"iteration": np.arange(args.permutations), "rsa_spearman": null}).to_csv(
        tables_dir / "rsa_label_shuffle_null.csv",
        index=False,
    )
    pair_bootstrap = bootstrap_pair_resample(
        pair_table,
        n_resamples=args.resamples,
        seed=args.seed + 1,
    )
    stimulus_subset = stimulus_subset_resample(
        aligned_neural,
        aligned_chemical,
        n_resamples=args.resamples,
        subset_fraction=args.subset_fraction,
        seed=args.seed + 2,
    )
    subset_permutation_summary, subset_permutation_null, subset_permutation_overall = (
        random_subset_label_shuffle_stability(
            aligned_neural,
            aligned_chemical,
            n_subsets=args.subset_permutation_subsets,
            subset_fraction=args.subset_fraction,
            n_permutations=args.subset_permutation_shuffles,
            seed=args.seed + 3,
        )
    )
    pair_bootstrap.to_csv(tables_dir / "rsa_pair_bootstrap_resamples.csv", index=False)
    stimulus_subset.to_csv(tables_dir / "rsa_stimulus_subset_resamples.csv", index=False)
    subset_permutation_summary.to_csv(tables_dir / "rsa_random_subset_label_shuffle_summary.csv", index=False)
    subset_permutation_null.to_csv(tables_dir / "rsa_random_subset_label_shuffle_null.csv", index=False)

    plot_heatmap(
        aligned_neural,
        figures_dir / "neural_shape_rdm_heatmap.png",
        title="Neural shape RDM",
    )
    plot_heatmap(
        aligned_chemical,
        figures_dir / "chemical_pca_rdm_heatmap.png",
        title="Chemical PCA RDM",
    )
    plot_rsa_scatter(
        pair_table,
        figures_dir / "rsa_paired_distance_scatter.png",
        spearman_r=observed_spearman,
    )
    plot_pca_variance(chemical_explained, figures_dir / "chemical_pca_variance.png")
    plot_distribution(
        null,
        figures_dir / "rsa_label_shuffle_null_distribution.png",
        title="Label-shuffle RSA null",
        xlabel="Spearman RSA",
        observed=observed_spearman,
        color="#64748B",
    )
    plot_resample_distributions(
        pair_bootstrap,
        stimulus_subset,
        figures_dir / "rsa_resample_distributions.png",
        observed=observed_spearman,
    )
    shuffle_tag = f"{args.permutations // 1000}k" if args.permutations % 1000 == 0 else str(args.permutations)
    subset_tag = (
        f"{args.subset_permutation_subsets}x{args.subset_permutation_shuffles}_"
        f"frac{int(round(args.subset_fraction * 100))}"
    )
    display_order = neural_cluster_order(aligned_neural)
    label_shuffle_p = empirical_p_greater(observed_spearman, null)
    plot_foundation_rdm_pair(
        neural=aligned_neural,
        chemical=aligned_chemical,
        display_order=display_order,
        observed_rsa=observed_spearman,
        chemical_pcs=int(chemical_scores.shape[1]),
        output_path=figures_dir / f"neural_chemical_rdm_foundation__86bac_shape_pca_label_shuffle_{shuffle_tag}_rdms.png",
    )
    plot_foundation_distribution(
        null_values=null,
        observed_rsa=observed_spearman,
        p_greater=label_shuffle_p,
        n_permutations=args.permutations,
        output_path=figures_dir
        / f"neural_chemical_rdm_foundation__86bac_shape_pca_label_shuffle_{shuffle_tag}_distribution.png",
        y_mode="count",
    )
    plot_foundation_distribution(
        null_values=null,
        observed_rsa=observed_spearman,
        p_greater=label_shuffle_p,
        n_permutations=args.permutations,
        output_path=figures_dir
        / f"neural_chemical_rdm_foundation__86bac_shape_pca_label_shuffle_{shuffle_tag}_distribution_fraction.png",
        y_mode="fraction",
    )
    plot_foundation_random_subset_distribution(
        subset_summary=subset_permutation_summary,
        subset_null=subset_permutation_null,
        subset_overall=subset_permutation_overall,
        output_path=figures_dir
        / f"neural_chemical_rdm_foundation__86bac_shape_pca_random_subsets_{subset_tag}_distribution_fraction.png",
    )

    sample_overlap = pd.DataFrame(
        {
            "sample_id": sorted(set(scaled_neural["sample_id"]), key=sample_number),
            "in_fold_change_matrix": [sample in fold_change.index for sample in sorted(set(scaled_neural["sample_id"]), key=sample_number)],
        }
    )
    sample_overlap.to_csv(tables_dir / "sample_overlap.csv", index=False)

    summary = {
        "neural": {
            "input": str(args.neural_parquet),
            "n_trials": int(trial_features["trial_id"].nunique()),
            "n_stimuli": int(sample_prototypes["stim_name"].nunique()),
            "n_sample_level_prototypes": int(len(sample_prototypes)),
            "merge_lr": True,
            "feature": "active-scaled flattened trace",
            "window_start_inclusive": int(args.window_start),
            "window_stop_exclusive": int(args.window_stop),
            "n_merged_neurons": len(MERGED_NEURONS),
            "n_trace_features": int(len(window_columns)),
            "shape_rdm_distance": "1 - Pearson correlation",
            "active_threshold_for_scale": float(args.active_threshold),
            "silent_neuron_scale": 1.0,
            "silent_neuron_scaling_policy": "preserve_raw_values",
        },
        "chemical": {
            "input": str(args.fold_change_matrix),
            "raw_metadata": str(args.raw_metabolites),
            "qcrsd_threshold": float(args.qcrsd_threshold),
            "missing_rate_threshold": float(args.missing_rate_threshold),
            "n_retained_features": int(len(retained_features)),
            "n_nonconstant_features_after_zscore": int(chemical_z.shape[1]),
            "transform": "log2 fold-change, per-feature z-score",
            "embedding": f"first {chemical_scores.shape[1]} PCs",
            "pca_cumulative_variance_percent": float(chemical_explained["cumulative_variance_percent"].iloc[-1]),
            "rdm_distance": "Euclidean distance in PCA score space",
        },
        "rsa": {
            "n_shared_samples": int(len(shared_samples)),
            "n_pairs": int(len(neural_vector)),
            "spearman_r": observed_spearman,
            "pearson_r": observed_pearson,
            "label_shuffle_permutations": int(args.permutations),
            "label_shuffle_p_greater": label_shuffle_p,
            "null_mean": float(np.nanmean(null)),
            "null_std": float(np.nanstd(null)),
            "null_q99": float(np.nanquantile(null, 0.99)),
            "null_max": float(np.nanmax(null)),
            "pair_bootstrap_resamples": int(args.resamples),
            "pair_bootstrap_mean": float(pair_bootstrap["rsa_spearman"].mean()),
            "pair_bootstrap_ci95": [
                float(pair_bootstrap["rsa_spearman"].quantile(0.025)),
                float(pair_bootstrap["rsa_spearman"].quantile(0.975)),
            ],
            "stimulus_subset_resamples": int(args.resamples),
            "stimulus_subset_fraction": float(args.subset_fraction),
            "stimulus_subset_n_samples": int(stimulus_subset["n_samples"].iloc[0]),
            "stimulus_subset_mean": float(stimulus_subset["rsa_spearman"].mean()),
            "stimulus_subset_ci95": [
                float(stimulus_subset["rsa_spearman"].quantile(0.025)),
                float(stimulus_subset["rsa_spearman"].quantile(0.975)),
            ],
            "random_subset_label_shuffle": subset_permutation_overall,
            "seed": int(args.seed),
        },
        "outputs": {
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_summary.md").write_text(format_summary_markdown(summary), encoding="utf-8")


def format_summary_markdown(summary: dict[str, object]) -> str:
    neural = summary["neural"]
    chemical = summary["chemical"]
    rsa = summary["rsa"]
    return "\n".join(
        [
            "# 86bac Neural Shape RDM vs Chemical PCA RDM",
            "",
            "## Neural RDM",
            "",
            f"- Input: `{neural['input']}`",
            f"- Prototypes: {neural['n_sample_level_prototypes']} sample-level stimuli from {neural['n_trials']} trials",
            f"- Feature: {neural['feature']}, window {neural['window_start_inclusive']}:{neural['window_stop_exclusive']}",
            f"- L/R merge: {neural['merge_lr']}; merged neurons: {neural['n_merged_neurons']}; trace features: {neural['n_trace_features']}",
            f"- Distance: {neural['shape_rdm_distance']}",
            f"- Silent-neuron scaling: preserve raw values with scale={neural['silent_neuron_scale']:.1f}",
            "",
            "## Chemical RDM",
            "",
            f"- Input: `{chemical['input']}`",
            f"- Filter: QCRSD <= {chemical['qcrsd_threshold']}, raw missing rate <= {chemical['missing_rate_threshold']}",
            f"- Retained features: {chemical['n_retained_features']} ({chemical['n_nonconstant_features_after_zscore']} non-constant after z-score)",
            f"- Transform: {chemical['transform']}",
            f"- Embedding: {chemical['embedding']}; cumulative variance = {chemical['pca_cumulative_variance_percent']:.2f}%",
            f"- Distance: {chemical['rdm_distance']}",
            "",
            "## RSA",
            "",
            f"- Shared samples: {rsa['n_shared_samples']}",
            f"- Pairwise distances: {rsa['n_pairs']}",
            f"- Spearman RSA: {rsa['spearman_r']:.6f}",
            f"- Pearson RSA: {rsa['pearson_r']:.6f}",
            f"- Label-shuffle p(greater): {rsa['label_shuffle_p_greater']:.6f} ({rsa['label_shuffle_permutations']} permutations)",
            f"- Null mean/std/q99/max: {rsa['null_mean']:.6f} / {rsa['null_std']:.6f} / {rsa['null_q99']:.6f} / {rsa['null_max']:.6f}",
            f"- Pair bootstrap mean / 95% CI: {rsa['pair_bootstrap_mean']:.6f} / [{rsa['pair_bootstrap_ci95'][0]:.6f}, {rsa['pair_bootstrap_ci95'][1]:.6f}]",
            f"- Stimulus subset ({rsa['stimulus_subset_fraction']:.2f}, n={rsa['stimulus_subset_n_samples']}) mean / 95% CI: {rsa['stimulus_subset_mean']:.6f} / [{rsa['stimulus_subset_ci95'][0]:.6f}, {rsa['stimulus_subset_ci95'][1]:.6f}]",
            f"- Random subset label-shuffle: {rsa['random_subset_label_shuffle']['subset_count']} subsets x {rsa['random_subset_label_shuffle']['subset_permutations']} shuffles; median observed = {rsa['random_subset_label_shuffle']['observed_rsa_median']:.6f}; pooled null q99 = {rsa['random_subset_label_shuffle']['pooled_null_q99']:.6f}",
            "",
            "## Interpretation Caveat",
            "",
            "This is a first-pass comparison between a neural shape RDM and a cleaner unsupervised chemical PCA RDM. The neural RDM intentionally excludes response strength/gain. The chemical RDM is variance-driven and should be treated as a baseline, not as a final pathway-aware metabolic-footprint embedding.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
