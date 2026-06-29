"""Plot readable neural-vs-chemical RDM review figures for the 76-bacteria batch."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from bacteria_analysis.analyses.rdm.builders import RdmResult, build_chemical_rdm, build_neural_rdm
from bacteria_analysis.features.chemical import build_chemical_feature_matrix
from bacteria_analysis.features.neural import build_trial_feature_matrix, neural_feature_columns
from bacteria_analysis._analysis_dataset_impl import AnalysisDataset
from bacteria_analysis._data_loaders import build_stimulus_sample_map, read_metabolite_matrix


DEFAULT_NEURAL_PATH = Path("data/76bac_20260311to20260429.parquet")
DEFAULT_MATRIX_PATH = Path("data/data_fc_missingto1_filtered.xlsx")
DEFAULT_CHEMICAL_SUMMARY_PATH = Path("data/data_fc_missingto1_filtered_summary.xlsx")
DEFAULT_METADATA_PATH: Path | None = None
DEFAULT_OUTPUT_ROOT = Path(
    "results/76bac_20260311to20260429/neural_chemical_rdm_missingto1_filtered/response_window"
)
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_SUBSET_COUNT = 200
DEFAULT_SUBSET_FRACTION = 0.8
DEFAULT_SUBSET_PERMUTATIONS = 500
DEFAULT_SEED = 7600
DEFAULT_CHEMICAL_STANDARDIZE = "none"
DEFAULT_NEURAL_MODE = "flatten_correlation"
DEFAULT_ACTIVE_MIN_ABS_PEAK = 0.05
DEFAULT_ACTIVE_MIN_STIMULUS_FRACTION = 0.10
MODEL_ID = "missingto1_filtered__log2_euclidean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-path", type=Path, default=DEFAULT_NEURAL_PATH)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--chemical-summary-path", type=Path, default=DEFAULT_CHEMICAL_SUMMARY_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--include-dates", default="")
    parser.add_argument(
        "--neural-mode",
        choices=("flatten_correlation", "active_signed_peak_euclidean"),
        default=DEFAULT_NEURAL_MODE,
    )
    parser.add_argument("--active-min-abs-peak", type=float, default=DEFAULT_ACTIVE_MIN_ABS_PEAK)
    parser.add_argument(
        "--active-min-stimulus-fraction",
        type=float,
        default=DEFAULT_ACTIVE_MIN_STIMULUS_FRACTION,
    )
    parser.add_argument("--qc-threshold", type=float, default=1.0)
    parser.add_argument("--chemical-standardize", choices=("none", "zscore"), default=DEFAULT_CHEMICAL_STANDARDIZE)
    parser.add_argument("--chemical-title", default="missing-to-one filtered, log2 Euclidean")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--subset-count", type=int, default=DEFAULT_SUBSET_COUNT)
    parser.add_argument("--subset-fraction", type=float, default=DEFAULT_SUBSET_FRACTION)
    parser.add_argument("--subset-permutations", type=int, default=DEFAULT_SUBSET_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        neural_path=args.neural_path,
        matrix_path=args.matrix_path,
        chemical_summary_path=args.chemical_summary_path,
        metadata_path=args.metadata_path,
        include_dates=parse_include_dates(str(args.include_dates)),
        neural_mode=str(args.neural_mode),
        active_min_abs_peak=float(args.active_min_abs_peak),
        active_min_stimulus_fraction=float(args.active_min_stimulus_fraction),
        qc_threshold=float(args.qc_threshold),
        chemical_standardize=str(args.chemical_standardize),
        chemical_title=str(args.chemical_title),
        output_root=args.output_root,
        n_permutations=int(args.permutations),
        subset_count=int(args.subset_count),
        subset_fraction=float(args.subset_fraction),
        subset_permutations=int(args.subset_permutations),
        seed=int(args.seed),
    )


def run(
    *,
    neural_path: Path,
    matrix_path: Path,
    chemical_summary_path: Path,
    metadata_path: Path | None,
    include_dates: tuple[str, ...],
    neural_mode: str,
    active_min_abs_peak: float,
    active_min_stimulus_fraction: float,
    qc_threshold: float,
    chemical_standardize: str,
    chemical_title: str,
    output_root: Path,
    n_permutations: int,
    subset_count: int,
    subset_fraction: float,
    subset_permutations: int,
    seed: int,
) -> None:
    validate_options(n_permutations, subset_count, subset_fraction, subset_permutations)
    output_root = Path(output_root)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(neural_path, matrix_path, chemical_summary_path, metadata_path, include_dates=include_dates)
    neural_result = build_script_neural_rdm(
        dataset,
        mode=neural_mode,
        active_min_abs_peak=active_min_abs_peak,
        active_min_stimulus_fraction=active_min_stimulus_fraction,
    )
    chemical_result = build_script_chemical_rdm(
        dataset,
        qc_threshold=qc_threshold,
        standardize=chemical_standardize,
    )
    neural = neural_result.matrix
    chemical = chemical_result.matrix
    labels = shared_labels(neural, chemical)
    label_meta = build_label_metadata(labels, dataset)
    display_order = neural_cluster_order(neural, labels)
    order_meta = build_order_meta(display_order, label_meta)
    pair_values = build_pair_values(neural, chemical, labels)
    observed = spearman(pair_values["neural_distance"], pair_values["chemical_distance"])
    null_values = label_shuffle_null(neural, chemical, labels, n_permutations, seed)
    summary = summarize_null(
        observed=observed,
        null_values=null_values,
        n_permutations=n_permutations,
        seed=seed,
        n_stimuli=len(labels),
        n_pairs=len(pair_values),
        neural_metadata=neural_result.metadata,
        chemical_metadata=chemical_result.metadata,
    )
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
    subset_prefix = (
        "neural_chemical_rdm_foundation__response_window_random_subsets_"
        f"{subset_count}x{subset_permutations}_frac{int(round(subset_fraction * 100)):02d}"
    )

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
        neural_title=neural_title(neural_result.metadata),
        chemical_title=chemical_title,
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


def validate_options(
    n_permutations: int,
    subset_count: int,
    subset_fraction: float,
    subset_permutations: int,
) -> None:
    if n_permutations < 1:
        raise ValueError("permutations must be >= 1")
    if subset_count < 1:
        raise ValueError("subset-count must be >= 1")
    if not 0 < subset_fraction <= 1:
        raise ValueError("subset-fraction must be in (0, 1]")
    if subset_permutations < 1:
        raise ValueError("subset-permutations must be >= 1")


def parse_include_dates(raw: str) -> tuple[str, ...]:
    values = [value.strip() for value in raw.replace(";", ",").split(",")]
    return tuple(value for value in values if value)


def build_dataset(
    neural_path: Path,
    matrix_path: Path,
    chemical_summary_path: Path,
    metadata_path: Path | None,
    include_dates: tuple[str, ...] = (),
) -> AnalysisDataset:
    neural = pd.read_parquet(neural_path)
    if "date" in neural.columns:
        neural = neural.copy()
        neural["date"] = neural["date"].fillna("").astype(str).str.strip()
        if include_dates:
            keep = set(include_dates)
            neural = neural.loc[neural["date"].isin(keep)].copy()
            if neural.empty:
                raise ValueError(f"no neural rows remain after include-dates filter: {', '.join(include_dates)}")
    matrix = read_metabolite_matrix(matrix_path)
    metadata = read_metadata(metadata_path) if metadata_path is not None else pd.DataFrame()
    stimulus_sample_map = build_stimulus_sample_map(neural, matrix_sample_ids=matrix.index)
    included_dates = tuple(sorted(value for value in neural["date"].dropna().astype(str).unique().tolist() if value))
    return AnalysisDataset(
        neural=neural.reset_index(drop=True),
        matrix=matrix,
        metadata=metadata.reset_index(drop=True),
        stimulus_sample_map=stimulus_sample_map,
        included_dates=included_dates,
        excluded_dates=(),
        parameters={
            "neural_path": str(neural_path),
            "matrix_path": str(matrix_path),
            "chemical_summary_path": str(chemical_summary_path),
            "metadata_path": str(metadata_path) if metadata_path is not None else "",
            "included_dates": included_dates,
        },
    )


def build_script_neural_rdm(
    dataset: AnalysisDataset,
    *,
    mode: str,
    active_min_abs_peak: float,
    active_min_stimulus_fraction: float,
) -> RdmResult:
    if mode == "flatten_correlation":
        return build_neural_rdm(dataset, view="response_window", aggregation="median", distance="correlation")
    if mode != "active_signed_peak_euclidean":
        raise ValueError(f"unsupported neural mode: {mode}")
    if active_min_abs_peak < 0:
        raise ValueError("active-min-abs-peak must be >= 0")
    if not 0 <= active_min_stimulus_fraction <= 1:
        raise ValueError("active-min-stimulus-fraction must be in [0, 1]")

    trial_features = build_trial_feature_matrix(dataset, view="response_window", merge_lr=True)
    trial_peaks = signed_peak_trial_features(trial_features)
    peak_columns = [column for column in trial_peaks.columns.astype(str) if column not in {"trial_id", "stimulus", "stim_name", "date"}]
    prototypes = signed_peak_stimulus_prototypes(trial_peaks, peak_columns=peak_columns).set_index("stimulus")
    raw_peaks = prototypes.loc[:, peak_columns].apply(pd.to_numeric, errors="coerce")
    active_columns = active_peak_columns(
        raw_peaks,
        min_abs_peak=active_min_abs_peak,
        min_stimulus_fraction=active_min_stimulus_fraction,
    )
    scaled = zscore_feature_columns(raw_peaks.loc[:, active_columns])
    return RdmResult(
        matrix=euclidean_rdm_from_feature_matrix(scaled),
        metadata={
            "view": "response_window",
            "mode": mode,
            "aggregation": "median",
            "merge_lr": True,
            "distance": "euclidean",
            "transform": "signed_peak+zscore",
            "active_min_abs_peak": active_min_abs_peak,
            "active_min_stimulus_fraction": active_min_stimulus_fraction,
            "active_neurons": tuple(active_columns),
            "feature_count": int(len(active_columns)),
            "n_trials": int(len(trial_peaks)),
            "n_stimuli": int(len(raw_peaks)),
        },
    )


def signed_peak_trial_features(trial_features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = neural_feature_columns(trial_features)
    neurons = sorted({str(column).split("__", maxsplit=1)[0] for column in feature_columns})
    rows = trial_features.loc[:, [column for column in ("trial_id", "stimulus", "stim_name", "date") if column in trial_features]]
    peaks = pd.DataFrame(index=trial_features.index)
    for neuron in neurons:
        neuron_columns = [column for column in feature_columns if str(column).startswith(f"{neuron}__")]
        values = trial_features.loc[:, neuron_columns].to_numpy(dtype=float, copy=False)
        peaks[neuron] = signed_peak(values)
    return pd.concat([rows.reset_index(drop=True), peaks.reset_index(drop=True)], axis=1)


def signed_peak_stimulus_prototypes(trial_peaks: pd.DataFrame, *, peak_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stimulus, group in trial_peaks.groupby("stimulus", sort=True, dropna=False):
        values = group.loc[:, peak_columns].to_numpy(dtype=float, copy=False)
        with np.errstate(all="ignore"):
            prototype = np.nanmedian(values, axis=0)
        row: dict[str, object] = {"stimulus": str(stimulus), "n_trials": int(len(group))}
        if "stim_name" in group.columns:
            row["stim_name"] = str(group["stim_name"].iloc[0])
        row.update(dict(zip(peak_columns, prototype, strict=True)))
        rows.append(row)
    id_columns = ["stimulus", "stim_name", "n_trials"]
    return pd.DataFrame.from_records(rows, columns=[column for column in id_columns if any(column in row for row in rows)] + peak_columns)


def signed_peak(values: np.ndarray) -> np.ndarray:
    peaks = np.full(values.shape[0], np.nan, dtype=float)
    finite = np.isfinite(values)
    for row_index in range(values.shape[0]):
        valid_indices = np.flatnonzero(finite[row_index])
        if valid_indices.size == 0:
            continue
        row_values = values[row_index, valid_indices]
        peaks[row_index] = float(row_values[np.argmax(np.abs(row_values))])
    return peaks


def active_peak_columns(
    peaks: pd.DataFrame,
    *,
    min_abs_peak: float,
    min_stimulus_fraction: float,
) -> list[str]:
    active: list[str] = []
    for column in peaks.columns.astype(str):
        values = peaks[column].to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            continue
        if float(np.std(finite)) <= 0.0:
            continue
        active_fraction = float(np.mean(np.abs(finite) >= min_abs_peak))
        if active_fraction >= min_stimulus_fraction:
            active.append(column)
    if not active:
        raise ValueError("no active neural signed-peak features passed the requested threshold")
    return active


def build_script_chemical_rdm(
    dataset: AnalysisDataset,
    *,
    qc_threshold: float,
    standardize: str,
) -> RdmResult:
    if standardize == "none":
        return build_chemical_rdm(dataset, qc_threshold=qc_threshold, transform="log2", distance="euclidean")
    if standardize != "zscore":
        raise ValueError(f"unsupported chemical standardization: {standardize}")

    features = build_chemical_feature_matrix(dataset, qc_threshold=qc_threshold, transform="log2")
    standardized = zscore_feature_columns(features.matrix)
    return RdmResult(
        matrix=euclidean_rdm_from_feature_matrix(standardized),
        metadata={
            **features.metadata,
            "transform": "log2+zscore",
            "standardize": "zscore",
            "distance": "euclidean",
            "feature_count": int(standardized.shape[1]),
        },
    )


def zscore_feature_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().any(axis=0)].copy()
    if numeric.empty:
        raise ValueError("no finite chemical features remain after log2 transform")

    means = numeric.mean(axis=0, skipna=True)
    stds = numeric.std(axis=0, skipna=True, ddof=0)
    safe_stds = stds.where(np.isfinite(stds) & stds.gt(0.0), 1.0)
    standardized = (numeric - means) / safe_stds
    constant_columns = stds.index[~np.isfinite(stds) | stds.le(0.0)]
    if len(constant_columns):
        standardized.loc[:, constant_columns] = 0.0
    return standardized


def euclidean_rdm_from_feature_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    labels = matrix.index.astype(str).tolist()
    values = matrix.to_numpy(dtype=float, copy=False)
    distances = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)

    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            left = values[left_index]
            right = values[right_index]
            valid = np.isfinite(left) & np.isfinite(right)
            distance = float(np.linalg.norm(left[valid] - right[valid])) if np.any(valid) else np.nan
            distances[left_index, right_index] = distance
            distances[right_index, left_index] = distance

    return pd.DataFrame(distances, index=labels, columns=labels)


def read_metadata(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported metadata file type: {path.suffix}")


def shared_labels(neural: pd.DataFrame, chemical: pd.DataFrame) -> list[str]:
    labels = [label for label in neural.index.astype(str) if label in chemical.index and label in chemical.columns]
    if len(labels) < 3:
        raise ValueError(f"Need at least 3 shared stimuli, found {len(labels)}")
    missing_columns = [label for label in labels if label not in neural.columns]
    if missing_columns:
        raise ValueError(f"Neural RDM is missing columns for: {missing_columns}")
    return labels


def build_label_metadata(labels: list[str], dataset: AnalysisDataset) -> pd.DataFrame:
    meta = pd.DataFrame({"stimulus": labels})
    mapping = dataset.stimulus_sample_map.copy()
    mapping["stimulus"] = mapping["stimulus"].astype(str)
    meta = meta.merge(mapping.drop_duplicates("stimulus"), on="stimulus", how="left")
    date_map = stimulus_date_map(dataset.neural)
    meta["date"] = meta["stimulus"].map(date_map).fillna("")
    return meta


def stimulus_date_map(neural: pd.DataFrame) -> dict[str, str]:
    if not {"stimulus", "date"}.issubset(neural.columns):
        return {}
    frame = neural.loc[:, ["stimulus", "date"]].dropna().copy()
    frame["stimulus"] = frame["stimulus"].astype(str).str.strip()
    frame["date"] = frame["date"].astype(str).str.strip()
    frame = frame.loc[(frame["stimulus"] != "") & (frame["date"] != "")]
    return {
        str(stimulus): str(group["date"].value_counts().sort_index().idxmax())
        for stimulus, group in frame.groupby("stimulus", sort=True)
    }


def neural_cluster_order(neural: pd.DataFrame, labels: list[str]) -> list[str]:
    matrix = neural.loc[labels, labels].to_numpy(float)
    if len(labels) < 3:
        return labels
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return labels
    fill_value = float(np.nanmedian(finite))
    matrix = np.where(np.isfinite(matrix), matrix, fill_value)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    if np.allclose(matrix, 0.0):
        return labels
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
    *,
    observed: float,
    null_values: np.ndarray,
    n_permutations: int,
    seed: int,
    n_stimuli: int,
    n_pairs: int,
    neural_metadata: dict[str, object],
    chemical_metadata: dict[str, object],
) -> dict[str, object]:
    finite = null_values[np.isfinite(null_values)]
    n_ge = int(np.sum(finite >= observed)) if np.isfinite(observed) else 0
    return {
        "model_id": MODEL_ID,
        "neural_rdm": neural_metadata.get("mode", "response_window_median_flattened"),
        "neural_transform": neural_metadata.get("transform", "flattened"),
        "neural_distance": neural_metadata.get("distance", "correlation"),
        "chemical_transform": chemical_metadata.get("transform", "log2"),
        "chemical_distance": chemical_metadata.get("distance", "euclidean"),
        "chemical_feature_filter": "data_fc_missingto1_filtered.xlsx prefiltered; no additional QC filter",
        "chemical_feature_count": chemical_metadata.get("feature_count"),
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
    neural_title: str,
    chemical_title: str,
    output_path: Path,
) -> None:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#FFFFFF")
    labels = order_meta["display_label"].astype(str).tolist()
    n_labels = len(labels)
    tick_positions = list(range(n_labels))
    label_fontsize = max(3.4, min(4.8, 240.0 / max(n_labels, 1)))
    figure_width = max(13.4, 0.20 * n_labels + 3.0)
    figure_height = max(5.7, 0.085 * n_labels + 1.6)

    fig = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.04, 1.0, 0.04])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2])]
    colorbar_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3])]
    neural_display = mask_diagonal(neural.loc[display_order, display_order])
    chemical_display = mask_diagonal(chemical.loc[display_order, display_order])

    panels = [
        (
            axes[0],
            colorbar_axes[0],
            neural_display,
            f"Neural RDM\n{neural_title}",
            neural_colorbar_label(summary),
        ),
        (
            axes[1],
            colorbar_axes[1],
            chemical_display,
            f"Chemical RDM\n{chemical_title}\nRSA={float(summary['observed_rsa']):.3f}",
            chemical_colorbar_label(summary),
        ),
    ]
    for ax, colorbar_axis, matrix, title, colorbar_label in panels:
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
        ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
        ax.set_yticklabels(labels, fontsize=label_fontsize)
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


def neural_title(metadata: dict[str, object]) -> str:
    if metadata.get("mode") == "active_signed_peak_euclidean":
        return "active signed peak, z-score Euclidean"
    return "response window"


def neural_colorbar_label(summary: dict[str, object]) -> str:
    if summary.get("neural_transform") == "signed_peak+zscore":
        return "signed peak z-score Euclidean distance"
    return "correlation distance"


def chemical_colorbar_label(summary: dict[str, object]) -> str:
    if summary.get("chemical_transform") == "log2+zscore":
        return "log2 z-score Euclidean distance"
    return "log2 Euclidean distance"


def mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.apply(pd.to_numeric, errors="coerce").copy()
    values = display.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


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
    ax.text(0.98, 0.92, text, ha="right", va="top", transform=ax.transAxes, fontsize=8)
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


if __name__ == "__main__":
    main()
