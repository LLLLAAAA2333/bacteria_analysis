"""Core Stage 1 reliability analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from bacteria_analysis.constants import EXPECTED_TIMEPOINTS, NEURON_ORDER
from bacteria_analysis.io import read_parquet

VIEW_WINDOWS: dict[str, tuple[int, ...]] = {
    "full_trajectory": tuple(EXPECTED_TIMEPOINTS),
    "on_window": tuple(range(6, 16)),
    "response_window": tuple(range(6, 21)),
    "post_window": tuple(range(16, 45)),
}
DEFAULT_DISTANCE_METRIC = "correlation"
SUPPORTED_DISTANCE_METRICS = (DEFAULT_DISTANCE_METRIC, "euclidean")
MIN_OVERLAP_NEURONS = 1
MIN_VALID_VALUES = 2
VALID_COMPARISON_STATUS = "ok"
SCORED_TRIAL_STATUS = "scored"
SUPPLEMENTARY_ANALYSIS_LABEL = "supplementary"


@dataclass(frozen=True)
class ReliabilityInputs:
    """Stage 1 inputs loaded from Stage 0 artifacts."""

    metadata: pd.DataFrame
    tensor: np.ndarray
    trial_ids: np.ndarray
    stimulus_labels: np.ndarray
    stim_name_labels: np.ndarray
    wide: pd.DataFrame | None = None


@dataclass(frozen=True)
class TrialView:
    """One view-specific slice of the Stage 0 trial tensor."""

    name: str
    timepoints: tuple[int, ...]
    metadata: pd.DataFrame
    values: np.ndarray


def build_individual_id(date: object, worm_key: object) -> str:
    """Build the canonical Stage 1 individual identifier."""

    return f"{str(date)}__{str(worm_key)}"


def add_individual_id(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return metadata with a deterministic Stage 1 individual_id column."""

    out = metadata.copy()
    out["date"] = out["date"].astype(str)
    out["worm_key"] = out["worm_key"].astype(str)
    out["trial_id"] = out["trial_id"].astype(str)
    out["stimulus"] = out["stimulus"].astype(str)
    if "stim_name" in out.columns:
        out["stim_name"] = out["stim_name"].astype(str)
    out["individual_id"] = out.apply(
        lambda row: build_individual_id(row["date"], row["worm_key"]),
        axis=1,
    )
    return out


def _load_tensor_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as archive:
        return {key: archive[key] for key in archive.files}


def _validate_tensor_contract(
    metadata: pd.DataFrame,
    tensor: np.ndarray,
    archive: dict[str, np.ndarray],
) -> None:
    if tensor.ndim != 3:
        raise ValueError("Stage 1 requires a 3D trial tensor")
    if tensor.shape[0] != len(metadata):
        raise ValueError("trial tensor and metadata must describe the same number of trials")
    if tensor.shape[1] != len(NEURON_ORDER):
        raise ValueError("trial tensor neuron axis does not match the canonical neuron order")
    if tensor.shape[2] != len(EXPECTED_TIMEPOINTS):
        raise ValueError("trial tensor time axis does not match the canonical 45-point grid")

    required_archive_keys = {"tensor", "trial_ids", "stimulus_labels", "stim_name_labels"}
    missing_archive_keys = sorted(required_archive_keys - set(archive))
    if missing_archive_keys:
        raise ValueError(f"tensor archive is missing required arrays: {', '.join(missing_archive_keys)}")

    metadata_trial_ids = metadata["trial_id"].astype(str).to_numpy()
    if not np.array_equal(archive["trial_ids"].astype(str), metadata_trial_ids):
        raise ValueError("tensor archive trial_ids must align exactly with metadata trial_id order")

    metadata_stimuli = metadata["stimulus"].astype(str).to_numpy()
    if not np.array_equal(archive["stimulus_labels"].astype(str), metadata_stimuli):
        raise ValueError("tensor archive stimulus_labels must align exactly with metadata stimulus order")

    if "stim_name" in metadata.columns:
        metadata_stim_names = metadata["stim_name"].astype(str).to_numpy()
        if not np.array_equal(archive["stim_name_labels"].astype(str), metadata_stim_names):
            raise ValueError("tensor archive stim_name_labels must align exactly with metadata stim_name order")


def _validate_wide_contract(metadata: pd.DataFrame, wide: pd.DataFrame) -> None:
    if len(wide) != len(metadata):
        raise ValueError("wide table and metadata must describe the same number of trials")
    if "trial_id" not in wide.columns:
        raise ValueError("wide table must include trial_id")
    if not metadata["trial_id"].astype(str).equals(wide["trial_id"].astype(str)):
        raise ValueError("wide table trial_id order must align with metadata")


def load_reliability_inputs(
    metadata_path: str | Path,
    tensor_path: str | Path,
    wide_path: str | Path | None = None,
) -> ReliabilityInputs:
    """Load and validate the canonical Stage 0 inputs used by Stage 1."""

    metadata = add_individual_id(read_parquet(metadata_path))
    archive = _load_tensor_archive(tensor_path)
    tensor = archive["tensor"].astype(float, copy=False)
    _validate_tensor_contract(metadata, tensor, archive)

    wide = None
    if wide_path is not None:
        wide = read_parquet(wide_path)
        _validate_wide_contract(metadata, wide)

    return ReliabilityInputs(
        metadata=metadata.reset_index(drop=True),
        tensor=tensor,
        trial_ids=archive["trial_ids"].astype(str),
        stimulus_labels=archive["stimulus_labels"].astype(str),
        stim_name_labels=archive["stim_name_labels"].astype(str),
        wide=wide,
    )


def build_trial_views(
    metadata: pd.DataFrame,
    tensor: np.ndarray,
    view_windows: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, TrialView]:
    """Materialize deterministic view-specific trial tensors."""

    if "individual_id" not in metadata.columns:
        metadata = add_individual_id(metadata)
    windows = view_windows or VIEW_WINDOWS
    views: dict[str, TrialView] = {}
    for view_name, timepoints in windows.items():
        views[view_name] = TrialView(
            name=view_name,
            timepoints=tuple(timepoints),
            metadata=metadata.reset_index(drop=True),
            values=tensor[:, :, list(timepoints)].copy(),
        )
    return views


def compute_vector_distance(left: np.ndarray, right: np.ndarray, metric: str = DEFAULT_DISTANCE_METRIC) -> tuple[float, str]:
    """Compute a distance on two aligned one-dimensional vectors."""

    if metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(f"unsupported metric: {metric}")

    if left.size < MIN_VALID_VALUES or right.size < MIN_VALID_VALUES:
        return float("nan"), "insufficient_valid_values"

    if metric == "euclidean":
        return float(np.linalg.norm(left - right)), VALID_COMPARISON_STATUS

    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return float("nan"), "constant_vector"

    correlation = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(correlation):
        return float("nan"), "invalid_correlation"

    return 1.0 - float(np.clip(correlation, -1.0, 1.0)), VALID_COMPARISON_STATUS


def compare_trial_arrays(
    left: np.ndarray,
    right: np.ndarray,
    metric: str = DEFAULT_DISTANCE_METRIC,
    min_overlap_neurons: int = MIN_OVERLAP_NEURONS,
) -> dict[str, float | int | str | bool]:
    """Compare two trial arrays using overlap-aware flattening."""

    shared_valid_timepoints = np.isfinite(left) & np.isfinite(right)
    overlapping_neurons = shared_valid_timepoints.any(axis=1)
    overlap_neuron_count = int(overlapping_neurons.sum())
    if overlap_neuron_count < min_overlap_neurons:
        return {
            "distance": float("nan"),
            "comparison_status": "insufficient_overlap_neurons",
            "overlap_neuron_count": overlap_neuron_count,
            "overlap_value_count": 0,
            "has_valid_distance": False,
        }

    left_flat = left[overlapping_neurons].reshape(-1)
    right_flat = right[overlapping_neurons].reshape(-1)
    valid_values = np.isfinite(left_flat) & np.isfinite(right_flat)
    overlap_value_count = int(valid_values.sum())
    if overlap_value_count < MIN_VALID_VALUES:
        return {
            "distance": float("nan"),
            "comparison_status": "insufficient_valid_values",
            "overlap_neuron_count": overlap_neuron_count,
            "overlap_value_count": overlap_value_count,
            "has_valid_distance": False,
        }

    distance, status = compute_vector_distance(left_flat[valid_values], right_flat[valid_values], metric=metric)
    return {
        "distance": distance,
        "comparison_status": status,
        "overlap_neuron_count": overlap_neuron_count,
        "overlap_value_count": overlap_value_count,
        "has_valid_distance": status == VALID_COMPARISON_STATUS and np.isfinite(distance),
    }


def compute_pairwise_distances(
    view: TrialView,
    metric: str = DEFAULT_DISTANCE_METRIC,
    min_overlap_neurons: int = MIN_OVERLAP_NEURONS,
) -> pd.DataFrame:
    """Compute overlap-aware pairwise distances for one view."""

    records: list[dict[str, object]] = []
    metadata = view.metadata.reset_index(drop=True)
    for left_index, right_index in combinations(range(len(metadata)), 2):
        left_row = metadata.iloc[left_index]
        right_row = metadata.iloc[right_index]
        comparison = compare_trial_arrays(
            view.values[left_index],
            view.values[right_index],
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )
        records.append(
            {
                "view_name": view.name,
                "trial_id_a": left_row["trial_id"],
                "trial_id_b": right_row["trial_id"],
                "stimulus_a": left_row["stimulus"],
                "stimulus_b": right_row["stimulus"],
                "individual_id_a": left_row["individual_id"],
                "individual_id_b": right_row["individual_id"],
                "date_a": left_row["date"],
                "date_b": right_row["date"],
                "same_stimulus": bool(left_row["stimulus"] == right_row["stimulus"]),
                "same_individual": bool(left_row["individual_id"] == right_row["individual_id"]),
                "same_date": bool(left_row["date"] == right_row["date"]),
                "overlap_neuron_count": comparison["overlap_neuron_count"],
                "overlap_value_count": comparison["overlap_value_count"],
                "comparison_status": comparison["comparison_status"],
                "distance": comparison["distance"],
            }
        )

    return pd.DataFrame.from_records(records)


def summarize_same_vs_different(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-stimulus versus different-stimulus distances per view."""

    rows: list[dict[str, object]] = []
    for view_name, group in comparisons.groupby("view_name", sort=False, dropna=False):
        valid = group[group["comparison_status"] == VALID_COMPARISON_STATUS].copy()
        same = valid[valid["same_stimulus"]]
        different = valid[~valid["same_stimulus"]]
        rows.append(
            {
                "view_name": view_name,
                "n_total_comparisons": int(len(group)),
                "n_valid_comparisons": int(len(valid)),
                "n_invalid_comparisons": int(len(group) - len(valid)),
                "same_count": int(len(same)),
                "different_count": int(len(different)),
                "same_mean_distance": float(same["distance"].mean()) if not same.empty else float("nan"),
                "same_median_distance": float(same["distance"].median()) if not same.empty else float("nan"),
                "different_mean_distance": float(different["distance"].mean()) if not different.empty else float("nan"),
                "different_median_distance": float(different["distance"].median()) if not different.empty else float("nan"),
                "distance_gap": (
                    float(different["distance"].mean() - same["distance"].mean())
                    if not same.empty and not different.empty
                    else float("nan")
                ),
                "overlap_neuron_mean": float(valid["overlap_neuron_count"].mean()) if not valid.empty else float("nan"),
                "overlap_neuron_median": float(valid["overlap_neuron_count"].median()) if not valid.empty else float("nan"),
                "overlap_value_mean": float(valid["overlap_value_count"].mean()) if not valid.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def select_within_date_cross_individual_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Keep only same-date comparisons that cross individuals."""

    filtered = comparisons.loc[
        comparisons["same_date"].astype(bool) & ~comparisons["same_individual"].astype(bool)
    ].copy()
    filtered["analysis_role"] = SUPPLEMENTARY_ANALYSIS_LABEL
    return filtered


def summarize_within_date_cross_individual_same_vs_different(
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize same-vs-different structure within each date across individuals."""

    filtered = select_within_date_cross_individual_comparisons(comparisons)
    summary = summarize_same_vs_different(filtered)
    if not summary.empty:
        summary["analysis_role"] = SUPPLEMENTARY_ANALYSIS_LABEL
    return filtered, summary


def _mean_trials(values: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=0)
    totals = np.nansum(values, axis=0)
    result = np.full(values.shape[1:], np.nan, dtype=float)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _build_stimulus_references(metadata: pd.DataFrame, values: np.ndarray) -> dict[str, np.ndarray]:
    references: dict[str, np.ndarray] = {}
    for stimulus, stimulus_rows in metadata.groupby("stimulus", sort=True, dropna=False):
        stimulus_indices = stimulus_rows.index.to_numpy()
        references[str(stimulus)] = _mean_trials(values[stimulus_indices])
    return references


def _score_trial_against_references(
    trial_vector: np.ndarray,
    true_stimulus: str,
    references: dict[str, np.ndarray],
    metric: str,
    min_overlap_neurons: int,
) -> dict[str, object]:
    candidate_records: list[dict[str, object]] = []
    for stimulus in sorted(references):
        comparison = compare_trial_arrays(
            trial_vector,
            references[stimulus],
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )
        candidate_records.append(
            {
                "candidate_stimulus": stimulus,
                "distance": comparison["distance"],
                "comparison_status": comparison["comparison_status"],
                "overlap_neuron_count": comparison["overlap_neuron_count"],
                "overlap_value_count": comparison["overlap_value_count"],
                "has_valid_distance": comparison["has_valid_distance"],
            }
        )

    valid_candidates = [record for record in candidate_records if record["has_valid_distance"]]
    if true_stimulus not in references:
        return {
            "predicted_stimulus": None,
            "best_distance": float("nan"),
            "true_distance": float("nan"),
            "n_candidate_stimuli": len(references),
            "score_status": "excluded",
            "exclusion_reason": "missing_training_stimulus",
            "is_correct": False,
            "best_overlap_neuron_count": 0,
            "best_overlap_value_count": 0,
        }
    if not valid_candidates:
        return {
            "predicted_stimulus": None,
            "best_distance": float("nan"),
            "true_distance": float("nan"),
            "n_candidate_stimuli": len(references),
            "score_status": "excluded",
            "exclusion_reason": "no_valid_reference_comparisons",
            "is_correct": False,
            "best_overlap_neuron_count": 0,
            "best_overlap_value_count": 0,
        }

    best_candidate = min(valid_candidates, key=lambda record: (float(record["distance"]), str(record["candidate_stimulus"])))
    true_candidate = next(
        (
            record
            for record in candidate_records
            if record["candidate_stimulus"] == true_stimulus and record["has_valid_distance"]
        ),
        None,
    )
    if true_candidate is None:
        return {
            "predicted_stimulus": None,
            "best_distance": float("nan"),
            "true_distance": float("nan"),
            "n_candidate_stimuli": len(references),
            "score_status": "excluded",
            "exclusion_reason": "true_stimulus_invalid_reference",
            "is_correct": False,
            "best_overlap_neuron_count": 0,
            "best_overlap_value_count": 0,
        }

    predicted_stimulus = str(best_candidate["candidate_stimulus"])
    return {
        "predicted_stimulus": predicted_stimulus,
        "best_distance": float(best_candidate["distance"]),
        "true_distance": float(true_candidate["distance"]),
        "n_candidate_stimuli": len(references),
        "score_status": SCORED_TRIAL_STATUS,
        "exclusion_reason": "",
        "is_correct": predicted_stimulus == true_stimulus,
        "best_overlap_neuron_count": int(best_candidate["overlap_neuron_count"]),
        "best_overlap_value_count": int(best_candidate["overlap_value_count"]),
    }


def run_leave_one_group_out(
    view: TrialView,
    holdout_column: str,
    holdout_type: str,
    metric: str = DEFAULT_DISTANCE_METRIC,
    min_overlap_neurons: int = MIN_OVERLAP_NEURONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score held-out reliability one individual/date at a time."""

    metadata = view.metadata.reset_index(drop=True)
    trial_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for holdout_value, holdout_frame in metadata.groupby(holdout_column, sort=True, dropna=False):
        holdout_indices = holdout_frame.index.to_numpy()
        train_mask = metadata[holdout_column] != holdout_value
        train_metadata = metadata.loc[train_mask].copy()
        test_metadata = metadata.loc[~train_mask].copy()
        references = _build_stimulus_references(train_metadata, view.values)

        heldout_trial_rows: list[dict[str, object]] = []
        for trial_index in holdout_indices:
            trial_row = metadata.iloc[trial_index]
            score = _score_trial_against_references(
                trial_vector=view.values[trial_index],
                true_stimulus=str(trial_row["stimulus"]),
                references=references,
                metric=metric,
                min_overlap_neurons=min_overlap_neurons,
            )
            row = {
                "view_name": view.name,
                "holdout_type": holdout_type,
                "heldout_group": str(holdout_value),
                "trial_id": str(trial_row["trial_id"]),
                "true_stimulus": str(trial_row["stimulus"]),
                "predicted_stimulus": score["predicted_stimulus"],
                "n_candidate_stimuli": int(score["n_candidate_stimuli"]),
                "best_distance": score["best_distance"],
                "true_distance": score["true_distance"],
                "best_overlap_neuron_count": int(score["best_overlap_neuron_count"]),
                "best_overlap_value_count": int(score["best_overlap_value_count"]),
                "score_status": str(score["score_status"]),
                "exclusion_reason": str(score["exclusion_reason"]),
                "is_correct": bool(score["is_correct"]),
            }
            heldout_trial_rows.append(row)
            trial_rows.append(row)

        holdout_scores_frame = pd.DataFrame(heldout_trial_rows)
        scored_rows = holdout_scores_frame[holdout_scores_frame["score_status"] == SCORED_TRIAL_STATUS]
        excluded_rows = holdout_scores_frame[holdout_scores_frame["score_status"] != SCORED_TRIAL_STATUS]
        group_rows.append(
            {
                "view_name": view.name,
                "holdout_type": holdout_type,
                "heldout_group": str(holdout_value),
                "n_training_trials": int(len(train_metadata)),
                "n_heldout_trials": int(len(test_metadata)),
                "n_trials_scored": int(len(scored_rows)),
                "n_trials_excluded": int(len(excluded_rows)),
                "accuracy": float(scored_rows["is_correct"].mean()) if not scored_rows.empty else float("nan"),
                "mean_true_distance": float(scored_rows["true_distance"].mean()) if not scored_rows.empty else float("nan"),
                "mean_best_distance": float(scored_rows["best_distance"].mean()) if not scored_rows.empty else float("nan"),
                "excluded_stimuli": "|".join(sorted(excluded_rows["true_stimulus"].unique())) if not excluded_rows.empty else "",
            }
        )

    trial_frame = pd.DataFrame(trial_rows)
    group_frame = pd.DataFrame(group_rows)
    summary_frame = summarize_holdout_results(group_frame)
    return trial_frame, group_frame, summary_frame


def summarize_holdout_results(group_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize leave-one-group-out results at the view level."""

    if group_results.empty:
        return pd.DataFrame(
            columns=[
                "view_name",
                "holdout_type",
                "n_groups_total",
                "n_groups_scored",
                "n_groups_excluded",
                "n_trials_scored",
                "accuracy_mean",
                "accuracy_median",
                "mean_true_distance",
                "mean_best_distance",
            ]
        )

    rows: list[dict[str, object]] = []
    grouping_columns = [column for column in group_results.columns if column in ("view_name", "holdout_type", "source_date")]
    for group_key, group in group_results.groupby(grouping_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_metadata = dict(zip(grouping_columns, group_key))
        scored_groups = group[group["n_trials_scored"] > 0]
        row = {
            "n_groups_total": int(len(group)),
            "n_groups_scored": int(len(scored_groups)),
            "n_groups_excluded": int(len(group) - len(scored_groups)),
            "n_trials_scored": int(group["n_trials_scored"].sum()),
            "accuracy_mean": float(scored_groups["accuracy"].mean()) if not scored_groups.empty else float("nan"),
            "accuracy_median": float(scored_groups["accuracy"].median()) if not scored_groups.empty else float("nan"),
            "mean_true_distance": float(scored_groups["mean_true_distance"].mean()) if not scored_groups.empty else float("nan"),
            "mean_best_distance": float(scored_groups["mean_best_distance"].mean()) if not scored_groups.empty else float("nan"),
        }
        row.update(group_metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_per_date_loio(
    view: TrialView,
    metric: str = DEFAULT_DISTANCE_METRIC,
    min_overlap_neurons: int = MIN_OVERLAP_NEURONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run LOIO separately within each date for one view."""

    metadata = view.metadata.reset_index(drop=True)
    trial_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []

    for source_date, date_frame in metadata.groupby("date", sort=True, dropna=False):
        date_indices = date_frame.index.to_numpy()
        date_view = TrialView(
            name=view.name,
            timepoints=view.timepoints,
            metadata=date_frame.reset_index(drop=True),
            values=view.values[date_indices].copy(),
        )
        trial_frame, group_frame, _ = run_leave_one_group_out(
            date_view,
            holdout_column="individual_id",
            holdout_type="per_date_individual",
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )
        trial_frame["source_date"] = str(source_date)
        group_frame["source_date"] = str(source_date)
        trial_frame["analysis_role"] = SUPPLEMENTARY_ANALYSIS_LABEL
        group_frame["analysis_role"] = SUPPLEMENTARY_ANALYSIS_LABEL
        trial_frames.append(trial_frame)
        group_frames.append(group_frame)

    trial_frame = _concat_frames(trial_frames)
    group_frame = _concat_frames(group_frames)
    summary_frame = summarize_holdout_results(group_frame)
    if not summary_frame.empty:
        summary_frame["analysis_role"] = SUPPLEMENTARY_ANALYSIS_LABEL
    return trial_frame, group_frame, summary_frame


def _iter_balanced_split_indices(indices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shuffled = rng.permutation(np.asarray(indices, dtype=int))
    midpoint = len(shuffled) // 2
    if midpoint == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return shuffled[:midpoint], shuffled[midpoint:]


def run_split_half_reliability(
    view: TrialView,
    n_repeats: int = 100,
    seed: int = 0,
    metric: str = DEFAULT_DISTANCE_METRIC,
    min_overlap_neurons: int = MIN_OVERLAP_NEURONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run seedable split-half reliability as a supplementary analysis."""

    rng = np.random.default_rng(seed)
    metadata = view.metadata.reset_index(drop=True)
    repeat_rows: list[dict[str, object]] = []

    for repeat in range(n_repeats):
        half_a_indices: dict[str, np.ndarray] = {}
        half_b_indices: dict[str, np.ndarray] = {}
        for stimulus, stimulus_rows in metadata.groupby("stimulus", sort=True, dropna=False):
            left, right = _iter_balanced_split_indices(stimulus_rows.index.to_numpy(), rng)
            if len(left) == 0 or len(right) == 0:
                continue
            half_a_indices[str(stimulus)] = left
            half_b_indices[str(stimulus)] = right

        stimuli = sorted(set(half_a_indices) & set(half_b_indices))
        if not stimuli:
            repeat_rows.append(
                {
                    "view_name": view.name,
                    "repeat": repeat,
                    "n_stimuli_scored": 0,
                    "n_stimuli_excluded": 0,
                    "accuracy": float("nan"),
                    "same_mean_distance": float("nan"),
                    "different_mean_distance": float("nan"),
                    "distance_gap": float("nan"),
                    "analysis_role": SUPPLEMENTARY_ANALYSIS_LABEL,
                }
            )
            continue

        half_a_refs = {
            stimulus: _mean_trials(view.values[indices])
            for stimulus, indices in half_a_indices.items()
            if stimulus in stimuli
        }
        half_b_refs = {
            stimulus: _mean_trials(view.values[indices])
            for stimulus, indices in half_b_indices.items()
            if stimulus in stimuli
        }

        scored = 0
        excluded = 0
        correct = 0
        same_distances: list[float] = []
        different_distances: list[float] = []

        for true_stimulus in stimuli:
            candidate_rows: list[dict[str, object]] = []
            for candidate_stimulus in stimuli:
                comparison = compare_trial_arrays(
                    half_a_refs[true_stimulus],
                    half_b_refs[candidate_stimulus],
                    metric=metric,
                    min_overlap_neurons=min_overlap_neurons,
                )
                if comparison["has_valid_distance"]:
                    candidate_rows.append(
                        {
                            "candidate_stimulus": candidate_stimulus,
                            "distance": float(comparison["distance"]),
                        }
                    )
                    if candidate_stimulus == true_stimulus:
                        same_distances.append(float(comparison["distance"]))
                    else:
                        different_distances.append(float(comparison["distance"]))

            if not candidate_rows:
                excluded += 1
                continue

            best_candidate = min(candidate_rows, key=lambda row: (row["distance"], row["candidate_stimulus"]))
            scored += 1
            correct += int(best_candidate["candidate_stimulus"] == true_stimulus)

        repeat_rows.append(
            {
                "view_name": view.name,
                "repeat": repeat,
                "n_stimuli_scored": scored,
                "n_stimuli_excluded": excluded,
                "accuracy": (correct / scored) if scored else float("nan"),
                "same_mean_distance": float(np.mean(same_distances)) if same_distances else float("nan"),
                "different_mean_distance": float(np.mean(different_distances)) if different_distances else float("nan"),
                "distance_gap": (
                    float(np.mean(different_distances) - np.mean(same_distances))
                    if same_distances and different_distances
                    else float("nan")
                ),
                "analysis_role": SUPPLEMENTARY_ANALYSIS_LABEL,
            }
        )

    repeat_frame = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, object]] = []
    for view_name, group in repeat_frame.groupby("view_name", sort=False, dropna=False):
        scored_repeats = group[group["n_stimuli_scored"] > 0]
        summary_rows.append(
            {
                "view_name": view_name,
                "n_repeats": int(len(group)),
                "n_scored_repeats": int(len(scored_repeats)),
                "accuracy_mean": float(scored_repeats["accuracy"].mean()) if not scored_repeats.empty else float("nan"),
                "accuracy_median": float(scored_repeats["accuracy"].median()) if not scored_repeats.empty else float("nan"),
                "same_mean_distance": float(scored_repeats["same_mean_distance"].mean()) if not scored_repeats.empty else float("nan"),
                "different_mean_distance": float(scored_repeats["different_mean_distance"].mean()) if not scored_repeats.empty else float("nan"),
                "distance_gap": float(scored_repeats["distance_gap"].mean()) if not scored_repeats.empty else float("nan"),
                "analysis_role": SUPPLEMENTARY_ANALYSIS_LABEL,
            }
        )
    return repeat_frame, pd.DataFrame(summary_rows)


def summarize_overlap_qc(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Summarize overlap-neuron usage for QC reporting."""

    valid = comparisons[comparisons["comparison_status"] == VALID_COMPARISON_STATUS].copy()
    rows: list[dict[str, object]] = []
    for view_name, group in valid.groupby("view_name", sort=False, dropna=False):
        rows.append(
            {
                "view_name": view_name,
                "n_valid_comparisons": int(len(group)),
                "overlap_neuron_min": int(group["overlap_neuron_count"].min()) if not group.empty else 0,
                "overlap_neuron_median": float(group["overlap_neuron_count"].median()) if not group.empty else float("nan"),
                "overlap_neuron_max": int(group["overlap_neuron_count"].max()) if not group.empty else 0,
                "overlap_value_min": int(group["overlap_value_count"].min()) if not group.empty else 0,
                "overlap_value_median": float(group["overlap_value_count"].median()) if not group.empty else float("nan"),
                "overlap_value_max": int(group["overlap_value_count"].max()) if not group.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_stimulus_distance_pairs(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid pairwise distances into stimulus-by-stimulus summaries."""

    valid = comparisons[comparisons["comparison_status"] == VALID_COMPARISON_STATUS].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "view_name",
                "stimulus_left",
                "stimulus_right",
                "same_stimulus",
                "n_pairs",
                "mean_distance",
                "median_distance",
                "analysis_role",
            ]
        )

    stimulus_a = valid["stimulus_a"].astype(str)
    stimulus_b = valid["stimulus_b"].astype(str)
    valid["stimulus_left"] = np.where(stimulus_a <= stimulus_b, stimulus_a, stimulus_b)
    valid["stimulus_right"] = np.where(stimulus_a <= stimulus_b, stimulus_b, stimulus_a)

    rows: list[dict[str, object]] = []
    for (view_name, stimulus_left, stimulus_right), group in valid.groupby(
        ["view_name", "stimulus_left", "stimulus_right"],
        sort=False,
        dropna=False,
    ):
        rows.append(
            {
                "view_name": view_name,
                "stimulus_left": stimulus_left,
                "stimulus_right": stimulus_right,
                "same_stimulus": bool(stimulus_left == stimulus_right),
                "n_pairs": int(len(group)),
                "mean_distance": float(group["distance"].mean()),
                "median_distance": float(group["distance"].median()),
                "analysis_role": SUPPLEMENTARY_ANALYSIS_LABEL,
            }
        )
    return pd.DataFrame(rows)


def run_reliability_pipeline(
    inputs: ReliabilityInputs,
    metric: str = DEFAULT_DISTANCE_METRIC,
    min_overlap_neurons: int = MIN_OVERLAP_NEURONS,
    split_half_repeats: int = 100,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Run the full Stage 1 core analysis over every view."""

    views = build_trial_views(inputs.metadata, inputs.tensor)
    comparisons = pd.concat(
        [
            compute_pairwise_distances(view, metric=metric, min_overlap_neurons=min_overlap_neurons)
            for view in views.values()
        ],
        ignore_index=True,
    )

    loio_trials: list[pd.DataFrame] = []
    loio_groups: list[pd.DataFrame] = []
    loio_summaries: list[pd.DataFrame] = []
    lodo_trials: list[pd.DataFrame] = []
    lodo_groups: list[pd.DataFrame] = []
    lodo_summaries: list[pd.DataFrame] = []
    per_date_loio_trials: list[pd.DataFrame] = []
    per_date_loio_groups: list[pd.DataFrame] = []
    per_date_loio_summaries: list[pd.DataFrame] = []
    split_half_results: list[pd.DataFrame] = []
    split_half_summaries: list[pd.DataFrame] = []

    for view in views.values():
        loio_trial_frame, loio_group_frame, loio_summary_frame = run_leave_one_group_out(
            view,
            holdout_column="individual_id",
            holdout_type="individual",
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )
        lodo_trial_frame, lodo_group_frame, lodo_summary_frame = run_leave_one_group_out(
            view,
            holdout_column="date",
            holdout_type="date",
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )
        per_date_loio_trial_frame, per_date_loio_group_frame, per_date_loio_summary_frame = run_per_date_loio(
            view,
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )
        split_half_frame, split_half_summary = run_split_half_reliability(
            view,
            n_repeats=split_half_repeats,
            seed=seed,
            metric=metric,
            min_overlap_neurons=min_overlap_neurons,
        )

        loio_trials.append(loio_trial_frame)
        loio_groups.append(loio_group_frame)
        loio_summaries.append(loio_summary_frame)
        lodo_trials.append(lodo_trial_frame)
        lodo_groups.append(lodo_group_frame)
        lodo_summaries.append(lodo_summary_frame)
        per_date_loio_trials.append(per_date_loio_trial_frame)
        per_date_loio_groups.append(per_date_loio_group_frame)
        per_date_loio_summaries.append(per_date_loio_summary_frame)
        split_half_results.append(split_half_frame)
        split_half_summaries.append(split_half_summary)

    metadata_summary = pd.DataFrame(
        [
            {
                "n_trials": int(inputs.metadata["trial_id"].nunique()),
                "n_individuals": int(inputs.metadata["individual_id"].nunique()),
                "n_dates": int(inputs.metadata["date"].nunique()),
                "n_stimuli": int(inputs.metadata["stimulus"].nunique()),
            }
        ]
    )

    within_date_cross_individual_comparisons, within_date_cross_individual_summary = (
        summarize_within_date_cross_individual_same_vs_different(comparisons)
    )
    stimulus_distance_pairs = summarize_stimulus_distance_pairs(comparisons)

    return {
        "metadata": inputs.metadata.copy(),
        "metadata_summary": metadata_summary,
        "comparisons": comparisons,
        "same_vs_different_summary": summarize_same_vs_different(comparisons),
        "loio_trials": pd.concat(loio_trials, ignore_index=True),
        "loio_groups": pd.concat(loio_groups, ignore_index=True),
        "loio_summary": pd.concat(loio_summaries, ignore_index=True),
        "lodo_trials": pd.concat(lodo_trials, ignore_index=True),
        "lodo_groups": pd.concat(lodo_groups, ignore_index=True),
        "lodo_summary": pd.concat(lodo_summaries, ignore_index=True),
        "per_date_loio_trials": _concat_frames(per_date_loio_trials),
        "per_date_loio_groups": _concat_frames(per_date_loio_groups),
        "per_date_loio_summary": _concat_frames(per_date_loio_summaries),
        "split_half_results": pd.concat(split_half_results, ignore_index=True),
        "split_half_summary": pd.concat(split_half_summaries, ignore_index=True),
        "overlap_qc_summary": summarize_overlap_qc(comparisons),
        "within_date_cross_individual_comparisons": within_date_cross_individual_comparisons,
        "within_date_cross_individual_summary": within_date_cross_individual_summary,
        "stimulus_distance_pairs": stimulus_distance_pairs,
    }
