"""Reliability permutation and bootstrap helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.reliability import (
    SCORED_TRIAL_STATUS,
    TrialView,
    VALID_COMPARISON_STATUS,
    add_individual_id,
)


def _coerce_seed(random_state: int | None = None, seed: int | None = None) -> int | None:
    return random_state if random_state is not None else seed


def permute_stimulus_labels(
    metadata: pd.DataFrame,
    group_column: str = "date",
    label_column: str = "stimulus",
    random_state: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Permute stimulus labels within groups while preserving per-group counts."""

    rng = np.random.default_rng(_coerce_seed(random_state=random_state, seed=seed))
    permuted = metadata[["trial_id", group_column, label_column]].copy()
    permuted[label_column] = permuted[label_column].astype(str)
    for _, group in permuted.groupby(group_column, sort=False, dropna=False):
        group_index = group.index.to_numpy()
        permuted.loc[group_index, label_column] = rng.permutation(group[label_column].to_numpy())
    return permuted


def build_permutation_null(
    view: TrialView,
    n_iterations: int = 100,
    group_column: str = "date",
    label_column: str = "stimulus",
    random_state: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return per-iteration permuted trial labels for one view."""

    metadata = view.metadata.copy()
    rows: list[dict[str, object]] = []
    base_seed = _coerce_seed(random_state=random_state, seed=seed) or 0
    for iteration in range(n_iterations):
        permuted = permute_stimulus_labels(
            metadata,
            group_column=group_column,
            label_column=label_column,
            random_state=base_seed + iteration,
        )
        for row in permuted.itertuples(index=False):
            rows.append(
                {
                    "view_name": view.name,
                    "iteration": iteration,
                    "trial_id": str(row.trial_id),
                    label_column: str(getattr(row, label_column)),
                }
            )
    return pd.DataFrame(rows)


def score_permutation_null(
    comparisons: pd.DataFrame,
    permutation_samples: pd.DataFrame,
    label_column: str = "stimulus",
) -> pd.DataFrame:
    """Score permutation samples against fixed pairwise distances."""

    valid = comparisons[comparisons["comparison_status"] == VALID_COMPARISON_STATUS].copy()
    rows: list[dict[str, object]] = []

    for iteration, sample in permutation_samples.groupby("iteration", sort=False, dropna=False):
        label_map = sample.set_index("trial_id")[label_column].astype(str).to_dict()
        permuted_left = valid["trial_id_a"].astype(str).map(label_map)
        permuted_right = valid["trial_id_b"].astype(str).map(label_map)
        same_mask = permuted_left.eq(permuted_right)
        different_mask = ~same_mask
        for view_name, view_group in valid.groupby("view_name", sort=False, dropna=False):
            view_mask = valid["view_name"].astype(str) == str(view_name)
            view_same = same_mask & view_mask
            view_different = different_mask & view_mask
            same_distances = valid.loc[view_same, "distance"].to_numpy(dtype=float)
            different_distances = valid.loc[view_different, "distance"].to_numpy(dtype=float)
            rows.append(
                {
                    "view_name": str(view_name),
                    "iteration": int(iteration),
                    "same_count": int(view_same.sum()),
                    "different_count": int(view_different.sum()),
                    "same_mean_distance": float(np.mean(same_distances)) if same_distances.size else float("nan"),
                    "different_mean_distance": float(np.mean(different_distances)) if different_distances.size else float("nan"),
                    "distance_gap": (
                        float(np.mean(different_distances) - np.mean(same_distances))
                        if same_distances.size and different_distances.size
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_permutation_null(
    observed_summary: pd.DataFrame,
    permutation_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Combine observed gaps with permutation null summaries."""

    observed_map = observed_summary.set_index("view_name")["distance_gap"].to_dict()
    rows: list[dict[str, object]] = []
    for view_name, group in permutation_scores.groupby("view_name", sort=False, dropna=False):
        observed_gap = float(observed_map.get(view_name, float("nan")))
        null_values = group["distance_gap"].to_numpy(dtype=float)
        rows.append(
            {
                "view_name": view_name,
                "observed_distance_gap": observed_gap,
                "permutation_iterations": int(len(group)),
                "null_mean_distance_gap": float(np.mean(null_values)) if null_values.size else float("nan"),
                "null_std_distance_gap": float(np.std(null_values, ddof=0)) if null_values.size else float("nan"),
                "null_ci_lower": float(pd.Series(null_values).quantile(0.025)) if null_values.size else float("nan"),
                "null_ci_upper": float(pd.Series(null_values).quantile(0.975)) if null_values.size else float("nan"),
                "p_value": (
                    float((1 + np.sum(null_values >= observed_gap)) / (len(null_values) + 1))
                    if null_values.size and np.isfinite(observed_gap)
                    else float("nan")
                ),
                "null_percentile": (
                    float(np.mean(null_values <= observed_gap))
                    if null_values.size and np.isfinite(observed_gap)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_grouped_bootstrap(
    view: TrialView,
    group_column: str = "individual_id",
    n_iterations: int = 100,
    random_state: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return bootstrap group samples for one view."""

    metadata = view.metadata.copy()
    if group_column not in metadata.columns and group_column == "individual_id":
        metadata = add_individual_id(metadata)

    rng = np.random.default_rng(_coerce_seed(random_state=random_state, seed=seed))
    group_index_map = {
        str(group_id): group_rows.index.to_numpy()
        for group_id, group_rows in metadata.groupby(group_column, sort=False, dropna=False)
    }
    group_ids = list(group_index_map)
    rows: list[dict[str, object]] = []
    for iteration in range(n_iterations):
        sampled_groups = tuple(rng.choice(group_ids, size=len(group_ids), replace=True).tolist())
        sampled_indices = np.concatenate([group_index_map[group_id] for group_id in sampled_groups])
        sampled_values = view.values[sampled_indices]
        rows.append(
            {
                "view_name": view.name,
                "iteration": iteration,
                "sampled_groups": sampled_groups,
                "score": float(np.nanmean(sampled_values)) if sampled_values.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_grouped_bootstrap_from_scores(
    trial_scores: pd.DataFrame,
    group_column: str = "heldout_group",
    score_column: str = "is_correct",
    n_iterations: int = 500,
    random_state: int | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap held-out trial scores by resampling groups with replacement."""

    rng = np.random.default_rng(_coerce_seed(random_state=random_state, seed=seed))
    eligible = trial_scores[trial_scores["score_status"] == SCORED_TRIAL_STATUS].copy()
    iteration_rows: list[dict[str, object]] = []

    for view_name, view_frame in eligible.groupby("view_name", sort=False, dropna=False):
        groups = {str(group): frame.copy() for group, frame in view_frame.groupby(group_column, sort=False, dropna=False)}
        group_ids = list(groups)
        if not group_ids:
            continue
        for iteration in range(n_iterations):
            sampled_group_ids = tuple(rng.choice(group_ids, size=len(group_ids), replace=True).tolist())
            sampled_group_scores = [
                float(groups[group_id][score_column].astype(float).mean())
                for group_id in sampled_group_ids
            ]
            sampled_rows = int(sum(len(groups[group_id]) for group_id in sampled_group_ids))
            iteration_rows.append(
                {
                    "view_name": view_name,
                    "iteration": iteration,
                    "sampled_groups": sampled_group_ids,
                    "bootstrap_iterations": n_iterations,
                    "n_groups_sampled": int(len(sampled_group_ids)),
                    "n_rows_sampled": sampled_rows,
                    "score_mean": float(np.mean(sampled_group_scores)) if sampled_group_scores else float("nan"),
                }
            )

    iteration_frame = pd.DataFrame(iteration_rows)
    summary_rows: list[dict[str, object]] = []
    for view_name, group in iteration_frame.groupby("view_name", sort=False, dropna=False):
        summary_rows.append(
            {
                "view_name": view_name,
                "bootstrap_iterations": int(len(group)),
                "bootstrap_mean": float(group["score_mean"].mean()),
                "ci_lower": float(group["score_mean"].quantile(0.025)),
                "ci_upper": float(group["score_mean"].quantile(0.975)),
            }
        )
    return iteration_frame, pd.DataFrame(summary_rows)


def build_final_summary_table(
    same_vs_different_summary: pd.DataFrame,
    loio_summary: pd.DataFrame,
    lodo_summary: pd.DataFrame,
    split_half_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the main reliability summaries into one per-view table."""

    loio_view = loio_summary[loio_summary["holdout_type"] == "individual"].copy()
    lodo_view = lodo_summary[lodo_summary["holdout_type"] == "date"].copy()
    permutation_view = permutation_summary.rename(columns={"observed_distance_gap": "permutation_observed_distance_gap"})
    bootstrap_view = bootstrap_summary.rename(
        columns={
            "bootstrap_mean": "bootstrap_loio_accuracy_mean",
            "ci_lower": "bootstrap_loio_ci_lower",
            "ci_upper": "bootstrap_loio_ci_upper",
        }
    )

    final = same_vs_different_summary.merge(
        loio_view[["view_name", "accuracy_mean", "accuracy_median"]].rename(
            columns={
                "accuracy_mean": "loio_accuracy_mean",
                "accuracy_median": "loio_accuracy_median",
            }
        ),
        on="view_name",
        how="left",
    )
    final = final.merge(
        lodo_view[["view_name", "accuracy_mean", "accuracy_median"]].rename(
            columns={
                "accuracy_mean": "lodo_accuracy_mean",
                "accuracy_median": "lodo_accuracy_median",
            }
        ),
        on="view_name",
        how="left",
    )
    final = final.merge(
        split_half_summary[["view_name", "accuracy_mean", "accuracy_median"]].rename(
            columns={
                "accuracy_mean": "split_half_accuracy_mean",
                "accuracy_median": "split_half_accuracy_median",
            }
        ),
        on="view_name",
        how="left",
    )
    final = final.merge(permutation_view, on="view_name", how="left")
    final = final.merge(bootstrap_view, on="view_name", how="left")
    return final
