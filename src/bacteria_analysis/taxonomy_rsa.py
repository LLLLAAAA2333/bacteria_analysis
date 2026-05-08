"""Reusable taxonomy-class RSA helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassCandidate:
    model_id: str
    taxonomy_level: str
    category: str
    metabolites: tuple[str, ...]
    chemical: pd.DataFrame
    primary_rank_matrix: np.ndarray
    primary_observed_rsa: float
    full_trajectory_rsa: float
    n_pairs: int


def class_observed_scores(candidates: list[ClassCandidate]) -> pd.DataFrame:
    rows = [
        {
            "model_id": candidate.model_id,
            "taxonomy_level": candidate.taxonomy_level,
            "category": candidate.category,
            "n_features": len(candidate.metabolites),
            "n_pairs": candidate.n_pairs,
            "response_window_rsa": candidate.primary_observed_rsa,
            "full_trajectory_rsa": candidate.full_trajectory_rsa,
            "metabolites": " | ".join(candidate.metabolites),
        }
        for candidate in candidates
    ]
    return pd.DataFrame(rows).sort_values("response_window_rsa", ascending=False).reset_index(drop=True)


def fixed_class_permutations(
    *,
    candidates: list[ClassCandidate],
    labels: list[str],
    neural_ranks: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []

    for candidate in candidates:
        null_values = permuted_scores(
            candidate.primary_rank_matrix,
            neural_ranks=neural_ranks,
            upper_i=upper_i,
            upper_j=upper_j,
            n_permutations=n_permutations,
            rng=rng,
        )
        for iteration, value in enumerate(null_values):
            null_rows.append(
                {
                    "iteration": iteration,
                    "model_id": candidate.model_id,
                    "category": candidate.category,
                    "rsa_similarity": value,
                }
            )

        observed = candidate.primary_observed_rsa
        summary_rows.append(
            {
                "model_id": candidate.model_id,
                "taxonomy_level": candidate.taxonomy_level,
                "category": candidate.category,
                "n_features": len(candidate.metabolites),
                "n_pairs": candidate.n_pairs,
                "observed_rsa": observed,
                "observed_percentile": percentile(observed, null_values),
                "p_one_sided_ge": empirical_p(observed, null_values),
                "null_mean": nan_stat(np.mean, null_values),
                "null_sd": nan_stat(np.std, null_values),
                "null_q95": nan_quantile(null_values, 0.95),
                "null_q99": nan_quantile(null_values, 0.99),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("observed_rsa", ascending=False).reset_index(drop=True)
    summary["p_fdr_bh"] = benjamini_hochberg(summary["p_one_sided_ge"].to_numpy(float))
    return summary, pd.DataFrame(null_rows)


def reselection_stability(
    *,
    candidates: list[ClassCandidate],
    primary_neural: pd.DataFrame,
    labels: list[str],
    date_map: dict[str, str],
    n_resamples: int,
    resample_fraction: float,
    top_k: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for resample_index in range(n_resamples):
        selected = date_stratified_resample(labels, date_map, resample_fraction, rng)
        scores = score_candidates_on_subset(candidates, primary_neural, selected)
        for selected_rank, row in enumerate(scores.itertuples(index=False), start=1):
            rows.append(
                {
                    "resample_index": resample_index,
                    "model_id": row.model_id,
                    "category": row.category,
                    "selected_rank": selected_rank,
                    "rank": selected_rank,
                    "selected_rsa": row.rsa_similarity,
                    "rsa_similarity": row.rsa_similarity,
                    "n_pairs": row.n_pairs,
                    "n_stimuli": len(selected),
                    "is_top1": selected_rank == 1,
                    "is_top3": selected_rank <= 3,
                    "is_top5": selected_rank <= 5,
                    "stimuli": ";".join(selected),
                }
            )

    runs = pd.DataFrame(rows)
    if runs.empty:
        columns = [
            "model_id",
            "category",
            "n_features",
            "top1_count",
            "top3_count",
            "top5_count",
            "top1_frequency",
            "top3_frequency",
            "top5_frequency",
            "mean_selected_rank",
            "mean_selected_rsa",
            "n_resamples",
        ]
        return runs, pd.DataFrame(columns=columns)

    feature_counts = {candidate.model_id: len(candidate.metabolites) for candidate in candidates}
    summary = (
        runs.groupby(["model_id", "category"], as_index=False)
        .agg(
            top1_count=("is_top1", "sum"),
            top3_count=("is_top3", "sum"),
            top5_count=("is_top5", "sum"),
            top1_frequency=("is_top1", "mean"),
            top3_frequency=("is_top3", "mean"),
            top5_frequency=("is_top5", "mean"),
            mean_selected_rank=("selected_rank", "mean"),
            mean_selected_rsa=("selected_rsa", "mean"),
            n_resamples=("resample_index", "nunique"),
        )
        .sort_values(["top1_frequency", "top3_frequency", "mean_selected_rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    summary["n_features"] = summary["model_id"].map(feature_counts).astype(int)
    return runs, summary.head(max(top_k, len(summary)))


def full_search_permutation(
    *,
    candidates: list[ClassCandidate],
    labels: list[str],
    neural_ranks: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_best = max(candidates, key=lambda candidate: candidate.primary_observed_rsa)
    null_rows: list[dict[str, object]] = []
    n_labels = len(labels)

    for iteration in range(n_permutations):
        permutation = rng.permutation(n_labels)
        best_value = float("nan")
        best_candidate: ClassCandidate | None = None
        for candidate in candidates:
            values = candidate.primary_rank_matrix[np.ix_(permutation, permutation)][upper_i, upper_j]
            score = pearson(neural_ranks, values)
            if best_candidate is None or _nan_safe_greater(score, best_value):
                best_value = score
                best_candidate = candidate
        null_rows.append(
            {
                "iteration": iteration,
                "best_null_model_id": best_candidate.model_id if best_candidate else "",
                "best_null_category": best_candidate.category if best_candidate else "",
                "best_null_rsa": best_value,
            }
        )

    null = pd.DataFrame(null_rows)
    null_values = null["best_null_rsa"].to_numpy(float) if "best_null_rsa" in null else np.array([], dtype=float)
    summary = pd.DataFrame(
        [
            {
                "observed_best_model_id": observed_best.model_id,
                "observed_best_category": observed_best.category,
                "observed_best_rsa": observed_best.primary_observed_rsa,
                "search_corrected_p": empirical_p(observed_best.primary_observed_rsa, null_values),
                "best_null_q95": nan_quantile(null_values, 0.95),
                "best_null_q99": nan_quantile(null_values, 0.99),
                "n_permutations": int(n_permutations),
            }
        ]
    )
    return summary, null


def build_final_shortlist(
    observed_scores: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    reselection_summary: pd.DataFrame,
    search_summary: pd.DataFrame,
) -> pd.DataFrame:
    scorecard = (
        observed_scores.merge(
            fixed_summary.loc[
                :,
                ["model_id", "observed_percentile", "p_one_sided_ge", "p_fdr_bh", "null_q95", "null_q99"],
            ],
            on="model_id",
            how="left",
        )
        .merge(
            reselection_summary.loc[
                :,
                ["model_id", "top1_frequency", "top3_frequency", "top5_frequency", "mean_selected_rank"],
            ],
            on="model_id",
            how="left",
        )
        .sort_values("response_window_rsa", ascending=False)
        .reset_index(drop=True)
    )
    observed_best = str(search_summary["observed_best_model_id"].iloc[0])
    search_p = float(search_summary["search_corrected_p"].iloc[0])
    scorecard["fixed_signal_pass"] = (
        scorecard["response_window_rsa"].to_numpy(float) > scorecard["null_q95"].to_numpy(float)
    )
    scorecard["reselection_preferred"] = scorecard["top3_frequency"].fillna(0.0).ge(0.20)
    scorecard["is_observed_best"] = scorecard["model_id"].eq(observed_best)
    scorecard["search_corrected_p_for_best"] = np.where(scorecard["is_observed_best"], search_p, np.nan)
    return scorecard.loc[scorecard["fixed_signal_pass"] | scorecard["reselection_preferred"] | scorecard["is_observed_best"]]


def class_chemical_rdm_similarity(
    *,
    candidates: list[ClassCandidate],
    labels: list[str],
    upper_i: np.ndarray,
    upper_j: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectors = {
        candidate.model_id: candidate.chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
        for candidate in candidates
    }
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            rows.append(
                {
                    "left_model_id": left.model_id,
                    "left_category": left.category,
                    "left_response_window_rsa": left.primary_observed_rsa,
                    "right_model_id": right.model_id,
                    "right_category": right.category,
                    "right_response_window_rsa": right.primary_observed_rsa,
                    "chemical_rdm_rsa": spearman(vectors[left.model_id], vectors[right.model_id]),
                    "left_n_features": len(left.metabolites),
                    "right_n_features": len(right.metabolites),
                }
            )
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise = pairwise.sort_values("chemical_rdm_rsa", ascending=False).reset_index(drop=True)
    summary = _similarity_summary(pairwise.get("chemical_rdm_rsa", pd.Series(dtype=float)), len(candidates))
    top_similarity = _top_class_similarity(pairwise, candidates[0].model_id) if not pairwise.empty else pairwise.copy()
    return pairwise, summary, top_similarity


def class_vs_full_chemical_similarity(
    *,
    candidates: list[ClassCandidate],
    full_chemical: pd.DataFrame,
    labels: list[str],
    upper_i: np.ndarray,
    upper_j: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_values = full_chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
    rows = []
    for candidate in candidates:
        class_values = candidate.chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
        rows.append(
            {
                "model_id": candidate.model_id,
                "category": candidate.category,
                "n_features": len(candidate.metabolites),
                "class_vs_full_chemical_rdm_rsa": spearman(class_values, full_values),
                "response_window_rsa": candidate.primary_observed_rsa,
                "full_trajectory_rsa": candidate.full_trajectory_rsa,
            }
        )
    similarity = pd.DataFrame(rows).sort_values("class_vs_full_chemical_rdm_rsa", ascending=False).reset_index(drop=True)
    summary = _similarity_summary(similarity["class_vs_full_chemical_rdm_rsa"], len(similarity))
    return similarity, summary


def score_candidates_on_subset(
    candidates: list[ClassCandidate],
    primary_neural: pd.DataFrame,
    selected_labels: list[str],
) -> pd.DataFrame:
    labels = [label for label in selected_labels if label in primary_neural.index and label in primary_neural.columns]
    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    neural_values = primary_neural.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
    neural_ranks = avg_rank(neural_values)
    rows = []
    for candidate in candidates:
        chemical_values = candidate.chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
        rows.append(
            {
                "model_id": candidate.model_id,
                "category": candidate.category,
                "rsa_similarity": pearson(neural_ranks, avg_rank(chemical_values)),
                "n_pairs": int(len(neural_values)),
            }
        )
    return pd.DataFrame(rows).sort_values("rsa_similarity", ascending=False, na_position="last").reset_index(drop=True)


def date_stratified_resample(
    labels: list[str],
    date_map: dict[str, str],
    fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    target_size = max(3, min(len(labels), int(np.floor(len(labels) * fraction))))
    if not _can_date_stratify(labels, date_map):
        return sorted(rng.choice(np.asarray(labels, dtype=object), size=target_size, replace=False).astype(str).tolist())

    by_date: dict[str, list[str]] = {}
    for label in labels:
        by_date.setdefault(str(date_map[str(label)]), []).append(str(label))

    selected: list[str] = []
    for group_labels in by_date.values():
        group_size = max(1, int(np.floor(len(group_labels) * fraction)))
        group_size = min(group_size, len(group_labels))
        selected.extend(rng.choice(np.asarray(group_labels, dtype=object), size=group_size, replace=False).astype(str))

    remaining = [label for label in labels if label not in set(selected)]
    if len(selected) < target_size and remaining:
        needed = min(target_size - len(selected), len(remaining))
        selected.extend(rng.choice(np.asarray(remaining, dtype=object), size=needed, replace=False).astype(str))
    if len(selected) > target_size:
        selected = rng.choice(np.asarray(selected, dtype=object), size=target_size, replace=False).astype(str).tolist()
    return [label for label in labels if label in set(selected)]


def permuted_scores(
    rank_matrix: np.ndarray,
    *,
    neural_ranks: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_labels = rank_matrix.shape[0]
    null_values = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        permutation = rng.permutation(n_labels)
        values = rank_matrix[np.ix_(permutation, permutation)][upper_i, upper_j]
        null_values[iteration] = pearson(neural_ranks, values)
    return null_values


def symmetric_rank_matrix(
    n_labels: int,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    ranks: np.ndarray,
) -> np.ndarray:
    matrix = np.full((n_labels, n_labels), np.nan, dtype=float)
    matrix[upper_i, upper_j] = ranks
    matrix[upper_j, upper_i] = ranks
    return matrix


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 3:
        return float("nan")
    return pearson(avg_rank(left_values[mask]), avg_rank(right_values[mask]))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 3:
        return float("nan")
    left_centered = left_values[mask] - np.mean(left_values[mask])
    right_centered = right_values[mask] - np.mean(right_values[mask])
    denominator = np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered))
    if denominator == 0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def avg_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float), copy=False).rank(method="average").to_numpy(float)


def empirical_p(observed: float, values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(observed) or finite.size == 0:
        return float("nan")
    return float((1 + np.sum(finite >= observed)) / (finite.size + 1))


def percentile(observed: float, values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(observed) or finite.size == 0:
        return float("nan")
    return float(np.mean(finite <= observed) * 100.0)


def nan_quantile(values: np.ndarray | pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(np.quantile(finite, quantile)) if finite.size else float("nan")


def nan_stat(func: object, values: np.ndarray | pd.Series) -> float:
    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return float(func(finite)) if finite.size else float("nan")


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return adjusted
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    scale = finite_values.size / np.arange(1, finite_values.size + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    finite_indices = np.flatnonzero(finite_mask)
    adjusted[finite_indices[order]] = ranked_adjusted
    return adjusted


def _similarity_summary(values: pd.Series, n_classes: int) -> pd.DataFrame:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return pd.DataFrame(
        [
            {
                "n_classes": int(n_classes),
                "n_class_pairs": int(len(values)),
                "mean": nan_stat(np.mean, finite),
                "std": nan_stat(np.std, finite),
                "min": float(np.min(finite)) if finite.size else np.nan,
                "q05": nan_quantile(finite, 0.05),
                "q25": nan_quantile(finite, 0.25),
                "median": nan_quantile(finite, 0.50),
                "q75": nan_quantile(finite, 0.75),
                "q95": nan_quantile(finite, 0.95),
                "max": float(np.max(finite)) if finite.size else np.nan,
            }
        ]
    )


def _top_class_similarity(pairwise: pd.DataFrame, top_model_id: str) -> pd.DataFrame:
    top = pairwise.loc[pairwise["left_model_id"].eq(top_model_id) | pairwise["right_model_id"].eq(top_model_id)].copy()
    top["other_category"] = np.where(
        top["left_model_id"].eq(top_model_id),
        top["right_category"],
        top["left_category"],
    )
    top["other_response_window_rsa"] = np.where(
        top["left_model_id"].eq(top_model_id),
        top["right_response_window_rsa"],
        top["left_response_window_rsa"],
    )
    return top.sort_values("chemical_rdm_rsa", ascending=False).reset_index(drop=True)


def _can_date_stratify(labels: list[str], date_map: dict[str, str]) -> bool:
    dates = [date_map.get(str(label), "") for label in labels]
    return all(dates) and len(set(dates)) > 1


def _nan_safe_greater(left: float, right: float) -> bool:
    if np.isfinite(left) and not np.isfinite(right):
        return True
    return bool(np.isfinite(left) and np.isfinite(right) and left > right)


__all__ = [
    "ClassCandidate",
    "avg_rank",
    "build_final_shortlist",
    "class_chemical_rdm_similarity",
    "class_observed_scores",
    "class_vs_full_chemical_similarity",
    "fixed_class_permutations",
    "full_search_permutation",
    "pearson",
    "reselection_stability",
    "spearman",
    "symmetric_rank_matrix",
]
