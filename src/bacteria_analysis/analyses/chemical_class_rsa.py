"""Chemical taxonomy-class RSA analysis."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from bacteria_analysis.analysis_dataset import AnalysisDataset
from bacteria_analysis.analysis_plotting import plot_rdm_heatmap, plot_rdm_heatmap_grid, plot_score_bars
from bacteria_analysis.analysis_results import AnalysisResult
from bacteria_analysis.chemical_features import build_chemical_class_rdms, build_chemical_rdm
from bacteria_analysis.neural_features import build_neural_rdm
from bacteria_analysis.rdm import align_square_rdms, rdm_pair_values, spearman_similarity
from bacteria_analysis.stats import empirical_p_value, label_shuffle_null

RESELECTION_FRACTION = 0.8


def run_chemical_class_rsa(
    dataset: AnalysisDataset,
    *,
    neural_rdm: pd.DataFrame | None = None,
    neural_view: str = "response_window",
    taxonomy_level: str = "Class",
    qc_threshold: float = 0.2,
    min_features: int = 3,
    fixed_permutations: int = 2000,
    resamples: int = 500,
    search_permutations: int = 2000,
    top_k: int = 5,
    seed: int = 0,
    include_debug: bool = False,
) -> AnalysisResult:
    """Score chemical taxonomy classes against a neural RDM."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if fixed_permutations < 0 or resamples < 0 or search_permutations < 0:
        raise ValueError("permutation and resample counts must be non-negative")

    neural = neural_rdm.copy() if neural_rdm is not None else build_neural_rdm(dataset, view=neural_view).matrix
    full_chemical = build_chemical_rdm(dataset, qc_threshold=qc_threshold)
    class_results = build_chemical_class_rdms(
        dataset,
        taxonomy_level=taxonomy_level,
        qc_threshold=qc_threshold,
        min_features=min_features,
    )
    if not class_results:
        raise ValueError("no chemical classes passed feature filters")

    observed_scores = _observed_scores(neural, class_results)
    fixed_summary, fixed_nulls = _fixed_class_permutation_summary(
        neural,
        class_results,
        observed_scores,
        fixed_permutations=fixed_permutations,
        seed=seed,
    )
    reselection_draws, reselection_summary = _reselection_stability(
        neural,
        class_results,
        observed_scores,
        date_map=_stimulus_date_map(dataset.neural),
        resamples=resamples,
        seed=seed + 10_000,
    )
    search_summary, search_null = _search_corrected_diagnostic_summary(
        neural,
        class_results,
        observed_scores,
        search_permutations=search_permutations,
        seed=seed + 20_000,
    )
    final_shortlist = fixed_summary.head(top_k).copy()
    final_shortlist["selected_for_audit"] = True

    class_similarity = _class_to_class_similarity(class_results)
    class_vs_full = _class_vs_full_similarity(full_chemical.matrix, class_results)
    class_similarity_matrix = _similarity_matrix(class_similarity, class_results.keys())
    result_rdms = _reported_rdms(neural, full_chemical.matrix, class_results, final_shortlist["class"].tolist())

    figures = {
        "fixed_class_permutation_scores": plot_score_bars(
            fixed_summary,
            label_column="class",
            value_column="observed_rsa",
            title="Fixed-class RSA",
        ),
        "reselection_stability": plot_score_bars(
            reselection_summary,
            label_column="class",
            value_column="top_fraction",
            title="Reselection stability",
            ylabel="top fraction",
        ),
        "top_class_rdm_comparison": plot_rdm_heatmap_grid(
            _top_class_comparison_rdms(neural, full_chemical.matrix, class_results, final_shortlist),
            title="Top class RDM comparison",
        ),
        "final_shortlist_scorecard": plot_score_bars(
            final_shortlist,
            label_column="class",
            value_column="observed_rsa",
            title="Final shortlist",
        ),
        "class_chemical_rdm_similarity": plot_rdm_heatmap(
            class_similarity_matrix,
            title="Class chemical RDM similarity",
            colorbar_label="similarity",
        ),
    }

    summary = {
        "top_class": str(final_shortlist.iloc[0]["class"]),
        "top_class_rsa": float(final_shortlist.iloc[0]["observed_rsa"]),
        "top_class_fixed_p_value": float(final_shortlist.iloc[0]["p_value"]),
        "top_class_fixed_q_value": float(final_shortlist.iloc[0]["q_value"]),
        "top_class_feature_count": int(final_shortlist.iloc[0]["feature_count"]),
        "evaluated_class_count": int(len(observed_scores)),
        "fixed_permutations": fixed_permutations,
        "resamples": resamples,
        "search_permutations": search_permutations,
        "search_corrected_results_are_diagnostic": True,
    }

    debug_tables = {}
    if include_debug:
        debug_tables = {
            "fixed_class_null": fixed_nulls,
            "search_max_null": search_null,
            "reselection_draws": reselection_draws,
        }

    return AnalysisResult(
        analysis_id="chemical_class_rsa",
        parameters={
            **dataset.parameters,
            "neural_rdm_provided": neural_rdm is not None,
            "neural_view": neural_view,
            "taxonomy_level": taxonomy_level,
            "qc_threshold": qc_threshold,
            "min_features": min_features,
            "fixed_permutations": fixed_permutations,
            "resamples": resamples,
            "search_permutations": search_permutations,
            "top_k": top_k,
            "seed": seed,
        },
        summary=summary,
        tables={
            "observed_class_scores": observed_scores,
            "fixed_class_permutation_summary": fixed_summary,
            "reselection_stability_summary": reselection_summary,
            "search_corrected_diagnostic_summary": search_summary,
            "final_class_shortlist": final_shortlist,
            "class_to_class_chemical_rdm_similarity": class_similarity,
            "class_vs_full_chemical_rdm_similarity": class_vs_full,
        },
        rdms=result_rdms,
        figures=figures,
        audit={
            "reported_rdm_keys": list(result_rdms),
            "full_chemical_retained_features": list(full_chemical.metadata["retained_features"]),
            "reselection_date_composition": _date_composition_audit(reselection_draws),
            "search_corrected_layer": {
                "diagnostic": True,
                "reason": "search-corrected results test exploratory class selection and are not fixed-class evidence",
            },
        },
        diagnostics={
            "class_count": int(len(class_results)),
            "search_corrected_results_are_diagnostic": True,
            "date_aware_resampling": bool(reselection_summary["date_aware_resampling"].any())
            if not reselection_summary.empty
            else False,
        },
        debug_tables=debug_tables,
    )


def _observed_scores(neural: pd.DataFrame, class_results: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_name, result in class_results.items():
        aligned_neural, aligned_class = align_square_rdms(neural, result.matrix)
        pair_values = rdm_pair_values(aligned_neural, aligned_class)
        rows.append(
            {
                "class": class_name,
                "rsa_similarity": _rsa(pair_values),
                "n_pairs": int(len(pair_values)),
                "feature_count": int(result.metadata["feature_count"]),
                "retained_features": ";".join(result.metadata["retained_features"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["rsa_similarity", "feature_count", "class"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _fixed_class_permutation_summary(
    neural: pd.DataFrame,
    class_results: dict[str, object],
    observed_scores: pd.DataFrame,
    *,
    fixed_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    null_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    observed_by_class = observed_scores.set_index("class")

    for offset, (class_name, result) in enumerate(class_results.items()):
        aligned_neural, aligned_class = align_square_rdms(neural, result.matrix)
        null_values = label_shuffle_null(
            aligned_neural,
            aligned_class,
            n_permutations=fixed_permutations,
            seed=seed + offset,
        )
        observed = float(observed_by_class.loc[class_name, "rsa_similarity"])
        p_value = empirical_p_value(observed, null_values, side="greater")
        summary_rows.append(
            {
                "class": class_name,
                "observed_rsa": observed,
                "p_value": p_value,
                "n_permutations": fixed_permutations,
                "n_pairs": int(observed_by_class.loc[class_name, "n_pairs"]),
                "feature_count": int(observed_by_class.loc[class_name, "feature_count"]),
            }
        )
        null_rows.append(
            pd.DataFrame(
                {
                    "class": class_name,
                    "iteration": np.arange(len(null_values)),
                    "rsa_similarity": null_values,
                }
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary["q_value"] = _benjamini_hochberg(summary["p_value"].to_numpy(dtype=float))
    summary = summary.sort_values(["observed_rsa", "feature_count", "class"], ascending=[False, False, True])
    nulls = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    return summary.reset_index(drop=True), nulls


def _reselection_stability(
    neural: pd.DataFrame,
    class_results: dict[str, object],
    observed_scores: pd.DataFrame,
    *,
    date_map: dict[str, str],
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(observed_scores_for_labels(neural, class_results), dtype=object)
    rng = np.random.default_rng(seed)
    draw_rows: list[dict[str, object]] = []
    date_aware = _can_resample_by_date(labels, date_map)

    for iteration in range(resamples):
        subset = _date_aware_subset(labels, date_map, rng) if date_aware else _random_subset(labels, rng)
        scores = _scores_on_subset(neural, class_results, subset)
        top = scores.iloc[0] if not scores.empty else pd.Series(dtype=object)
        composition = _date_composition(subset, date_map)
        draw_rows.append(
            {
                "iteration": iteration,
                "top_class": top.get("class", ""),
                "top_score": top.get("rsa_similarity", np.nan),
                "n_stimuli": int(len(subset)),
                "date_composition": composition,
                "date_aware_resampling": date_aware,
            }
        )

    draws = pd.DataFrame(
        draw_rows,
        columns=["iteration", "top_class", "top_score", "n_stimuli", "date_composition", "date_aware_resampling"],
    )
    summary_rows: list[dict[str, object]] = []
    feature_counts = observed_scores.set_index("class")["feature_count"].to_dict()
    for class_name in observed_scores["class"].tolist():
        class_draws = draws.loc[draws["top_class"] == class_name]
        scores = pd.to_numeric(class_draws["top_score"], errors="coerce").dropna()
        summary_rows.append(
            {
                "class": class_name,
                "top_count": int(len(class_draws)),
                "top_fraction": float(len(class_draws) / resamples) if resamples else np.nan,
                "score_median": float(scores.median()) if not scores.empty else np.nan,
                "score_q01": float(scores.quantile(0.01)) if not scores.empty else np.nan,
                "score_q99": float(scores.quantile(0.99)) if not scores.empty else np.nan,
                "feature_count": int(feature_counts[class_name]),
                "date_aware_resampling": date_aware,
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["top_fraction", "score_median", "class"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return draws, summary


def observed_scores_for_labels(neural: pd.DataFrame, class_results: dict[str, object]) -> list[str]:
    labels: set[str] = set(neural.index.astype(str))
    for result in class_results.values():
        labels &= set(result.matrix.index.astype(str))
    return sorted(labels)


def _scores_on_subset(neural: pd.DataFrame, class_results: dict[str, object], subset: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_name, result in class_results.items():
        aligned_neural, aligned_class = align_square_rdms(neural, result.matrix)
        labels = [label for label in aligned_neural.index.astype(str).tolist() if label in set(subset)]
        if len(labels) < 3:
            score = np.nan
            n_pairs = 0
        else:
            pair_values = rdm_pair_values(
                aligned_neural.loc[labels, labels],
                aligned_class.loc[labels, labels],
            )
            score = _rsa(pair_values)
            n_pairs = int(len(pair_values))
        rows.append({"class": class_name, "rsa_similarity": score, "n_pairs": n_pairs})
    return (
        pd.DataFrame(rows)
        .sort_values(["rsa_similarity", "class"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )


def _search_corrected_diagnostic_summary(
    neural: pd.DataFrame,
    class_results: dict[str, object],
    observed_scores: pd.DataFrame,
    *,
    search_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if search_permutations == 0:
        max_null = np.array([], dtype=float)
    else:
        null_arrays = []
        for offset, result in enumerate(class_results.values()):
            aligned_neural, aligned_class = align_square_rdms(neural, result.matrix)
            null_arrays.append(
                label_shuffle_null(
                    aligned_neural,
                    aligned_class,
                    n_permutations=search_permutations,
                    seed=seed + offset,
                )
            )
        max_null = np.nanmax(np.vstack(null_arrays), axis=0)

    rows = []
    for _, row in observed_scores.iterrows():
        rows.append(
            {
                "class": row["class"],
                "observed_rsa": row["rsa_similarity"],
                "search_corrected_p_value": empirical_p_value(row["rsa_similarity"], max_null, side="greater"),
                "n_permutations": search_permutations,
                "diagnostic": True,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["observed_rsa", "class"], ascending=[False, True]).reset_index(drop=True)
    null = pd.DataFrame({"iteration": np.arange(len(max_null)), "max_rsa_similarity": max_null})
    return summary, null


def _class_to_class_similarity(class_results: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    class_names = list(class_results)
    for left_index, left_class in enumerate(class_names):
        for right_class in class_names[left_index + 1 :]:
            left, right = align_square_rdms(class_results[left_class].matrix, class_results[right_class].matrix)
            pair_values = rdm_pair_values(left, right)
            rows.append(
                {
                    "class_left": left_class,
                    "class_right": right_class,
                    "rdm_similarity": _rsa(pair_values),
                }
            )
    return pd.DataFrame(rows, columns=["class_left", "class_right", "rdm_similarity"])


def _class_vs_full_similarity(full_chemical: pd.DataFrame, class_results: dict[str, object]) -> pd.DataFrame:
    rows = []
    for class_name, result in class_results.items():
        left, right = align_square_rdms(full_chemical, result.matrix)
        pair_values = rdm_pair_values(left, right)
        rows.append({"class": class_name, "full_similarity": _rsa(pair_values)})
    return (
        pd.DataFrame(rows)
        .sort_values(["full_similarity", "class"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _similarity_matrix(summary: pd.DataFrame, class_names) -> pd.DataFrame:
    labels = sorted(str(class_name) for class_name in class_names)
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels, dtype=float)
    for _, row in summary.iterrows():
        matrix.loc[row["class_left"], row["class_right"]] = row["rdm_similarity"]
        matrix.loc[row["class_right"], row["class_left"]] = row["rdm_similarity"]
    return matrix


def _reported_rdms(
    neural: pd.DataFrame,
    full_chemical: pd.DataFrame,
    class_results: dict[str, object],
    shortlisted_classes: list[str],
) -> dict[str, pd.DataFrame]:
    rdms = {"neural": neural.copy(), "chemical_full": full_chemical.copy()}
    for class_name in shortlisted_classes:
        rdms[f"class_{_safe_name(class_name)}"] = class_results[class_name].matrix.copy()
    return rdms


def _top_class_comparison_rdms(
    neural: pd.DataFrame,
    full_chemical: pd.DataFrame,
    class_results: dict[str, object],
    final_shortlist: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    top_class = str(final_shortlist.iloc[0]["class"])
    aligned_neural, aligned_full = align_square_rdms(neural, full_chemical)
    aligned_neural, aligned_top = align_square_rdms(aligned_neural, class_results[top_class].matrix)
    aligned_full = aligned_full.loc[aligned_neural.index, aligned_neural.columns]
    return {
        "Neural": aligned_neural,
        "Full chemical": aligned_full,
        top_class: aligned_top,
    }


def _date_composition_audit(draws: pd.DataFrame) -> pd.DataFrame:
    if draws.empty:
        return pd.DataFrame(columns=["iteration", "date_composition", "date_aware_resampling"])
    return draws.loc[:, ["iteration", "date_composition", "date_aware_resampling"]].copy()


def _rsa(pair_values: pd.DataFrame) -> float:
    return spearman_similarity(pair_values["neural_distance"], pair_values["chemical_distance"])


def _stimulus_date_map(neural: pd.DataFrame) -> dict[str, str]:
    if not {"stimulus", "date"}.issubset(neural.columns):
        return {}
    frame = neural.loc[:, ["stimulus", "date"]].copy()
    frame["stimulus"] = frame["stimulus"].fillna("").astype(str).str.strip()
    frame["date"] = frame["date"].fillna("").astype(str).str.strip()
    frame = frame.loc[(frame["stimulus"] != "") & (frame["date"] != "")]
    return {str(stimulus): str(group["date"].mode().sort_values().iloc[0]) for stimulus, group in frame.groupby("stimulus")}


def _can_resample_by_date(labels: np.ndarray, date_map: dict[str, str]) -> bool:
    dates = [date_map.get(str(label), "") for label in labels]
    return all(dates) and len(set(dates)) > 1


def _date_aware_subset(labels: np.ndarray, date_map: dict[str, str], rng: np.random.Generator) -> list[str]:
    by_date: dict[str, list[str]] = {}
    for label in labels:
        by_date.setdefault(date_map[str(label)], []).append(str(label))
    selected: list[str] = []
    for group in by_date.values():
        sample_size = max(1, int(np.ceil(len(group) * RESELECTION_FRACTION)))
        selected.extend(rng.choice(group, size=min(len(group), sample_size), replace=False).tolist())
    if len(selected) < 3 and len(labels) >= 3:
        remaining = [str(label) for label in labels if str(label) not in set(selected)]
        selected.extend(rng.choice(remaining, size=3 - len(selected), replace=False).tolist())
    return [str(label) for label in labels if str(label) in set(selected)]


def _random_subset(labels: np.ndarray, rng: np.random.Generator) -> list[str]:
    sample_size = max(3, int(np.ceil(len(labels) * RESELECTION_FRACTION)))
    sample_size = min(len(labels), sample_size)
    subset = rng.choice(labels, size=sample_size, replace=False)
    return [str(label) for label in labels if label in set(subset)]


def _date_composition(subset: list[str], date_map: dict[str, str]) -> str:
    if not date_map:
        return f"unknown:{len(subset)}"
    counts: dict[str, int] = {}
    for label in subset:
        counts[date_map.get(str(label), "unknown")] = counts.get(date_map.get(str(label), "unknown"), 0) + 1
    return ";".join(f"{date}:{counts[date]}" for date in sorted(counts))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    q_values = np.full_like(p_values, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(p_values))
    if finite_positions.size == 0:
        return q_values
    finite = p_values[finite_positions]
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    n_tests = len(ranked)
    for index in range(n_tests - 1, -1, -1):
        running = min(running, ranked[index] * n_tests / (index + 1))
        adjusted[index] = running
    q_values[finite_positions[order]] = adjusted
    return q_values


def _safe_name(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("._") or "class"


__all__ = [
    "run_chemical_class_rsa",
]
