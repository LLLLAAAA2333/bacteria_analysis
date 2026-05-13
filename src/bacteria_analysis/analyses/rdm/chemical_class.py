"""Chemical taxonomy-class RSA analysis."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from bacteria_analysis.io import AnalysisDataset, AnalysisResult
from bacteria_analysis.analyses.rdm.builders import (
    build_chemical_class_rdms,
    build_chemical_rdm,
    build_neural_rdm,
)
from bacteria_analysis.analyses.rdm.plots import (
    plot_class_chemical_rdm_similarity,
    plot_class_vs_full_chemical_similarity,
    plot_fixed_class_permutation,
    plot_reselection_stability,
    plot_summary_scorecard,
    plot_top_class_rdm_comparison,
)
from bacteria_analysis.features.taxonomy import (
    ClassCandidate,
    avg_rank,
    build_final_shortlist,
    class_chemical_rdm_similarity,
    class_observed_scores,
    class_vs_full_chemical_similarity,
    fixed_class_permutations,
    full_search_permutation,
    pearson,
    reselection_stability,
    spearman,
    symmetric_rank_matrix,
)

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
    if fixed_permutations < 1 or resamples < 1 or search_permutations < 1:
        raise ValueError("permutation and resample counts must be at least 1")

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

    candidates, labels, upper_i, upper_j, primary_ranks = _class_candidates(
        neural=neural,
        full_chemical=full_chemical.matrix,
        class_results=class_results,
        taxonomy_level=taxonomy_level,
    )
    if not candidates:
        raise ValueError("no chemical classes had enough shared stimuli")

    rng = np.random.default_rng(seed)
    observed_scores = _observed_scores_from_candidates(class_observed_scores(candidates))
    fixed_summary, fixed_nulls = fixed_class_permutations(
        candidates=candidates,
        labels=labels,
        neural_ranks=primary_ranks,
        upper_i=upper_i,
        upper_j=upper_j,
        n_permutations=fixed_permutations,
        rng=rng,
    )
    fixed_summary = _fixed_summary_with_aliases(fixed_summary)
    reselection_draws, reselection_summary = reselection_stability(
        candidates=candidates,
        primary_neural=neural.loc[labels, labels],
        labels=labels,
        date_map=_stimulus_date_map(dataset.neural),
        n_resamples=resamples,
        resample_fraction=RESELECTION_FRACTION,
        top_k=max(top_k, 5),
        rng=rng,
    )
    reselection_summary = _reselection_summary_with_aliases(
        reselection_summary,
        date_aware=_can_resample_by_date(np.asarray(labels, dtype=object), _stimulus_date_map(dataset.neural)),
    )
    search_summary, search_null = full_search_permutation(
        candidates=candidates,
        labels=labels,
        neural_ranks=primary_ranks,
        upper_i=upper_i,
        upper_j=upper_j,
        n_permutations=search_permutations,
        rng=rng,
    )
    search_summary = _search_summary_with_aliases(search_summary)
    final_shortlist = _ensure_shortlist_size(
        build_final_shortlist(
            observed_scores,
            fixed_summary,
            reselection_summary,
            search_summary,
        ),
        observed_scores,
        fixed_summary,
        reselection_summary,
        search_summary,
        top_k=top_k,
    )
    final_shortlist = _shortlist_with_aliases(final_shortlist)

    class_similarity, class_similarity_summary, top_class_similarity = class_chemical_rdm_similarity(
        candidates=candidates,
        labels=labels,
        upper_i=upper_i,
        upper_j=upper_j,
    )
    class_vs_full, class_vs_full_summary = class_vs_full_chemical_similarity(
        candidates=candidates,
        full_chemical=full_chemical.matrix.loc[labels, labels],
        labels=labels,
        upper_i=upper_i,
        upper_j=upper_j,
    )
    result_rdms = _reported_rdms_from_candidates(
        neural.loc[labels, labels],
        full_chemical.matrix.loc[labels, labels],
        candidates,
        final_shortlist["category"].astype(str).tolist(),
    )

    top_candidate = candidates[0]
    figures = {
        "fixed_class_permutation.png": lambda output_path: plot_fixed_class_permutation(
            fixed_summary,
            output_path,
            class_limit=24,
        ),
        "reselection_stability.png": lambda output_path: plot_reselection_stability(
            reselection_summary,
            observed_scores,
            output_path,
            class_limit=24,
        ),
        "top_class_rdm_comparison.png": lambda output_path: plot_top_class_rdm_comparison(
            primary_neural=neural.loc[labels, labels],
            full_chemical=full_chemical.matrix.loc[labels, labels],
            candidate=top_candidate,
            labels=labels,
            stimulus_sample_map=dataset.stimulus_sample_map,
            output_path=output_path,
        ),
        "taxonomy_class_stability_summary.png": lambda output_path: plot_summary_scorecard(
            final_shortlist,
            output_path,
            class_limit=min(24, 18),
        ),
        "class_chemical_rdm_similarity_matrix.png": lambda output_path: plot_class_chemical_rdm_similarity(
            pairwise=class_similarity,
            top_class_similarity=top_class_similarity,
            output_path=output_path,
        ),
        "class_vs_full_chemical_rdm_similarity.png": lambda output_path: plot_class_vs_full_chemical_similarity(
            similarity=class_vs_full,
            output_path=output_path,
        ),
    }

    summary = {
        "top_class": str(final_shortlist.iloc[0]["category"]),
        "top_class_rsa": float(final_shortlist.iloc[0]["response_window_rsa"]),
        "top_class_fixed_p_value": float(final_shortlist.iloc[0]["p_one_sided_ge"]),
        "top_class_fixed_q_value": float(final_shortlist.iloc[0]["p_fdr_bh"]),
        "top_class_feature_count": int(final_shortlist.iloc[0]["n_features"]),
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
            "search_corrected_diagnostic_summary": search_summary,
            "class_to_class_chemical_rdm_similarity": class_similarity,
            "class_to_class_chemical_rdm_similarity_summary": class_similarity_summary,
            "class_vs_full_chemical_rdm_similarity_summary": class_vs_full_summary,
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
            "final_class_shortlist": final_shortlist,
            "class_vs_full_chemical_rdm_similarity": class_vs_full,
        },
        rdms=result_rdms,
        figures=figures,
        audit={
            "reported_rdm_keys": list(result_rdms),
            "full_chemical_retained_features": list(full_chemical.metadata["retained_features"]),
            "reselection_date_composition": _date_composition_audit_from_runs(reselection_draws, _stimulus_date_map(dataset.neural)),
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


def _class_candidates(
    *,
    neural: pd.DataFrame,
    full_chemical: pd.DataFrame,
    class_results: dict[str, object],
    taxonomy_level: str,
) -> tuple[list[ClassCandidate], list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels = observed_scores_for_labels(neural, class_results)
    labels = [label for label in labels if label in full_chemical.index and label in full_chemical.columns]
    if len(labels) < 3:
        return [], labels, np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    primary_values = neural.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
    primary_ranks = avg_rank(primary_values)
    full_values = full_chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
    candidates: list[ClassCandidate] = []

    for class_name, result in class_results.items():
        chemical = result.matrix.loc[labels, labels]
        chemical_values = chemical.to_numpy(float)[upper_i, upper_j]
        chemical_ranks = avg_rank(chemical_values)
        candidates.append(
            ClassCandidate(
                model_id=f"{taxonomy_level}::{class_name}",
                taxonomy_level=str(taxonomy_level),
                category=str(class_name),
                metabolites=tuple(result.metadata["retained_features"]),
                chemical=chemical,
                primary_rank_matrix=symmetric_rank_matrix(
                    len(labels),
                    upper_i,
                    upper_j,
                    chemical_ranks,
                ),
                primary_observed_rsa=pearson(primary_ranks, chemical_ranks),
                full_trajectory_rsa=spearman(full_values, chemical_values),
                n_pairs=int(len(chemical_values)),
            )
        )

    candidates = sorted(candidates, key=lambda candidate: candidate.primary_observed_rsa, reverse=True)
    return candidates, labels, upper_i, upper_j, primary_ranks


def _observed_scores_from_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    observed = frame.copy()
    observed["class"] = observed["category"]
    observed["feature_count"] = observed["n_features"]
    observed["rsa_similarity"] = observed["response_window_rsa"]
    observed["retained_features"] = observed["metabolites"].astype(str).str.replace(" | ", ";", regex=False)
    return observed


def _fixed_summary_with_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.copy()
    summary["class"] = summary["category"]
    summary["feature_count"] = summary["n_features"]
    summary["p_value"] = summary["p_one_sided_ge"]
    summary["q_value"] = summary["p_fdr_bh"]
    return summary


def _reselection_summary_with_aliases(frame: pd.DataFrame, *, date_aware: bool) -> pd.DataFrame:
    summary = frame.copy()
    summary["class"] = summary["category"]
    summary["feature_count"] = summary["n_features"]
    summary["top_fraction"] = summary["top1_frequency"]
    summary["date_aware_resampling"] = date_aware
    return summary


def _search_summary_with_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.copy()
    summary["class"] = summary["observed_best_category"]
    summary["observed_rsa"] = summary["observed_best_rsa"]
    summary["search_corrected_p_value"] = summary["search_corrected_p"]
    summary["diagnostic"] = True
    return summary


def _shortlist_with_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    shortlist = frame.copy()
    shortlist["class"] = shortlist["category"]
    shortlist["feature_count"] = shortlist["n_features"]
    shortlist["observed_rsa"] = shortlist["response_window_rsa"]
    shortlist["selected_for_audit"] = True
    return shortlist


def _ensure_shortlist_size(
    shortlist: pd.DataFrame,
    observed: pd.DataFrame,
    fixed: pd.DataFrame,
    reselection: pd.DataFrame,
    search_summary: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    scorecard = observed.merge(
        fixed.loc[:, ["model_id", "observed_percentile", "p_one_sided_ge", "p_fdr_bh", "null_q95", "null_q99"]],
        on="model_id",
        how="left",
    ).merge(
        reselection.loc[:, ["model_id", "top1_frequency", "top3_frequency", "top5_frequency", "mean_selected_rank"]],
        on="model_id",
        how="left",
    )
    search_p = float(search_summary["search_corrected_p"].iloc[0])
    observed_best = str(search_summary["observed_best_model_id"].iloc[0])
    scorecard["fixed_signal_pass"] = (
        scorecard["response_window_rsa"].to_numpy(float) > scorecard["null_q95"].to_numpy(float)
    )
    scorecard["reselection_preferred"] = scorecard["top3_frequency"].fillna(0.0).ge(0.20)
    scorecard["is_observed_best"] = scorecard["model_id"].eq(observed_best)
    scorecard["search_corrected_p_for_best"] = np.where(scorecard["is_observed_best"], search_p, np.nan)
    missing = scorecard.loc[~scorecard["model_id"].isin(set(shortlist["model_id"].astype(str)))].copy()
    return pd.concat([shortlist, missing], ignore_index=True).head(top_k)


def _reported_rdms_from_candidates(
    neural: pd.DataFrame,
    full_chemical: pd.DataFrame,
    candidates: list[object],
    shortlisted_classes: list[str],
) -> dict[str, pd.DataFrame]:
    rdms = {"neural": neural.copy(), "chemical_full": full_chemical.copy()}
    by_class = {candidate.category: candidate.chemical for candidate in candidates}
    for class_name in shortlisted_classes:
        rdms[f"class_{_safe_name(class_name)}"] = by_class[class_name].copy()
    return rdms


def _date_composition_audit_from_runs(runs: pd.DataFrame, date_map: dict[str, str]) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(columns=["iteration", "date_composition", "date_aware_resampling"])
    rows = []
    for resample_index, group in runs.groupby("resample_index", sort=True):
        stimuli = str(group["stimuli"].iloc[0]).split(";") if "stimuli" in group else []
        rows.append(
            {
                "iteration": int(resample_index),
                "date_composition": _date_composition([stimulus for stimulus in stimuli if stimulus], date_map),
                "date_aware_resampling": _can_resample_by_date(np.asarray(stimuli, dtype=object), date_map),
            }
        )
    return pd.DataFrame(rows)


def observed_scores_for_labels(neural: pd.DataFrame, class_results: dict[str, object]) -> list[str]:
    labels: set[str] = set(neural.index.astype(str))
    for result in class_results.values():
        labels &= set(result.matrix.index.astype(str))
    return sorted(labels)


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


def _date_composition(subset: list[str], date_map: dict[str, str]) -> str:
    if not date_map:
        return f"unknown:{len(subset)}"
    counts: dict[str, int] = {}
    for label in subset:
        counts[date_map.get(str(label), "unknown")] = counts.get(date_map.get(str(label), "unknown"), 0) + 1
    return ";".join(f"{date}:{counts[date]}" for date in sorted(counts))


def _safe_name(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("._") or "class"


__all__ = [
    "run_chemical_class_rsa",
]
