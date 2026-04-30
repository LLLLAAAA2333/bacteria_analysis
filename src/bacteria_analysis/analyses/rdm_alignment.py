"""High-level neural-chemical RDM alignment analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.analysis_dataset import AnalysisDataset
from bacteria_analysis.analysis_plotting import (
    plot_null_distribution,
    plot_rdm_heatmap_pair,
    plot_subset_stability,
)
from bacteria_analysis.analysis_results import AnalysisResult
from bacteria_analysis.chemical_features import build_chemical_rdm
from bacteria_analysis.neural_features import build_neural_rdm
from bacteria_analysis.rdm import align_square_rdms, rdm_pair_values, spearman_similarity
from bacteria_analysis.stats import (
    date_preserving_label_shuffle_null,
    empirical_p_value,
    label_shuffle_null,
    stimulus_subset_rsa,
)

DATE_STRUCTURE_CAVEAT = (
    "Within-date and cross-date RSA are descriptive date-structure summaries; "
    "cross-date RSA is a stress test, not proof of generalization, because "
    "stimulus identity and date can be confounded."
)


def run_rdm_alignment(
    dataset: AnalysisDataset,
    *,
    neural_view: str = "response_window",
    neural_aggregation: str = "median",
    chemical_qc_threshold: float = 0.2,
    chemical_transform: str = "log2",
    chemical_distance: str = "euclidean",
    permutations: int = 2000,
    subset_count: int = 200,
    subset_fraction: float = 0.8,
    seed: int = 0,
    include_debug: bool = False,
) -> AnalysisResult:
    """Run the broad neural-chemical RDM alignment analysis in memory."""

    neural_result = build_neural_rdm(dataset, view=neural_view, aggregation=neural_aggregation)
    chemical_result = build_chemical_rdm(
        dataset,
        qc_threshold=chemical_qc_threshold,
        transform=chemical_transform,
        distance=chemical_distance,
    )
    neural_rdm, chemical_rdm = align_square_rdms(neural_result.matrix, chemical_result.matrix)
    if len(neural_rdm) < 2:
        raise ValueError("at least 2 shared stimuli are required for RDM alignment")

    pair_values = rdm_pair_values(neural_rdm, chemical_rdm)
    observed = _rsa(pair_values)
    label_null = label_shuffle_null(neural_rdm, chemical_rdm, n_permutations=permutations, seed=seed)
    subset_results = stimulus_subset_rsa(
        neural_rdm,
        chemical_rdm,
        subset_count=subset_count,
        subset_fraction=subset_fraction,
        seed=seed + 1,
    )

    date_map = _stimulus_date_map(dataset.neural)
    pair_values_with_dates = _attach_pair_dates(pair_values, date_map)
    date_preserving_null = _date_preserving_null(
        neural_rdm,
        chemical_rdm,
        date_map,
        permutations=permutations,
        seed=seed + 2,
    )
    scope_summary = _scope_summary_table(pair_values_with_dates)
    subset_summary = _subset_summary(subset_results)

    summary = {
        "all_pairs_rsa": observed,
        "within_date_rsa": _scope_rsa(pair_values_with_dates, "within_date"),
        "cross_date_rsa": _scope_rsa(pair_values_with_dates, "cross_date"),
        "n_pairs_all": int(len(pair_values)),
        "n_pairs_within_date": _scope_pair_count(pair_values_with_dates, "within_date"),
        "n_pairs_cross_date": _scope_pair_count(pair_values_with_dates, "cross_date"),
        "label_shuffle_p_value": empirical_p_value(observed, label_null, side="greater"),
        "date_preserving_p_value": empirical_p_value(observed, date_preserving_null, side="greater"),
        "subset_rsa_median": subset_summary["median"],
        "subset_rsa_q01": subset_summary["q01"],
        "subset_rsa_q99": subset_summary["q99"],
        "date_structure_caveat": DATE_STRUCTURE_CAVEAT,
    }

    tables = {
        "rsa_summary_by_scope": scope_summary,
        "subset_stability_summary": pd.DataFrame([subset_summary]),
    }
    figures = {
        "aligned_rdms": plot_rdm_heatmap_pair(neural_rdm, chemical_rdm),
        "label_shuffle_null": plot_null_distribution(label_null, observed),
        "subset_stability": plot_subset_stability(subset_results),
    }
    audit = {
        "aligned_stimulus_order": neural_rdm.index.astype(str).tolist(),
        "retained_features": list(chemical_result.metadata["retained_features"]),
        "date_coverage": _date_coverage(date_map, neural_rdm.index),
        "date_pair_coverage": _date_pair_coverage(pair_values_with_dates),
        "n_pairs_by_scope": {
            "all": int(len(pair_values)),
            "within_date": _scope_pair_count(pair_values_with_dates, "within_date"),
            "cross_date": _scope_pair_count(pair_values_with_dates, "cross_date"),
        },
        "source_parameters": dataset.parameters,
    }
    diagnostics = {
        "date_structure_caveat": DATE_STRUCTURE_CAVEAT,
        "date_preserving_null_available": bool(date_preserving_null.size),
        "neural_metadata": neural_result.metadata,
        "chemical_metadata": chemical_result.metadata,
    }

    debug_tables = {}
    if include_debug:
        debug_tables = {
            "pair_values": pair_values_with_dates,
            "label_shuffle_null": pd.DataFrame(
                {"iteration": np.arange(len(label_null)), "rsa_similarity": label_null}
            ),
            "date_preserving_label_shuffle_null": pd.DataFrame(
                {"iteration": np.arange(len(date_preserving_null)), "rsa_similarity": date_preserving_null}
            ),
            "subset_rsa": subset_results,
        }

    return AnalysisResult(
        analysis_id="rdm_alignment",
        parameters={
            **dataset.parameters,
            "neural_view": neural_view,
            "neural_aggregation": neural_aggregation,
            "chemical_qc_threshold": chemical_qc_threshold,
            "chemical_transform": chemical_transform,
            "chemical_distance": chemical_distance,
            "permutations": permutations,
            "subset_count": subset_count,
            "subset_fraction": subset_fraction,
            "seed": seed,
        },
        summary=summary,
        tables=tables,
        rdms={"neural": neural_rdm, "chemical": chemical_rdm},
        figures=figures,
        audit=audit,
        diagnostics=diagnostics,
        debug_tables=debug_tables,
    )


def _rsa(pair_values: pd.DataFrame) -> float:
    return spearman_similarity(pair_values["neural_distance"], pair_values["chemical_distance"])


def _stimulus_date_map(neural: pd.DataFrame) -> dict[str, str]:
    if not {"stimulus", "date"}.issubset(neural.columns):
        return {}
    frame = neural.loc[:, ["stimulus", "date"]].copy()
    frame["stimulus"] = frame["stimulus"].fillna("").astype(str).str.strip()
    frame["date"] = frame["date"].fillna("").astype(str).str.strip()
    frame = frame.loc[(frame["stimulus"] != "") & (frame["date"] != "")]
    date_map: dict[str, str] = {}
    for stimulus, group in frame.groupby("stimulus", sort=True):
        counts = group["date"].value_counts()
        date_map[str(stimulus)] = str(counts.sort_index().idxmax())
    return date_map


def _attach_pair_dates(pair_values: pd.DataFrame, date_map: dict[str, str]) -> pd.DataFrame:
    with_dates = pair_values.copy()
    with_dates["date_left"] = with_dates["stimulus_left"].map(date_map).fillna("")
    with_dates["date_right"] = with_dates["stimulus_right"].map(date_map).fillna("")
    known_dates = (with_dates["date_left"] != "") & (with_dates["date_right"] != "")
    with_dates["date_scope"] = "unknown"
    with_dates.loc[known_dates & (with_dates["date_left"] == with_dates["date_right"]), "date_scope"] = "within_date"
    with_dates.loc[known_dates & (with_dates["date_left"] != with_dates["date_right"]), "date_scope"] = "cross_date"
    return with_dates


def _date_preserving_null(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    date_map: dict[str, str],
    *,
    permutations: int,
    seed: int,
) -> np.ndarray:
    if permutations <= 0 or not _can_use_date_preserving_null(date_map, neural.index):
        return np.array([], dtype=float)
    return date_preserving_label_shuffle_null(
        neural,
        chemical,
        date_map,
        n_permutations=permutations,
        seed=seed,
    )


def _can_use_date_preserving_null(date_map: dict[str, str], labels: pd.Index) -> bool:
    dates = pd.Series(date_map, dtype=object).reindex(labels.astype(str))
    if dates.isna().any() or dates.nunique() < 2:
        return False
    return bool(dates.value_counts().gt(1).any())


def _scope_summary_table(pair_values: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "scope": "all",
            "rsa_similarity": _rsa(pair_values),
            "n_pairs": int(len(pair_values)),
        }
    ]
    for scope in ("within_date", "cross_date"):
        rows.append(
            {
                "scope": scope,
                "rsa_similarity": _scope_rsa(pair_values, scope),
                "n_pairs": _scope_pair_count(pair_values, scope),
            }
        )
    return pd.DataFrame(rows, columns=["scope", "rsa_similarity", "n_pairs"])


def _scope_rsa(pair_values: pd.DataFrame, scope: str) -> float:
    subset = pair_values.loc[pair_values["date_scope"] == scope]
    if len(subset) < 2:
        return float("nan")
    return _rsa(subset)


def _scope_pair_count(pair_values: pd.DataFrame, scope: str) -> int:
    return int((pair_values["date_scope"] == scope).sum())


def _subset_summary(subset_results: pd.DataFrame) -> dict[str, float | int]:
    values = pd.to_numeric(subset_results.get("rsa_similarity", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return {"median": float("nan"), "q01": float("nan"), "q99": float("nan"), "n_draws": 0}
    return {
        "median": float(values.median()),
        "q01": float(values.quantile(0.01)),
        "q99": float(values.quantile(0.99)),
        "n_draws": int(len(values)),
    }


def _date_coverage(date_map: dict[str, str], labels: pd.Index) -> pd.DataFrame:
    rows = [
        {"date": date_map.get(str(label), ""), "stimulus": str(label)}
        for label in labels
        if date_map.get(str(label), "")
    ]
    if not rows:
        return pd.DataFrame(columns=["date", "n_stimuli"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "n_stimuli"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def _date_pair_coverage(pair_values: pd.DataFrame) -> pd.DataFrame:
    known = pair_values.loc[pair_values["date_scope"] != "unknown"].copy()
    if known.empty:
        return pd.DataFrame(columns=["date_left", "date_right", "date_scope", "n_pairs"])
    return (
        known.groupby(["date_left", "date_right", "date_scope"], as_index=False)
        .size()
        .rename(columns={"size": "n_pairs"})
        .sort_values(["date_scope", "date_left", "date_right"])
        .reset_index(drop=True)
    )


__all__ = [
    "run_rdm_alignment",
]
