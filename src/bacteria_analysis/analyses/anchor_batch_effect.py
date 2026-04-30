"""Anchor-stimulus batch/date-effect analysis."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from bacteria_analysis.analysis_dataset import AnchorDataset
from bacteria_analysis.analysis_plotting import plot_rdm_heatmap
from bacteria_analysis.analysis_results import AnalysisResult
from bacteria_analysis.neural_features import build_trial_feature_matrix
from bacteria_analysis.reliability import VALID_COMPARISON_STATUS, compute_vector_distance

FEATURE_COLUMN_PATTERN = re.compile(r"^[A-Za-z0-9]+__t\d{2}$")
SUPPORTED_AGGREGATIONS = ("median", "mean")


def run_anchor_batch_effect(
    anchor_dataset: AnchorDataset,
    *,
    views: tuple[str, ...] = ("response_window", "full_trajectory"),
    aggregation: str = "median",
    seed: int = 0,
    include_debug: bool = False,
) -> AnalysisResult:
    """Summarize anchor stimulus distances across dates."""

    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"unknown anchor aggregation {aggregation!r}")
    if not views:
        raise ValueError("views must include at least one neural view")

    neural = _filter_anchor_neural(anchor_dataset)
    coverage_tables: list[pd.DataFrame] = []
    pairwise_tables: list[pd.DataFrame] = []
    rdms: dict[str, pd.DataFrame] = {}
    figures = {}
    debug_tables = {}

    for view in views:
        trial_features = build_trial_feature_matrix(neural, view=view, merge_lr=True)
        prototypes = _stimulus_date_prototypes(trial_features, view=view, aggregation=aggregation)
        coverage_tables.append(_coverage_table(trial_features, view=view))
        rdm, pairwise = _prototype_rdm_and_pairs(prototypes, view=view)
        pairwise_tables.append(pairwise)
        rdms[f"{view}_anchor"] = rdm
        figures[f"{view}_anchor_rdm"] = plot_rdm_heatmap(rdm, title=f"{view} anchor RDM")
        if include_debug:
            debug_tables[f"{view}_prototypes"] = prototypes

    anchor_coverage = _combine(coverage_tables, ["view", "stimulus", "date", "n_trials"])
    anchor_pairwise_distances = _combine(pairwise_tables, _pairwise_columns())
    same_anchor_summary = _same_anchor_cross_date_summary(anchor_pairwise_distances)
    stimulus_vs_date_contrast = _stimulus_vs_date_contrast(anchor_pairwise_distances)

    return AnalysisResult(
        analysis_id="anchor_batch_effect",
        parameters={
            **anchor_dataset.parameters,
            "views": views,
            "aggregation": aggregation,
            "seed": seed,
        },
        summary={
            "anchor_count": int(len(anchor_dataset.anchor_stimuli)),
            "view_count": int(len(views)),
            "same_anchor_cross_date_median": _overall_median(same_anchor_summary),
            "stimulus_vs_date_contrast_median": _overall_contrast(stimulus_vs_date_contrast),
        },
        tables={
            "anchor_coverage": anchor_coverage,
            "anchor_pairwise_distances": anchor_pairwise_distances,
            "same_anchor_cross_date_summary": same_anchor_summary,
            "stimulus_vs_date_contrast": stimulus_vs_date_contrast,
        },
        rdms=rdms,
        figures=figures,
        audit={
            "anchor_stimuli": list(anchor_dataset.anchor_stimuli),
            "included_dates": list(anchor_dataset.included_dates),
            "excluded_dates": list(anchor_dataset.excluded_dates),
        },
        diagnostics={
            "interpretation": (
                "Same-anchor cross-date distances summarize batch/date effects; "
                "different-anchor within-date distances provide the stimulus-effect reference."
            ),
        },
        debug_tables=debug_tables,
    )


def _filter_anchor_neural(anchor_dataset: AnchorDataset) -> pd.DataFrame:
    neural = anchor_dataset.neural.copy()
    anchors = {str(value) for value in anchor_dataset.anchor_stimuli}
    if anchors and "stimulus" in neural.columns:
        neural = neural.loc[neural["stimulus"].astype(str).isin(anchors)].copy()
    return neural


def _stimulus_date_prototypes(features: pd.DataFrame, *, view: str, aggregation: str) -> pd.DataFrame:
    if "date" not in features.columns:
        raise ValueError("anchor neural features must include date")
    feature_columns = _feature_columns(features)
    if not feature_columns:
        raise ValueError("anchor neural features must include neural feature columns")

    rows: list[dict[str, object]] = []
    for (stimulus, date), group in features.groupby(["stimulus", "date"], sort=True, dropna=False):
        values = group.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if aggregation == "median":
                prototype = np.nanmedian(values, axis=0)
            else:
                prototype = np.nanmean(values, axis=0)
        row: dict[str, object] = {
            "view": view,
            "prototype_id": f"{stimulus}|{date}",
            "stimulus": str(stimulus),
            "date": str(date),
            "n_trials": int(len(group)),
        }
        row.update(dict(zip(feature_columns, prototype, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows, columns=["view", "prototype_id", "stimulus", "date", "n_trials", *feature_columns])


def _coverage_table(features: pd.DataFrame, *, view: str) -> pd.DataFrame:
    coverage = (
        features.groupby(["stimulus", "date"], as_index=False)
        .size()
        .rename(columns={"size": "n_trials"})
        .sort_values(["stimulus", "date"])
        .reset_index(drop=True)
    )
    coverage.insert(0, "view", view)
    return coverage.loc[:, ["view", "stimulus", "date", "n_trials"]]


def _prototype_rdm_and_pairs(prototypes: pd.DataFrame, *, view: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = _feature_columns(prototypes)
    labels = prototypes["prototype_id"].astype(str).tolist()
    values = prototypes.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
    distances = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(distances, 0.0)
    rows: list[dict[str, object]] = []

    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            left_values = values[left_index]
            right_values = values[right_index]
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            distance, status = compute_vector_distance(left_values[valid], right_values[valid], metric="correlation")
            if status == VALID_COMPARISON_STATUS and np.isfinite(distance):
                distances[left_index, right_index] = distance
                distances[right_index, left_index] = distance
            left = prototypes.iloc[left_index]
            right = prototypes.iloc[right_index]
            rows.append(
                {
                    "view": view,
                    "prototype_left": labels[left_index],
                    "prototype_right": labels[right_index],
                    "stimulus_left": left["stimulus"],
                    "stimulus_right": right["stimulus"],
                    "date_left": left["date"],
                    "date_right": right["date"],
                    "same_anchor": bool(left["stimulus"] == right["stimulus"]),
                    "same_date": bool(left["date"] == right["date"]),
                    "distance": distance,
                    "comparison_status": status,
                }
            )

    rdm = pd.DataFrame(distances, index=labels, columns=labels)
    return rdm, pd.DataFrame(rows, columns=_pairwise_columns())


def _same_anchor_cross_date_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    subset = pairwise.loc[pairwise["same_anchor"] & ~pairwise["same_date"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["view", "n_pairs", "median_distance", "mean_distance"])
    return (
        subset.groupby("view", as_index=False)
        .agg(n_pairs=("distance", "count"), median_distance=("distance", "median"), mean_distance=("distance", "mean"))
        .sort_values("view")
        .reset_index(drop=True)
    )


def _stimulus_vs_date_contrast(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for view, group in pairwise.groupby("view", sort=True):
        same_anchor_cross_date = group.loc[group["same_anchor"] & ~group["same_date"], "distance"].dropna()
        different_anchor_within_date = group.loc[~group["same_anchor"] & group["same_date"], "distance"].dropna()
        rows.append(
            {
                "view": view,
                "same_anchor_cross_date_median": _median(same_anchor_cross_date),
                "different_anchor_within_date_median": _median(different_anchor_within_date),
                "date_effect_minus_stimulus_effect": _median(same_anchor_cross_date)
                - _median(different_anchor_within_date),
                "n_same_anchor_cross_date": int(len(same_anchor_cross_date)),
                "n_different_anchor_within_date": int(len(different_anchor_within_date)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "view",
            "same_anchor_cross_date_median",
            "different_anchor_within_date_median",
            "date_effect_minus_stimulus_effect",
            "n_same_anchor_cross_date",
            "n_different_anchor_within_date",
        ],
    )


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if FEATURE_COLUMN_PATTERN.match(str(column))]


def _pairwise_columns() -> list[str]:
    return [
        "view",
        "prototype_left",
        "prototype_right",
        "stimulus_left",
        "stimulus_right",
        "date_left",
        "date_right",
        "same_anchor",
        "same_date",
        "distance",
        "comparison_status",
    ]


def _combine(tables: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame(columns=columns)
    return pd.concat(tables, ignore_index=True).loc[:, columns]


def _median(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.median())


def _overall_median(summary: pd.DataFrame) -> float:
    if summary.empty:
        return float("nan")
    return float(pd.to_numeric(summary["median_distance"], errors="coerce").median())


def _overall_contrast(contrast: pd.DataFrame) -> float:
    if contrast.empty:
        return float("nan")
    return float(pd.to_numeric(contrast["date_effect_minus_stimulus_effect"], errors="coerce").median())


__all__ = [
    "run_anchor_batch_effect",
]
