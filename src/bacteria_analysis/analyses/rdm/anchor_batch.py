"""Anchor-stimulus batch/date-effect analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacteria_analysis.io import AnchorDataset, AnalysisResult
from bacteria_analysis.analyses.rdm.plots import (
    plot_anchor_clustered_rdm_heatmaps,
    plot_anchor_ideal_models,
    plot_anchor_rdm_heatmaps,
    plot_anchor_stimulus_date_mds,
    plot_anchor_stimulus_neuron_activity,
    plot_anchor_stimulus_neuron_time_heatmaps,
    plot_anchor_stimulus_same_vs_other_distributions,
    plot_stimulus_resolved_date_pair_heatmap,
)
from bacteria_analysis.features.anchor import (
    build_anchor_stimulus_neuron_activity,
    build_anchor_stimulus_neuron_time_activity,
    build_coverage,
    build_date_pair_same_vs_other_contrasts,
    build_pairwise_prototype_distances,
    build_prototypes,
    build_trial_features,
    summarize_date_anchors,
    summarize_date_pair_anchors,
    summarize_distance_categories,
    summarize_ideal_models,
    summarize_stimulus_anchors,
)

SUPPORTED_AGGREGATIONS = ("median", "mean")
REQUIRED_VIEWS = ("response_window", "full_trajectory")


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
    if tuple(views) != REQUIRED_VIEWS:
        raise ValueError("anchor-stimulus review figures require views=('response_window', 'full_trajectory')")

    base = _prepare_anchor_rows(anchor_dataset)
    trial_features = build_trial_features(base)
    prototypes = build_prototypes(trial_features)
    pairwise = build_pairwise_prototype_distances(prototypes)
    coverage = build_coverage(base)
    model_summary = summarize_ideal_models(pairwise)
    distance_summary = summarize_distance_categories(pairwise)
    stimulus_anchor_summary = summarize_stimulus_anchors(pairwise)
    date_pair_anchor_summary = summarize_date_pair_anchors(pairwise)
    date_anchor_summary = summarize_date_anchors(pairwise)
    same_vs_other_contrasts = build_date_pair_same_vs_other_contrasts(pairwise)
    activity_summary, activity_matrix, activity_distances = build_anchor_stimulus_neuron_activity(
        trial_features,
        view_name="response_window",
    )
    trajectory_summary = build_anchor_stimulus_neuron_time_activity(
        trial_features,
        view_name="full_trajectory",
        aggregator="median",
    )
    trajectory_mean_summary = build_anchor_stimulus_neuron_time_activity(
        trial_features,
        view_name="full_trajectory",
        aggregator="mean",
    )
    date_order = list(anchor_dataset.included_dates) or None
    rdms = _prototype_rdms(pairwise)
    figures = {
        "anchor_stimulus_date_prototype_rdms.png": lambda output_path: plot_anchor_rdm_heatmaps(
            pairwise,
            output_path,
        ),
        "anchor_stimulus_date_prototype_rdms__neural_clustered_stim_name.png": lambda output_path: plot_anchor_clustered_rdm_heatmaps(
            pairwise,
            output_path,
            view_name="response_window",
        ),
        "anchor_stimulus_date_mds__response_window.png": lambda output_path: plot_anchor_stimulus_date_mds(
            pairwise,
            output_path,
            view_name="response_window",
            date_order=date_order,
        ),
        "anchor_stimulus_date_pair_heatmap_by_stimulus__response_window.png": lambda output_path: plot_stimulus_resolved_date_pair_heatmap(
            pairwise,
            output_path,
            view_name="response_window",
            date_order=date_order,
        ),
        "anchor_stimulus_neuron_activity_heatmap__response_window.png": lambda output_path: plot_anchor_stimulus_neuron_activity(
            activity_matrix,
            output_path,
            view_name="response_window",
        ),
        "anchor_stimulus_neuron_time_heatmap__full_trajectory.png": lambda output_path: plot_anchor_stimulus_neuron_time_heatmaps(
            trajectory_summary,
            output_path,
            view_name="full_trajectory",
            aggregator_label="median",
        ),
        "anchor_stimulus_neuron_time_heatmap__full_trajectory__mean.png": lambda output_path: plot_anchor_stimulus_neuron_time_heatmaps(
            trajectory_mean_summary,
            output_path,
            view_name="full_trajectory",
            aggregator_label="mean",
        ),
        "anchor_stimulus_same_vs_other_stimuli__response_window.png": lambda output_path: plot_anchor_stimulus_same_vs_other_distributions(
            same_vs_other_contrasts,
            output_path,
            view_name="response_window",
        ),
        "anchor_stimulus_ideal_model_similarity.png": lambda output_path: plot_anchor_ideal_models(
            model_summary,
            output_path,
        ),
    }
    debug_tables = {}
    if include_debug:
        debug_tables = {
            "anchor_stimulus_trial_features": trial_features,
            "anchor_stimulus_date_prototypes": prototypes,
            "anchor_stimulus_pairwise_prototype_distances": pairwise,
            "anchor_stimulus_neuron_activity_summary__response_window": activity_summary,
            "anchor_stimulus_neuron_activity_matrix__response_window": activity_matrix.reset_index(),
            "anchor_stimulus_neuron_activity_distances__response_window": activity_distances,
            "anchor_stimulus_neuron_time_activity__full_trajectory": trajectory_summary,
            "anchor_stimulus_neuron_time_activity__full_trajectory__mean": trajectory_mean_summary,
        }

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
            "same_anchor_cross_date_median": _overall_anchor_median(stimulus_anchor_summary),
            "stimulus_vs_date_contrast_median": _overall_anchor_contrast(same_vs_other_contrasts),
        },
        tables={
            "anchor_stimulus_coverage": coverage,
            "anchor_stimulus_ideal_model_similarity": model_summary,
            "anchor_stimulus_distance_category_summary": distance_summary,
            "anchor_stimulus_cross_date_anchor_summary": stimulus_anchor_summary,
            "anchor_stimulus_same_stimulus_date_pair_summary": date_pair_anchor_summary,
            "anchor_stimulus_date_anchor_summary": date_anchor_summary,
            "anchor_stimulus_date_pair_same_vs_other_contrasts": same_vs_other_contrasts,
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
    if not anchors:
        return neural.iloc[0:0].copy()
    if "stimulus" in neural.columns:
        return neural.loc[neural["stimulus"].astype(str).isin(anchors)].copy()
    if "stim_name" in neural.columns:
        return neural.loc[neural["stim_name"].astype(str).isin(anchors)].copy()
    return neural.iloc[0:0].copy()


def _prepare_anchor_rows(anchor_dataset: AnchorDataset) -> pd.DataFrame:
    base = _filter_anchor_neural(anchor_dataset)
    if base.empty:
        raise ValueError("no anchor-stimulus rows found")
    base = base.copy()
    base["date"] = base["date"].astype(str)
    base["stimulus"] = base["stimulus"].astype(str)
    if "stim_name" not in base.columns:
        base["stim_name"] = base["stimulus"]
    base["stim_name"] = base["stim_name"].fillna("").astype(str).str.strip()
    base["trial_id"] = (
        base["date"]
        + "__"
        + base["worm_key"].astype(str)
        + "__"
        + base["segment_index"].astype(str)
    )
    if base["date"].nunique() < 2:
        raise ValueError("at least two dates are required to review anchor-stimulus date effects")
    return base


def _prototype_rdms(pairwise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rdms: dict[str, pd.DataFrame] = {}
    for view_name, view in pairwise.groupby("view_name", sort=True):
        labels = sorted(set(view["left_label"].astype(str)) | set(view["right_label"].astype(str)))
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
        np.fill_diagonal(matrix.values, 0.0)
        for _, row in view.iterrows():
            left = str(row["left_label"])
            right = str(row["right_label"])
            matrix.loc[left, right] = row["distance"]
            matrix.loc[right, left] = row["distance"]
        rdms[f"{view_name}_anchor"] = matrix
    return rdms


def _overall_anchor_median(summary: pd.DataFrame) -> float:
    column = "cross_date_same_stimulus_distance_median"
    if summary.empty or column not in summary.columns:
        return float("nan")
    return float(pd.to_numeric(summary[column], errors="coerce").median())


def _overall_anchor_contrast(contrast: pd.DataFrame) -> float:
    if contrast.empty:
        return float("nan")
    response = contrast.loc[contrast["view_name"].astype(str).eq("response_window")].copy()
    same = response.loc[response["contrast"].astype(str).eq("same"), "distance"]
    different = response.loc[response["contrast"].astype(str).eq("different"), "distance"]
    if same.empty or different.empty:
        return float("nan")
    return float(pd.to_numeric(same, errors="coerce").median() - pd.to_numeric(different, errors="coerce").median())


__all__ = [
    "run_anchor_batch_effect",
]
