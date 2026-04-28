"""Use anchor stimuli to estimate date effects."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


DEFAULT_INPUT_PATH = Path("data/202604/202604_data_withbaseodor.parquet")
DEFAULT_OUT = Path("results/202604_without_20260331/date_controlled_rsa_review/anchor_stimulus_date_effect")
ANCHOR_STIMULI = ("s3_0", "s6_1", "s8_2")
VIEW_WINDOWS = {
    "response_window": tuple(range(6, 21)),
    "full_trajectory": tuple(range(45)),
}
VIEW_LABELS = {
    "full_trajectory": "t0-44",
    "response_window": "t6-20",
}
BASELINE_TIMEPOINTS = tuple(range(6))
MERGE_GROUPS = {
    "ADF": ("ADFL", "ADFR"),
    "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"),
    "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"),
    "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"),
    "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
}
KEEP_NEURONS = ("ASEL", "ASER", "AWCOFF", "AWCON")


def main() -> None:
    args = parse_args()
    run(
        args.input_path,
        args.output_root,
        min_date_exclusive=args.min_date_exclusive,
        keep_dates=args.keep_dates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Parquet file containing anchor-stimulus trials.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory where review tables and figures are written.",
    )
    parser.add_argument(
        "--min-date-exclusive",
        type=str,
        default=None,
        help="Keep only rows with date > this YYYYMMDD value.",
    )
    parser.add_argument(
        "--keep-dates",
        nargs="+",
        default=None,
        help="Keep only these YYYYMMDD dates, in the order provided.",
    )
    return parser.parse_args()


def run(
    input_path: Path,
    output_root: Path,
    *,
    min_date_exclusive: str | None = None,
    keep_dates: list[str] | None = None,
) -> None:
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(input_path)
    base = raw.loc[raw["stimulus"].astype(str).isin(ANCHOR_STIMULI)].copy()
    if base.empty:
        raise ValueError(f"No anchor-stimulus rows found in {input_path}")
    base["date"] = base["date"].astype(str)
    base["stimulus"] = base["stimulus"].astype(str)
    base["stim_name"] = base["stim_name"].astype(str).str.strip()
    if keep_dates is not None:
        keep_dates = [str(date) for date in keep_dates]
        keep_set = set(keep_dates)
        base = base.loc[base["date"].isin(keep_set)].copy()
        if base.empty:
            raise ValueError(f"No anchor-stimulus rows remain after filtering keep_dates={keep_dates}")
        base["date"] = pd.Categorical(base["date"], categories=keep_dates, ordered=True)
    if min_date_exclusive is not None:
        base = base.loc[base["date"].gt(str(min_date_exclusive))].copy()
        if base.empty:
            raise ValueError(f"No anchor-stimulus rows remain after filtering dates > {min_date_exclusive}")
    if keep_dates is not None:
        base["date"] = base["date"].astype(str)
    base["trial_id"] = base["date"] + "__" + base["worm_key"].astype(str) + "__" + base["segment_index"].astype(str)
    if base["date"].nunique() < 2:
        raise ValueError("At least two dates are required after filtering to review date effects")

    coverage = build_coverage(base)
    trial_features = build_trial_features(base)
    prototypes = build_prototypes(trial_features)
    pairwise = build_pairwise_prototype_distances(prototypes)
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
    trajectory_summaries = {
        "full_trajectory": build_anchor_stimulus_neuron_time_activity(
            trial_features,
            view_name="full_trajectory",
            aggregator="median",
        )
    }
    trajectory_mean_summaries = {
        "full_trajectory": build_anchor_stimulus_neuron_time_activity(
            trial_features,
            view_name="full_trajectory",
            aggregator="mean",
        )
    }

    coverage.to_csv(tables / "anchor_stimulus_coverage.csv", index=False)
    trial_features.to_parquet(tables / "anchor_stimulus_trial_features.parquet", index=False)
    prototypes.to_parquet(tables / "anchor_stimulus_date_prototypes.parquet", index=False)
    pairwise.to_csv(tables / "anchor_stimulus_pairwise_prototype_distances.csv", index=False)
    model_summary.to_csv(tables / "anchor_stimulus_ideal_model_similarity.csv", index=False)
    distance_summary.to_csv(tables / "anchor_stimulus_distance_category_summary.csv", index=False)
    stimulus_anchor_summary.to_csv(tables / "anchor_stimulus_cross_date_anchor_summary.csv", index=False)
    date_pair_anchor_summary.to_csv(tables / "anchor_stimulus_same_stimulus_date_pair_summary.csv", index=False)
    date_anchor_summary.to_csv(tables / "anchor_stimulus_date_anchor_summary.csv", index=False)
    same_vs_other_contrasts.to_csv(tables / "anchor_stimulus_date_pair_same_vs_other_contrasts.csv", index=False)
    activity_summary.to_csv(tables / "anchor_stimulus_neuron_activity_summary__response_window.csv", index=False)
    activity_matrix.to_csv(tables / "anchor_stimulus_neuron_activity_matrix__response_window.csv")
    activity_distances.to_csv(tables / "anchor_stimulus_neuron_activity_distances__response_window.csv", index=False)
    for view_name, trajectory_summary in trajectory_summaries.items():
        trajectory_summary.to_csv(tables / f"anchor_stimulus_neuron_time_activity__{view_name}.csv", index=False)
    for view_name, trajectory_summary in trajectory_mean_summaries.items():
        trajectory_summary.to_csv(
            tables / f"anchor_stimulus_neuron_time_activity__{view_name}__mean.csv",
            index=False,
        )
    stale_trajectory_table = tables / "anchor_stimulus_neuron_time_activity__response_window.csv"
    if stale_trajectory_table.exists():
        stale_trajectory_table.unlink()

    plot_rdm_heatmaps(pairwise, figures / "anchor_stimulus_date_prototype_rdms.png")
    plot_clustered_rdm_heatmaps(
        pairwise,
        figures / "anchor_stimulus_date_prototype_rdms__neural_clustered_stim_name.png",
        view_name="response_window",
    )
    plot_anchor_stimulus_date_mds(
        pairwise,
        figures / "anchor_stimulus_date_mds__response_window.png",
        view_name="response_window",
        date_order=keep_dates,
    )
    stale_heatmap_paths = [
        figures / "base_odor_same_odor_date_pair_heatmaps.png",
        figures / "base_odor_same_odor_date_pair_heatmap__response_window.png",
        figures / "anchor_stimulus_same_stimulus_date_pair_heatmaps.png",
        figures / "anchor_stimulus_same_stimulus_date_pair_heatmap__response_window.png",
    ]
    for stale_heatmap_path in stale_heatmap_paths:
        if stale_heatmap_path.exists():
            stale_heatmap_path.unlink()
    plot_stimulus_resolved_date_pair_heatmap(
        pairwise,
        figures / "anchor_stimulus_date_pair_heatmap_by_stimulus__response_window.png",
        view_name="response_window",
        date_order=keep_dates,
    )
    plot_anchor_stimulus_neuron_activity(
        activity_matrix,
        figures / "anchor_stimulus_neuron_activity_heatmap__response_window.png",
        view_name="response_window",
    )
    for view_name, trajectory_summary in trajectory_summaries.items():
        plot_anchor_stimulus_neuron_time_heatmaps(
            trajectory_summary,
            figures / f"anchor_stimulus_neuron_time_heatmap__{view_name}.png",
            view_name=view_name,
            aggregator_label="median",
        )
    for view_name, trajectory_summary in trajectory_mean_summaries.items():
        plot_anchor_stimulus_neuron_time_heatmaps(
            trajectory_summary,
            figures / f"anchor_stimulus_neuron_time_heatmap__{view_name}__mean.png",
            view_name=view_name,
            aggregator_label="mean",
        )
    stale_trajectory_figure = figures / "anchor_stimulus_neuron_time_heatmap__response_window.png"
    if stale_trajectory_figure.exists():
        stale_trajectory_figure.unlink()
    stale_category_figure = figures / "base_odor_distance_categories.png"
    if stale_category_figure.exists():
        stale_category_figure.unlink()
    stale_identity_figure = figures / "base_odor_anchor_identity_contrast.png"
    if stale_identity_figure.exists():
        stale_identity_figure.unlink()
    stale_same_vs_other_figure = figures / "base_odor_matched_vs_mismatched_by_date_pair__response_window.png"
    if stale_same_vs_other_figure.exists():
        stale_same_vs_other_figure.unlink()
    plot_anchor_stimulus_same_vs_other_distributions(
        same_vs_other_contrasts,
        figures / "anchor_stimulus_same_vs_other_stimuli__response_window.png",
        view_name="response_window",
    )
    plot_ideal_models(model_summary, figures / "anchor_stimulus_ideal_model_similarity.png")
    write_summary(
        input_path,
        output_root,
        coverage,
        model_summary,
        stimulus_anchor_summary,
        date_anchor_summary,
        min_date_exclusive=min_date_exclusive,
        keep_dates=keep_dates,
    )


def build_coverage(base: pd.DataFrame) -> pd.DataFrame:
    trials = base[["date", "stimulus", "stim_name", "trial_id", "worm_key", "segment_index"]].drop_duplicates()
    return (
        trials.groupby(["stimulus", "stim_name", "date"], as_index=False)
        .agg(n_trials=("trial_id", "nunique"), n_worms=("worm_key", "nunique"))
        .sort_values(["stim_name", "date"])
    )


def build_trial_features(base: pd.DataFrame) -> pd.DataFrame:
    centered = baseline_center(base)
    merged = merge_non_ase_lr(centered)
    id_cols = ["trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index"]
    records = []
    for trial_id, trial in merged.groupby("trial_id", sort=True):
        meta = trial[id_cols].iloc[0].to_dict()
        pivot = trial.pivot_table(
            index="merged_neuron",
            columns="time_point",
            values="dff_baseline_centered",
            aggfunc="mean",
        )
        for view_name, timepoints in VIEW_WINDOWS.items():
            record = dict(meta)
            record["view_name"] = view_name
            for neuron in merged_neuron_order():
                for timepoint in timepoints:
                    record[f"{neuron}__t{timepoint:02d}"] = (
                        float(pivot.loc[neuron, timepoint])
                        if neuron in pivot.index and timepoint in pivot.columns and pd.notna(pivot.loc[neuron, timepoint])
                        else np.nan
                    )
            records.append(record)
    return pd.DataFrame(records)


def baseline_center(base: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["trial_id", "stimulus", "neuron"]
    baseline = (
        base.loc[base["time_point"].isin(BASELINE_TIMEPOINTS)]
        .groupby(group_cols, sort=False)["delta_F_over_F0"]
        .mean()
        .rename("baseline_mean")
        .reset_index()
    )
    centered = base.merge(baseline, on=group_cols, how="left")
    centered["dff_baseline_centered"] = centered["delta_F_over_F0"] - centered["baseline_mean"]
    return centered


def merge_non_ase_lr(centered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for merged_label, members in MERGE_GROUPS.items():
        part = centered.loc[centered["neuron"].isin(members)].copy()
        if part.empty:
            continue
        averaged = (
            part.groupby(
                ["trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index", "time_point"],
                as_index=False,
                sort=False,
            )["dff_baseline_centered"]
            .mean()
        )
        averaged["merged_neuron"] = merged_label
        rows.append(averaged)

    kept = centered.loc[centered["neuron"].isin(KEEP_NEURONS)].copy()
    kept["merged_neuron"] = kept["neuron"]
    rows.append(
        kept[
            [
                "trial_id",
                "date",
                "stimulus",
                "stim_name",
                "worm_key",
                "segment_index",
                "time_point",
                "dff_baseline_centered",
                "merged_neuron",
            ]
        ]
    )
    return pd.concat(rows, ignore_index=True)


def merged_neuron_order() -> list[str]:
    return list(MERGE_GROUPS) + list(KEEP_NEURONS)


def build_prototypes(trial_features: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["view_name", "date", "stimulus", "stim_name"]
    feature_cols = feature_columns(trial_features)
    rows = []
    for group_key, group in trial_features.groupby(id_cols, sort=True):
        row = dict(zip(id_cols, group_key, strict=True))
        row["n_trials"] = int(group["trial_id"].nunique())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.nanmedian(group[feature_cols].to_numpy(float), axis=0)
        row.update(dict(zip(feature_cols, values, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def build_anchor_stimulus_neuron_activity(
    trial_features: pd.DataFrame,
    *,
    view_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    view = trial_features.loc[trial_features["view_name"].astype(str).eq(view_name)].copy()
    if view.empty:
        raise ValueError(f"No trial features available for view {view_name!r}")

    id_cols = ["view_name", "trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index"]
    records = []
    for neuron in merged_neuron_order():
        neuron_cols = [column for column in feature_columns(view) if column.startswith(f"{neuron}__t")]
        if not neuron_cols:
            continue
        values = view[neuron_cols].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            trial_medians = np.nanmedian(values, axis=1)
        part = view[id_cols].copy()
        part["neuron"] = neuron
        part["trial_response_window_median"] = trial_medians
        records.append(part)

    if not records:
        raise ValueError(f"No neuron activity columns available for view {view_name!r}")

    long = pd.concat(records, ignore_index=True)
    long = long.loc[np.isfinite(long["trial_response_window_median"].to_numpy(float))].copy()
    if long.empty:
        raise ValueError(f"No finite neuron activity values available for view {view_name!r}")

    summary = (
        long.groupby(["view_name", "stimulus", "stim_name", "neuron"], as_index=False, sort=True)
        .agg(
            n_trials=("trial_id", "nunique"),
            n_valid_trial_neuron_values=("trial_response_window_median", "count"),
            activity_median=("trial_response_window_median", "median"),
            activity_q25=("trial_response_window_median", lambda values: float(values.quantile(0.25))),
            activity_q75=("trial_response_window_median", lambda values: float(values.quantile(0.75))),
        )
        .sort_values(["stim_name", "neuron"], kind="stable")
    )
    summary["stimulus_label"] = summary.apply(activity_stimulus_label, axis=1)

    matrix = (
        summary.pivot(index="neuron", columns="stimulus_label", values="activity_median")
        .reindex(index=merged_neuron_order())
        .dropna(axis=0, how="all")
    )
    matrix.index.name = "neuron"

    distances = build_activity_stimulus_distances(matrix, summary)
    return summary, matrix, distances


def activity_stimulus_label(row: pd.Series) -> str:
    return f"{row['stim_name']} ({row['stimulus']})"


def build_activity_stimulus_distances(matrix: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    label_meta = summary[["stimulus", "stim_name", "stimulus_label"]].drop_duplicates("stimulus_label")
    label_meta = label_meta.set_index("stimulus_label")
    rows = []
    for left, right in combinations(matrix.columns.astype(str), 2):
        distance = correlation_distance(matrix[left].to_numpy(float), matrix[right].to_numpy(float))
        rows.append(
            {
                "stimulus_left": label_meta.loc[left, "stimulus"],
                "stim_name_left": label_meta.loc[left, "stim_name"],
                "stimulus_right": label_meta.loc[right, "stimulus"],
                "stim_name_right": label_meta.loc[right, "stim_name"],
                "distance": distance,
            }
        )
    return pd.DataFrame(rows)


def build_anchor_stimulus_neuron_time_activity(
    trial_features: pd.DataFrame,
    *,
    view_name: str,
    aggregator: str = "median",
) -> pd.DataFrame:
    view = trial_features.loc[trial_features["view_name"].astype(str).eq(view_name)].copy()
    if view.empty:
        raise ValueError(f"No trial features available for view {view_name!r}")

    id_cols = ["view_name", "trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index"]
    records = []
    for neuron in merged_neuron_order():
        for timepoint in VIEW_WINDOWS[view_name]:
            column = f"{neuron}__t{timepoint:02d}"
            if column not in view.columns:
                continue
            part = view[id_cols].copy()
            part["neuron"] = neuron
            part["time_point"] = int(timepoint)
            part["activity"] = view[column].to_numpy(float)
            records.append(part)

    if not records:
        raise ValueError(f"No neuron time activity columns available for view {view_name!r}")

    long = pd.concat(records, ignore_index=True)
    long = long.loc[np.isfinite(long["activity"].to_numpy(float))].copy()
    if long.empty:
        raise ValueError(f"No finite neuron time activity values available for view {view_name!r}")

    if aggregator not in {"median", "mean"}:
        raise ValueError(f"Unsupported aggregator {aggregator!r}")

    activity_column = f"activity_{aggregator}"
    summary = (
        long.groupby(["view_name", "stimulus", "stim_name", "neuron", "time_point"], as_index=False, sort=True)
        .agg(
            n_trials=("trial_id", "nunique"),
            n_valid_trial_neuron_values=("activity", "count"),
            **{
                activity_column: ("activity", aggregator),
                "activity_q25": ("activity", lambda values: float(values.quantile(0.25))),
                "activity_q75": ("activity", lambda values: float(values.quantile(0.75))),
            },
        )
        .sort_values(["stimulus", "neuron", "time_point"], kind="stable")
    )
    summary["activity_value"] = summary[activity_column].to_numpy(float)
    summary["activity_aggregator"] = aggregator
    summary["stimulus_label"] = summary.apply(activity_stimulus_label, axis=1)
    return summary


def build_pairwise_prototype_distances(prototypes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view_name, view in prototypes.groupby("view_name", sort=True):
        feature_cols = feature_columns(view)
        view = view.reset_index(drop=True)
        for left_idx, right_idx in combinations(range(len(view)), 2):
            left = view.iloc[left_idx]
            right = view.iloc[right_idx]
            distance = correlation_distance(left[feature_cols].to_numpy(float), right[feature_cols].to_numpy(float))
            rows.append(
                {
                    "view_name": view_name,
                    "left_label": prototype_label(left),
                    "right_label": prototype_label(right),
                    "left_date": left["date"],
                    "right_date": right["date"],
                    "left_stimulus": left["stimulus"],
                    "right_stimulus": right["stimulus"],
                    "left_stim_name": left["stim_name"],
                    "right_stim_name": right["stim_name"],
                    "left_n_trials": int(left["n_trials"]),
                    "right_n_trials": int(right["n_trials"]),
                    "same_date": bool(left["date"] == right["date"]),
                    "same_stimulus": bool(left["stimulus"] == right["stimulus"]),
                    "pair_category": pair_category(left["date"] == right["date"], left["stimulus"] == right["stimulus"]),
                    "distance": distance,
                    "date_ideal_distance": 0.0 if left["date"] == right["date"] else 1.0,
                    "stimulus_ideal_distance": 0.0 if left["stimulus"] == right["stimulus"] else 1.0,
                }
            )
    return pd.DataFrame(rows)


def prototype_label(row: pd.Series) -> str:
    return f"{row['date']}__{row['stimulus']}__{row['stim_name']}"


def pair_category(same_date: bool, same_stimulus: bool) -> str:
    if same_date and same_stimulus:
        return "same_date_same_stimulus"
    if same_date:
        return "same_date_different_stimulus"
    if same_stimulus:
        return "different_date_same_stimulus"
    return "different_date_different_stimulus"


def correlation_distance(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    left = left[valid]
    right = right[valid]
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return 1.0 - float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def summarize_ideal_models(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view_name, view in pairwise.groupby("view_name", sort=True):
        stimulus_r = spearman(view["distance"], view["stimulus_ideal_distance"])
        date_r = spearman(view["distance"], view["date_ideal_distance"])
        partial_stimulus = partial_corr(view["distance"], view["stimulus_ideal_distance"], view["date_ideal_distance"])
        partial_date = partial_corr(view["distance"], view["date_ideal_distance"], view["stimulus_ideal_distance"])
        rows.append(
            {
                "view_name": view_name,
                "n_pairs": int(len(view)),
                "stimulus_ideal_spearman": stimulus_r,
                "date_ideal_spearman": date_r,
                "stimulus_ideal_partial_r": partial_stimulus,
                "date_ideal_partial_r": partial_date,
            }
        )
    return pd.DataFrame(rows)


def summarize_distance_categories(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (view_name, category), group in pairwise.groupby(["view_name", "pair_category"], sort=True):
        values = group["distance"].dropna().astype(float)
        rows.append(
            {
                "view_name": view_name,
                "pair_category": category,
                "n_pairs": int(len(values)),
                "distance_mean": float(values.mean()) if len(values) else np.nan,
                "distance_median": float(values.median()) if len(values) else np.nan,
                "distance_q25": float(values.quantile(0.25)) if len(values) else np.nan,
                "distance_q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "distance_min": float(values.min()) if len(values) else np.nan,
                "distance_max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_stimulus_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_stimulus")].copy()
    rows = []
    for (view_name, stimulus, stim_name), group in anchors.groupby(["view_name", "left_stimulus", "left_stim_name"], sort=True):
        values = group["distance"].dropna().astype(float)
        rows.append(
            {
                "view_name": view_name,
                "stimulus": stimulus,
                "stim_name": stim_name,
                "n_cross_date_pairs": int(len(values)),
                "cross_date_same_stimulus_distance_median": float(values.median()) if len(values) else np.nan,
                "cross_date_same_stimulus_distance_min": float(values.min()) if len(values) else np.nan,
                "cross_date_same_stimulus_distance_max": float(values.max()) if len(values) else np.nan,
                "date_pairs": ";".join(
                    sorted({f"{left}|{right}" for left, right in zip(group["left_date"], group["right_date"], strict=False)})
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_date_pair_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_stimulus")].copy()
    rows = []
    for (view_name, left_date, right_date), group in anchors.groupby(["view_name", "left_date", "right_date"], sort=True):
        values = group["distance"].dropna().astype(float)
        rows.append(
            {
                "view_name": view_name,
                "left_date": left_date,
                "right_date": right_date,
                "date_pair": f"{left_date}|{right_date}",
                "n_stimulus_anchors": int(group["left_stimulus"].nunique()),
                "stimulus_anchors": ";".join(sorted(group["left_stim_name"].astype(str).unique())),
                "distance_mean": float(values.mean()) if len(values) else np.nan,
                "distance_median": float(values.median()) if len(values) else np.nan,
                "distance_max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_date_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_stimulus")].copy()
    rows = []
    for _, row in anchors.iterrows():
        rows.append(
            {
                "view_name": row["view_name"],
                "date": row["left_date"],
                "other_date": row["right_date"],
                "stim_name": row["left_stim_name"],
                "distance": row["distance"],
            }
        )
        rows.append(
            {
                "view_name": row["view_name"],
                "date": row["right_date"],
                "other_date": row["left_date"],
                "stim_name": row["left_stim_name"],
                "distance": row["distance"],
            }
        )
    expanded = pd.DataFrame(rows)
    output = []
    for (view_name, date), group in expanded.groupby(["view_name", "date"], sort=True):
        values = group["distance"].dropna().astype(float)
        output.append(
            {
                "view_name": view_name,
                "date": date,
                "n_cross_date_same_stimulus_pairs": int(len(values)),
                "n_other_dates": int(group["other_date"].nunique()),
                "n_stimulus_anchors": int(group["stim_name"].nunique()),
                "distance_mean": float(values.mean()) if len(values) else np.nan,
                "distance_median": float(values.median()) if len(values) else np.nan,
                "distance_q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "distance_max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(output)


def build_date_pair_same_vs_other_contrasts(pairwise: pd.DataFrame) -> pd.DataFrame:
    cross_date = pairwise.loc[~pairwise["same_date"].astype(bool)].copy()
    rows: list[dict[str, object]] = []

    for (view_name, left_date, right_date), group in cross_date.groupby(["view_name", "left_date", "right_date"], sort=True):
        matched = group.loc[group["pair_category"].eq("different_date_same_stimulus")].copy()
        mismatched = group.loc[group["pair_category"].eq("different_date_different_stimulus")].copy()
        stimulus_order = sorted(set(matched["left_stim_name"].astype(str)) | set(matched["right_stim_name"].astype(str)))
        for anchor_stimulus in stimulus_order:
            matched_rows = matched.loc[
                matched["left_stim_name"].astype(str).eq(anchor_stimulus)
                & matched["right_stim_name"].astype(str).eq(anchor_stimulus)
            ]
            for _, row in matched_rows.iterrows():
                rows.append(
                    {
                        "view_name": view_name,
                        "left_date": left_date,
                        "right_date": right_date,
                        "date_pair": f"{left_date}|{right_date}",
                        "anchor_stimulus": anchor_stimulus,
                        "contrast": "same",
                        "anchor_side": "both",
                        "other_stimulus": anchor_stimulus,
                        "distance": row["distance"],
                    }
                )

            mismatched_rows = mismatched.loc[
                mismatched["left_stim_name"].astype(str).eq(anchor_stimulus)
                | mismatched["right_stim_name"].astype(str).eq(anchor_stimulus)
            ]
            for _, row in mismatched_rows.iterrows():
                anchor_on_left = bool(str(row["left_stim_name"]) == anchor_stimulus)
                rows.append(
                    {
                        "view_name": view_name,
                        "left_date": left_date,
                        "right_date": right_date,
                        "date_pair": f"{left_date}|{right_date}",
                        "anchor_stimulus": anchor_stimulus,
                        "contrast": "different",
                        "anchor_side": "left_date" if anchor_on_left else "right_date",
                        "other_stimulus": row["right_stim_name"] if anchor_on_left else row["left_stim_name"],
                        "distance": row["distance"],
                    }
                )

    return pd.DataFrame(rows)


def build_anchor_identity_contrasts(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    same_stimulus = pairwise.loc[pairwise["pair_category"].eq("different_date_same_stimulus")].copy()
    for _, row in same_stimulus.iterrows():
        rows.append(
            {
                "view_name": row["view_name"],
                "anchor_stimulus": row["left_stim_name"],
                "contrast": "same stimulus across dates",
                "date_context": f"{row['left_date']}|{row['right_date']}",
                "distance": row["distance"],
            }
        )

    same_date_different = pairwise.loc[pairwise["pair_category"].eq("same_date_different_stimulus")].copy()
    for _, row in same_date_different.iterrows():
        rows.append(
            {
                "view_name": row["view_name"],
                "anchor_stimulus": row["left_stim_name"],
                "contrast": "different stimulus same date",
                "date_context": str(row["left_date"]),
                "distance": row["distance"],
            }
        )
        rows.append(
            {
                "view_name": row["view_name"],
                "anchor_stimulus": row["right_stim_name"],
                "contrast": "different stimulus same date",
                "date_context": str(row["right_date"]),
                "distance": row["distance"],
            }
        )

    return pd.DataFrame(rows)


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left = pd.Series(left, dtype="float64")
    right = pd.Series(right, dtype="float64")
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    left_rank = left[valid].rank(method="average")
    right_rank = right[valid].rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def partial_corr(target: pd.Series, predictor: pd.Series, covariate: pd.Series) -> float:
    frame = pd.DataFrame({"target": target, "predictor": predictor, "covariate": covariate}).dropna()
    if len(frame) < 4 or frame["predictor"].nunique() < 2 or frame["covariate"].nunique() < 2:
        return np.nan
    target_rank = frame["target"].rank(method="average").to_numpy(float)
    predictor_rank = frame["predictor"].rank(method="average").to_numpy(float)
    covariate_rank = frame["covariate"].rank(method="average").to_numpy(float)
    target_resid = residualize(target_rank, covariate_rank)
    predictor_resid = residualize(predictor_rank, covariate_rank)
    if np.std(target_resid) == 0 or np.std(predictor_resid) == 0:
        return np.nan
    return float(np.corrcoef(target_resid, predictor_resid)[0, 1])


def residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariate)), covariate])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index", "view_name", "n_trials"}
    return [column for column in frame.columns if column not in excluded]


def plot_anchor_stimulus_neuron_activity(
    matrix: pd.DataFrame,
    path: Path,
    *,
    view_name: str,
) -> None:
    if matrix.empty:
        raise ValueError("Cannot plot an empty anchor-stimulus activity matrix")

    neuron_order = profile_cluster_order(matrix)
    stimulus_distance = profile_distance_matrix(matrix.T)
    stimulus_order = clustered_order(stimulus_distance) if can_cluster(stimulus_distance) else matrix.columns.astype(str).tolist()
    ordered = matrix.loc[neuron_order, stimulus_order]
    ordered_distances = stimulus_distance.loc[stimulus_order, stimulus_order]

    values = ordered.to_numpy(float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot plot anchor-stimulus activity without finite values")
    vmax = float(np.nanquantile(np.abs(finite_values), 0.98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(finite_values)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    dist_values = ordered_distances.to_numpy(float)
    finite_distances = dist_values[np.isfinite(dist_values)]
    dist_vmax = float(np.nanmax(finite_distances)) if finite_distances.size else 1.0
    if not np.isfinite(dist_vmax) or dist_vmax == 0.0:
        dist_vmax = 1.0

    activity_cmap = plt.get_cmap("RdBu_r").copy()
    activity_cmap.set_bad("#F1F1F1")
    distance_cmap = plt.get_cmap("magma").copy()
    distance_cmap.set_bad("#F1F1F1")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, max(5.0, 0.32 * len(ordered.index) + 2.0)),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    activity_ax, distance_ax = axes

    image = activity_ax.imshow(values, cmap=activity_cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    activity_ax.set_xticks(range(len(stimulus_order)))
    activity_ax.set_yticks(range(len(neuron_order)))
    activity_ax.set_xticklabels(stimulus_order, rotation=35, ha="right", fontsize=8)
    activity_ax.set_yticklabels(neuron_order, fontsize=8)
    activity_ax.set_title(f"Median activity ({view_label(view_name)})", fontsize=12)
    activity_ax.set_xlabel("Anchor stimulus", fontsize=9)
    activity_ax.set_ylabel("Neuron (non-ASE L/R merged; ASEL/ASER separate)", fontsize=9)
    activity_ax.set_xticks(np.arange(-0.5, len(stimulus_order), 1.0), minor=True)
    activity_ax.set_yticks(np.arange(-0.5, len(neuron_order), 1.0), minor=True)
    activity_ax.grid(which="minor", color="#FFFFFF", linewidth=0.8)
    activity_ax.tick_params(which="minor", bottom=False, left=False)
    annotate_heatmap(activity_ax, values, vmax=vmax)
    colorbar = fig.colorbar(image, ax=activity_ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Median baseline-centered dF/F0", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    distance_image = distance_ax.imshow(
        ordered_distances.to_numpy(float),
        cmap=distance_cmap,
        vmin=0.0,
        vmax=dist_vmax,
        interpolation="nearest",
    )
    distance_ax.set_xticks(range(len(stimulus_order)))
    distance_ax.set_yticks(range(len(stimulus_order)))
    distance_ax.set_xticklabels(stimulus_order, rotation=35, ha="right", fontsize=8)
    distance_ax.set_yticklabels(stimulus_order, fontsize=8)
    distance_ax.set_title("Stimulus profile distance", fontsize=12)
    distance_ax.set_xticks(np.arange(-0.5, len(stimulus_order), 1.0), minor=True)
    distance_ax.set_yticks(np.arange(-0.5, len(stimulus_order), 1.0), minor=True)
    distance_ax.grid(which="minor", color="#FFFFFF", linewidth=0.8)
    distance_ax.tick_params(which="minor", bottom=False, left=False)
    annotate_heatmap(distance_ax, ordered_distances.to_numpy(float), vmax=dist_vmax, decimals=2, low_values_dark=True)
    distance_colorbar = fig.colorbar(distance_image, ax=distance_ax, fraction=0.046, pad=0.03)
    distance_colorbar.set_label("1 - r across neuron medians", fontsize=9)
    distance_colorbar.ax.tick_params(labelsize=8)

    fig.suptitle("Anchor-stimulus neuron activity and profile relationship", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_anchor_stimulus_neuron_time_heatmaps(
    summary: pd.DataFrame,
    path: Path,
    *,
    view_name: str,
    aggregator_label: str | None = None,
) -> None:
    view = summary.loc[summary["view_name"].astype(str).eq(view_name)].copy()
    if view.empty:
        raise ValueError(f"No neuron time activity rows available for view {view_name!r}")

    stimulus_labels = ordered_activity_stimulus_labels(view)
    timepoints = list(VIEW_WINDOWS[view_name])
    neuron_order = [neuron for neuron in merged_neuron_order() if neuron in set(view["neuron"].astype(str))]

    matrices: dict[str, pd.DataFrame] = {}
    finite_values: list[float] = []
    for stimulus_label in stimulus_labels:
        stimulus_view = view.loc[view["stimulus_label"].astype(str).eq(stimulus_label)].copy()
        matrix = (
            stimulus_view.pivot(index="neuron", columns="time_point", values="activity_value")
            .reindex(index=neuron_order, columns=timepoints)
            .astype(float)
        )
        matrices[stimulus_label] = matrix
        values = matrix.to_numpy(float)
        finite_values.extend(values[np.isfinite(values)].tolist())

    if not finite_values:
        raise ValueError(f"No finite neuron time activity values available for view {view_name!r}")

    vmax = float(np.nanquantile(np.abs(np.asarray(finite_values, dtype=float)), 0.98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(np.asarray(finite_values, dtype=float))))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    cmap = anchor_trajectory_cmap()
    cmap.set_bad("#F7F7F7")
    fig, axes = plt.subplots(
        1,
        len(stimulus_labels),
        figsize=(max(13.6, 4.9 * len(stimulus_labels)), max(7.2, 0.48 * len(neuron_order) + 2.6)),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    last_image = None
    for ax, stimulus_label in zip(axes, stimulus_labels, strict=False):
        matrix = matrices[stimulus_label]
        image = ax.imshow(matrix.to_numpy(float), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        last_image = image
        ax.set_title(clean_stimulus_label(stimulus_label), fontsize=13, pad=10)
        ax.set_xticks(time_tick_positions(timepoints, view_name))
        ax.set_xticklabels(time_tick_labels(timepoints, view_name), fontsize=10)
        ax.set_yticks(range(len(neuron_order)))
        ax.set_yticklabels(neuron_order, fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(timepoints), 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, len(neuron_order), 1.0), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=0.35, alpha=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)
        draw_time_boundaries(ax, timepoints)

    axes[0].set_ylabel("Neuron", fontsize=11)
    axes[len(axes) // 2].set_xlabel("Time (s)", fontsize=11, labelpad=6)
    fig.subplots_adjust(left=0.09, right=0.865, bottom=0.13, top=0.90, wspace=0.10)
    colorbar_axis = fig.add_axes([0.885, 0.22, 0.016, 0.50])
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    label_prefix = f"{aggregator_label.capitalize()} " if aggregator_label else ""
    colorbar.set_label(f"{label_prefix}" + r"$\Delta F/F_0$", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def anchor_trajectory_cmap() -> plt.Colormap:
    return LinearSegmentedColormap.from_list(
        "anchor_trajectory",
        ["#3E5C76", "#98C1D9", "#F8F5F1", "#EEB479", "#B23A48"],
        N=256,
    )


def clean_stimulus_label(label: str) -> str:
    return label.split(" (", maxsplit=1)[0]


def ordered_activity_stimulus_labels(summary: pd.DataFrame) -> list[str]:
    meta = summary[["stimulus", "stimulus_label"]].drop_duplicates()
    label_by_stimulus = dict(zip(meta["stimulus"].astype(str), meta["stimulus_label"].astype(str), strict=False))
    labels = [label_by_stimulus[stimulus] for stimulus in ANCHOR_STIMULI if stimulus in label_by_stimulus]
    labels.extend(sorted(set(meta["stimulus_label"].astype(str)) - set(labels)))
    return labels


def time_tick_positions(timepoints: list[int], view_name: str) -> list[int]:
    ticks_by_view = {
        "full_trajectory": [0, 6, 16, 26, 36, 44],
        "response_window": [6, 10, 16, 20],
    }
    tick_values = ticks_by_view.get(view_name, list(np.linspace(timepoints[0], timepoints[-1], 5, dtype=int)))
    return [timepoints.index(value) for value in tick_values if value in timepoints]


def time_tick_labels(timepoints: list[int], view_name: str) -> list[str]:
    if view_name == "full_trajectory":
        tick_values = [0, 6, 16, 26, 36, 44]
        return [str(value - 6) for value in tick_values if value in timepoints]
    ticks_by_view = {
        "response_window": [6, 10, 16, 20],
    }
    tick_values = ticks_by_view.get(view_name, list(np.linspace(timepoints[0], timepoints[-1], 5, dtype=int)))
    return [str(value - 6) for value in tick_values if value in timepoints]


def draw_time_boundaries(ax: plt.Axes, timepoints: list[int]) -> None:
    min_time = min(timepoints)
    max_time = max(timepoints)
    for boundary in (6, 16):
        if min_time < boundary <= max_time:
            ax.axvline(boundary - min_time - 0.5, color="#2F2F2F", linestyle="--", linewidth=0.8)


def label_time_phases(ax: plt.Axes, timepoints: list[int]) -> None:
    min_time = min(timepoints)
    max_time = max(timepoints)
    for label, start, end in (("base", 0, 5), ("stim", 6, 15), ("post", 16, 44)):
        visible_start = max(start, min_time)
        visible_end = min(end, max_time)
        if visible_start > visible_end:
            continue
        center = ((visible_start + visible_end) / 2.0) - min_time
        ax.text(
            center,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    vmax: float,
    decimals: int = 2,
    low_values_dark: bool = False,
) -> None:
    threshold = 0.55 * vmax
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                ax.text(col_idx, row_idx, "NA", ha="center", va="center", fontsize=7, color="#555555")
                continue
            if low_values_dark:
                text_color = "#F8F8F8" if float(value) <= threshold else "#222222"
            else:
                text_color = "#F8F8F8" if abs(float(value)) >= threshold else "#222222"
            ax.text(
                col_idx,
                row_idx,
                f"{float(value):.{decimals}f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )


def profile_cluster_order(matrix: pd.DataFrame) -> list[str]:
    distances = profile_distance_matrix(matrix)
    if can_cluster(distances):
        return clustered_order(distances)
    return matrix.index.astype(str).tolist()


def profile_distance_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    labels = profiles.index.astype(str).tolist()
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    np.fill_diagonal(matrix.values, 0.0)
    for left, right in combinations(labels, 2):
        distance = correlation_distance(profiles.loc[left].to_numpy(float), profiles.loc[right].to_numpy(float))
        matrix.loc[left, right] = distance
        matrix.loc[right, left] = distance
    return matrix


def can_cluster(matrix: pd.DataFrame) -> bool:
    return len(matrix) >= 3 and bool(np.isfinite(matrix.to_numpy(float)).all())


def plot_rdm_heatmaps(pairwise: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for ax, (view_name, view) in zip(axes, pairwise.groupby("view_name", sort=True)):
        labels = sorted(set(view["left_label"]) | set(view["right_label"]))
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
        np.fill_diagonal(matrix.values, 0.0)
        for _, row in view.iterrows():
            matrix.loc[row["left_label"], row["right_label"]] = row["distance"]
            matrix.loc[row["right_label"], row["left_label"]] = row["distance"]
        image = ax.imshow(matrix.to_numpy(float), cmap="magma", interpolation="nearest")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([short_label(label) for label in labels], rotation=90, fontsize=7)
        ax.set_yticklabels([short_label(label) for label in labels], fontsize=7)
        ax.set_xlabel(view_label(view_name))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_clustered_rdm_heatmaps(pairwise: pd.DataFrame, path: Path, *, view_name: str) -> None:
    view = pairwise.loc[pairwise["view_name"].astype(str).eq(view_name)].copy()
    if view.empty:
        raise ValueError(f"No prototype distance rows available for view {view_name!r}")

    labels = sorted(set(view["left_label"]) | set(view["right_label"]))
    matrix = build_distance_matrix(view, labels)
    ordered_labels = clustered_order(matrix)
    ordered = matrix.loc[ordered_labels, ordered_labels]

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    image = ax.imshow(ordered.to_numpy(float), cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_xticklabels([prototype_display_label(label) for label in ordered_labels], rotation=90, fontsize=8)
    ax.set_yticklabels([prototype_display_label(label) for label in ordered_labels], fontsize=8)
    ax.set_title("Response-window clustered prototype RDM", fontsize=12)
    ax.set_xlabel("Clustered date / anchor stimulus labels", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Prototype distance", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_anchor_stimulus_date_mds(
    pairwise: pd.DataFrame,
    path: Path,
    *,
    view_name: str,
    date_order: list[str] | None = None,
) -> None:
    view = pairwise.loc[pairwise["view_name"].astype(str).eq(view_name)].copy()
    if view.empty:
        raise ValueError(f"No prototype distance rows available for view {view_name!r}")

    labels = sorted(set(view["left_label"]) | set(view["right_label"]))
    matrix = build_distance_matrix(view, labels)
    coords = classical_mds(matrix)
    coords["date"] = coords["label"].map(label_date)
    coords["stim_name"] = coords["label"].map(label_stim_name)

    if date_order is None:
        ordered_dates = sorted(coords["date"].astype(str).unique())
    else:
        ordered_dates = [str(date) for date in date_order if str(date) in set(coords["date"].astype(str))]

    stimulus_order = sorted(coords["stim_name"].astype(str).unique())
    stimulus_colors = {
        stimulus: color
        for stimulus, color in zip(
            stimulus_order,
            ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"],
            strict=False,
        )
    }
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    date_markers = {date: marker_cycle[idx % len(marker_cycle)] for idx, date in enumerate(ordered_dates)}

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for stimulus_name in stimulus_order:
        stimulus_points = coords.loc[coords["stim_name"].astype(str).eq(stimulus_name)].copy()
        stimulus_points["date_sort"] = stimulus_points["date"].astype(str).map(
            {date: idx for idx, date in enumerate(ordered_dates)}
        )
        stimulus_points = stimulus_points.sort_values("date_sort")
        ax.plot(
            stimulus_points["x"].to_numpy(float),
            stimulus_points["y"].to_numpy(float),
            color=stimulus_colors[stimulus_name],
            linewidth=1.4,
            alpha=0.65,
            zorder=1,
        )

    for date in ordered_dates:
        date_points = coords.loc[coords["date"].astype(str).eq(date)].copy()
        if date_points.empty:
            continue
        colors = [stimulus_colors[stimulus] for stimulus in date_points["stim_name"].astype(str)]
        ax.scatter(
            date_points["x"],
            date_points["y"],
            s=78,
            c=colors,
            marker=date_markers[date],
            edgecolors="#FFFFFF",
            linewidths=0.9,
            zorder=3,
            label=date,
        )

    x_pad = max(0.05, (float(coords["x"].max()) - float(coords["x"].min())) * 0.08)
    y_pad = max(0.05, (float(coords["y"].max()) - float(coords["y"].min())) * 0.08)
    for _, row in coords.iterrows():
        ax.text(
            float(row["x"]) + x_pad * 0.20,
            float(row["y"]) + y_pad * 0.18,
            short_date_label(str(row["date"])),
            fontsize=8,
            color="#555555",
            ha="left",
            va="bottom",
            zorder=4,
        )

    stimulus_handles = [
        plt.Line2D([0], [0], color=stimulus_colors[stimulus], linewidth=2.0, label=stimulus)
        for stimulus in stimulus_order
    ]
    date_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=date_markers[date],
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.9,
            markersize=7,
            label=date,
        )
        for date in ordered_dates
    ]
    stimulus_legend = ax.legend(
        handles=stimulus_handles,
        title="anchor stimulus",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(stimulus_legend)
    ax.legend(
        handles=date_handles,
        title="date",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_title("Anchor-stimulus prototypes across dates", fontsize=12)
    ax.set_xlabel("MDS axis 1", fontsize=9)
    ax.set_ylabel("MDS axis 2", fontsize=9)
    ax.grid(color="#E3E3E3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")
    fig.text(
        0.5,
        0.02,
        "Distances are correlation-distance relationships in 2D; lines connect the same stimulus across dates.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.86, 0.98))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_stimulus_resolved_date_pair_heatmap(
    pairwise: pd.DataFrame,
    path: Path,
    *,
    view_name: str,
    date_order: list[str] | None = None,
) -> None:
    view = pairwise.loc[
        pairwise["view_name"].astype(str).eq(view_name)
        & pairwise["same_stimulus"].astype(bool)
        & (~pairwise["same_date"].astype(bool))
    ].copy()
    if view.empty:
        raise ValueError(f"No same-stimulus cross-date rows available for view {view_name!r}")

    stimulus_order = sorted(view["left_stim_name"].astype(str).unique())
    if date_order is None:
        dates = sorted(set(view["left_date"].astype(str)) | set(view["right_date"].astype(str)))
    else:
        dates = [str(date) for date in date_order]

    matrices: dict[str, pd.DataFrame] = {}
    finite_values: list[float] = []
    for stimulus_name in stimulus_order:
        stimulus_view = view.loc[view["left_stim_name"].astype(str).eq(stimulus_name)].copy()
        matrix = pd.DataFrame(np.nan, index=dates, columns=dates, dtype=float)
        for _, row in stimulus_view.iterrows():
            left_date = str(row["left_date"])
            right_date = str(row["right_date"])
            if left_date not in matrix.index or right_date not in matrix.columns:
                continue
            value = float(row["distance"])
            matrix.loc[left_date, right_date] = value
            matrix.loc[right_date, left_date] = value
            finite_values.append(value)
        matrices[stimulus_name] = matrix

    if not finite_values:
        raise ValueError(f"No valid same-stimulus cross-date distances available for view {view_name!r}")

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#F3F1EE")
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    threshold = float(np.nanmedian(finite_values))

    fig, axes = plt.subplots(1, len(stimulus_order), figsize=(4.2 * len(stimulus_order), 4.8), sharex=True, sharey=True)
    if len(stimulus_order) == 1:
        axes = [axes]

    last_image = None
    for ax, stimulus_name in zip(axes, stimulus_order):
        display = matrices[stimulus_name].copy()
        image = ax.imshow(display.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        last_image = image
        ax.set_xticks(range(len(dates)))
        ax.set_yticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(dates, fontsize=8)
        ax.set_title(stimulus_name, fontsize=11)
        ax.set_xlabel(view_label(view_name), fontsize=9)

        for row_index, left_date in enumerate(dates):
            for col_index, right_date in enumerate(dates):
                value = display.iat[row_index, col_index]
                if row_index == col_index:
                    ax.add_patch(
                        plt.Rectangle(
                            (col_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            facecolor="#F8F7F4",
                            edgecolor="none",
                            zorder=3,
                        )
                    )
                    ax.text(
                        col_index,
                        row_index,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#777777",
                        zorder=4,
                    )
                    continue
                if not np.isfinite(value):
                    ax.add_patch(
                        plt.Rectangle(
                            (col_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            facecolor="#D9D9D9",
                            edgecolor="none",
                            zorder=3,
                        )
                    )
                    ax.text(
                        col_index,
                        row_index,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#555555",
                        zorder=4,
                    )
                    continue
                text_color = "#111111" if value >= threshold else "#F8F7F4"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    zorder=4,
                )

        for boundary in np.arange(-0.5, len(dates), 1.0):
            ax.axhline(boundary, color="#FFFFFF", linewidth=0.8, alpha=0.65)
            ax.axvline(boundary, color="#FFFFFF", linewidth=0.8, alpha=0.65)

    axes[0].set_ylabel("Date", fontsize=9)
    fig.suptitle("Anchor-stimulus cross-date distance by stimulus", fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.18, top=0.82, wspace=0.10)
    colorbar_axis = fig.add_axes([0.905, 0.20, 0.015, 0.60])
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Raw matched distance\nLower = less drift", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_anchor_stimulus_same_vs_other_distributions(
    contrast: pd.DataFrame,
    path: Path,
    *,
    view_name: str,
) -> None:
    view = contrast.loc[contrast["view_name"].astype(str).eq(view_name)].copy()
    view["distance"] = pd.to_numeric(view["distance"], errors="coerce")
    view = view.dropna(subset=["distance"])
    if view.empty:
        raise ValueError(f"No same-vs-different rows available for view {view_name!r}")

    stimulus_order = sorted(view["anchor_stimulus"].astype(str).unique())
    category_order = ["same", "different"]
    category_colors = {"same": "#0072B2", "different": "#8F8F8F"}
    category_labels = {"same": "same stimulus", "different": "other stimuli"}

    values = view["distance"].to_numpy(float)
    y_min = max(0.0, float(np.nanmin(values)) - 0.08)
    y_max = float(np.nanmax(values)) + 0.08

    fig, axes = plt.subplots(1, len(stimulus_order), figsize=(4.2 * len(stimulus_order), 4.8), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(20260423)

    for ax, stimulus_name in zip(axes, stimulus_order):
        stimulus_view = view.loc[view["anchor_stimulus"].astype(str).eq(stimulus_name)].copy()
        box_data: list[np.ndarray] = []
        box_positions: list[float] = []
        box_categories: list[str] = []
        tick_labels: list[str] = []

        for position, category in enumerate(category_order):
            category_values = stimulus_view.loc[stimulus_view["contrast"].astype(str).eq(category), "distance"].to_numpy(
                float
            )
            tick_labels.append(f"{category_labels[category]}\n(n={len(category_values)})")
            if category_values.size == 0:
                continue
            box_data.append(category_values)
            box_positions.append(float(position))
            box_categories.append(category)

        if box_data:
            boxplot = ax.boxplot(
                box_data,
                positions=box_positions,
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 1.4},
                whiskerprops={"color": "#777777", "linewidth": 1.0},
                capprops={"color": "#777777", "linewidth": 1.0},
                boxprops={"linewidth": 1.2},
            )
            for box, category in zip(boxplot["boxes"], box_categories, strict=False):
                box.set_facecolor(category_colors[category])
                box.set_alpha(0.18)
                box.set_edgecolor(category_colors[category])

        for position, category in enumerate(category_order):
            category_values = stimulus_view.loc[stimulus_view["contrast"].astype(str).eq(category), "distance"].to_numpy(
                float
            )
            if category_values.size == 0:
                continue
            jitter = rng.normal(0.0, 0.045, size=category_values.size)
            ax.scatter(
                np.full(category_values.size, position, dtype=float) + jitter,
                category_values,
                s=24,
                color=category_colors[category],
                edgecolors="none",
                alpha=0.8,
                zorder=3,
            )

        ax.set_title(stimulus_name, fontsize=11)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("correlation distance (1 - r)", fontsize=10)
    fig.suptitle("Anchor-stimulus cross-date distances", fontsize=13, y=0.98)
    fig.text(
        0.5,
        0.02,
        "same stimulus = the stimulus compared to itself across dates; other stimuli = the stimulus compared to different stimuli across dates.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.06, 1, 0.90))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_anchor_identity_contrast(pairwise: pd.DataFrame, path: Path) -> None:
    contrast = build_anchor_identity_contrasts(pairwise)
    contrast = contrast.dropna(subset=["distance"]).copy()
    if contrast.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        ax.text(
            0.5,
            0.5,
            "No anchor-stimulus contrasts available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return

    stimulus_order = sorted(contrast["anchor_stimulus"].astype(str).unique())
    contrast_order = ["same stimulus across dates", "different stimulus same date"]
    contrast_labels = {
        "same stimulus across dates": "same stimulus\ncross date",
        "different stimulus same date": "different stimulus\nsame date",
    }
    colors = {
        "same stimulus across dates": "#0072B2",
        "different stimulus same date": "#D55E00",
    }
    offsets = {
        "same stimulus across dates": -0.18,
        "different stimulus same date": 0.18,
    }

    views = [view for view in ("response_window", "full_trajectory") if view in set(contrast["view_name"])]
    if not views:
        views = sorted(contrast["view_name"].astype(str).unique())
    fig, axes = plt.subplots(1, len(views), figsize=(5.8 * len(views), 4.6), sharey=True)
    if len(views) == 1:
        axes = [axes]

    rng = np.random.default_rng(20260420)
    for ax, view_name in zip(axes, views):
        view = contrast.loc[contrast["view_name"].eq(view_name)].copy()
        for stimulus_index, stimulus in enumerate(stimulus_order):
            stimulus_rows = view.loc[view["anchor_stimulus"].astype(str).eq(stimulus)]
            for contrast_name in contrast_order:
                rows = stimulus_rows.loc[stimulus_rows["contrast"].eq(contrast_name)]
                values = pd.to_numeric(rows["distance"], errors="coerce").dropna().to_numpy(float)
                if values.size == 0:
                    continue
                x = stimulus_index + offsets[contrast_name] + rng.normal(0.0, 0.025, size=values.size)
                ax.scatter(x, values, s=18, alpha=0.55, color=colors[contrast_name], linewidths=0)
                median = float(np.median(values))
                ax.plot(
                    [
                        stimulus_index + offsets[contrast_name] - 0.10,
                        stimulus_index + offsets[contrast_name] + 0.10,
                    ],
                    [median, median],
                    color="#222222",
                    linewidth=2.0,
                )

        ax.set_title(view_label(view_name))
        ax.set_xticks(range(len(stimulus_order)))
        ax.set_xticklabels(stimulus_order, rotation=25, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[name], markersize=6)
            for name in contrast_order
        ]
        ax.legend(handles, [contrast_labels[name].replace("\n", " ") for name in contrast_order], frameon=False, fontsize=8)

    axes[0].set_ylabel("correlation distance")
    fig.suptitle("Anchor-stimulus identity across dates", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_distance_matrix(view: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    np.fill_diagonal(matrix.values, 0.0)
    for _, row in view.iterrows():
        matrix.loc[row["left_label"], row["right_label"]] = row["distance"]
        matrix.loc[row["right_label"], row["left_label"]] = row["distance"]
    return matrix


def classical_mds(matrix: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    values = matrix.to_numpy(float)
    n_items = len(matrix)
    identity = np.eye(n_items)
    centering = identity - np.full((n_items, n_items), 1.0 / n_items)
    gram = -0.5 * centering @ (values**2) @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = eigvals > 0
    eigvals = eigvals[positive][:n_components]
    eigvecs = eigvecs[:, positive][:, :n_components]
    if eigvals.size == 0:
        coords = np.zeros((n_items, n_components), dtype=float)
    else:
        coords = eigvecs * np.sqrt(eigvals)
        if coords.shape[1] < n_components:
            coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])), constant_values=0.0)
    return pd.DataFrame({"label": matrix.index, "x": coords[:, 0], "y": coords[:, 1]})


def clustered_order(matrix: pd.DataFrame) -> list[str]:
    values = matrix.to_numpy(float)
    if not np.isfinite(values).all() or len(matrix) < 3:
        return matrix.index.tolist()
    condensed = squareform(values, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=True)
    return matrix.index.to_numpy()[leaves_list(tree)].tolist()


def short_label(label: str) -> str:
    parts = label.split("__")
    if len(parts) < 3:
        return label
    return f"{parts[0]} {parts[1]}"


def stim_name_label(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    if len(parts) < 3:
        return label
    date, _, stim_name = parts
    return f"{date} {stim_name}"


def prototype_display_label(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    if len(parts) < 3:
        return label
    date, _, stim_name = parts
    return f"{date}\n{stim_name}"


def label_date(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    return parts[0] if len(parts) >= 1 else label


def label_stim_name(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    return parts[2] if len(parts) >= 3 else label


def short_date_label(date: str) -> str:
    if len(date) != 8:
        return date
    return f"{date[4:6]}-{date[6:8]}"


def view_label(view_name: str) -> str:
    return VIEW_LABELS.get(view_name, view_name)


def plot_distance_categories(summary: pd.DataFrame, path: Path) -> None:
    order = ["different_date_same_stimulus", "same_date_different_stimulus", "different_date_different_stimulus"]
    labels = {
        "different_date_same_stimulus": "same stimulus\ncross date",
        "same_date_different_stimulus": "same date\ndifferent stimulus",
        "different_date_different_stimulus": "different stimulus\ncross date",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (view_name, view) in zip(axes, summary.groupby("view_name", sort=True)):
        view = view.set_index("pair_category")
        x = np.arange(len(order))
        y = [view.loc[item, "distance_median"] if item in view.index else np.nan for item in order]
        low = [view.loc[item, "distance_q25"] if item in view.index else np.nan for item in order]
        high = [view.loc[item, "distance_q75"] if item in view.index else np.nan for item in order]
        ax.vlines(x, low, high, color="#777777", linewidth=2)
        ax.scatter(x, y, color="#0072B2", s=40)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[item] for item in order])
        ax.set_xlabel(view_label(view_name))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("correlation distance")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_ideal_models(summary: pd.DataFrame, path: Path) -> None:
    metrics = ["stimulus_ideal_spearman", "date_ideal_spearman", "stimulus_ideal_partial_r", "date_ideal_partial_r"]
    labels = ["stimulus", "date", "stimulus partial", "date partial"]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(metrics))
    width = 0.32
    for offset, (view_name, view) in zip([-width / 2, width / 2], summary.groupby("view_name", sort=True)):
        row = view.iloc[0]
        ax.bar(x + offset, [row[metric] for metric in metrics], width=width, label=view_label(view_name))
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("similarity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(
    input_path: Path,
    output_root: Path,
    coverage: pd.DataFrame,
    model_summary: pd.DataFrame,
    stimulus_anchor_summary: pd.DataFrame,
    date_anchor_summary: pd.DataFrame,
    *,
    min_date_exclusive: str | None,
    keep_dates: list[str] | None,
) -> None:
    stimulus_names = ", ".join(f"`{name}`" for name in sorted(coverage["stim_name"].astype(str).unique()))
    lines = [
        "# Anchor-Stimulus Date-Effect Review",
        "",
        f"- Input: `{input_path.as_posix()}`",
        f"- Anchor stimuli: {stimulus_names}.",
        "- Neural representation: baseline-centered, non-ASE L/R merged, ASEL/ASER kept separate, trial-median prototypes.",
        "- Purpose: use anchor stimuli to estimate date effects that could not be separated from disjoint bacteria panels.",
    ]
    if keep_dates is not None:
        lines.append("- Date filter: keep only dates " + ", ".join(f"`{date}`" for date in keep_dates) + ".")
    if min_date_exclusive is not None:
        lines.append(f"- Date filter: keep only dates after `{min_date_exclusive}`.")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            markdown_table(coverage),
            "",
            "## Ideal Model Similarity",
            "",
            markdown_table(model_summary),
            "",
            "## Cross-Date Same-Stimulus Anchors",
            "",
            markdown_table(stimulus_anchor_summary),
            "",
            "## Per-Date Same-Stimulus Cross-Date Anchors",
            "",
            markdown_table(date_anchor_summary),
            "",
            "## Output Files",
            "",
            "- `tables/anchor_stimulus_coverage.csv`",
            "- `tables/anchor_stimulus_ideal_model_similarity.csv`",
            "- `tables/anchor_stimulus_cross_date_anchor_summary.csv`",
            "- `tables/anchor_stimulus_same_stimulus_date_pair_summary.csv`",
            "- `tables/anchor_stimulus_date_anchor_summary.csv`",
            "- `tables/anchor_stimulus_date_pair_same_vs_other_contrasts.csv`",
            "- `tables/anchor_stimulus_neuron_activity_summary__response_window.csv`",
            "- `tables/anchor_stimulus_neuron_activity_matrix__response_window.csv`",
            "- `tables/anchor_stimulus_neuron_activity_distances__response_window.csv`",
            "- `tables/anchor_stimulus_neuron_time_activity__full_trajectory.csv`",
            "- `tables/anchor_stimulus_neuron_time_activity__full_trajectory__mean.csv`",
            "- `figures/anchor_stimulus_date_prototype_rdms.png`",
            "- `figures/anchor_stimulus_date_prototype_rdms__neural_clustered_stim_name.png`",
            "- `figures/anchor_stimulus_date_mds__response_window.png`",
            "- `figures/anchor_stimulus_date_pair_heatmap_by_stimulus__response_window.png`",
            "- `figures/anchor_stimulus_neuron_activity_heatmap__response_window.png`",
            "- `figures/anchor_stimulus_neuron_time_heatmap__full_trajectory.png`",
            "- `figures/anchor_stimulus_neuron_time_heatmap__full_trajectory__mean.png`",
            "- `figures/anchor_stimulus_same_vs_other_stimuli__response_window.png`",
            "- `figures/anchor_stimulus_ideal_model_similarity.png`",
        ]
    )
    (output_root / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns]
    for _, row in frame.iterrows():
        rows.append([format_cell(row[column]) for column in columns])
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(columns))]
    rendered = ["| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(rows[0])) + " |"]
    rendered.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows[1:]:
        rendered.append("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |")
    return "\n".join(rendered)


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
