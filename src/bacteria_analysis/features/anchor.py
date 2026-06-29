"""Reusable anchor-stimulus date-effect helpers."""

from __future__ import annotations

from itertools import combinations
import warnings

import numpy as np
import pandas as pd

VIEW_WINDOWS = {
    "response_window": tuple(range(5, 25)),
    "full_trajectory": tuple(range(45)),
}
BASELINE_TIMEPOINTS = tuple(range(6))
NEURON_GROUPS = {
    "ADF": ("ADFL", "ADFR"),
    "ADL": ("ADLL", "ADLR"),
    "ASG": ("ASGL", "ASGR"),
    "ASH": ("ASHL", "ASHR"),
    "ASI": ("ASIL", "ASIR"),
    "ASJ": ("ASJL", "ASJR"),
    "ASK": ("ASKL", "ASKR"),
    "AWA": ("AWAL", "AWAR"),
    "AWB": ("AWBL", "AWBR"),
    "ASEL": ("ASEL",),
    "ASER": ("ASER",),
    "AWCOFF": ("AWCOFF",),
    "AWCON": ("AWCON",),
}


def build_coverage(base: pd.DataFrame) -> pd.DataFrame:
    trials = base[["date", "stimulus", "stim_name", "trial_id", "worm_key", "segment_index"]].drop_duplicates()
    return (
        trials.groupby(["stimulus", "stim_name", "date"], as_index=False)
        .agg(n_trials=("trial_id", "nunique"), n_worms=("worm_key", "nunique"))
        .sort_values(["stim_name", "date"])
    )


def build_trial_features(base: pd.DataFrame) -> pd.DataFrame:
    centered = baseline_center(base)
    merged = merge_neurons(centered)
    id_cols = ["trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index"]
    records = []
    for _, trial in merged.groupby("trial_id", sort=True):
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
                    value = pivot.loc[neuron, timepoint] if neuron in pivot.index and timepoint in pivot.columns else np.nan
                    record[f"{neuron}__t{timepoint:02d}"] = float(value) if pd.notna(value) else np.nan
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


def merge_neurons(
    centered: pd.DataFrame,
    neuron_groups: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    groups = neuron_groups or NEURON_GROUPS
    group_cols = ["trial_id", "date", "stimulus", "stim_name", "worm_key", "segment_index", "time_point"]
    rows = []
    for merged_label, members in groups.items():
        part = centered.loc[centered["neuron"].isin(members)].copy()
        if part.empty:
            continue
        averaged = (
            part.groupby(
                group_cols,
                as_index=False,
                sort=False,
            )["dff_baseline_centered"]
            .mean()
        )
        averaged["merged_neuron"] = merged_label
        rows.append(averaged)

    merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if merged.empty:
        raise ValueError("no supported anchor-stimulus neurons found")
    return merged


def merged_neuron_order() -> list[str]:
    return list(NEURON_GROUPS)


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
    if aggregator not in {"median", "mean"}:
        raise ValueError(f"Unsupported aggregator {aggregator!r}")

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
        rows.append(
            {
                "view_name": view_name,
                "n_pairs": int(len(view)),
                "stimulus_ideal_spearman": spearman(view["distance"], view["stimulus_ideal_distance"]),
                "date_ideal_spearman": spearman(view["distance"], view["date_ideal_distance"]),
                "stimulus_ideal_partial_r": partial_corr(
                    view["distance"],
                    view["stimulus_ideal_distance"],
                    view["date_ideal_distance"],
                ),
                "date_ideal_partial_r": partial_corr(
                    view["distance"],
                    view["date_ideal_distance"],
                    view["stimulus_ideal_distance"],
                ),
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
    for (view_name, stimulus, stim_name), group in anchors.groupby(
        ["view_name", "left_stimulus", "left_stim_name"],
        sort=True,
    ):
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
                    sorted(
                        {
                            f"{left}|{right}"
                            for left, right in zip(group["left_date"], group["right_date"], strict=False)
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_date_pair_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_stimulus")].copy()
    rows = []
    for (view_name, left_date, right_date), group in anchors.groupby(
        ["view_name", "left_date", "right_date"],
        sort=True,
    ):
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

    for (view_name, left_date, right_date), group in cross_date.groupby(
        ["view_name", "left_date", "right_date"],
        sort=True,
    ):
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


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_values = pd.Series(left, dtype="float64")
    right_values = pd.Series(right, dtype="float64")
    valid = left_values.notna() & right_values.notna() & np.isfinite(left_values) & np.isfinite(right_values)
    if valid.sum() < 2:
        return np.nan
    left_rank = left_values[valid].rank(method="average")
    right_rank = right_values[valid].rank(method="average")
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


__all__ = [
    "build_anchor_stimulus_neuron_activity",
    "build_anchor_stimulus_neuron_time_activity",
    "build_coverage",
    "build_date_pair_same_vs_other_contrasts",
    "build_pairwise_prototype_distances",
    "build_prototypes",
    "build_trial_features",
    "merge_neurons",
    "summarize_date_anchors",
    "summarize_date_pair_anchors",
    "summarize_distance_categories",
    "summarize_ideal_models",
    "summarize_stimulus_anchors",
]
