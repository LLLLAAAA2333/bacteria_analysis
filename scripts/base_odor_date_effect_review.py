"""Use repeated base odors as anchors to estimate date effects."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


DEFAULT_INPUT_PATH = Path("data/202604/202604_data_withbaseodor.parquet")
DEFAULT_OUT = Path("results/202604_without_20260331/date_controlled_rsa_review/base_odor_date_effect")
BASE_ODOR_STIMULI = ("s3_0", "s6_1", "s8_2")
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
        help="Parquet file containing repeated base-odor trials.",
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
    base = raw.loc[raw["stimulus"].astype(str).isin(BASE_ODOR_STIMULI)].copy()
    if base.empty:
        raise ValueError(f"No base-odor rows found in {input_path}")
    base["date"] = base["date"].astype(str)
    base["stimulus"] = base["stimulus"].astype(str)
    base["stim_name"] = base["stim_name"].astype(str).str.strip()
    if keep_dates is not None:
        keep_dates = [str(date) for date in keep_dates]
        keep_set = set(keep_dates)
        base = base.loc[base["date"].isin(keep_set)].copy()
        if base.empty:
            raise ValueError(f"No base-odor rows remain after filtering keep_dates={keep_dates}")
        base["date"] = pd.Categorical(base["date"], categories=keep_dates, ordered=True)
    if min_date_exclusive is not None:
        base = base.loc[base["date"].gt(str(min_date_exclusive))].copy()
        if base.empty:
            raise ValueError(f"No base-odor rows remain after filtering dates > {min_date_exclusive}")
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
    odor_anchor_summary = summarize_odor_anchors(pairwise)
    date_pair_anchor_summary = summarize_date_pair_anchors(pairwise)
    date_anchor_summary = summarize_date_anchors(pairwise)
    same_vs_other_contrasts = build_date_pair_same_vs_other_contrasts(pairwise)

    coverage.to_csv(tables / "base_odor_coverage.csv", index=False)
    trial_features.to_parquet(tables / "base_odor_trial_features.parquet", index=False)
    prototypes.to_parquet(tables / "base_odor_date_prototypes.parquet", index=False)
    pairwise.to_csv(tables / "base_odor_pairwise_prototype_distances.csv", index=False)
    model_summary.to_csv(tables / "base_odor_ideal_model_similarity.csv", index=False)
    distance_summary.to_csv(tables / "base_odor_distance_category_summary.csv", index=False)
    odor_anchor_summary.to_csv(tables / "base_odor_cross_date_anchor_summary.csv", index=False)
    date_pair_anchor_summary.to_csv(tables / "base_odor_same_odor_date_pair_summary.csv", index=False)
    date_anchor_summary.to_csv(tables / "base_odor_date_anchor_summary.csv", index=False)
    same_vs_other_contrasts.to_csv(tables / "base_odor_date_pair_same_vs_other_contrasts.csv", index=False)

    plot_rdm_heatmaps(pairwise, figures / "base_odor_date_prototype_rdms.png")
    plot_clustered_rdm_heatmaps(
        pairwise,
        figures / "base_odor_date_prototype_rdms__neural_clustered_stim_name.png",
        view_name="response_window",
    )
    plot_base_odor_date_mds(
        pairwise,
        figures / "base_odor_date_mds__response_window.png",
        view_name="response_window",
        date_order=keep_dates,
    )
    stale_heatmap_paths = [
        figures / "base_odor_same_odor_date_pair_heatmaps.png",
        figures / "base_odor_same_odor_date_pair_heatmap__response_window.png",
    ]
    for stale_heatmap_path in stale_heatmap_paths:
        if stale_heatmap_path.exists():
            stale_heatmap_path.unlink()
    plot_odor_resolved_date_pair_heatmap(
        pairwise,
        figures / "base_odor_date_pair_heatmap_by_odor__response_window.png",
        view_name="response_window",
        date_order=keep_dates,
    )
    stale_category_figure = figures / "base_odor_distance_categories.png"
    if stale_category_figure.exists():
        stale_category_figure.unlink()
    stale_identity_figure = figures / "base_odor_anchor_identity_contrast.png"
    if stale_identity_figure.exists():
        stale_identity_figure.unlink()
    stale_same_vs_other_figure = figures / "base_odor_matched_vs_mismatched_by_date_pair__response_window.png"
    if stale_same_vs_other_figure.exists():
        stale_same_vs_other_figure.unlink()
    plot_base_odor_same_vs_other_distributions(
        same_vs_other_contrasts,
        figures / "base_odor_same_vs_other_odors__response_window.png",
        view_name="response_window",
    )
    plot_ideal_models(model_summary, figures / "base_odor_ideal_model_similarity.png")
    write_summary(
        input_path,
        output_root,
        coverage,
        model_summary,
        odor_anchor_summary,
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
                    "same_odor": bool(left["stimulus"] == right["stimulus"]),
                    "pair_category": pair_category(left["date"] == right["date"], left["stimulus"] == right["stimulus"]),
                    "distance": distance,
                    "date_ideal_distance": 0.0 if left["date"] == right["date"] else 1.0,
                    "odor_ideal_distance": 0.0 if left["stimulus"] == right["stimulus"] else 1.0,
                }
            )
    return pd.DataFrame(rows)


def prototype_label(row: pd.Series) -> str:
    return f"{row['date']}__{row['stimulus']}__{row['stim_name']}"


def pair_category(same_date: bool, same_odor: bool) -> str:
    if same_date and same_odor:
        return "same_date_same_odor"
    if same_date:
        return "same_date_different_odor"
    if same_odor:
        return "different_date_same_odor"
    return "different_date_different_odor"


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
        odor_r = spearman(view["distance"], view["odor_ideal_distance"])
        date_r = spearman(view["distance"], view["date_ideal_distance"])
        partial_odor = partial_corr(view["distance"], view["odor_ideal_distance"], view["date_ideal_distance"])
        partial_date = partial_corr(view["distance"], view["date_ideal_distance"], view["odor_ideal_distance"])
        rows.append(
            {
                "view_name": view_name,
                "n_pairs": int(len(view)),
                "odor_ideal_spearman": odor_r,
                "date_ideal_spearman": date_r,
                "odor_ideal_partial_r": partial_odor,
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


def summarize_odor_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_odor")].copy()
    rows = []
    for (view_name, stimulus, stim_name), group in anchors.groupby(["view_name", "left_stimulus", "left_stim_name"], sort=True):
        values = group["distance"].dropna().astype(float)
        rows.append(
            {
                "view_name": view_name,
                "stimulus": stimulus,
                "stim_name": stim_name,
                "n_cross_date_pairs": int(len(values)),
                "cross_date_same_odor_distance_median": float(values.median()) if len(values) else np.nan,
                "cross_date_same_odor_distance_min": float(values.min()) if len(values) else np.nan,
                "cross_date_same_odor_distance_max": float(values.max()) if len(values) else np.nan,
                "date_pairs": ";".join(
                    sorted({f"{left}|{right}" for left, right in zip(group["left_date"], group["right_date"], strict=False)})
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_date_pair_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_odor")].copy()
    rows = []
    for (view_name, left_date, right_date), group in anchors.groupby(["view_name", "left_date", "right_date"], sort=True):
        values = group["distance"].dropna().astype(float)
        rows.append(
            {
                "view_name": view_name,
                "left_date": left_date,
                "right_date": right_date,
                "date_pair": f"{left_date}|{right_date}",
                "n_odor_anchors": int(group["left_stimulus"].nunique()),
                "odor_anchors": ";".join(sorted(group["left_stim_name"].astype(str).unique())),
                "distance_mean": float(values.mean()) if len(values) else np.nan,
                "distance_median": float(values.median()) if len(values) else np.nan,
                "distance_max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_date_anchors(pairwise: pd.DataFrame) -> pd.DataFrame:
    anchors = pairwise.loc[pairwise["pair_category"].eq("different_date_same_odor")].copy()
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
                "n_cross_date_same_odor_pairs": int(len(values)),
                "n_other_dates": int(group["other_date"].nunique()),
                "n_odor_anchors": int(group["stim_name"].nunique()),
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
        matched = group.loc[group["pair_category"].eq("different_date_same_odor")].copy()
        mismatched = group.loc[group["pair_category"].eq("different_date_different_odor")].copy()
        odor_order = sorted(set(matched["left_stim_name"].astype(str)) | set(matched["right_stim_name"].astype(str)))
        for anchor_odor in odor_order:
            matched_rows = matched.loc[
                matched["left_stim_name"].astype(str).eq(anchor_odor) & matched["right_stim_name"].astype(str).eq(anchor_odor)
            ]
            for _, row in matched_rows.iterrows():
                rows.append(
                    {
                        "view_name": view_name,
                        "left_date": left_date,
                        "right_date": right_date,
                        "date_pair": f"{left_date}|{right_date}",
                        "anchor_odor": anchor_odor,
                        "contrast": "same",
                        "anchor_side": "both",
                        "other_odor": anchor_odor,
                        "distance": row["distance"],
                    }
                )

            mismatched_rows = mismatched.loc[
                mismatched["left_stim_name"].astype(str).eq(anchor_odor)
                | mismatched["right_stim_name"].astype(str).eq(anchor_odor)
            ]
            for _, row in mismatched_rows.iterrows():
                anchor_on_left = bool(str(row["left_stim_name"]) == anchor_odor)
                rows.append(
                    {
                        "view_name": view_name,
                        "left_date": left_date,
                        "right_date": right_date,
                        "date_pair": f"{left_date}|{right_date}",
                        "anchor_odor": anchor_odor,
                        "contrast": "different",
                        "anchor_side": "left_date" if anchor_on_left else "right_date",
                        "other_odor": row["right_stim_name"] if anchor_on_left else row["left_stim_name"],
                        "distance": row["distance"],
                    }
                )

    return pd.DataFrame(rows)


def build_anchor_identity_contrasts(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    same_odor = pairwise.loc[pairwise["pair_category"].eq("different_date_same_odor")].copy()
    for _, row in same_odor.iterrows():
        rows.append(
            {
                "view_name": row["view_name"],
                "anchor_odor": row["left_stim_name"],
                "contrast": "same odor across dates",
                "date_context": f"{row['left_date']}|{row['right_date']}",
                "distance": row["distance"],
            }
        )

    same_date_different = pairwise.loc[pairwise["pair_category"].eq("same_date_different_odor")].copy()
    for _, row in same_date_different.iterrows():
        rows.append(
            {
                "view_name": row["view_name"],
                "anchor_odor": row["left_stim_name"],
                "contrast": "different odor same date",
                "date_context": str(row["left_date"]),
                "distance": row["distance"],
            }
        )
        rows.append(
            {
                "view_name": row["view_name"],
                "anchor_odor": row["right_stim_name"],
                "contrast": "different odor same date",
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
    ax.set_xlabel("Clustered date / base odor labels", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Prototype distance", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_base_odor_date_mds(
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

    odor_order = sorted(coords["stim_name"].astype(str).unique())
    odor_colors = {
        odor: color
        for odor, color in zip(
            odor_order,
            ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"],
            strict=False,
        )
    }
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    date_markers = {date: marker_cycle[idx % len(marker_cycle)] for idx, date in enumerate(ordered_dates)}

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for odor_name in odor_order:
        odor_points = coords.loc[coords["stim_name"].astype(str).eq(odor_name)].copy()
        odor_points["date_sort"] = odor_points["date"].astype(str).map({date: idx for idx, date in enumerate(ordered_dates)})
        odor_points = odor_points.sort_values("date_sort")
        ax.plot(
            odor_points["x"].to_numpy(float),
            odor_points["y"].to_numpy(float),
            color=odor_colors[odor_name],
            linewidth=1.4,
            alpha=0.65,
            zorder=1,
        )

    for date in ordered_dates:
        date_points = coords.loc[coords["date"].astype(str).eq(date)].copy()
        if date_points.empty:
            continue
        colors = [odor_colors[odor] for odor in date_points["stim_name"].astype(str)]
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

    odor_handles = [
        plt.Line2D([0], [0], color=odor_colors[odor], linewidth=2.0, label=odor) for odor in odor_order
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
    odor_legend = ax.legend(
        handles=odor_handles,
        title="base odor",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(odor_legend)
    ax.legend(
        handles=date_handles,
        title="date",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_title("Base-odor prototypes across dates", fontsize=12)
    ax.set_xlabel("MDS axis 1", fontsize=9)
    ax.set_ylabel("MDS axis 2", fontsize=9)
    ax.grid(color="#E3E3E3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")
    fig.text(
        0.5,
        0.02,
        "Distances are correlation-distance relationships in 2D; lines connect the same odor across dates.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.86, 0.98))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_odor_resolved_date_pair_heatmap(
    pairwise: pd.DataFrame,
    path: Path,
    *,
    view_name: str,
    date_order: list[str] | None = None,
) -> None:
    view = pairwise.loc[
        pairwise["view_name"].astype(str).eq(view_name)
        & pairwise["same_odor"].astype(bool)
        & (~pairwise["same_date"].astype(bool))
    ].copy()
    if view.empty:
        raise ValueError(f"No same-odor cross-date rows available for view {view_name!r}")

    odor_order = sorted(view["left_stim_name"].astype(str).unique())
    if date_order is None:
        dates = sorted(set(view["left_date"].astype(str)) | set(view["right_date"].astype(str)))
    else:
        dates = [str(date) for date in date_order]

    matrices: dict[str, pd.DataFrame] = {}
    finite_values: list[float] = []
    for odor_name in odor_order:
        odor_view = view.loc[view["left_stim_name"].astype(str).eq(odor_name)].copy()
        matrix = pd.DataFrame(np.nan, index=dates, columns=dates, dtype=float)
        for _, row in odor_view.iterrows():
            left_date = str(row["left_date"])
            right_date = str(row["right_date"])
            if left_date not in matrix.index or right_date not in matrix.columns:
                continue
            value = float(row["distance"])
            matrix.loc[left_date, right_date] = value
            matrix.loc[right_date, left_date] = value
            finite_values.append(value)
        matrices[odor_name] = matrix

    if not finite_values:
        raise ValueError(f"No valid same-odor cross-date distances available for view {view_name!r}")

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#F3F1EE")
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    threshold = float(np.nanmedian(finite_values))

    fig, axes = plt.subplots(1, len(odor_order), figsize=(4.2 * len(odor_order), 4.8), sharex=True, sharey=True)
    if len(odor_order) == 1:
        axes = [axes]

    last_image = None
    for ax, odor_name in zip(axes, odor_order):
        display = matrices[odor_name].copy()
        image = ax.imshow(display.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        last_image = image
        ax.set_xticks(range(len(dates)))
        ax.set_yticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(dates, fontsize=8)
        ax.set_title(odor_name, fontsize=11)
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
    fig.suptitle("Base-odor cross-date distance by anchor odor", fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.18, top=0.82, wspace=0.10)
    colorbar_axis = fig.add_axes([0.905, 0.20, 0.015, 0.60])
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Raw matched distance\nLower = less drift", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_base_odor_same_vs_other_distributions(
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

    odor_order = sorted(view["anchor_odor"].astype(str).unique())
    category_order = ["same", "different"]
    category_colors = {"same": "#0072B2", "different": "#8F8F8F"}
    category_labels = {"same": "same odor", "different": "other odors"}

    values = view["distance"].to_numpy(float)
    y_min = max(0.0, float(np.nanmin(values)) - 0.08)
    y_max = float(np.nanmax(values)) + 0.08

    fig, axes = plt.subplots(1, len(odor_order), figsize=(4.2 * len(odor_order), 4.8), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(20260423)

    for ax, odor_name in zip(axes, odor_order):
        odor_view = view.loc[view["anchor_odor"].astype(str).eq(odor_name)].copy()
        box_data: list[np.ndarray] = []
        box_positions: list[float] = []
        box_categories: list[str] = []
        tick_labels: list[str] = []

        for position, category in enumerate(category_order):
            category_values = odor_view.loc[odor_view["contrast"].astype(str).eq(category), "distance"].to_numpy(float)
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
            category_values = odor_view.loc[odor_view["contrast"].astype(str).eq(category), "distance"].to_numpy(float)
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

        ax.set_title(odor_name, fontsize=11)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("correlation distance (1 - r)", fontsize=10)
    fig.suptitle("Base-odor cross-date distances", fontsize=13, y=0.98)
    fig.text(
        0.5,
        0.02,
        "same odor = the odor compared to itself across dates; other odors = the odor compared to different odors across dates.",
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
        ax.text(0.5, 0.5, "No base-odor anchor contrasts available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return

    odor_order = sorted(contrast["anchor_odor"].astype(str).unique())
    contrast_order = ["same odor across dates", "different odor same date"]
    contrast_labels = {
        "same odor across dates": "same odor\ncross date",
        "different odor same date": "different odor\nsame date",
    }
    colors = {
        "same odor across dates": "#0072B2",
        "different odor same date": "#D55E00",
    }
    offsets = {
        "same odor across dates": -0.18,
        "different odor same date": 0.18,
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
        for odor_index, odor in enumerate(odor_order):
            odor_rows = view.loc[view["anchor_odor"].astype(str).eq(odor)]
            for contrast_name in contrast_order:
                rows = odor_rows.loc[odor_rows["contrast"].eq(contrast_name)]
                values = pd.to_numeric(rows["distance"], errors="coerce").dropna().to_numpy(float)
                if values.size == 0:
                    continue
                x = odor_index + offsets[contrast_name] + rng.normal(0.0, 0.025, size=values.size)
                ax.scatter(x, values, s=18, alpha=0.55, color=colors[contrast_name], linewidths=0)
                median = float(np.median(values))
                ax.plot(
                    [odor_index + offsets[contrast_name] - 0.10, odor_index + offsets[contrast_name] + 0.10],
                    [median, median],
                    color="#222222",
                    linewidth=2.0,
                )

        ax.set_title(view_label(view_name))
        ax.set_xticks(range(len(odor_order)))
        ax.set_xticklabels(odor_order, rotation=25, ha="right")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[name], markersize=6)
            for name in contrast_order
        ]
        ax.legend(handles, [contrast_labels[name].replace("\n", " ") for name in contrast_order], frameon=False, fontsize=8)

    axes[0].set_ylabel("correlation distance")
    fig.suptitle("Base-odor identity across dates", fontsize=12)
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
    order = ["different_date_same_odor", "same_date_different_odor", "different_date_different_odor"]
    labels = {
        "different_date_same_odor": "same odor\ncross date",
        "same_date_different_odor": "same date\ndifferent odor",
        "different_date_different_odor": "different odor\ncross date",
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
    metrics = ["odor_ideal_spearman", "date_ideal_spearman", "odor_ideal_partial_r", "date_ideal_partial_r"]
    labels = ["odor", "date", "odor partial", "date partial"]
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
    odor_anchor_summary: pd.DataFrame,
    date_anchor_summary: pd.DataFrame,
    *,
    min_date_exclusive: str | None,
    keep_dates: list[str] | None,
) -> None:
    odor_names = ", ".join(f"`{name}`" for name in sorted(coverage["stim_name"].astype(str).unique()))
    lines = [
        "# Base-Odor Date-Effect Review",
        "",
        f"- Input: `{input_path.as_posix()}`",
        f"- Base odors: {odor_names}.",
        "- Neural representation: baseline-centered, non-ASE L/R merged, trial-median prototypes.",
        "- Purpose: use repeated odors as anchors to estimate date effects that could not be separated from disjoint bacteria panels.",
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
            "## Cross-Date Same-Odor Anchors",
            "",
            markdown_table(odor_anchor_summary),
            "",
            "## Per-Date Same-Odor Cross-Date Anchors",
            "",
            markdown_table(date_anchor_summary),
            "",
            "## Output Files",
            "",
            "- `tables/base_odor_coverage.csv`",
            "- `tables/base_odor_ideal_model_similarity.csv`",
            "- `tables/base_odor_cross_date_anchor_summary.csv`",
            "- `tables/base_odor_same_odor_date_pair_summary.csv`",
            "- `tables/base_odor_date_anchor_summary.csv`",
            "- `tables/base_odor_date_pair_same_vs_other_contrasts.csv`",
            "- `figures/base_odor_date_prototype_rdms.png`",
            "- `figures/base_odor_date_prototype_rdms__neural_clustered_stim_name.png`",
            "- `figures/base_odor_date_mds__response_window.png`",
            "- `figures/base_odor_date_pair_heatmap_by_odor__response_window.png`",
            "- `figures/base_odor_same_vs_other_odors__response_window.png`",
            "- `figures/base_odor_ideal_model_similarity.png`",
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
