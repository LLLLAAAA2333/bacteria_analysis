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
    run(args.input_path, args.output_root)


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
    return parser.parse_args()


def run(input_path: Path, output_root: Path) -> None:
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
    base["trial_id"] = base["date"] + "__" + base["worm_key"].astype(str) + "__" + base["segment_index"].astype(str)

    coverage = build_coverage(base)
    trial_features = build_trial_features(base)
    prototypes = build_prototypes(trial_features)
    pairwise = build_pairwise_prototype_distances(prototypes)
    model_summary = summarize_ideal_models(pairwise)
    distance_summary = summarize_distance_categories(pairwise)
    odor_anchor_summary = summarize_odor_anchors(pairwise)
    date_pair_anchor_summary = summarize_date_pair_anchors(pairwise)
    date_anchor_summary = summarize_date_anchors(pairwise)

    coverage.to_csv(tables / "base_odor_coverage.csv", index=False)
    trial_features.to_parquet(tables / "base_odor_trial_features.parquet", index=False)
    prototypes.to_parquet(tables / "base_odor_date_prototypes.parquet", index=False)
    pairwise.to_csv(tables / "base_odor_pairwise_prototype_distances.csv", index=False)
    model_summary.to_csv(tables / "base_odor_ideal_model_similarity.csv", index=False)
    distance_summary.to_csv(tables / "base_odor_distance_category_summary.csv", index=False)
    odor_anchor_summary.to_csv(tables / "base_odor_cross_date_anchor_summary.csv", index=False)
    date_pair_anchor_summary.to_csv(tables / "base_odor_same_odor_date_pair_summary.csv", index=False)
    date_anchor_summary.to_csv(tables / "base_odor_date_anchor_summary.csv", index=False)

    plot_rdm_heatmaps(pairwise, figures / "base_odor_date_prototype_rdms.png")
    plot_clustered_rdm_heatmaps(pairwise, figures / "base_odor_date_prototype_rdms__neural_clustered_stim_name.png")
    plot_date_pair_anchor_heatmaps(date_pair_anchor_summary, figures / "base_odor_same_odor_date_pair_heatmaps.png")
    plot_distance_categories(distance_summary, figures / "base_odor_distance_categories.png")
    plot_ideal_models(model_summary, figures / "base_odor_ideal_model_similarity.png")
    write_summary(
        input_path,
        output_root,
        coverage,
        model_summary,
        distance_summary,
        odor_anchor_summary,
        date_anchor_summary,
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


def plot_clustered_rdm_heatmaps(pairwise: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    for ax, (view_name, view) in zip(axes, pairwise.groupby("view_name", sort=True)):
        labels = sorted(set(view["left_label"]) | set(view["right_label"]))
        matrix = build_distance_matrix(view, labels)
        ordered_labels = clustered_order(matrix)
        ordered = matrix.loc[ordered_labels, ordered_labels]
        image = ax.imshow(ordered.to_numpy(float), cmap="magma", interpolation="nearest")
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_yticks(range(len(ordered_labels)))
        ax.set_xticklabels([stim_name_label(label) for label in ordered_labels], rotation=90, fontsize=7)
        ax.set_yticklabels([stim_name_label(label) for label in ordered_labels], fontsize=7)
        ax.set_xlabel(view_label(view_name))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_date_pair_anchor_heatmaps(summary: pd.DataFrame, path: Path) -> None:
    dates = sorted(set(summary["left_date"].astype(str)) | set(summary["right_date"].astype(str)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    for ax, (view_name, view) in zip(axes, summary.groupby("view_name", sort=True)):
        matrix = pd.DataFrame(0.0, index=dates, columns=dates, dtype=float)
        for _, row in view.iterrows():
            matrix.loc[row["left_date"], row["right_date"]] = row["distance_mean"]
            matrix.loc[row["right_date"], row["left_date"]] = row["distance_mean"]
        image = ax.imshow(matrix.to_numpy(float), cmap="magma", interpolation="nearest")
        ax.set_xticks(range(len(dates)))
        ax.set_yticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=90, fontsize=7)
        ax.set_yticklabels(dates, fontsize=7)
        ax.set_xlabel(view_label(view_name))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_distance_matrix(view: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    np.fill_diagonal(matrix.values, 0.0)
    for _, row in view.iterrows():
        matrix.loc[row["left_label"], row["right_label"]] = row["distance"]
        matrix.loc[row["right_label"], row["left_label"]] = row["distance"]
    return matrix


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
    distance_summary: pd.DataFrame,
    odor_anchor_summary: pd.DataFrame,
    date_anchor_summary: pd.DataFrame,
) -> None:
    odor_names = ", ".join(f"`{name}`" for name in sorted(coverage["stim_name"].astype(str).unique()))
    lines = [
        "# Base-Odor Date-Effect Review",
        "",
        f"- Input: `{input_path.as_posix()}`",
        f"- Base odors: {odor_names}.",
        "- Neural representation: baseline-centered, non-ASE L/R merged, trial-median prototypes.",
        "- Purpose: use repeated odors as anchors to estimate date effects that could not be separated from disjoint bacteria panels.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Ideal Model Similarity",
        "",
        markdown_table(model_summary),
        "",
        "## Distance Categories",
        "",
        markdown_table(distance_summary),
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
        "- `tables/base_odor_distance_category_summary.csv`",
        "- `tables/base_odor_cross_date_anchor_summary.csv`",
        "- `tables/base_odor_same_odor_date_pair_summary.csv`",
        "- `tables/base_odor_date_anchor_summary.csv`",
        "- `figures/base_odor_date_prototype_rdms.png`",
        "- `figures/base_odor_date_prototype_rdms__neural_clustered_stim_name.png`",
        "- `figures/base_odor_same_odor_date_pair_heatmaps.png`",
        "- `figures/base_odor_distance_categories.png`",
        "- `figures/base_odor_ideal_model_similarity.png`",
    ]
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
