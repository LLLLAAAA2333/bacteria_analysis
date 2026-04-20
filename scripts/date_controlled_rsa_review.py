"""Standalone date-controlled RSA review for the filtered 202604 batch."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("results/202604_without_20260331")
OUT = BASE / "date_controlled_rsa_review"
N_ITER = 2000
SEED = 20260420


@dataclass(frozen=True)
class RdmSpec:
    id: str
    label: str
    path: Path


NEURAL = [
    RdmSpec(
        "response_window_median_flattened",
        "response window median",
        BASE / "neural_median_peak_review_lr_merge" / "neural_rdm__response_window__median_flattened.parquet",
    ),
    RdmSpec(
        "full_trajectory_median_flattened",
        "full trajectory median",
        BASE / "neural_median_peak_review_lr_merge" / "neural_rdm__full_trajectory__median_flattened.parquet",
    ),
]

MODELS = [
    RdmSpec(
        "global_qc20_log2_euclidean",
        "QC20 log2 Euclidean",
        BASE / "neural_common_response_residualization_review" / "model_rdm__Global_qc20_log2_euclidean.parquet",
    ),
    RdmSpec(
        "global_profile_default_correlation",
        "global profile correlation",
        BASE / "rsa" / "tables" / "model_rdm__global_profile.parquet",
    ),
    RdmSpec(
        "best_weighted_fusion_fixed_weights",
        "best weighted fusion",
        BASE / "neural_common_response_residualization_review" / "model_rdm__best_weighted_fusion_fixed_weights.parquet",
    ),
]

RED_BOX_TABLES = {
    "global_qc20_log2_euclidean": BASE
    / "joint_consensus_rdm_review"
    / "tables"
    / "shared_low_blocks__neural_rdm_median_flattened__model_global_qc20_log2_euclidean__minimal.csv",
    "global_profile_default_correlation": BASE
    / "joint_consensus_rdm_review"
    / "tables"
    / "shared_low_blocks__neural_rdm_median_flattened__model_global_profile_default_correlation__minimal.csv",
}


def main() -> None:
    tables = OUT / "tables"
    figures = OUT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    date_map = load_date_map()
    neural_rdms = {spec.id: load_rdm(spec.path) for spec in NEURAL}
    model_rdms = {spec.id: load_rdm(spec.path) for spec in MODELS}

    pair_frames = []
    scope_rows = []
    per_date_rows = []
    date_pair_rows = []
    perm_summary_rows = []
    perm_null_rows = []
    rng = np.random.default_rng(SEED)

    for neural in NEURAL:
        for model in MODELS:
            pairs = build_pairs(neural_rdms[neural.id], model_rdms[model.id], date_map)
            pairs.insert(0, "model_id", model.id)
            pairs.insert(0, "neural_rdm", neural.id)
            pair_frames.append(pairs)
            scope_rows.extend(summarize_scopes(pairs, neural, model))
            per_date_rows.extend(summarize_per_date(pairs, neural, model))
            date_pair_rows.extend(summarize_date_pairs(pairs, neural, model))
            summary, null = date_preserving_null(
                pairs,
                model_rdms[model.id],
                date_map,
                neural=neural,
                model=model,
                rng=rng,
                n_iter=N_ITER,
            )
            perm_summary_rows.extend(summary)
            perm_null_rows.extend(null)

    pair_level = pd.concat(pair_frames, ignore_index=True)
    scope_summary = pd.DataFrame(scope_rows)
    per_date = pd.DataFrame(per_date_rows)
    date_pair_summary = pd.DataFrame(date_pair_rows)
    perm_summary = pd.DataFrame(perm_summary_rows)
    perm_null = pd.DataFrame(perm_null_rows)
    red_summary, red_null = red_box_date_matched_null(
        date_map,
        neural_rdms,
        model_rdms,
        rng=np.random.default_rng(SEED + 1),
        n_iter=N_ITER,
    )
    date_counts = (
        pd.Series(date_map, name="date")
        .rename_axis("stimulus")
        .reset_index()
        .groupby("date", as_index=False)
        .agg(n_stimuli=("stimulus", "size"), stimuli=("stimulus", lambda x: ";".join(sorted(x))))
    )

    pair_level.to_csv(tables / "pair_level_distances_with_date.csv", index=False)
    scope_summary.to_csv(tables / "rsa_all_vs_within_date_vs_cross_date.csv", index=False)
    per_date.to_csv(tables / "rsa_per_date.csv", index=False)
    date_pair_summary.to_csv(tables / "rsa_by_date_pair.csv", index=False)
    perm_summary.to_csv(tables / "date_preserving_permutation_summary.csv", index=False)
    perm_null.to_csv(tables / "date_preserving_permutation_null.csv", index=False)
    red_summary.to_csv(tables / "red_box_date_matched_null_summary.csv", index=False)
    red_null.to_csv(tables / "red_box_date_matched_null.csv", index=False)
    date_counts.to_csv(tables / "stimulus_counts_by_date.csv", index=False)

    plot_scope(scope_summary, figures / "rsa_scope_comparison.png")
    plot_per_date(per_date, figures / "rsa_per_date.png")
    plot_perm(perm_summary, figures / "date_preserving_permutation_summary.png")
    plot_red(red_summary, figures / "red_box_date_matched_null_summary.png")
    write_summary(scope_summary, perm_summary, red_summary, date_counts)


def load_date_map() -> dict[str, str]:
    path = BASE / "rsa" / "qc" / "aggregated_response_support__per_date.parquet"
    support = pd.read_parquet(path)[["date", "stimulus"]].drop_duplicates()
    counts = support.groupby("stimulus")["date"].nunique()
    repeated = counts.loc[counts > 1]
    if not repeated.empty:
        raise ValueError(f"Expected one date per stimulus, got repeated labels: {repeated.index.tolist()}")
    return dict(zip(support["stimulus"].astype(str), support["date"].astype(str)))


def load_rdm(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "stimulus_row" not in frame.columns:
        raise ValueError(f"{path} is missing stimulus_row")
    matrix = frame.set_index("stimulus_row")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


def build_pairs(neural: pd.DataFrame, model: pd.DataFrame, date_map: dict[str, str]) -> pd.DataFrame:
    labels = [label for label in neural.index.astype(str) if label in model.index and label in date_map]
    neural = neural.loc[labels, labels]
    model = model.loc[labels, labels]
    rows = []
    for i, j in combinations(range(len(labels)), 2):
        left, right = labels[i], labels[j]
        date_left, date_right = date_map[left], date_map[right]
        date_pair = "|".join(sorted([date_left, date_right]))
        rows.append(
            {
                "stimulus_left": left,
                "stimulus_right": right,
                "date_left": date_left,
                "date_right": date_right,
                "date_pair": date_pair,
                "date_pair_type": "within_date" if date_left == date_right else "cross_date",
                "neural_value": float(neural.iat[i, j]),
                "model_value": float(model.iat[i, j]),
            }
        )
    pairs = pd.DataFrame(rows)
    pairs["neural_global_rank_pct"] = pct_rank(pairs["neural_value"].to_numpy(float))
    pairs["model_global_rank_pct"] = pct_rank(pairs["model_value"].to_numpy(float))
    pairs["neural_date_pair_rank_pct"] = stratified_pct_rank(pairs["neural_value"], pairs["date_pair"])
    pairs["model_date_pair_rank_pct"] = stratified_pct_rank(pairs["model_value"], pairs["date_pair"])
    return pairs


def summarize_scopes(pairs: pd.DataFrame, neural: RdmSpec, model: RdmSpec) -> list[dict[str, object]]:
    within = pairs["date_pair_type"].eq("within_date")
    scopes = [
        ("all_pairs", pairs, "global_spearman"),
        ("within_date_only", pairs.loc[within], "global_spearman"),
        ("cross_date_only", pairs.loc[~within], "global_spearman"),
        ("date_pair_stratified_rank_all", pairs, "date_pair_stratified_rank"),
        ("date_pair_stratified_rank_within", pairs.loc[within], "date_pair_stratified_rank"),
        ("date_pair_stratified_rank_cross", pairs.loc[~within], "date_pair_stratified_rank"),
    ]
    return [score_row(neural, model, scope, method, subset) for scope, subset, method in scopes]


def summarize_per_date(pairs: pd.DataFrame, neural: RdmSpec, model: RdmSpec) -> list[dict[str, object]]:
    rows = []
    within = pairs.loc[pairs["date_pair_type"].eq("within_date")]
    for date, subset in within.groupby("date_left", sort=True):
        row = score_row(neural, model, "per_date_within", "global_spearman", subset)
        row["date"] = date
        rows.append(row)
    return rows


def summarize_date_pairs(pairs: pd.DataFrame, neural: RdmSpec, model: RdmSpec) -> list[dict[str, object]]:
    rows = []
    for date_pair, subset in pairs.groupby("date_pair", sort=True):
        row = score_row(neural, model, "date_pair", "global_spearman", subset)
        row.update(
            {
                "date_pair": date_pair,
                "date_pair_type": subset["date_pair_type"].iloc[0],
                "neural_mean": float(subset["neural_value"].mean()),
                "model_mean": float(subset["model_value"].mean()),
                "neural_global_rank_pct_mean": float(subset["neural_global_rank_pct"].mean()),
                "model_global_rank_pct_mean": float(subset["model_global_rank_pct"].mean()),
            }
        )
        rows.append(row)
    return rows


def score_row(neural: RdmSpec, model: RdmSpec, scope: str, method: str, subset: pd.DataFrame) -> dict[str, object]:
    if method == "date_pair_stratified_rank":
        rsa = pearson(subset["neural_date_pair_rank_pct"], subset["model_date_pair_rank_pct"])
    else:
        rsa = spearman(subset["neural_value"], subset["model_value"])
    return {
        "neural_rdm": neural.id,
        "neural_label": neural.label,
        "model_id": model.id,
        "model_label": model.label,
        "scope": scope,
        "score_method": method,
        "n_pairs": int(len(subset)),
        "rsa_similarity": rsa,
        "score_status": "ok" if np.isfinite(rsa) else "invalid",
    }


def date_preserving_null(
    pairs: pd.DataFrame,
    model_rdm: pd.DataFrame,
    date_map: dict[str, str],
    *,
    neural: RdmSpec,
    model: RdmSpec,
    rng: np.random.Generator,
    n_iter: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = sorted(pd.Index(pairs["stimulus_left"]).append(pd.Index(pairs["stimulus_right"])).drop_duplicates())
    label_to_index = {label: i for i, label in enumerate(labels)}
    model_values = model_rdm.loc[labels, labels].to_numpy(float)
    by_date = {
        date: np.array([label_to_index[label] for label in labels if date_map[label] == date])
        for date in sorted(set(date_map.values()))
    }
    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    pair_index = pd.MultiIndex.from_arrays([[labels[i] for i in upper_i], [labels[j] for j in upper_j]])
    ordered = pairs.set_index(["stimulus_left", "stimulus_right"]).loc[pair_index].reset_index()
    neural_values = ordered["neural_value"].to_numpy(float)
    original_model = ordered["model_value"].to_numpy(float)
    date_pairs = ordered["date_pair"].astype(str).to_numpy()
    scopes = mask_specs(ordered)

    observed = {
        scope: score_values(neural_values[mask], original_model[mask], date_pairs[mask], method)
        for scope, mask, method in scopes
    }
    null_by_scope = {scope: [] for scope, _, _ in scopes}

    for _ in range(n_iter):
        permuted = np.arange(len(labels))
        for indices in by_date.values():
            permuted[indices] = rng.permutation(indices)
        permuted_model = model_values[np.ix_(permuted, permuted)]
        permuted_upper = permuted_model[upper_i, upper_j]
        for scope, mask, method in scopes:
            null_by_scope[scope].append(score_values(neural_values[mask], permuted_upper[mask], date_pairs[mask], method))

    summary_rows = []
    null_rows = []
    for scope, _, method in scopes:
        values = np.asarray(null_by_scope[scope], dtype=float)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "neural_rdm": neural.id,
                "model_id": model.id,
                "scope": scope,
                "score_method": method,
                "observed_rsa_similarity": observed[scope],
                "n_iterations": n_iter,
                "null_mean": nan_mean(finite),
                "null_sd": nan_sd(finite),
                "null_q025": nan_quantile(finite, 0.025),
                "null_q50": nan_quantile(finite, 0.5),
                "null_q975": nan_quantile(finite, 0.975),
                "p_value_one_sided_ge": empirical_p(observed[scope], finite),
                "n_valid_null": int(finite.size),
            }
        )
        null_rows.extend(
            {
                "neural_rdm": neural.id,
                "model_id": model.id,
                "scope": scope,
                "iteration": iteration,
                "rsa_similarity": value,
            }
            for iteration, value in enumerate(values)
        )
    return summary_rows, null_rows


def mask_specs(pairs: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    within = pairs["date_pair_type"].eq("within_date").to_numpy()
    cross = ~within
    all_pairs = np.ones(len(pairs), dtype=bool)
    return [
        ("all_pairs", all_pairs, "global_spearman"),
        ("within_date_only", within, "global_spearman"),
        ("cross_date_only", cross, "global_spearman"),
        ("date_pair_stratified_rank_all", all_pairs, "date_pair_stratified_rank"),
        ("date_pair_stratified_rank_within", within, "date_pair_stratified_rank"),
        ("date_pair_stratified_rank_cross", cross, "date_pair_stratified_rank"),
    ]


def score_values(neural_values: np.ndarray, model_values: np.ndarray, date_pairs: np.ndarray, method: str) -> float:
    if method == "date_pair_stratified_rank":
        return pearson(stratified_pct_rank_array(neural_values, date_pairs), stratified_pct_rank_array(model_values, date_pairs))
    return spearman(neural_values, model_values)


def red_box_date_matched_null(
    date_map: dict[str, str],
    neural_rdms: dict[str, pd.DataFrame],
    model_rdms: dict[str, pd.DataFrame],
    *,
    rng: np.random.Generator,
    n_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_by_date = {
        date: sorted([label for label, label_date in date_map.items() if label_date == date])
        for date in sorted(set(date_map.values()))
    }
    summary_rows = []
    null_rows = []
    for model_id, red_path in RED_BOX_TABLES.items():
        if not red_path.exists():
            continue
        red = pd.read_csv(red_path)
        for neural in NEURAL:
            view = neural.id.replace("_median_flattened", "")
            blocks = red.loc[red["view"].eq(view)]
            if blocks.empty:
                continue
            selected = sorted(
                {
                    stimulus
                    for value in blocks["stimuli"].astype(str)
                    for stimulus in value.split(";")
                    if stimulus
                }
            )
            pairs = build_pairs(neural_rdms[neural.id], model_rdms[model_id], date_map)
            inside_obs = score_sample_set(pairs, selected)
            outside_obs = score_sample_set(pairs, sorted(set(date_map) - set(selected)))
            delta_obs = inside_obs - outside_obs
            counts = {date: sum(date_map[label] == date for label in selected) for date in labels_by_date}
            inside_null, outside_null, delta_null = [], [], []
            for iteration in range(n_iter):
                sampled = []
                for date, count in counts.items():
                    if count:
                        sampled.extend(rng.choice(labels_by_date[date], size=count, replace=False).tolist())
                sampled = sorted(sampled)
                outside = sorted(set(date_map) - set(sampled))
                inside_score = score_sample_set(pairs, sampled)
                outside_score = score_sample_set(pairs, outside)
                delta_score = inside_score - outside_score
                inside_null.append(inside_score)
                outside_null.append(outside_score)
                delta_null.append(delta_score)
                null_rows.append(
                    {
                        "neural_rdm": neural.id,
                        "model_id": model_id,
                        "iteration": iteration,
                        "inside_rsa_similarity": inside_score,
                        "outside_rsa_similarity": outside_score,
                        "delta_inside_minus_outside": delta_score,
                    }
                )
            summary_rows.append(
                {
                    "neural_rdm": neural.id,
                    "model_id": model_id,
                    "n_selected_samples": len(selected),
                    "selected_counts_by_date": ";".join(f"{date}={count}" for date, count in counts.items()),
                    "observed_inside_rsa": inside_obs,
                    "observed_outside_rsa": outside_obs,
                    "observed_delta_inside_minus_outside": delta_obs,
                    "n_iterations": n_iter,
                    **null_fields("inside", inside_null, inside_obs),
                    **null_fields("outside", outside_null, outside_obs),
                    **null_fields("delta", delta_null, delta_obs),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


def score_sample_set(pairs: pd.DataFrame, selected: list[str]) -> float:
    selected_set = set(selected)
    subset = pairs.loc[pairs["stimulus_left"].isin(selected_set) & pairs["stimulus_right"].isin(selected_set)]
    return spearman(subset["neural_value"], subset["model_value"])


def null_fields(prefix: str, values: list[float], observed: float) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        f"null_{prefix}_mean": nan_mean(finite),
        f"null_{prefix}_sd": nan_sd(finite),
        f"null_{prefix}_q025": nan_quantile(finite, 0.025),
        f"null_{prefix}_q50": nan_quantile(finite, 0.5),
        f"null_{prefix}_q975": nan_quantile(finite, 0.975),
        f"p_value_{prefix}_one_sided_ge": empirical_p(observed, finite),
        f"n_valid_null_{prefix}": int(finite.size),
    }


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return np.nan
    return pearson(avg_rank(left[mask]), avg_rank(right[mask]))


def pearson(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return np.nan
    left, right = left[mask], right[mask]
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def avg_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, copy=False).rank(method="average").to_numpy(float)


def pct_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        result[finite] = (avg_rank(values[finite]) - 0.5) / finite.sum()
    return result


def stratified_pct_rank(values: pd.Series, groups: pd.Series) -> np.ndarray:
    return stratified_pct_rank_array(values.to_numpy(float), groups.astype(str).to_numpy())


def stratified_pct_rank_array(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=object)
    result = np.full(values.shape, np.nan, dtype=float)
    for group in sorted(pd.unique(groups)):
        mask = groups == group
        result[mask] = pct_rank(values[mask])
    return result


def empirical_p(observed: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        return np.nan
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def nan_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def nan_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def nan_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else np.nan


def plot_scope(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[
        summary["scope"].isin(["all_pairs", "within_date_only", "cross_date_only", "date_pair_stratified_rank_all"])
    ]
    scopes = ["all_pairs", "within_date_only", "cross_date_only", "date_pair_stratified_rank_all"]
    scope_labels = ["all", "within", "cross", "stratified"]
    colors = model_colors()
    fig, axes = plt.subplots(1, len(NEURAL), figsize=(13.0, 4.3), sharey=True)
    if len(NEURAL) == 1:
        axes = [axes]
    x = np.arange(len(scopes))
    offsets = np.linspace(-0.22, 0.22, len(MODELS))
    for ax, neural in zip(axes, NEURAL):
        view = selected.loc[selected["neural_rdm"].eq(neural.id)]
        for offset, model in zip(offsets, MODELS):
            model_rows = view.loc[view["model_id"].eq(model.id)].set_index("scope")
            y = [model_rows.loc[scope, "rsa_similarity"] if scope in model_rows.index else np.nan for scope in scopes]
            ax.plot(x + offset, y, marker="o", linewidth=1.4, color=colors[model.id], label=model.id)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scope_labels)
        ax.set_xlabel(neural.id.replace("_median_flattened", ""))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("RSA similarity")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_per_date(per_date: pd.DataFrame, path: Path) -> None:
    colors = model_colors()
    dates = sorted(per_date["date"].astype(str).unique())
    fig, axes = plt.subplots(1, len(NEURAL), figsize=(13.0, 4.3), sharey=True)
    if len(NEURAL) == 1:
        axes = [axes]
    x = np.arange(len(dates))
    offsets = np.linspace(-0.2, 0.2, len(MODELS))
    for ax, neural in zip(axes, NEURAL):
        view = per_date.loc[per_date["neural_rdm"].eq(neural.id)]
        for offset, model in zip(offsets, MODELS):
            model_rows = view.loc[view["model_id"].eq(model.id)].set_index("date")
            y = [model_rows.loc[date, "rsa_similarity"] if date in model_rows.index else np.nan for date in dates]
            ax.plot(x + offset, y, marker="o", linewidth=1.4, color=colors[model.id], label=model.id)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=35, ha="right")
        ax.set_xlabel(neural.id.replace("_median_flattened", ""))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("within-date RSA")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_perm(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[summary["scope"].isin(["all_pairs", "date_pair_stratified_rank_all"])].copy()
    selected["combo"] = selected["neural_rdm"].str.replace("_median_flattened", "", regex=False) + "\n" + selected["model_id"]
    selected = selected.sort_values(["scope", "neural_rdm", "model_id"])
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharey=True)
    for ax, (scope, rows) in zip(axes, selected.groupby("scope", sort=False)):
        y = np.arange(len(rows))
        ax.hlines(y, rows["null_q025"], rows["null_q975"], color="#999999", linewidth=2)
        ax.scatter(rows["null_mean"], y, color="#999999", s=24, label="null mean")
        ax.scatter(rows["observed_rsa_similarity"], y, color="#D55E00", s=34, label="observed")
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(rows["combo"], fontsize=7)
        ax.set_xlabel(scope)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_red(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    rows = summary.copy()
    rows["combo"] = rows["neural_rdm"].str.replace("_median_flattened", "", regex=False) + "\n" + rows["model_id"]
    rows = rows.sort_values(["neural_rdm", "model_id"])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.hlines(y, rows["null_delta_q025"], rows["null_delta_q975"], color="#999999", linewidth=2)
    ax.scatter(rows["null_delta_mean"], y, color="#999999", s=24, label="date-matched null")
    ax.scatter(rows["observed_delta_inside_minus_outside"], y, color="#D55E00", s=34, label="observed")
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["combo"], fontsize=8)
    ax.set_xlabel("inside minus outside RSA")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def model_colors() -> dict[str, str]:
    return {
        "global_qc20_log2_euclidean": "#0072B2",
        "global_profile_default_correlation": "#009E73",
        "best_weighted_fusion_fixed_weights": "#D55E00",
    }


def write_summary(
    scope_summary: pd.DataFrame,
    perm_summary: pd.DataFrame,
    red_summary: pd.DataFrame,
    date_counts: pd.DataFrame,
) -> None:
    scopes = ["all_pairs", "within_date_only", "cross_date_only", "date_pair_stratified_rank_all"]
    key_scope = scope_summary.loc[scope_summary["scope"].isin(scopes)].sort_values(["model_id", "neural_rdm", "scope"])
    all_perm = perm_summary.loc[perm_summary["scope"].eq("all_pairs")].sort_values(["model_id", "neural_rdm"])
    strat_perm = perm_summary.loc[perm_summary["scope"].eq("date_pair_stratified_rank_all")].sort_values(
        ["model_id", "neural_rdm"]
    )
    lines = [
        "# Date-Controlled RSA Review",
        "",
        "- Scope: standalone exploratory review for `results/202604_without_20260331`; existing RSA code and figures were not modified.",
        "- Neural RDM: non-ASE L/R merged trial-median correlation RDMs.",
        "- Date control is descriptive because every stimulus belongs to exactly one date in this filtered batch.",
        f"- Date-preserving permutation iterations: {N_ITER}; seed: {SEED}.",
        "",
        "## Stimulus Counts by Date",
        "",
        markdown_table(date_counts[["date", "n_stimuli"]]),
        "",
        "## Main Scope Summary",
        "",
        markdown_table(key_scope[["neural_rdm", "model_id", "scope", "n_pairs", "rsa_similarity"]]),
        "",
        "## Date-Preserving Null: All Pairs",
        "",
        markdown_table(
            all_perm[
            [
                "neural_rdm",
                "model_id",
                "observed_rsa_similarity",
                "null_mean",
                "null_q025",
                "null_q975",
                "p_value_one_sided_ge",
            ]
            ]
        ),
        "",
        "## Date-Pair Stratified Null",
        "",
        markdown_table(
            strat_perm[
            [
                "neural_rdm",
                "model_id",
                "observed_rsa_similarity",
                "null_mean",
                "null_q025",
                "null_q975",
                "p_value_one_sided_ge",
            ]
            ]
        ),
        "",
        "## Red-Box Date-Matched Null",
        "",
    ]
    if red_summary.empty:
        lines.append("No red-box table was found.")
    else:
        lines.append(
            markdown_table(
                red_summary[
                [
                    "neural_rdm",
                    "model_id",
                    "n_selected_samples",
                    "selected_counts_by_date",
                    "observed_inside_rsa",
                    "observed_outside_rsa",
                    "observed_delta_inside_minus_outside",
                    "null_delta_mean",
                    "null_delta_q025",
                    "null_delta_q975",
                    "p_value_delta_one_sided_ge",
                ]
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `tables/rsa_all_vs_within_date_vs_cross_date.csv`",
            "- `tables/rsa_per_date.csv`",
            "- `tables/rsa_by_date_pair.csv`",
            "- `tables/date_preserving_permutation_summary.csv`",
            "- `tables/red_box_date_matched_null_summary.csv`",
            "- `figures/rsa_scope_comparison.png`",
            "- `figures/rsa_per_date.png`",
            "- `figures/date_preserving_permutation_summary.png`",
            "- `figures/red_box_date_matched_null_summary.png`",
        ]
    )
    (OUT / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns]
    for _, row in frame.iterrows():
        rows.append([format_cell(row[column]) for column in columns])
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(columns))]
    rendered = []
    rendered.append("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(rows[0])) + " |")
    rendered.append("| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |")
    for row in rows[1:]:
        rendered.append("| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |")
    return "\n".join(rendered)


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
