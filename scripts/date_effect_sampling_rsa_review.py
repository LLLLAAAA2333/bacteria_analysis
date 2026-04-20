"""Bootstrap date-effect sensitivity for the filtered 202604 RSA review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REVIEW_ROOT = Path("results/202604_without_20260331/date_controlled_rsa_review")
TABLES = REVIEW_ROOT / "tables"
FIGURES = REVIEW_ROOT / "figures"
PAIR_LEVEL_PATH = TABLES / "pair_level_distances_with_date.csv"
N_ITER = 5000
SEED = 20260420 + 17


@dataclass(frozen=True)
class SamplingScheme:
    id: str
    label: str
    filter_id: str
    balanced_by_date_pair: bool = False
    per_date_pair_n: int | None = None


SCHEMES = [
    SamplingScheme("all_pairs_unrestricted", "all pairs", "all"),
    SamplingScheme("all_date_pair_balanced", "date-pair balanced", "all", True, 78),
    SamplingScheme("within_date_unrestricted", "within-date", "within"),
    SamplingScheme("cross_date_unrestricted", "cross-date", "cross"),
    SamplingScheme("cross_excluding_20260311", "cross no 0311", "cross_no_0311"),
    SamplingScheme("cross_only_20260410_20260414", "04/10 vs 04/14", "cross_0410_0414"),
]


MODEL_ORDER = [
    "global_qc20_log2_euclidean",
    "global_profile_default_correlation",
    "best_weighted_fusion_fixed_weights",
]
NEURAL_ORDER = ["response_window_median_flattened", "full_trajectory_median_flattened"]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIR_LEVEL_PATH)
    for column in ["date_left", "date_right", "date_pair"]:
        pairs[column] = pairs[column].astype(str)

    rng = np.random.default_rng(SEED)
    sampling_summary, sampling_draws = run_sampling_schemes(pairs, rng)
    date_pair_summary = summarize_date_pairs(pairs, rng)
    heterogeneity = summarize_heterogeneity(date_pair_summary)
    leave_one_out = summarize_leave_one_date_pair_out(pairs)

    sampling_summary.to_csv(TABLES / "date_effect_sampling_summary.csv", index=False)
    sampling_draws.to_csv(TABLES / "date_effect_sampling_draws.csv", index=False)
    date_pair_summary.to_csv(TABLES / "date_pair_bootstrap_summary.csv", index=False)
    heterogeneity.to_csv(TABLES / "date_pair_heterogeneity_summary.csv", index=False)
    leave_one_out.to_csv(TABLES / "date_pair_leave_one_out_rsa.csv", index=False)

    plot_sampling_summary(sampling_summary, FIGURES / "date_effect_sampling_summary.png")
    plot_date_pair_summary(date_pair_summary, FIGURES / "date_pair_bootstrap_summary.png")
    write_summary(sampling_summary, heterogeneity, leave_one_out)


def run_sampling_schemes(pairs: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    draw_rows = []
    for (neural, model), combo in pairs.groupby(["neural_rdm", "model_id"], sort=False):
        for scheme in SCHEMES:
            source = filter_pairs(combo, scheme.filter_id)
            if source.empty:
                continue
            source_rsa = spearman(source["neural_value"], source["model_value"])
            draws = bootstrap_scheme(source, scheme, rng)
            finite = draws[np.isfinite(draws)]
            summary_rows.append(
                {
                    "neural_rdm": neural,
                    "model_id": model,
                    "scheme": scheme.id,
                    "scheme_label": scheme.label,
                    "n_source_pairs": int(len(source)),
                    "n_sampled_pairs_per_iteration": sampled_n(source, scheme),
                    "n_date_pairs": int(source["date_pair"].nunique()),
                    "date_pairs": ";".join(sorted(source["date_pair"].unique())),
                    "source_rsa_similarity": source_rsa,
                    "bootstrap_median": quantile(finite, 0.5),
                    "bootstrap_mean": mean(finite),
                    "bootstrap_sd": sd(finite),
                    "bootstrap_q025": quantile(finite, 0.025),
                    "bootstrap_q975": quantile(finite, 0.975),
                    "bootstrap_minus_source_median": quantile(finite, 0.5) - source_rsa,
                    "n_iterations": N_ITER,
                    "n_valid_iterations": int(len(finite)),
                }
            )
            draw_rows.extend(
                {
                    "neural_rdm": neural,
                    "model_id": model,
                    "scheme": scheme.id,
                    "iteration": i,
                    "rsa_similarity": value,
                }
                for i, value in enumerate(draws)
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def filter_pairs(pairs: pd.DataFrame, filter_id: str) -> pd.DataFrame:
    if filter_id == "all":
        return pairs.copy()
    if filter_id == "within":
        return pairs.loc[pairs["date_pair_type"].eq("within_date")].copy()
    if filter_id == "cross":
        return pairs.loc[pairs["date_pair_type"].eq("cross_date")].copy()
    if filter_id == "cross_no_0311":
        mask = (
            pairs["date_pair_type"].eq("cross_date")
            & ~pairs["date_left"].eq("20260311")
            & ~pairs["date_right"].eq("20260311")
        )
        return pairs.loc[mask].copy()
    if filter_id == "cross_0410_0414":
        return pairs.loc[pairs["date_pair"].eq("20260410|20260414")].copy()
    raise ValueError(f"Unknown filter_id: {filter_id}")


def bootstrap_scheme(source: pd.DataFrame, scheme: SamplingScheme, rng: np.random.Generator) -> np.ndarray:
    draws = np.full(N_ITER, np.nan, dtype=float)
    if scheme.balanced_by_date_pair:
        groups = [group.index.to_numpy() for _, group in source.groupby("date_pair", sort=True)]
        if not groups:
            return draws
        per_group = scheme.per_date_pair_n or min(len(group) for group in groups)
        for iteration in range(N_ITER):
            sampled_index = np.concatenate([rng.choice(group, size=per_group, replace=True) for group in groups])
            sampled = source.loc[sampled_index]
            draws[iteration] = spearman(sampled["neural_value"], sampled["model_value"])
        return draws

    index = source.index.to_numpy()
    n = len(index)
    for iteration in range(N_ITER):
        sampled = source.loc[rng.choice(index, size=n, replace=True)]
        draws[iteration] = spearman(sampled["neural_value"], sampled["model_value"])
    return draws


def sampled_n(source: pd.DataFrame, scheme: SamplingScheme) -> int:
    if not scheme.balanced_by_date_pair:
        return int(len(source))
    per_group = scheme.per_date_pair_n or int(source.groupby("date_pair").size().min())
    return int(per_group * source["date_pair"].nunique())


def summarize_date_pairs(pairs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (neural, model, date_pair), source in pairs.groupby(["neural_rdm", "model_id", "date_pair"], sort=True):
        values = bootstrap_simple(source, rng)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "neural_rdm": neural,
                "model_id": model,
                "date_pair": date_pair,
                "date_pair_type": source["date_pair_type"].iloc[0],
                "n_pairs": int(len(source)),
                "source_rsa_similarity": spearman(source["neural_value"], source["model_value"]),
                "bootstrap_median": quantile(finite, 0.5),
                "bootstrap_q025": quantile(finite, 0.025),
                "bootstrap_q975": quantile(finite, 0.975),
                "bootstrap_sd": sd(finite),
                "n_iterations": N_ITER,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_simple(source: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    index = source.index.to_numpy()
    draws = np.full(N_ITER, np.nan, dtype=float)
    for iteration in range(N_ITER):
        sampled = source.loc[rng.choice(index, size=len(index), replace=True)]
        draws[iteration] = spearman(sampled["neural_value"], sampled["model_value"])
    return draws


def summarize_heterogeneity(date_pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (neural, model, date_pair_type), group in date_pair_summary.groupby(
        ["neural_rdm", "model_id", "date_pair_type"], sort=False
    ):
        values = group["source_rsa_similarity"].astype(float)
        min_row = group.loc[values.idxmin()]
        max_row = group.loc[values.idxmax()]
        rows.append(
            {
                "neural_rdm": neural,
                "model_id": model,
                "date_pair_type": date_pair_type,
                "n_date_pairs": int(len(group)),
                "date_pair_rsa_mean": float(values.mean()),
                "date_pair_rsa_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "date_pair_rsa_min": float(values.min()),
                "date_pair_rsa_max": float(values.max()),
                "date_pair_rsa_range": float(values.max() - values.min()),
                "min_date_pair": min_row["date_pair"],
                "max_date_pair": max_row["date_pair"],
            }
        )
    return pd.DataFrame(rows)


def summarize_leave_one_date_pair_out(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (neural, model), combo in pairs.groupby(["neural_rdm", "model_id"], sort=False):
        for scope, source in [
            ("all_pairs", combo),
            ("cross_date_only", combo.loc[combo["date_pair_type"].eq("cross_date")]),
        ]:
            baseline = spearman(source["neural_value"], source["model_value"])
            for date_pair in sorted(source["date_pair"].unique()):
                reduced = source.loc[~source["date_pair"].eq(date_pair)]
                rsa = spearman(reduced["neural_value"], reduced["model_value"])
                rows.append(
                    {
                        "neural_rdm": neural,
                        "model_id": model,
                        "scope": scope,
                        "left_out_date_pair": date_pair,
                        "n_pairs_after_drop": int(len(reduced)),
                        "baseline_rsa_similarity": baseline,
                        "drop_one_rsa_similarity": rsa,
                        "delta_vs_baseline": rsa - baseline,
                    }
                )
    return pd.DataFrame(rows)


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left = pd.Series(left, dtype="float64")
    right = pd.Series(right, dtype="float64")
    mask = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return np.nan
    left_rank = left[mask].rank(method="average")
    right_rank = right[mask].rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else np.nan


def plot_sampling_summary(summary: pd.DataFrame, path: Path) -> None:
    schemes = [scheme.id for scheme in SCHEMES]
    scheme_labels = {scheme.id: scheme.label for scheme in SCHEMES}
    colors = model_colors()
    fig, axes = plt.subplots(1, len(NEURAL_ORDER), figsize=(14.0, 5.0), sharey=True)
    if len(NEURAL_ORDER) == 1:
        axes = [axes]
    y = np.arange(len(schemes))
    offsets = np.linspace(-0.22, 0.22, len(MODEL_ORDER))
    for ax, neural in zip(axes, NEURAL_ORDER):
        view = summary.loc[summary["neural_rdm"].eq(neural)]
        for offset, model in zip(offsets, MODEL_ORDER):
            model_rows = view.loc[view["model_id"].eq(model)].set_index("scheme")
            median = [model_rows.loc[scheme, "bootstrap_median"] if scheme in model_rows.index else np.nan for scheme in schemes]
            low = [model_rows.loc[scheme, "bootstrap_q025"] if scheme in model_rows.index else np.nan for scheme in schemes]
            high = [model_rows.loc[scheme, "bootstrap_q975"] if scheme in model_rows.index else np.nan for scheme in schemes]
            y_pos = y + offset
            ax.hlines(y_pos, low, high, color=colors[model], linewidth=1.7, alpha=0.7)
            ax.scatter(median, y_pos, color=colors[model], s=24, label=model)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([scheme_labels[scheme] for scheme in schemes])
        ax.set_xlabel(neural.replace("_median_flattened", ""))
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("sampling scheme")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_date_pair_summary(summary: pd.DataFrame, path: Path) -> None:
    qc = summary.loc[summary["model_id"].eq("global_qc20_log2_euclidean")].copy()
    date_pairs = sorted(qc["date_pair"].unique())
    colors = {
        "within_date": "#0072B2",
        "cross_date": "#D55E00",
    }
    fig, axes = plt.subplots(1, len(NEURAL_ORDER), figsize=(13.0, 5.2), sharey=True)
    if len(NEURAL_ORDER) == 1:
        axes = [axes]
    y = np.arange(len(date_pairs))
    for ax, neural in zip(axes, NEURAL_ORDER):
        rows = qc.loc[qc["neural_rdm"].eq(neural)].set_index("date_pair")
        for idx, date_pair in enumerate(date_pairs):
            if date_pair not in rows.index:
                continue
            row = rows.loc[date_pair]
            color = colors[row["date_pair_type"]]
            ax.hlines(idx, row["bootstrap_q025"], row["bootstrap_q975"], color=color, linewidth=1.8, alpha=0.75)
            ax.scatter(row["bootstrap_median"], idx, color=color, s=25)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(date_pairs)
        ax.set_xlabel(neural.replace("_median_flattened", ""))
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("date pair, QC20 log2 Euclidean")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def model_colors() -> dict[str, str]:
    return {
        "global_qc20_log2_euclidean": "#0072B2",
        "global_profile_default_correlation": "#009E73",
        "best_weighted_fusion_fixed_weights": "#D55E00",
    }


def write_summary(summary: pd.DataFrame, heterogeneity: pd.DataFrame, leave_one_out: pd.DataFrame) -> None:
    key_schemes = [
        "all_pairs_unrestricted",
        "all_date_pair_balanced",
        "cross_date_unrestricted",
        "cross_excluding_20260311",
        "cross_only_20260410_20260414",
    ]
    key_summary = summary.loc[summary["scheme"].isin(key_schemes)].sort_values(["model_id", "neural_rdm", "scheme"])
    cross_heterogeneity = heterogeneity.loc[heterogeneity["date_pair_type"].eq("cross_date")].sort_values(
        ["model_id", "neural_rdm"]
    )
    loo = (
        leave_one_out.loc[leave_one_out["scope"].eq("cross_date_only")]
        .assign(abs_delta=lambda frame: frame["delta_vs_baseline"].abs())
        .sort_values(["model_id", "neural_rdm", "abs_delta"], ascending=[True, True, False])
        .groupby(["neural_rdm", "model_id"], as_index=False)
        .head(2)
    )

    lines = [
        "# Date-Effect Sampling RSA Review",
        "",
        f"- Input: `{PAIR_LEVEL_PATH.as_posix()}`",
        f"- Bootstrap iterations: {N_ITER}; seed: {SEED}.",
        "- Sampling is pair-level and exploratory; it tests sensitivity to date-pair composition, not biological repeatability of identical stimuli across dates.",
        "",
        "## Sampling Schemes",
        "",
        markdown_table(
            key_summary[
                [
                    "neural_rdm",
                    "model_id",
                    "scheme",
                    "n_source_pairs",
                    "n_sampled_pairs_per_iteration",
                    "source_rsa_similarity",
                    "bootstrap_median",
                    "bootstrap_q025",
                    "bootstrap_q975",
                    "bootstrap_minus_source_median",
                ]
            ]
        ),
        "",
        "## Cross-Date Pair Heterogeneity",
        "",
        markdown_table(
            cross_heterogeneity[
                [
                    "neural_rdm",
                    "model_id",
                    "n_date_pairs",
                    "date_pair_rsa_mean",
                    "date_pair_rsa_sd",
                    "date_pair_rsa_min",
                    "date_pair_rsa_max",
                    "date_pair_rsa_range",
                    "min_date_pair",
                    "max_date_pair",
                ]
            ]
        ),
        "",
        "## Largest Cross-Date Leave-One-Pair Effects",
        "",
        markdown_table(
            loo[
                [
                    "neural_rdm",
                    "model_id",
                    "left_out_date_pair",
                    "baseline_rsa_similarity",
                    "drop_one_rsa_similarity",
                    "delta_vs_baseline",
                ]
            ]
        ),
        "",
        "## Output Files",
        "",
        "- `tables/date_effect_sampling_summary.csv`",
        "- `tables/date_effect_sampling_draws.csv`",
        "- `tables/date_pair_bootstrap_summary.csv`",
        "- `tables/date_pair_heterogeneity_summary.csv`",
        "- `tables/date_pair_leave_one_out_rsa.csv`",
        "- `figures/date_effect_sampling_summary.png`",
        "- `figures/date_pair_bootstrap_summary.png`",
    ]
    (REVIEW_ROOT / "date_effect_sampling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns]
    for _, row in frame.iterrows():
        rows.append([format_cell(row[column]) for column in columns])
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(columns))]
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
    return str(value)


if __name__ == "__main__":
    main()
