"""Output writers and figures for reliability analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from bacteria_analysis.io import write_json, write_parquet


def ensure_reliability_output_dirs(output_root: str | Path) -> dict[str, Path]:
    """Create and return the standard reliability output tree."""

    root = Path(output_root)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    qc_dir = root / "qc"

    for directory in (root, tables_dir, figures_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": root,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "qc_dir": qc_dir,
    }


def _prepare_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        if prepared[column].dtype != "object":
            continue
        prepared[column] = prepared[column].map(
            lambda value: "|".join(map(str, value)) if isinstance(value, (list, tuple, set)) else value
        )
    return prepared


def _write_markdown_summary(summary: dict[str, Any], path: str | Path) -> Path:
    lines = [
        "# Reliability Analysis Run Summary",
        "",
        "## Dataset",
        f"- Trials: {summary['n_trials']}",
        f"- Individuals: {summary['n_individuals']}",
        f"- Dates: {summary['n_dates']}",
        f"- Stimuli: {summary['n_stimuli']}",
        "",
        "## Focus View",
        f"- Focus view: {summary['focus_view']}",
        f"- Distance gap: {summary['focus_view_distance_gap']:.4f}",
        f"- LOIO mean accuracy: {summary['focus_view_loio_accuracy_mean']:.4f}",
        f"- LODO mean accuracy: {summary['focus_view_lodo_accuracy_mean']:.4f}",
        "",
        "## Strongest Views",
    ]
    strongest_views = summary.get("strongest_views", [])
    if strongest_views:
        for row in strongest_views:
            lines.append(f"- {row['view_name']}: distance_gap={row['distance_gap']:.4f}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Output Paths",
            f"- Final summary table: {summary['final_summary_path']}",
            f"- Figures directory: {summary['figures_dir']}",
            f"- QC directory: {summary['qc_dir']}",
        ]
    )
    per_date_figure_names = summary.get("per_date_same_vs_different_figure_names", [])
    if per_date_figure_names:
        lines.extend(["", "## Per-Date Same vs Different Figures"])
        lines.extend(f"- {figure_name}" for figure_name in per_date_figure_names)
    pooled_per_stimulus_figure_name = summary.get("pooled_per_stimulus_same_vs_different_figure_name")
    if pooled_per_stimulus_figure_name:
        lines.extend(["", "## Per-Stimulus Same vs Different Figures", f"- {pooled_per_stimulus_figure_name}"])
    per_date_per_stimulus_figure_names = summary.get("per_date_per_stimulus_same_vs_different_figure_names", [])
    if per_date_per_stimulus_figure_names:
        lines.extend(f"- {figure_name}" for figure_name in per_date_per_stimulus_figure_names)

    output_path = Path(path)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _remove_retired_reliability_figures(figures_dir: Path) -> None:
    retired_figure_names = (
        "same_vs_different_distributions.png",
        "same_vs_different_distributions__raincloud.png",
        "same_vs_different_distributions__violin_clean.png",
        "cross_view_reliability_comparison.png",
        "leave_one_individual_out_summary.png",
        "leave_one_date_out_summary.png",
        "split_half_summary.png",
    )
    for figure_name in retired_figure_names:
        figure_path = figures_dir / figure_name
        if figure_path.exists():
            figure_path.unlink()

    for stale_matrix_figure in figures_dir.glob("stimulus_distance_matrix__*.png"):
        stale_matrix_figure.unlink()
    for stale_per_date_figure in figures_dir.glob("same_vs_different_by_date__*.png"):
        stale_per_date_figure.unlink()
    for stale_per_stimulus_figure in figures_dir.glob("per_stimulus_same_vs_different__*.png"):
        stale_per_stimulus_figure.unlink()


def _format_plot_value(value: object, fmt: str = ".4f") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return format(float(value), fmt)


def _compute_quantile_axis_limits(
    values: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    pad_ratio: float = 0.08,
) -> tuple[float, float] | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None

    lower = float(numeric.quantile(lower_quantile))
    upper = float(numeric.quantile(upper_quantile))
    if not pd.notna(lower) or not pd.notna(upper):
        return None

    if upper <= lower:
        center = lower
        span = max(abs(center) * 0.1, 0.1)
        return center - span, center + span

    span = upper - lower
    padding = span * pad_ratio
    return lower - padding, upper + padding


SAME_VS_DIFFERENT_ORDER = ["same", "different"]
SAME_VS_DIFFERENT_PALETTE = {
    "same": "#5BA4A4",
    "different": "#C98C7A",
}
SAME_VS_DIFFERENT_POINT_PALETTE = {
    "same": "#2E6F6F",
    "different": "#9C6656",
}
SAME_VS_DIFFERENT_LOWER_QUANTILE = 0.02
SAME_VS_DIFFERENT_UPPER_QUANTILE = 0.98


def _build_focus_view_same_vs_different_plot_frame(
    comparisons: pd.DataFrame,
    focus_view: str,
) -> pd.DataFrame:
    valid = comparisons[
        (comparisons["comparison_status"].astype(str) == "ok")
        & (comparisons["view_name"].astype(str) == str(focus_view))
    ].copy()
    if valid.empty:
        valid["comparison_label"] = pd.Series(dtype="object")
        return valid

    valid["comparison_label"] = valid["same_stimulus"].map({True: "same", False: "different"})
    valid["comparison_label"] = pd.Categorical(
        valid["comparison_label"],
        categories=SAME_VS_DIFFERENT_ORDER,
        ordered=True,
    )
    return valid


def _build_focus_view_same_vs_different_plot_frame_for_date(
    comparisons: pd.DataFrame,
    focus_view: str,
    date_value: object,
) -> pd.DataFrame:
    valid = _build_focus_view_same_vs_different_plot_frame(comparisons, focus_view)
    if valid.empty or not {"date_a", "date_b"}.issubset(valid.columns):
        return valid.iloc[0:0].copy()

    date_text = str(date_value)
    if "same_date" in valid.columns:
        valid = valid.loc[valid["same_date"].astype(bool)].copy()

    return valid.loc[
        valid["date_a"].astype(str).eq(date_text) & valid["date_b"].astype(str).eq(date_text)
    ].copy()


def _summarize_same_vs_different_plot_frame(valid: pd.DataFrame) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame(
            index=SAME_VS_DIFFERENT_ORDER,
            columns=["median", "q1", "q3", "count"],
            dtype=float,
        )

    return (
        valid.groupby("comparison_label", sort=False, observed=False)["distance"]
        .agg(
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            count="size",
        )
        .reindex(SAME_VS_DIFFERENT_ORDER)
    )


def _sample_same_vs_different_points(
    valid: pd.DataFrame,
    max_points_per_group: int = 700,
    random_state: int = 0,
) -> pd.DataFrame:
    sampled_groups: list[pd.DataFrame] = []
    for label in SAME_VS_DIFFERENT_ORDER:
        group = valid.loc[valid["comparison_label"] == label].copy()
        if group.empty:
            continue
        if len(group) > max_points_per_group:
            group = group.sample(n=max_points_per_group, random_state=random_state)
        sampled_groups.append(group)

    if not sampled_groups:
        return valid.iloc[0:0].copy()
    return pd.concat(sampled_groups, ignore_index=True)


def _build_same_vs_different_tick_labels(summary: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for label in SAME_VS_DIFFERENT_ORDER:
        count = summary.loc[label, "count"] if label in summary.index else 0
        count_value = int(count) if pd.notna(count) else 0
        labels.append(f"{label}\nn = {count_value:,}")
    return labels


def _compute_same_vs_different_center_gap(summary: pd.DataFrame, metric: str = "median") -> float | None:
    if metric not in {"median", "q1", "q3"}:
        raise ValueError(f"Unsupported metric for center gap: {metric}")
    if not set(SAME_VS_DIFFERENT_ORDER).issubset(summary.index):
        return None

    same_value = summary.loc["same", metric]
    different_value = summary.loc["different", metric]
    if pd.isna(same_value) or pd.isna(different_value):
        return None
    return float(different_value) - float(same_value)


def _build_same_vs_different_subtitle(
    focus_row: pd.Series,
    summary: pd.DataFrame,
    metric: str = "gap",
) -> str:
    p_value_text = _format_plot_value(focus_row.get("p_value"), ".4g")
    if metric == "gap":
        return f"Gap = {_format_plot_value(focus_row.get('distance_gap'), '.3f')} | permutation p = {p_value_text}"
    if metric == "median":
        center_gap = _compute_same_vs_different_center_gap(summary, metric="median")
        return f"Delta median = {_format_plot_value(center_gap, '.3f')} | permutation p = {p_value_text}"
    raise ValueError(f"Unsupported subtitle metric: {metric}")


def _remove_plot_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _add_same_vs_different_summary_markers(ax: plt.Axes, summary: pd.DataFrame) -> None:
    for x_position, label in enumerate(SAME_VS_DIFFERENT_ORDER):
        if label not in summary.index or pd.isna(summary.loc[label, "median"]):
            continue
        ax.vlines(
            x_position,
            float(summary.loc[label, "q1"]),
            float(summary.loc[label, "q3"]),
            color="#111111",
            linewidth=2.2,
            zorder=4,
        )
        ax.scatter(
            [x_position],
            [float(summary.loc[label, "median"])],
            color="#111111",
            s=26,
            zorder=5,
        )


def _add_same_vs_different_gap_bracket(ax: plt.Axes, summary: pd.DataFrame) -> None:
    median_gap = _compute_same_vs_different_center_gap(summary, metric="median")
    if median_gap is None:
        return

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        return

    bottom = ymax - 0.12 * span
    top = ymax - 0.08 * span
    ax.plot([0, 0, 1, 1], [bottom, top, top, bottom], color="#444444", linewidth=1.2, clip_on=False, zorder=6)
    ax.text(
        0.5,
        top + 0.015 * span,
        f"Delta median = {_format_plot_value(median_gap, '.3f')}",
        ha="center",
        va="bottom",
        color="#444444",
        fontsize=9,
    )


def _clip_violin_halves(ax: plt.Axes, centers: list[float], sides: list[str], half_width: float = 0.34) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1.0)
    violin_collections = list(ax.collections[: len(centers)])
    for collection, center, side in zip(violin_collections, centers, sides, strict=True):
        if side == "left":
            clip_rect = Rectangle(
                (center - half_width, ymin - 0.1 * span),
                half_width,
                span * 1.2,
                transform=ax.transData,
            )
        else:
            clip_rect = Rectangle(
                (center, ymin - 0.1 * span),
                half_width,
                span * 1.2,
                transform=ax.transData,
            )
        collection.set_clip_path(clip_rect)


def _add_same_vs_different_half_points(
    ax: plt.Axes,
    sampled: pd.DataFrame,
    max_points_per_group: int = 650,
    random_state: int = 0,
) -> None:
    reduced = _sample_same_vs_different_points(
        sampled,
        max_points_per_group=max_points_per_group,
        random_state=random_state,
    )
    if reduced.empty:
        return

    rng = np.random.default_rng(random_state)
    offsets = {
        "same": rng.uniform(0.03, 0.13, size=len(reduced.loc[reduced["comparison_label"] == "same"])),
        "different": rng.uniform(-0.13, -0.03, size=len(reduced.loc[reduced["comparison_label"] == "different"])),
    }
    center_map = {"same": 0.0, "different": 1.0}

    for label in SAME_VS_DIFFERENT_ORDER:
        group = reduced.loc[reduced["comparison_label"] == label].copy()
        if group.empty:
            continue
        x_positions = center_map[label] + offsets[label]
        ax.scatter(
            x_positions,
            group["distance"],
            color=SAME_VS_DIFFERENT_POINT_PALETTE[label],
            alpha=0.14,
            s=8,
            linewidths=0,
            zorder=3,
        )


def _style_same_vs_different_axis(
    ax: plt.Axes,
    valid: pd.DataFrame,
    focus_row: pd.Series,
    summary: pd.DataFrame,
    focus_view: str,
    lower_quantile: float = SAME_VS_DIFFERENT_LOWER_QUANTILE,
    upper_quantile: float = SAME_VS_DIFFERENT_UPPER_QUANTILE,
    title_text: str | None = None,
    subtitle_metric: str = "gap",
    subtitle_text: str | None = None,
) -> None:
    axis_limits = _compute_quantile_axis_limits(
        valid["distance"],
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    if axis_limits is not None:
        ax.set_ylim(*axis_limits)

    ax.figure.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xticks(range(len(SAME_VS_DIFFERENT_ORDER)))
    ax.set_xticklabels(_build_same_vs_different_tick_labels(summary))
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Distance")
    if title_text is None:
        title_text = f"Same vs Different Trial Distances ({focus_view})"
    resolved_subtitle = subtitle_text
    if resolved_subtitle is None:
        resolved_subtitle = _build_same_vs_different_subtitle(focus_row, summary, metric=subtitle_metric)
    ax.set_title(title_text, pad=16)
    ax.text(
        0.5,
        1.02,
        resolved_subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="y", linestyle="--", color="#E4E4E4", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def _resolve_focus_view_row(final_summary: pd.DataFrame, focus_view: str) -> pd.Series:
    focus_rows = final_summary.loc[final_summary["view_name"].astype(str) == focus_view]
    if focus_rows.empty:
        return pd.Series(dtype=object)
    return focus_rows.iloc[0]


def _pick_stimulus_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    columns = {"stimulus"}
    if "stim_name" in metadata.columns:
        columns.add("stim_name")
    if "stim_color" in metadata.columns:
        columns.add("stim_color")

    stimulus_metadata = metadata.loc[:, sorted(columns)].copy()
    stimulus_metadata["stimulus"] = stimulus_metadata["stimulus"].astype(str)
    if "stim_name" not in stimulus_metadata.columns:
        stimulus_metadata["stim_name"] = stimulus_metadata["stimulus"]
    if "stim_color" not in stimulus_metadata.columns:
        stimulus_metadata["stim_color"] = ""

    def _first_non_blank(series: pd.Series, fallback: str = "") -> str:
        for value in series:
            if pd.notna(value) and str(value).strip():
                return str(value)
        return fallback

    grouped = (
        stimulus_metadata.groupby("stimulus", sort=True, dropna=False)
        .agg(
            stim_name=("stim_name", lambda values: _first_non_blank(values, fallback="")),
            stim_color=("stim_color", lambda values: _first_non_blank(values, fallback="")),
        )
        .reset_index()
    )
    grouped["stim_name"] = grouped["stim_name"].where(grouped["stim_name"].astype(str).str.len() > 0, grouped["stimulus"])
    return grouped


def _build_focus_view_stimulus_gap_summary(
    comparisons: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_view: str,
    date_value: object | None = None,
) -> pd.DataFrame:
    focus_comparisons = comparisons[
        (comparisons["comparison_status"].astype(str) == "ok")
        & (comparisons["view_name"].astype(str) == str(focus_view))
    ].copy()
    metadata_source = metadata.copy()
    if date_value is not None:
        date_text = str(date_value)
        if "same_date" in focus_comparisons.columns:
            focus_comparisons = focus_comparisons.loc[focus_comparisons["same_date"].astype(bool)].copy()
        focus_comparisons = focus_comparisons.loc[
            focus_comparisons["date_a"].astype(str).eq(date_text) & focus_comparisons["date_b"].astype(str).eq(date_text)
        ].copy()
        if "date" in metadata_source.columns:
            metadata_source = metadata_source.loc[metadata_source["date"].astype(str).eq(date_text)].copy()

    stimulus_metadata = _pick_stimulus_metadata(metadata_source)
    if focus_comparisons.empty:
        return stimulus_metadata.assign(
            same_count=0,
            same_mean_distance=float("nan"),
            different_count=0,
            different_mean_distance=float("nan"),
            distance_gap=float("nan"),
        )

    same_summary = (
        focus_comparisons.loc[focus_comparisons["same_stimulus"].astype(bool), ["stimulus_a", "distance"]]
        .rename(columns={"stimulus_a": "stimulus"})
        .assign(stimulus=lambda frame: frame["stimulus"].astype(str))
        .groupby("stimulus", sort=True, dropna=False)
        .agg(
            same_count=("distance", "size"),
            same_mean_distance=("distance", "mean"),
        )
        .reset_index()
    )

    different_comparisons = focus_comparisons.loc[
        ~focus_comparisons["same_stimulus"].astype(bool),
        ["stimulus_a", "stimulus_b", "distance"],
    ].copy()
    different_long = pd.concat(
        [
            different_comparisons.loc[:, ["stimulus_a", "distance"]].rename(columns={"stimulus_a": "stimulus"}),
            different_comparisons.loc[:, ["stimulus_b", "distance"]].rename(columns={"stimulus_b": "stimulus"}),
        ],
        ignore_index=True,
    )
    different_long["stimulus"] = different_long["stimulus"].astype(str)
    different_summary = (
        different_long.groupby("stimulus", sort=True, dropna=False)
        .agg(
            different_count=("distance", "size"),
            different_mean_distance=("distance", "mean"),
        )
        .reset_index()
    )

    summary = stimulus_metadata.merge(same_summary, on="stimulus", how="left").merge(
        different_summary,
        on="stimulus",
        how="left",
    )
    summary["same_count"] = summary["same_count"].fillna(0).astype(int)
    summary["different_count"] = summary["different_count"].fillna(0).astype(int)
    summary["distance_gap"] = summary["different_mean_distance"] - summary["same_mean_distance"]
    return summary.sort_values(
        by=["distance_gap", "different_mean_distance", "stimulus"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def _build_stimulus_availability_matrix(metadata: pd.DataFrame) -> pd.DataFrame:
    availability = metadata.copy()
    availability["date"] = availability["date"].astype(str)
    availability["stimulus"] = availability["stimulus"].astype(str)

    count_column = "trial_id" if "trial_id" in availability.columns else "stimulus"
    counts = (
        availability.groupby(["date", "stimulus"], sort=True, dropna=False)[count_column]
        .nunique()
        .unstack(fill_value=0)
    )
    return counts.sort_index(axis=0).sort_index(axis=1)


def _plot_same_vs_different_raincloud(
    comparisons: pd.DataFrame,
    final_summary: pd.DataFrame,
    focus_view: str,
    path: Path,
) -> Path:
    valid = _build_focus_view_same_vs_different_plot_frame(comparisons, focus_view)
    focus_row = _resolve_focus_view_row(final_summary, focus_view)
    summary = _summarize_same_vs_different_plot_frame(valid)
    plt.figure(figsize=(6.8, 4.8))
    ax = plt.gca()
    if valid.empty:
        ax.text(0.5, 0.5, f"No valid comparisons for {focus_view}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save_figure(path)

    sns.violinplot(
        data=valid,
        x="comparison_label",
        hue="comparison_label",
        y="distance",
        order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_PALETTE,
        dodge=False,
        inner=None,
        cut=0,
        linewidth=0.9,
        width=0.58,
        density_norm="width",
        ax=ax,
    )
    _clip_violin_halves(ax, centers=[0.0, 1.0], sides=["left", "right"], half_width=0.30)
    for collection in ax.collections[:2]:
        collection.set_alpha(0.62)

    sns.boxplot(
        data=valid,
        x="comparison_label",
        y="distance",
        order=SAME_VS_DIFFERENT_ORDER,
        width=0.16,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#4A4A4A", "alpha": 0.95},
        whiskerprops={"color": "#4A4A4A", "linewidth": 1.0},
        capprops={"color": "#4A4A4A", "linewidth": 1.0},
        medianprops={"color": "#333333", "linewidth": 1.5},
        ax=ax,
    )
    _add_same_vs_different_half_points(ax, valid, max_points_per_group=650, random_state=0)
    _add_same_vs_different_summary_markers(ax, summary)
    _style_same_vs_different_axis(ax, valid, focus_row, summary, focus_view)
    _add_same_vs_different_gap_bracket(ax, summary)
    return _save_figure(path)


def _plot_same_vs_different_violin_clean(
    comparisons: pd.DataFrame,
    final_summary: pd.DataFrame,
    focus_view: str,
    path: Path,
) -> Path:
    valid = _build_focus_view_same_vs_different_plot_frame(comparisons, focus_view)
    focus_row = _resolve_focus_view_row(final_summary, focus_view)
    summary = _summarize_same_vs_different_plot_frame(valid)
    plt.figure(figsize=(6.7, 4.8))
    ax = plt.gca()
    if valid.empty:
        ax.text(0.5, 0.5, f"No valid comparisons for {focus_view}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save_figure(path)

    sns.violinplot(
        data=valid,
        x="comparison_label",
        hue="comparison_label",
        y="distance",
        order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_PALETTE,
        dodge=False,
        inner=None,
        cut=0,
        linewidth=1.0,
        width=0.68,
        density_norm="width",
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.8)

    _remove_plot_legend(ax)
    _add_same_vs_different_summary_markers(ax, summary)
    _style_same_vs_different_axis(ax, valid, focus_row, summary, focus_view)
    return _save_figure(path)


def _plot_same_vs_different_boxen_points(
    comparisons: pd.DataFrame,
    final_summary: pd.DataFrame,
    focus_view: str,
    path: Path,
) -> Path:
    valid = _build_focus_view_same_vs_different_plot_frame(comparisons, focus_view)
    focus_row = _resolve_focus_view_row(final_summary, focus_view)
    summary = _summarize_same_vs_different_plot_frame(valid)
    sampled = _sample_same_vs_different_points(valid, max_points_per_group=1000, random_state=0)
    plt.figure(figsize=(6.9, 4.9))
    ax = plt.gca()
    if valid.empty:
        ax.text(0.5, 0.5, f"No valid comparisons for {focus_view}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save_figure(path)

    sns.boxenplot(
        data=valid,
        x="comparison_label",
        hue="comparison_label",
        y="distance",
        order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_PALETTE,
        dodge=False,
        width=0.56,
        linewidth=0.8,
        ax=ax,
    )
    for patch in ax.patches:
        patch.set_edgecolor("#5B5B5B")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.72)
    sns.stripplot(
        data=sampled,
        x="comparison_label",
        y="distance",
        hue="comparison_label",
        order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_POINT_PALETTE,
        dodge=False,
        jitter=0.14,
        alpha=0.09,
        size=1.7,
        linewidth=0,
        ax=ax,
    )
    _remove_plot_legend(ax)
    _style_same_vs_different_axis(
        ax,
        valid,
        focus_row,
        summary,
        focus_view,
        title_text="Same vs Different Trial Distances",
        subtitle_metric="median",
    )
    return _save_figure(path)


def _plot_same_vs_different_ecdf(
    comparisons: pd.DataFrame,
    final_summary: pd.DataFrame,
    focus_view: str,
    path: Path,
) -> Path:
    valid = _build_focus_view_same_vs_different_plot_frame(comparisons, focus_view)
    focus_row = _resolve_focus_view_row(final_summary, focus_view)
    summary = _summarize_same_vs_different_plot_frame(valid)
    plt.figure(figsize=(6.8, 4.8))
    ax = plt.gca()
    if valid.empty:
        ax.text(0.5, 0.5, f"No valid comparisons for {focus_view}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save_figure(path)

    sns.ecdfplot(
        data=valid,
        x="distance",
        hue="comparison_label",
        hue_order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_PALETTE,
        linewidth=2.4,
        ax=ax,
    )
    median_gap = _compute_same_vs_different_center_gap(summary, metric="median")
    for label in SAME_VS_DIFFERENT_ORDER:
        if label not in summary.index or pd.isna(summary.loc[label, "median"]):
            continue
        ax.axvline(
            float(summary.loc[label, "median"]),
            color=SAME_VS_DIFFERENT_POINT_PALETTE[label],
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )

    ax.figure.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(f"Same vs Different Distance ECDF ({focus_view})", pad=16)
    ax.text(
        0.5,
        1.02,
        (
            f"Gap = {_format_plot_value(focus_row.get('distance_gap'), '.3f')} | "
            f"permutation p = {_format_plot_value(focus_row.get('p_value'), '.4g')}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    if median_gap is not None:
        ax.text(
            0.03,
            0.94,
            f"Delta median = {_format_plot_value(median_gap, '.3f')}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.25"},
        )
    ax.grid(axis="both", linestyle="--", color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    _remove_plot_legend(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color=SAME_VS_DIFFERENT_PALETTE["same"], linewidth=2.4, label="same"),
            Line2D([0], [0], color=SAME_VS_DIFFERENT_PALETTE["different"], linewidth=2.4, label="different"),
        ],
        loc="lower right",
        frameon=False,
        title=None,
    )
    return _save_figure(path)


def _write_same_vs_different_variant_figures(
    comparisons: pd.DataFrame,
    final_summary: pd.DataFrame,
    focus_view: str,
    figures_dir: Path,
) -> dict[str, Path]:
    return {
        "same_vs_different_figure_boxen_points": _plot_same_vs_different_boxen_points(
            comparisons,
            final_summary,
            focus_view,
            figures_dir / "same_vs_different_distributions__boxen_points.png",
        ),
        "same_vs_different_figure_ecdf": _plot_same_vs_different_ecdf(
            comparisons,
            final_summary,
            focus_view,
            figures_dir / "same_vs_different_distributions__ecdf.png",
        ),
    }


def _sanitize_path_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "blank"
    invalid_chars = '<>:"/\\|?*'
    sanitized = text.translate(str.maketrans({char: "_" for char in invalid_chars}))
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def _build_same_vs_different_date_subtitle(summary: pd.DataFrame) -> str:
    same_count = summary.loc["same", "count"] if "same" in summary.index else 0
    different_count = summary.loc["different", "count"] if "different" in summary.index else 0
    median_gap = _compute_same_vs_different_center_gap(summary, metric="median")
    same_count_value = int(same_count) if pd.notna(same_count) else 0
    different_count_value = int(different_count) if pd.notna(different_count) else 0
    return (
        "Within-date pairs only"
        f" | same n = {same_count_value:,}"
        f" | different n = {different_count_value:,}"
        f" | Delta median = {_format_plot_value(median_gap, '.3f')}"
    )


def _plot_same_vs_different_boxen_points_for_date(
    comparisons: pd.DataFrame,
    focus_view: str,
    date_value: object,
    path: Path,
) -> Path:
    valid = _build_focus_view_same_vs_different_plot_frame_for_date(comparisons, focus_view, date_value)
    summary = _summarize_same_vs_different_plot_frame(valid)
    sampled = _sample_same_vs_different_points(valid, max_points_per_group=1000, random_state=0)
    date_text = str(date_value)

    plt.figure(figsize=(6.9, 4.9))
    ax = plt.gca()
    if valid.empty:
        ax.text(
            0.5,
            0.5,
            f"No valid within-date comparisons for {date_text} ({focus_view})",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return _save_figure(path)

    sns.boxenplot(
        data=valid,
        x="comparison_label",
        hue="comparison_label",
        y="distance",
        order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_PALETTE,
        dodge=False,
        width=0.56,
        linewidth=0.8,
        ax=ax,
    )
    for patch in ax.patches:
        patch.set_edgecolor("#5B5B5B")
        patch.set_linewidth(0.8)
        patch.set_alpha(0.72)
    sns.stripplot(
        data=sampled,
        x="comparison_label",
        y="distance",
        hue="comparison_label",
        order=SAME_VS_DIFFERENT_ORDER,
        palette=SAME_VS_DIFFERENT_POINT_PALETTE,
        dodge=False,
        jitter=0.14,
        alpha=0.09,
        size=1.7,
        linewidth=0,
        ax=ax,
    )
    _remove_plot_legend(ax)
    _style_same_vs_different_axis(
        ax,
        valid,
        pd.Series(dtype=object),
        summary,
        focus_view,
        title_text=f"Same vs Different Trial Distances ({date_text})",
        subtitle_text=_build_same_vs_different_date_subtitle(summary),
    )
    return _save_figure(path)


def _write_same_vs_different_per_date_figures(
    comparisons: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_view: str,
    figures_dir: Path,
) -> dict[str, Path]:
    if "date" not in metadata.columns:
        return {}

    written: dict[str, Path] = {}
    date_values = sorted(metadata["date"].dropna().astype(str).unique().tolist())
    for date_value in date_values:
        safe_date = _sanitize_path_token(date_value)
        written[f"same_vs_different_by_date_{safe_date}"] = _plot_same_vs_different_boxen_points_for_date(
            comparisons,
            focus_view,
            date_value,
            figures_dir / f"same_vs_different_by_date__{safe_date}.png",
        )
    return written


def _plot_holdout_summary(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    focus_view: str | None = None,
) -> Path:
    plot_summary = summary.copy()
    if focus_view is not None:
        plot_summary = plot_summary.loc[plot_summary["view_name"].astype(str) == focus_view].copy()
    plt.figure(figsize=(5.5 if focus_view is not None else 8, 4.5))
    ax = sns.barplot(data=plot_summary, x="view_name", y="accuracy_mean", color="#4c78a8")
    plt.ylim(0, 1)
    if focus_view is not None and not plot_summary.empty:
        value = float(plot_summary.iloc[0]["accuracy_mean"])
        ax.text(0, value + 0.03, f"{value:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("Focus View" if focus_view is not None else "View")
    plt.ylabel("Accuracy")
    plt.title(f"{title} ({focus_view})" if focus_view is not None else title)
    return _save_figure(path)


def _plot_focus_view_stimulus_gap(
    comparisons: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_view: str,
    path: Path,
    date_value: object | None = None,
) -> Path:
    summary = _build_focus_view_stimulus_gap_summary(comparisons, metadata, focus_view, date_value=date_value)
    plot_data = summary.dropna(subset=["same_mean_distance", "different_mean_distance"]).copy()
    date_text = str(date_value) if date_value is not None else None

    plt.figure(figsize=(5.8, 5.6))
    ax = plt.gca()
    if plot_data.empty:
        empty_label = (
            f"No scorable per-stimulus gaps for {date_text} ({focus_view})"
            if date_text is not None
            else f"No scorable per-stimulus gaps for {focus_view}"
        )
        ax.text(0.5, 0.5, empty_label, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save_figure(path)

    ax.figure.set_facecolor("white")
    ax.set_facecolor("white")
    same_x = 0.0
    different_x = 1.0
    for row in plot_data.itertuples(index=False):
        ax.plot(
            [same_x, different_x],
            [float(row.same_mean_distance), float(row.different_mean_distance)],
            color="#B8B8B8",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )
        ax.scatter(
            [same_x],
            [float(row.same_mean_distance)],
            color=SAME_VS_DIFFERENT_PALETTE["same"],
            alpha=0.7,
            marker="o",
            s=18,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            [different_x],
            [float(row.different_mean_distance)],
            color=SAME_VS_DIFFERENT_PALETTE["different"],
            alpha=0.7,
            marker="o",
            s=18,
            linewidths=0,
            zorder=2,
        )

    paired_summary = pd.DataFrame(
        {
            "median": {
                "same": float(plot_data["same_mean_distance"].median()),
                "different": float(plot_data["different_mean_distance"].median()),
            },
            "q1": {
                "same": float(plot_data["same_mean_distance"].quantile(0.25)),
                "different": float(plot_data["different_mean_distance"].quantile(0.25)),
            },
            "q3": {
                "same": float(plot_data["same_mean_distance"].quantile(0.75)),
                "different": float(plot_data["different_mean_distance"].quantile(0.75)),
            },
            "count": {
                "same": int(len(plot_data)),
                "different": int(len(plot_data)),
            },
        }
    )
    _add_same_vs_different_summary_markers(ax, paired_summary)

    num_increasing = int((plot_data["same_mean_distance"] < plot_data["different_mean_distance"]).sum())
    total_pairs = int(len(plot_data))
    median_gap = _compute_same_vs_different_center_gap(paired_summary, metric="median")

    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([same_x, different_x])
    ax.set_xticklabels(["same", "different"])
    ax.grid(axis="y", linestyle="--", color="#E4E4E4", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Distance")
    if date_text is None:
        title_text = f"Per-Stimulus Same vs Different Distances ({focus_view})"
        prefix_text = None
    else:
        title_text = f"Per-Stimulus Same vs Different Distances ({date_text})"
        prefix_text = f"focus view = {focus_view} | "
    ax.set_title(title_text, pad=24)
    ax.text(
        0.5,
        1.01,
        (
            f"{prefix_text or ''}{num_increasing}/{total_pairs} paired stimuli show same < different"
            f" | Delta median = {_format_plot_value(median_gap, '.3f')}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    omitted = int(len(summary) - len(plot_data))
    if omitted > 0:
        ax.text(
            0.01,
            -0.08,
            f"{omitted} stimuli omitted because same or different comparisons were unavailable.",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    return _save_figure(path)


def _write_per_stimulus_same_vs_different_figures(
    comparisons: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_view: str,
    figures_dir: Path,
) -> dict[str, Path]:
    written = {
        "per_stimulus_same_vs_different_pooled": _plot_focus_view_stimulus_gap(
            comparisons,
            metadata,
            focus_view,
            figures_dir / "per_stimulus_same_vs_different__pooled.png",
        )
    }
    if "date" not in metadata.columns:
        return written

    date_values = sorted(metadata["date"].dropna().astype(str).unique().tolist())
    for date_value in date_values:
        safe_date = _sanitize_path_token(date_value)
        written[f"per_stimulus_same_vs_different_{safe_date}"] = _plot_focus_view_stimulus_gap(
            comparisons,
            metadata,
            focus_view,
            figures_dir / f"per_stimulus_same_vs_different__{safe_date}.png",
            date_value=date_value,
        )
    return written


def _plot_within_date_cross_individual_same_vs_different(comparisons: pd.DataFrame, path: Path) -> Path:
    valid = comparisons[comparisons["comparison_status"] == "ok"].copy()
    valid["comparison_label"] = valid["same_stimulus"].map({True: "same", False: "different"})
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=valid, x="view_name", y="distance", hue="comparison_label", showfliers=False)
    plt.xticks(rotation=15, ha="right")
    plt.xlabel("View")
    plt.ylabel("Distance")
    plt.title("Within-Date Cross-Individual Same vs Different Distances")
    return _save_figure(path)


def _plot_stimulus_availability_by_date(metadata: pd.DataFrame, path: Path) -> Path:
    matrix = _build_stimulus_availability_matrix(metadata)
    figure_width = max(8.0, 0.4 * max(len(matrix.columns), 1) + 2.5)
    figure_height = max(3.5, 0.55 * max(len(matrix.index), 1) + 1.6)
    plt.figure(figsize=(figure_width, figure_height))
    ax = plt.gca()
    if matrix.empty:
        ax.text(0.5, 0.5, "No date-by-stimulus metadata available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return _save_figure(path)

    annotate = len(matrix.columns) <= 24 and len(matrix.index) <= 12
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        annot=annotate,
        fmt=".0f",
        cbar_kws={"label": "Trial Count"},
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Stimulus")
    plt.ylabel("Date")
    plt.title("Stimulus Availability By Date")
    return _save_figure(path)


def _plot_per_date_loio_overview(summary: pd.DataFrame, path: Path) -> Path:
    pivot = (
        summary.pivot(index="source_date", columns="view_name", values="accuracy_mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    plt.figure(figsize=(8, max(3.5, 0.6 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.xlabel("View")
    plt.ylabel("Date")
    plt.title("Per-Date LOIO Accuracy Overview")
    return _save_figure(path)


def _build_stimulus_distance_matrix_frames(pair_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for view_name, group in pair_summary.groupby("view_name", sort=False, dropna=False):
        forward = group[["stimulus_left", "stimulus_right", "mean_distance"]].rename(
            columns={"stimulus_left": "stimulus_row", "stimulus_right": "stimulus_column"}
        )
        reverse = (
            group.loc[group["stimulus_left"] != group["stimulus_right"], ["stimulus_right", "stimulus_left", "mean_distance"]]
            .rename(columns={"stimulus_right": "stimulus_row", "stimulus_left": "stimulus_column"})
        )
        matrix_long = pd.concat([forward, reverse], ignore_index=True)
        stimuli = sorted(set(matrix_long["stimulus_row"]).union(matrix_long["stimulus_column"]))
        matrix = matrix_long.pivot(index="stimulus_row", columns="stimulus_column", values="mean_distance")
        matrix = matrix.reindex(index=stimuli, columns=stimuli)
        frames[str(view_name)] = matrix.reset_index()
    return frames


def _plot_stimulus_distance_matrix(matrix_frame: pd.DataFrame, view_name: str, path: Path) -> Path:
    heatmap_frame = matrix_frame.set_index("stimulus_row")
    size = max(6.5, 0.45 * len(heatmap_frame.columns) + 2.5)
    plt.figure(figsize=(size, size))
    sns.heatmap(heatmap_frame, cmap="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.xlabel("Stimulus")
    plt.ylabel("Stimulus")
    plt.title(f"Stimulus-by-Stimulus Distance Matrix ({view_name})")
    return _save_figure(path)


def build_run_summary(
    core_outputs: dict[str, pd.DataFrame],
    stats_outputs: dict[str, pd.DataFrame],
    written_paths: dict[str, Path],
    focus_view: str,
) -> dict[str, Any]:
    metadata_summary = core_outputs["metadata_summary"].iloc[0]
    final_summary = stats_outputs["final_summary"].sort_values("distance_gap", ascending=False, kind="stable")
    strongest = final_summary[["view_name", "distance_gap"]].head(2).to_dict(orient="records")
    focus_view_row = final_summary.loc[final_summary["view_name"] == focus_view].iloc[0]
    per_date_same_vs_different_figure_names = sorted(
        path.name for key, path in written_paths.items() if key.startswith("same_vs_different_by_date_")
    )
    pooled_per_stimulus_same_vs_different_figure_name = (
        written_paths["per_stimulus_same_vs_different_pooled"].name
        if "per_stimulus_same_vs_different_pooled" in written_paths
        else None
    )
    per_date_per_stimulus_same_vs_different_figure_names = sorted(
        path.name for key, path in written_paths.items() if key.startswith("per_stimulus_same_vs_different_") and key != "per_stimulus_same_vs_different_pooled"
    )
    return {
        "n_trials": int(metadata_summary["n_trials"]),
        "n_individuals": int(metadata_summary["n_individuals"]),
        "n_dates": int(metadata_summary["n_dates"]),
        "n_stimuli": int(metadata_summary["n_stimuli"]),
        "focus_view": focus_view,
        "focus_view_distance_gap": float(focus_view_row["distance_gap"]),
        "focus_view_loio_accuracy_mean": float(focus_view_row["loio_accuracy_mean"]),
        "focus_view_lodo_accuracy_mean": float(focus_view_row["lodo_accuracy_mean"]),
        "strongest_views": strongest,
        "final_summary_path": str(written_paths["final_summary"]),
        "figures_dir": str(written_paths["figures_dir"]),
        "qc_dir": str(written_paths["qc_dir"]),
        "per_date_same_vs_different_figure_names": per_date_same_vs_different_figure_names,
        "pooled_per_stimulus_same_vs_different_figure_name": pooled_per_stimulus_same_vs_different_figure_name,
        "per_date_per_stimulus_same_vs_different_figure_names": per_date_per_stimulus_same_vs_different_figure_names,
    }


def write_reliability_outputs(
    core_outputs: dict[str, pd.DataFrame],
    stats_outputs: dict[str, pd.DataFrame],
    output_root: str | Path,
    focus_view: str,
) -> dict[str, Path]:
    """Write reliability tables, QC outputs, figures, and summaries."""

    dirs = ensure_reliability_output_dirs(output_root)
    _remove_retired_reliability_figures(dirs["figures_dir"])
    written: dict[str, Path] = {
        "figures_dir": dirs["figures_dir"],
        "qc_dir": dirs["qc_dir"],
    }
    final_summary_with_focus = stats_outputs["final_summary"].copy()
    final_summary_with_focus["is_focus_view"] = final_summary_with_focus["view_name"].astype(str) == focus_view

    for view_name, group in core_outputs["comparisons"].groupby("view_name", sort=False, dropna=False):
        written[f"comparisons_{view_name}"] = write_parquet(
            _prepare_for_parquet(group.reset_index(drop=True)),
            dirs["tables_dir"] / f"comparisons__{view_name}.parquet",
        )

    table_names = (
        "same_vs_different_summary",
        "loio_trials",
        "loio_groups",
        "loio_summary",
        "lodo_trials",
        "lodo_groups",
        "lodo_summary",
        "split_half_results",
        "split_half_summary",
        "permutation_iterations",
        "permutation_summary",
        "bootstrap_iterations",
        "bootstrap_summary",
        "final_summary",
    )
    for table_name in table_names:
        table_frame = stats_outputs.get(table_name, core_outputs.get(table_name)).reset_index(drop=True)
        if table_name == "final_summary":
            table_frame = final_summary_with_focus.reset_index(drop=True)
        written[table_name] = write_parquet(
            _prepare_for_parquet(table_frame),
            dirs["tables_dir"] / f"{table_name}.parquet",
        )

    written["reliability_summary"] = write_parquet(
        _prepare_for_parquet(final_summary_with_focus.reset_index(drop=True)),
        dirs["tables_dir"] / "reliability_summary.parquet",
    )
    written["focus_view_summary"] = write_parquet(
        _prepare_for_parquet(
            final_summary_with_focus.loc[final_summary_with_focus["is_focus_view"]].reset_index(drop=True)
        ),
        dirs["tables_dir"] / "focus_view_summary.parquet",
    )
    written["leave_one_individual_out"] = write_parquet(
        _prepare_for_parquet(core_outputs["loio_groups"].reset_index(drop=True)),
        dirs["tables_dir"] / "leave_one_individual_out.parquet",
    )
    written["leave_one_date_out"] = write_parquet(
        _prepare_for_parquet(core_outputs["lodo_groups"].reset_index(drop=True)),
        dirs["tables_dir"] / "leave_one_date_out.parquet",
    )
    written["split_half_reliability"] = write_parquet(
        _prepare_for_parquet(core_outputs["split_half_results"].reset_index(drop=True)),
        dirs["tables_dir"] / "split_half_reliability.parquet",
    )
    written["permutation_null"] = write_parquet(
        _prepare_for_parquet(stats_outputs["permutation_iterations"].reset_index(drop=True)),
        dirs["tables_dir"] / "permutation_null.parquet",
    )
    written["grouped_bootstrap"] = write_parquet(
        _prepare_for_parquet(stats_outputs["bootstrap_iterations"].reset_index(drop=True)),
        dirs["tables_dir"] / "grouped_bootstrap.parquet",
    )
    written["within_date_cross_individual_comparisons"] = write_parquet(
        _prepare_for_parquet(core_outputs["within_date_cross_individual_comparisons"].reset_index(drop=True)),
        dirs["tables_dir"] / "within_date_cross_individual_comparisons.parquet",
    )
    written["within_date_cross_individual_same_vs_different_summary"] = write_parquet(
        _prepare_for_parquet(core_outputs["within_date_cross_individual_summary"].reset_index(drop=True)),
        dirs["tables_dir"] / "within_date_cross_individual_same_vs_different_summary.parquet",
    )
    written["per_date_loio_trials"] = write_parquet(
        _prepare_for_parquet(core_outputs["per_date_loio_trials"].reset_index(drop=True)),
        dirs["tables_dir"] / "per_date_loio_trials.parquet",
    )
    written["per_date_loio_groups"] = write_parquet(
        _prepare_for_parquet(core_outputs["per_date_loio_groups"].reset_index(drop=True)),
        dirs["tables_dir"] / "per_date_loio_groups.parquet",
    )
    written["per_date_loio_summary"] = write_parquet(
        _prepare_for_parquet(core_outputs["per_date_loio_summary"].reset_index(drop=True)),
        dirs["tables_dir"] / "per_date_loio_summary.parquet",
    )
    written["stimulus_distance_pairs"] = write_parquet(
        _prepare_for_parquet(core_outputs["stimulus_distance_pairs"].reset_index(drop=True)),
        dirs["tables_dir"] / "stimulus_distance_pairs.parquet",
    )
    stimulus_distance_matrices = _build_stimulus_distance_matrix_frames(core_outputs["stimulus_distance_pairs"])
    for view_name, matrix_frame in stimulus_distance_matrices.items():
        written[f"stimulus_distance_matrix_{view_name}"] = write_parquet(
            _prepare_for_parquet(matrix_frame.reset_index(drop=True)),
            dirs["tables_dir"] / f"stimulus_distance_matrix__{view_name}.parquet",
        )

    written["overlap_qc_summary"] = write_parquet(
        _prepare_for_parquet(core_outputs["overlap_qc_summary"].reset_index(drop=True)),
        dirs["qc_dir"] / "overlap_neuron_summary.parquet",
    )
    written["overlap_neuron_counts"] = write_parquet(
        _prepare_for_parquet(core_outputs["overlap_qc_summary"].reset_index(drop=True)),
        dirs["qc_dir"] / "overlap_neuron_counts.parquet",
    )
    written["excluded_comparisons"] = write_parquet(
        _prepare_for_parquet(
            core_outputs["comparisons"]
            .loc[core_outputs["comparisons"]["comparison_status"] != "ok"]
            .reset_index(drop=True)
        ),
        dirs["qc_dir"] / "excluded_comparisons.parquet",
    )
    excluded_holdout = pd.concat(
        [
            core_outputs["loio_trials"].loc[core_outputs["loio_trials"]["score_status"] != "scored"],
            core_outputs["lodo_trials"].loc[core_outputs["lodo_trials"]["score_status"] != "scored"],
        ],
        ignore_index=True,
    )
    written["excluded_holdout_trials"] = write_parquet(
        _prepare_for_parquet(excluded_holdout.reset_index(drop=True)),
        dirs["qc_dir"] / "excluded_holdout_trials.parquet",
    )

    written.update(
        _write_same_vs_different_variant_figures(
            core_outputs["comparisons"],
            final_summary_with_focus,
            focus_view,
            dirs["figures_dir"],
        )
    )
    written.update(
        _write_same_vs_different_per_date_figures(
            core_outputs["comparisons"],
            core_outputs["metadata"],
            focus_view,
            dirs["figures_dir"],
        )
    )
    written.update(
        _write_per_stimulus_same_vs_different_figures(
            core_outputs["comparisons"],
            core_outputs["metadata"],
            focus_view,
            dirs["figures_dir"],
        )
    )
    written["within_date_cross_individual_figure"] = _plot_within_date_cross_individual_same_vs_different(
        core_outputs["within_date_cross_individual_comparisons"],
        dirs["figures_dir"] / "within_date_cross_individual_same_vs_different.png",
    )
    written["per_date_loio_figure"] = _plot_per_date_loio_overview(
        core_outputs["per_date_loio_summary"],
        dirs["figures_dir"] / "per_date_loio_overview.png",
    )
    written["stimulus_availability_figure"] = _plot_stimulus_availability_by_date(
        core_outputs["metadata"],
        dirs["figures_dir"] / "overlap_neuron_qc_summary.png",
    )

    stats_outputs_with_focus = dict(stats_outputs)
    stats_outputs_with_focus["final_summary"] = final_summary_with_focus
    summary = build_run_summary(core_outputs, stats_outputs_with_focus, written, focus_view=focus_view)
    written["run_summary_json"] = write_json(summary, dirs["output_root"] / "run_summary.json")
    written["run_summary_md"] = _write_markdown_summary(summary, dirs["output_root"] / "run_summary.md")
    return written


ensure_stage1_output_dirs = ensure_reliability_output_dirs
write_stage1_outputs = write_reliability_outputs
