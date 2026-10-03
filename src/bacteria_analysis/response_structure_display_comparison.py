"""Redraw cached animal-prediction errors; does not load raw data or fit models."""

from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd


CELL_ORDER = (
    "ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
    "ASEL", "ASER", "AWCON", "AWCOFF",
)
MODELS = ("B", "M0", "M1", "M2")
COLORS = {"B": "#8B959D", "M0": "#786B9D", "M1": "#207F9C", "M2": "#BC692B"}
LABELS = {"B": "各菌株相同预测", "M0": "所有细胞一起变强弱",
          "M1": "各细胞改变方向/幅度", "M2": "各菌株独立时程"}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "text.color": "#24323D", "axes.labelcolor": "#24323D",
    "xtick.color": "#66717A", "ytick.color": "#66717A",
    "axes.edgecolor": "#C5CCD0", "axes.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "path",
}


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _panel_axes(title, subtitle):
    figure, axes = plt.subplots(4, 4, figsize=(13.8, 10.7))
    figure.subplots_adjust(left=0.061, right=0.975, bottom=0.075, top=0.80,
                           wspace=0.40, hspace=1.00)
    figure.text(0.061, 0.964, title, fontsize=21, fontweight=600, va="top")
    figure.text(0.061, 0.918, subtitle, fontsize=11, color="#66717A", va="top")
    for ax in axes.flat:
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, labelsize=9, pad=4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    for ax in axes.flat[len(CELL_ORDER):]:
        ax.set_visible(False)
    return figure, axes.ravel()


def _coverage_title(ax, cell, row):
    ax.text(0, 1.28, cell, transform=ax.transAxes, fontsize=13.5, fontweight=600)
    ax.text(1, 1.28, f"n = {int(row.n_animals)} / {int(row.n_conditions)}",
            transform=ax.transAxes, fontsize=9, ha="right", color="#737C83")


def _save(figure, directory, stem):
    files = {}
    for suffix in ("png", "pdf", "svg"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=230, facecolor="white")
        files[suffix] = str(path.resolve())
    plt.close(figure)
    return files


def _comparison_figure(rows, limits, window):
    figure, axes = _panel_axes(
        f"C  跨动物预测偏差  ·  {window.replace('s', ' 秒')}",
        "越靠左，预测偏差越小",
    )
    legend = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], markersize=7,
                     label=LABELS[m]) for m in MODELS]
    figure.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.055, 0.889),
                  ncol=4, frameon=False, handletextpad=0.5, columnspacing=1.8,
                  fontsize=11)
    for ax, cell in zip(axes, CELL_ORDER):
        row = rows.loc[cell]
        values = np.sqrt(row[[f"mse_{m}" for m in MODELS]].to_numpy(dtype=float))
        _coverage_title(ax, cell, row)
        ax.set_xlim(0, limits[cell])
        ax.set_ylim(-0.42, 3.42)
        ax.set_yticks([])
        ax.axvline(values[0], ymin=0.07, ymax=0.91, color=COLORS["B"],
                   linestyle=(0, (2, 3)), lw=0.85, alpha=0.75, zorder=1)
        # The direction of each segment faithfully keeps worsening predictions.
        ax.plot(values[1:3], [2, 1], color=COLORS["M1"], lw=1.7, alpha=0.7, zorder=2)
        ax.plot(values[2:4], [1, 0], color=COLORS["M2"], lw=1.7, alpha=0.7, zorder=2)
        for value, y, model in zip(values, [3, 2, 1, 0], MODELS):
            ax.scatter(value, y, color=COLORS[model], s=41, zorder=3,
                       edgecolor="white", linewidth=0.6)
            ax.annotate(f"{value:.4f}", (value, y), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        color=COLORS[model], fontsize=9.3)
    figure.text(0.52, 0.030, "预测偏差（ΔF/F₀）", ha="center", fontsize=12)
    figure.text(0.345, 0.159, "n = 动物数 / 菌株×采集块条件数；各细胞横轴不同",
                fontsize=10, color="#737C83")
    return figure


def _gain_figure(rows, animals, limits, window):
    figure, axes = _panel_axes(
        f"C 补充  逐动物预测增益  ·  {window.replace('s', ' 秒')}",
        "零线右侧表示预测能力增加    ·    每个细胞独立横轴；所有负增益保留",
    )
    legend = [
        Line2D([], [], marker="D", linestyle="none", color=COLORS["M1"], markersize=6,
               label="细胞权重的收益"),
        Line2D([], [], marker="D", linestyle="none", color=COLORS["M2"], markersize=6,
               label="额外时程的收益"),
        Line2D([], [], marker="o", linestyle="none", color="#B7BDC2", markersize=4,
               label="逐动物"),
    ]
    figure.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.055, 0.889),
                  ncol=3, frameon=False, handletextpad=0.6, columnspacing=2.2, fontsize=11)
    for ax, cell in zip(axes, CELL_ORDER):
        row = rows.loc[cell]
        _coverage_title(ax, cell, row)
        ax.set_xlim(-limits[cell], limits[cell])
        ax.set_ylim(-0.42, 1.42)
        ax.set_yticks([])
        ax.axvline(0, color="#8B959D", lw=0.85, zorder=1)
        selected = animals.loc[animals.cell.eq(cell)].sort_values("animal_id")
        for y, key, model in [(1, "delta_combination", "M1"), (0, "delta_timing", "M2")]:
            values = selected[key].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            # A deterministic jitter only separates markers; it is not another variable.
            jitter = np.sin(np.arange(len(values)) * 2.399963) * 0.15
            ax.scatter(values, y + jitter, color="#AAB2B9", alpha=0.48, s=11,
                       linewidth=0, zorder=2)
            ax.scatter(float(row[key]), y, marker="D", s=54, color=COLORS[model],
                       edgecolor="white", linewidth=0.8, zorder=3)
    figure.text(0.52, 0.030, "预测增益：前一模型 MSE − 后一模型 MSE（ΔF/F₀）²",
                ha="center", fontsize=12)
    figure.text(0.345, 0.176, "菱形：原加权汇总    灰点：逐动物", fontsize=11, color="#53616C")
    figure.text(0.345, 0.137, "n = 动物数 / 菌株×采集块条件数", fontsize=10, color="#737C83")
    return figure


def plot_comparison(source_dir, output_dir):
    """Write separate C figures and exact cached supporting data for both windows.

    ``source_dir`` may be the analysis report directory or its ``tables`` folder.
    RMSE is sqrt(the existing hierarchical weighted MSE), never a mean of animal
    RMSEs. No predictions, scientific weights, fits, or assessments are changed.
    Returns a dictionary of output paths, source hashes, and concise captions.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not (source_dir / "cell_summary.csv").is_file():
        source_dir = source_dir / "tables"
    summary_path = source_dir / "cell_summary.csv"
    animal_path = source_dir / "animal_cell_errors.csv"
    summary, animals = pd.read_csv(summary_path), pd.read_csv(animal_path)
    windows = ("0-40s", "0-25s")
    required = {"window", "cell", "n_animals", "n_conditions", "delta_combination", "delta_timing"}
    required.update(f"mse_{m}" for m in MODELS)
    if not required.issubset(summary.columns):
        raise ValueError(f"Cached summary lacks required columns: {sorted(required - set(summary.columns))}")
    if not {"window", "cell", "animal_id", "delta_combination", "delta_timing"}.issubset(animals.columns):
        raise ValueError("Cached animal table lacks the required identifiers or paired gains")
    for window in windows:
        part = summary.loc[summary.window.eq(window)]
        if len(part) != len(CELL_ORDER) or set(part.cell) != set(CELL_ORDER):
            raise ValueError(f"Expected one summary per neuron class in {window}")
        if not np.isfinite(part[[f"mse_{m}" for m in MODELS]].to_numpy()).all():
            raise ValueError(f"Nonfinite cached model error in {window}; cannot silently omit it")
    if (summary[[f"mse_{m}" for m in MODELS]] < 0).any().any():
        raise ValueError("A cached MSE is negative")
    # A given cell uses the same original-unit scale in both time-range reports.
    error_limits = {
        cell: 1.34 * np.sqrt(summary.loc[summary.cell.eq(cell), [f"mse_{m}" for m in MODELS]]
                             .to_numpy(dtype=float)).max()
        for cell in CELL_ORDER
    }
    gain_limits = {}
    for cell in CELL_ORDER:
        values = animals.loc[animals.cell.eq(cell), ["delta_combination", "delta_timing"]].to_numpy()
        center = summary.loc[summary.cell.eq(cell), ["delta_combination", "delta_timing"]].to_numpy()
        gain_limits[cell] = max(1e-8, 1.12 * np.nanmax(np.abs(np.r_[values.ravel(), center.ravel()])))
        error_limits[cell] = max(1e-8, error_limits[cell])
    figure_dir, data_dir = output_dir / "figures", output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "source_sha256": {"cell_summary.csv": _hash(summary_path),
                           "animal_cell_errors.csv": _hash(animal_path)},
        "windows": {},
    }
    with plt.rc_context(STYLE):
        for window in windows:
            rows = summary.loc[summary.window.eq(window)].set_index("cell").loc[list(CELL_ORDER)].copy()
            for model in MODELS:
                rows[f"rmse_{model}"] = np.sqrt(rows[f"mse_{model}"])
            selected_animals = animals.loc[animals.window.eq(window)].copy()
            comparison_stem, gain_stem = f"C_model_comparison_{window}", f"C_gain_animals_{window}"
            files = _save(_comparison_figure(rows, error_limits, window), figure_dir, comparison_stem)
            gain_files = _save(_gain_figure(rows, selected_animals, gain_limits, window), figure_dir, gain_stem)
            data_path = data_dir / f"{comparison_stem}.csv"
            rows.reset_index().to_csv(data_path, index=False)
            animal_data_path = data_dir / f"{gain_stem}.csv"
            selected_animals.to_csv(animal_data_path, index=False)
            caption = (
                f"C，{window}：四种模型在整动物留出预测中的误差。图中“预测偏差”为 RMSE；每个点为原分析加权 MSE 的平方根，"
                "不是逐动物 RMSE 的平均；加权规则、测试条目与原分析完全一致。各细胞横轴不同，"
                "同一细胞在两个时间范围的横轴相同。灰虚线为所有条件共用训练平均曲线的误差；"
                "蓝线连接所有细胞一起变强弱与各细胞改变方向/幅度，橙线连接各细胞改变方向/幅度与各菌株独立时程。"
                "左移为预测能力提高，右移为降低。n 为该细胞可评分动物数/菌株×采集块条件数。"
                "正式预测增益保持原 MSE 单位并完整保存在对应 CSV。\n"
                "C 补充：灰点为原表中的逐动物配对 MSE 差；菱形为原分析的层级加权汇总，"
                "并非灰点的简单平均。上排为群体统一增益减去细胞各自权重，下排为细胞各自权重"
                "减去独立时间曲线。保留所有负增益；无置信区间或独立重复数推断。"
            )
            caption_path = output_dir / f"{comparison_stem}_caption.txt"
            caption_path.write_text(caption + "\n", encoding="utf-8")
            result["windows"][window] = {
                "main": files, "animal_gains": gain_files,
                "summary_data": str(data_path.resolve()), "animal_data": str(animal_data_path.resolve()),
                "caption_file": str(caption_path.resolve()), "caption": caption,
                "n_cells": len(rows), "n_animal_cell_rows": len(selected_animals),
            }
    return result
