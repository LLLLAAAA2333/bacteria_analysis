"""Display the cached 0–40 s cell-response combinations without model refitting.

Call ``plot_combinations(source_dir, output_dir)`` from a Notebook.  Seven
coefficient columns are contextualized by their original RMS-one templates;
the other six cells retain their eight observed response-window means.  The
four pages retain every strain/date condition in identifier/date order.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


COEFFICIENT_CELLS = ["AWCON", "AWB", "AWA", "ASH", "ADF", "ASK", "ASJ"]
ORIGINAL_WINDOW_CELLS = ["AWCOFF", "ASEL", "ADL", "ASI", "ASER", "ASG"]
TIME_MIDPOINTS = np.arange(2.5, 40, 5)
COLOR_LIMIT = .6
ROWS_PER_PAGE = 28


def _load_display_tables(source_dir):
    """Read saved results and aggregate only the saved animal-level observations."""
    tables = source_dir / "tables"
    coefficients = pd.read_csv(tables / "coefficients.csv", dtype={"strain": str, "block": str})
    coefficients = coefficients.loc[coefficients.window.eq("0-40s") & coefficients.cell.isin(COEFFICIENT_CELLS)].copy()
    templates = pd.read_csv(tables / "templates.csv")
    templates = templates.loc[templates.window.eq("0-40s") & templates.fit_type.eq("full")
                              & templates.cell.isin(COEFFICIENT_CELLS)].copy()
    observations = pd.read_parquet(source_dir / "data" / "observations.parquet")
    observations = observations.rename(columns={"sample_id": "strain", "neuron_class": "cell"})
    observations["strain"] = observations.strain.astype(str)
    observations["block"] = observations.block.astype(str)
    observations = observations.loc[observations.bin_index.between(0, 7)]
    keys = ["strain", "block", "animal_id", "cell", "bin_index"]
    if observations.duplicated(keys).any():
        raise ValueError("Expected one saved response per animal, condition, cell and window.")
    conditions = observations[["strain", "block"]].drop_duplicates().sort_values(["strain", "block"]).reset_index(drop=True)
    conditions["condition_index"] = np.arange(len(conditions))
    conditions["page"] = conditions.condition_index // ROWS_PER_PAGE + 1
    conditions["row_on_page"] = conditions.condition_index % ROWS_PER_PAGE
    conditions["condition"] = conditions.strain + " | " + conditions.block

    coef_grid = conditions.merge(pd.DataFrame({"cell": COEFFICIENT_CELLS}), how="cross")
    coefficients = coef_grid.merge(coefficients, on=["strain", "block", "cell"], how="left", validate="one_to_one")
    coefficients["has_extra_timing"] = coefficients.cell.ne("AWCON")
    coefficients["display_clipped"] = coefficients.coefficient.abs().gt(COLOR_LIMIT)
    coefficients["low_coverage"] = coefficients.n_animals_min.lt(3) & coefficients.coefficient.notna()
    coefficients["display_value"] = coefficients.coefficient.clip(-COLOR_LIMIT, COLOR_LIMIT)
    coefficients["coefficient_column"] = coefficients.cell.map(dict(zip(COEFFICIENT_CELLS, range(len(COEFFICIENT_CELLS)))))

    observed = observations.loc[observations.cell.isin(ORIGINAL_WINDOW_CELLS) & observations.response.notna()]
    raw = observed.groupby(["strain", "block", "cell", "bin_index"], observed=True).agg(
        response=("response", "mean"), n_animals=("animal_id", "nunique")).reset_index()
    raw_grid = conditions.merge(pd.DataFrame({"cell": ORIGINAL_WINDOW_CELLS}), how="cross").merge(
        pd.DataFrame({"bin_index": np.arange(8)}), how="cross")
    raw = raw_grid.merge(raw, on=["strain", "block", "cell", "bin_index"], how="left", validate="one_to_one")
    raw["n_animals"] = raw.n_animals.fillna(0).astype(int)  # Counts only; responses stay missing.
    raw["time_start"] = raw.bin_index * 5
    raw["time_end"] = raw.time_start + 5
    raw["time_s"] = raw.time_start + 2.5
    raw["display_clipped"] = raw.response.abs().gt(COLOR_LIMIT)
    raw["low_coverage"] = raw.n_animals.between(1, 2) & raw.response.notna()
    raw["display_value"] = raw.response.clip(-COLOR_LIMIT, COLOR_LIMIT)
    raw["cell_group_column"] = raw.cell.map(dict(zip(ORIGINAL_WINDOW_CELLS, range(len(ORIGINAL_WINDOW_CELLS)))))
    assessments = pd.read_csv(tables / "cell_assessments.csv")
    assessments = assessments.loc[assessments.window.eq("0-40s")].copy()
    return conditions, coefficients, templates, raw, assessments


def _colormap(colors):
    cmap = LinearSegmentedColormap.from_list("response_display", colors)
    cmap.set_bad("#D5D8DA")
    return cmap


def _add_hatching(ax, mask, x_edges):
    """Hatch low-support entries without substituting a response or coefficient."""
    for row, col in np.argwhere(np.asarray(mask, bool)):
        ax.add_patch(Rectangle((x_edges[col], row - .5), x_edges[col + 1] - x_edges[col], 1,
                               fill=False, hatch="///", edgecolor="#666666", linewidth=0, zorder=3))


def _add_overflow(ax, mask, x_midpoints):
    rows, cols = np.where(np.asarray(mask, bool))
    ax.scatter(np.asarray(x_midpoints)[cols], rows, s=11, c="#252A2D", edgecolors="white",
               linewidths=.45, zorder=5)


def _frame_matrix(ax, n_rows):
    ax.set_ylim(n_rows - .5, -.5)
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for y in np.arange(n_rows - 1) + .5:
        ax.axhline(y, color="white", lw=.24, zorder=2)


def _draw_page(page_conditions, coefficients, templates, raw, page_number, n_pages):
    coef_cmap = _colormap(["#75559A", "#FCFCFC", "#087F83"])
    raw_cmap = _colormap(["#2679A4", "#FCFCFC", "#AC6E40"])
    norm = Normalize(-COLOR_LIMIT, COLOR_LIMIT, clip=True)
    fig = plt.figure(figsize=(16.5, 10.2), facecolor="white")
    coef_left, coef_width = .165, .295
    raw_left, raw_right, gap = .492, .98, .009
    raw_width = (raw_right - raw_left - 5 * gap) / 6
    bottom, height, template_bottom, template_height = .18, .505, .748, .12
    n_rows = len(page_conditions)
    row_order = page_conditions.condition_index.tolist()
    fig.text(.027, .943, "B  响应组合", fontsize=20, fontweight=600, fontfamily="PingFang SC", color="#202B33")
    fig.text(.978, .947, f"0–40 s   ·   {page_number}/{n_pages}", ha="right", fontsize=10, color="#65717A")
    fig.text(coef_left, .906, "细胞权重", fontsize=13, fontweight=600, fontfamily="PingFang SC", color="#27333B")
    fig.text(raw_left, .906, "保留原时程", fontsize=13, fontweight=600, fontfamily="PingFang SC", color="#27333B")
    fig.text(coef_left, .727, "模板 · RMS = 1", fontsize=8, color="#68737C")
    fig.text(raw_left, .866, "8 个 5 秒窗", fontsize=9, color="#68737C")

    for j, cell in enumerate(COEFFICIENT_CELLS):
        ax = fig.add_axes([coef_left + j * coef_width / 7, template_bottom, coef_width / 7, template_height])
        profile = templates.loc[templates.cell.eq(cell)].sort_values("time_s")
        ax.axvspan(0, 10, color="#F0F1ED", zorder=0)
        ax.axhline(0, color="#CBD1D4", lw=.5)
        ax.plot(profile.time_s, profile.template, color="#207F9C", lw=1.4)
        ax.set(xlim=(0, 40), ylim=(-3, 3), xticks=[], yticks=[])
        ax.tick_params(axis="x", length=0, labelsize=6.5, colors="#778189")
        ax.set_title(cell + ("†" if j else ""), fontsize=9, pad=5, color="#26343C")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(coef_left + coef_width, .727, "0–40 s", fontsize=8, color="#68737C", ha="right")

    coef = coefficients.loc[coefficients.condition_index.isin(row_order)]
    matrix = coef.pivot(index="condition_index", columns="cell", values="coefficient").reindex(index=row_order, columns=COEFFICIENT_CELLS)
    low = coef.pivot(index="condition_index", columns="cell", values="low_coverage").reindex_like(matrix)
    overflow = coef.pivot(index="condition_index", columns="cell", values="display_clipped").reindex_like(matrix)
    ax = fig.add_axes([coef_left, bottom, coef_width, height])
    coefficient_image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), aspect="auto", interpolation="none", norm=norm, cmap=coef_cmap)
    _frame_matrix(ax, n_rows)
    ax.set_xticks([])
    _add_hatching(ax, low.to_numpy(bool), np.arange(8) - .5)
    _add_overflow(ax, overflow.to_numpy(bool), np.arange(7))
    for j in np.arange(6) + .5:
        ax.axvline(j, color="white", lw=.7, zorder=2)
    for i, record in enumerate(page_conditions.itertuples(index=False)):
        date = str(record.block)
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
        ax.text(-.67, i, f"{record.strain}  ·  {date}", ha="right", va="center", fontsize=8.1,
                color="#394650", clip_on=False)
    fig.text(coef_left - .007, .698, "菌株 · 日期", ha="right", fontsize=8, color="#75818A")

    for j, cell in enumerate(ORIGINAL_WINDOW_CELLS):
        entries = raw.loc[raw.condition_index.isin(row_order) & raw.cell.eq(cell)]
        response = entries.pivot(index="condition_index", columns="bin_index", values="response").reindex(index=row_order, columns=range(8))
        low = entries.pivot(index="condition_index", columns="bin_index", values="low_coverage").reindex_like(response)
        overflow = entries.pivot(index="condition_index", columns="bin_index", values="display_clipped").reindex_like(response)
        ax = fig.add_axes([raw_left + j * (raw_width + gap), bottom, raw_width, height])
        raw_image = ax.imshow(np.ma.masked_invalid(response.to_numpy(float)), aspect="auto", interpolation="none", norm=norm,
                              cmap=raw_cmap, extent=(0, 40, n_rows - .5, -.5))
        _frame_matrix(ax, n_rows)
        ax.set_xticks([0, 10, 40], ["0", "10", "40"])
        ax.tick_params(labelsize=7, colors="#75818A")
        ax.axvline(10, color="#4D5D66", lw=.55, ls=(0, (2, 2)), alpha=.7)
        ax.set_title(cell, fontsize=9, pad=9, color="#26343C")
        _add_hatching(ax, low.to_numpy(bool), np.arange(9) * 5)
        _add_overflow(ax, overflow.to_numpy(bool), TIME_MIDPOINTS)
    fig.text(raw_right, bottom - .036, "时间 / s", ha="right", fontsize=8, color="#75818A")

    coefficient_bar = fig.add_axes([coef_left, .101, coef_width, .012])
    cbar = fig.colorbar(coefficient_image, cax=coefficient_bar, orientation="horizontal", extend="both", ticks=[-.6, 0, .6])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8, pad=3)
    cbar.set_label("系数 a  /  ΔF/F₀", fontsize=9, labelpad=3)
    response_bar = fig.add_axes([raw_left, .101, raw_right - raw_left, .012])
    cbar = fig.colorbar(raw_image, cax=response_bar, orientation="horizontal", extend="both", ticks=[-.6, 0, .6])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8, pad=3)
    cbar.set_label("动物均值  /  ΔF/F₀", fontsize=9, labelpad=3)
    fig.text(.027, .025, "±：沿 / 翻转模板     † 仍有时程差异     · 实心点：超出色标     ///：有效动物少于 3     灰色：缺失",
             fontsize=8, color="#6D7880")
    return fig


def plot_combinations(source_dir, output_dir):
    """Export four pages of original-unit coefficients and window-response data.

    Files under ``source_dir`` are read only.  Display saturation at ±0.6 is
    explicit and never changes the unbounded numerical values exported to CSV.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    # A direct call must retain the same review guard as the Notebook entry point.
    from .response_structure_display import validate_evidence
    validate_evidence(source_dir)
    conditions, coefficients, templates, raw, assessments = _load_display_tables(source_dir)
    figures, figure_data = output_dir / "figures", output_dir / "figure_data"
    figures.mkdir(parents=True, exist_ok=True)
    figure_data.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(figure_data / "B_coefficients.csv", index=False)
    templates.to_csv(figure_data / "B_templates.csv", index=False)
    raw.to_csv(figure_data / "B_original_windows.csv", index=False)
    conditions.to_csv(figure_data / "B_condition_order.csv", index=False)
    assessments.to_csv(figure_data / "B_source_assessments.csv", index=False)

    files = ["tables/coefficients.csv", "tables/templates.csv", "tables/cell_assessments.csv", "data/observations.parquet"]
    fingerprints = {}
    for name in files:
        with (source_dir / name).open("rb") as handle:
            fingerprints[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    parameters = {
        "window": "0-40s", "source_dir": str(source_dir.resolve()), "source_sha256": fingerprints,
        "coefficient_cells": COEFFICIENT_CELLS, "original_window_cells": ORIGINAL_WINDOW_CELLS,
        "coefficient_color_limits": [-COLOR_LIMIT, COLOR_LIMIT], "response_color_limits": [-COLOR_LIMIT, COLOR_LIMIT],
        "color_limit_rule": "Shared original-unit ±0.6 display domain selected for readability after inspecting all displayed values; no cell-specific scaling.",
        "n_conditions": len(conditions), "conditions_per_page": ROWS_PER_PAGE,
        "condition_order": "strain identifier ascending, then acquisition date ascending; no clustering or response sorting",
        "selection_rule": "Every recorded condition is displayed; no selected-condition main panel.",
        "coefficient_clipped_entries": int(coefficients.display_clipped.sum()),
        "raw_window_clipped_entries": int(raw.display_clipped.sum()),
        "coefficient_low_coverage_entries": int(coefficients.low_coverage.sum()),
        "raw_window_low_coverage_entries": int(raw.low_coverage.sum()),
        "low_coverage_marker": "Thin diagonal hatching; valid animal count <3; counts stay in exact data tables",
        "overflow_marker": "Dark point with white edge; absolute unbounded value exceeds 0.6",
        "missing_color": "#D5D8DA", "template_color": "#207F9C",
        "coefficient_colors": ["#75559A", "#FCFCFC", "#087F83"],
        "raw_window_colors": ["#2679A4", "#FCFCFC", "#AC6E40"],
    }
    (figure_data / "B_display_parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    awcon_low = int(coefficients.loc[coefficients.cell.eq("AWCON"), "low_coverage"].sum())
    caption = (
        "# B｜响应组合\n\n"
        f"展示当前刺激协议下 0–40 秒的钙响应表型。全部 {len(conditions)} 个菌株×采集日期条件按菌株编号、日期固定排序，每页 {ROWS_PER_PAGE} 个条件；同菌株跨日期分别保留。没有按响应排序或聚类，也没有挑选代表条件。\n\n"
        "左侧七列是已保存的全数据 M1 系数，来自各细胞未经额外中心化的单模板拟合。其上方模板保持原有 RMS=1、绝对值最大窗取正的约定，全部模板共用 −3 至 3 的纵轴。AWCON 可用单模板近似；AWB、AWA、ASH、ADF、ASK、ASJ 标 †，表示 M1 保留菌株差异，但仍有额外时程差异。展示这些系数用于比较可预测的响应组合，不把 † 细胞重新判为固定时程。ASJ 的这一标记针对主窗口 0–40 秒。\n\n"
        "右侧六类细胞保留原来的八个 5 秒窗，颜色为该条件各有效动物的原 ΔF/F₀ 均值。刺激为 0–10 秒；原分窗中的虚线及模板中的浅色背景标明刺激终点或范围。所有曲线、系数均复用已保存数据，没有重新拟合模型。\n\n"
        "两种颜色条分别表示有符号系数和原分窗响应，均使用跨细胞、跨页统一的原单位范围 −0.6 至 0.6；紫—白—青用于系数，蓝—白—棕用于分窗响应。范围以外仅在显示时饱和，并用实心点标出，颜色条两端也标有延伸。数据未缩放、未截断，精确导出同时保留原值与 display_value。系数的正负只表示相对于对应模板的缩放与整体翻转，不能直接读作激活或抑制。\n\n"
        f"斜线表示有效动物数小于 3：系数取该条件、该细胞各窗中的最小动物数，原分窗按每个窗记录。低覆盖值仅作描述，不能视为已经验证的跨动物预测；特别是 AWCON 的 {awcon_low} 个低覆盖条件仍被保留。灰色为缺失，响应未填零。有效动物数在导出表中逐条件、细胞与时间窗保存。\n\n"
        f"显示饱和条目：系数 {parameters['coefficient_clipped_entries']} / {coefficients.coefficient.notna().sum()}；原分窗 {parameters['raw_window_clipped_entries']} / {raw.response.notna().sum()}。\n"
    )
    (output_dir / "B_caption.md").write_text(caption, encoding="utf-8")
    n_pages = (len(conditions) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    paths = []
    style = {"font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
             "font.size": 9, "axes.unicode_minus": False, "pdf.fonttype": 42,
             "svg.fonttype": "none", "savefig.facecolor": "white", "hatch.linewidth": .25}
    with plt.rc_context(style), PdfPages(figures / "B_response_combinations.pdf") as pdf:
        for page in range(1, n_pages + 1):
            selected = conditions.loc[conditions.page.eq(page)]
            fig = _draw_page(selected, coefficients, templates, raw, page, n_pages)
            stem = figures / f"B_response_combinations_page{page}"
            fig.savefig(stem.with_suffix(".png"), dpi=200)
            fig.savefig(stem.with_suffix(".svg"))
            pdf.savefig(fig)
            plt.close(fig)
            paths.extend([str(stem.with_suffix(".png")), str(stem.with_suffix(".svg"))])
    return {"pdf": str(figures / "B_response_combinations.pdf"), "pages": paths,
            "figure_data": str(figure_data), "caption": str(output_dir / "B_caption.md")}
