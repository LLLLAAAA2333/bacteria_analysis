"""Scientific figures for the animal-held-out shared-time-template analysis.

``plot_results(output_dir)`` reads the saved analysis tables and writes three
figure groups under ``figures/``.  Plotting never refits a model.  Gray heatmap
entries are missing, not zero; signed template coefficients are not activity
labels.  All response curves remain in the original delta-F/F0 units.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


MODEL_COLORS = {"B": "#808080", "M0": "#C38D32", "M1": "#2166AC", "M2": "#A33B57"}
CELL_ORDER = ["ASK", "ADL", "ASI", "AWA", "AWB", "ASG", "ADF", "ASH", "ASJ",
              "ASEL", "ASER", "AWCON", "AWCOFF"]
MODEL_ORDER = ["B", "M0", "M1", "M2"]
STATUS_LABELS = {
    "useful": "Single template useful",
    "timing": "Timing variation",
    "uncertain": "Currently undetermined",
}


def _status_label(record):
    if record.get("status") == "timing" and record.get("scope") == "local":
        return "Local timing evidence"
    return STATUS_LABELS.get(record.get("status"), str(record.get("status", "")))


def _style():
    return plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def _cell_order(values):
    values = set(map(str, values))
    return [x for x in CELL_ORDER if x in values] + sorted(values - set(CELL_ORDER))


def _conditions(frame):
    out = frame.copy()
    out["condition"] = out["strain"].astype(str) + " | " + out["block"].astype(str)
    return out


def _finite_limit(values):
    values = np.asarray(values, dtype=float)
    good = np.abs(values[np.isfinite(values)])
    return float(good.max()) if len(good) and good.max() > 0 else 1.0


def _heatmap(ax, matrix, times=None, vlim=None):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E0E0E0")
    vlim = _finite_limit(matrix.to_numpy()) if vlim is None else vlim
    im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), aspect="auto",
                   interpolation="none", cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns if times is None else times)
    ax.tick_params(axis="both", length=0)
    return im


def _time_axis(ax, max_time):
    ax.axhline(0, color="#B5B5B5", lw=.5, zorder=0)
    ax.axvspan(0, 10, color="#ECE9DC", alpha=.75, zorder=0)
    ax.set_xlim(0, max_time)
    ax.set_xticks(sorted(set(np.arange(0, max_time + 1, 10).tolist() + [max_time])))


def _save(fig, pdf, folder, stem, dpi=160):
    pdf.savefig(fig)
    fig.savefig(folder / (stem + ".png"), dpi=dpi)
    plt.close(fig)


def _caption(fig, text):
    fig.text(.02, .012, text, fontsize=7, ha="left", va="bottom", wrap=True)


def _animal_curve(frame, value, times):
    """Animal rows by time; no imputation and no averaging across conditions."""
    return frame.pivot_table(index="animal_id", columns="time_s", values=value,
                             aggfunc="first").reindex(columns=times)


def _paired_animal_gains(predictions):
    """Preserve animal identity; macro-average windows, blocks, then strains.

    These points describe each held-out animal's available conditions. They are
    neither independent validation folds nor the overall weighted estimator.
    """
    valid = predictions.loc[predictions["eligible"].astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame(columns=["window", "animal_id", "cell", "delta_combination", "delta_timing"])
    for model in MODEL_ORDER:
        valid["se_" + model] = (valid["actual"] - valid["pred_" + model]) ** 2
    valid["delta_combination"] = valid["se_M0"] - valid["se_M1"]
    valid["delta_timing"] = valid["se_M1"] - valid["se_M2"]
    columns = ["delta_combination", "delta_timing"]
    keys = ["window", "animal_id", "cell", "strain", "block"]
    out = valid.groupby(keys, observed=True)[columns].mean().reset_index()
    out = out.groupby(keys[:-1], observed=True)[columns].mean().reset_index()
    return out.groupby(keys[:3], observed=True)[columns].mean().reset_index()


def _representatives(predictions):
    """Select examples from held-out data by a recorded, deterministic rule.

    Typical: median M1 MSE among conditions with positive B-minus-M1 gain.
    Deviation: largest mean M1-minus-M2 gain where M2 beats B and both gains
    are positive in the same at least two animals and a majority of animals.
    Uncertain: lowest coverage among unscored conditions, or least favorable
    B-minus-M2 gain when all conditions have eligible observations.
    Example selection is descriptive; no inferential test follows selection.
    """
    p = _conditions(predictions)
    rows = []
    for (window, cell), group in p.groupby(["window", "cell"], sort=False):
        summaries = []
        for condition, subset in group.groupby("condition", sort=True):
            score = subset.loc[subset["eligible"].astype(bool)]
            item = {"window": window, "cell": cell, "condition": condition,
                    "strain": subset["strain"].iloc[0], "block": subset["block"].iloc[0],
                    "n_animals": subset.loc[subset["actual"].notna(), "animal_id"].nunique(),
                    "n_scored_animals": score["animal_id"].nunique()}
            for model in MODEL_ORDER:
                mse = ((score["actual"] - score["pred_" + model]) ** 2)
                item["mse_" + model] = mse.groupby(score["time_s"]).mean().mean()
            by_animal = score.copy()
            by_animal["gain"] = ((score.actual - score.pred_M1) ** 2
                                 - (score.actual - score.pred_M2) ** 2)
            by_animal["gain_B_M2"] = ((score.actual - score.pred_B) ** 2
                                      - (score.actual - score.pred_M2) ** 2)
            gains = by_animal.groupby("animal_id")[["gain", "gain_B_M2"]].mean()
            joint = (gains.gain > 0) & (gains.gain_B_M2 > 0)
            item["n_timing_positive"] = int((gains.gain > 0).sum())
            item["fraction_timing_positive"] = float((gains.gain > 0).mean()) if len(gains) else np.nan
            item["n_joint_positive"] = int(joint.sum())
            item["fraction_joint_positive"] = float(joint.mean()) if len(gains) else np.nan
            summaries.append(item)
        stats = pd.DataFrame(summaries)
        if stats.empty:
            continue
        stats["gain_M1_vs_B"] = stats.mse_B - stats.mse_M1
        stats["gain_M2_vs_B"] = stats.mse_B - stats.mse_M2
        stats["gain_timing"] = stats.mse_M1 - stats.mse_M2
        typical = stats.loc[stats.gain_M1_vs_B > 0].sort_values(["mse_M1", "condition"])
        deviant = stats.loc[(stats.gain_M2_vs_B > 0) & (stats.gain_timing > 0)
                            & (stats.n_joint_positive >= 2)
                            & (stats.fraction_joint_positive > .5)]
        missing = stats.loc[stats.n_scored_animals < 3].sort_values(["n_animals", "condition"])
        choices = {
            "Typical fit": typical.iloc[len(typical) // 2] if len(typical) else None,
            "Timing deviation": deviant.sort_values(["gain_timing", "condition"], ascending=[False, True]).iloc[0]
            if len(deviant) else None,
            "Insufficient evidence": missing.iloc[0] if len(missing) else
            (stats.loc[stats.gain_M2_vs_B <= 0].sort_values(["gain_M2_vs_B", "condition"]).iloc[0]
             if (stats.gain_M2_vs_B <= 0).any() else None),
        }
        for kind, row in choices.items():
            if row is not None:
                record = row.to_dict()
                record["example_type"] = kind
                candidate = group.loc[group.condition == row.condition].copy()
                scored = candidate.loc[candidate.eligible.astype(bool)].copy()
                if len(scored):
                    scored["animal_mse"] = (scored.actual - scored.pred_M1) ** 2
                    scored["animal_timing_gain"] = scored.animal_mse - (scored.actual - scored.pred_M2) ** 2
                    scored["animal_B_M2_gain"] = ((scored.actual - scored.pred_B) ** 2
                                                   - (scored.actual - scored.pred_M2) ** 2)
                    animals = scored.groupby("animal_id")[["animal_mse", "animal_timing_gain", "animal_B_M2_gain"]].mean()
                    if kind == "Timing deviation":
                        animals = animals.loc[(animals.animal_timing_gain > 0) & (animals.animal_B_M2_gain > 0)].sort_values("animal_timing_gain", kind="stable")
                        animal_rule = "Median timing gain among animals jointly benefiting from M2 vs M1 and B"
                    else:
                        animals = animals.sort_values("animal_mse", kind="stable")
                        animal_rule = "Median M1 animal prediction error"
                else:
                    candidate["response_squared"] = candidate.actual ** 2
                    animals = candidate.groupby("animal_id").response_squared.mean().sort_values(kind="stable").to_frame()
                    animal_rule = "Median observed animal RMS; no eligible prediction"
                record["representative_animal"] = animals.index[len(animals) // 2]
                record["animal_selection_rule"] = animal_rule
                rows.append(record)
    return pd.DataFrame(rows)


def _template_pages(templates, summary, folder, figure_data, pdf):
    display = []
    for window in ["0-40s", "0-25s"]:
        selected = templates.loc[templates.window == window]
        cells = _cell_order(selected.cell)
        if not cells:
            continue
        fig, axes = plt.subplots(7, 2, figsize=(11, 13), squeeze=False)
        for ax, cell in zip(axes.flat, cells):
            sub = selected.loc[selected.cell == cell]
            full = sub.loc[sub.fit_type == "full"].sort_values("time_s")
            if "identified" in full.columns and not full.identified.isin([True, "True"]).all():
                ax.text(.5, .5, "Template is not identifiable", transform=ax.transAxes,
                        ha="center", va="center", color="#666666")
                ax.set_title(cell, loc="left")
                _time_axis(ax, int(window.split("-")[1][:-1]))
                continue
            times = full.time_s.to_numpy()
            reference = full.template.to_numpy()
            stress = []
            informative = sub.fit_type != "full"
            if "informative_fold" in sub.columns:
                informative &= sub.informative_fold.isin([True, "True"])
            if "identified" in sub.columns:
                informative &= sub.identified.isin([True, "True"])
            for (kind, fold), curve in sub.loc[informative].groupby(["fit_type", "fold"]):
                curve = curve.set_index("time_s").reindex(times)
                value = curve.template.to_numpy()
                mask = np.isfinite(reference) & np.isfinite(value)
                flip = -1 if mask.any() and np.dot(reference[mask], value[mask]) < 0 else 1
                value = value * flip
                display.extend({"window": window, "cell": cell, "fit_type": kind, "fold": fold,
                                "time_s": t, "display_template": h, "display_sign": flip}
                               for t, h in zip(times, value))
                if kind == "loao":
                    ax.plot(times, value, color="#777777", alpha=.12, lw=.6)
                elif kind == "top3_loao":
                    stress.append(value)
            if stress:
                ax.plot(times, pd.DataFrame(stress).median().to_numpy(), color=MODEL_COLORS["M0"],
                        lw=1.2, ls="--", label="Top-3 exclusion: median")
            ax.plot(times, reference, color=MODEL_COLORS["M1"], lw=1.7, label="Full-data template")
            _time_axis(ax, int(window.split("-")[1][:-1]))
            bound = np.sqrt(len(times)) * 1.05
            ax.set_ylim(-bound, bound)
            record = summary.loc[(summary.window == window) & (summary.cell == cell)]
            state = _status_label(record.iloc[0]) if len(record) else ""
            ax.set_title(f"{cell}  |  {state}", loc="left")
            ax.set_ylabel("Template (RMS = 1)")
            ax.set_xlabel("Time from stimulus onset (s)")
        for ax in list(axes.flat)[len(cells):]:
            ax.set_axis_off()
        fig.suptitle(f"Shared temporal shape across animal-held-out fits: {window}", fontsize=12)
        handles = [Line2D([0], [0], color=MODEL_COLORS["M1"], lw=1.7, label="Full-data template"),
                   Line2D([0], [0], color="#888888", lw=.7, label="Animal-held-out templates")]
        if (selected.fit_type == "top3_loao").any():
            handles.append(Line2D([0], [0], color=MODEL_COLORS["M0"], ls="--",
                                  label="Exclude strongest 3 strains: median shape"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .038), ncol=len(handles), frameon=False)
        _caption(fig, "Shading: stimulus, 0–10 s. Held-out signs are aligned to the full-data shape for display only. "
                 "Thin curves show overlapping fits, not a confidence interval. Templates are dimensionless.")
        fig.tight_layout(rect=[0, .075, 1, .96])
        _save(fig, pdf, folder, f"01_templates_{window}")
    pd.DataFrame(display).to_csv(figure_data / "template_display_values.csv", index=False)


def _representative_pages(predictions, representatives, folder, pdf):
    main = _conditions(predictions.loc[predictions.window == "0-40s"])
    selected = representatives.loc[representatives.window == "0-40s"] if len(representatives) else representatives
    cells = _cell_order(main.cell)
    kinds = ["Typical fit", "Timing deviation", "Insufficient evidence"]
    for page, start in enumerate(range(0, len(cells), 7), 1):
        page_cells = cells[start:start + 7]
        fig, axes = plt.subplots(len(page_cells), 3, figsize=(14, 2.0 * len(page_cells) + 1.2), squeeze=False)
        for row, cell in enumerate(page_cells):
            limits = []
            for col, kind in enumerate(kinds):
                ax = axes[row, col]
                choices = selected.loc[(selected.cell == cell) & (selected.example_type == kind)] if len(selected) else selected
                _time_axis(ax, 40)
                ax.set_xlabel("Time (s)")
                if col == 0:
                    ax.set_ylabel(f"{cell}\nΔF/F₀")
                if choices.empty:
                    ax.text(.5, .5, "No example meets the selection rule", ha="center", va="center",
                            transform=ax.transAxes, color="#666666", fontsize=8)
                    ax.set_title(kind, loc="left")
                    continue
                choice = choices.iloc[0]
                part = main.loc[(main.cell == cell) & (main.condition == choice.condition)]
                times = np.arange(2.5, 40, 5)
                actual = _animal_curve(part, "actual", times)
                for _, curve in actual.iterrows():
                    ax.plot(times, curve, color="#A0A0A0", lw=.65, alpha=.7)
                animal = choice.representative_animal
                ax.plot(times, actual.loc[animal].to_numpy(), color="#222222", lw=1.4, ls="--")
                limits.extend(actual.to_numpy().ravel().tolist())
                score = part.loc[part.eligible.astype(bool) & (part.animal_id == animal)]
                for model in ["M1", "M2"]:
                    if len(score):
                        curve = _animal_curve(score, "pred_" + model, times).iloc[0].to_numpy()
                        ax.plot(times, curve, color=MODEL_COLORS[model], lw=1.5)
                        limits.extend(curve.tolist())
                name = str(choice.condition)
                ax.set_title(f"{kind}: {name}\n{int(choice.n_animals)} observed / "
                             f"{int(choice.n_scored_animals)} scored; shown animal {animal}", loc="left", fontsize=7)
            finite = np.asarray(limits, float)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                lo, hi = min(0, finite.min()), max(0, finite.max())
                pad = max((hi - lo) * .08, .001)
                for ax in axes[row]:
                    ax.set_ylim(lo - pad, hi + pad)
        fig.suptitle(f"Recorded examples: observed animals and held-out predictions (page {page})", fontsize=12)
        fig.legend(handles=[Line2D([0], [0], color="#AAAAAA", lw=.7, label="Observed animals"),
                            Line2D([0], [0], color="#222222", ls="--", label="Selected held-out animal"),
                            Line2D([0], [0], color=MODEL_COLORS["M1"], label="M1 prediction for that animal"),
                            Line2D([0], [0], color=MODEL_COLORS["M2"], label="M2 prediction for that animal")],
                   loc="lower center", bbox_to_anchor=(.5, .037), ncol=4, frameon=False)
        _caption(fig, "Examples are selected from prediction errors by the recorded rule; they do not supply independent inferential evidence. "
                 "Gray lines: all observed animals. Black and predictions refer to the same selected animal; predictions use eligible entries only. Row scales match.")
        fig.tight_layout(rect=[0, .085, 1, .96])
        _save(fig, pdf, folder, f"01_representatives_page{page}")


def _coefficient_pages(coefficients, templates, summary, predictions, folder, figure_data, pdf):
    window = "0-40s"
    valid_cells = _cell_order(summary.loc[(summary.window == window) & (summary.status == "useful"), "cell"])
    all_cells = _cell_order(summary.loc[summary.window == window, "cell"])
    coef = _conditions(coefficients.loc[coefficients.window == window])
    means = _conditions(predictions.loc[predictions.window == window]).groupby(
        ["condition", "strain", "block", "cell", "time_s"], observed=True).actual.agg(["mean", "count"]).reset_index()
    means.to_csv(figure_data / "raw_condition_window_means.csv", index=False)
    coef.loc[coef.cell.isin(valid_cells)].to_csv(figure_data / "displayed_coefficients.csv", index=False)
    if valid_cells:
        matrix = coef.pivot(index="condition", columns="cell", values="coefficient").reindex(columns=valid_cells).sort_index()
        vlim = _finite_limit(matrix.to_numpy())
        for page, start in enumerate(range(0, len(matrix), 60), 1):
            part = matrix.iloc[start:start + 60]
            fig = plt.figure(figsize=(max(8, len(valid_cells) * .7 + 3), max(7, len(part) * .16 + 3)))
            grid = fig.add_gridspec(2, len(valid_cells), height_ratios=[1, max(4, len(part) * .17)],
                                   left=.23, right=.86, bottom=.1, top=.9, hspace=.35)
            for j, cell in enumerate(valid_cells):
                ax = fig.add_subplot(grid[0, j])
                t = templates.loc[(templates.window == window) & (templates.cell == cell) & (templates.fit_type == "full")].sort_values("time_s")
                ax.plot(t.time_s, t.template, color=MODEL_COLORS["M1"], lw=1.2)
                _time_axis(ax, 40)
                ax.set_ylim(-2.9, 2.9)
                ax.set_title(cell)
                ax.set_xticks([0, 40])
                ax.set_xlabel("Time (s)")
                if j == 0:
                    ax.set_ylabel("Template\n(RMS = 1)")
                else:
                    ax.set_yticklabels([])
            ax = fig.add_subplot(grid[1, :])
            im = _heatmap(ax, part, vlim=vlim)
            ax.set_yticks(range(len(part)), part.index, fontsize=6)
            ax.set_ylabel("Strain | acquisition block")
            cax = fig.add_axes([.89, .2, .025, .5])
            fig.colorbar(im, cax=cax, label="Signed coefficient (ΔF/F₀)")
            fig.suptitle(f"Cell response combinations with a useful single-template approximation: page {page}", fontsize=11)
            _caption(fig, "Full-data fits are descriptive, not held-out predictions. Shared color scale across cells and pages; gray = missing. "
                     "Template shading: 0–10 s stimulus. A coefficient scales and can reverse its template; color is not an excitation/inhibition label.")
            _save(fig, pdf, folder, f"02_coefficients_page{page}")
    else:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.set_axis_off()
        ax.text(.5, .55, "No cell currently meets the evidence criteria for a useful single-template summary.",
                ha="center", va="center", wrap=True, transform=ax.transAxes)
        _caption(fig, "All cells remain visible in the original-window response panels. No zero coefficients are assigned.")
        _save(fig, pdf, folder, "02_coefficients_no_supported_cells")
    other = [c for c in all_cells if c not in valid_cells]
    # Four cells per page retain all strain-block rows without reducing missingness.
    conditions = sorted(means.condition.unique())
    for page, start in enumerate(range(0, len(other), 4), 1):
        cells = other[start:start + 4]
        fig = plt.figure(figsize=(4.1 * len(cells) + 2, max(9, len(conditions) * .13 + 2.6)))
        grid = fig.add_gridspec(1, len(cells) * 2, width_ratios=[8, 1] * len(cells),
                               left=.13, right=.95, top=.88, bottom=.1, wspace=.22)
        for j, cell in enumerate(cells):
            part = means.loc[means.cell == cell]
            matrix = part.pivot(index="condition", columns="time_s", values="mean").reindex(index=conditions, columns=np.arange(2.5, 40, 5))
            count = part.pivot(index="condition", columns="time_s", values="count").reindex_like(matrix)
            matrix.to_csv(figure_data / f"raw_window_matrix_{cell}.csv")
            ax = fig.add_subplot(grid[0, j * 2])
            im = _heatmap(ax, matrix, times=["0–5", "5–10", "10–15", "15–20", "20–25", "25–30", "30–35", "35–40"])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            if j == 0:
                ax.set_yticks(range(len(matrix)), matrix.index, fontsize=5.5)
            else:
                ax.set_yticks([])
            ax.set_xlabel("Time window (s)")
            state = _status_label(summary.loc[(summary.window == window) & (summary.cell == cell)].iloc[0])
            ax.set_title(f"{cell}\n{state}", fontsize=8)
            cbar = fig.colorbar(im, ax=ax, orientation="horizontal", location="top", fraction=.018, pad=.065, aspect=12)
            cbar.set_label("Mean ΔF/F₀; cell-specific scale", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            ca = fig.add_subplot(grid[0, j * 2 + 1])
            # The response colorbar shrinks its parent axes; match coverage rows
            # to that post-colorbar position rather than to the original grid.
            response_position, count_position = ax.get_position(), ca.get_position()
            ca.set_position([count_position.x0, response_position.y0,
                             count_position.width, response_position.height])
            # Minimum across available windows makes partial time coverage visible.
            n = count.fillna(0).min(axis=1).to_numpy()
            ca.imshow(n[:, None], aspect="auto", cmap="Greys", interpolation="none", vmin=0,
                      vmax=max(1, np.nanmax(n)))
            ca.set_yticks([])
            ca.set_xticks([0], ["min n"], rotation=90)
            for k, value in enumerate(n):
                ca.text(0, k, str(int(value)), ha="center", va="center", fontsize=5,
                        color="white" if value > max(1, np.nanmax(n)) / 2 else "black")
        fig.suptitle(f"Original response windows for cells without a supported single-template summary: page {page}", fontsize=11)
        _caption(fig, "Means over observed animals; all recorded strain-block conditions retained. Gray = missing response, not zero. "
                 "Each cell has its own symmetric response color scale. Coverage is minimum animal count over all eight windows.")
        _save(fig, pdf, folder, f"02_original_windows_page{page}", dpi=180)


def _comparison_pages(summary, animal_gains, folder, pdf):
    for window in ["0-40s", "0-25s"]:
        part = summary.loc[summary.window == window]
        cells = _cell_order(part.cell)
        if not cells:
            continue
        fig, axes = plt.subplots(7, 2, figsize=(11, 13), squeeze=False)
        for ax, cell in zip(axes.flat, cells):
            rec = part.loc[part.cell == cell].iloc[0]
            values = [rec.get("mse_" + model, np.nan) for model in MODEL_ORDER]
            ax.plot(np.arange(4), values, color="#C0C0C0", lw=1, zorder=1)
            for j, (model, value) in enumerate(zip(MODEL_ORDER, values)):
                ax.scatter(j, value, color=MODEL_COLORS[model], s=30, zorder=2)
            ax.set_xticks(range(4), MODEL_ORDER)
            finite = np.asarray(values, float)
            finite = finite[np.isfinite(finite)]
            maximum = float(finite.max()) if len(finite) else 0
            ax.set_ylim(0, maximum * 1.12 if maximum > 0 else 1)
            ax.set_ylabel("MSE [(ΔF/F₀)²]")
            n_animals = int(rec.get("n_animals", 0))
            n_conditions = int(rec.get("n_conditions", 0))
            ax.set_title(f"{cell}  |  {n_animals} animals; {n_conditions} conditions", loc="left")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        for ax in list(axes.flat)[len(cells):]:
            ax.set_axis_off()
        fig.suptitle(f"Prediction of held-out animals: {window}", fontsize=12)
        _caption(fig, "MSE in original squared calcium-response units; cell panels have separate vertical scales. "
                 "Within each bin: animals equal; then windows, blocks within strain, strains, and cells equal. "
                 "M2 is an uncompressed training mean, not truth or a noise ceiling.")
        fig.tight_layout(rect=[0, .055, 1, .96])
        _save(fig, pdf, folder, f"03_model_errors_{window}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        for ax, metric, label in zip(axes, ["delta_combination", "delta_timing"],
                                      ["Combination gain: M0 − M1", "Additional timing gain: M1 − M2"]):
            ax.axvline(0, color="#999999", lw=.8)
            for j, cell in enumerate(cells):
                animals = animal_gains.loc[(animal_gains.window == window) & (animal_gains.cell == cell)].sort_values("animal_id")
                points = animals[metric].to_numpy()
                offsets = (np.arange(len(points)) % 7 - 3) * .045
                ax.scatter(points, j + offsets, s=12, color="#999999", alpha=.65, linewidths=0)
                value = part.loc[part.cell == cell, metric].iloc[0]
                ax.scatter(value, j, marker="D", s=42, color=MODEL_COLORS["M1"] if metric == "delta_combination" else MODEL_COLORS["M2"],
                           edgecolor="white", linewidth=.4, zorder=3)
            ax.set_yticks(range(len(cells)), cells)
            ax.set_ylim(len(cells) - .5, -.5)
            ax.set_title(label)
            ax.set_xlabel("Reduction in squared prediction error [(ΔF/F₀)²]")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useMathText=True)
            ax.grid(axis="y", color="#EEEEEE", zorder=0)
        fig.suptitle(f"Prediction gains and individual-animal variation: {window}", fontsize=12)
        fig.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999",
                                  label="One held-out animal (available conditions)"),
                            Line2D([0], [0], marker="D", color="none", markerfacecolor="#444444",
                                  label="Overall hierarchical-weight estimate")],
                   loc="lower center", bbox_to_anchor=(.5, .065), frameon=False, ncol=2)
        _caption(fig, "Positive = improved prediction; negative gains are retained. Animal points macro-average available windows, blocks and strains. "
                 "Diamonds use the prespecified overall weights, so need not equal the mean of points. Overlapping training folds are not independent replicates.")
        fig.tight_layout(rect=[0, .13, 1, .95])
        _save(fig, pdf, folder, f"03_prediction_gains_{window}")


def plot_results(output_dir):
    """Read analysis tables, export three PDF figure groups, PNGs and plot data.

    Required columns are documented by the corresponding data creation helper.
    ``predictions.parquet`` must retain observed but ineligible response entries.
    No new scientific data are inferred or imputed during plotting.
    """
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    folder = output_dir / "figures"
    figure_data = output_dir / "figure_data"
    folder.mkdir(parents=True, exist_ok=True)
    figure_data.mkdir(parents=True, exist_ok=True)
    templates = pd.read_csv(tables / "templates.csv", dtype={"fold": str})
    coefficients = pd.read_csv(tables / "coefficients.csv")
    predictions = pd.read_parquet(tables / "predictions.parquet")
    summary = pd.read_csv(tables / "cell_summary.csv")
    assessments_path = tables / "cell_assessments.csv"
    if assessments_path.exists():
        from .response_structure import assessment_fingerprint, evidence_fingerprints
        metadata_path = tables / "cell_assessments_metadata.json"
        if not metadata_path.exists():
            raise ValueError("Manual cell assessments require cell_assessments_metadata.json with the reviewed numeric fingerprint.")
        metadata = json.loads(metadata_path.read_text())
        if assessment_fingerprint(summary) != metadata.get("numeric_sha256"):
            raise ValueError("Cell-summary numbers changed after manual review; review the new evidence before reusing cell assessments.")
        if evidence_fingerprints(output_dir) != metadata.get("evidence_sha256"):
            raise ValueError("The reviewed evidence files changed; review the new evidence before reusing cell assessments.")
        assessment = pd.read_csv(assessments_path)
        columns = ["window", "cell", "status"] + (["scope"] if "scope" in assessment else [])
        assessment = assessment[columns].rename(columns={"status": "reviewed_status"})
        summary = summary.merge(assessment, on=["window", "cell"], how="left", validate="one_to_one")
        summary["status"] = summary.reviewed_status.combine_first(summary.status)
        summary = summary.drop(columns="reviewed_status")
    coverage = pd.read_csv(tables / "coverage.csv")
    animal_gains = _paired_animal_gains(predictions)
    representatives = _representatives(predictions)
    # Rerunning after status review can reduce page counts; remove only this
    # module's old generated PNGs after validating the input and reviewed values.
    for pattern in ["01_templates_*.png", "01_representatives_page*.png",
                    "02_coefficients_page*.png", "02_coefficients_no_supported_cells.png",
                    "02_original_windows_page*.png", "03_model_errors_*.png",
                    "03_prediction_gains_*.png"]:
        for old in folder.glob(pattern):
            old.unlink()
    for old in figure_data.glob("raw_window_matrix_*.csv"):
        old.unlink()
    animal_gains.to_csv(figure_data / "paired_animal_gains.csv", index=False)
    representatives.to_csv(figure_data / "representative_selection.csv", index=False)
    summary.to_csv(figure_data / "model_comparison_data.csv", index=False)
    coverage.to_csv(figure_data / "coverage.csv", index=False)
    if len(representatives):
        selected_keys = representatives[["window", "strain", "block", "cell", "example_type"]]
        predictions.merge(selected_keys, on=["window", "strain", "block", "cell"], how="inner").to_csv(
            figure_data / "representative_response_values.csv", index=False)
    with _style():
        with PdfPages(folder / "01_time_profiles_and_examples.pdf") as pdf:
            _template_pages(templates, summary, folder, figure_data, pdf)
            _representative_pages(predictions, representatives, folder, pdf)
        with PdfPages(folder / "02_cell_response_combinations.pdf") as pdf:
            _coefficient_pages(coefficients, templates, summary, predictions, folder, figure_data, pdf)
        with PdfPages(folder / "03_animal_heldout_model_comparison.pdf") as pdf:
            _comparison_pages(summary, animal_gains, folder, pdf)
    return {"figures": folder, "figure_data": figure_data}
