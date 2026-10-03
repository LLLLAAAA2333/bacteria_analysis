"""Notebook-facing, presentation-focused redraw of cached response results.

No fitting or raw-data processing is performed. Descriptive examples were
chosen after inspecting the results; they are not independent validation sets.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BLUE, ORANGE = "#207F9C", "#BC692B"
EXAMPLES = [
    (0, 0, "AWCON", "A300", "20260520", "主体近似；强正向响应例"),
    (1, 0, "AWCON", "A264", "20260429", "主体近似；弱负向响应例"),
    (0, 1, "AWB", "A002", "20260520", "已有定位证据；10–25秒转换延迟"),
    (1, 1, "AWCOFF", "A024", "20260601", "已有定位证据；10–25秒恢复延迟"),
    (0, 2, "ASG", "A305", "20260414", "已有证据不足例；两种预测均未优于基准"),
    (1, 2, "ASI", "A237", "20260311", "两动物条件中，按动物均值曲线RMS取中位条件"),
]


def validate_evidence(source_dir):
    """Stop before drawing if the saved evidence no longer matches its review."""
    from .response_structure import assessment_fingerprint, evidence_fingerprints
    source = Path(source_dir)
    meta = json.loads((source / "tables/cell_assessments_metadata.json").read_text())
    summary = pd.read_csv(source / "tables/cell_summary.csv")
    if (assessment_fingerprint(summary) != meta["numeric_sha256"]
            or evidence_fingerprints(source) != meta["evidence_sha256"]):
        raise ValueError("Cached evidence changed; review the cell assessments before redrawing.")
    # These display inputs were not part of the original review fingerprint.
    # Check them against its bound predictions instead of trusting mixed caches.
    predictions = pd.read_parquet(source / "tables/predictions.parquet")
    observations = pd.read_parquet(source / "data/observations.parquet").rename(
        columns={"sample_id": "strain", "neuron_class": "cell", "response": "actual"})
    keys = ["animal_id", "strain", "block", "cell", "bin_index"]
    observed = observations.loc[observations.bin_index.lt(8) & observations.actual.notna()]
    expected = predictions.loc[predictions.window.eq("0-40s")]
    left = observed.set_index(keys).actual.sort_index()
    right = expected.set_index(keys).actual.sort_index()
    if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-12, atol=1e-14):
        raise ValueError("Saved observations do not match the reviewed prediction records.")
    scored = predictions.loc[predictions.eligible].copy()
    columns = ["mse_"+model for model in ("B", "M0", "M1", "M2")]
    for model in ("B", "M0", "M1", "M2"):
        scored["mse_"+model] = (scored.actual - scored["pred_"+model])**2
    keys = ["window", "animal_id", "cell"]
    errors = scored.groupby(keys+["strain", "block"])[columns].mean().groupby(keys).mean()
    errors["delta_combination"] = errors.mse_M0 - errors.mse_M1
    errors["delta_timing"] = errors.mse_M1 - errors.mse_M2
    saved = pd.read_csv(source / "tables/animal_cell_errors.csv").set_index(keys)
    left, right = errors.sort_index(), saved[errors.columns].sort_index()
    if not left.index.equals(right.index) or not np.allclose(left, right, rtol=1e-12, atol=1e-14):
        raise ValueError("Saved animal errors do not match the reviewed prediction records.")
    return meta


def _input_hashes(source):
    names = ("data/observations.parquet", "tables/animal_cell_errors.csv", "tables/cell_assessments.csv")
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}


def _code_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("response_structure_display*.py")}


def display_cache_current(source_dir, output_dir):
    """Check that cached figures belong to the current inputs and plotting code."""
    from .response_structure import assessment_fingerprint, evidence_fingerprints
    source = Path(source_dir)
    path = Path(output_dir) / "display_parameters.json"
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    return (record.get("reviewed_numeric_sha256") == assessment_fingerprint(
                pd.read_csv(source / "tables/cell_summary.csv"))
            and record.get("reviewed_evidence_sha256") == evidence_fingerprints(source)
            and record.get("display_input_sha256") == _input_hashes(source)
            and record.get("code_sha256") == _code_hashes())


def _select_examples(predictions):
    records = []
    for row, col, cell, strain, block, reason in EXAMPLES:
        p = predictions.loc[(predictions.window == "0-40s") & (predictions.cell == cell)
                            & (predictions.strain == strain) & (predictions.block == block)].copy()
        if p.empty:
            raise ValueError(f"Missing reviewed example: {cell}, {strain}, {block}")
        observed = p.loc[p.actual.notna()]
        scored = p.loc[p.eligible]
        n = observed.animal_id.nunique()
        if len(scored):
            scored = scored.assign(mse=(scored.actual - scored.pred_M1)**2)
            ranked = scored.groupby("animal_id").mse.mean().reset_index().sort_values(
                ["mse", "animal_id"], kind="stable")
            animal = ranked.iloc[len(ranked)//2].animal_id
            rule = "All scored animals sorted by 0–40 s M1 MSE, then animal_id; index n//2. No benefit filtering."
        else:
            animal = ""
            rule = "All available animals; no prediction and no representative animal selected."
        records.append(dict(row=row, column=col, cell=cell, strain=strain, block=block,
                            condition_selection_reason=reason, n_animals=n,
                            representative_animal=animal, animal_selection_rule=rule))
    return pd.DataFrame(records)


def plot_examples(source_dir, output_dir, windows=("0-40s", "0-25s")):
    """Export A as two separate 2×3 panels in original response units.

    Predictions belong to the highlighted animal's own leave-one-animal-out
    fold. Faint curves are observations, never averaged out-of-fold predictions.
    The short-window display reuses the main-window case/animal selection.
    """
    source, out = Path(source_dir), Path(output_dir)
    validate_evidence(source)
    folder, data_dir = out / "figures", out / "figure_data"
    folder.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(source / "tables/predictions.parquet")
    p["block"] = p.block.astype(str)
    selection = _select_examples(p)
    selection.to_csv(data_dir / "A_example_selection.csv", index=False)
    export = []
    paths = []
    style = {"font.family": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
             "font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
             "axes.spines.right": False, "axes.edgecolor": "#ADB7BD",
             "axes.labelcolor": "#35444D", "text.color": "#253640",
             "xtick.color": "#596973", "ytick.color": "#596973", "pdf.fonttype": 42}
    with plt.rc_context(style):
        for window in windows:
            end = int(window.split("-")[1][:-1])
            fig, axes = plt.subplots(2, 3, figsize=(12.7, 7.4))
            fig.subplots_adjust(left=.075, right=.985, bottom=.155, top=.79,
                                wspace=.29, hspace=.58)
            fig.suptitle(f"A  共享时间轮廓的适用范围   |   0–{end} 秒", x=.06, ha="left",
                         fontsize=17, y=.975)
            for ax, title in zip(axes[0], ["缩放 / 翻转", "额外时程", "证据不足"]):
                pos = ax.get_position()
                fig.text(pos.x0+pos.width/2, .898, title, ha="center", fontsize=14)
            for item in selection.to_dict("records"):
                ax = axes[item["row"], item["column"]]
                frame = p.loc[(p.window == window) & (p.cell == item["cell"])
                              & (p.strain == item["strain"]) & (p.block == item["block"])].copy()
                chosen = frame.animal_id == item["representative_animal"]
                frame["highlighted_animal"] = chosen
                frame["panel_row"], frame["panel_column"] = item["row"], item["column"]
                export.append(frame)
                for _, animal in frame.loc[~chosen].groupby("animal_id", sort=True):
                    animal = animal.sort_values("time_s")
                    ax.plot(animal.time_s, animal.actual, color="#ADB6BD", lw=1,
                            alpha=.65, marker="o", ms=2, zorder=1)
                if chosen.any():
                    animal = frame.loc[chosen].sort_values("time_s")
                    ax.plot(animal.time_s, animal.actual, color="#26343D", lw=1.9,
                            marker="o", ms=3.5, zorder=4)
                    for model, color, ls in [("M1", BLUE, "--"), ("M2", ORANGE, "-")]:
                        prediction = animal["pred_"+model].where(animal.eligible)
                        ax.plot(animal.time_s, prediction, color=color, lw=1.9,
                                ls=ls, zorder=3)
                ax.axvspan(0, 10, color="#EEF1F3", zorder=-3)
                ax.axhline(0, lw=.65, color="#BDC6CC", zorder=-2)
                ax.set(xlim=(0, end), xticks=sorted(set([0, 10, 25, end])))
                ax.tick_params(labelsize=10, length=3)
                ax.set_title(f"{item['cell']}  ·  {item['strain']}", loc="left", pad=22, fontsize=12)
                date = pd.to_datetime(item["block"]).strftime("%Y-%m-%d")
                detail = f"{date}   n = {item['n_animals']}"
                if not chosen.any():
                    detail += "   覆盖不足"
                ax.text(0, 1.035, detail, transform=ax.transAxes, fontsize=9.5, color="#6E7D86")
                ax.set_ylabel("ΔF/F₀", fontsize=11)
                if item["row"] == 1:
                    ax.set_xlabel("时间（秒）")
            handles = [Line2D([], [], color="#26343D", marker="o", ms=3, lw=1.8, label="示例动物"),
                       Line2D([], [], color="#ADB6BD", lw=1.1, label="其余动物"),
                       Line2D([], [], color=BLUE, ls="--", lw=1.9, label="共享时程预测"),
                       Line2D([], [], color=ORANGE, lw=1.9, label="独立时程预测")]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .066),
                       ncol=4, frameon=False, fontsize=10)
            fig.text(.075, .028, "预测仅由其他动物估计  ·  灰底：刺激期  ·  n：动物数  ·  各面板纵轴范围不同",
                     fontsize=9.5, color="#6E7D86")
            stem = folder / f"A_time_profiles_{window}"
            for ext in ("png", "pdf", "svg"):
                fig.savefig(stem.with_suffix("."+ext), dpi=185, facecolor="white")
            paths.append(str(stem.with_suffix(".png")))
            plt.close(fig)
    pd.concat(export, ignore_index=True).to_csv(data_dir / "A_response_and_prediction_values.csv", index=False)
    return paths


def plot_display(source_dir, output_dir):
    """Single Notebook call for the three display groups, reading cached tables."""
    from .response_structure_display_combinations import plot_combinations
    from .response_structure_display_comparison import plot_comparison
    source, out = Path(source_dir), Path(output_dir)
    metadata = validate_evidence(source)
    out.mkdir(parents=True, exist_ok=True)
    results = {"A": plot_examples(source, out),
               "B": plot_combinations(source, out),
               "C": plot_comparison(source, out)}
    record = {"source_dir": str(source.resolve()), "display_only": True, "refit": False,
              "reviewed_numeric_sha256": metadata["numeric_sha256"],
              "reviewed_evidence_sha256": metadata["evidence_sha256"],
              "display_input_sha256": _input_hashes(source),
              "main_window": "0-40s", "sensitivity_window": "0-25s",
              "code_sha256": _code_hashes()}
    (out / "display_parameters.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return results
