"""Compact display drafts of the existing held-out chemistry/neural profiles.

No candidate selection, model fitting, or data exclusion is performed. The two
drafts differ only in whether the neural discovery mean is subtracted for display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


def _symmetric_max(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(finite.max()), 0.001) if finite.size else 1.0


def make_figures(out: Path) -> list[Path]:
    """Create raw and discovery-mean-centered display drafts and audit mappings.

    Input/output directory is the existing direct-workbook report. Every held-out
    strain is included, with columns sorted by its saved chemical score. Chemistry
    keeps its source row order. All neurons are ordered by the absolute saved
    discovery slope, largest first; ties keep source order. Centering subtracts
    one saved discovery mean per neuron, without variance rescaling.
    """
    out = Path(out)
    tables, figures = out / "tables", out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(tables / "sample_scores.csv", dtype={"strain": str})
    members = pd.read_csv(tables / "selected_members.csv")
    slopes = pd.read_csv(tables / "neural_slopes.csv")
    slopes = slopes.assign(discovery_abs_slope=slopes["discovery"].abs()).sort_values(
        "discovery_abs_slope", ascending=False, kind="stable"
    )
    chemistry = pd.read_csv(tables / "selected_chemical_standardized.csv", index_col=0)
    neural = pd.read_csv(tables / "neural_unit_coefficients.csv", index_col=0)
    chemistry.index = chemistry.index.astype(str)
    neural.index = neural.index.astype(str)

    held = scores.loc[scores["split"].eq("holdout")].sort_values(["chemical_score", "strain"])
    held_ids = held["strain"].tolist()
    discovery_ids = scores.loc[scores["split"].eq("discovery"), "strain"].tolist()
    chemicals = members["metabolite"].astype(str).tolist()
    cells = slopes["cell"].astype(str).tolist()
    chem_values = chemistry.loc[held_ids, chemicals].to_numpy(float).T
    raw_neural = neural.loc[held_ids, cells].to_numpy(float).T
    discovery_means = neural.loc[discovery_ids, cells].mean(axis=0)
    centered_neural = raw_neural - discovery_means.to_numpy(float)[:, None]

    paths = []
    strain_map = held[["strain", "genus", "chemical_score"]].reset_index(drop=True)
    strain_map.insert(0, "display_column_1based", np.arange(1, len(held) + 1))
    strain_map_path = figures / "simple_story_strain_order.csv"
    strain_map.to_csv(strain_map_path, index=False)
    paths.append(strain_map_path)
    neuron_map = pd.DataFrame({
        "display_row_1based": np.arange(1, len(cells) + 1),
        "cell": cells,
        "discovery_slope": slopes["discovery"].to_numpy(float),
        "discovery_abs_slope": slopes["discovery_abs_slope"].to_numpy(float),
        "discovery_mean_unit_coefficient": discovery_means.to_numpy(float),
    })
    neuron_map_path = figures / "simple_story_neuron_order.csv"
    neuron_map.to_csv(neuron_map_path, index=False)
    paths.append(neuron_map_path)

    variants = [
        ("raw", raw_neural, "Unit coefficient"),
        ("centered", centered_neural, "Unit coefficient - discovery mean"),
    ]
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 15, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 11,
        "svg.fonttype": "none",
    }):
        for variant, neural_values, neural_scale_label in variants:
            fig = plt.figure(figsize=(12, 7))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.022],
                                  height_ratios=[len(chemicals), len(cells)],
                                  hspace=0.30, wspace=0.045)
            fig.subplots_adjust(left=0.16, right=0.90, top=0.94, bottom=0.12)
            axc, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
            cbac, cban = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#D9DDDF")
            chem_max = _symmetric_max(chem_values)
            neural_max = _symmetric_max(neural_values)
            imc = axc.imshow(chem_values, aspect="auto", interpolation="nearest",
                             cmap=cmap, vmin=-chem_max, vmax=chem_max)
            imn = axn.imshow(neural_values, aspect="auto", interpolation="nearest",
                             cmap=cmap, vmin=-neural_max, vmax=neural_max)
            axc.set_yticks(np.arange(len(chemicals)), chemicals)
            axn.set_yticks(np.arange(len(cells)), cells)
            axc.set_title("Chemical levels", loc="left", pad=12)
            axn.set_title("Neural responses", loc="left", pad=12)
            axn.set_xlabel(f"Strains: lower to higher chemical levels  (n = {len(held_ids)})", labelpad=16)
            for ax in (axc, axn):
                ax.set_xticks([])
                ax.tick_params(axis="both", length=0, pad=9)
                ax.spines[:].set_visible(False)
            cbc = fig.colorbar(imc, cax=cbac, ticks=[-chem_max, 0, chem_max])
            cbn = fig.colorbar(imn, cax=cban, ticks=[-neural_max, 0, neural_max])
            cbc.set_label("Chemical level (z score)", fontsize=10, labelpad=10)
            cbn.set_label(neural_scale_label, fontsize=10, labelpad=10)
            for cb in (cbc, cbn):
                cb.outline.set_visible(False)
                cb.ax.tick_params(length=2, labelsize=10, pad=4)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            stem = figures / f"03_simple_story_{variant}"
            for suffix in (".png", ".svg"):
                path = stem.with_suffix(suffix)
                fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                paths.append(path)
            plt.close(fig)

    caption = (
        "Compact display drafts of the existing selected chemical module and complete neural "
        f"profiles: {len(chemicals)} × {len(held_ids)} chemistry values above and "
        f"{len(cells)} × {len(held_ids)} neural values below. Every held-out strain appears once "
        "as a column, in the same position in both panels. Columns are ordered by the saved "
        "chemical score; this makes the chemical gradient expected by construction. Equal "
        "column spacing represents rank, not equal chemical distance. No strain or neuron is "
        "excluded. Chemistry keeps its source row order. All 13 neural rows are ordered "
        "by the absolute discovery slope (largest first), with ties retaining source order. "
        "A slope is the unit-coefficient change per chemical-score SD. The ordering uses "
        "discovery data only and indicates association magnitude in this fitted combination, "
        "not causal importance or statistical significance. The complete strain mapping, "
        "neuron order and ordering values are saved beside the figures.\n\n"
        "Chemistry values are the existing global discovery z scores of log2-transformed "
        "reported concentrations in ng/mL. In the raw draft, neural values are the existing "
        "observed unit-vector coefficients. In the centered draft, each neuron's discovery "
        "mean coefficient is subtracted solely for display; those means are saved in "
        "simple_story_neuron_order.csv. No per-neuron variance scaling is applied. Thus red "
        "and blue in the centered neural panel indicate above and below the discovery mean, "
        "not excitatory and inhibitory effects. Both drafts use a single shared scale across "
        "all neural rows. Chemistry and neural values have separate scales; all color scales "
        "span the complete displayed ranges without clipping. Missing values, if present, "
        "are gray.\n\n"
        "The visual question is whether parts of the neural pattern change across the "
        "chemistry-ordered strains, including all exceptions. This display adds no new fit, "
        "candidate selection, validation test, or causal claim. The analysis reuses a previously "
        "seen dataset and cached neural templates. Independently cultured chemical material "
        "does not measure the concentrations delivered in the neural stimulus.\n"
    )
    caption_path = figures / "simple_story_caption.txt"
    caption_path.write_text(caption, encoding="utf-8")
    paths.append(caption_path)
    return paths
