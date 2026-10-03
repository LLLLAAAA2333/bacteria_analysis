"""Inspection map of cached chemical-neighbor, paired amplitude differences.

Notebook usage::
    from bacteria_analysis.neighborhood_amplitudes_display import plot_amplitude_neighborhoods
    plot_amplitude_neighborhoods(
        root / "reports/examples/poster_neighborhood_amplitudes_20261001")

Reads summary tables only. No model fitting or raw-data processing is performed.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


CELLS = ("AWCON", "ASK", "ADF", "ASJ", "AWA", "AWB", "ASH")
INK, MUTED, RULE = "#20374D", "#71808F", "#DCE3E9"
TEAL, MISSING = "#147D78", "#D5D8DA"
PAIR_COLUMNS = (
    "pair_id", "strain_a", "strain_b", "date", "genus", "reference_group",
    "joint_rms_log2fc", "n_joint_reported", "selected_main", "is_benchmark",
)
CELL_COLUMNS = (
    "pair_id", "cell", "n_animals", "eligible", "mean_delta", "abs_mean_delta",
    "sem_delta", "energy", "loo_energy_min", "loo_energy_max", "mean_curve_rms",
    "model_residual_fraction", "mean_amp_a", "mean_amp_b",
)


def _booleans(values, name):
    """Read CSV booleans explicitly; a nonempty 'False' string is not True."""
    parsed = values.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"{name} must contain only True/False values.")
    return parsed.astype(bool)


def _read_values(output_dir):
    pairs = pd.read_csv(output_dir / "tables/pair_summary.csv", dtype={"date": str})
    cells = pd.read_csv(output_dir / "tables/pair_cell_summary.csv")
    for frame, columns, name in ((pairs, PAIR_COLUMNS, "pair_summary"),
                                  (cells, CELL_COLUMNS, "pair_cell_summary")):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    if pairs.pair_id.isna().any() or pairs.pair_id.duplicated().any():
        raise ValueError("Pair identifiers must be nonmissing and unique.")
    if cells[["pair_id", "cell"]].isna().any().any() or cells.duplicated(["pair_id", "cell"]).any():
        raise ValueError("Pair-cell identifiers must be nonmissing and unique.")
    for name in ("selected_main", "is_benchmark"):
        pairs[name] = _booleans(pairs[name], name)
    cells["eligible"] = _booleans(cells.eligible, "eligible")
    pairs = pairs.loc[pairs.selected_main, list(PAIR_COLUMNS)].sort_values(
        ["joint_rms_log2fc", "pair_id"], kind="stable").reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No selected_main pairs; no inspection figure was written.")
    if pairs[["strain_a", "strain_b", "date"]].isna().any().any():
        raise ValueError("Selected pair labels require strains and dates.")
    chemical = pairs[["joint_rms_log2fc", "n_joint_reported"]].to_numpy(dtype=float)
    if not np.isfinite(chemical).all() or (chemical[:, 0] < 0).any():
        raise ValueError("Selected chemical distances and coverage must be finite and nonnegative.")
    if (chemical[:, 1] < 1).any() or (chemical[:, 1] != np.floor(chemical[:, 1])).any():
        raise ValueError("Jointly reported feature counts must be positive integers.")
    cells = cells.loc[cells.pair_id.isin(pairs.pair_id) & cells.cell.isin(CELLS), list(CELL_COLUMNS)]
    expected = pd.MultiIndex.from_product([pairs.pair_id, CELLS], names=["pair_id", "cell"])
    indexed = cells.set_index(["pair_id", "cell"])
    if set(indexed.index) != set(expected):
        raise ValueError("Every selected pair must have exactly one row for each of the seven cells.")
    values = indexed.reindex(expected).reset_index().merge(pairs, on="pair_id", validate="many_to_one")
    n = values.n_animals.to_numpy(dtype=float)
    if not np.isfinite(n).all() or (n < 0).any() or (n != np.floor(n)).any():
        raise ValueError("Animal counts must be nonnegative integers.")
    if not np.array_equal(values.eligible.to_numpy(), n >= 3):
        raise ValueError("The inspection map requires eligible = (n_animals >= 3).")
    supported = values.loc[values.eligible]
    numeric = supported[["mean_delta", "abs_mean_delta", "loo_energy_min", "loo_energy_max"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Eligible cells require finite effects and leave-one-animal energy limits.")
    if not np.allclose(supported.abs_mean_delta, supported.mean_delta.abs(), rtol=1e-10, atol=1e-12):
        raise ValueError("Absolute and signed mean differences disagree.")
    if (supported.loo_energy_min > supported.loo_energy_max).any():
        raise ValueError("Leave-one-animal energy limits are reversed.")
    values["row_index"] = values.pair_id.map(dict(zip(pairs.pair_id, range(len(pairs)))))
    values["column_index"] = values.cell.map(dict(zip(CELLS, range(len(CELLS)))))
    values["plotted_abs_mean_delta"] = values.abs_mean_delta.where(values.eligible)
    values["positive_energy_after_every_omission"] = values.eligible & values.loo_energy_min.gt(0)
    return pairs, values


def plot_amplitude_neighborhoods(output_dir):
    """Save PNG, SVG, exact plotted values and an external methods caption.

    One row is a selected strain pair in one acquisition date. All seven cell
    classes share an absolute amplitude scale in ΔF/F₀. Cells with fewer than
    three paired animals are gray, including cells with no observations.
    """
    output_dir = Path(output_dir)
    pairs, values = _read_values(output_dir)
    n_benchmarks = int(pairs.is_benchmark.sum())
    n_candidates = len(pairs) - n_benchmarks
    selection_label = (f"{n_candidates} chemical candidates + {n_benchmarks} "
                       f"benchmark{'s' if n_benchmarks != 1 else ''}")
    matrix = values.plotted_abs_mean_delta.to_numpy().reshape(len(pairs), len(CELLS))
    finite = matrix[np.isfinite(matrix)]
    limit = max(.1, float(np.ceil(finite.max() * 10) / 10)) if finite.size else .1
    cmap = LinearSegmentedColormap.from_list("absolute_amplitude_difference", ["#FBFBFC", TEAL])
    cmap.set_bad(MISSING)
    height = max(4.6, .29 * len(pairs) + 2.3)
    bottom, top = 1.05 / height, 1 - 1.0 / height
    style = {"font.family": "DejaVu Sans", "font.size": 9,
             "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": MUTED, "ytick.color": MUTED,
             "axes.edgecolor": RULE, "axes.linewidth": .7,
             "svg.fonttype": "none", "savefig.facecolor": "white"}
    folder = output_dir / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / "amplitude_difference_map"
    paths = {"png": stem.with_suffix(".png"), "svg": stem.with_suffix(".svg"),
             "values": folder / "amplitude_difference_map_values.csv",
             "caption": folder / "caption.txt"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(10.6, height))
        chemical_ax = fig.add_axes([.285, bottom, .105, top - bottom])
        ax = fig.add_axes([.43, bottom, .48, top - bottom])
        cax = fig.add_axes([.935, bottom, .014, min(2.0 / height, top - bottom)])
        fig.text(.05, 1 - .24 / height, "Amplitude differences in chemically selected pairs",
                 va="top", fontsize=16, fontweight="bold")
        fig.text(.05, 1 - .56 / height, f"{selection_label} · 0–40 s templates · inspection map",
                 va="top", fontsize=9, color=MUTED)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=limit,
                          interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(CELLS)), CELLS)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=8)
        ax.set_yticks([])
        ax.set_xticks(np.arange(-.5, len(CELLS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(pairs), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.8)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        marked = values.loc[values.positive_energy_after_every_omission]
        ax.scatter(marked.column_index, marked.row_index, s=15, c=INK,
                   edgecolors="white", linewidths=.45)
        chemical_ax.scatter(pairs.joint_rms_log2fc, np.arange(len(pairs)),
                            s=20, color="#16839C", edgecolors="white", linewidths=.4)
        chemical_ax.set_ylim(len(pairs) - .5, -.5)
        chemical_ax.set_xlim(0, max(.1, pairs.joint_rms_log2fc.max() * 1.12))
        chemical_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        chemical_ax.set_title("Chemical RMS", fontsize=9, pad=8)
        chemical_ax.set_xlabel("log₂ fold change", fontsize=8, labelpad=5)
        labels = []
        for pair in pairs.itertuples():
            date = str(pair.date)
            if len(date) == 8 and date.isdigit():
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            labels.append(f"{pair.strain_a}/{pair.strain_b}  ·  {date}")
        chemical_ax.set_yticks(np.arange(len(pairs)), labels)
        chemical_ax.tick_params(axis="y", length=0, pad=10, labelsize=8.5)
        chemical_ax.tick_params(axis="x", length=3, labelsize=8)
        for label, pair in zip(chemical_ax.get_yticklabels(), pairs.itertuples()):
            if {pair.strain_a, pair.strain_b} == {"A231", "A232"}:
                label.set_fontweight("bold")
                label.set_color(INK)
        for name in ("top", "left", "right"):
            chemical_ax.spines[name].set_visible(False)
        chemical_ax.grid(axis="x", color=RULE, linewidth=.55, zorder=0)
        chemical_ax.set_axisbelow(True)
        colorbar = fig.colorbar(image, cax=cax, ticks=[0, limit / 2, limit])
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=2, labelsize=8)
        colorbar.set_label("|Mean paired Δamplitude| (ΔF/F₀)", fontsize=9, labelpad=8)
        handles = [Patch(facecolor=MISSING, edgecolor="none", label="Fewer than 3 paired animals"),
                   Line2D([], [], marker="o", linestyle="none", color=INK,
                          markeredgecolor="white", markeredgewidth=.45, markersize=4,
                          label="Positive energy after every animal omission (influence check)")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05, .13 / height),
                   frameon=False, ncol=1, fontsize=8, handlelength=1.1, labelspacing=.5)
        for ext in ("png", "svg"):
            fig.savefig(paths[ext], dpi=200, facecolor="white", bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
    values["color_scale_min"] = 0.0
    values["color_scale_max"] = limit
    values.to_csv(paths["values"], index=False)
    caption = (
        "Amplitude differences in chemically selected pairs (analysis inspection map). "
        f"The {len(pairs)} rows comprise {n_candidates} candidates from the Figure 5 "
        f"chemistry-only shortlist and {n_benchmarks} separately designated benchmark "
        "pair(s). They are exactly selected_main=True in tables/pair_summary.csv, ordered by "
        "joint_rms_log2fc and then pair_id; A231/A232 is bold. No neural-effect threshold "
        "is applied to row selection. Chemical RMS is the root-mean-square difference "
        "between log2 fold changes over features jointly reported for each pair; the "
        "feature mask and its size can differ among pairs. n_joint_reported records its "
        "size. Columns are the fixed seven cell classes used in Figures 4 and 5. "
        "Rows may share acquisition dates, animal sets or strains; they are descriptive "
        "comparisons and are not treated as independent replicates.\n\n"
        "Each heat-map value is the absolute mean within-animal amplitude difference "
        "between the two strains. Signed differences in the exported table use the "
        "arbitrary lexicographic strain_a minus strain_b convention. The fixed full-data "
        "0–40 s templates have RMS=1, so amplitude differences retain ΔF/F₀ units. "
        "No new template fitting is performed. n_animals counts paired animals with valid "
        "amplitudes for both strains in the pair's acquisition date and cell class. "
        "Cells with n<3 are gray and are masked, not set to zero. Pair-cell masks, animal "
        "counts and SEM are retained in amplitude_difference_map_values.csv. "
        f"All columns share the range 0–{limit:g} ΔF/F₀, obtained by rounding the global "
        "displayed maximum upward to 0.1 (minimum upper limit 0.1). A small amplitude "
        "difference does not establish equivalence or preserved responses.\n\n"
        "A dot marks cross-animal energy that remains strictly positive after every "
        "leave-one-animal omission, with the same full-data template held fixed. "
        "This checks sensitivity to an individual animal; it is not a significance "
        "test or independent validation. Unmarked cells are not evidence of no difference."
    )
    paths["caption"].write_text(caption + "\n", encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}
