"""Plot saved exploratory associations; this script does not select or fit them.

Run with the repository's .pixi Python. Each candidate page retains all 106
strains. Four reference panels share a coefficient axis but have independent
chemical axes because their log2FC values use different reference samples.
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
FIGURES = OUT / "figures"
PDF = REPO / "output/pdf/neural_compound_relation_candidates.pdf"
REFERENCES = ("A050", "A250", "A306", "ref12")
COLORS = dict(zip(REFERENCES, ("#226CA7", "#BE7A28", "#269084", "#905D9C")))
INK = "#26343E"
GRAY = "#66717A"


def padded_limits(values, fraction=0.08):
    """Include every value with modest padding, including constant vectors."""
    values = np.asarray(values, float)
    low, high = np.nanmin(values), np.nanmax(values)
    span = high - low
    pad = fraction * span if span > 0 else max(abs(low) * fraction, 0.1)
    return low - pad, high + pad


def style_axis(ax, small=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#ABB3BA")
    ax.tick_params(colors=GRAY, labelsize=7.6 if small else 8.3, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(4 if small else 5))
    ax.yaxis.set_major_locator(MaxNLocator(4 if small else 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8EBEE", lw=0.55)


def draw_points(ax, frame, x, y, references=REFERENCES, size=24):
    for ref in references:
        group = frame[frame.reference_group.eq(ref)]
        ax.scatter(group[x], group[y], s=size, c=COLORS[ref],
                   alpha=0.83, edgecolors="white", linewidths=0.35,
                   label=ref, zorder=3)
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(frame)


def guide(candidates, parameters):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(0.065, 0.925, "Neural responses and annotated chemicals", fontsize=23,
             color=INK, weight="bold")
    fig.text(0.065, 0.884, "Twelve candidate relationships for individual inspection",
             fontsize=14, color=GRAY)
    fig.text(0.065, 0.82, "106 strains  /  162 fully reported chemicals  /  13 cells",
             fontsize=12, color=INK)
    sections = [
        ("Reading each page", [
            "Top left: original chemical log2FC and signed template coefficient, colored by chemical reference.",
            "Top right: global ranks after removing reference and recording-date effects from both variables.",
            "Bottom: each reference group separately; y scales match, while chemical x scales are group-specific.",
            "Every point is one strain. All 106 strains are retained on each of the two upper panels."]),
        ("Association and response units", [
            "The displayed rho is the Pearson correlation of residual ranks, not a regression slope.",
            "Within-reference rho also adjusts for recording dates. Adjustments are the saved analysis outputs.",
            "Coefficient: signed amplitude along that cell's RMS-normalized response template, in delta F/F0.",
            "Displayed coefficients use SNR >= 0.5 screening; screened zeros do not establish absence of a response.",
            "Unit coefficient: the signed coefficient divided by the strain's 13-cell L2 norm; it is dimensionless."]),
        ("How these twelve were selected", [
            "Only chemicals reported in every strain and with report QC RSD <= 0.30 entered the primary screen.",
            "Candidates are ranked by the smallest absolute association across six ref/date-adjusted checks:",
            "coefficient, unit coefficient, each with genus adjustment, pre-gate coefficient, and true unfiltered coefficient.",
            "All six associations must have the same sign. The same data select and display these candidates."]),
        ("Interpretation", [
            "These are exploratory associations without independent validation or a significance claim.",
            "Genus adjustment uses the available within-genus variation; singleton genera add no residual information.",
            "Chemical identities are report annotations. A negative association does not by itself establish inhibition.",
            "Reference groups can overlap other study structure; the panels do not remove every possible confound."]),
    ]
    y = 0.756
    for title, lines in sections:
        fig.text(0.065, y, title, fontsize=11.4, color=INK, weight="bold")
        y -= 0.033
        for line in lines:
            fig.text(0.065, y, line, fontsize=9.4, color=INK)
            y -= 0.026
        y -= 0.021
    fig.text(0.065, 0.033, "2 October 2026  |  Source: saved strain-level exploratory tables", fontsize=8, color=GRAY)
    fig.text(0.955, 0.033, "1", fontsize=8, color=GRAY, ha="right")
    return fig


def candidate_page(candidate, points, within, page_number):
    feature, cell = candidate.feature, candidate.cell
    frame = points[points.feature.eq(feature) & points.cell.eq(cell)].copy()
    assert len(frame) == 106 and frame.sample_id.is_unique
    assert np.isfinite(frame[["chemical_log2fc", "coefficient", "chemical_rank_residual",
                               "coefficient_rank_residual"]].to_numpy()).all()
    residual_rho = np.corrcoef(frame.chemical_rank_residual, frame.coefficient_rank_residual)[0, 1]
    assert np.isclose(residual_rho, candidate["coefficient__reference_date"], atol=1e-12)
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(0.06, 0.946, f"{feature}  |  {cell}", fontsize=19, weight="bold", color=INK)
    subtitle = (f"n = 106    |    Ref + date: coefficient ρ = {candidate['coefficient__reference_date']:+.2f}; "
                f"unit ρ = {candidate['unit__reference_date']:+.2f}    |    + genus: "
                f"coefficient ρ = {candidate['coefficient__reference_date_genus']:+.2f}; "
                f"unit ρ = {candidate['unit__reference_date_genus']:+.2f}")
    fig.text(0.06, 0.906, subtitle, fontsize=9.4, color=INK)
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[ref],
                      markersize=5.4, label=ref) for ref in REFERENCES]
    fig.legend(handles=handles, title="Chemical reference", loc="upper center",
               bbox_to_anchor=(0.505, 0.886), ncol=4, frameon=False,
               fontsize=8.7, title_fontsize=8.2, handletextpad=0.35, columnspacing=2)

    raw = fig.add_axes([0.092, 0.513, 0.370, 0.276])
    adjusted = fig.add_axes([0.58, 0.513, 0.370, 0.276])
    draw_points(raw, frame, "chemical_log2fc", "coefficient")
    raw.set_title("Original chemical and neural values", fontsize=11, pad=11)
    raw.set_xlabel("Chemical value (log₂FC vs. reference)", fontsize=9, labelpad=7)
    raw.set_ylabel(f"{cell} coefficient (ΔF/F₀)", fontsize=9, labelpad=7)
    raw.set_xlim(padded_limits(frame.chemical_log2fc))
    coefficient_limits = padded_limits(frame.coefficient)
    raw.set_ylim(coefficient_limits)
    draw_points(adjusted, frame, "chemical_rank_residual", "coefficient_rank_residual")
    adjusted.set_title("After reference + date adjustment", fontsize=11, pad=11)
    adjusted.set_xlabel("Chemical rank residual", fontsize=9, labelpad=7)
    adjusted.set_ylabel("Coefficient rank residual", fontsize=9, labelpad=7)
    adjusted.set_xlim(padded_limits(frame.chemical_rank_residual))
    adjusted.set_ylim(padded_limits(frame.coefficient_rank_residual))
    for ax in (raw, adjusted):
        style_axis(ax)

    fig.text(0.092, 0.414, "Inspecting each reference group", fontsize=11, weight="bold", color=INK)
    fig.text(0.95, 0.414, "Shared y scale; separate x scales", fontsize=8.5, color=GRAY, ha="right")
    axes = []
    for i, ref in enumerate(REFERENCES):
        ax = fig.add_axes([0.092 + i * 0.226, 0.19, 0.178, 0.165])
        group = frame[frame.reference_group.eq(ref)]
        stats = within[within.feature.eq(feature) & within.cell.eq(cell) &
                       within.reference_group.eq(ref) & within.representation.eq("coefficient")]
        assert len(stats) == 1 and len(group) == int(stats.iloc[0].n_samples)
        rho = float(stats.iloc[0].rho)
        draw_points(ax, group, "chemical_log2fc", "coefficient", references=(ref,), size=24)
        ax.set_title(f"{ref}  |  n = {len(group)}\nDate-adjusted ρ = {rho:+.2f}",
                     fontsize=8.5, color=COLORS[ref], linespacing=1.55, pad=8)
        ax.set_xlabel("Chemical log₂FC", fontsize=8.2, labelpad=6)
        if i == 0:
            ax.set_ylabel("Coefficient (ΔF/F₀)", fontsize=8.2, labelpad=7)
        ax.set_xlim(padded_limits(group.chemical_log2fc, fraction=0.12))
        ax.set_ylim(coefficient_limits)
        style_axis(ax, small=True)
        axes.append(ax)
    assert all(ax.get_ylim() == coefficient_limits for ax in axes)
    assert sum(len(c.get_offsets()) for ax in axes for c in ax.collections) == 106
    fig.text(0.06, 0.089,
             "Selection: top 12 by minimum |ρ| across six same-data checks with matching signs (162 chemicals × 13 cells).",
             fontsize=8, color=GRAY)
    fig.text(0.06, 0.064,
             "Exploratory; report annotations; no independent validation. Negative associations do not by themselves establish inhibition.",
             fontsize=8, color=GRAY)
    fig.text(0.06, 0.034, f"Candidate {int(candidate.display_rank)} of 12  |  All chemical values originally reported  |  Neural SNR >= 0.5",
             fontsize=8, color=GRAY)
    fig.text(0.955, 0.034, str(page_number), fontsize=8, color=GRAY, ha="right")
    return fig


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    tables = OUT / "tables"
    files = [tables / name for name in ("selected_candidates.csv", "candidate_points.csv",
                                         "within_reference_associations.csv")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    candidates, points, within = (pd.read_csv(path) for path in files)
    candidates = candidates.sort_values("display_rank")
    parameters = json.loads((OUT / "parameters.json").read_text())
    assert len(candidates) == 12
    assert set(points.reference_group.unique()) == set(REFERENCES)
    FIGURES.mkdir(exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF, metadata={"Title": "Neural and chemical candidate relationships",
                                "Author": "Bacteria analysis - exploratory figures"}) as pdf:
        fig = guide(candidates, parameters)
        pdf.savefig(fig)
        fig.savefig(FIGURES / "00_reading_guide.png", dpi=180)
        fig.savefig(FIGURES / "00_reading_guide.svg")
        plt.close(fig)
        for _, candidate in candidates.iterrows():
            fig = candidate_page(candidate, points, within, int(candidate.display_rank) + 1)
            pdf.savefig(fig)
            if candidate.display_rank <= 6:
                stem = re.sub(r"[^a-z0-9]+", "_", f"{candidate.feature}_{candidate.cell}".lower()).strip("_")
                stem = f"{int(candidate.display_rank):02d}_{stem}"
                fig.savefig(FIGURES / f"{stem}.png", dpi=200)
                fig.savefig(FIGURES / f"{stem}.svg")
            plt.close(fig)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
    verification = {"pdf": str(PDF), "n_pages": 13, "n_candidates": 12,
                    "n_strains_each_upper_panel": 106,
                    "within_reference_panel_sample_counts": {r: int(points[points.feature.eq(candidates.iloc[0].feature) &
                                                    points.cell.eq(candidates.iloc[0].cell)].reference_group.eq(r).sum()) for r in REFERENCES},
                    "residual_panel_correlations_match_saved_results": True,
                    "all_points_included_and_axis_limits_checked": True,
                    "same_y_limits_for_reference_panels": True,
                    "first_six_candidates_png_and_svg": True,
                    "source_sha256": hashes,
                    "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest()}
    (FIGURES / "plot_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
