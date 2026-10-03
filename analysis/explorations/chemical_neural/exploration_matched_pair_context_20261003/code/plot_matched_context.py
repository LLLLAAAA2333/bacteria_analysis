"""Render saved strict-context pairs and every qualifying matched triplet.

This script does not select cases or fit scientific models. The saved case order,
global labels, and top-18 chemical rankings are retained. Run with .pixi Python.
"""
from pathlib import Path
import hashlib
import json

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
PDF = REPO / "output/pdf/neural_chemical_matched_context.pdf"
INK, GRAY, LIGHT = "#26343E", "#66717A", "#E5E9EC"
ANCHOR, BLUE, RED = "#7B838B", "#226CA7", "#C64245"
FIGSIZE = (11.7, 8.3)


def padded(values, fraction=0.08, include_zero=False):
    values = np.asarray(values, float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if include_zero:
        lo, hi = min(lo, 0), max(hi, 0)
    margin = (hi - lo) * fraction if hi > lo else max(abs(lo) * fraction, 0.1)
    return lo - margin, hi + margin


def style_axis(ax, small=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A9B1B8")
    ax.tick_params(labelsize=7.2 if small else 8.1, length=3, colors=GRAY)
    ax.yaxis.set_major_locator(MaxNLocator(4 if small else 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=LIGHT, linewidth=0.5)


def date_label(value):
    return str(value).replace("|", " / ")


def page_footer(fig, page, line1, line2):
    fig.text(0.055, 0.048, line1, fontsize=7.1, color=GRAY)
    fig.text(0.055, 0.027, line2, fontsize=7.1, color=GRAY)
    fig.text(0.965, 0.027, str(page), fontsize=7.4, color=GRAY, ha="right")


def profile_panel(fig, rect, data, samples, colors, title, ylabel, unit=False):
    ax = fig.add_axes(rect)
    values = data.loc[samples].to_numpy(float)
    count = len(samples)
    x = np.arange(data.shape[1])
    width = 0.75 / count
    for j, (sample, color) in enumerate(zip(samples, colors)):
        ax.bar(x + (j - (count - 1) / 2) * width, values[j], width=width,
               color=color, edgecolor="none", label=sample)
    ax.set_xlim(-0.65, data.shape[1] - 0.35)
    ax.set_xticks(x, data.columns, rotation=52, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9.5, pad=9)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=5)
    ax.set_ylim((-1.05, 1.05) if unit else padded(values, include_zero=True))
    ax.axhline(0, color=GRAY, linewidth=0.55)
    style_axis(ax, small=True)
    assert len(ax.patches) == len(samples) * 13
    assert np.allclose(np.concatenate([values[j] for j in range(count)]),
                       [p.get_height() for p in ax.patches])
    return ax


def change_scatter(fig, frame, title="Chemical changes (162)"):
    ax = fig.add_axes([0.795, 0.496, 0.168, 0.232])
    ax.scatter(frame.delta_b, frame.delta_c, s=9, c=ANCHOR, alpha=0.72,
               edgecolors="none", zorder=3)
    limit = max(np.max(np.abs(frame.delta_b)), np.max(np.abs(frame.delta_c))) * 1.08
    ax.plot([-limit, limit], [-limit, limit], color=GRAY, linewidth=0.65)
    ax.axhline(0, color=LIGHT, linewidth=0.7)
    ax.axvline(0, color=LIGHT, linewidth=0.7)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9.2, pad=9)
    ax.set_xlabel("B - anchor (log₂ change)", fontsize=8, color=BLUE)
    ax.set_ylabel("C - anchor (log₂ change)", fontsize=8, color=RED, labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    style_axis(ax, small=True)
    assert len(ax.collections[0].get_offsets()) == 162
    return ax


def chemical_dots(fig, frame, title, single=False):
    """Use the saved common top-18 list, without changing feature identities."""
    top = frame.loc[frame.display_rank.le(18)].sort_values("display_rank")
    assert len(top) == 18 and top.feature.is_unique
    fig.text(0.055, 0.405, title, fontsize=10, color=INK, weight="bold")
    ax = fig.add_axes([0.317, 0.131, 0.632, 0.249])
    y = np.arange(len(top))
    if single:
        ax.hlines(y, 0, top.delta_b, color="#B8C0C7", linewidth=0.8)
        ax.scatter(top.delta_b, y, c=RED, s=16, edgecolors="none", zorder=3)
        values = top.delta_b.to_numpy()
    else:
        ax.hlines(y, top.delta_b, top.delta_c, color="#A8B1B8", linewidth=0.8)
        ax.scatter(top.delta_b, y, c=BLUE, s=16, edgecolors="none", zorder=3)
        ax.scatter(top.delta_c, y, c=RED, s=16, edgecolors="none", zorder=3)
        values = top[["delta_b", "delta_c"]].to_numpy()
        handles = [Line2D([], [], color=BLUE, marker="o", linestyle="", label="B - anchor", markersize=4),
                   Line2D([], [], color=RED, marker="o", linestyle="", label="C - anchor", markersize=4)]
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.952, 0.409),
                   ncol=2, frameon=False, fontsize=7.8, handletextpad=0.3, columnspacing=1.3)
    ax.set_yticks(y, top.feature.str.strip(), fontsize=6.8)
    ax.set_ylim(17.65, -0.65)
    ax.set_xlim(padded(values, include_zero=True))
    ax.axvline(0, color=GRAY, linewidth=0.65)
    ax.set_xlabel("Signed change in log₂(reported value + 1)", fontsize=8.2, labelpad=4)
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#A9B1B8")
    ax.tick_params(axis="y", length=0, pad=7, colors=INK)
    ax.tick_params(axis="x", labelsize=7.2, colors=GRAY)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=LIGHT, linewidth=0.5)
    assert [t.get_text() for t in ax.get_yticklabels()] == top.feature.str.strip().tolist()
    return top


def guide(parameters, observations):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    fig.text(0.065, 0.93, "Chemical-distance matching within strict backgrounds",
             fontsize=22, weight="bold", color=INK)
    fig.text(0.065, 0.883, "Individual pairs and shared-anchor triplets for inspection", fontsize=13, color=GRAY)
    fig.text(0.065, 0.815, "100 comparable pairs  /  121 matched triplets  /  31 strains in 6 matched contexts",
             fontsize=11.5, color=INK)
    sections = [
        ("Strict scope", [
            "All three strains share the same reference, complete neural recording-date set, and genus.",
            "B and C share anchor A, and their chemical distances to A differ by no more than 10%:",
            "mismatch = 2 x |d(A,B) - d(A,C)| / [d(A,B) + d(A,C)]. All 121 qualifying triplets are shown."]),
        ("What is present in these data", [
            "Among the 100 strict-background pairs, global neural labels are 82 near, 17 middle, and 1 far.",
            "No strict matched triplet joins the global near and far tails at the 25% cutoffs.",
            "The displayed triplets comprise 85 near/near, 28 near/middle, and 8 middle/middle cases.",
            "Coverage is uneven: 112 / 121 cases are Bacteroides; 79 are from A050 / 20260601.",
            "Page 3 retains the sole far pair, A291 / A296, which has no qualifying matched triplet."]),
        ("Reading the case pages", [
            "B has the lower neural distance to A; C has the higher. This ordering does not make C a far-tail sample.",
            "Gray = anchor; blue = B; red = C. Unit profiles divide the 13-cell coefficient vector by its L2 norm.",
            "Chemical panels show signed partner-minus-anchor changes for the same 162 fully reported chemicals.",
            "The common top 18 is ranked by sqrt[(change B squared + change C squared) / 2]."]),
        ("Measures and limits", [
            "Neural distance = 1 - cosine of SNR >= 0.5 template coefficients; screened zeros do not establish absence.",
            "Valid-bootstrap-draw fractions report available comparisons, not confidence or a reliability score.",
            "Chemical distance = RMS difference of log2(original reported value + 1); no chemical values are imputed.",
            "Similar chemical distance lengths do not establish matching chemical directions or compositions.",
            "Same genus is not necessarily same species; matching does not remove all experimental confounding.",
            "The assays do not share an actual stimulus aliquot. Cases reuse strains and are not independent replicates.",
            "This is descriptive inspection: no significance, causal, equivalence, or performance claim."]),
    ]
    y = 0.755
    for title, lines in sections:
        fig.text(0.065, y, title, fontsize=11.1, weight="bold", color=INK)
        y -= 0.030
        for line in lines:
            fig.text(0.065, y, line, fontsize=9.0, color=INK)
            y -= 0.025
        y -= 0.019
    page_footer(fig, 1, "Case order follows reference, full date support, genus, anchor, B, and C. No cases were selected by neural gap.",
                "3 October 2026  |  Chemical identities are report annotations; reference IDs are numerical denominator groups.")
    return fig


def support_page(pairs, parameters):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    fig.text(0.065, 0.93, "All pairs within the strict background", fontsize=23, weight="bold", color=INK)
    fig.text(0.065, 0.884, "100 pairs: same reference, full date-support set, and genus", fontsize=12.4, color=GRAY)
    ax = fig.add_axes([0.095, 0.20, 0.72, 0.585])
    far = pairs.neural_label.eq("far")
    ax.scatter(pairs.loc[~far, "chemical_distance"], pairs.loc[~far, "neural_distance"],
               s=27, color=BLUE, alpha=0.70, linewidths=0.4, edgecolors="white", zorder=3)
    ax.scatter(pairs.loc[far, "chemical_distance"], pairs.loc[far, "neural_distance"],
               s=55, color=RED, edgecolors="white", linewidths=0.7, zorder=4)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 100
    near, distant = parameters["neural_thresholds"]["0.25"]
    for threshold in (near, distant):
        ax.axhline(threshold, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.85)
    for threshold in parameters["chemical_thresholds"]:
        ax.axvline(threshold, color="#9DA6AF", linestyle=(0, (2, 3)), linewidth=0.75)
    ax.set_xlim(0, max(pairs.chemical_distance.max() * 1.08, parameters["chemical_thresholds"][1] * 1.15))
    ax.set_ylim(-0.025, max(1.0, pairs.neural_distance.max() * 1.1))
    ax.set_xlabel("Chemical distance (RMS log₂ difference; 162 chemicals)", fontsize=10, labelpad=8)
    ax.set_ylabel("Neural distance (1 - cosine)", fontsize=10, labelpad=8)
    style_axis(ax)
    for label, y in (("Near: 82 pairs", near / 2),
                     ("Middle: 17 pairs", (near + distant) / 2),
                     ("Far: 1 pair", (distant + 1.0) / 2)):
        ax.text(1.025, y, label, transform=ax.get_yaxis_transform(), fontsize=9.1, color=INK, va="center")
    only_far = pairs.loc[far].iloc[0]
    ax.annotate("A291 / A296", (only_far.chemical_distance, only_far.neural_distance),
                xytext=(12, 5), textcoords="offset points", fontsize=9, color=RED)
    fig.text(0.095, 0.121,
             "Dashed cutoffs use all 5,565 pairs: neural near <= 0.283, far >= 0.700; chemical near <= 1.203, far >= 1.647.",
             fontsize=8.4, color=GRAY)
    page_footer(fig, 2, "Each point is a pair, not an independent replicate; strains can contribute to multiple points.",
                "No group averaging. These background restrictions leave no chemical-matched global near/far triplet at the 25% cutoffs.")
    return fig


def case_page(row, data):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    samples = [row.anchor, row.partner_b, row.partner_c]
    colors = [ANCHOR, BLUE, RED]
    frame = data["changes"][data["changes"].case_id.eq(row.case_id)]
    assert len(frame) == 162 and frame.feature.is_unique
    context = data["context"].loc[samples]
    assert context.reference.nunique() == context.dates.nunique() == context.genus.nunique() == 1
    assert row.all_same_genus and row.neural_distance_b <= row.neural_distance_c
    assert row.chemical_mismatch <= 0.1 + 1e-12
    chem = data["chemical"].loc[samples, frame.feature]
    assert np.allclose(chem.iloc[1] - chem.iloc[0], frame.delta_b)
    assert np.allclose(chem.iloc[2] - chem.iloc[0], frame.delta_c)
    assert np.isclose(np.sqrt(np.mean(frame.delta_b ** 2)), row.chemical_distance_b)
    assert np.isclose(np.sqrt(np.mean(frame.delta_c ** 2)), row.chemical_distance_c)
    assert np.allclose(np.sqrt((frame.delta_b ** 2 + frame.delta_c ** 2) / 2), frame.joint_change_rms)
    units = data["unit"].loc[samples].to_numpy()
    assert np.isclose(1 - units[0] @ units[1], row.neural_distance_b)
    assert np.isclose(1 - units[0] @ units[2], row.neural_distance_c)
    fig.text(0.055, 0.949, f"{row.case_id}  |  Anchor {row.anchor}  /  B {row.partner_b}  /  C {row.partner_c}",
             fontsize=17.5, weight="bold", color=INK)
    fig.text(0.055, 0.911, f"All three: {row.anchor_genus}  |  Reference {row.reference}  |  Recording dates {date_label(row.dates)}",
             fontsize=9.3, color=INK)
    for x, sample, color, role in zip([0.055, 0.373, 0.694], samples, colors, ["Anchor", "Partner B", "Partner C"]):
        norm = float(data["context"].loc[sample, "coefficient_norm"])
        fig.text(x, 0.876, f"{role}: {sample}     norm = {norm:.3f}", fontsize=8.2, weight="bold", color=color)
        fig.text(x, 0.854, str(data["context"].loc[sample, "species"]), fontsize=7.5, color=color)
    fig.text(0.055, 0.819,
             f"B: chemical {row.chemical_distance_b:.3f} [{row.chemical_label_b}]   |   neural {row.neural_distance_b:.4f} [{row.neural_label_b}]",
             fontsize=8.5, color=BLUE)
    fig.text(0.524, 0.819,
             f"C: chemical {row.chemical_distance_c:.3f} [{row.chemical_label_c}]   |   neural {row.neural_distance_c:.4f} [{row.neural_label_c}]",
             fontsize=8.5, color=RED)
    fig.text(0.055, 0.789,
             f"Chemical mismatch {row.chemical_mismatch:.1%}   |   Change-direction cosine {row.chemical_change_cosine:.3f}"
             f"   |   Neural gap screened / unfiltered: {row.neural_gap:+.3f} / {row.neural_gap_unfiltered:+.3f}"
             f"   |   Unfiltered B / C: {row.neural_unfiltered_b:.3f} / {row.neural_unfiltered_c:.3f}",
             fontsize=7.6, color=GRAY)
    profile_panel(fig, [0.069, 0.496, 0.274, 0.232], data["coefficient"], samples, colors,
                  "Signed neural coefficients", "Coefficient (ΔF/F₀)")
    profile_panel(fig, [0.427, 0.496, 0.274, 0.232], data["unit"], samples, colors,
                  "Unit-normalized neural profiles", "Unit coefficient", unit=True)
    change_scatter(fig, frame)
    chemical_dots(fig, frame, "Top 18 chemical changes")
    valid_b = f"{row.bootstrap_valid_fraction_b:.1%}" if np.isfinite(row.bootstrap_valid_fraction_b) else "unavailable"
    valid_c = f"{row.bootstrap_valid_fraction_c:.1%}" if np.isfinite(row.bootstrap_valid_fraction_c) else "unavailable"
    page_footer(fig, int(row.pdf_page),
                "Same features in both arms; ranked by joint RMS change. Matching chemical distance length does not match full composition.",
                f"Valid bootstrap draws B/C: {valid_b} / {valid_c}. B/C order is relative neural distance; cases reuse strains.")
    return fig


def far_pair_page(row, data):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    samples = [row.strain_a, row.strain_b]
    colors = [BLUE, RED]
    chemical = data["chemical"].loc[samples]
    delta = chemical.iloc[1] - chemical.iloc[0]
    order = np.argsort(-np.abs(delta.to_numpy()), kind="stable")
    changes = pd.DataFrame({"feature": chemical.columns[order], "delta_b": delta.iloc[order].to_numpy(),
                            "display_rank": np.arange(1, 163)})
    fig.text(0.055, 0.949, f"{row.strain_a}  /  {row.strain_b}  |  The sole strict-background far pair",
             fontsize=17.5, weight="bold", color=INK)
    fig.text(0.055, 0.911,
             f"Same genus: {row.genus_a}  |  Reference {row.reference_a}  |  Recording dates {date_label(row.dates_a)}",
             fontsize=9.3, color=INK)
    for x, sample, color in zip([0.055, 0.524], samples, colors):
        fig.text(x, 0.876, f"{sample}     norm = {data['context'].loc[sample, 'coefficient_norm']:.3f}",
                 fontsize=8.5, weight="bold", color=color)
        fig.text(x, 0.854, str(data["context"].loc[sample, "species"]), fontsize=8.0, color=color)
    fig.text(0.055, 0.819,
             f"Chemical distance {row.chemical_distance:.3f} [{row.chemical_label}]   |   Neural distance {row.neural_distance:.4f} [{row.neural_label}]"
             f"   |   Unfiltered neural distance {row.neural_unfiltered:.4f}", fontsize=8.5, color=INK)
    fig.text(0.055, 0.789, "No qualifying third strain forms a strict matched triplet with this pair at the 10% chemical-distance caliper.",
             fontsize=8.2, color=GRAY)
    profile_panel(fig, [0.069, 0.496, 0.274, 0.232], data["coefficient"], samples, colors,
                  "Signed neural coefficients", "Coefficient (ΔF/F₀)")
    profile_panel(fig, [0.427, 0.496, 0.274, 0.232], data["unit"], samples, colors,
                  "Unit-normalized neural profiles", "Unit coefficient", unit=True)
    ax = fig.add_axes([0.795, 0.496, 0.168, 0.232])
    ax.scatter(chemical.iloc[0], chemical.iloc[1], s=9, c=ANCHOR, alpha=0.72, edgecolors="none")
    limits = padded(chemical.to_numpy())
    ax.plot(limits, limits, color=GRAY, linewidth=0.65)
    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Chemical profiles (162)", fontsize=9.2, pad=9)
    ax.set_xlabel(f"{samples[0]} log₂(value + 1)", fontsize=7.5, color=BLUE)
    ax.set_ylabel(f"{samples[1]} log₂(value + 1)", fontsize=7.5, color=RED, labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    style_axis(ax, small=True)
    chemical_dots(fig, changes, f"Largest 18 chemical differences  |  {samples[1]} minus {samples[0]}", single=True)
    page_footer(fig, 3, "All 162 chemical values were originally reported. The top 18 are ranked by absolute pairwise change.",
                "An unmatched pair provides an inspection case, not evidence from a chemical-distance matched near/far comparison.")
    return fig


def load_data():
    names = {"cases": "matched_cases.csv", "pairs": "strict_context_pairs.csv", "far": "unmatched_far_pairs.csv",
             "coefficient": "neural_coefficients.csv", "unit": "neural_unit_coefficients.csv",
             "context": "sample_context.csv", "chemical": "chemical_log2_report_plus1.csv",
             "changes": "case_chemical_changes.csv"}
    data = {}
    hashes = {}
    indexed = {"coefficient", "unit", "context", "chemical"}
    for key, name in names.items():
        path = OUT / "tables" / name
        data[key] = pd.read_csv(path, index_col=0 if key in indexed else None,
                                dtype={"dates": str, "dates_a": str, "dates_b": str})
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(data["cases"]) == 121
    assert data["cases"].pdf_page.tolist() == list(range(4, 125))
    assert len(data["pairs"]) == 100 and len(data["far"]) == 1
    assert data["chemical"].shape == (106, 162)
    assert data["coefficient"].shape == data["unit"].shape == (106, 13)
    return data, hashes


def save_preview(fig, stem):
    fig.savefig(FIGURES / f"{stem}.png", dpi=180)
    fig.savefig(FIGURES / f"{stem}.svg")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    data, hashes = load_data()
    parameters = json.loads((OUT / "parameters.json").read_text())
    observations = json.loads((OUT / "observations.json").read_text())
    FIGURES.mkdir(exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    # Previews include all context starts and the three largest gaps. PDF includes
    # every case in the saved order; preview choices do not change its content.
    starts = data["cases"].groupby(["reference", "dates", "anchor_genus"], sort=False).head(1).case_id.tolist()
    largest = data["cases"].nlargest(3, "neural_gap").case_id.tolist()
    previews = set(starts + largest)
    with PdfPages(PDF, metadata={"Title": "Neural and chemical comparison in strict matched contexts",
                                "Author": "Bacteria analysis - exploratory figures"}) as pdf:
        fig = guide(parameters, observations); pdf.savefig(fig); save_preview(fig, "00_reading_guide"); plt.close(fig)
        fig = support_page(data["pairs"], parameters); pdf.savefig(fig); save_preview(fig, "01_strict_pair_support"); plt.close(fig)
        fig = far_pair_page(data["far"].iloc[0], data); pdf.savefig(fig); save_preview(fig, "02_unmatched_far_pair"); plt.close(fig)
        for row in data["cases"].itertuples(index=False):
            fig = case_page(row, data)
            pdf.savefig(fig)
            if row.case_id in previews:
                save_preview(fig, f"{int(row.pdf_page):03d}_{row.case_id}_{row.anchor}_{row.partner_b}_{row.partner_c}")
            plt.close(fig)
            if row.pdf_page % 20 == 0:
                print(f"Rendered {row.pdf_page}/124 pages", flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
    verification = {"pdf": str(PDF), "n_pages": 124, "n_case_pages": 121,
                    "case_order_identical_to_saved_table": True,
                    "all_case_chemical_changes_and_distances_verified": True,
                    "all_case_neural_distances_verified": True,
                    "all_case_genus_reference_full_date_sets_verified": True,
                    "all_case_shared_top18_names_preserved": True,
                    "all_13_raw_and_unit_coefficients_included": True,
                    "all_162_chemical_changes_included": True,
                    "support_panel_pair_count": 100,
                    "preview_context_start_cases": starts, "preview_largest_gap_cases": largest,
                    "preview_selection_does_not_change_complete_pdf": True,
                    "source_sha256": hashes, "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest()}
    (FIGURES / "plot_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({k: verification[k] for k in ["pdf", "n_pages", "n_case_pages", "pdf_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
