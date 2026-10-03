"""Poster figures of whole neighborhoods; no new model or pair selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1]
T, F = OUT / "tables", OUT / "figures"
CELLS = ["ASK","ADL","ASI","AWA","AWB","ASG","ADF","ASH","ASJ","ASEL","ASER","AWCON","AWCOFF"]
SAME, DIFFERENT = "#b46834", "#277c8e"


def save(fig, name):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(F / f"{name}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def atlas(pairs, cells, raw=False):
    rows = pairs[pairs.nearest_either.eq(1) & pairs.group_n_strains.ge(3)].copy()
    # Descriptive order only; it cannot count as independent evidence of groups.
    rows = rows.sort_values(["energy_fixed","pair_id"], ascending=[False, True])
    rows.to_csv(T / "poster_neighborhood_rows.csv", index=False)
    matrix = cells.pivot(index="pair_id", columns="neuron",
                         values="contribution_raw" if raw else "contribution_fixed").reindex(
                             index=rows.pair_id, columns=CELLS)
    matrix.to_csv(T / ("poster_cell_contributions_raw.csv" if raw else "poster_cell_contributions.csv"))
    fig = plt.figure(figsize=(13.4, 10.2), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.6, 6.1, 2.6], wspace=.05)
    chem = fig.add_subplot(grid[0, 0])
    heat = fig.add_subplot(grid[0, 1], sharey=chem)
    signal = fig.add_subplot(grid[0, 2], sharey=chem)
    y = np.arange(len(rows))
    colors = np.where(rows.same_species.eq(1), SAME, DIFFERENT)
    chem.scatter(rows.chemical_rms_log2fc, y, color=colors, s=30, zorder=3)
    chem.set(yticks=y, yticklabels=[f"{a} / {b}" for a,b in zip(rows.strain_a,rows.strain_b)],
             xlim=(1.65, 3.35), xticks=[2, 3], xlabel="Chemical difference\nRMS Δlog₂FC",
             title="Chemical neighbors")
    chem.set_ylim(len(rows)-.5, -.5)
    for ax in [chem, signal]:
        ax.grid(axis="y", color=".93", lw=.5)
        ax.tick_params(axis="y", length=0)
    # Signed energy, not activation/inhibition. Negative estimates remain on
    # the color scale. Missing cells receive a separate background and cross.
    lo, hi = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    extent = max(abs(lo), abs(hi), 1e-8)
    cmap = LinearSegmentedColormap.from_list("signed_component",
                                             ["#686868","#fafafa","#00576c"])
    cmap.set_bad("#eee6d8")
    # Symmetric units on the two sides: equal magnitude has equal distance
    # from zero in the color scale. Do not visually amplify small negatives.
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)
    im = heat.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, norm=norm,
                     interpolation="nearest")
    for r, c in zip(*np.where(matrix.isna().to_numpy())):
        heat.text(c, r, "×", ha="center", va="center", color="#7a6b54", fontsize=12)
    heat.set_xticks(range(13), CELLS, rotation=55, ha="right")
    heat.tick_params(axis="y", left=False, labelleft=False)
    heat.set_title("Where the response difference occurs")
    heat.set_xlabel("All 13 neuron classes")
    bar = fig.colorbar(im, ax=heat, orientation="horizontal", shrink=.82, pad=.03,
                      label="Contribution to neural separation (ΔF/F₀)²" if raw else
                            "Contribution to neural separation (cell-scaled units²)")
    if not raw:
        bar.set_ticks([-.8, -.4, 0, .4, .8])
    stem = "raw" if raw else "fixed"
    for i, row in enumerate(rows.itertuples()):
        value = getattr(row, f"energy_{stem}")
        lower = getattr(row, f"loo_{stem}_min")
        upper = getattr(row, f"loo_{stem}_max")
        signal.plot([lower, upper], [i, i], color=colors[i], lw=1.5, alpha=.52)
        signal.scatter(value, i, color=colors[i], s=28, zorder=3)
    signal.axvline(0, color=".5", lw=.8, ls="--")
    signal.set_xlabel("Cross-animal neural separation\n" +
                      ("(ΔF/F₀)²" if raw else "(cell-scaled units²)"))
    signal.set_title("Population difference")
    signal.tick_params(axis="y", left=False, labelleft=False)
    if not raw:
        signal.set_xlim(-.3, 3.2)
        signal.set_xticks([0, 1, 2, 3])
    fig.suptitle(("Chemical neighbors have heterogeneous population response differences" if not raw else
                  "The same chemical-neighbor pairs in original calcium units") +
                 "\n28 chemistry-selected pairs · 40 strains · 38 animals", fontsize=16)
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=SAME,
                      markeredgecolor="none", label="Same species (7 pairs)"),
               Line2D([], [], marker="o", color="none", markerfacecolor=DIFFERENT,
                      markeredgecolor="none", label="Different species (21 pairs)")]
    chem.legend(handles=handles, frameon=False, loc="lower left",
                bbox_to_anchor=(0, -.21), fontsize=9)
    fig.supxlabel("Only sets with ≥3 candidate strains; each row is a chemical-neighbor pair, ordered by the observed neural score.\n"
                  "26 pairs: 13 eligible cells; 2 pairs: 10. Population score = mean across eligible cells.\n"
                  "Bars: delete-one-animal range, not confidence intervals. Gray values: negative estimates; ×: fewer than 3 paired animals.\n"
                  "Cell colors show contribution to separation, not activation or inhibition. Small estimates do not establish equivalent responses.",
                  fontsize=9)
    save(fig, "poster_neighborhood_atlas_raw" if raw else "poster_neighborhood_atlas")


def landscape(pairs):
    chosen = pairs.nearest_either.eq(1) & pairs.group_n_strains.ge(3)
    fig, ax = plt.subplots(figsize=(9.5, 6.6), layout="constrained")
    background = pairs[~chosen]
    ax.scatter(background.chemical_rms_log2fc, background.energy_fixed,
               s=24, color="#c5c9ca", alpha=.65, edgecolor="none",
               label="Other comparable pairs", zorder=1)
    for same, label, color in [(0,"Chemical neighbors: different species",DIFFERENT),
                                (1,"Chemical neighbors: same species",SAME)]:
        z = pairs[chosen & pairs.same_species.eq(same)]
        ax.vlines(z.chemical_rms_log2fc, z.loo_fixed_min, z.loo_fixed_max,
                  color=color, alpha=.28, lw=1)
        ax.scatter(z.chemical_rms_log2fc, z.energy_fixed, s=49, color=color,
                   edgecolor="white", linewidth=.6, label=label, zorder=3)
    ax.axhline(0, color=".45", lw=.85, ls="--")
    ax.set(xlabel="Chemical profile difference\nRMS Δlog₂FC across all 380 features",
           ylabel="Cross-animal neural separation\n(cell-scaled calcium units²)",
           title="Chemical neighbors span a range of neural differences",
           xlim=(1.62, 3.69))
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.supxlabel("147 within-genus, reference-matched comparisons; 28 non-forced chemical-neighbor pairs highlighted.\n"
                  "Each point is a strain pair, not an independent replicate. Lines: delete-one-animal ranges, not confidence intervals.\n"
                  "No fitted global correlation; a low or negative estimate is not evidence of biological equivalence.", fontsize=9)
    pairs.to_csv(T / "poster_landscape_points.csv", index=False)
    save(fig, "poster_chemical_neural_landscape")


def species(pairs):
    groups = pd.read_csv(T / "pair_context_pattern_groups.csv", dtype={"date":str})
    g = groups[(groups.neural_coverage == "available_cells_147") &
               (groups.min_pairs_per_group == 3) &
               (groups.effect == "same_minus_different_species") &
               (groups.covariate == "energy")].copy()
    g = g.sort_values(["genus","n_strains","comparison_group"])
    g.to_csv(T / "poster_species_group_effects.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.3, 5.6), layout="constrained")
    y = np.arange(len(g))
    x = g.estimate
    ax.hlines(y, 0, x, colors=np.where(x < 0, DIFFERENT, SAME), lw=2, alpha=.5)
    ax.scatter(x, y, color=np.where(x < 0, DIFFERENT, SAME), s=60, zorder=3)
    labels = [f"{r.genus} ({r.n_strains} strains; {r.n_same_species}/{r.n_different_species} pairs)"
              for r in g.itertuples()]
    ax.set(yticks=y, yticklabels=labels, ylim=(len(g)-.6,-.6), xlim=(-1.05,.2),
           xlabel="Mean neural separation: same species − different species\n"
                  "(cell-scaled calcium units²)\n"
                  "← Same species closer                 Same species farther →",
           title="Species identity gives a tendency, not a boundary")
    ax.axvline(0, color=".55", lw=.8)
    ax.grid(axis="x", color=".94")
    fig.supxlabel("Seven matched comparison sets contain both pair types; labels show same-species / different-species pair counts.\n"
                  "Means use all comparable pairs within each set, not only chemical neighbors.\n"
                  "Pairs share animals and strains. This is a descriptive group comparison without p-values.", fontsize=9)
    save(fig, "poster_species_tendency")


def main():
    F.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11, "axes.titlesize":12, "axes.spines.top":False,
                         "axes.spines.right":False, "pdf.fonttype":42, "svg.fonttype":"none"})
    pairs = pd.read_csv(T / "pair_signal_summary.csv")
    cells = pd.read_csv(T / "pair_signal_percell.csv")
    assert len(pairs) == 147
    for _, g in cells.groupby("pair_id"):
        p = pairs.set_index("pair_id").loc[g.pair_id.iloc[0]]
        assert np.isclose(g.contribution_fixed.sum(), p.energy_fixed, atol=1e-12)
    atlas(pairs, cells)
    landscape(pairs)
    species(pairs)
    atlas(pairs, cells, raw=True)
    (OUT / "logs/figure_verification.json").write_text(json.dumps(dict(
        status="passed", n_pairs=147, highlighted_pairs=28, neuron_classes=CELLS,
        figure_pairs_selected_by_chemistry_only=True,
        sum_cell_contributions_equals_population_score=True,
        interval="Leave-one-whole-animal range, fixed cell scales/panel; not confidence intervals",
        row_sort="Observed population score; descriptive order, not clustering or independent validation",
        missing="No zero imputation; x marks cells with fewer than three paired animals",
        formats=["png","pdf","svg"]), indent=2))
    print("Rendered four figures: main atlas, landscape context, species tendency, raw-unit sensitivity.")


if __name__ == "__main__":
    main()
