"""Display observed profiles in five chemistry-defined groups; do not refit models."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


def make_figures(out: Path) -> list[Path]:
    """Save a group-mean main draft and its complete individual-strain companion.

    All 106 strains are sorted by the saved chemical score, then split into five
    near-equal rank groups, independently of neural values. Neural coefficients
    are centered across all strains without per-neuron variance scaling. Chemical
    log values are standardized across all strains solely for this display.
    The original discovery/holdout fits and their inputs are never changed.
    """
    out = Path(out)
    tables, figures = out / "tables", out / "figures"
    figures.mkdir(exist_ok=True)
    sources = [tables / name for name in (
        "sample_scores.csv", "selected_chemical_standardized.csv",
        "neural_unit_coefficients.csv", "neural_slopes.csv",
    )]
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sources}
    scores = pd.read_csv(sources[0], dtype={"strain": str}).sort_values(
        ["chemical_score", "strain"], kind="stable"
    )
    chemistry = pd.read_csv(sources[1], index_col=0)
    neural = pd.read_csv(sources[2], index_col=0)
    slopes = pd.read_csv(sources[3])
    cells = slopes.iloc[np.argsort(-slopes.discovery.abs().to_numpy(), kind="stable")].cell.tolist()
    ids = scores.strain.tolist()
    if (len(ids) != 106 or len(set(ids)) != 106 or len(cells) != 13
            or set(ids) != set(chemistry.index) or set(ids) != set(neural.index)):
        raise ValueError("Expected the complete saved 106-strain, 13-neuron panel.")
    chemistry, neural = chemistry.loc[ids], neural.loc[ids, cells]
    if not (np.isfinite(chemistry).all().all() and np.isfinite(neural).all().all()):
        raise ValueError("Missing values require an explicit display decision.")
    # Re-standardizing the saved affine-transformed log concentrations is exactly
    # equivalent to z-scoring the original log concentrations across these strains.
    chem_mean, chem_sd = chemistry.mean(), chemistry.std(ddof=1)
    if (chem_sd <= 0).any():
        raise ValueError("A chemical has no across-strain variation.")
    chemical_z = (chemistry - chem_mean) / chem_sd
    neural_mean = neural.mean()
    neural_centered = neural - neural_mean
    groups = np.array_split(np.arange(len(ids)), 5)
    names = ["Lowest", "Low", "Middle", "High", "Highest"]
    group_labels = [f"{name}\nn = {len(pos)}" for name, pos in zip(names, groups)]
    chem_group = pd.DataFrame({name: chemical_z.iloc[pos].mean()
                               for name, pos in zip(names, groups)})
    neural_group = pd.DataFrame({name: neural_centered.iloc[pos].mean()
                                 for name, pos in zip(names, groups)})
    membership = scores[["strain", "genus", "split", "chemical_score"]].copy()
    membership.insert(0, "display_column_1based", np.arange(1, len(ids) + 1))
    membership["chemical_group"] = np.concatenate([
        np.repeat(name, len(pos)) for name, pos in zip(names, groups)
    ])
    outputs = []
    for name, frame, use_index in (
        ("grouped_story_strain_groups.csv", membership, False),
        ("grouped_story_chemical_means.csv", chem_group, True),
        ("grouped_story_neural_means.csv", neural_group, True),
        ("grouped_story_neural_center.csv", neural_mean.rename("all_strain_mean").to_frame(), True),
    ):
        path = figures / name
        frame.to_csv(path, index=use_index)
        outputs.append(path)

    scales = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.titlesize": 14}):
        for individual in (False, True):
            key = "individuals" if individual else "means"
            cv = chemical_z.to_numpy().T if individual else chem_group.to_numpy()
            nv = neural_centered.to_numpy().T if individual else neural_group.to_numpy()
            limits = [float(np.abs(cv).max()), float(np.abs(nv).max())]
            scales[key] = {"chemical_symmetric_limit": limits[0],
                           "neural_symmetric_limit": limits[1]}
            fig = plt.figure(figsize=(12.0 if individual else 8.5, 8.6))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.025],
                                  height_ratios=[3, 13], hspace=0.43, wspace=0.075)
            fig.subplots_adjust(left=0.21 if not individual else 0.15,
                                right=0.84 if not individual else 0.89,
                                bottom=0.11, top=0.82)
            axc, axn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
            caxes = [fig.add_subplot(gs[row, 1]) for row in (0, 1)]
            for ax, cax, values, limit, rows, scale_label in zip(
                (axc, axn), caxes, (cv, nv), limits,
                (chemical_z.columns.tolist(), cells),
                ("Chemical level (z score)", "Coefficient minus overall mean"),
            ):
                im = ax.imshow(values, aspect="auto", interpolation="nearest",
                               cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_yticks(np.arange(len(rows)), rows)
                ax.set_xticks([])
                ax.tick_params(axis="both", length=0, pad=9)
                ax.spines[:].set_visible(False)
                ax.set_yticks(np.arange(len(rows) - 1) + 0.5, minor=True)
                ax.grid(which="minor", axis="y", color="white", linewidth=1.4)
                ax.tick_params(which="minor", length=0)
                if not individual:
                    ax.set_xticks(np.arange(4) + 0.5, minor=True)
                    ax.grid(which="minor", axis="x", color="white", linewidth=2.2)
                else:
                    for boundary in np.cumsum([len(g) for g in groups])[:-1] - 0.5:
                        ax.axvline(boundary, color="white", linewidth=2.0)
                cb = fig.colorbar(im, cax=cax, ticks=[-limit, 0, limit])
                cb.outline.set_visible(False)
                cb.set_label(scale_label, fontsize=10, labelpad=10)
                cb.ax.tick_params(length=2, labelsize=9)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            positions = [float(pos.mean()) for pos in groups] if individual else np.arange(5)
            axc.set_xticks(positions, group_labels, fontsize=11)
            axc.xaxis.tick_top()
            axc.tick_params(axis="x", pad=10)
            axn.set_xticks(positions, names, fontsize=10)
            axn.set_xlabel("Chemical level  →", labelpad=13, fontsize=12)
            # Panel labels sit above their own matrix, away from group headers.
            axc.text(0, 1.72, "Chemical levels", transform=axc.transAxes,
                     fontsize=13, fontweight="bold", va="bottom")
            axn.set_title("Neural pattern", loc="left", pad=12, fontweight="bold")
            title = "Individual strains" if individual else "Average profiles across chemical levels"
            subtitle = "106 strains · one column per strain" if individual else "106 strains · one column per group mean"
            fig.suptitle(title, x=0.51, y=0.985, fontsize=18)
            fig.text(0.51, 0.941, subtitle, ha="center", fontsize=11, color="#53616A")
            stem = "05_grouped_story_individuals" if individual else "05_grouped_story"
            for suffix in (".png", ".svg"):
                path = figures / (stem + suffix)
                fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                outputs.append(path)
            plt.close(fig)

    parameters = {
        "source_sha256": hashes, "n_strains": len(ids), "n_neurons": len(cells),
        "grouping": "Five near-equal rank groups, sorted by saved chemical score then strain ID",
        "group_count_fixed_before_inspecting_group_means": True,
        "group_sizes": {name: len(pos) for name, pos in zip(names, groups)},
        "chemical_score_ranges": {
            name: [float(scores.iloc[pos].chemical_score.min()),
                   float(scores.iloc[pos].chemical_score.max())]
            for name, pos in zip(names, groups)
        },
        "aggregation": "Arithmetic mean across strains of observed values, equal strain weight",
        "chemistry_display": "All-106-strain z scores of log concentrations, sample SD ddof=1",
        "neural_display": "Observed unit coefficient minus that neuron's all-106-strain mean",
        "neural_per_row_variance_scaling": False,
        "neuron_order": cells,
        "neuron_order_rule": "Descending absolute saved discovery slope; stable ties",
        "color_scales": scales, "clipping": False,
        "discovery_and_holdout_pooled_for_description_only": True,
        "model_refitted": False,
    }
    path = figures / "grouped_story_display_parameters.json"
    path.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    outputs.append(path)
    caption = (
        "Observed average chemical and neural profiles across five chemical-level groups. "
        "All 106 strains are ordered by the previously saved score for Glucaric acid, "
        "Lumichrome and Vitamin B1, then divided into five near-equal rank groups "
        "(22, 21, 21, 21, 21 strains). Groups use chemical values only; no neural-based "
        "selection, boundary adjustment, smoothing, interpolation or fitted predictions "
        "are used. Each column is the arithmetic mean of the same strains in both panels. "
        "Equal column spacing represents rank groups, not equal chemical-score intervals.\n\n"
        "Chemical colors show mean all-strain z scores of log concentrations (sample SD). "
        "Neural colors show the group mean of observed unit template coefficients minus "
        "each neuron's mean across all 106 strains. All 13 neurons share one color scale, "
        "without per-neuron SD scaling; their order is the saved discovery absolute-slope "
        "order. Red and blue indicate above and below each row's across-strain mean, "
        "not excitation/inhibition. Chemistry and neural scales differ. Each scale spans "
        "the full displayed range without clipping. No chemical denominator group or "
        "legacy fold-change result is introduced.\n\n"
        "The main figure describes group means, not a common response in every strain. "
        "The chemical gradient is expected from grouping by its own score. Averaging "
        "reduces visible strain-to-strain heterogeneity and does not increase the strength "
        "of the underlying evidence. Means need not change monotonically. Genus composition "
        "can differ between groups; groups are descriptive bins, not experimental replicates "
        "or established biological classes. No confidence intervals or significance tests "
        "are shown. The companion 05_grouped_story_individuals displays every strain with "
        "the same order, group boundaries, row order and centering; it has separately "
        "labeled full-range color scales because individual deviations are larger.\n\n"
        "Discovery and holdout strains are pooled for descriptive presentation only. "
        "This plot does not constitute validation, a new model or a new causal result. "
        "The existing split comparison and full-vector prediction evaluation remain "
        "supporting evidence. The previously explored dataset and cached neural templates "
        "are reused. Chemical cultures were independent of neural stimulus preparations.\n"
    )
    path = figures / "grouped_story_caption.txt"
    path.write_text(caption, encoding="utf-8")
    outputs.append(path)
    for source in sources:
        assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[str(source.relative_to(out))]
    return outputs
