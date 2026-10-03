"""Consistent RdBu_r inspection figures for the trial-SNR exploration."""
from pathlib import Path
import json
import hashlib

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Circle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from trial_representation import aggregate_strains

CMAP = "RdBu_r"
INK, MUTED, MISSING = "#243746", "#73808C", "#DCDCDC"
BLUE, RED = plt.get_cmap(CMAP)(.18), plt.get_cmap(CMAP)(.82)
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#AAB7C2", "svg.fonttype": "none"}


def _save(fig, folder, name):
    for ext in ("png", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _map():
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(MISSING)
    return cmap


def _matrix(tables, name):
    return pd.read_csv(tables / f"{name}.csv", index_col=0)


def _hist(ax, values, color, label, bins=None):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    bins = np.arange(-1., 1.0001, .05) if bins is None else bins
    ax.hist(values, bins=bins, density=True, color=color, alpha=.32,
            edgecolor=color, linewidth=.5)
    ax.hist(values, bins=bins, density=True, histtype="step", color=color,
            linewidth=1.1, label=f"{label} (mean {values.mean():.2f})")
    ax.axvline(values.mean(), color=color, linestyle=(0, (4, 3)), linewidth=1.3)


def _gate_by_strain(main, conditions, cells, ids):
    rows = []
    strains = np.asarray([s for s, _ in conditions])
    for strain in ids:
        for c, cell in enumerate(cells):
            status = main["status"][strains == strain, c]
            kept = np.isin(status, ["retained", "observed_zero"])
            suppressed = status == "below_snr"
            n_available = int((kept | suppressed).sum())
            value = np.nan if not n_available else (0 if kept.any() and suppressed.any() else (1 if kept.any() else -1))
            label = "unavailable" if not n_available else {1: "retained", -1: "zeroed", 0: "mixed_dates"}[value]
            rows.append(dict(strain=strain, cell=cell, n_dates=len(status),
                             retained_dates=int(kept.sum()), zeroed_dates=int(suppressed.sum()),
                             unavailable_dates=int(len(status)-n_available), state=label, code=value))
    return pd.DataFrame(rows)


def plot_exploration(data, fits, output_dir):
    out = Path(output_dir)
    tables, folder = out / "tables", out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    cells, conditions = list(data["cells"]), data["conditions"]
    main = fits["1"]
    coeff, strains = aggregate_strains(main["coefficients"], conditions)
    reconstruction, _ = aggregate_strains(main["reconstruction"], conditions)
    raw = _matrix(tables, "rdm_raw").loc[strains, strains].to_numpy()
    if np.isfinite(raw).all():
        d = np.clip((raw + raw.T)/2, 0, 2)
        np.fill_diagonal(d, 0)
        order = leaves_list(linkage(squareform(d), method="average"))
        order_rule = "Average linkage of unfiltered 0–40 s eight-bin cosine distances; display only"
    else:
        order = np.arange(len(strains))
        order_rule = "Sample ID order; missing unfiltered distances are not imputed"
    ids = [strains[i] for i in order]
    pd.DataFrame({"position": range(len(ids)), "strain": ids}).to_csv(tables / "figure_row_order.csv", index=False)
    cmap, norm = _map(), TwoSlopeNorm(vmin=-.5, vcenter=0, vmax=1.)
    captions = []
    with plt.rc_context(STYLE):
        # Each class occupies five bins, exactly the notebook's 0–25 s profile window.
        fig, axes = plt.subplots(2, 1, figsize=(16, 12.8), height_ratios=[1.1, 10], layout="constrained")
        template_panel = main["templates"][:, :5].reshape(-1)
        x = np.arange(len(cells)*5)
        for c in range(len(cells)):
            axes[0].plot(x[c*5:(c+1)*5], template_panel[c*5:(c+1)*5], color=RED, lw=1.6)
            axes[0].axvspan(c*5-.5, c*5+1.5, color="#EEF0F2", zorder=-2)
        axes[0].axhline(0, color=MUTED, lw=.5)
        axes[0].set(xticks=np.arange(len(cells))*5+2, xticklabels=cells,
                    xlim=(-.5, len(cells)*5-.5), ylabel="Template", title="Shared temporal templates · displayed 0–25 s")
        axes[0].tick_params(axis="x", length=0)
        values = reconstruction[order, :, :5].reshape(len(ids), -1)
        im = axes[1].imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        for boundary in np.arange(5, len(cells)*5, 5)-.5:
            axes[1].axvline(boundary, color="white", lw=1.4)
        axes[1].set(xticks=np.arange(len(cells))*5+2, xticklabels=cells,
                    ylabel="Bacterial samples", xlabel="Five 5-s bins per neuron class (0–25 s)")
        ticks = np.unique(np.r_[0, np.arange(9, len(ids), 10), len(ids)-1]).astype(int)
        axes[1].set_yticks(ticks, [ids[i] for i in ticks], fontsize=8)
        # Shared colorbar allocation keeps upper template blocks and lower columns aligned.
        fig.colorbar(im, ax=list(axes), shrink=.6, label="Reconstructed ΔF/F₀", extend="both")
        fig.suptitle("Trial-screened response profile · amplitude × template", fontsize=20, weight="bold")
        _save(fig, folder, "01_response_profile_5bin")
        pd.DataFrame(values, index=pd.Index(ids, name="strain"),
                     columns=[f"{c}_{b*5}-{(b+1)*5}s" for c in cells for b in range(5)]).to_csv(tables / "display_profile_5bin.csv")
        clipped = int(((values < -.5) | (values > 1.)).sum())
        captions.append(f"01_response_profile_5bin: All {len(ids)} strains and {len(cells)} classes. Each class occupies the notebook's five 5-s bins over 0–25 s. Values are reconstructed amplitude × template; templates and the SNR gate were fitted over 0–40 s, and only the first five bins are displayed. Upper templates are likewise truncated for display; their full eight bins are saved. Blank upper traces mean no identifiable template at this cutoff. Gray indicates unavailable data; zeroing is an analysis decision, not measured absence. Color limits −0.5 to 1 ΔF/F₀ match the notebook, with {clipped} saturated displayed bins. {order_rule}.")

        # An explicit, fully labelled audit of every neuron–bacteria decision.
        audit = _gate_by_strain(main, conditions, cells, ids)
        audit.to_csv(tables / "filter_state_by_strain_cell.csv", index=False)
        codes = audit.pivot(index="strain", columns="cell", values="code").loc[ids, cells]
        states = ListedColormap([cmap(.18), cmap(.5), cmap(.82)])
        states.set_bad(MISSING)
        state_norm = BoundaryNorm([-1.5, -.5, .5, 1.5], 3)
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 13.8), layout="constrained")
        cut = (len(ids)+1)//2
        for ax, subset in zip(axes, [ids[:cut], ids[cut:]]):
            ax.imshow(codes.loc[subset], aspect="auto", cmap=states, norm=state_norm, interpolation="nearest")
            ax.set(xticks=range(len(cells)), xticklabels=cells, yticks=range(len(subset)),
                   yticklabels=subset, xlabel="Neuron class")
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", labelsize=8, length=0)
            incomplete = audit.loc[audit.strain.isin(subset) & audit.unavailable_dates.gt(0) & audit.code.notna()]
            for row in incomplete.itertuples():
                ax.plot(cells.index(row.cell), subset.index(row.strain), ".", color=INK, ms=3)
        handles = [Patch(facecolor=cmap(.82), label="Retained"),
                   Patch(facecolor=cmap(.18), label="Zeroed"),
                   Patch(facecolor=cmap(.5), edgecolor=MUTED, label="Mixed across dates"),
                   Patch(facecolor=MISSING, label="Unavailable")]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle("Trial-SNR filtering by neuron and bacterium · SNR ≥ 1", fontsize=18, weight="bold")
        _save(fig, folder, "00_filter_status_heatmap")
        captions.append("00_filter_status_heatmap: Every sample ID is labelled, with the atlas order continued from left to right. Red: all evaluated dates retained; blue: all evaluated dates zeroed; white: retained and zeroed dates both present; gray: no evaluable date. A small dark dot, if present, indicates additional unavailable dates in an otherwise evaluated entry. Mixed dates are not forced to one binary strain-level decision: averaging the condition amplitudes may leave a nonzero strain coefficient. Full date-level decisions, trial counts, animal counts and SNR are in condition_metrics.csv; this categorical map is not response magnitude.")

        fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
        summary = pd.read_csv(tables / "sensitivity_metrics.csv", dtype={"threshold": str})
        for i, label in enumerate(["0.5", "1", "1.5", "2"]):
            rows = summary[summary.threshold.eq(label)].set_index("cell").loc[cells]
            ax.plot(range(len(cells)), rows.retained/rows.eligible, marker="o", ms=4,
                    color=cmap([.12, .32, .68, .88][i]), label=f"SNR ≥ {label}")
        ax.set(xticks=range(len(cells)), xticklabels=cells, ylabel="Retained fraction", ylim=(-.03, 1.03),
               title="Sensitivity to the trial-SNR cutoff")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(frameon=False, ncol=4)
        _save(fig, folder, "00b_trial_snr_thresholds")
        captions.append("00b_trial_snr_thresholds: Retained fractions among trial- and animal-covered condition × cell entries at each descriptive cutoff. Each threshold refits templates; no threshold is chosen using chemical agreement.")

        split = _matrix(tables, "split_filtered_cosine").loc[ids, ids].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), layout="constrained")
        im = axes[0].imshow(split, cmap=cmap, norm=norm, interpolation="nearest")
        axes[0].set(title="Cross-half response similarity", xlabel="Sample", ylabel="Sample", xticks=[], yticks=[])
        fig.colorbar(im, ax=axes[0], shrink=.8, label="Cosine similarity", extend="min")
        _hist(axes[1], split[np.triu_indices(len(ids), 1)], BLUE, "Different stimuli")
        _hist(axes[1], np.diag(split), RED, "Same stimulus")
        axes[1].set(xlabel="Cross-half cosine similarity", ylabel="Density", xlim=(-1, 1), title="Same versus different stimuli")
        axes[1].legend(frameon=False, fontsize=9)
        _save(fig, folder, "02_repeatability_distribution")
        counts = _matrix(tables, "split_filtered_valid_splits").loc[ids, ids].to_numpy()
        dc = np.diag(counts)
        pc = counts[np.triu_indices(len(ids), 1)]
        captions.append(f"02_repeatability_distribution: Independent whole-animal halves within each date; trial-SNR gates and templates refitted in each half. The right panel restores the notebook's separately normalized density histograms, bin width 0.05, filled plus step outlines and dashed means. Inputs are per-pair means over valid splits, not pooled split-level measurements. Same-sample entries have {int(dc.min())}–{int(dc.max())} valid splits; {int((pc==0).sum())} different-sample pairs have none. Zero norms remain undefined; at least four shared cells are required. Reconstruction comparisons use all eight 0–40 s bins. Repeated partitions and overlapping pairs are not independent biological replicates. Heatmap limits match the notebook; values below −0.5 saturate but remain unchanged in the histogram and tables.")

        errors = pd.read_csv(tables / "loao_cell_summary.csv").set_index("cell").loc[cells]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
        for method, color, label in [("raw", cmap(.12), "Raw"), ("unfiltered", cmap(.32), "Template, no gate"), ("filtered", cmap(.82), "Template, trial gate")]:
            _hist(axes[0], np.diag(_matrix(tables, f"split_{method}_paired_cosine")), color, label)
        axes[0].set(xlabel="Same-sample cross-half cosine", ylabel="Density", xlim=(-1, 1), title="Matched-support repeatability")
        axes[0].legend(frameon=False, fontsize=8)
        a, b = np.sqrt(errors.mse_unfiltered), np.sqrt(errors.mse_filtered)
        y = np.arange(len(cells))
        axes[1].hlines(y, np.minimum(a,b), np.maximum(a,b), color=MUTED, lw=1)
        axes[1].scatter(a, y, color=BLUE, label="No gate", s=24)
        axes[1].scatter(b, y, color=RED, label="Trial gate", s=24)
        axes[1].set(yticks=y, yticklabels=cells, ylim=(len(cells)-.5, -.5),
                    xlabel="Held-out raw-target RMSE (ΔF/F₀)", title="Prediction of another animal")
        axes[1].legend(frameon=False, fontsize=8)
        _save(fig, folder, "02b_representation_validation")
        captions.append("02b_representation_validation: Repeatability controls use identical cells and valid splits. Leave-one-animal-out fits exclude that animal and all its trials before computing SNR or templates. Held-out raw eight-bin targets are never zeroed. RMSE uses equal animals within condition, dates within strain, then strains within cell. These diagnostics evaluate the stated trial gate, not all possible SNR definitions.")

        neural = _matrix(tables, "rdm_filtered").loc[ids, ids].to_numpy()
        chemical = _matrix(tables, "rdm_chemical").loc[ids, ids].to_numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), layout="constrained")
        for ax, values, title, maximum, label in [(axes[0], neural, "Neural RDM", 2., "1 − cosine"),
                                                  (axes[1], chemical, "Chemical RDM", float(np.nanmax(chemical)), "RMS Δlog₂FC")]:
            im = ax.imshow(values, cmap=cmap, vmin=0, vmax=maximum, interpolation="nearest")
            ax.set(title=title, xlabel="Sample", ylabel="Sample", xticks=[], yticks=[])
            fig.colorbar(im, ax=ax, shrink=.75, label=label)
        tri = np.triu_indices(len(ids), 1)
        xx, yy = chemical[tri], neural[tri]
        valid = np.isfinite(xx) & np.isfinite(yy)
        im = axes[2].hexbin(xx[valid], yy[valid], gridsize=27, mincnt=1, cmap=cmap, linewidths=0)
        axes[2].set(xlabel="Chemical RMS Δlog₂FC", ylabel="Neural 1 − cosine", ylim=(0, 2), title="Matched sample pairs")
        fig.colorbar(im, ax=axes[2], shrink=.75, label="Pairs per bin")
        _save(fig, folder, "03_neural_chemical_rdm")
        captions.append(f"03_neural_chemical_rdm: Identical {len(ids)}-sample ordering. Neural distances use 0–40 s template amplitudes, equivalent to fixed-template reconstructed curve cosine on the same cell support. All-zero profiles have undefined cosine. Chemical distance uses the existing 380 log₂FC features and inherits their reference/filling limitations. Both RDMs and the hexbin use RdBu_r, with separate colorbar units and limits. Pairwise scatter is descriptive; pairs sharing samples are not independent observations.")
    (folder / "captions.txt").write_text("\n\n".join(captions)+"\n")
    settings = dict(colormap=CMAP, response_limits=[-.5, 0, 1], cosine_limits=[-.5, 0, 1],
                    profile_display_window=[0, 25], profile_bins=5, model_window=[0, 40], model_bins=8,
                    primary_snr=1, row_order=ids, row_order_rule=order_rule,
                    repeatability_histogram_bin_width=.05, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / "plot_parameters.json").write_text(json.dumps(settings, indent=2)+"\n")
    return settings


def plot_hmds(output_dir):
    """Plot true Poincaré coordinates returned by the existing HMDS fitter."""
    out = Path(output_dir)
    hmds, folder = out / "hmds", out / "figures"
    colors = pd.read_csv(hmds / "sample_colors.csv", index_col=0)
    score = colors.iloc[:, 0]
    cmap = _map()
    bound = float(np.nanmax(np.abs(score)))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    neural = pd.read_csv(hmds / "neural_coordinates.csv", index_col=0)
    selected = neural.index
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), layout="constrained")
    with plt.rc_context(STYLE):
        for ax, domain, title in zip(axes, ["neural", "chemical"], ["Neural HMDS · chord distance", "Chemical HMDS · RMS Δlog₂FC"]):
            frame = pd.read_csv(hmds / f"{domain}_coordinates.csv", index_col=0).loc[selected]
            ax.add_patch(Circle((0,0), 1, fill=False, color=MUTED, lw=.8))
            scatter = ax.scatter(frame.x, frame.y, c=score.reindex(frame.index), cmap=cmap, norm=norm,
                                 s=25, edgecolors="white", linewidths=.35)
            ax.set(xlim=(-1.03,1.03), ylim=(-1.03,1.03), aspect="equal", xticks=[], yticks=[], title=title)
            ax.set_axis_off()
        fig.colorbar(scatter, ax=list(axes), shrink=.75, label="Chemical PCo1 (shared sample colors)")
        fig.suptitle(f"Matched HMDS views · {len(selected)} samples with neural bootstrap support", fontsize=15)
        _save(fig, folder, "03b_neural_chemical_hmds")
    with (folder / "captions.txt").open("a") as handle:
        handle.write(f"\n03b_neural_chemical_hmds: Two-dimensional Poincaré-disk displays of the same {len(selected)} samples in both panels. Neural input is sqrt(2 × neural cosine distance), matching the original notebook's chord-distance HMDS; its uncertainty is recomputed by 1000 joint within-date whole-animal bootstrap draws with trial gates and templates refitted. Pairs require at least 80% valid draws; only the largest connected eligible component can be fitted together. Full-sample zero profiles and inadequate bootstrap support are listed in hmds/neural_sample_coverage.csv. Chemical coordinates are shown for the same subset but come from the saved, verified full 106-sample 2D RMS-distance fit. Chemical PCo1 supplies one frozen scalar per sample and the same RdBu_r mapping in both panels; colors are not clusters inferred from neural coordinates. Angular orientation is arbitrary and raw coordinate distances across panels are not directly comparable. Native hyperbolic isometric centering is retained without Euclidean shrinkage. Coverage and convergence diagnostics are in hmds/result.json.\n")
    return "03b_neural_chemical_hmds.png"
