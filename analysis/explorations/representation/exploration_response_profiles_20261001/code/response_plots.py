"""Exploration figures for screened template-amplitude profiles (PNG and SVG)."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from response_representation import aggregate_strains

INK, MUTED, TEAL, ORANGE = "#20374D", "#71808F", "#16839C", "#CB7539"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
         "text.color": INK, "axes.labelcolor": INK,
         "xtick.color": MUTED, "ytick.color": MUTED,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#AAB7C2", "svg.fonttype": "none"}


def _save(fig, folder, name):
    for ext in ("png", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _read_matrix(table, name):
    return pd.read_csv(table / f"{name}.csv", index_col=0)


def _ecdf(ax, values, label, color):
    x = np.sort(np.asarray(values)[np.isfinite(values)])
    if len(x):
        ax.step(x, np.arange(1, len(x)+1)/len(x), where="post", label=label, color=color, lw=2)


def plot_exploration(data, fits, output_dir):
    """Plot all three stages and supporting threshold/curve diagnostics."""
    out = Path(output_dir)
    table, folder = out / "tables", out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    main = fits["1"]
    cells, conditions = list(data["cells"]), data["conditions"]
    coeff, strains = aggregate_strains(main["coefficients"], conditions)
    strains = list(strains)
    raw_rdm = _read_matrix(table, "rdm_raw").reindex(index=strains, columns=strains)
    if np.isfinite(raw_rdm.to_numpy()).all():
        distance = np.clip(raw_rdm.to_numpy(), 0, 2)
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform((distance+distance.T)/2, checks=True), method="average"))
        ordering = "Average linkage on unfiltered 0–40 s eight-bin raw-curve cosine distances; display only."
    else:
        order = np.arange(len(strains))
        ordering = "Sample ID order because the unfiltered raw RDM contains undefined values; no imputation."
    ids = [strains[i] for i in order]
    pd.DataFrame({"position": np.arange(len(ids)), "sample_id": ids}).to_csv(table / "figure_row_order.csv", index=False)
    cmap = LinearSegmentedColormap.from_list("signed", ["#74509C", "#FBFBFC", "#147D78"])
    cmap.set_bad("#D3D7DC")
    captions = []
    with plt.rc_context(STYLE):
        # Thresholds are shown before response inspection; selection is not optimized for RSA.
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for label, fit in fits.items():
            if label == "unfiltered":
                continue
            eligible = fit["counts"] >= 3
            kept = eligible & (fit["snr"] >= float(label))
            denom = eligible.sum(axis=0)
            fraction = np.divide(kept.sum(axis=0), denom, out=np.full(len(cells), np.nan), where=denom > 0)
            axes[0].plot(range(len(cells)), fraction, marker="o", ms=4, label=f"SNR ≥ {label}")
            if label == "1":
                _ecdf(axes[1], fit["snr"][eligible], "Eligible conditions", TEAL)
        axes[0].set(xticks=range(len(cells)), xticklabels=cells, ylim=(-.03, 1.03), ylabel="Retained conditions / eligible conditions", title="Threshold sensitivity")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].set(xlabel="Across-animal SNR", ylabel="Cumulative fraction", xlim=(0, 4), title="Raw-curve SNR distribution")
        for cutoff in (.5, 1, 1.5, 2):
            axes[1].axvline(cutoff, color=MUTED, lw=.7, ls="--", alpha=.6)
        _save(fig, folder, "00_snr_thresholds")
        captions.append("00_snr_thresholds: Complete 0–40 s raw curves determine SNR before template fitting. Fractions use condition × cell entries with at least three animals; conditions are strain × acquisition date. SNR is sqrt(max(P−V/n,0)/V), where P is squared RMS of the animal mean curve and V is mean across-time sample variance across animals. The cumulative distribution is unchanged by the threshold; vertical lines show the four candidate cutoffs. No threshold is selected by neural–chemical correlation.")

        # One common amplitude unit and aligned per-cell templates.
        fig = plt.figure(figsize=(13.2, 13.6), layout="constrained")
        grid = fig.add_gridspec(2, len(cells)+1, height_ratios=[1.15, 11],
                               width_ratios=[1]*len(cells)+[.25], hspace=.08)
        template_max = float(np.nanmax(np.abs(main["templates"])))
        for c, cell in enumerate(cells):
            ax = fig.add_subplot(grid[0, c])
            ax.plot(np.arange(2.5, 40, 5), main["templates"][c], color=TEAL, lw=1.7)
            ax.axhline(0, color=MUTED, lw=.5)
            ax.axvspan(0, 10, color="#E7ECF1", zorder=-2)
            ax.set(xlim=(0, 40), ylim=(-template_max*1.08, template_max*1.08), title=cell, xticks=[0, 40], yticks=[])
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("Template", fontsize=9)
        ax = fig.add_subplot(grid[1, :len(cells)])
        limit = .6
        matrix = coeff[order]
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
        ax.set(xticks=range(len(cells)), xticklabels=cells, ylabel="Bacterial samples", xlabel="Neuron class")
        ticks = np.unique(np.r_[0, np.arange(9, len(ids), 10), len(ids)-1]).astype(int)
        ax.set_yticks(ticks, [ids[i] for i in ticks], fontsize=8)
        cax = fig.add_subplot(grid[1, -1])
        fig.colorbar(im, cax=cax, extend="both", label="Amplitude (ΔF/F₀)")
        fig.suptitle("Amplitude × template response atlas · SNR ≥ 1", x=.03, ha="left", fontsize=20, weight="bold")
        clipped = int((np.abs(matrix) > limit).sum())
        _save(fig, folder, "01_response_profile")
        captions.append(f"01_response_profile: All {len(ids)} samples and 13 neuron classes. Upper traces are independently fitted unit-RMS eight-bin templates; lower entries are the coefficients after raw-curve SNR ≥ 1 screening. A blank upper trace means no identifiable template; ASG has no retained fitting conditions. Gate-failed adequately measured conditions have analysis coefficient zero; insufficient coverage is gray. Multiple dates are equally weighted after condition-level screening, using available supported dates. White does not establish physiological absence of response. Templates have maximum-absolute bin positive, so signed coefficients do not generically indicate excitation/inhibition. Color limits ±0.6 ΔF/F₀ saturate {clipped} entries; no analytical value is clipped. {ordering}")

        # Diagnostic examples selected by fixed rules, not hand-picked fit quality.
        counts = main["counts"]
        mean_rms = np.sqrt(np.nanmean(main["means"]**2, axis=-1))
        valid = counts >= 3
        asg = cells.index("ASG")
        eligible_asg = np.flatnonzero(valid[:, asg])
        strongest = eligible_asg[np.nanargmax(main["snr"][eligible_asg, asg])]
        failed_asg = eligible_asg[main["snr"][eligible_asg, asg] < 1]
        median_failed = failed_asg[np.argsort(mean_rms[failed_asg, asg])[len(failed_asg)//2]] if len(failed_asg) else strongest
        variable = np.where(valid & (main["snr"] < 1), mean_rms, np.nan)
        kv, cv = np.unravel_index(np.nanargmax(variable), variable.shape)
        selected = [(strongest, asg, "Highest ASG SNR"), (median_failed, asg, "Median suppressed ASG RMS"), (kv, cv, "Largest suppressed mean RMS")]
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), layout="constrained")
        example_rows = []
        for ax, (k, c, rule) in zip(axes, selected):
            y = data["raw"][:, k, c]
            y = y[np.isfinite(y).all(axis=1)]
            mu, sem = y.mean(axis=0), y.std(axis=0, ddof=1)/np.sqrt(len(y))
            ax.axvspan(0, 10, color="#E7ECF1", zorder=-3)
            ax.axhline(0, color=MUTED, lw=.6)
            ax.fill_between(np.arange(40), mu-sem, mu+sem, color=TEAL, alpha=.18)
            ax.plot(np.arange(40), mu, color=TEAL, lw=2, label="Raw mean ± SEM")
            ax.plot(np.arange(2.5, 40, 5), main["reconstruction"][k, c], color=ORANGE, ls="--", lw=1.8, label="Screened reconstruction")
            ax.set(title=f"{cells[c]} · {conditions[k][0]}", xlabel="Time (s)", ylabel="ΔF/F₀", xlim=(0, 40))
            example_rows.append(dict(strain=conditions[k][0], block=conditions[k][1], cell=cells[c], rule=rule,
                                     n_animals=len(y), snr=main["snr"][k, c], coefficient=main["coefficients"][k, c]))
        axes[-1].legend(frameon=False, fontsize=8)
        _save(fig, folder, "01b_curve_screening_examples")
        pd.DataFrame(example_rows).to_csv(table / "curve_example_selection.csv", index=False)
        captions.append("01b_curve_screening_examples: Original 1-s animal means ± SEM, with no smoothing or rebaselining; orange traces are screened eight-bin reconstructions. Example rules are highest ASG SNR, median raw mean-curve RMS among suppressed ASG conditions, and largest raw mean-curve RMS among all suppressed conditions (n≥3). These are inspection examples, not independently selected findings. A zero reconstruction means the representation suppresses that condition; original animal responses are retained. Axes have cell-specific scales. Exact selections, dates, SNR and counts are saved separately.")

        split = _read_matrix(table, "split_filtered_cosine").reindex(index=ids, columns=ids)
        scmap = plt.get_cmap("RdBu_r").copy()
        scmap.set_bad("#D3D7DC")
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained")
        im = axes[0].imshow(split, cmap=scmap, vmin=-1, vmax=1, interpolation="nearest")
        axes[0].set(title="Cross-half response similarity", xlabel="Sample", ylabel="Sample", xticks=[], yticks=[])
        fig.colorbar(im, ax=axes[0], shrink=.75, label="Cosine similarity")
        a = split.to_numpy()
        _ecdf(axes[1], np.diag(a), "Same sample", TEAL)
        _ecdf(axes[1], a[np.triu_indices(len(a), 1)], "Different samples", MUTED)
        axes[1].set(xlabel="Cross-half cosine similarity", ylabel="Cumulative fraction", xlim=(-1, 1), ylim=(0, 1.02), title="Same versus different samples")
        axes[1].legend(frameon=False)
        _save(fig, folder, "02_repeatability")
        valid_splits = _read_matrix(table, "split_filtered_valid_splits").reindex(index=ids, columns=ids).to_numpy()
        upper = np.triu_indices(len(ids), 1)
        diag_counts, pair_counts = np.diag(valid_splits), valid_splits[upper]
        coverage = dict(same_sample_valid_split_min=int(diag_counts.min()),
                        same_sample_valid_split_median=float(np.median(diag_counts)),
                        same_sample_valid_split_max=int(diag_counts.max()),
                        different_pairs_without_valid_splits=int((pair_counts == 0).sum()),
                        different_pair_valid_split_median=float(np.median(pair_counts)))
        captions.append(f"02_repeatability: Animals are globally assigned to two groups within each acquisition date, moving all their records together. Both SNR gates and templates are independently re-estimated in each half (minimum two animals per condition × cell). Comparisons use reconstructed 13 × 8 profiles, avoiding arbitrary template-sign alignment across halves. Symmetric directions use the same four-profile complete shared-cell support, at least four cells. Zero-norm profiles are undefined and remain gray. Each cell of the matrix is the mean over its valid splits, not necessarily all 100. Same-sample entries have {int(diag_counts.min())}–{int(diag_counts.max())} valid splits (median {np.median(diag_counts):g}); {int((pair_counts == 0).sum())}/{len(pair_counts)} different-sample pairs have none. Three-animal condition × cell entries cannot satisfy minimum two in both halves. The 100 partitions and overlapping sample pairs are not independent biological replicates. This evaluates repeatability of processed profiles; unfiltered held-out targets are evaluated separately. Row order is inherited from the unfiltered raw RDM for display.")

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), layout="constrained")
        for method, label, color in (("raw", "Raw curves", MUTED),
                                      ("unfiltered", "Template, no gate", ORANGE),
                                      ("filtered", "Template, SNR ≥ 1", TEAL)):
            matched = _read_matrix(table, f"split_{method}_paired_cosine").to_numpy()
            _ecdf(axes[0], np.diag(matched), label, color)
        axes[0].set(xlabel="Same-sample cross-half cosine", ylabel="Cumulative fraction",
                    xlim=(-1, 1), ylim=(0, 1.02), title="Matched-support repeatability")
        axes[0].legend(frameon=False, fontsize=8)
        errors = pd.read_csv(table / "loao_cell_summary.csv").set_index("cell").reindex(cells)
        ungated, gated = np.sqrt(errors.mse_unfiltered), np.sqrt(errors.mse_filtered)
        positions = np.arange(len(cells))
        axes[1].hlines(positions, np.minimum(ungated, gated), np.maximum(ungated, gated), color=MUTED, lw=1)
        axes[1].scatter(ungated, positions, color=ORANGE, s=24, label="No gate", zorder=3)
        axes[1].scatter(gated, positions, color=TEAL, s=24, label="SNR ≥ 1", zorder=4)
        axes[1].set(yticks=positions, yticklabels=cells, ylim=(len(cells)-.5, -.5),
                    xlabel="Held-out raw-target RMSE (ΔF/F₀)", title="Prediction of another animal")
        axes[1].legend(frameon=False, fontsize=8)
        _ecdf(axes[2], diag_counts, "Same sample", TEAL)
        _ecdf(axes[2], pair_counts, "Different samples", MUTED)
        axes[2].set(xlabel="Valid splits / 100", ylabel="Cumulative fraction", xlim=(0, 102),
                    ylim=(0, 1.02), title="Screened comparison coverage")
        axes[2].legend(frameon=False, fontsize=8)
        _save(fig, folder, "02b_representation_validation")
        captions.append("02b_representation_validation: Left: all three representations use identical shared cells and valid split instances for each same-sample comparison; empirical curves summarize the 106 sample means, without treating splits as independent replicates. Center: leave-one-whole-animal-out fits re-estimate gates and templates; raw held-out eight-bin targets are never screened or zeroed. RMSE is the square root of MSE averaged equally across animals within condition, dates within strain, and strains within cell; lower is better. No gate and SNR ≥ 1 are evaluated on common valid observations. Right: comparison coverage includes zero-count pairs. Coverage and zero-vector audits are saved in split tables. This is a diagnostic check of this cutoff, not a tuned validation of the screening method.")

        chemical = _read_matrix(table, "rdm_chemical").reindex(index=ids, columns=ids)
        neural = _read_matrix(table, "rdm_filtered").reindex(index=ids, columns=ids)
        dmap = plt.get_cmap("viridis").copy()
        dmap.set_bad("#D3D7DC")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        for ax, matrix, title, upper, unit in zip(axes[:2], [neural, chemical], ["Neural response distance", "Chemical distance"], [2., float(np.nanmax(chemical))], ["1 − cosine", "RMS Δlog₂FC"]):
            im = ax.imshow(matrix, cmap=dmap, vmin=0, vmax=upper, interpolation="nearest")
            ax.set(title=title, xlabel="Sample", ylabel="Sample", xticks=[], yticks=[])
            fig.colorbar(im, ax=ax, shrink=.72, label=unit)
        tri = np.triu_indices(len(ids), 1)
        x, y = chemical.to_numpy()[tri], neural.to_numpy()[tri]
        ok = np.isfinite(x) & np.isfinite(y)
        hb = axes[2].hexbin(x[ok], y[ok], gridsize=27, mincnt=1, cmap="Blues", linewidths=0)
        axes[2].set(xlabel="Chemical RMS Δlog₂FC", ylabel="Neural 1 − cosine", ylim=(0, 2), title="Matched sample pairs")
        fig.colorbar(hb, ax=axes[2], shrink=.72, label="Pairs per bin")
        _save(fig, folder, "03_neural_chemical_rdm")
        captions.append("03_neural_chemical_rdm: Both matrices use the identical 106-sample order. Neural distance is 1 − cosine of screened 13-cell amplitudes; unit-RMS fixed templates make this exactly equivalent to cosine of the corresponding reconstructed 13 × 8 profiles on the same cell support. All-zero comparisons are undefined, not zero distance. Chemical distance retains all 380 existing log₂FC features with their upstream filling/reference effects; chemistry is from separate cultures, not neural stimulus aliquots. Color scales have different units. The hexbin plot is descriptive; overlapping pairs are not independent observations, and no ordinary pair-wise correlation p value is reported. Threshold comparisons use a common valid pair set in the tables.")
    (folder / "captions.txt").write_text("\n\n".join(captions)+"\n")
    (folder / "plot_parameters.json").write_text(json.dumps(dict(primary_snr=1, cells=cells, row_order=ids,
         row_order_rule=ordering, coefficient_color_limits=[-.6, .6], profile_clipped_entries=clipped,
         split_coverage=coverage,
         code_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+"\n")
    return {"row_order": ids, "clipped_entries": clipped, "figures": sorted(p.name for p in folder.glob('*.png'))}
