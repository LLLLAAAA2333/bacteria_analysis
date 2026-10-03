"""Display-only profile and model figures from the saved individual-SNR fit.

The response profile follows Notebook 02's final Panel A layout, without a
template strip. A separate figure explains compression and displays the saved
templates above their amplitudes. No raw data are loaded and no models refit.
"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.interpolate import PchipInterpolator
from scipy.spatial.distance import squareform


INK, MUTED, BLUE, RED = "#132C4E", "#687F99", "#2166AC", "#B2182B"
PROFILE_LIMITS = (-.5, 1.)
AMPLITUDE_LIMITS = (-.6, .6)
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .8,
    "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "figure.dpi": 110, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _save(fig, directory, stem):
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _cmap():
    result = plt.get_cmap("RdBu_r").copy()
    result.set_bad("#DCDCDC")
    return result


def _extend(values, limits):
    finite = np.asarray(values)[np.isfinite(values)]
    low, high = finite.min() < limits[0], finite.max() > limits[1]
    return "both" if low and high else "min" if low else "max" if high else "neither"


def _date_mean(values, conditions, ids):
    """Preserve the saved fit's equal-available-date aggregation."""
    result = []
    for sample in ids:
        selected = values[conditions[:, 0] == sample]
        finite = np.isfinite(selected)
        counts = finite.sum(axis=0)
        total = np.where(finite, selected, 0.).sum(axis=0)
        result.append(np.divide(total, counts, out=np.full(selected.shape[1:], np.nan),
                                 where=counts > 0))
    return np.asarray(result)


def _draw_tree(ax, tree, expected_order):
    drawn = dendrogram(tree, ax=ax, orientation="left", no_labels=True,
                       color_threshold=.7 * tree[:, 2].max(), above_threshold_color=MUTED)
    if not np.array_equal(drawn["leaves"], expected_order):
        raise ValueError("Dendrogram leaves differ from the displayed data order")
    ax.set_ylim(10 * len(expected_order), 0)
    ax.axis("off")
    for collection in ax.collections:
        collection.set_linewidth(.7)


def _profile_figure(directory, values, cells, ids, tree, order):
    """Restore final Notebook 02 Panel A's axes, class gaps and labeling."""
    n_bins = 5
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14.5, 8.2))
        fig.text(.018, .985, "A", fontsize=21, weight="bold", va="top")
        fig.text(.07, .978, "Population response profile", fontsize=14, weight="bold", va="top")
        ax_tree = fig.add_axes([.035, .22, .055, .68])
        ax = fig.add_axes([.095, .22, .78, .68])
        _draw_tree(ax_tree, tree, order)
        display = np.full((len(ids), len(cells) * (n_bins + 1) - 1), np.nan)
        for c in range(len(cells)):
            display[:, c * (n_bins + 1):c * (n_bins + 1) + n_bins] = values[:, c, :n_bins]
        image = ax.imshow(display, cmap=_cmap(),
                          norm=TwoSlopeNorm(vmin=PROFILE_LIMITS[0], vcenter=0, vmax=PROFILE_LIMITS[1]),
                          origin="upper", aspect="auto", interpolation="nearest", rasterized=True)
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(cells)))
        for c, (cell, color) in enumerate(zip(cells, colors)):
            start = c * (n_bins + 1)
            if c:
                ax.axvspan(start - 1.5, start - .5, color="white", linewidth=0)
            ax.axvline(start + 1.5, color="#36465A", linestyle=(0, (3, 3)),
                       linewidth=.8, alpha=.85, zorder=4)
            ax.plot([start - .45, start + n_bins - .55], [1.012, 1.012], color=color,
                    linewidth=2.3, transform=ax.get_xaxis_transform(), clip_on=False)
            ax.text(start + (n_bins - 1) / 2, 1.034, cell, color=color,
                    ha="center", fontsize=9, weight="bold", transform=ax.get_xaxis_transform())
        ticks = [c * (n_bins + 1) + b for c in range(len(cells)) for b in range(n_bins)]
        ax.set_xticks(ticks, list(range(1, n_bins + 1)) * len(cells), fontsize=6)
        rows = np.arange(0, len(ids), 10)
        ax.set_yticks(rows, np.asarray(ids)[rows], fontsize=7)
        ax.yaxis.tick_right()
        ax.tick_params(length=0, pad=3)
        ax.set_xlabel("Time bins (5 s each; dashed line = stimulus offset at 10 s)", labelpad=9)
        ax_tree.text(-.13, .5, "Bacterial stimuli (clustered)", rotation=90,
                     ha="right", va="center", transform=ax_tree.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes([.936, .34, .013, .43])
        cax.set_title(r"$\Delta F/F_0$", fontsize=11, pad=12)
        colorbar = fig.colorbar(image, cax=cax, ticks=[-.5, 0, 1],
                               extend=_extend(values[:, :, :5], PROFILE_LIMITS))
        colorbar.outline.set_visible(False)
        return _save(fig, directory, "01_response_profile_5bin")


def _time_axis(ax, ylabel=None):
    ax.axvspan(0, 10, color="#D8DEE5", alpha=.45, zorder=-3)
    ax.axhline(0, color=INK, alpha=.4, lw=.6, zorder=-2)
    ax.set(xlim=(0, 40), xticks=[0, 10, 40], xlabel="Time (s)")
    ax.tick_params(length=3, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)


def _model_figure(directory, arrays, cells, ids, amplitudes, tree, order, example):
    """Combine the saved compression example and aligned template/amplitude map."""
    k, c = example
    raw = arrays["means"][k, c]
    binned = arrays["mean_bins"][k, c]
    templates = arrays["templates"]
    weight = float(arrays["coefficients"][k, c])
    nodes = np.arange(2.5, 40, 5)
    smooth_times = np.linspace(nodes[0], nodes[-1], 351)
    smooth_template = PchipInterpolator(nodes, templates[c], extrapolate=False)(smooth_times)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14.8, 14.0))
        fig.text(.043, .981, "From response curves to an amplitude atlas", fontsize=20, weight="bold")
        fig.text(.055, .948, "AWCON · A300", fontsize=10.5, color=MUTED)
        for x, label in [(.148, "Mean response"), (.414, "8 time bins"),
                          (.679, "Shared template"), (.895, "Amplitude")]:
            fig.text(x, .919, label, fontsize=14, ha="center")
        raw_ax = fig.add_axes([.055, .768, .185, .133])
        _time_axis(raw_ax, r"$\Delta F/F_0$")
        response_limits = (-.18, max(float(raw.max()), float(binned.max())) * 1.09)
        raw_ax.set_ylim(response_limits)
        raw_ax.plot(np.arange(40), raw, color="#26354A", lw=2.1)
        bin_ax = fig.add_axes([.322, .768, .185, .133])
        _time_axis(bin_ax)
        bin_ax.set_ylim(response_limits)
        for b, value in enumerate(binned):
            bin_ax.fill_between([5*b, 5*(b+1)], [0, 0], [value, value],
                                color="#AFC4D4", alpha=.65, linewidth=0)
            bin_ax.plot([5*b+.25, 5*(b+1)-.25], [value, value], color="#4B718A", lw=2.4)
            bin_ax.axvline(5*b, color="white", lw=1.1, zorder=4)
        shape_ax = fig.add_axes([.593, .768, .174, .133])
        _time_axis(shape_ax)
        shape_ax.set_ylim(-.1, max(2.1, float(templates[c].max()) * 1.08))
        shape_ax.plot(smooth_times, smooth_template, color=BLUE, lw=2.4)
        shape_ax.scatter(nodes, templates[c], s=10, color=BLUE, zorder=3)
        fig.add_artist(FancyArrowPatch((.257, .834), (.302, .834), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=15, lw=1.3, color=MUTED))
        fig.text(.55, .833, "≈", fontsize=26, ha="center", va="center", color=INK)
        fig.text(.803, .832, "×", fontsize=26, ha="center", va="center", color=INK)
        fig.text(.895, .828, f"{weight:+.2f}", fontsize=30, ha="center", color=RED)
        fig.text(.148, .721, "1-s sampling", fontsize=10, ha="center", color=MUTED)
        fig.text(.414, .721, "8 × 5 s", fontsize=10, ha="center", color=MUTED)
        fig.text(.679, .727, r"$h_c(t)$", fontsize=15, ha="center", color=BLUE)
        fig.text(.679, .705, "shared · RMS = 1", fontsize=10, ha="center", color=MUTED)
        fig.text(.895, .727, r"$a_{kc}$", fontsize=15, ha="center", color=RED)
        fig.text(.895, .705, "one per condition", fontsize=10, ha="center", color=MUTED)
        fig.add_artist(Line2D([.043, .963], [.680, .680], transform=fig.transFigure,
                             color="#D8DEE5", lw=.7))
        fig.text(.055, .657, "Shared templates and sample amplitudes", fontsize=14, weight="bold")

        left, width = .095, .78
        template_limit = float(np.nanmax(np.abs(templates))) * 1.08
        for j, cell in enumerate(cells):
            col_width = width / len(cells)
            ax = fig.add_axes([left + j*col_width + .004, .554, col_width - .008, .075])
            ax.plot(nodes, templates[j], color=BLUE, lw=1.5)
            ax.axhline(0, color=MUTED, lw=.5)
            ax.axvspan(0, 10, color="#E7ECF1", zorder=-2)
            ax.set(xlim=(0, 40), ylim=(-template_limit, template_limit), title=cell,
                   xticks=[0, 40], yticks=[])
            ax.title.set_fontsize(9.5)
            ax.tick_params(labelsize=6.5, length=2)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#AAB7C2")
            if j == 0:
                ax.set_ylabel("Template", fontsize=9)
        ax_tree = fig.add_axes([.035, .065, .055, .455])
        _draw_tree(ax_tree, tree, order)
        ax_tree.text(-.13, .5, "Bacterial stimuli (same order)", rotation=90,
                     ha="right", va="center", transform=ax_tree.transAxes)
        ax = fig.add_axes([left, .065, width, .455])
        image = ax.imshow(amplitudes, origin="upper", aspect="auto", interpolation="nearest",
                          cmap=_cmap(), norm=TwoSlopeNorm(vmin=-.6, vcenter=0, vmax=.6), rasterized=True)
        ax.set(xticks=range(len(cells)), xticklabels=cells, xlabel="Neuron class")
        rows = np.arange(0, len(ids), 10)
        ax.set_yticks(rows, np.asarray(ids)[rows], fontsize=7)
        ax.yaxis.tick_right()
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes([.936, .155, .013, .28])
        cax.set_title("Amplitude", fontsize=10, pad=12)
        colorbar = fig.colorbar(image, cax=cax, ticks=[-.6, 0, .6],
                               extend=_extend(amplitudes, AMPLITUDE_LIMITS))
        colorbar.set_label(r"$\Delta F/F_0$", labelpad=8)
        colorbar.outline.set_visible(False)
        return _save(fig, directory, "01b_response_model")


def plot_profile_and_model(output_dir):
    """Render the two figures solely from saved arrays and tables; return paths."""
    out = Path(output_dir).resolve()
    tables, figures, repo = out / "tables", out / "figures", out.parent.parent
    figures.mkdir(parents=True, exist_ok=True)
    parameters_path = out / "representation_parameters.json"
    parameters = json.loads(parameters_path.read_text())
    if parameters["snr_basis"] != "individual" or parameters["primary_threshold"] != .5:
        raise ValueError("These requested figures require the saved individual-SNR ≥ 0.5 representation")
    arrays_path = tables / "representation_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    cells = arrays["cells"].tolist()
    conditions = arrays["conditions"]
    if arrays["templates"].shape != (len(cells), 8) or arrays["means"].shape[-1] != 40:
        raise ValueError("Expected saved 0–40 s curves and eight 5-s template bins")
    coefficients_path = tables / "strain_coefficients.csv"
    coefficients = pd.read_csv(coefficients_path, index_col=0).loc[:, cells]
    ids = coefficients.index.astype(str)
    reconstructed = _date_mean(arrays["reconstruction"], conditions, ids)
    aggregated_coefficients = _date_mean(arrays["coefficients"], conditions, ids)
    if not np.allclose(aggregated_coefficients, coefficients, rtol=1e-12, atol=1e-14, equal_nan=True):
        raise ValueError("Saved strain amplitudes do not match saved condition arrays")
    rdm_path = tables / "rdm_raw.csv"
    distance = pd.read_csv(rdm_path, index_col=0).loc[ids, ids].to_numpy(float)
    if not np.isfinite(distance).all() or not np.allclose(distance, distance.T, atol=1e-12):
        raise ValueError("A complete symmetric raw RDM is required for the unchanged display hierarchy")
    distance = np.clip((distance + distance.T) / 2, 0, 2)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=True), method="average")
    order = leaves_list(tree)
    ordered_ids = ids.take(order).tolist()
    row_path = tables / "figure_row_order.csv"
    previous_order = pd.read_csv(row_path)
    previous_ids = previous_order["strain" if "strain" in previous_order else "sample_id"].tolist()
    if previous_ids != ordered_ids:
        raise ValueError("Rebuilt dendrogram would change the saved figure row order")
    display_path = tables / "display_profile_5bin.csv"
    if display_path.exists():
        previous_display = pd.read_csv(display_path, index_col=0).loc[ordered_ids].to_numpy(float)
        if not np.allclose(previous_display, reconstructed[order, :, :5].reshape(len(ids), -1),
                           rtol=1e-12, atol=1e-14, equal_nan=True):
            raise ValueError("Restored profile would change the saved displayed values")
    k = np.flatnonzero((conditions[:, 0] == "A300") & (conditions[:, 1] == "20260520"))
    if len(k) != 1 or "AWCON" not in cells:
        raise ValueError("The previously reviewed AWCON–A300 illustration is unavailable")
    k, c = int(k[0]), cells.index("AWCON")
    if arrays["status"][k, c] != "retained" or not np.isfinite(arrays["templates"][c]).all():
        raise ValueError("The reviewed illustration is not retained under the saved current gate")
    if not np.allclose(arrays["means"][k, c].reshape(8, 5).mean(axis=1), arrays["mean_bins"][k, c]):
        raise ValueError("Saved raw illustration and eight-bin means differ")
    files = _profile_figure(figures, reconstructed[order], cells, ordered_ids, tree, order)
    files += _model_figure(figures, arrays, cells, ordered_ids,
                           aggregated_coefficients[order], tree, order, (k, c))
    profile_clip = int(((reconstructed[order, :, :5] < -.5) | (reconstructed[order, :, :5] > 1.)).sum())
    amplitude_clip = int((np.abs(aggregated_coefficients) > .6).sum())
    references = [repo / "notebook/02_reproducibility_inspection.ipynb",
                  repo / "notebook/response_structure_poster.py",
                  repo / "reports/exploration_response_profiles_20261001/figures/01_response_profile.png",
                  repo / "reports/response_structure_poster_20261001/curve_compression_poster.svg"]
    inputs = [arrays_path, parameters_path, coefficients_path, rdm_path, row_path]
    if display_path.exists():
        inputs.append(display_path)
    metadata = dict(
        display_only=True, refit=False, primary_snr=.5, min_animals=parameters["min_animals"],
        source_sha256={str(path): _hash(path) for path in inputs},
        reference_sha256={str(path): _hash(path) for path in references}, code_sha256=_hash(__file__),
        profile_style="Notebook 02 final Panel A, source cells 27 and 29; no template strip",
        profile_figsize_inches=[14.5, 8.2], model_figsize_inches=[14.8, 14.0], dpi=300,
        profile_display_window=[0, 25], profile_bins=5, model_fit_window=[0, 40], model_bins=8,
        cells=cells, row_order=ordered_ids,
        row_order_rule="Existing average-linkage hierarchy of the unfiltered 0–40 s raw-curve RDM; reconstructed saved amplitudes do not determine the hierarchy",
        row_order_matches_existing=True, profile_values_match_existing=True,
        response_cmap="RdBu_r", profile_color_limits=[-.5, 0, 1], amplitude_color_limits=[-.6, 0, .6],
        profile_saturated_bins=profile_clip, amplitude_saturated_entries=amplitude_clip,
        illustration=dict(strain="A300", cell="AWCON", block="20260520", n_animals=int(arrays["counts"][k, c]),
                          rule="Same previously inspected strong AWCON example as the poster reference; illustration only",
                          snr=float(arrays["snr"][k, c]), coefficient=float(arrays["coefficients"][k, c]),
                          mean_curve=arrays["means"][k, c].tolist(), mean_bins=arrays["mean_bins"][k, c].tolist(),
                          template=arrays["templates"][c].tolist(),
                          smooth_template="Display-only PCHIP through eight saved template nodes; no extrapolation"),
        files=files,
    )
    metadata_path = figures / "profile_display_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    caption_path = figures / "profile_model_captions.txt"
    caption_path.write_text(
        "01_response_profile_5bin: Saved individual-SNR ≥ 0.5 amplitude × template reconstructions for all 106 samples and 13 neuron classes, displayed as five 5-s bins from 0–25 s. The model and gate use the full 0–40 s window. Notebook 02 Panel A's dendrogram, white class gaps, colored headers, right-side sample labels, offset markers and color scale are restored; no template strip is included. The display hierarchy is the existing average-linkage hierarchy of unfiltered 0–40 s raw-curve distances, not a new clustering of screened reconstructions. All data and sample order match the preceding saved display. Missing values are gray; screened entries have analysis value zero, which does not establish physiological absence. Multiple dates are averaged equally over available supported dates. "
        f"The −0.5 to 1 ΔF/F₀ display range saturates {profile_clip} bins without changing saved values.\n\n"
        "01b_response_model: Top illustrates the previously selected AWCON–A300 condition using its saved raw mean (7 animals), eight 5-s means, and the current shared template × coefficient. The approximate sign denotes a model approximation, not exact equality. The template was fitted across retained conditions, not from this example alone. Gray shading marks stimulation (0–10 s); smooth template interpolation is display-only through the eight model nodes. This strong example is not a representative validation result. Below, 13 cell-specific eight-bin templates are aligned to the corresponding columns of the full 106-sample signed-amplitude matrix, using the same sample order as the separate profile. Templates have unit RMS and largest-absolute bin positive; coefficient sign is relative to that template and does not generically identify excitation or inhibition. The matrix uses individual SNR ≥ 0.5, minimum two animals, and equal available-date aggregation. Zero is a screening decision; gray denotes unavailable values. "
        f"The ±0.6 ΔF/F₀ amplitude range saturates {amplitude_clip} entries without changing saved coefficients.\n",
        encoding="utf-8")
    return dict(files=files, parameters=str(metadata_path), captions=str(caption_path),
                profile_saturated_bins=profile_clip, amplitude_saturated_entries=amplitude_clip,
                row_order=ordered_ids)
