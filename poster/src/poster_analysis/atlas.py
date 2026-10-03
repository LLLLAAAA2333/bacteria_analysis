"""Saved response representation: loading, aggregation and reusable figure layouts.

No raw-data processing or template fitting occurs here. Arrays retain their saved
missing values and screened zeros; figure functions return open Matplotlib figures.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .constants import NEURON_COLORS


def load_atlas_data(report):
    """Load the reviewed individual-SNR representation and its figure row order."""
    report = Path(report)
    tables = report / "tables"
    with np.load(tables / "representation_arrays.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    neurons = arrays["cells"].tolist()
    coefficients = pd.read_csv(tables / "strain_coefficients.csv", index_col=0).loc[:, neurons]
    order = pd.read_csv(tables / "figure_row_order.csv")["strain"].astype(str).tolist()
    parameters = json.loads((report / "representation_parameters.json").read_text())
    if parameters["snr_basis"] != "individual" or parameters["primary_threshold"] != .5:
        raise ValueError("The poster currently uses the saved individual-SNR >= 0.5 representation")
    if arrays["templates"].shape != (len(neurons), 8):
        raise ValueError("Expected eight 5-s template bins per neuron")
    if len(set(order)) != len(coefficients) or set(order) != set(coefficients.index):
        raise ValueError("Saved row order and coefficient sample IDs differ")
    return dict(arrays=arrays, neurons=neurons, coefficients=coefficients,
                sample_order=order, parameters=parameters,
                conditions=pd.read_csv(tables / "condition_metrics.csv"),
                sensitivity=pd.read_csv(tables / "sensitivity_metrics.csv", dtype={"threshold": str}),
                filter_states=pd.read_csv(tables / "filter_state_by_strain_cell.csv"))


def aggregate_conditions(values, conditions, sample_ids):
    """Finite-value mean of available condition blocks, equally weighted per sample.

    ``values`` is condition × any trailing dimensions; ``conditions`` is condition
    × (sample ID, block). No supported block leaves a value NaN, rather than zero.
    """
    values, conditions = np.asarray(values, float), np.asarray(conditions)
    if values.shape[0] != len(conditions) or conditions.ndim != 2:
        raise ValueError("Values and condition labels must share their first dimension")
    result = []
    for sample in sample_ids:
        selected = values[conditions[:, 0] == sample]
        finite = np.isfinite(selected)
        counts = finite.sum(axis=0)
        totals = np.where(finite, selected, 0).sum(axis=0)
        result.append(np.divide(totals, counts, out=np.full(values.shape[1:], np.nan), where=counts > 0))
    return np.asarray(result)


def _response_cmap():
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DCDCDC")
    return cmap


def _extend(values, limits):
    finite = np.asarray(values)[np.isfinite(values)]
    low = finite.size and finite.min() < limits[0]
    high = finite.size and finite.max() > limits[1]
    return "both" if low and high else "min" if low else "max" if high else "neither"


def plot_response_atlas(profile, neurons, sample_ids, limits=(-.5, 1.)):
    """Plot sample × neuron × five-bin reconstructions in caller-supplied order."""
    profile = np.asarray(profile, float)
    if profile.shape != (len(sample_ids), len(neurons), 5):
        raise ValueError("Profile must have sample × neuron × 5 dimensions")
    display = np.full((len(sample_ids), len(neurons) * 6 - 1), np.nan)
    for column in range(len(neurons)):
        display[:, column * 6:column * 6 + 5] = profile[:, column]
    fig, ax = plt.subplots(figsize=(14.5, 8.0), layout="constrained")
    image = ax.imshow(display, cmap=_response_cmap(),
                      norm=TwoSlopeNorm(vmin=limits[0], vcenter=0, vmax=limits[1]),
                      aspect="auto", interpolation="nearest", rasterized=True)
    colors = [NEURON_COLORS.get(neuron, "#26354A") for neuron in neurons]
    for column, (neuron, color) in enumerate(zip(neurons, colors)):
        start = column * 6
        if column:
            ax.axvspan(start - 1.5, start - .5, color="white", linewidth=0)
        ax.axvline(start + 1.5, color="#36465A", ls=(0, (3, 3)), lw=.7)
        ax.text(start + 2, 1.02, neuron, color=color, ha="center", weight="bold",
                fontsize=10, transform=ax.get_xaxis_transform())
    ax.set_xticks([j * 6 + k for j in range(len(neurons)) for k in range(5)],
                  list(range(1, 6)) * len(neurons), fontsize=6)
    rows = np.arange(0, len(sample_ids), 10)
    ax.set_yticks(rows, np.asarray(sample_ids)[rows], fontsize=8)
    ax.set(xlabel="Time bins (5 s each; dashed line: stimulus offset at 10 s)",
           ylabel=f"Bacterial stimuli (n = {len(sample_ids)}; fixed profile order)")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=.65, pad=.02, extend=_extend(profile, limits))
    bar.set_label(r"Reconstructed $\Delta F/F_0$")
    return fig


def plot_method_strip(mean_curve, binned, template, coefficient, label="AWCON · A300"):
    """Explain saved 40-s mean → eight bins ≈ template × signed coefficient."""
    mean_curve, binned, template = map(np.asarray, (mean_curve, binned, template))
    if mean_curve.shape != (40,) or binned.shape != (8,) or template.shape != (8,):
        raise ValueError("Expected 40 one-second means and eight model bins")
    fig = plt.figure(figsize=(14, 2.9), layout="constrained")
    grid = fig.add_gridspec(1, 6, width_ratios=[1, .12, 1, .12, 1, .55])
    axes = [fig.add_subplot(grid[0, j]) for j in (0, 2, 4, 5)]
    for j, symbol in [(1, "→"), (3, "≈")]:
        separator = fig.add_subplot(grid[0, j])
        separator.axis("off")
        separator.text(.5, .5, symbol, ha="center", va="center", fontsize=25, color="#687F99")
    nodes = np.arange(2.5, 40, 5)
    axes[0].plot(np.arange(40), mean_curve, color="#26354A", lw=2)
    axes[0].set(title=f"Mean response · {label}", ylabel=r"$\Delta F/F_0$")
    axes[1].stairs(binned, np.arange(0, 41, 5), color="#4B718A", fill=True, alpha=.55)
    axes[1].set_title("Eight 5-s means")
    axes[2].plot(nodes, template, "o-", color="#2166AC", lw=1.8, ms=3)
    axes[2].set_title("Shared template (RMS = 1)")
    for ax in axes[:3]:
        ax.axvspan(0, 10, color="#D8DEE5", alpha=.35, zorder=-2)
        ax.axhline(0, color="#687F99", lw=.6)
        ax.set(xlim=(0, 40), xticks=[0, 10, 40], xlabel="Time (s)")
        ax.spines[["top", "right"]].set_visible(False)
    limits = (min(0, np.nanmin(mean_curve), np.nanmin(binned)) - .08,
              max(np.nanmax(mean_curve), np.nanmax(binned)) * 1.08)
    axes[0].set_ylim(limits)
    axes[1].set_ylim(limits)
    axes[3].axis("off")
    axes[3].text(.5, .64, "Signed amplitude", ha="center", fontsize=11)
    axes[3].text(.5, .39, f"× {coefficient:+.2f}", color="#B2182B", ha="center", fontsize=25)
    return fig


def plot_template_atlas(templates, coefficients, neurons, sample_ids, limits=(-.6, .6)):
    """Supporting view of full eight-bin templates and signed strain amplitudes."""
    fig = plt.figure(figsize=(13.5, 7.8), layout="constrained")
    grid = fig.add_gridspec(2, len(neurons), height_ratios=[1, 5])
    vmax = np.nanmax(np.abs(templates)) * 1.1
    for j, neuron in enumerate(neurons):
        ax = fig.add_subplot(grid[0, j])
        ax.plot(np.arange(2.5, 40, 5), templates[j], color="#2166AC", lw=1.5)
        ax.axvspan(0, 10, color="#D8DEE5", alpha=.4)
        ax.axhline(0, color="#687F99", lw=.5)
        ax.set(title=neuron, ylim=(-vmax, vmax), xlim=(0, 40), xticks=[0, 40], yticks=[])
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right", "left"]].set_visible(False)
    ax = fig.add_subplot(grid[1, :])
    image = ax.imshow(coefficients, cmap=_response_cmap(),
                      norm=TwoSlopeNorm(vmin=limits[0], vcenter=0, vmax=limits[1]),
                      aspect="auto", interpolation="nearest", rasterized=True)
    rows = np.arange(0, len(sample_ids), 10)
    ax.set(xticks=np.arange(len(neurons)), xticklabels=neurons,
           yticks=rows, yticklabels=np.asarray(sample_ids)[rows],
           xlabel="Neuron class", ylabel="Bacterial stimuli (fixed profile order)")
    fig.colorbar(image, ax=ax, shrink=.8, pad=.01, extend=_extend(coefficients, limits)).set_label(
        r"Signed amplitude ($\Delta F/F_0$)")
    return fig
