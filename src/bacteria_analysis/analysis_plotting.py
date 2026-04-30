"""Compact plotting primitives for function-first analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

plt.rcParams["figure.max_open_warning"] = 0


def plot_rdm_heatmap_pair(neural: pd.DataFrame, chemical: pd.DataFrame, *, title: str = "Aligned RDMs") -> Figure:
    """Return a two-panel heatmap for aligned neural and chemical RDMs."""

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    matrices = (("Neural", neural), ("Chemical", chemical))
    finite_values = np.concatenate(
        [
            matrix.to_numpy(dtype=float, copy=False)[np.isfinite(matrix.to_numpy(dtype=float, copy=False))]
            for _, matrix in matrices
        ]
    )
    vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
    for axis, (label, matrix) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix.to_numpy(dtype=float, copy=False), cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(label)
        axis.set_xticks(range(len(matrix.columns)), matrix.columns.astype(str), rotation=90)
        axis.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
    figure.colorbar(image, ax=axes, shrink=0.85, label="distance")
    figure.suptitle(title)
    return figure


def plot_rdm_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str = "RDM",
    colorbar_label: str = "distance",
) -> Figure:
    """Return a single compact RDM heatmap."""

    figure, axis = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    values = matrix.to_numpy(dtype=float, copy=False)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns.astype(str), rotation=90)
    axis.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.85, label=colorbar_label)
    return figure


def plot_rdm_heatmap_grid(rdms: dict[str, pd.DataFrame], *, title: str = "RDM comparison") -> Figure:
    """Return a compact row of RDM heatmaps."""

    count = max(1, len(rdms))
    figure, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.3), constrained_layout=True)
    if count == 1:
        axes = [axes]

    finite_values = []
    for matrix in rdms.values():
        values = matrix.to_numpy(dtype=float, copy=False)
        finite_values.extend(values[np.isfinite(values)].tolist())
    vmax = float(np.nanmax(finite_values)) if finite_values else 1.0

    image = None
    for axis, (label, matrix) in zip(axes, rdms.items(), strict=False):
        values = matrix.to_numpy(dtype=float, copy=False)
        image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(label)
        axis.set_xticks(range(len(matrix.columns)), matrix.columns.astype(str), rotation=90)
        axis.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="distance")
    figure.suptitle(title)
    return figure


def plot_score_bars(
    table: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
    ylabel: str = "RSA similarity",
) -> Figure:
    """Return a concise horizontal score bar plot."""

    figure, axis = plt.subplots(figsize=(5.0, max(2.6, 0.32 * max(len(table), 1))), constrained_layout=True)
    if not table.empty:
        labels = table[label_column].astype(str).tolist()
        values = pd.to_numeric(table[value_column], errors="coerce").fillna(0.0).to_numpy()
        positions = np.arange(len(labels))
        axis.barh(positions, values, color="#80b1d3", edgecolor="#4a5568")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(ylabel)
    return figure


def plot_null_distribution(null_values: np.ndarray, observed: float, *, title: str = "Label-shuffle null") -> Figure:
    """Return a compact null-distribution plot with the observed RSA marked."""

    figure, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        axis.hist(finite, bins=min(30, max(5, finite.size)), color="#9ecae1", edgecolor="#4a5568")
    if np.isfinite(observed):
        axis.axvline(observed, color="#c2410c", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("RSA similarity")
    axis.set_ylabel("count")
    return figure


def plot_subset_stability(subset_results: pd.DataFrame, *, title: str = "Subset stability") -> Figure:
    """Return a histogram for stimulus-subset RSA stability draws."""

    figure, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    if not subset_results.empty and "rsa_similarity" in subset_results.columns:
        values = pd.to_numeric(subset_results["rsa_similarity"], errors="coerce").dropna()
        if not values.empty:
            axis.hist(values, bins=min(30, max(5, len(values))), color="#a7f3d0", edgecolor="#4a5568")
    axis.set_title(title)
    axis.set_xlabel("RSA similarity")
    axis.set_ylabel("count")
    return figure


__all__ = [
    "plot_null_distribution",
    "plot_rdm_heatmap",
    "plot_rdm_heatmap_grid",
    "plot_rdm_heatmap_pair",
    "plot_score_bars",
    "plot_subset_stability",
]
