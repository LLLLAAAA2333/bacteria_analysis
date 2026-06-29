"""Compact plotting primitives for function-first analyses."""

from __future__ import annotations

import warnings
from itertools import combinations, cycle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from ...features.anchor import VIEW_WINDOWS, merged_neuron_order

plt.rcParams["figure.max_open_warning"] = 0


def finish_figure(
    figure: Figure, output_path: str | Path | None, *, dpi: int = 150
) -> Figure | None:
    """Save a figure when a path is provided, otherwise return it for display."""

    if output_path is None:
        return figure
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return None


def _finish_anchor_reference_figure(
    figure: Figure, output_path: str | Path | None, *, dpi: int = 220
) -> Figure | None:
    """Save anchor-review figures with the legacy reference sizing contract."""

    if output_path is None:
        return figure
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return None


def plot_rdm_heatmap_pair(
    neural: pd.DataFrame, chemical: pd.DataFrame, *, title: str = "Aligned RDMs"
) -> Figure:
    """Return a two-panel heatmap for aligned neural and chemical RDMs."""

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    matrices = (("Neural", neural), ("Chemical", chemical))
    finite_values = np.concatenate(
        [
            matrix.to_numpy(dtype=float, copy=False)[
                np.isfinite(matrix.to_numpy(dtype=float, copy=False))
            ]
            for _, matrix in matrices
        ]
    )
    vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
    for axis, (label, matrix) in zip(axes, matrices, strict=True):
        image = axis.imshow(
            matrix.to_numpy(dtype=float, copy=False),
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set_title(label)
        axis.set_xticks(
            range(len(matrix.columns)), matrix.columns.astype(str), rotation=90
        )
        axis.set_yticks(range(len(matrix.index)), matrix.index.astype(str))
    figure.colorbar(image, ax=axes, shrink=0.85, label="distance")
    figure.suptitle(title)
    return figure


def write_rdm_heatmap_pair(
    neural: pd.DataFrame,
    chemical: pd.DataFrame,
    output_path: str | Path | None,
    *,
    display_order: list[str] | None = None,
    title: str = "Aligned RDMs",
    dpi: int = 150,
) -> Figure | None:
    """Render an aligned neural/chemical RDM pair and optionally save it."""

    neural_display = _ordered_square(neural, display_order)
    chemical_display = _ordered_square(chemical, display_order)
    return finish_figure(
        plot_rdm_heatmap_pair(neural_display, chemical_display, title=title),
        output_path,
        dpi=dpi,
    )


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


def plot_rdm_heatmap_grid(
    rdms: dict[str, pd.DataFrame], *, title: str = "RDM comparison"
) -> Figure:
    """Return a compact row of RDM heatmaps."""

    count = max(1, len(rdms))
    figure, axes = plt.subplots(
        1, count, figsize=(3.2 * count, 3.3), constrained_layout=True
    )
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
        axis.set_xticks(
            range(len(matrix.columns)), matrix.columns.astype(str), rotation=90
        )
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

    figure, axis = plt.subplots(
        figsize=(5.0, max(2.6, 0.32 * max(len(table), 1))), constrained_layout=True
    )
    if not table.empty:
        labels = table[label_column].astype(str).tolist()
        values = (
            pd.to_numeric(table[value_column], errors="coerce").fillna(0.0).to_numpy()
        )
        positions = np.arange(len(labels))
        axis.barh(positions, values, color="#80b1d3", edgecolor="#4a5568")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(ylabel)
    return figure


def plot_null_distribution(
    null_values: np.ndarray, observed: float, *, title: str = "Label-shuffle null"
) -> Figure:
    """Return a compact null-distribution plot with the observed RSA marked."""

    figure, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        axis.hist(
            finite,
            bins=min(30, max(5, finite.size)),
            color="#9ecae1",
            edgecolor="#4a5568",
        )
    if np.isfinite(observed):
        axis.axvline(observed, color="#c2410c", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("RSA similarity")
    axis.set_ylabel("count")
    return figure


def write_null_distribution(
    null_values: np.ndarray,
    observed: float,
    output_path: str | Path | None,
    *,
    title: str = "Label-shuffle null",
    y_mode: str = "count",
    dpi: int = 150,
) -> Figure | None:
    """Render a compact null-distribution figure and optionally save it."""

    if y_mode not in {"count", "fraction"}:
        raise ValueError("y_mode must be 'count' or 'fraction'")

    figure, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        weights = None
        if y_mode == "fraction":
            weights = np.ones_like(finite, dtype=float) / finite.size
        axis.hist(
            finite,
            bins=min(30, max(5, finite.size)),
            weights=weights,
            color="#9ecae1",
            edgecolor="#4a5568",
        )
    if np.isfinite(observed):
        axis.axvline(observed, color="#c2410c", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("RSA similarity")
    axis.set_ylabel("fraction" if y_mode == "fraction" else "count")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_subset_stability(
    subset_results: pd.DataFrame, *, title: str = "Subset stability"
) -> Figure:
    """Return a histogram for stimulus-subset RSA stability draws."""

    figure, axis = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    if not subset_results.empty and "rsa_similarity" in subset_results.columns:
        values = pd.to_numeric(
            subset_results["rsa_similarity"], errors="coerce"
        ).dropna()
        if not values.empty:
            axis.hist(
                values,
                bins=min(30, max(5, len(values))),
                color="#a7f3d0",
                edgecolor="#4a5568",
            )
    axis.set_title(title)
    axis.set_xlabel("RSA similarity")
    axis.set_ylabel("count")
    return figure


def write_subset_stability(
    subset_results: pd.DataFrame,
    output_path: str | Path | None,
    *,
    title: str = "Subset stability",
    dpi: int = 150,
) -> Figure | None:
    """Render stimulus-subset RSA stability and optionally save it."""

    return finish_figure(
        plot_subset_stability(subset_results, title=title), output_path, dpi=dpi
    )


def plot_fixed_class_permutation(
    summary: pd.DataFrame,
    output_path: str | Path | None,
    *,
    class_limit: int,
    dpi: int = 150,
) -> Figure | None:
    """Render observed class RSA against fixed-class permutation bands."""

    plot_frame = (
        summary.sort_values("observed_rsa", ascending=False)
        .head(class_limit)
        .iloc[::-1]
        .copy()
    )
    figure, axis = plt.subplots(
        figsize=(7.0, max(3.2, 0.28 * len(plot_frame) + 1.4)), constrained_layout=True
    )
    if not plot_frame.empty:
        y = np.arange(len(plot_frame))
        axis.hlines(
            y,
            plot_frame["null_q95"],
            plot_frame["null_q99"],
            color="#8a8a8a",
            linewidth=2.0,
        )
        points = axis.scatter(
            plot_frame["observed_rsa"],
            y,
            c=plot_frame["observed_percentile"],
            cmap="viridis",
            vmin=50,
            vmax=100,
            s=42,
            edgecolor="#222222",
            linewidth=0.35,
        )
        axis.set_yticks(y, plot_frame["category"].astype(str).tolist())
        figure.colorbar(points, ax=axis, pad=0.02, label="observed percentile")
    axis.axvline(0.0, color="#c9c9c9", linewidth=0.9)
    axis.set_xlabel("response-window RSA")
    axis.set_title("Fixed-class permutation")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_reselection_stability(
    stability: pd.DataFrame,
    observed: pd.DataFrame,
    output_path: str | Path | None,
    *,
    class_limit: int,
    dpi: int = 150,
) -> Figure | None:
    """Render how often each class reappears in stimulus resamples."""

    merged = stability.merge(
        observed.loc[:, ["model_id", "response_window_rsa"]], on="model_id", how="left"
    )
    plot_frame = merged.sort_values(
        ["top3_frequency", "response_window_rsa"], ascending=False
    )
    plot_frame = plot_frame.head(class_limit).iloc[::-1].copy()
    figure, axis = plt.subplots(
        figsize=(7.2, max(3.2, 0.32 * len(plot_frame) + 1.5)), constrained_layout=True
    )
    if not plot_frame.empty:
        y = np.arange(len(plot_frame))
        axis.barh(
            y - 0.18,
            plot_frame["top1_frequency"],
            height=0.16,
            color="#4c78a8",
            label="top 1",
        )
        axis.barh(
            y, plot_frame["top3_frequency"], height=0.16, color="#f58518", label="top 3"
        )
        axis.barh(
            y + 0.18,
            plot_frame["top5_frequency"],
            height=0.16,
            color="#54a24b",
            label="top 5",
        )
        axis.set_yticks(y, plot_frame["category"].astype(str).tolist())
        axis.legend(fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_xlabel("reselection frequency")
    axis.set_title("Reselection stability")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_top_class_rdm_comparison(
    *,
    primary_neural: pd.DataFrame,
    full_chemical: pd.DataFrame,
    candidate: object,
    labels: list[str],
    stimulus_sample_map: pd.DataFrame,
    output_path: str | Path | None,
    dpi: int = 150,
) -> Figure | None:
    """Render neural, full-chemical, and top-class chemical RDMs side by side."""

    order = [
        label
        for label in labels
        if label in primary_neural.index and label in primary_neural.columns
    ]
    display_labels = _stimulus_display_labels(order, stimulus_sample_map)
    panels = {
        "Neural\nresponse window": primary_neural.loc[order, order],
        "Chemical\nall retained": full_chemical.loc[order, order],
        f"Chemical\n{candidate.category}": candidate.chemical.loc[order, order],
    }

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#ffffff")
    for axis, (title, matrix) in zip(axes, panels.items(), strict=True):
        display = _mask_diagonal(matrix)
        values = display.to_numpy(float)
        finite = values[np.isfinite(values)]
        image = axis.imshow(
            values,
            cmap=cmap,
            vmin=float(np.nanmin(finite)) if finite.size else None,
            vmax=float(np.nanmax(finite)) if finite.size else None,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xticks(
            range(len(display_labels)), display_labels, rotation=90, fontsize=6
        )
        axis.set_yticks(range(len(display_labels)), display_labels, fontsize=6)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.colorbar(image, ax=axis, shrink=0.78, label="distance")
    figure.suptitle(f"Top class RDM comparison: {candidate.category}")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_summary_scorecard(
    shortlist: pd.DataFrame,
    output_path: str | Path | None,
    *,
    class_limit: int,
    dpi: int = 150,
) -> Figure | None:
    """Render a compact evidence scorecard for shortlisted classes."""

    plot_frame = shortlist.head(class_limit).iloc[::-1].copy()
    figure, axis = plt.subplots(
        figsize=(7.8, max(3.0, 0.34 * len(plot_frame) + 1.6)), constrained_layout=True
    )
    if plot_frame.empty:
        axis.text(0.5, 0.5, "No shortlisted classes", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    observed = plot_frame["response_window_rsa"].to_numpy(float)
    span = max(1e-9, float(np.nanmax(observed) - np.nanmin(observed)))
    values = np.column_stack(
        [
            (observed - float(np.nanmin(observed))) / span,
            plot_frame["observed_percentile"].to_numpy(float) / 100.0,
            plot_frame["top3_frequency"].fillna(0.0).to_numpy(float),
            plot_frame["fixed_signal_pass"].astype(float).to_numpy(),
        ]
    )
    image = axis.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.set_yticks(
        np.arange(len(plot_frame)), plot_frame["category"].astype(str).tolist()
    )
    axis.set_xticks(
        np.arange(4),
        ["RSA", "fixed pct", "top3 freq", "q95 pass"],
        rotation=25,
        ha="right",
    )
    for row in range(values.shape[0]):
        annotations = [
            f"{plot_frame['response_window_rsa'].iloc[row]:.3f}",
            f"{plot_frame['observed_percentile'].iloc[row]:.1f}",
            f"{plot_frame['top3_frequency'].iloc[row]:.2f}",
            "yes" if bool(plot_frame["fixed_signal_pass"].iloc[row]) else "no",
        ]
        for col, text in enumerate(annotations):
            axis.text(
                col, row, text, ha="center", va="center", fontsize=7, color="white"
            )
    axis.set_title("Taxonomy class scorecard")
    figure.colorbar(image, ax=axis, pad=0.02, label="scaled evidence")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_class_chemical_rdm_similarity(
    *,
    pairwise: pd.DataFrame,
    top_class_similarity: pd.DataFrame,
    output_path: str | Path | None,
    dpi: int = 150,
) -> Figure | None:
    """Render class-to-class chemical RDM similarity and top-class neighbors."""

    matrix = _class_similarity_matrix(pairwise)
    if matrix.empty:
        figure, axis = plt.subplots(figsize=(5.0, 3.0), constrained_layout=True)
        axis.text(0.5, 0.5, "No class pairs", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, max(4.0, 0.28 * len(matrix) + 1.4)),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
        constrained_layout=True,
    )
    values = matrix.to_numpy(float)
    off_diagonal = values[~np.eye(len(matrix), dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    vmin = min(-0.1, float(np.nanmin(finite)) if finite.size else -0.1)
    vmax = max(0.85, float(np.nanmax(finite)) if finite.size else 0.85)
    image = axes[0].imshow(
        values, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    labels = matrix.index.astype(str).tolist()
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=6)
    axes[0].tick_params(length=0)
    axes[0].set_title("Class chemical RDM RSA")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03, label="RSA")

    top = top_class_similarity.sort_values("chemical_rdm_rsa", ascending=True).copy()
    if not top.empty:
        y = np.arange(len(top))
        axes[1].barh(y, top["chemical_rdm_rsa"], color="#4c78a8")
        axes[1].set_yticks(y, top["other_category"].astype(str).tolist(), fontsize=7)
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_xlabel("RSA vs top class")
    axes[1].set_title("Top-class neighbors")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_class_vs_full_chemical_similarity(
    *,
    similarity: pd.DataFrame,
    output_path: str | Path | None,
    dpi: int = 150,
) -> Figure | None:
    """Render each class RDM's similarity to the full chemical RDM."""

    plot_frame = similarity.sort_values(
        "class_vs_full_chemical_rdm_rsa", ascending=True
    ).copy()
    figure, axis = plt.subplots(
        figsize=(7.2, max(3.2, 0.28 * len(plot_frame) + 1.4)), constrained_layout=True
    )
    if not plot_frame.empty:
        y = np.arange(len(plot_frame))
        values = plot_frame["class_vs_full_chemical_rdm_rsa"].to_numpy(float)
        axis.barh(y, values, color="#4c78a8")
        axis.set_yticks(y, plot_frame["category"].astype(str).tolist())
        median = float(np.nanmedian(values))
        axis.axvline(
            median,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=f"median = {median:.2f}",
        )
        axis.legend(fontsize=8)
    axis.set_xlabel("RSA vs full chemical RDM")
    axis.set_title("Class similarity to full chemical RDM")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_rdm_heatmaps(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    dpi: int = 220,
) -> Figure | None:
    """Render one prototype-distance RDM per anchor view."""

    groups = (
        list(pairwise.groupby("view_name", sort=True)) if not pairwise.empty else []
    )
    count = max(1, len(groups))
    figure, axes = plt.subplots(1, count, figsize=(6.0 * count, 5.0))
    axes = np.atleast_1d(axes).ravel()
    if not groups:
        axes[0].text(0.5, 0.5, "No anchor distances", ha="center", va="center")
        axes[0].axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    for axis, (view_name, view) in zip(axes, groups, strict=False):
        labels = sorted(
            set(view["left_label"].astype(str)) | set(view["right_label"].astype(str))
        )
        matrix = _anchor_distance_matrix(view, labels)
        image = axis.imshow(
            matrix.to_numpy(float), cmap="magma", interpolation="nearest"
        )
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(
            [_short_anchor_label(label) for label in labels], rotation=90, fontsize=7
        )
        axis.set_yticklabels(
            [_short_anchor_label(label) for label in labels], fontsize=7
        )
        axis.set_xlabel(_anchor_view_label(str(view_name)))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_anchor_clustered_rdm_heatmaps(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    dpi: int = 220,
) -> Figure | None:
    """Render the requested anchor prototype RDM with clustered labels."""

    view = pairwise.loc[pairwise["view_name"].astype(str).eq(view_name)].copy()
    figure, axis = plt.subplots(figsize=(8.8, 7.6))
    if view.empty:
        axis.text(0.5, 0.5, "No anchor distances", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    labels = sorted(
        set(view["left_label"].astype(str)) | set(view["right_label"].astype(str))
    )
    matrix = _anchor_distance_matrix(view, labels)
    ordered_labels = _clustered_anchor_order(matrix)
    ordered = matrix.loc[ordered_labels, ordered_labels]
    image = axis.imshow(ordered.to_numpy(float), cmap="magma", interpolation="nearest")
    axis.set_xticks(range(len(ordered_labels)))
    axis.set_yticks(range(len(ordered_labels)))
    axis.set_xticklabels(
        [_prototype_display_label(label) for label in ordered_labels],
        rotation=90,
        fontsize=8,
    )
    axis.set_yticklabels(
        [_prototype_display_label(label) for label in ordered_labels], fontsize=8
    )
    axis.set_title("Response-window clustered prototype RDM", fontsize=12)
    axis.set_xlabel("Clustered date / anchor stimulus labels", fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Prototype distance", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.tight_layout()
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_date_mds(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    date_order: list[str] | None = None,
    dpi: int = 220,
) -> Figure | None:
    """Render a two-dimensional view of anchor date prototypes."""

    view = pairwise.loc[pairwise["view_name"].astype(str).eq(view_name)].copy()
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    if view.empty:
        axis.text(0.5, 0.5, "No anchor distances", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    labels = sorted(
        set(view["left_label"].astype(str)) | set(view["right_label"].astype(str))
    )
    matrix = _anchor_distance_matrix(view, labels)
    coords = _classical_mds_frame(matrix)
    coords["date"] = coords["label"].map(_label_date)
    coords["stim_name"] = coords["label"].map(_label_stim_name)

    if date_order is None:
        ordered_dates = sorted(coords["date"].astype(str).unique())
    else:
        ordered_dates = [
            str(date)
            for date in date_order
            if str(date) in set(coords["date"].astype(str))
        ]

    stimulus_order = sorted(coords["stim_name"].astype(str).unique())
    _stimulus_palette = [
        "#0072B2",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
    ]
    stimulus_colors = {
        stimulus: color
        for stimulus, color in zip(
            stimulus_order,
            cycle(_stimulus_palette),
        )
    }
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    date_markers = {
        date: marker_cycle[index % len(marker_cycle)]
        for index, date in enumerate(ordered_dates)
    }

    for stimulus_name in stimulus_order:
        stimulus_points = coords.loc[
            coords["stim_name"].astype(str).eq(stimulus_name)
        ].copy()
        stimulus_points["date_sort"] = (
            stimulus_points["date"]
            .astype(str)
            .map({date: index for index, date in enumerate(ordered_dates)})
        )
        stimulus_points = stimulus_points.sort_values("date_sort")
        axis.plot(
            stimulus_points["x"].to_numpy(float),
            stimulus_points["y"].to_numpy(float),
            color=stimulus_colors[stimulus_name],
            linewidth=1.4,
            alpha=0.65,
            zorder=1,
        )

    for date in ordered_dates:
        date_points = coords.loc[coords["date"].astype(str).eq(date)].copy()
        if date_points.empty:
            continue
        colors = [
            stimulus_colors[stimulus]
            for stimulus in date_points["stim_name"].astype(str)
        ]
        axis.scatter(
            date_points["x"],
            date_points["y"],
            s=78,
            c=colors,
            marker=date_markers[date],
            edgecolors="#FFFFFF",
            linewidths=0.9,
            zorder=3,
            label=date,
        )

    x_span = float(coords["x"].max()) - float(coords["x"].min())
    y_span = float(coords["y"].max()) - float(coords["y"].min())
    x_pad = max(0.05, x_span * 0.08)
    y_pad = max(0.05, y_span * 0.08)
    for _, row in coords.iterrows():
        axis.text(
            float(row["x"]) + x_pad * 0.20,
            float(row["y"]) + y_pad * 0.18,
            _short_date_label(str(row["date"])),
            fontsize=8,
            color="#555555",
            ha="left",
            va="bottom",
            zorder=4,
        )

    stimulus_handles = [
        plt.Line2D(
            [0], [0], color=stimulus_colors[stimulus], linewidth=2.0, label=stimulus
        )
        for stimulus in stimulus_order
    ]
    date_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=date_markers[date],
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.9,
            markersize=7,
            label=date,
        )
        for date in ordered_dates
    ]
    stimulus_legend = axis.legend(
        handles=stimulus_handles,
        title="anchor stimulus",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    axis.add_artist(stimulus_legend)
    axis.legend(
        handles=date_handles,
        title="date",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    axis.set_title("Anchor-stimulus across dates", fontsize=12)
    axis.set_xlabel("MDS axis 1", fontsize=9)
    axis.set_ylabel("MDS axis 2", fontsize=9)
    axis.grid(color="#E3E3E3", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_aspect("equal", adjustable="box")
    figure.text(
        0.5,
        0.02,
        "Distances are correlation-distance relationships in 2D; lines connect the same stimulus across dates.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.06, 0.86, 0.98))
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_stimulus_resolved_date_pair_heatmap(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    date_order: list[str] | None = None,
    dpi: int = 220,
) -> Figure | None:
    """Render same-anchor cross-date distances by stimulus and date pair."""

    view = pairwise.loc[
        pairwise["view_name"].astype(str).eq(view_name)
        & pairwise["same_stimulus"].astype(bool)
        & (~pairwise["same_date"].astype(bool))
    ].copy()
    if view.empty:
        figure, axis = plt.subplots(figsize=(4.2, 4.8))
        axis.text(0.5, 0.5, "No cross-date anchors", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    stimulus_order = sorted(view["left_stim_name"].astype(str).unique())
    if date_order is None:
        dates = sorted(
            set(view["left_date"].astype(str)) | set(view["right_date"].astype(str))
        )
    else:
        dates = [str(date) for date in date_order]

    matrices: dict[str, pd.DataFrame] = {}
    finite_values: list[float] = []
    for stimulus_name in stimulus_order:
        stimulus_view = view.loc[
            view["left_stim_name"].astype(str).eq(stimulus_name)
        ].copy()
        matrix = pd.DataFrame(np.nan, index=dates, columns=dates, dtype=float)
        for _, row in stimulus_view.iterrows():
            left_date = str(row["left_date"])
            right_date = str(row["right_date"])
            if left_date not in matrix.index or right_date not in matrix.columns:
                continue
            value = float(row["distance"])
            matrix.loc[left_date, right_date] = value
            matrix.loc[right_date, left_date] = value
            finite_values.append(value)
        matrices[stimulus_name] = matrix

    figure, axes = plt.subplots(
        1,
        len(stimulus_order),
        figsize=(4.2 * len(stimulus_order), 4.8),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    if not finite_values:
        axes[0].text(0.5, 0.5, "No finite cross-date anchors", ha="center", va="center")
        axes[0].axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#F3F1EE")
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    threshold = float(np.nanmedian(finite_values))

    last_image = None
    for axis, stimulus_name in zip(axes, stimulus_order, strict=False):
        display = matrices[stimulus_name].copy()
        image = axis.imshow(
            display.to_numpy(float),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        last_image = image
        axis.set_xticks(range(len(dates)))
        axis.set_yticks(range(len(dates)))
        axis.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(dates, fontsize=8)
        axis.set_title(stimulus_name, fontsize=11)
        axis.set_xlabel(_anchor_view_label(view_name), fontsize=9)

        for row_index, _left_date in enumerate(dates):
            for col_index, _right_date in enumerate(dates):
                value = display.iat[row_index, col_index]
                if row_index == col_index:
                    axis.add_patch(
                        plt.Rectangle(
                            (col_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            facecolor="#F8F7F4",
                            edgecolor="none",
                            zorder=3,
                        )
                    )
                    axis.text(
                        col_index,
                        row_index,
                        "\u2014",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#777777",
                        zorder=4,
                    )
                    continue
                if not np.isfinite(value):
                    axis.add_patch(
                        plt.Rectangle(
                            (col_index - 0.5, row_index - 0.5),
                            1.0,
                            1.0,
                            facecolor="#D9D9D9",
                            edgecolor="none",
                            zorder=3,
                        )
                    )
                    axis.text(
                        col_index,
                        row_index,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#555555",
                        zorder=4,
                    )
                    continue
                text_color = "#111111" if value >= threshold else "#F8F7F4"
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    zorder=4,
                )

        for boundary in np.arange(-0.5, len(dates), 1.0):
            axis.axhline(boundary, color="#FFFFFF", linewidth=0.8, alpha=0.65)
            axis.axvline(boundary, color="#FFFFFF", linewidth=0.8, alpha=0.65)

    axes[0].set_ylabel("Date", fontsize=9)
    figure.suptitle("Anchor-stimulus cross-date distance by stimulus", fontsize=13)
    figure.subplots_adjust(left=0.07, right=0.89, bottom=0.18, top=0.82, wspace=0.10)
    if last_image is not None:
        colorbar_axis = figure.add_axes([0.905, 0.20, 0.015, 0.60])
        colorbar = figure.colorbar(last_image, cax=colorbar_axis)
        colorbar.set_label("Raw matched distance\nLower = less drift", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_neuron_activity(
    matrix: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    dpi: int = 220,
) -> Figure | None:
    """Render median neuron activity by anchor stimulus."""

    if matrix.empty:
        figure, axis = plt.subplots(figsize=(10.8, 5.0))
        axis.text(0.5, 0.5, "No activity matrix", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    neuron_order = _profile_cluster_order(matrix)
    stimulus_distance = _profile_distance_matrix(matrix.T)
    stimulus_order = (
        _clustered_anchor_order(stimulus_distance)
        if _can_cluster_anchor_matrix(stimulus_distance)
        else matrix.columns.astype(str).tolist()
    )
    ordered = matrix.loc[neuron_order, stimulus_order]
    ordered_distances = stimulus_distance.loc[stimulus_order, stimulus_order]

    values = ordered.to_numpy(float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        figure, axis = plt.subplots(figsize=(10.8, 5.0))
        axis.text(0.5, 0.5, "No finite activity values", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)
    vmax = float(np.nanquantile(np.abs(finite_values), 0.98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(finite_values)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    distance_values = ordered_distances.to_numpy(float)
    finite_distances = distance_values[np.isfinite(distance_values)]
    distance_vmax = float(np.nanmax(finite_distances)) if finite_distances.size else 1.0
    if not np.isfinite(distance_vmax) or distance_vmax == 0.0:
        distance_vmax = 1.0

    activity_cmap = plt.get_cmap("RdBu_r").copy()
    activity_cmap.set_bad("#F1F1F1")
    distance_cmap = plt.get_cmap("magma").copy()
    distance_cmap.set_bad("#F1F1F1")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, max(5.0, 0.32 * len(ordered.index) + 2.0)),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    activity_axis, distance_axis = axes

    image = activity_axis.imshow(
        values, cmap=activity_cmap, vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    activity_axis.set_xticks(range(len(stimulus_order)))
    activity_axis.set_yticks(range(len(neuron_order)))
    activity_axis.set_xticklabels(stimulus_order, rotation=35, ha="right", fontsize=8)
    activity_axis.set_yticklabels(neuron_order, fontsize=8)
    activity_axis.set_title(
        f"Median activity ({_anchor_view_label(view_name)})", fontsize=12
    )
    activity_axis.set_xlabel("Anchor stimulus", fontsize=9)
    activity_axis.set_ylabel(
        "Neuron (non-ASE L/R merged; ASEL/ASER separate)", fontsize=9
    )
    activity_axis.set_xticks(np.arange(-0.5, len(stimulus_order), 1.0), minor=True)
    activity_axis.set_yticks(np.arange(-0.5, len(neuron_order), 1.0), minor=True)
    activity_axis.grid(which="minor", color="#FFFFFF", linewidth=0.8)
    activity_axis.tick_params(which="minor", bottom=False, left=False)
    _annotate_anchor_heatmap(activity_axis, values, vmax=vmax)
    colorbar = figure.colorbar(image, ax=activity_axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Median baseline-centered dF/F0", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    distance_image = distance_axis.imshow(
        ordered_distances.to_numpy(float),
        cmap=distance_cmap,
        vmin=0.0,
        vmax=distance_vmax,
        interpolation="nearest",
    )
    distance_axis.set_xticks(range(len(stimulus_order)))
    distance_axis.set_yticks(range(len(stimulus_order)))
    distance_axis.set_xticklabels(stimulus_order, rotation=35, ha="right", fontsize=8)
    distance_axis.set_yticklabels(stimulus_order, fontsize=8)
    distance_axis.set_title("Stimulus profile distance", fontsize=12)
    distance_axis.set_xticks(np.arange(-0.5, len(stimulus_order), 1.0), minor=True)
    distance_axis.set_yticks(np.arange(-0.5, len(stimulus_order), 1.0), minor=True)
    distance_axis.grid(which="minor", color="#FFFFFF", linewidth=0.8)
    distance_axis.tick_params(which="minor", bottom=False, left=False)
    _annotate_anchor_heatmap(
        distance_axis,
        ordered_distances.to_numpy(float),
        vmax=distance_vmax,
        decimals=2,
        low_values_dark=True,
    )
    distance_colorbar = figure.colorbar(
        distance_image, ax=distance_axis, fraction=0.046, pad=0.03
    )
    distance_colorbar.set_label("1 - r across neuron medians", fontsize=9)
    distance_colorbar.ax.tick_params(labelsize=8)

    figure.suptitle(
        "Anchor-stimulus neuron activity and profile relationship", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_neuron_time_heatmaps(
    summary: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    aggregator_label: str,
    dpi: int = 220,
) -> Figure | None:
    """Render neuron-by-time activity heatmaps for each anchor stimulus."""

    view = (
        summary.loc[summary["view_name"].astype(str).eq(view_name)].copy()
        if not summary.empty
        else summary.copy()
    )
    if view.empty:
        figure, axis = plt.subplots(figsize=(13.6, 7.2))
        axis.text(0.5, 0.5, "No trajectory summary", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    stimulus_labels = _ordered_activity_stimulus_labels(view)
    timepoints = list(
        VIEW_WINDOWS.get(
            view_name, sorted(view["time_point"].astype(int).unique().tolist())
        )
    )
    neuron_order = [
        neuron
        for neuron in merged_neuron_order()
        if neuron in set(view["neuron"].astype(str))
    ]

    matrices: dict[str, pd.DataFrame] = {}
    finite_values: list[float] = []
    for stimulus_label in stimulus_labels:
        stimulus_view = view.loc[
            view["stimulus_label"].astype(str).eq(stimulus_label)
        ].copy()
        matrix = (
            stimulus_view.pivot(
                index="neuron", columns="time_point", values="activity_value"
            )
            .reindex(index=neuron_order, columns=timepoints)
            .astype(float)
        )
        matrices[stimulus_label] = matrix
        values = matrix.to_numpy(float)
        finite_values.extend(values[np.isfinite(values)].tolist())

    if not finite_values:
        figure, axis = plt.subplots(figsize=(13.6, 7.2))
        axis.text(0.5, 0.5, "No finite trajectory values", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    value_array = np.asarray(finite_values, dtype=float)
    vmax = float(np.nanquantile(np.abs(value_array), 0.98))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(value_array)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    cmap = _anchor_trajectory_cmap()
    cmap.set_bad("#F7F7F7")
    figure, axes = plt.subplots(
        1,
        len(stimulus_labels),
        figsize=(
            max(13.6, 4.9 * len(stimulus_labels)),
            max(7.2, 0.48 * len(neuron_order) + 2.6),
        ),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    last_image = None
    for axis, stimulus_label in zip(axes, stimulus_labels, strict=False):
        matrix = matrices[stimulus_label]
        image = axis.imshow(
            matrix.to_numpy(float),
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        last_image = image
        axis.set_title(_clean_stimulus_label(stimulus_label), fontsize=13, pad=10)
        axis.set_xticks(_time_tick_positions(timepoints, view_name))
        axis.set_xticklabels(_time_tick_labels(timepoints, view_name), fontsize=10)
        axis.set_yticks(range(len(neuron_order)))
        axis.set_yticklabels(neuron_order, fontsize=10)
        axis.set_xticks(np.arange(-0.5, len(timepoints), 1.0), minor=True)
        axis.set_yticks(np.arange(-0.5, len(neuron_order), 1.0), minor=True)
        axis.grid(which="minor", color="#FFFFFF", linewidth=0.35, alpha=0.45)
        axis.tick_params(which="minor", bottom=False, left=False)
        _draw_time_boundaries(axis, timepoints)

    axes[0].set_ylabel("Neuron", fontsize=11)
    axes[len(axes) // 2].set_xlabel("Time (s)", fontsize=11, labelpad=6)
    figure.subplots_adjust(left=0.09, right=0.865, bottom=0.13, top=0.90, wspace=0.10)
    if last_image is not None:
        colorbar_axis = figure.add_axes([0.885, 0.22, 0.016, 0.50])
        colorbar = figure.colorbar(last_image, cax=colorbar_axis)
        label_prefix = f"{aggregator_label.capitalize()} " if aggregator_label else ""
        colorbar.set_label(f"{label_prefix}" + r"$\Delta F/F_0$", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_same_vs_other_distributions(
    contrasts: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    dpi: int = 220,
) -> Figure | None:
    """Render same-anchor versus other-anchor cross-date distances."""

    view = contrasts.loc[contrasts["view_name"].astype(str).eq(view_name)].copy()
    view["distance"] = pd.to_numeric(view["distance"], errors="coerce")
    view = view.dropna(subset=["distance"])
    if view.empty:
        figure, axis = plt.subplots(figsize=(4.2, 4.8))
        axis.text(0.5, 0.5, "No contrasts", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    stimulus_order = sorted(view["anchor_stimulus"].astype(str).unique())
    category_order = ["same", "different"]
    category_colors = {"same": "#0072B2", "different": "#8F8F8F"}
    category_labels = {"same": "same stimulus", "different": "other stimuli"}

    values = view["distance"].to_numpy(float)
    y_min = max(0.0, float(np.nanmin(values)) - 0.08)
    y_max = float(np.nanmax(values)) + 0.08

    figure, axes = plt.subplots(
        1, len(stimulus_order), figsize=(4.2 * len(stimulus_order), 4.8), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(20260423)

    for axis, stimulus_name in zip(axes, stimulus_order, strict=False):
        stimulus_view = view.loc[
            view["anchor_stimulus"].astype(str).eq(stimulus_name)
        ].copy()
        box_data: list[np.ndarray] = []
        box_positions: list[float] = []
        box_categories: list[str] = []
        tick_labels: list[str] = []

        for position, category in enumerate(category_order):
            category_values = stimulus_view.loc[
                stimulus_view["contrast"].astype(str).eq(category),
                "distance",
            ].to_numpy(float)
            tick_labels.append(
                f"{category_labels[category]}\n(n={len(category_values)})"
            )
            if category_values.size == 0:
                continue
            box_data.append(category_values)
            box_positions.append(float(position))
            box_categories.append(category)

        if box_data:
            boxplot = axis.boxplot(
                box_data,
                positions=box_positions,
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#222222", "linewidth": 1.4},
                whiskerprops={"color": "#777777", "linewidth": 1.0},
                capprops={"color": "#777777", "linewidth": 1.0},
                boxprops={"linewidth": 1.2},
            )
            for box, category in zip(boxplot["boxes"], box_categories, strict=False):
                box.set_facecolor(category_colors[category])
                box.set_alpha(0.18)
                box.set_edgecolor(category_colors[category])

        for position, category in enumerate(category_order):
            category_values = stimulus_view.loc[
                stimulus_view["contrast"].astype(str).eq(category),
                "distance",
            ].to_numpy(float)
            if category_values.size == 0:
                continue
            jitter = rng.normal(0.0, 0.045, size=category_values.size)
            axis.scatter(
                np.full(category_values.size, position, dtype=float) + jitter,
                category_values,
                s=24,
                color=category_colors[category],
                edgecolors="none",
                alpha=0.8,
                zorder=3,
            )

        axis.set_title(stimulus_name, fontsize=11)
        axis.set_xticks([0.0, 1.0])
        axis.set_xticklabels(tick_labels, fontsize=9)
        axis.set_xlim(-0.55, 1.55)
        axis.set_ylim(y_min, y_max)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("correlation distance (1 - r)", fontsize=10)
    figure.suptitle("Anchor-stimulus cross-date distances", fontsize=13, y=0.98)
    figure.text(
        0.5,
        0.02,
        "same stimulus = the stimulus compared to itself across dates; other stimuli = the stimulus compared to different stimuli across dates.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.06, 1, 0.90))
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def plot_anchor_ideal_models(
    summary: pd.DataFrame,
    output_path: str | Path | None,
    *,
    dpi: int = 220,
) -> Figure | None:
    """Render anchor distance alignment with stimulus/date ideal models."""

    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    if summary.empty:
        axis.text(0.5, 0.5, "No ideal-model summary", ha="center", va="center")
        axis.axis("off")
        return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)

    metrics = [
        "stimulus_ideal_spearman",
        "date_ideal_spearman",
        "stimulus_ideal_partial_r",
        "date_ideal_partial_r",
    ]
    labels = ["stimulus", "date", "stimulus partial", "date partial"]
    x = np.arange(len(metrics))
    width = 0.35
    for offset, (view_name, view) in zip(
        [-width / 2, width / 2], summary.groupby("view_name", sort=True), strict=False
    ):
        row = view.iloc[0]
        axis.bar(
            x + offset,
            [row[metric] for metric in metrics],
            width=width,
            label=_anchor_view_label(str(view_name)),
        )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("similarity")
    axis.legend(frameon=False)
    figure.tight_layout()
    return _finish_anchor_reference_figure(figure, output_path, dpi=dpi)


def create_rdm_panel_figure(
    *,
    nrows: int,
    figsize: tuple[float, float],
) -> tuple[Figure, np.ndarray, np.ndarray]:
    """Create a two-column RDM panel grid with dedicated colorbar axes."""

    figure = plt.figure(figsize=figsize)
    grid = figure.add_gridspec(
        nrows=nrows,
        ncols=4,
        width_ratios=(1.0, 0.06, 1.0, 0.06),
        left=0.07,
        right=0.96,
        bottom=0.08,
        top=0.9,
        wspace=0.28,
        hspace=0.34,
    )
    axes = np.empty((nrows, 2), dtype=object)
    colorbar_axes = np.empty((nrows, 2), dtype=object)
    for row_index in range(nrows):
        axes[row_index, 0] = figure.add_subplot(grid[row_index, 0])
        colorbar_axes[row_index, 0] = figure.add_subplot(grid[row_index, 1])
        axes[row_index, 1] = figure.add_subplot(grid[row_index, 2])
        colorbar_axes[row_index, 1] = figure.add_subplot(grid[row_index, 3])
    return figure, axes, colorbar_axes


def render_prepared_rdm_panels(
    figure: Figure,
    axes: np.ndarray,
    colorbar_axes: np.ndarray,
    panels: list[tuple[int, int, pd.DataFrame | None, str, str]],
) -> None:
    """Render prepared RDM heatmaps into a two-column panel grid."""

    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#f2f2f2")

    for row_index, col_index, frame, title, fallback_message in panels:
        axis = axes[row_index, col_index]
        colorbar_axis = colorbar_axes[row_index, col_index]
        axis.set_title(title)
        image = _render_prepared_rdm_axis(
            axis,
            frame,
            fallback_message=fallback_message,
            cmap=cmap,
        )
        if image is None:
            colorbar_axis.set_visible(False)
            continue
        try:
            colorbar_axis.set_visible(True)
            figure.colorbar(image, cax=colorbar_axis, label="RDM dissimilarity")
        except Exception as exc:
            warnings.warn(
                f"RDM colorbar failed for panel ({row_index}, {col_index}): {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            colorbar_axis.set_visible(False)


def _ordered_square(matrix: pd.DataFrame, order: list[str] | None) -> pd.DataFrame:
    if order is None:
        return matrix.copy()
    kept = [
        label for label in order if label in matrix.index and label in matrix.columns
    ]
    return matrix.loc[kept, kept].copy()


def _mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.apply(pd.to_numeric, errors="coerce").copy()
    values = display.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


def _stimulus_display_labels(
    labels: list[str], stimulus_sample_map: pd.DataFrame
) -> list[str]:
    if not {"stimulus", "sample_id"}.issubset(stimulus_sample_map.columns):
        return labels
    sample_map = dict(
        zip(
            stimulus_sample_map["stimulus"].astype(str),
            stimulus_sample_map["sample_id"].fillna("").astype(str),
        )
    )
    return [sample_map.get(label, label) or label for label in labels]


def _class_similarity_matrix(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    categories = sorted(
        set(pairwise["left_category"].astype(str).tolist())
        | set(pairwise["right_category"].astype(str).tolist())
    )
    matrix = pd.DataFrame(np.nan, index=categories, columns=categories, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)
    for row in pairwise.itertuples(index=False):
        left = str(row.left_category)
        right = str(row.right_category)
        value = float(row.chemical_rdm_rsa)
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def _anchor_view_label(view_name: str) -> str:
    labels = {
        "full_trajectory": "t0-44",
        "response_window": "t5-24",
    }
    return labels.get(view_name, view_name)


def _anchor_distance_matrix(view: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    np.fill_diagonal(matrix.values, 0.0)
    for row in view.itertuples(index=False):
        left = str(row.left_label)
        right = str(row.right_label)
        matrix.loc[left, right] = float(row.distance)
        matrix.loc[right, left] = float(row.distance)
    return matrix


def _classical_mds_frame(matrix: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.0
    distances = np.where(np.isfinite(values), values, fill)
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)

    item_count = len(matrix)
    if item_count == 0:
        coords = np.zeros((0, n_components), dtype=float)
    else:
        centering = np.eye(item_count) - np.full(
            (item_count, item_count), 1.0 / item_count
        )
        gram = -0.5 * centering @ (distances**2) @ centering
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        positive = eigvals > 0
        eigvals = eigvals[positive][:n_components]
        eigvecs = eigvecs[:, positive][:, :n_components]
        if eigvals.size == 0:
            coords = np.zeros((item_count, n_components), dtype=float)
        else:
            coords = eigvecs * np.sqrt(eigvals)
            if coords.shape[1] < n_components:
                coords = np.pad(
                    coords,
                    ((0, 0), (0, n_components - coords.shape[1])),
                    constant_values=0.0,
                )
    return pd.DataFrame(
        {"label": matrix.index.astype(str), "x": coords[:, 0], "y": coords[:, 1]}
    )


def _clustered_anchor_order(matrix: pd.DataFrame) -> list[str]:
    values = matrix.to_numpy(float)
    if not np.isfinite(values).all() or len(matrix) < 3:
        return matrix.index.astype(str).tolist()

    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(values, checks=False)
        tree = linkage(condensed, method="average", optimal_ordering=True)
        return matrix.index.to_numpy()[leaves_list(tree)].astype(str).tolist()
    except Exception:
        return _average_linkage_anchor_order(matrix)


def _average_linkage_anchor_order(matrix: pd.DataFrame) -> list[str]:
    labels = matrix.index.astype(str).tolist()
    values = pd.DataFrame(matrix.to_numpy(float), index=labels, columns=labels)
    clusters: list[tuple[str, ...]] = [(label,) for label in labels]

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = np.inf
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left = clusters[left_index]
                right = clusters[right_index]
                block = values.loc[list(left), list(right)].to_numpy(float)
                distance = float(np.nanmean(block))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left_index, right_index)
        if best_pair is None:
            return labels

        left_index, right_index = best_pair
        merged = clusters[left_index] + clusters[right_index]
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: cluster[0])
    return list(clusters[0])


def _short_anchor_label(label: str) -> str:
    parts = label.split("__")
    if len(parts) < 3:
        return label
    return f"{parts[0]} {parts[1]}"


def _prototype_display_label(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    if len(parts) < 3:
        return label
    date, _, stim_name = parts
    return f"{date}\n{stim_name}"


def _label_date(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    return parts[0] if len(parts) >= 1 else label


def _label_stim_name(label: str) -> str:
    parts = label.split("__", maxsplit=2)
    return parts[2] if len(parts) >= 3 else label


def _short_date_label(date: str) -> str:
    if len(date) != 8:
        return date
    return f"{date[4:6]}-{date[6:8]}"


def _anchor_trajectory_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "anchor_trajectory",
        ["#3E5C76", "#98C1D9", "#F8F5F1", "#EEB479", "#B23A48"],
        N=256,
    )


def _clean_stimulus_label(label: str) -> str:
    return label.split(" (", maxsplit=1)[0]


def _ordered_activity_stimulus_labels(summary: pd.DataFrame) -> list[str]:
    meta = summary[["stimulus", "stimulus_label"]].drop_duplicates()
    meta = meta.sort_values("stimulus", kind="stable")
    return meta["stimulus_label"].astype(str).tolist()


def _time_tick_positions(timepoints: list[int], view_name: str) -> list[int]:
    ticks_by_view = {
        "full_trajectory": [0, 6, 16, 26, 36, 44],
        "response_window": [5, 12, 18, 24],
    }
    tick_values = ticks_by_view.get(
        view_name, list(np.linspace(timepoints[0], timepoints[-1], 5, dtype=int))
    )
    return [timepoints.index(value) for value in tick_values if value in timepoints]


def _time_tick_labels(timepoints: list[int], view_name: str) -> list[str]:
    if view_name == "full_trajectory":
        tick_values = [0, 6, 16, 26, 36, 44]
        return [str(value - 6) for value in tick_values if value in timepoints]
    tick_values = {
        "response_window": [5, 12, 18, 24],
    }.get(view_name, list(np.linspace(timepoints[0], timepoints[-1], 5, dtype=int)))
    return [str(value - 6) for value in tick_values if value in timepoints]


def _draw_time_boundaries(axis: plt.Axes, timepoints: list[int]) -> None:
    min_time = min(timepoints)
    max_time = max(timepoints)
    for boundary in (6, 16):
        if min_time < boundary <= max_time:
            axis.axvline(
                boundary - min_time - 0.5,
                color="#2F2F2F",
                linestyle="--",
                linewidth=0.8,
            )


def _annotate_anchor_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    vmax: float,
    decimals: int = 2,
    low_values_dark: bool = False,
) -> None:
    threshold = 0.55 * vmax
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if not np.isfinite(value):
                axis.text(
                    col_index,
                    row_index,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#555555",
                )
                continue
            if low_values_dark:
                text_color = "#F8F8F8" if float(value) <= threshold else "#222222"
            else:
                text_color = "#F8F8F8" if abs(float(value)) >= threshold else "#222222"
            axis.text(
                col_index,
                row_index,
                f"{float(value):.{decimals}f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )


def _profile_cluster_order(matrix: pd.DataFrame) -> list[str]:
    distances = _profile_distance_matrix(matrix)
    if _can_cluster_anchor_matrix(distances):
        return _clustered_anchor_order(distances)
    return matrix.index.astype(str).tolist()


def _profile_distance_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    labels = profiles.index.astype(str).tolist()
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    np.fill_diagonal(matrix.values, 0.0)
    for left, right in combinations(labels, 2):
        distance = _correlation_distance(
            profiles.loc[left].to_numpy(float), profiles.loc[right].to_numpy(float)
        )
        matrix.loc[left, right] = distance
        matrix.loc[right, left] = distance
    return matrix


def _can_cluster_anchor_matrix(matrix: pd.DataFrame) -> bool:
    return len(matrix) >= 3 and bool(np.isfinite(matrix.to_numpy(float)).all())


def _correlation_distance(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    left = left[valid]
    right = right[valid]
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return 1.0 - float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _anchor_matrices_from_pairwise(pairwise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    for view_name, view in pairwise.groupby("view_name", sort=True):
        labels = sorted(
            set(view["left_label"].astype(str)) | set(view["right_label"].astype(str))
        )
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
        np.fill_diagonal(matrix.values, 0.0)
        for row in view.itertuples(index=False):
            matrix.loc[str(row.left_label), str(row.right_label)] = float(row.distance)
            matrix.loc[str(row.right_label), str(row.left_label)] = float(row.distance)
        matrices[str(view_name)] = matrix
    return matrices


def _plot_square_matrix(
    axis: plt.Axes, matrix: pd.DataFrame, *, title: str, cmap: str
) -> object:
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    image = axis.imshow(
        values,
        cmap=cmap,
        vmin=float(np.nanmin(finite)) if finite.size else None,
        vmax=float(np.nanmax(finite)) if finite.size else None,
    )
    axis.set_xticks(
        range(len(matrix.columns)), matrix.columns.astype(str), rotation=90, fontsize=6
    )
    axis.set_yticks(range(len(matrix.index)), matrix.index.astype(str), fontsize=6)
    axis.set_title(title)
    axis.tick_params(length=0)
    return image


def _classical_mds(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.0
    distances = np.where(np.isfinite(values), values, fill)
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)
    if len(distances) == 1:
        return np.zeros((1, 2), dtype=float)
    squared = distances * distances
    centering = np.eye(len(distances)) - np.ones_like(distances) / len(distances)
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    selected = np.clip(eigenvalues[order], 0.0, None)
    coords = eigenvectors[:, order] * np.sqrt(selected)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords


def _anchor_label_metadata(labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        parts = str(label).split("__", 2)
        rows.append(
            {
                "label": label,
                "date": parts[0] if len(parts) > 0 else "",
                "stimulus": parts[1] if len(parts) > 1 else label,
                "stim_name": parts[2] if len(parts) > 2 else label,
            }
        )
    return pd.DataFrame(rows)


def _render_prepared_rdm_axis(
    axis: plt.Axes,
    heatmap_frame: pd.DataFrame | None,
    *,
    fallback_message: str,
    cmap: matplotlib.colors.Colormap,
) -> object | None:
    if heatmap_frame is None or heatmap_frame.empty:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return None

    display_frame = _coerce_square_frame(heatmap_frame)
    norm = _rdm_power_norm(display_frame)
    if norm is None:
        axis.text(0.5, 0.5, fallback_message, ha="center", va="center")
        axis.axis("off")
        return None

    values = display_frame.to_numpy(dtype=float)
    axis.set_axis_on()
    image = axis.imshow(values, cmap=cmap, norm=norm)
    axis.set_xticks(
        np.arange(len(display_frame.columns)),
        display_frame.columns.tolist(),
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(display_frame.index)), display_frame.index.tolist())
    return image


def _rdm_power_norm(
    heatmap_frame: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> matplotlib.colors.PowerNorm | None:
    finite = _finite_off_diagonal_values(heatmap_frame)
    if finite.size == 0:
        return None

    quantiles = np.quantile(finite, [lower_quantile, upper_quantile])
    if np.all(np.isfinite(quantiles)):
        vmin = float(quantiles[0])
        vmax = float(quantiles[1])
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))

    if vmin > vmax:
        vmin, vmax = vmax, vmin
    finite_min = float(np.min(finite))
    finite_max = float(np.max(finite))
    if vmin == vmax:
        if finite_min != finite_max:
            vmin, vmax = finite_min, finite_max
        else:
            padding = max(abs(finite_min) * 0.05, 1e-6)
            vmin = finite_min - padding
            vmax = finite_min + padding
    return matplotlib.colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax, clip=True)


def _finite_off_diagonal_values(heatmap_frame: pd.DataFrame) -> np.ndarray:
    values = _coerce_square_frame(heatmap_frame).to_numpy(dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    finite_mask = np.isfinite(values)
    diagonal_length = min(values.shape)
    if diagonal_length:
        diagonal_indices = np.arange(diagonal_length)
        finite_mask[diagonal_indices, diagonal_indices] = False
    return values[finite_mask]


def _coerce_square_frame(matrix_frame: pd.DataFrame) -> pd.DataFrame:
    if "stimulus_row" in matrix_frame.columns:
        heatmap_frame = matrix_frame.set_index("stimulus_row").copy()
    else:
        heatmap_frame = matrix_frame.copy()
    if heatmap_frame.empty:
        return heatmap_frame
    heatmap_frame.index = pd.Index(heatmap_frame.index.astype(str))
    heatmap_frame.columns = pd.Index(heatmap_frame.columns.astype(str))
    if set(heatmap_frame.index) != set(heatmap_frame.columns):
        raise ValueError("RDM heatmap requires matching row and column labels")
    heatmap_frame = heatmap_frame.reindex(columns=heatmap_frame.index)
    return heatmap_frame.apply(pd.to_numeric, errors="coerce")


__all__ = [
    "create_rdm_panel_figure",
    "finish_figure",
    "plot_anchor_clustered_rdm_heatmaps",
    "plot_anchor_ideal_models",
    "plot_anchor_rdm_heatmaps",
    "plot_anchor_stimulus_date_mds",
    "plot_anchor_stimulus_neuron_activity",
    "plot_anchor_stimulus_neuron_time_heatmaps",
    "plot_anchor_stimulus_same_vs_other_distributions",
    "plot_class_chemical_rdm_similarity",
    "plot_class_vs_full_chemical_similarity",
    "plot_fixed_class_permutation",
    "plot_null_distribution",
    "plot_rdm_heatmap",
    "plot_rdm_heatmap_grid",
    "plot_rdm_heatmap_pair",
    "plot_score_bars",
    "plot_reselection_stability",
    "plot_stimulus_resolved_date_pair_heatmap",
    "plot_summary_scorecard",
    "plot_subset_stability",
    "plot_top_class_rdm_comparison",
    "render_prepared_rdm_panels",
    "write_null_distribution",
    "write_rdm_heatmap_pair",
    "write_subset_stability",
]
