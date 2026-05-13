"""Compact plotting primitives for function-first analyses."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

plt.rcParams["figure.max_open_warning"] = 0


def finish_figure(figure: Figure, output_path: str | Path | None, *, dpi: int = 150) -> Figure | None:
    """Save a figure when a path is provided, otherwise return it for display."""

    if output_path is None:
        return figure
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return None


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


def write_subset_stability(
    subset_results: pd.DataFrame,
    output_path: str | Path | None,
    *,
    title: str = "Subset stability",
    dpi: int = 150,
) -> Figure | None:
    """Render stimulus-subset RSA stability and optionally save it."""

    return finish_figure(plot_subset_stability(subset_results, title=title), output_path, dpi=dpi)


def plot_fixed_class_permutation(
    summary: pd.DataFrame,
    output_path: str | Path | None,
    *,
    class_limit: int,
    dpi: int = 150,
) -> Figure | None:
    """Render observed class RSA against fixed-class permutation bands."""

    plot_frame = summary.sort_values("observed_rsa", ascending=False).head(class_limit).iloc[::-1].copy()
    figure, axis = plt.subplots(figsize=(7.0, max(3.2, 0.28 * len(plot_frame) + 1.4)), constrained_layout=True)
    if not plot_frame.empty:
        y = np.arange(len(plot_frame))
        axis.hlines(y, plot_frame["null_q95"], plot_frame["null_q99"], color="#8a8a8a", linewidth=2.0)
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

    merged = stability.merge(observed.loc[:, ["model_id", "response_window_rsa"]], on="model_id", how="left")
    plot_frame = merged.sort_values(["top3_frequency", "response_window_rsa"], ascending=False)
    plot_frame = plot_frame.head(class_limit).iloc[::-1].copy()
    figure, axis = plt.subplots(figsize=(7.2, max(3.2, 0.32 * len(plot_frame) + 1.5)), constrained_layout=True)
    if not plot_frame.empty:
        y = np.arange(len(plot_frame))
        axis.barh(y - 0.18, plot_frame["top1_frequency"], height=0.16, color="#4c78a8", label="top 1")
        axis.barh(y, plot_frame["top3_frequency"], height=0.16, color="#f58518", label="top 3")
        axis.barh(y + 0.18, plot_frame["top5_frequency"], height=0.16, color="#54a24b", label="top 5")
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

    order = [label for label in labels if label in primary_neural.index and label in primary_neural.columns]
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
        axis.set_xticks(range(len(display_labels)), display_labels, rotation=90, fontsize=6)
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
    figure, axis = plt.subplots(figsize=(7.8, max(3.0, 0.34 * len(plot_frame) + 1.6)), constrained_layout=True)
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
    axis.set_yticks(np.arange(len(plot_frame)), plot_frame["category"].astype(str).tolist())
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
            axis.text(col, row, text, ha="center", va="center", fontsize=7, color="white")
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
    image = axes[0].imshow(values, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
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

    plot_frame = similarity.sort_values("class_vs_full_chemical_rdm_rsa", ascending=True).copy()
    figure, axis = plt.subplots(figsize=(7.2, max(3.2, 0.28 * len(plot_frame) + 1.4)), constrained_layout=True)
    if not plot_frame.empty:
        y = np.arange(len(plot_frame))
        values = plot_frame["class_vs_full_chemical_rdm_rsa"].to_numpy(float)
        axis.barh(y, values, color="#4c78a8")
        axis.set_yticks(y, plot_frame["category"].astype(str).tolist())
        median = float(np.nanmedian(values))
        axis.axvline(median, color="#222222", linestyle="--", linewidth=1.0, label=f"median = {median:.2f}")
        axis.legend(fontsize=8)
    axis.set_xlabel("RSA vs full chemical RDM")
    axis.set_title("Class similarity to full chemical RDM")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_rdm_heatmaps(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    dpi: int = 150,
) -> Figure | None:
    """Render one prototype-distance RDM per anchor view."""

    matrices = _anchor_matrices_from_pairwise(pairwise)
    count = max(1, len(matrices))
    figure, axes = plt.subplots(1, count, figsize=(4.0 * count, 3.8), constrained_layout=True)
    if count == 1:
        axes = [axes]
    for axis, (view_name, matrix) in zip(axes, matrices.items(), strict=False):
        image = _plot_square_matrix(axis, matrix, title=str(view_name), cmap="viridis")
        figure.colorbar(image, ax=axis, shrink=0.78, label="distance")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_clustered_rdm_heatmaps(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    dpi: int = 150,
) -> Figure | None:
    """Render the requested anchor prototype RDM with stable label ordering."""

    matrices = _anchor_matrices_from_pairwise(pairwise)
    matrix = matrices.get(view_name, next(iter(matrices.values()), pd.DataFrame()))
    figure, axis = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    if matrix.empty:
        axis.text(0.5, 0.5, "No anchor distances", ha="center", va="center")
        axis.axis("off")
    else:
        image = _plot_square_matrix(axis, matrix, title=f"{view_name} anchor RDM", cmap="viridis")
        figure.colorbar(image, ax=axis, shrink=0.82, label="distance")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_date_mds(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    date_order: list[str] | None = None,
    dpi: int = 150,
) -> Figure | None:
    """Render a compact two-dimensional view of anchor date prototypes."""

    matrices = _anchor_matrices_from_pairwise(pairwise)
    matrix = matrices.get(view_name, pd.DataFrame())
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    if matrix.empty:
        axis.text(0.5, 0.5, "No anchor distances", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    coords = _classical_mds(matrix)
    meta = _anchor_label_metadata(matrix.index.astype(str).tolist())
    dates = date_order or sorted(meta["date"].astype(str).unique())
    palette = {date: plt.get_cmap("tab10")(index % 10) for index, date in enumerate(dates)}
    for row_index, row in meta.iterrows():
        label = str(row["label"])
        axis.scatter(coords[row_index, 0], coords[row_index, 1], color=palette.get(str(row["date"]), "#4c78a8"), s=48)
        axis.text(coords[row_index, 0], coords[row_index, 1], str(row["stim_name"]), fontsize=8, ha="left", va="bottom")
    axis.axhline(0, color="#d0d0d0", linewidth=0.8)
    axis.axvline(0, color="#d0d0d0", linewidth=0.8)
    axis.set_title(f"{view_name} anchor prototype map")
    axis.set_xlabel("MDS 1")
    axis.set_ylabel("MDS 2")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_stimulus_resolved_date_pair_heatmap(
    pairwise: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    date_order: list[str] | None = None,
    dpi: int = 150,
) -> Figure | None:
    """Render same-anchor cross-date distances by stimulus and date pair."""

    view = pairwise.loc[
        pairwise["view_name"].astype(str).eq(view_name)
        & pairwise["pair_category"].astype(str).eq("different_date_same_stimulus")
    ].copy()
    figure, axis = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    if view.empty:
        axis.text(0.5, 0.5, "No cross-date anchors", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    view["date_pair"] = view["left_date"].astype(str) + "|" + view["right_date"].astype(str)
    if date_order:
        ordered_pairs = [
            f"{left}|{right}"
            for index, left in enumerate(date_order)
            for right in date_order[index + 1 :]
            if f"{left}|{right}" in set(view["date_pair"])
        ]
    else:
        ordered_pairs = sorted(view["date_pair"].astype(str).unique())
    pivot = view.pivot_table(
        index="left_stim_name",
        columns="date_pair",
        values="distance",
        aggfunc="median",
    ).reindex(columns=ordered_pairs)
    image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index.astype(str).tolist())
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns.astype(str).tolist(), rotation=45, ha="right")
    axis.set_title(f"{view_name} cross-date anchor distances")
    figure.colorbar(image, ax=axis, pad=0.02, label="distance")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_neuron_activity(
    matrix: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    dpi: int = 150,
) -> Figure | None:
    """Render median neuron activity by anchor stimulus."""

    figure, axis = plt.subplots(figsize=(5.8, max(2.8, 0.32 * len(matrix) + 1.4)), constrained_layout=True)
    if matrix.empty:
        axis.text(0.5, 0.5, "No activity matrix", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index.astype(str).tolist())
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns.astype(str).tolist(), rotation=45, ha="right")
    axis.set_title(f"{view_name} anchor neuron activity")
    figure.colorbar(image, ax=axis, pad=0.02, label="baseline-centered dF/F0")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_neuron_time_heatmaps(
    summary: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    aggregator_label: str,
    dpi: int = 150,
) -> Figure | None:
    """Render neuron-by-time activity heatmaps for each anchor stimulus."""

    stimuli = summary["stimulus_label"].astype(str).drop_duplicates().tolist() if not summary.empty else []
    count = max(1, len(stimuli))
    figure, axes = plt.subplots(1, count, figsize=(4.2 * count, 3.4), constrained_layout=True)
    if count == 1:
        axes = [axes]
    if summary.empty:
        axes[0].text(0.5, 0.5, "No trajectory summary", ha="center", va="center")
        axes[0].axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    finite = pd.to_numeric(summary["activity_value"], errors="coerce").dropna().to_numpy(float)
    limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    image = None
    for axis, stimulus in zip(axes, stimuli, strict=False):
        part = summary.loc[summary["stimulus_label"].astype(str).eq(stimulus)].copy()
        pivot = part.pivot_table(index="neuron", columns="time_point", values="activity_value", aggfunc="median")
        image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(stimulus)
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index.astype(str).tolist(), fontsize=7)
        axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns.astype(str).tolist(), rotation=90, fontsize=6)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="activity")
    figure.suptitle(f"{view_name} anchor trajectories ({aggregator_label})")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_stimulus_same_vs_other_distributions(
    contrasts: pd.DataFrame,
    output_path: str | Path | None,
    *,
    view_name: str,
    dpi: int = 150,
) -> Figure | None:
    """Render same-anchor versus other-anchor cross-date distances."""

    view = contrasts.loc[contrasts["view_name"].astype(str).eq(view_name)].copy()
    figure, axis = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    if view.empty:
        axis.text(0.5, 0.5, "No contrasts", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    groups = [
        pd.to_numeric(view.loc[view["contrast"].eq(label), "distance"], errors="coerce").dropna().to_numpy(float)
        for label in ("same", "different")
    ]
    axis.boxplot(groups, tick_labels=["same", "different"], showfliers=True)
    axis.set_ylabel("distance")
    axis.set_title(f"{view_name} anchor contrast")
    return finish_figure(figure, output_path, dpi=dpi)


def plot_anchor_ideal_models(
    summary: pd.DataFrame,
    output_path: str | Path | None,
    *,
    dpi: int = 150,
) -> Figure | None:
    """Render anchor distance alignment with stimulus/date ideal models."""

    figure, axis = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    if summary.empty:
        axis.text(0.5, 0.5, "No ideal-model summary", ha="center", va="center")
        axis.axis("off")
        return finish_figure(figure, output_path, dpi=dpi)

    x = np.arange(len(summary))
    width = 0.35
    axis.bar(x - width / 2, summary["stimulus_ideal_spearman"], width=width, label="stimulus")
    axis.bar(x + width / 2, summary["date_ideal_spearman"], width=width, label="date")
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_xticks(x, summary["view_name"].astype(str).tolist(), rotation=20, ha="right")
    axis.set_ylabel("Spearman r")
    axis.set_title("Anchor ideal-model similarity")
    axis.legend(fontsize=8)
    return finish_figure(figure, output_path, dpi=dpi)


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
    kept = [label for label in order if label in matrix.index and label in matrix.columns]
    return matrix.loc[kept, kept].copy()


def _mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.apply(pd.to_numeric, errors="coerce").copy()
    values = display.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


def _stimulus_display_labels(labels: list[str], stimulus_sample_map: pd.DataFrame) -> list[str]:
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


def _anchor_matrices_from_pairwise(pairwise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    for view_name, view in pairwise.groupby("view_name", sort=True):
        labels = sorted(set(view["left_label"].astype(str)) | set(view["right_label"].astype(str)))
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
        np.fill_diagonal(matrix.values, 0.0)
        for row in view.itertuples(index=False):
            matrix.loc[str(row.left_label), str(row.right_label)] = float(row.distance)
            matrix.loc[str(row.right_label), str(row.left_label)] = float(row.distance)
        matrices[str(view_name)] = matrix
    return matrices


def _plot_square_matrix(axis: plt.Axes, matrix: pd.DataFrame, *, title: str, cmap: str) -> object:
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    image = axis.imshow(
        values,
        cmap=cmap,
        vmin=float(np.nanmin(finite)) if finite.size else None,
        vmax=float(np.nanmax(finite)) if finite.size else None,
    )
    axis.set_xticks(range(len(matrix.columns)), matrix.columns.astype(str), rotation=90, fontsize=6)
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
    axis.set_xticks(np.arange(len(display_frame.columns)), display_frame.columns.tolist(), rotation=45, ha="right")
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
