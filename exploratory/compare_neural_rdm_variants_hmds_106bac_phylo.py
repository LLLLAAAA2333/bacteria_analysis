"""106bac neural RDM variants → 2D + 3D HMDS with 16S phylogeny colormap.

Four RDM construction variants:
  1. response_window (t05:t24) + active_scale
  2. response_window (t05:t24), raw (no active_scale)
  3. full_trajectory (t00:t44) + active_scale
  4. full_trajectory (t00:t44), raw (no active_scale)

Each variant → correlation RDM → chord → normalize → HMDS (2D + 3D).
All visualisations use the 16S phylogeny Ward leaf-order colormap.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, leaves_list, linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = str(PROJECT_ROOT / "src")
EXPLORATORY = str(PROJECT_ROOT / "exploratory")
for p in (SRC, EXPLORATORY):
    if p not in sys.path:
        sys.path.insert(0, p)

import compare_86bac_chord_hmds as base

from bacteria_analysis._data_loaders import enrich_neural_dataframe
from bacteria_analysis.features.neural import (
    build_trial_feature_matrix,
    neural_feature_columns,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
ALL_NEURONS = (
    "ADF", "ADL", "ASEL", "ASER", "ASG", "ASH", "ASI", "ASJ", "ASK",
    "AWA", "AWB", "AWCOFF", "AWCON",
)
TREE_PATH = Path("data/16S.aln.trim.fa.treefile")
DEFAULT_OUTPUT_DIR = Path("results/neural_rdm_variants_hmds_106bac_phylo")


# ---------------------------------------------------------------------------
# helpers (replicated from compute_86bac_shape_pca_rsa to avoid importing main)
# ---------------------------------------------------------------------------

def _sample_id_from_stim_name(stim_name: object) -> str:
    parts = str(stim_name).strip().split()
    return parts[0] if parts else ""


def _feature_window_columns(
    columns: list[str], *, window_start: int, window_stop: int
) -> list[str]:
    wanted = {f"t{t:02d}" for t in range(window_start, window_stop)}
    return [c for c in columns if c.rsplit("__", 1)[-1] in wanted]


def _aggregate_features(
    features: pd.DataFrame,
    *,
    group_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group in features.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prototype = np.nanmedian(values, axis=0)
        row = dict(zip(group_columns, group_key, strict=True))
        row["n_trials"] = int(group["trial_id"].nunique())
        row.update(dict(zip(feature_columns, prototype, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_active_scales(
    date_stim: pd.DataFrame,
    *,
    feature_columns: list[str],
    active_threshold: float = 0.2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for neuron in ALL_NEURONS:
        neuron_cols = [c for c in feature_columns if c.startswith(f"{neuron}__")]
        vals = date_stim.loc[:, neuron_cols].to_numpy(dtype=float, copy=False).ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            scale, n_active, method = 1.0, 0, "fallback_no_finite_values"
        else:
            active = finite[np.abs(finite) >= active_threshold]
            n_active = int(active.size)
            if active.size:
                scale = float(np.mean(np.abs(active)))
                method = "mean_abs_active_frames"
            else:
                scale, method = 1.0, "silent_neuron_unit_scale"
        if not np.isfinite(scale) or scale <= 0:
            scale, method = 1.0, "fallback_unit_scale"
        rows.append({
            "neuron": neuron, "active_scale": scale,
            "n_active_frames_for_scale": n_active, "scaling_method": method,
        })
    return pd.DataFrame(rows)


def _apply_active_scale(
    prototypes: pd.DataFrame,
    *,
    feature_columns: list[str],
    scales: pd.DataFrame,
) -> pd.DataFrame:
    scale_map = scales.set_index("neuron")["active_scale"].to_dict()
    scaled = prototypes.copy()
    for col in feature_columns:
        neuron = col.split("__", 1)[0]
        scaled[col] = scaled[col].astype(float) / float(scale_map[neuron])
    return scaled


def _build_correlation_rdm(
    values: pd.DataFrame, *, label_column: str, feature_columns: list[str],
) -> pd.DataFrame:
    labels = values[label_column].astype(str).tolist()
    arr = values.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
    dist = np.full((len(labels), len(labels)), np.nan, dtype=float)
    np.fill_diagonal(dist, 0.0)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            valid = np.isfinite(arr[i]) & np.isfinite(arr[j])
            if valid.sum() < 2:
                continue
            left, right = arr[i, valid], arr[j, valid]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            d = float(np.clip(1.0 - np.corrcoef(left, right)[0, 1], 0.0, 2.0))
            dist[i, j] = dist[j, i] = d
    rdm = pd.DataFrame(dist, index=labels, columns=labels)
    rdm.index.name = label_column
    return rdm


# ---------------------------------------------------------------------------
# 1. Phylogeny colormap
# ---------------------------------------------------------------------------

def build_phylogeny_colormap(
    tree_path: Path, cmap_name: str = "turbo",
) -> tuple[dict[str, str], list[str], pd.DataFrame]:
    """Parse 16S tree → Ward clustering → leaf-order turbo colormap.

    Returns
    -------
    aid_to_color : dict  AID → hex colour
    ordered_aids : list  AIDs in dendrogram leaf order
    colormap_df  : DataFrame with aid, color_hex, leaf_order_position, color_normed
    """
    from Bio import Phylo

    tree = Phylo.read(str(tree_path), "newick")
    terminals = tree.get_terminals()
    leaf_names = [str(t.name) for t in terminals]
    n = len(leaf_names)

    # pairwise cophenetic distances
    tree_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(terminals[i], terminals[j]) or 0.0
            tree_dist[i, j] = tree_dist[j, i] = d
    print(f"16S tree: {n} leaves, cophenetic range [{tree_dist.min():.2f}, {tree_dist.max():.2f}]")

    # Ward clustering
    Z = linkage(squareform(tree_dist), method="ward")
    order = leaves_list(Z)
    coph = squareform(cophenet(Z))
    gaps = np.array([coph[order[i], order[i + 1]] for i in range(n - 1)])
    med = float(np.median(gaps)) or 1.0
    cumdist = np.concatenate([[0.0], np.cumsum(np.arcsinh(gaps / med))])
    normed = (cumdist - cumdist.min()) / (cumdist.max() - cumdist.min() + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    colors_rgba = cmap(normed)
    colors_orig = np.zeros((n, 4))
    for leaf_pos, orig_idx in enumerate(order):
        colors_orig[orig_idx] = colors_rgba[leaf_pos]

    colors_hex = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in colors_orig
    ]

    ordered_aids = [leaf_names[i] for i in order]
    aid_to_color = dict(zip(leaf_names, colors_hex))

    colormap_df = pd.DataFrame({
        "aid": leaf_names,
        "color_hex": colors_hex,
        "leaf_order_position": [int(list(order).index(i)) for i in range(n)],
        "color_normed": [float(normed[list(order).index(i)]) for i in range(n)],
    })

    # intersect with 106bac AIDs will happen later
    return aid_to_color, ordered_aids, colormap_df, Z, leaf_names


def plot_phylogeny_dendrogram(
    Z, leaf_labels: list[str], colors_hex: list[str], output_path: Path,
) -> None:
    """Ward dendrogram with leaf colours."""
    n = len(leaf_labels)
    fig, ax = plt.subplots(figsize=(max(16, n * 0.22), 5.5), constrained_layout=True)
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(
        Z, ax=ax, labels=leaf_labels, leaf_font_size=6,
        color_threshold=0, above_threshold_color="#2c3e50",
        link_color_func=lambda k: "#2c3e50",
    )
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        try:
            idx = list(leaf_labels).index(label_text)
            tick_label.set_color(colors_hex[idx])
        except (ValueError, IndexError):
            pass
    ax.set_title(f"16S Phylogeny Ward dendrogram — {n} strains", fontsize=11)
    ax.set_ylabel("Ward merge cost")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. RDM variant builder
# ---------------------------------------------------------------------------

def _build_variant_rdm(
    trial_features: pd.DataFrame,
    feature_columns: list[str],
    *,
    apply_scale: bool,
    active_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """Build correlation RDM for one variant."""
    sample_prototypes = _aggregate_features(
        trial_features, group_columns=["stim_name"], feature_columns=feature_columns,
    )
    sample_prototypes["sample_id"] = sample_prototypes["stim_name"].map(_sample_id_from_stim_name)

    extra = {"n_features": len(feature_columns)}

    if apply_scale:
        date_stim = _aggregate_features(
            trial_features, group_columns=["date", "stim_name"],
            feature_columns=feature_columns,
        )
        scales = _compute_active_scales(
            date_stim, feature_columns=feature_columns,
            active_threshold=active_threshold,
        )
        extra["active_scales"] = scales.set_index("neuron")["active_scale"].to_dict()
        scaled = _apply_active_scale(
            sample_prototypes, feature_columns=feature_columns, scales=scales,
        )
        rdm = _build_correlation_rdm(scaled, label_column="sample_id", feature_columns=feature_columns)
    else:
        rdm = _build_correlation_rdm(sample_prototypes, label_column="sample_id", feature_columns=feature_columns)

    valid = rdm.notna().all(axis=1)
    rdm = rdm.loc[valid, valid]
    extra["n_samples_in_rdm"] = int(valid.sum())
    extra["n_nan_samples"] = int((~valid).sum())
    return rdm, extra


# ---------------------------------------------------------------------------
# 3. HMDS pipeline (2D + 3D)
# ---------------------------------------------------------------------------

def _run_hmds(
    distance: np.ndarray,
    *,
    dim: int,
    method: str,
    input_distance_name: str,
    analysis_label: str,
    seed: int,
    hmds_starts: int,
    hmds_maxiter: int,
) -> base.EmbeddingResult:
    """Fit HMDS at given dimension."""
    coords_lorentz, embedded_hyp, lambda_value, metadata = base.scipy_hyperbolic_mds(
        distance, dim=dim, starts=hmds_starts, maxiter=hmds_maxiter, seed=seed,
    )
    poincare = base.recenter_poincare(base.lorentz_to_poincare(coords_lorentz))
    predicted = embedded_hyp / lambda_value
    n_params = distance.shape[0] * dim + 1 - dim * (dim - 1) / 2
    metrics = base.preservation_metrics(distance, embedded_hyp, predicted, n_params=n_params)

    radius = np.linalg.norm(poincare, axis=1)
    hyp_radius = 2.0 * np.arctanh(np.clip(radius, 0.0, 0.999999))
    metrics.update({
        "lambda": float(lambda_value),
        "dim": dim,
        "analysis_label": analysis_label,
        "poincare_radius_min": float(radius.min()),
        "poincare_radius_median": float(np.median(radius)),
        "poincare_radius_mean": float(radius.mean()),
        "poincare_radius_q90": float(np.quantile(radius, 0.90)),
        "poincare_radius_q95": float(np.quantile(radius, 0.95)),
        "poincare_radius_max": float(radius.max()),
        "hyperbolic_radius_median": float(np.median(hyp_radius)),
        "fraction_radius_gt_0_90": float(np.mean(radius > 0.90)),
        "fraction_radius_gt_0_95": float(np.mean(radius > 0.95)),
        **metadata,
    })
    return base.EmbeddingResult(
        method=f"{method}_dim{dim}",
        input_distance_name=input_distance_name,
        coordinates=poincare,
        embedded_distance=embedded_hyp,
        predicted_distance=predicted,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 4. Plot functions
# ---------------------------------------------------------------------------

def _format_metrics_line(metrics: dict, dim: int) -> str:
    return (
        f"ρ={metrics['distance_spearman']:.3f}  "
        f"stress={metrics['normalized_raw_stress']:.3f}  "
        f"λ={metrics['lambda']:.3f}"
    )


# --- 2D Poincaré disk ---

def plot_2d_disk_png(
    coords: np.ndarray,
    sample_ids: list[str],
    colors_hex: list[str],
    metrics: dict,
    title_label: str,
    output_path: Path,
) -> None:
    """Static 2D Poincaré disk with AID labels."""
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, lw=1)
    ax.add_artist(circle)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors_hex, s=60, alpha=0.90, edgecolors="white", linewidths=0.6,
    )
    for i, label in enumerate(sample_ids):
        ax.annotate(
            str(label), (coords[i, 0], coords[i, 1]),
            fontsize=4, alpha=0.55, ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 3),
        )
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("Poincaré 1", fontsize=9)
    ax.set_ylabel("Poincaré 2", fontsize=9)
    ax.grid(True, color="#e5e7eb", linewidth=0.4, alpha=0.5)
    ax.set_title(
        f"{title_label}  2D HMDS — 16S Phylogeny colormap\n{_format_metrics_line(metrics, 2)}",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_2d_disk_html(
    coords: np.ndarray,
    sample_ids: list[str],
    colors_hex: list[str],
    metrics: dict,
    title_label: str,
    output_path: Path,
) -> None:
    """Interactive 2D Poincaré disk (plotly)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers+text",
        marker={"size": 8, "color": colors_hex, "opacity": 0.92,
                "line": {"color": "white", "width": 0.5}},
        text=[str(s) for s in sample_ids],
        textposition="top center",
        textfont={"size": 7, "color": "#333333"},
        hovertext=[f"<b>{s}</b>" for s in sample_ids],
        hoverinfo="text",
        showlegend=False,
    ))
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode="lines",
        line={"color": "#6b7280", "width": 0.8},
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        title={
            "text": f"{title_label}<br><sup>{_format_metrics_line(metrics, 2)}</sup>",
            "font": {"size": 14},
        },
        xaxis={"range": [-1.08, 1.08], "constrain": "domain", "title": "Poincaré 1"},
        yaxis={"range": [-1.08, 1.08], "scaleanchor": "x", "scaleratio": 1,
               "title": "Poincaré 2"},
        width=950, height=850,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


# --- 3D Poincaré ball ---

def plot_3d_ball_html(
    coords: np.ndarray,
    sample_ids: list[str],
    colors_hex: list[str],
    metrics: dict,
    title_label: str,
    output_path: Path,
) -> None:
    """Interactive 3D Poincaré ball (plotly)."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # points
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        marker={"size": 5, "color": colors_hex, "opacity": 0.90,
                "line": {"color": "white", "width": 0.4}},
        text=[str(s) for s in sample_ids],
        textposition="top center",
        textfont={"size": 7, "color": "#444444"},
        hovertext=[f"<b>{s}</b><br>r={np.linalg.norm(coords[i]):.4f}"
                   for i, s in enumerate(sample_ids)],
        hoverinfo="text",
        showlegend=False,
    ))

    # wireframe sphere (3 great circles)
    theta = np.linspace(0, 2 * np.pi, 120)
    zero = np.zeros_like(theta)
    for plane_label, xs, ys, zs, color in [
        ("xy", np.cos(theta), np.sin(theta), zero, "#6b7280"),
        ("xz", np.cos(theta), zero, np.sin(theta), "#9ca3af"),
        ("yz", zero, np.cos(theta), np.sin(theta), "#9ca3af"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line={"color": color, "width": 0.8},
            hoverinfo="skip", showlegend=False,
            name=f"sphere_{plane_label}",
        ))

    fig.update_layout(
        title={
            "text": f"{title_label}<br><sup>{_format_metrics_line(metrics, 3)}</sup>",
            "font": {"size": 14},
        },
        scene={
            "xaxis": {"title": "P1", "range": [-1.05, 1.05]},
            "yaxis": {"title": "P2", "range": [-1.05, 1.05]},
            "zaxis": {"title": "P3", "range": [-1.05, 1.05]},
            "aspectmode": "cube",
        },
        width=900, height=850,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


# --- Shepard ---

def plot_shepard(
    result: base.EmbeddingResult,
    original: np.ndarray,
    output_path: Path,
    dim: int,
) -> None:
    """Shepard diagram: input vs embedded predicted distances."""
    orig = base.upper_triangle(original)
    pred = base.upper_triangle(result.predicted_distance)
    limit = float(max(orig.max(), pred.max()) * 1.04)
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.scatter(orig, pred, s=5, color="#4a5568", alpha=0.18, linewidths=0)
    ax.plot([0, limit], [0, limit], "--", color="#111827", lw=0.8)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Input chord distance", fontsize=9)
    ax.set_ylabel("Embedded predicted distance", fontsize=9)
    ax.set_title(
        f"Shepard  ({dim}D HMDS)\n{_format_metrics_line(result.metrics, dim)}",
        fontsize=10,
    )
    ax.grid(True, color="#e5e7eb", lw=0.4)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Per-variant orchestrator
# ---------------------------------------------------------------------------

def _run_one_variant(
    *,
    variant_id: str,
    variant_label: str,
    feature_columns: list[str],
    apply_scale: bool,
    trial_features: pd.DataFrame,
    aid_to_color: dict[str, str],
    output_dir: Path,
    active_threshold: float,
    seed: int,
    hmds_starts: int,
    hmds_maxiter: int,
) -> dict:
    """Full pipeline for one variant: RDM → 2D HMDS → 3D HMDS → outputs."""
    print(f"\n{'=' * 60}")
    print(f"  {variant_label}")
    print(f"  features={len(feature_columns)}, active_scale={apply_scale}")
    print(f"{'=' * 60}")

    var_dir = output_dir / variant_id
    fig_dir = var_dir / "figures"
    tbl_dir = var_dir / "tables"
    for d in (fig_dir, tbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- build RDM ---
    rdm, rdm_meta = _build_variant_rdm(
        trial_features, feature_columns,
        apply_scale=apply_scale, active_threshold=active_threshold,
    )
    print(f"  RDM: {rdm.shape[0]} samples, {rdm_meta['n_features']} features")
    if "active_scales" in rdm_meta:
        s = rdm_meta["active_scales"]
        print(f"  active_scales: [{min(s.values()):.4f}, {max(s.values()):.4f}]")

    # --- intersect with phylogeny AIDs ---
    rdm_aids = set(rdm.index.astype(str))
    common_aids = sorted(rdm_aids & set(aid_to_color.keys()))
    if len(common_aids) < 4:
        msg = f"Only {len(common_aids)} AIDs intersect phylogeny; need ≥4"
        print(f"  SKIP: {msg}")
        return {"variant_id": variant_id, "status": "skipped", "reason": msg}

    rdm = rdm.loc[common_aids, common_aids]
    colors = [aid_to_color[a] for a in common_aids]
    print(f"  Intersection with phylogeny: {len(common_aids)} AIDs")

    # --- chord + normalize ---
    linear = base.clean_distance_matrix(rdm)
    chord = base.chord_from_linear(linear)
    distance = base.normalize_to_max_two(chord)
    pd.DataFrame(distance, index=common_aids, columns=common_aids).to_csv(
        tbl_dir / "distance__chord_normalized.csv"
    )
    rdm.to_csv(tbl_dir / "neural_correlation_rdm.csv")
    with open(tbl_dir / "rdm_build_metadata.json", "w") as fh:
        json.dump({k: v for k, v in rdm_meta.items() if k != "active_scales"},
                  fh, indent=2, default=str)

    results_2d, results_3d = None, None

    # --- 2D HMDS ---
    try:
        result_2d = _run_hmds(
            distance, dim=2,
            method=f"{variant_id}_chord_hmds",
            input_distance_name=f"{variant_id}_chord_normalized",
            analysis_label=variant_id,
            seed=seed, hmds_starts=hmds_starts, hmds_maxiter=hmds_maxiter,
        )
        print(f"  2D: stress={result_2d.metrics['normalized_raw_stress']:.4f}  "
              f"ρ={result_2d.metrics['distance_spearman']:.4f}  "
              f"λ={result_2d.metrics['lambda']:.3f}  "
              f"r_q90={result_2d.metrics['poincare_radius_q90']:.3f}")

        # save
        pd.DataFrame(result_2d.coordinates, index=common_aids,
                     columns=["poincare1", "poincare2"]).to_csv(tbl_dir / "coordinates_2d.csv")
        base.save_pair_distances(result_2d, distance, common_aids, tbl_dir / "pair_distances_2d.csv")
        with open(tbl_dir / "metrics_2d.json", "w") as fh:
            json.dump(result_2d.metrics, fh, indent=2, default=str)

        prefix = f"106bac {variant_label}"
        plot_shepard(result_2d, distance, fig_dir / "shepard_2d.png", dim=2)
        plot_2d_disk_png(result_2d.coordinates, common_aids, colors,
                         result_2d.metrics, prefix, fig_dir / "poincare_disk_2d.png")
        plot_2d_disk_html(result_2d.coordinates, common_aids, colors,
                          result_2d.metrics, prefix, fig_dir / "poincare_disk_2d.html")
        results_2d = result_2d
    except Exception:
        print(f"  2D HMDS FAILED:\n{traceback.format_exc()}")

    # --- 3D HMDS ---
    try:
        result_3d = _run_hmds(
            distance, dim=3,
            method=f"{variant_id}_chord_hmds",
            input_distance_name=f"{variant_id}_chord_normalized",
            analysis_label=variant_id,
            seed=seed, hmds_starts=hmds_starts, hmds_maxiter=hmds_maxiter,
        )
        print(f"  3D: stress={result_3d.metrics['normalized_raw_stress']:.4f}  "
              f"ρ={result_3d.metrics['distance_spearman']:.4f}  "
              f"λ={result_3d.metrics['lambda']:.3f}  "
              f"r_q90={result_3d.metrics['poincare_radius_q90']:.3f}")

        # save
        pd.DataFrame(result_3d.coordinates, index=common_aids,
                     columns=["poincare1", "poincare2", "poincare3"]).to_csv(
            tbl_dir / "coordinates_3d.csv")
        base.save_pair_distances(result_3d, distance, common_aids, tbl_dir / "pair_distances_3d.csv")
        with open(tbl_dir / "metrics_3d.json", "w") as fh:
            json.dump(result_3d.metrics, fh, indent=2, default=str)

        prefix = f"106bac {variant_label}"
        plot_shepard(result_3d, distance, fig_dir / "shepard_3d.png", dim=3)
        plot_3d_ball_html(result_3d.coordinates, common_aids, colors,
                          result_3d.metrics, prefix, fig_dir / "poincare_ball_3d.html")
        results_3d = result_3d
    except Exception:
        print(f"  3D HMDS FAILED:\n{traceback.format_exc()}")

    print(f"  Done → {var_dir}")
    return {
        "variant_id": variant_id,
        "status": "ok",
        "n_samples": len(common_aids),
        "n_features": rdm_meta["n_features"],
        "metrics_2d": results_2d.metrics if results_2d else None,
        "metrics_3d": results_3d.metrics if results_3d else None,
    }


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="106bac neural RDM variants → 2D+3D HMDS, phylogeny colormap"
    )
    parser.add_argument("--neural-parquet", type=Path, default=Path("data/106bac.parquet"))
    parser.add_argument("--tree-path", type=Path, default=TREE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-start", type=int, default=5)
    parser.add_argument("--window-stop", type=int, default=25)
    parser.add_argument("--active-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--hmds-starts", type=int, default=8)
    parser.add_argument("--hmds-maxiter", type=int, default=900)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Phylogeny colormap ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Building 16S phylogeny colormap ...")
    print("=" * 60)
    aid_to_color, ordered_aids, colormap_df, Z, tree_leaves = build_phylogeny_colormap(
        args.tree_path,
    )
    colormap_df.to_csv(output_dir / "phylogeny_colormap.csv", index=False)
    print(f"  Colormap: {len(aid_to_color)} tree leaves → "
          f"{output_dir / 'phylogeny_colormap.csv'}")

    # ── 2. Load neural data ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Loading 106bac neural data ...")
    print("=" * 60)
    raw_neural = pd.read_parquet(args.neural_parquet)
    trial_features = build_trial_feature_matrix(
        raw_neural, view="full_trajectory", merge_lr=True,
    )
    all_feature_columns = neural_feature_columns(trial_features)
    response_window_cols = _feature_window_columns(
        all_feature_columns,
        window_start=args.window_start, window_stop=args.window_stop,
    )
    print(f"  trial_features: {trial_features.shape[0]} trials, "
          f"{len(all_feature_columns)} columns")
    print(f"  response_window (t{args.window_start:02d}:t{args.window_stop:02d}): "
          f"{len(response_window_cols)} columns")
    print(f"  full_trajectory: {len(all_feature_columns)} columns")

    # ── 3. Build RDM sample_id → AID preview ───────────────────────────
    sample_ids_in_data = set()
    for sn in trial_features["stim_name"].dropna().unique():
        sample_ids_in_data.add(_sample_id_from_stim_name(sn))
    tree_aids = set(aid_to_color.keys())
    common_preview = sample_ids_in_data & tree_aids
    print(f"  Data AIDs ∩ Tree leaves: {len(common_preview)} strains")
    if len(common_preview) < 10:
        print(f"  WARNING: very few shared AIDs! Data: {len(sample_ids_in_data)}, "
              f"Tree: {len(tree_aids)}")

    # ── 4. Dendrogram for all tree leaves ──────────────────────────────
    print("\nPlotting phylogeny dendrogram ...")
    # We don't have genus/species for all 299 tree leaves
    leaf_labels_full = list(tree_leaves)
    colors_full = [aid_to_color.get(n, "#999999") for n in leaf_labels_full]
    plot_phylogeny_dendrogram(Z, leaf_labels_full, colors_full,
                              output_dir / "phylogeny_dendrogram.png")
    print(f"  → phylogeny_dendrogram.png")

    # ── 5. Run 4 variants ──────────────────────────────────────────────
    variants: list[dict] = [
        {
            "variant_id": "response_window_active",
            "variant_label": "Response Window + Active Scale  (RW+AS)",
            "feature_columns": response_window_cols,
            "apply_scale": True,
        },
        {
            "variant_id": "response_window_raw",
            "variant_label": "Response Window  Raw  (RW raw)",
            "feature_columns": response_window_cols,
            "apply_scale": False,
        },
        {
            "variant_id": "full_trajectory_active",
            "variant_label": "Full Trajectory + Active Scale  (FT+AS)",
            "feature_columns": all_feature_columns,
            "apply_scale": True,
        },
        {
            "variant_id": "full_trajectory_raw",
            "variant_label": "Full Trajectory  Raw  (FT raw)",
            "feature_columns": all_feature_columns,
            "apply_scale": False,
        },
    ]

    all_results = []
    for v in variants:
        result = _run_one_variant(
            trial_features=trial_features,
            aid_to_color=aid_to_color,
            output_dir=output_dir,
            active_threshold=args.active_threshold,
            seed=args.seed,
            hmds_starts=args.hmds_starts,
            hmds_maxiter=args.hmds_maxiter,
            **{k: v[k] for k in [
                "variant_id", "variant_label", "feature_columns", "apply_scale",
            ]},
        )
        all_results.append(result)

    # ── 6. Summary ─────────────────────────────────────────────────────
    summary = {
        "parameters": {
            "neural_parquet": str(args.neural_parquet),
            "tree_path": str(args.tree_path),
            "window_start": args.window_start,
            "window_stop": args.window_stop,
            "active_threshold": args.active_threshold,
            "response_window_n_features": len(response_window_cols),
            "full_trajectory_n_features": len(all_feature_columns),
            "seed": args.seed,
            "hmds_starts": args.hmds_starts,
            "hmds_maxiter": args.hmds_maxiter,
        },
        "variants": [],
    }
    for r in all_results:
        entry = {"variant_id": r["variant_id"], "status": r["status"]}
        for dim_tag, m in [("2d", r.get("metrics_2d")), ("3d", r.get("metrics_3d"))]:
            if m is None:
                continue
            entry[f"{dim_tag}_stress"] = m.get("normalized_raw_stress")
            entry[f"{dim_tag}_spearman_rho"] = m.get("distance_spearman")
            entry[f"{dim_tag}_lambda"] = m.get("lambda")
            entry[f"{dim_tag}_radius_q90"] = m.get("poincare_radius_q90")
            entry[f"{dim_tag}_fraction_gt_0_90"] = m.get("fraction_radius_gt_0_90")
            entry[f"{dim_tag}_backend"] = m.get("backend")
        if "reason" in r:
            entry["reason"] = r["reason"]
        summary["variants"].append(entry)

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"Summary → {summary_path}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
