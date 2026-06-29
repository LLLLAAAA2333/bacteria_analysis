from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
DEFAULT_BAYESIAN_HMDS = Path("H:/Process_temporary/WJH/BayesianHMDS")
NEURAL_RDM_CANDIDATES = (
    "neural_shape_rdm__active_scaled_flattened_correlation.csv",
    "neural_correlation_distance_matrix.csv",
)
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"
SEED = 20260604
AID_COLOR_MIN = 1
AID_COLOR_MAX = 300
AID_COLORBAR_TICKS = [1, 50, 100, 150, 200, 250, 300]


@dataclass
class EmbeddingResult:
    method: str
    input_distance_name: str
    coordinates: np.ndarray
    embedded_distance: np.ndarray
    predicted_distance: np.ndarray
    metrics: dict[str, float | int | str | None]


def aid_sort_key(aid: str) -> tuple[str, int, str]:
    text = str(aid)
    prefix = "".join(char for char in text if not char.isdigit())
    digits = "".join(char for char in text if char.isdigit())
    numeric = int(digits) if digits else -1
    return prefix, numeric, text


def aid_number(aid: str) -> int:
    return aid_sort_key(aid)[1]


def aid_label(number: int) -> str:
    return f"A{int(number):03d}"


def color_to_hex(color: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(color[0] * 255)),
        int(round(color[1] * 255)),
        int(round(color[2] * 255)),
    )


def genus_color_table(taxonomy: pd.DataFrame) -> pd.DataFrame:
    genus_counts = taxonomy["genus"].value_counts()
    aid_ordered = taxonomy.copy()
    aid_ordered["_aid_sort_key"] = aid_ordered["AID"].map(aid_sort_key)
    aid_ordered = aid_ordered.sort_values("_aid_sort_key")
    labels = list(dict.fromkeys(aid_ordered["genus"].astype(str)))
    first_aid = aid_ordered.groupby("genus", sort=False)["AID"].first()
    first_aids = first_aid.loc[labels].to_numpy(str)
    first_aid_numbers = np.asarray([aid_number(aid) for aid in first_aids], dtype=float)
    color_norm = plt.Normalize(vmin=AID_COLOR_MIN, vmax=AID_COLOR_MAX)
    color_positions = np.clip(color_norm(first_aid_numbers), 0.0, 1.0)
    colors = plt.cm.turbo(color_positions)
    return pd.DataFrame(
        {
            "genus": labels,
            "genus_index": np.arange(len(labels)),
            "first_AID_in_color_order": first_aids,
            "first_AID_number": first_aid_numbers.astype(int),
            "first_AID_color_position": color_positions,
            "n_strains": genus_counts.loc[labels].to_numpy(int),
            "color_hex": [color_to_hex(color) for color in colors],
        }
    )


def species_color_table(taxonomy: pd.DataFrame) -> pd.DataFrame:
    ordered = taxonomy.copy()
    ordered["_aid_sort_key"] = ordered["AID"].map(aid_sort_key)
    ordered = ordered.sort_values("_aid_sort_key")
    labels = list(dict.fromkeys(ordered["species"].astype(str)))
    first_aid = ordered.groupby("species", sort=False)["AID"].first()
    genus = ordered.groupby("species", sort=False)["genus"].first()
    counts = ordered["species"].value_counts()
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(labels)))
    return pd.DataFrame(
        {
            "species": labels,
            "species_index": np.arange(len(labels)),
            "genus": genus.loc[labels].to_numpy(str),
            "first_AID": first_aid.loc[labels].to_numpy(str),
            "n_strains": counts.loc[labels].to_numpy(int),
            "color_hex": [color_to_hex(color) for color in colors],
        }
    )


def _detect_color_mode(color_table: pd.DataFrame) -> dict:
    if "species_index" in color_table.columns and "species" in color_table.columns:
        n = len(color_table)
        return {
            "merge_key": "species",
            "color_column": "species_index",
            "norm_vmin": -0.5,
            "norm_vmax": max(n - 0.5, 0.5),
            "mode": "species",
            "merge_columns": ["species", "species_index", "color_hex", "first_AID"],
            "n_colors": n,
        }
    if "first_AID_number" in color_table.columns and "genus" in color_table.columns:
        return {
            "merge_key": "genus",
            "color_column": "first_AID_number",
            "norm_vmin": AID_COLOR_MIN,
            "norm_vmax": AID_COLOR_MAX,
            "mode": "genus",
            "merge_columns": ["genus", "first_AID_number", "first_AID_in_color_order", "n_strains"],
            "n_colors": 0,
        }
    raise ValueError(f"Unrecognized color table format. Columns: {list(color_table.columns)}")


def read_rdm(path: Path) -> pd.DataFrame:
    rdm = pd.read_csv(path, index_col=0)
    rdm.index = rdm.index.astype(str)
    rdm.columns = rdm.columns.astype(str)
    if not rdm.index.equals(rdm.columns):
        raise ValueError(f"RDM index and columns do not match: {path}")
    values = rdm.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"RDM contains NaN values: {path}")
    return values


def read_neural_rdm(tables_dir: Path) -> tuple[pd.DataFrame, Path]:
    for name in NEURAL_RDM_CANDIDATES:
        path = tables_dir / name
        if path.exists():
            return read_rdm(path), path
    names = ", ".join(NEURAL_RDM_CANDIDATES)
    raise FileNotFoundError(f"could not find neural RDM; tried: {names}")


def clean_distance_matrix(rdm: pd.DataFrame) -> np.ndarray:
    values = rdm.to_numpy(dtype=float, copy=True)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    if np.any(values < -1e-10):
        raise ValueError("distance matrix contains negative values")
    values[values < 0] = 0.0
    return values


def normalize_to_max_two(distance: np.ndarray) -> np.ndarray:
    max_value = float(np.max(distance))
    if max_value <= 0 or not np.isfinite(max_value):
        raise ValueError("cannot normalize a degenerate distance matrix")
    return 2.0 * distance / max_value


def chord_from_linear(linear_distance: np.ndarray) -> np.ndarray:
    corr = 1.0 - linear_distance
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    chord = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(chord, 0.0)
    return chord


def upper_triangle(values: np.ndarray) -> np.ndarray:
    return values[np.triu_indices_from(values, k=1)]


def euclidean_metric_mds(distance: np.ndarray, *, dim: int, seed: int, n_init: int) -> tuple[np.ndarray, np.ndarray]:
    common_kwargs = {
        "n_components": dim,
        "random_state": seed,
        "n_init": n_init,
        "init": "random",
        "max_iter": 1000,
        "eps": 1e-9,
    }
    parameters = inspect.signature(MDS).parameters
    if "metric_mds" in parameters:
        kwargs = {
            **common_kwargs,
            "metric": "precomputed",
            "metric_mds": True,
            "normalized_stress": False,
        }
    else:
        kwargs = {
            **common_kwargs,
            "dissimilarity": "precomputed",
            "metric": True,
        }
        if "normalized_stress" in parameters:
            kwargs["normalized_stress"] = False
    mds = MDS(**kwargs)
    coords = mds.fit_transform(distance)
    return coords, squareform(pdist(coords, metric="euclidean"))


def calculate_global_variance_bic(original: np.ndarray, predicted: np.ndarray, n_params: float) -> float:
    residuals = upper_triangle(original) - upper_triangle(predicted)
    n_pairs = residuals.size
    sigma2 = float(np.mean(residuals**2))
    sigma2 = max(sigma2, 1e-12)
    log_likelihood = -0.5 * float(np.sum(np.log(2.0 * np.pi * sigma2) + residuals**2 / sigma2))
    return float(n_params * np.log(n_pairs) - 2.0 * log_likelihood)


def preservation_metrics(
    original: np.ndarray,
    embedded: np.ndarray,
    predicted: np.ndarray,
    *,
    n_params: float,
) -> dict[str, float | int]:
    original_pairs = upper_triangle(original)
    embedded_pairs = upper_triangle(embedded)
    predicted_pairs = upper_triangle(predicted)
    residual = predicted_pairs - original_pairs
    return {
        "n_samples": int(original.shape[0]),
        "n_pairs": int(original_pairs.size),
        "distance_spearman": float(spearmanr(original_pairs, predicted_pairs).statistic),
        "distance_pearson": float(pearsonr(original_pairs, predicted_pairs).statistic),
        "embedded_distance_spearman": float(spearmanr(original_pairs, embedded_pairs).statistic),
        "embedded_distance_pearson": float(pearsonr(original_pairs, embedded_pairs).statistic),
        "normalized_raw_stress": float(np.sqrt(np.sum(residual**2) / np.sum(original_pairs**2))),
        "mean_absolute_residual": float(np.mean(np.abs(residual))),
        "median_absolute_residual": float(np.median(np.abs(residual))),
        "q95_absolute_residual": float(np.quantile(np.abs(residual), 0.95)),
        "bic_global_variance": calculate_global_variance_bic(original, predicted, n_params),
    }


def lorentz_distances(coords: np.ndarray) -> np.ndarray:
    time = np.sqrt(1.0 + np.sum(coords**2, axis=1))
    xi = np.outer(time, time) - coords @ coords.T
    xi = np.maximum(xi, 1.0)
    distance = np.arccosh(xi)
    np.fill_diagonal(distance, 0.0)
    return distance


def lorentz_to_poincare(coords: np.ndarray) -> np.ndarray:
    time = np.sqrt(1.0 + np.sum(coords**2, axis=1))
    return coords / (time[:, None] + 1.0)


def poincare_translation(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    dp = float(v.dot(x))
    v2 = float(v.dot(v))
    x2 = float(x.dot(x))
    denominator = 1.0 + 2.0 * dp + x2 * v2
    if denominator <= 1e-12:
        return x
    return ((1.0 + 2.0 * dp + x2) * v + (1.0 - v2) * x) / denominator


def recenter_poincare(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm <= 0 or norm >= 0.95:
        return points
    return np.asarray([poincare_translation(-center, point) for point in points])


def initial_hyperbolic_coords(distance: np.ndarray, *, dim: int, seed: int) -> np.ndarray:
    coords, _ = euclidean_metric_mds(distance, dim=dim, seed=seed, n_init=2)
    coords = coords - coords.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(coords, axis=1)))
    if radius <= 0 or not np.isfinite(radius):
        rng = np.random.default_rng(seed)
        coords = rng.normal(scale=0.05, size=(distance.shape[0], dim))
    else:
        coords = coords / radius * 0.6
    return coords


def hyperbolic_objective_and_grad(flat_coords: np.ndarray, target: np.ndarray, dim: int) -> tuple[float, np.ndarray]:
    n = target.shape[0]
    coords = flat_coords.reshape(n, dim)
    time = np.sqrt(1.0 + np.sum(coords**2, axis=1))
    xi = np.outer(time, time) - coords @ coords.T
    xi = np.maximum(xi, 1.0 + 1e-9)
    hyp = np.arccosh(xi)
    np.fill_diagonal(hyp, 0.0)

    i_idx, j_idx = np.triu_indices(n, k=1)
    target_pairs = target[i_idx, j_idx]
    hyp_pairs = hyp[i_idx, j_idx]
    denom = float(np.sum(hyp_pairs**2))
    scale = float(np.sum(target_pairs * hyp_pairs) / denom) if denom > 1e-12 else 1.0
    scale = max(scale, 1e-9)
    residual = scale * hyp_pairs - target_pairs
    loss = 0.5 * float(np.mean(residual**2))

    grad = np.zeros_like(coords)
    pair_weight = residual * scale / np.sqrt(np.maximum(xi[i_idx, j_idx] ** 2 - 1.0, 1e-12))
    pair_weight = pair_weight / target_pairs.size
    for pair_index, (i, j) in enumerate(zip(i_idx, j_idx)):
        weight = pair_weight[pair_index]
        grad_i = weight * ((time[j] / time[i]) * coords[i] - coords[j])
        grad_j = weight * ((time[i] / time[j]) * coords[j] - coords[i])
        grad[i] += grad_i
        grad[j] += grad_j

    ridge = 1e-5
    loss += 0.5 * ridge * float(np.mean(coords**2))
    grad += ridge * coords / coords.size
    return loss, grad.ravel()


def scipy_hyperbolic_mds(
    distance: np.ndarray,
    *,
    dim: int,
    starts: int,
    maxiter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    initial = initial_hyperbolic_coords(distance, dim=dim, seed=seed)
    best = None
    start_summaries = []
    for start_index in range(starts):
        if start_index == 0:
            x0 = initial
        else:
            x0 = initial + rng.normal(scale=0.08, size=initial.shape)
        result = minimize(
            fun=lambda x: hyperbolic_objective_and_grad(x, distance, dim),
            x0=x0.ravel(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7, "maxls": 30},
        )
        start_summaries.append(
            {
                "start": start_index,
                "success": bool(result.success),
                "final_loss": float(result.fun),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("hyperbolic optimizer did not run")

    coords = best.x.reshape(distance.shape[0], dim)
    hyp = lorentz_distances(coords)
    i_idx, j_idx = np.triu_indices(distance.shape[0], k=1)
    hyp_pairs = hyp[i_idx, j_idx]
    distance_pairs = distance[i_idx, j_idx]
    denom = float(np.sum(hyp_pairs**2))
    scale = float(np.sum(distance_pairs * hyp_pairs) / denom) if denom > 1e-12 else 1.0
    scale = max(scale, 1e-9)
    lambda_value = 1.0 / scale
    metadata = {
        "backend": "scipy_lorentz_stress_fallback",
        "optimizer_success": bool(best.success),
        "optimizer_loss": float(best.fun),
        "optimizer_iterations": int(best.nit),
        "optimizer_message": str(best.message),
        "starts": int(starts),
        "maxiter": int(maxiter),
        "start_summaries": json.dumps(start_summaries),
    }
    return coords, hyp, lambda_value, metadata


def run_bayesian_hmds_if_available(
    distance: np.ndarray,
    *,
    dim: int,
    trials: int,
    bayesian_hmds: Path,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int | str]] | None:
    if importlib.util.find_spec("pystan") is None:
        return None
    if not bayesian_hmds.exists():
        return None
    sys.path.insert(0, str(bayesian_hmds))
    try:
        from analysis_from_mat import run_embedding_trials

        trial_results = run_embedding_trials(distance, embedding_dim=dim, n_trials=trials, dmat_unc=None)
        fit = trial_results["best_fit"]
        metadata = {
            "backend": "BayesianHMDS_pystan",
            "trials": int(trials),
            "lambda_mean": float(trial_results["lambda_mean"]),
            "lambda_std": float(trial_results["lambda_std"]),
            "best_bic_bayesian_hmds": float(np.min(trial_results["bic_all"])),
        }
        return fit["euc"], fit["emb_mat"], float(fit["lambda"]), metadata
    finally:
        try:
            sys.path.remove(str(bayesian_hmds))
        except ValueError:
            pass


def set_equal_3d_axes(axis, coords: np.ndarray) -> None:
    center = coords.mean(axis=0)
    radius = float(np.max(np.ptp(coords, axis=0)) / 2.0)
    if radius <= 0 or not np.isfinite(radius):
        radius = 1.0
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def set_first_aid_colorbar(colorbar, color_table: pd.DataFrame, *, label_size: int) -> None:
    _ = color_table
    colorbar.set_ticks(AID_COLORBAR_TICKS)
    colorbar.set_ticklabels([aid_label(tick) for tick in AID_COLORBAR_TICKS])
    colorbar.ax.tick_params(labelsize=label_size)
    colorbar.set_label("Genus color by first AID on A001-A300 axis", fontsize=9)


def set_species_colorbar(colorbar, color_table: pd.DataFrame, *, label_size: int) -> None:
    n_species = len(color_table)
    tick_count = min(6, n_species)
    ticks = np.linspace(0, n_species - 1, tick_count, dtype=int)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"S{tick + 1:02d}" for tick in ticks])
    colorbar.ax.tick_params(labelsize=label_size)
    colorbar.set_label("Species color code", fontsize=9)


def plotly_turbo_colorscale(n_colors: int = 256) -> list[list[float | str]]:
    values = np.linspace(0.0, 1.0, n_colors)
    return [[float(value), color_to_hex(np.asarray(plt.cm.turbo(value)))] for value in values]


def plotly_species_colorscale(color_table: pd.DataFrame) -> list[list[float | str]]:
    n_colors = len(color_table)
    colorscale: list[list[float | str]] = []
    for index, color_hex in enumerate(color_table["color_hex"]):
        colorscale.append([index / n_colors, color_hex])
        colorscale.append([(index + 1) / n_colors, color_hex])
    return colorscale


def analysis_label(result: EmbeddingResult) -> str:
    return str(result.metrics.get("analysis_label") or "neural")


def hmds_distance_label(result: EmbeddingResult) -> str:
    return str(result.metrics.get("hmds_distance_label") or "Chord")


def plot_3d_embedding(
    result: EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
) -> None:
    cfg = _detect_color_mode(color_table)
    coords = result.coordinates
    merged = sample_table.copy()
    for dim_index in range(coords.shape[1]):
        merged[f"x{dim_index + 1}"] = coords[:, dim_index]
    merged = merged.merge(color_table[cfg["merge_columns"]], on=cfg["merge_key"], how="left")

    figure = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    norm = plt.Normalize(vmin=cfg["norm_vmin"], vmax=cfg["norm_vmax"])
    scatter = axis.scatter(
        merged["x1"],
        merged["x2"],
        merged["x3"],
        c=merged[cfg["color_column"]],
        cmap=plt.cm.turbo,
        norm=norm,
        s=48,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.45,
    )
    axis.set_title(result.method.replace("_", " "), fontsize=12)
    axis.set_xlabel("Dim 1", fontsize=9)
    axis.set_ylabel("Dim 2", fontsize=9)
    axis.set_zlabel("Dim 3", fontsize=9)
    axis.tick_params(labelsize=8)
    set_equal_3d_axes(axis, coords)

    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.04)
    if cfg["mode"] == "species":
        set_species_colorbar(colorbar, color_table, label_size=7)
    else:
        set_first_aid_colorbar(colorbar, color_table, label_size=7)

    subtitle = (
        f"stress={result.metrics['normalized_raw_stress']:.3f}; "
        f"rho={result.metrics['distance_spearman']:.3f}"
    )
    if result.metrics.get("lambda") is not None:
        subtitle += f"; lambda={result.metrics['lambda']:.3f}"
    figure.suptitle(f"86bac {analysis_label(result)} embedding colored by {cfg['mode']}\n{subtitle}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_poincare_projections(
    result: EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
) -> None:
    cfg = _detect_color_mode(color_table)
    coords = result.coordinates
    merged = sample_table.copy()
    for dim_index in range(coords.shape[1]):
        merged[f"x{dim_index + 1}"] = coords[:, dim_index]
    merged = merged.merge(color_table[cfg["merge_columns"]], on=cfg["merge_key"], how="left")

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True)
    norm = plt.Normalize(vmin=cfg["norm_vmin"], vmax=cfg["norm_vmax"])
    projections = [(1, 2), (1, 3), (2, 3)]
    scatter = None
    for axis, (left, right) in zip(axes, projections):
        circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.5, linewidth=1.0)
        axis.add_artist(circle)
        scatter = axis.scatter(
            merged[f"x{left}"],
            merged[f"x{right}"],
            c=merged[cfg["color_column"]],
            cmap=plt.cm.turbo,
            norm=norm,
            s=34,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.35,
        )
        axis.set_xlim(-1.04, 1.04)
        axis.set_ylim(-1.04, 1.04)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"Poincare {left}", fontsize=9)
        axis.set_ylabel(f"Poincare {right}", fontsize=9)
        axis.set_title(f"{left} vs {right}", fontsize=10)
        axis.grid(True, color="#e5e7eb", linewidth=0.6)
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=axes, fraction=0.025, pad=0.02)
        if cfg["mode"] == "species":
            set_species_colorbar(colorbar, color_table, label_size=6)
        else:
            set_first_aid_colorbar(colorbar, color_table, label_size=6)
    figure.suptitle(
        f"86bac {analysis_label(result)} HMDS Poincare projections colored by {cfg['mode']}",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_poincare_ball_3d(
    result: EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
) -> None:
    cfg = _detect_color_mode(color_table)
    coords = result.coordinates
    merged = sample_table.copy()
    for dim_index in range(coords.shape[1]):
        merged[f"x{dim_index + 1}"] = coords[:, dim_index]
    merged = merged.merge(color_table[cfg["merge_columns"]], on=cfg["merge_key"], how="left")

    figure = plt.figure(figsize=(8.6, 8.0), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")

    u, v = np.mgrid[0 : 2 * np.pi : 32j, 0 : np.pi : 16j]
    sphere_x = np.cos(u) * np.sin(v)
    sphere_y = np.sin(u) * np.sin(v)
    sphere_z = np.cos(v)
    axis.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#6b7280",
        linewidth=0.45,
        alpha=0.22,
        rstride=2,
        cstride=2,
    )

    norm = plt.Normalize(vmin=cfg["norm_vmin"], vmax=cfg["norm_vmax"])
    scatter = axis.scatter(
        merged["x1"],
        merged["x2"],
        merged["x3"],
        c=merged[cfg["color_column"]],
        cmap=plt.cm.turbo,
        norm=norm,
        s=46,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
    )
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-1.02, 1.02)
    axis.set_zlim(-1.02, 1.02)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.set_zlabel("Poincare 3", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.set_title("HMDS in the Poincare ball", fontsize=12)

    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.04)
    if cfg["mode"] == "species":
        set_species_colorbar(colorbar, color_table, label_size=6)
    else:
        set_first_aid_colorbar(colorbar, color_table, label_size=6)

    figure.suptitle(
        f"86bac {analysis_label(result)} HMDS Poincare ball colored by {cfg['mode']}\n"
        f"stress={result.metrics['normalized_raw_stress']:.3f}; "
        f"rho={result.metrics['distance_spearman']:.3f}; "
        f"lambda={result.metrics['lambda']:.3f}",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_poincare_ball_3d_html(
    result: EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    cfg = _detect_color_mode(color_table)
    coords = result.coordinates
    merged = sample_table.copy()
    for dim_index in range(coords.shape[1]):
        merged[f"x{dim_index + 1}"] = coords[:, dim_index]
    merged = merged.merge(color_table[cfg["merge_columns"]], on=cfg["merge_key"], how="left")
    figure = go.Figure()
    wire_line = {"color": "rgba(107,114,128,0.28)", "width": 1}
    theta = np.linspace(0.0, np.pi, 90)
    phi = np.linspace(0.0, 2.0 * np.pi, 140)
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        figure.add_trace(
            go.Scatter3d(
                x=np.cos(angle) * np.sin(theta),
                y=np.sin(angle) * np.sin(theta),
                z=np.cos(theta),
                mode="lines",
                line=wire_line,
                hoverinfo="skip",
                showlegend=False,
                name="Poincare ball boundary",
            )
        )
    for z_value in np.linspace(-0.8, 0.8, 5):
        radius = math.sqrt(max(1.0 - z_value**2, 0.0))
        figure.add_trace(
            go.Scatter3d(
                x=radius * np.cos(phi),
                y=radius * np.sin(phi),
                z=np.full_like(phi, z_value),
                mode="lines",
                line=wire_line,
                hoverinfo="skip",
                showlegend=False,
                name="Poincare ball boundary",
            )
        )

    if cfg["mode"] == "species":
        colorscale = plotly_species_colorscale(color_table)
        tick_count = min(6, cfg["n_colors"])
        cbar_tickvals = np.linspace(0, cfg["n_colors"] - 1, tick_count, dtype=int)
        cbar_ticktext = [f"S{tick + 1:02d}" for tick in cbar_tickvals]
        cbar_title = "Species"
        customdata = np.column_stack(
            [
                merged["genus"].astype(str),
                merged["species"].astype(str),
                merged["first_AID"].astype(str),
                (merged["species_index"].astype(int) + 1).astype(str),
            ]
        )
        hovertemplate = (
            "name=%{text}<br>"
            "genus=%{customdata[0]}<br>"
            "species=%{customdata[1]}<br>"
            "species color code=S%{customdata[3]}<br>"
            "species first AID=%{customdata[2]}<br>"
            "Poincare1=%{x:.4f}<br>"
            "Poincare2=%{y:.4f}<br>"
            "Poincare3=%{z:.4f}<extra></extra>"
        )
    else:
        colorscale = plotly_turbo_colorscale()
        cbar_tickvals = AID_COLORBAR_TICKS
        cbar_ticktext = [aid_label(tick) for tick in AID_COLORBAR_TICKS]
        cbar_title = "First AID"
        customdata = np.column_stack(
            [
                merged["genus"].astype(str),
                merged["species"].astype(str),
                merged["first_AID_in_color_order"].astype(str),
                merged["first_AID_number"].astype(int).astype(str),
            ]
        )
        hovertemplate = (
            "name=%{text}<br>"
            "genus=%{customdata[0]}<br>"
            "species=%{customdata[1]}<br>"
            "genus color from %{customdata[2]} on A001-A300 axis<br>"
            "first AID number=%{customdata[3]}<br>"
            "Poincare1=%{x:.4f}<br>"
            "Poincare2=%{y:.4f}<br>"
            "Poincare3=%{z:.4f}<extra></extra>"
        )

    figure.add_trace(
        go.Scatter3d(
            x=merged["x1"],
            y=merged["x2"],
            z=merged["x3"],
            mode="markers",
            name="86bac strains",
            showlegend=False,
            marker={
                "size": 5,
                "color": merged[cfg["color_column"]],
                "colorscale": colorscale,
                "cmin": cfg["norm_vmin"],
                "cmax": cfg["norm_vmax"],
                "line": {"width": 0.5, "color": "white"},
                "colorbar": {
                    "title": cbar_title,
                    "tickmode": "array",
                    "tickvals": cbar_tickvals,
                    "ticktext": cbar_ticktext,
                    "tickfont": {"size": 9},
                },
            },
            text=merged["sample_id"].astype(str),
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )

    figure.update_layout(
        title=(
            f"86bac {analysis_label(result)} HMDS Poincare ball"
            f"<br>stress={result.metrics['normalized_raw_stress']:.3f}, "
            f"rho={result.metrics['distance_spearman']:.3f}, "
            f"lambda={result.metrics['lambda']:.3f}"
        ),
        width=1080,
        height=840,
        margin={"l": 10, "r": 130, "t": 70, "b": 10},
        scene={
            "xaxis": {"title": "Poincare 1", "range": [-1.02, 1.02]},
            "yaxis": {"title": "Poincare 2", "range": [-1.02, 1.02]},
            "zaxis": {"title": "Poincare 3", "range": [-1.02, 1.02]},
            "aspectmode": "cube",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def plot_shepard(result: EmbeddingResult, original: np.ndarray, output_path: Path) -> None:
    original_pairs = upper_triangle(original)
    predicted_pairs = upper_triangle(result.predicted_distance)
    limit = float(max(original_pairs.max(), predicted_pairs.max()) * 1.04)

    figure, axis = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    axis.scatter(original_pairs, predicted_pairs, s=10, color="#4a5568", alpha=0.24, linewidths=0)
    axis.plot([0, limit], [0, limit], linestyle="--", color="#111827", linewidth=1.0)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Input distance", fontsize=10)
    axis.set_ylabel("Embedded predicted distance", fontsize=10)
    axis.set_title(
        result.method.replace("_", " ")
        + "\n"
        + f"rho={result.metrics['distance_spearman']:.3f}; "
        + f"r={result.metrics['distance_pearson']:.3f}; "
        + f"stress={result.metrics['normalized_raw_stress']:.3f}",
        fontsize=12,
    )
    axis.grid(True, color="#e5e7eb", linewidth=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def save_coordinates(result: EmbeddingResult, sample_table: pd.DataFrame, output_path: Path) -> None:
    table = sample_table.copy()
    prefix = "poincare" if "hmds" in result.method else "mds"
    for dim_index in range(result.coordinates.shape[1]):
        table[f"{prefix}{dim_index + 1}"] = result.coordinates[:, dim_index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)


def save_pair_distances(
    result: EmbeddingResult,
    original: np.ndarray,
    samples: list[str],
    output_path: Path,
) -> None:
    i_idx, j_idx = np.triu_indices(len(samples), k=1)
    table = pd.DataFrame(
        {
            "sample_left": np.asarray(samples, dtype=object)[i_idx],
            "sample_right": np.asarray(samples, dtype=object)[j_idx],
            "input_distance": original[i_idx, j_idx],
            "embedded_distance": result.embedded_distance[i_idx, j_idx],
            "predicted_distance": result.predicted_distance[i_idx, j_idx],
            "residual_predicted_minus_input": result.predicted_distance[i_idx, j_idx] - original[i_idx, j_idx],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)


def build_sample_table(samples: list[str], taxonomy: pd.DataFrame) -> pd.DataFrame:
    table = taxonomy.loc[samples, ["AID", "genus", "species"]].rename(columns={"AID": "sample_id"}).reset_index(drop=True)
    table["_aid_sort_key"] = table["sample_id"].map(aid_sort_key)
    return table


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.root
    tables_dir = root / "tables"
    output_dir = root / "hmds_comparison"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    neural_rdm, neural_path = read_neural_rdm(tables_dir)
    taxonomy = pd.read_csv(tables_dir / TAXONOMY)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    taxonomy = taxonomy.set_index("AID", drop=False)
    samples = [sample for sample in neural_rdm.index.astype(str) if sample in taxonomy.index]
    if len(samples) < 4:
        raise ValueError(f"need at least 4 samples with taxonomy; found {len(samples)}")
    neural_rdm = neural_rdm.loc[samples, samples]
    taxonomy = taxonomy.loc[samples].copy()
    sample_table = build_sample_table(samples, taxonomy)
    color_table = species_color_table(taxonomy)
    color_table.to_csv(table_dir / "species_color_map.csv", index=False)

    linear_raw = clean_distance_matrix(neural_rdm)
    chord_raw = chord_from_linear(linear_raw)
    linear = normalize_to_max_two(linear_raw)
    chord = normalize_to_max_two(chord_raw)
    pd.DataFrame(linear, index=samples, columns=samples).to_csv(table_dir / "neural_distance__linear_normalized.csv")
    pd.DataFrame(chord, index=samples, columns=samples).to_csv(table_dir / "neural_distance__chord_normalized.csv")

    results: list[EmbeddingResult] = []
    for method, input_name, distance in [
        ("linear_euclidean_mds_3d", "linear_normalized", linear),
        ("chord_euclidean_mds_3d", "chord_normalized", chord),
    ]:
        coords, embedded = euclidean_metric_mds(distance, dim=args.dim, seed=args.seed, n_init=args.mds_n_init)
        n_params = distance.shape[0] * args.dim + 1
        metrics = preservation_metrics(distance, embedded, embedded, n_params=n_params)
        metrics.update({"lambda": None, "backend": "sklearn_metric_mds"})
        results.append(
            EmbeddingResult(
                method=method,
                input_distance_name=input_name,
                coordinates=coords,
                embedded_distance=embedded,
                predicted_distance=embedded,
                metrics=metrics,
            )
        )

    hmds_backend_status = "not_attempted"
    try:
        bayesian = run_bayesian_hmds_if_available(
            chord,
            dim=args.dim,
            trials=args.hmds_trials,
            bayesian_hmds=args.bayesian_hmds,
        )
        if bayesian is None:
            hmds_backend_status = "pystan_unavailable_or_bayesian_hmds_missing; used scipy fallback"
            coords_lorentz, embedded_hyp, lambda_value, metadata = scipy_hyperbolic_mds(
                chord,
                dim=args.dim,
                starts=args.hmds_starts,
                maxiter=args.hmds_maxiter,
                seed=args.seed,
            )
        else:
            hmds_backend_status = "BayesianHMDS_pystan"
            coords_lorentz, embedded_hyp, lambda_value, metadata = bayesian
        poincare = recenter_poincare(lorentz_to_poincare(coords_lorentz))
        predicted = embedded_hyp / lambda_value
        n_params = chord.shape[0] * args.dim + 1 - args.dim * (args.dim - 1) / 2
        metrics = preservation_metrics(chord, embedded_hyp, predicted, n_params=n_params)
        metrics.update({"lambda": float(lambda_value), **metadata})
        results.append(
            EmbeddingResult(
                method="chord_hmds_poincare_3d",
                input_distance_name="chord_normalized",
                coordinates=poincare,
                embedded_distance=embedded_hyp,
                predicted_distance=predicted,
                metrics=metrics,
            )
        )
    except Exception as exc:
        hmds_backend_status = f"failed: {exc}"

    summary_rows = []
    for result in results:
        save_coordinates(result, sample_table, table_dir / f"coordinates__{result.method}.csv")
        original = linear if result.input_distance_name.startswith("linear") else chord
        save_pair_distances(result, original, samples, table_dir / f"pair_distances__{result.method}.csv")
        plot_3d_embedding(result, sample_table, color_table, figure_dir / f"embedding__{result.method}.png")
        plot_shepard(result, original, figure_dir / f"shepard__{result.method}.png")
        if result.method == "chord_hmds_poincare_3d":
            plot_poincare_projections(result, sample_table, color_table, figure_dir / "poincare_projections__chord_hmds.png")
            plot_poincare_ball_3d(result, sample_table, color_table, figure_dir / "poincare_ball__chord_hmds_3d.png")
            plot_poincare_ball_3d_html(result, sample_table, color_table, figure_dir / "poincare_ball__chord_hmds_3d.html")
            plot_poincare_ball_3d_html(
                result,
                sample_table,
                color_table,
                figure_dir / "poincare_ball__chord_hmds_3d_hover_labels.html",
            )
        row = {
            "method": result.method,
            "input_distance": result.input_distance_name,
            **result.metrics,
        }
        summary_rows.append(row)

    summary = {
        "input": {
            "root": str(root),
            "neural_rdm": str(neural_path),
            "taxonomy": str(tables_dir / TAXONOMY),
            "bayesian_hmds": str(args.bayesian_hmds),
        },
        "parameters": {
            "dim": int(args.dim),
            "seed": int(args.seed),
            "mds_n_init": int(args.mds_n_init),
            "hmds_trials": int(args.hmds_trials),
            "hmds_starts": int(args.hmds_starts),
            "hmds_maxiter": int(args.hmds_maxiter),
        },
        "n_samples": len(samples),
        "n_species": int(taxonomy["species"].nunique()),
        "hmds_backend_status": hmds_backend_status,
        "outputs": {
            "summary_csv": str(table_dir / "embedding_comparison_summary.csv"),
            "summary_json": str(output_dir / "run_summary.json"),
            "figures_dir": str(figure_dir),
            "tables_dir": str(table_dir),
        },
    }
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(table_dir / "embedding_comparison_summary.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare linear/chord Euclidean MDS and chord HMDS for 86bac neural RDM.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bayesian-hmds", type=Path, default=DEFAULT_BAYESIAN_HMDS)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mds-n-init", type=int, default=8)
    parser.add_argument("--hmds-trials", type=int, default=3)
    parser.add_argument("--hmds-starts", type=int, default=6)
    parser.add_argument("--hmds-maxiter", type=int, default=700)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
