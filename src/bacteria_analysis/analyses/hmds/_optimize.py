"""Hyperbolic MDS stress optimisation via L-BFGS-B."""

from __future__ import annotations

import inspect
import json

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from bacteria_analysis.analyses.hmds._geometry import lorentz_distances


def euclidean_metric_mds(
    distance: np.ndarray, *, dim: int, seed: int, n_init: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical / metric MDS with a precomputed distance matrix.

    Adapts to both old and new sklearn MDS signatures.

    Parameters
    ----------
    distance : (n, n) ndarray
        Precomputed distance matrix.
    dim : int
        Embedding dimension.
    seed : int
        Random seed.
    n_init : int
        Number of initialisations.

    Returns
    -------
    coords : (n, dim) ndarray
    embedded : (n, n) ndarray
        Euclidean distances between embedded points.
    """
    common_kwargs: dict = {
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


def initial_hyperbolic_coords(
    distance: np.ndarray, *, dim: int, seed: int,
) -> np.ndarray:
    """Generate starting coords via Euclidean MDS, scaled to radius 0.6.

    Parameters
    ----------
    distance : (n, n) ndarray
        Input distance matrix.
    dim : int
        Embedding dimension.
    seed : int
        Random seed passed to ``euclidean_metric_mds``.

    Returns
    -------
    (n, dim) ndarray
        Initial Poincaré-ball coordinates (radius ≤ 0.6).
    """
    coords, _ = euclidean_metric_mds(distance, dim=dim, seed=seed, n_init=2)
    coords = coords - coords.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(coords, axis=1)))
    if radius <= 0 or not np.isfinite(radius):
        rng = np.random.default_rng(seed)
        coords = rng.normal(scale=0.05, size=(distance.shape[0], dim))
    else:
        coords = coords / radius * 0.6
    return coords


def hyperbolic_objective_and_grad(
    flat_coords: np.ndarray, target: np.ndarray, dim: int,
) -> tuple[float, np.ndarray]:
    """Scale-optimised stress loss + analytic gradient for Lorentz coords.

    Parameters
    ----------
    flat_coords : (n*dim,) ndarray
        Flattened Lorentz spatial coordinates.
    target : (n, n) ndarray
        Target distance matrix.
    dim : int
        Embedding dimension.

    Returns
    -------
    loss : float
        Regularised stress value.
    grad : (n*dim,) ndarray
        Gradient w.r.t. *flat_coords*.
    """
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
    scale = (
        float(np.sum(target_pairs * hyp_pairs) / denom) if denom > 1e-12 else 1.0
    )
    scale = max(scale, 1e-9)
    residual = scale * hyp_pairs - target_pairs
    loss = 0.5 * float(np.mean(residual**2))

    grad = np.zeros_like(coords)
    pair_weight = (
        residual * scale
        / np.sqrt(np.maximum(xi[i_idx, j_idx] ** 2 - 1.0, 1e-12))
    )
    pair_weight = pair_weight / target_pairs.size
    for pair_idx, (i, j) in enumerate(zip(i_idx, j_idx)):
        w = pair_weight[pair_idx]
        grad[i] += w * ((time[j] / time[i]) * coords[i] - coords[j])
        grad[j] += w * ((time[i] / time[j]) * coords[j] - coords[i])

    ridge = 1e-5
    loss += 0.5 * ridge * float(np.mean(coords**2))
    grad += ridge * coords / coords.size
    return loss, grad.ravel()


def scipy_hyperbolic_mds(
    distance: np.ndarray,
    *,
    dim: int,
    starts: int = 8,
    maxiter: int = 900,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Multi-start hyperbolic MDS via L-BFGS-B.

    Parameters
    ----------
    distance : (n, n) ndarray
        Input distance matrix (should be normalised to max ≈ 2).
    dim : int
        Embedding dimension (typically 2 or 3).
    starts : int
        Number of random-restart initialisations.
    maxiter : int
        Max L-BFGS-B iterations per start.
    seed : int
        Random seed.

    Returns
    -------
    lorentz_coords : (n, dim) ndarray
        Best Lorentz spatial coordinates.
    hyperbolic_distances : (n, n) ndarray
        Hyperbolic distance matrix from the best embedding.
    lambda_value : float
        Optimal scale factor (``1 / scale``).
    metadata : dict
        Optimiser diagnostics (backend, success, loss, iterations,
        per-start summaries).
    """
    rng = np.random.default_rng(seed)
    initial = initial_hyperbolic_coords(distance, dim=dim, seed=seed)
    best: object = None
    start_summaries: list[dict] = []
    for start_idx in range(starts):
        if start_idx == 0:
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
        start_summaries.append({
            "start": start_idx,
            "success": bool(result.success),
            "final_loss": float(result.fun),
            "iterations": int(result.nit),
            "message": str(result.message),
        })
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("hyperbolic optimizer did not run")

    coords = best.x.reshape(distance.shape[0], dim)
    hyp = lorentz_distances(coords)
    i_idx, j_idx = np.triu_indices(distance.shape[0], k=1)
    hyp_pairs = hyp[i_idx, j_idx]
    dist_pairs = distance[i_idx, j_idx]
    denom = float(np.sum(hyp_pairs**2))
    scale = (
        float(np.sum(dist_pairs * hyp_pairs) / denom)
        if denom > 1e-12
        else 1.0
    )
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
