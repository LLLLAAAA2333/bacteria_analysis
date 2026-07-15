"""Convenience pipeline: distance matrix → hyperbolic MDS embedding."""

from __future__ import annotations

import numpy as np

from bacteria_analysis.analyses.hmds._distance import (
    normalize_to_max_two,
    preservation_metrics,
)
from bacteria_analysis.analyses.hmds._geometry import (
    lorentz_to_poincare,
    recenter_poincare,
)
from bacteria_analysis.analyses.hmds._optimize import scipy_hyperbolic_mds
from bacteria_analysis.analyses.hmds._result import EmbeddingResult


def fit_hyperbolic_mds(
    distance: np.ndarray,
    *,
    dim: int = 2,
    starts: int = 8,
    maxiter: int = 900,
    seed: int = 42,
    method: str = "hmds",
    input_distance_name: str = "distance",
    normalize: bool = True,
) -> EmbeddingResult:
    """Fit hyperbolic MDS to a distance matrix.

    Convenience wrapper that normalises the distance, runs multi-start
    L-BFGS-B optimisation, converts to Poincaré coordinates, recentres,
    and computes preservation metrics.

    Parameters
    ----------
    distance : (n, n) ndarray
        Input distance matrix.
    dim : int
        Embedding dimension (2 or 3).
    starts : int
        Number of random-restart initialisations.
    maxiter : int
        Max L-BFGS-B iterations per start.
    seed : int
        Random seed.
    method : str
        Label for the embedding method.
    input_distance_name : str
        Label for the input distance.
    normalize : bool
        If True (default), apply ``normalize_to_max_two`` to *distance*.

    Returns
    -------
    EmbeddingResult
    """
    dist = normalize_to_max_two(distance) if normalize else distance

    lorentz, embedded_hyp, lam, metadata = scipy_hyperbolic_mds(
        dist, dim=dim, starts=starts, maxiter=maxiter, seed=seed,
    )
    poincare = recenter_poincare(lorentz_to_poincare(lorentz))
    predicted = embedded_hyp / lam
    n_params = distance.shape[0] * dim + 1 - dim * (dim - 1) / 2
    metrics = preservation_metrics(dist, embedded_hyp, predicted, n_params=n_params)

    radius = np.linalg.norm(poincare, axis=1)
    hyp_radius = 2.0 * np.arctanh(np.clip(radius, 0.0, 0.999999))
    metrics.update({
        "lambda": float(lam),
        "dim": dim,
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
    return EmbeddingResult(
        method=method,
        input_distance_name=input_distance_name,
        coordinates=poincare,
        embedded_distance=embedded_hyp,
        predicted_distance=predicted,
        metrics=metrics,
    )


def radius_summary(coords: np.ndarray) -> dict[str, float]:
    """Summarise the radial distribution of Poincaré coordinates.

    Parameters
    ----------
    coords : (n, dim) ndarray
        Poincaré ball coordinates.

    Returns
    -------
    dict
        Keys: ``poincare_radius_min``, ``poincare_radius_median``,
        ``poincare_radius_mean``, ``poincare_radius_q90``,
        ``poincare_radius_q95``, ``poincare_radius_max``,
        ``hyperbolic_radius_median``, ``fraction_radius_gt_0_90``,
        ``fraction_radius_gt_0_95``.
    """
    radius = np.linalg.norm(coords, axis=1)
    hyperbolic_radius = 2.0 * np.arctanh(np.clip(radius, 0.0, 0.999999))
    return {
        "poincare_radius_min": float(radius.min()),
        "poincare_radius_median": float(np.median(radius)),
        "poincare_radius_mean": float(radius.mean()),
        "poincare_radius_q90": float(np.quantile(radius, 0.90)),
        "poincare_radius_q95": float(np.quantile(radius, 0.95)),
        "poincare_radius_max": float(radius.max()),
        "hyperbolic_radius_median": float(np.median(hyperbolic_radius)),
        "fraction_radius_gt_0_90": float(np.mean(radius > 0.90)),
        "fraction_radius_gt_0_95": float(np.mean(radius > 0.95)),
    }
