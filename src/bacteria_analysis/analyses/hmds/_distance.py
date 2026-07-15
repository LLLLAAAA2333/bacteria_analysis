"""Distance-matrix utilities for HMDS pre- and post-processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def clean_distance_matrix(rdm: pd.DataFrame) -> np.ndarray:
    """Symmetrise, zero-diagonal, and clip-negatives of an RDM.

    Parameters
    ----------
    rdm : pd.DataFrame
        Square distance / RDM matrix.

    Returns
    -------
    (n, n) ndarray
        Clean float64 distance matrix.
    """
    values = rdm.to_numpy(dtype=float, copy=True)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    if np.any(values < -1e-10):
        raise ValueError("distance matrix contains negative values")
    values[values < 0] = 0.0
    return values


def chord_from_linear(linear_distance: np.ndarray) -> np.ndarray:
    """Convert correlation-based distance to chord distance.

    ``chord = sqrt(2 * (1 - correlation))``, which makes a correlation-based
    dissimilarity Euclidean-embeddable (including hyperbolic).

    Parameters
    ----------
    linear_distance : (n, n) ndarray
        Linear distance matrix (e.g. ``1 - Pearson r``).

    Returns
    -------
    (n, n) ndarray
        Chord distance matrix.
    """
    corr = 1.0 - linear_distance
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    chord = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(chord, 0.0)
    return chord


def normalize_to_max_two(distance: np.ndarray) -> np.ndarray:
    """Scale a distance matrix so its maximum value is exactly 2.0.

    Standard pre-processing for hyperbolic MDS.

    Parameters
    ----------
    distance : (n, n) ndarray
        Non-negative distance matrix.

    Returns
    -------
    (n, n) ndarray
        Normalised distance matrix.
    """
    max_value = float(np.max(distance))
    if max_value <= 0 or not np.isfinite(max_value):
        raise ValueError("cannot normalize a degenerate distance matrix")
    return 2.0 * distance / max_value


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the strict upper-triangle entries as a flat 1-D array.

    Parameters
    ----------
    matrix : (n, n) ndarray
        Square matrix.

    Returns
    -------
    (n*(n-1)/2,) ndarray
    """
    return matrix[np.triu_indices_from(matrix, k=1)]


def _calculate_global_variance_bic(
    original: np.ndarray, predicted: np.ndarray, n_params: float,
) -> float:
    """BIC from global-variance Gaussian likelihood of residual pairs."""
    residuals = upper_triangle(original) - upper_triangle(predicted)
    n_pairs = residuals.size
    sigma2 = float(np.mean(residuals**2))
    sigma2 = max(sigma2, 1e-12)
    log_likelihood = -0.5 * float(
        np.sum(np.log(2.0 * np.pi * sigma2) + residuals**2 / sigma2)
    )
    return float(n_params * np.log(n_pairs) - 2.0 * log_likelihood)


def preservation_metrics(
    original: np.ndarray,
    embedded: np.ndarray,
    predicted: np.ndarray,
    *,
    n_params: float,
) -> dict[str, float | int]:
    """Compute embedding quality metrics from pairwise distances.

    Parameters
    ----------
    original : (n, n) ndarray
        Input distance matrix.
    embedded : (n, n) ndarray
        Raw embedded (hyperbolic) distances.
    predicted : (n, n) ndarray
        Scale-optimised predicted distances (``embedded / lambda``).
    n_params : float
        Effective number of parameters for BIC.

    Returns
    -------
    dict
        Keys: ``n_samples``, ``n_pairs``, ``distance_spearman``,
        ``distance_pearson``, ``embedded_distance_spearman``,
        ``embedded_distance_pearson``, ``normalized_raw_stress``,
        ``mean_absolute_residual``, ``median_absolute_residual``,
        ``q95_absolute_residual``, ``bic_global_variance``.
    """
    orig_pairs = upper_triangle(original)
    emb_pairs = upper_triangle(embedded)
    pred_pairs = upper_triangle(predicted)
    residual = pred_pairs - orig_pairs
    return {
        "n_samples": int(original.shape[0]),
        "n_pairs": int(orig_pairs.size),
        "distance_spearman": float(spearmanr(orig_pairs, pred_pairs).statistic),
        "distance_pearson": float(pearsonr(orig_pairs, pred_pairs).statistic),
        "embedded_distance_spearman": float(spearmanr(orig_pairs, emb_pairs).statistic),
        "embedded_distance_pearson": float(pearsonr(orig_pairs, emb_pairs).statistic),
        "normalized_raw_stress": float(
            np.sqrt(np.sum(residual**2) / np.sum(orig_pairs**2))
        ),
        "mean_absolute_residual": float(np.mean(np.abs(residual))),
        "median_absolute_residual": float(np.median(np.abs(residual))),
        "q95_absolute_residual": float(np.quantile(np.abs(residual), 0.95)),
        "bic_global_variance": _calculate_global_variance_bic(
            original, predicted, n_params,
        ),
    }
