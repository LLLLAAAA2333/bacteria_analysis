"""Poincaré / Lorentz geometry utilities for hyperbolic MDS.

All functions are dimension-agnostic (work for any ``dim >= 2``).
"""

from __future__ import annotations

import numpy as np


def lorentz_distances(coords: np.ndarray) -> np.ndarray:
    """Pairwise hyperbolic distances between Lorentz-model points.

    Parameters
    ----------
    coords : (n, dim) ndarray
        Lorentz coordinates (spatial components of the hyperboloid model).

    Returns
    -------
    (n, n) ndarray
        Hyperbolic distance matrix (zero diagonal).
    """
    time = np.sqrt(1.0 + np.sum(coords**2, axis=1))
    xi = np.outer(time, time) - coords @ coords.T
    xi = np.maximum(xi, 1.0)
    distance = np.arccosh(xi)
    np.fill_diagonal(distance, 0.0)
    return distance


def lorentz_to_poincare(coords: np.ndarray) -> np.ndarray:
    """Stereographic projection from Lorentz hyperboloid to Poincaré ball.

    Parameters
    ----------
    coords : (n, dim) ndarray
        Lorentz spatial coordinates.

    Returns
    -------
    (n, dim) ndarray
        Poincaré ball coordinates (radius < 1).
    """
    time = np.sqrt(1.0 + np.sum(coords**2, axis=1))
    return coords / (time[:, None] + 1.0)


def poincare_translation(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Möbius translation of point *x* by vector *v* in the Poincaré ball.

    Parameters
    ----------
    v : (dim,) ndarray
        Translation vector.
    x : (dim,) ndarray
        Point to translate.

    Returns
    -------
    (dim,) ndarray
        Translated point.
    """
    dp = float(v.dot(x))
    v2 = float(v.dot(v))
    x2 = float(x.dot(x))
    denominator = 1.0 + 2.0 * dp + x2 * v2
    if denominator <= 1e-12:
        return x
    return ((1.0 + 2.0 * dp + x2) * v + (1.0 - v2) * x) / denominator


def recenter_poincare(points: np.ndarray) -> np.ndarray:
    """Centre a set of Poincaré points by translating their mean to the origin.

    Parameters
    ----------
    points : (n, dim) ndarray
        Poincaré ball coordinates.

    Returns
    -------
    (n, dim) ndarray
        Recentred coordinates.
    """
    centre = points.mean(axis=0)
    norm = float(np.linalg.norm(centre))
    if norm <= 0 or norm >= 0.95:
        return points
    return np.asarray([poincare_translation(-centre, pt) for pt in points])
