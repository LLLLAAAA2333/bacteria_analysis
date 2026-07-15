"""EmbeddingResult dataclass — canonical return type for HMDS fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingResult:
    """Result of a single hyperbolic MDS embedding run.

    Attributes
    ----------
    method : str
        Identifier for the embedding method (e.g. ``"chord_hmds_poincare_2d"``).
    input_distance_name : str
        Label for the input distance used (e.g. ``"chord_normalized"``).
    coordinates : np.ndarray  (n_samples, dim)
        Poincaré ball coordinates (re-centred to origin).
    embedded_distance : np.ndarray  (n_samples, n_samples)
        Raw hyperbolic distances from the Lorentz embedding.
    predicted_distance : np.ndarray  (n_samples, n_samples)
        Scale-optimised predicted distances (`embedded_distance / lambda`).
    metrics : dict
        Quality metrics (stress, Spearman ρ, λ, radius summary, …).
    """

    method: str
    input_distance_name: str
    coordinates: np.ndarray
    embedded_distance: np.ndarray
    predicted_distance: np.ndarray
    metrics: dict[str, float | int | str | None]
