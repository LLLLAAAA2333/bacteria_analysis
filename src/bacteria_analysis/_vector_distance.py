"""Low-level vector distance computation for neural feature construction."""

from __future__ import annotations

import numpy as np

MIN_VALID_VALUES = 2
VALID_COMPARISON_STATUS = "ok"


def compute_vector_distance(
    left: np.ndarray, right: np.ndarray, metric: str = "correlation"
) -> tuple[float, str]:
    """Compute a distance on two aligned one-dimensional vectors."""

    if metric not in ("correlation", "euclidean"):
        raise ValueError(f"unsupported metric: {metric}")
    if left.size < MIN_VALID_VALUES or right.size < MIN_VALID_VALUES:
        return float("nan"), "insufficient_valid_values"

    if metric == "euclidean":
        return float(np.linalg.norm(left - right)), VALID_COMPARISON_STATUS

    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return float("nan"), "constant_vector"

    correlation = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(correlation):
        return float("nan"), "invalid_correlation"

    return float(np.clip(1.0 - correlation, 0.0, 2.0)), VALID_COMPARISON_STATUS
