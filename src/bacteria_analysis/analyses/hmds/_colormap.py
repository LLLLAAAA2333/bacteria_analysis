"""Ward-dendrogram leaf-order colormap.

Assigns colours to leaves based on cophenetic gaps in a Ward dendrogram,
producing a continuous colormap that respects hierarchical structure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cophenet, leaves_list
from scipy.spatial.distance import squareform


def leaf_order_colormap(
    Z: np.ndarray, n: int, cmap_name: str = "turbo",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign colours to *n* leaves from a Ward linkage *Z*.

    Colours are ordered by the dendrogram leaf order, with spacing
    proportional to ``arcsinh(cophenetic_gap / median_gap)`` so that
    deep phylogenetic splits produce visible colour transitions.

    Parameters
    ----------
    Z : (n-1, 4) ndarray
        Scipy linkage matrix (``method="ward"``).
    n : int
        Number of leaves.
    cmap_name : str
        Matplotlib colormap name (default ``"turbo"``).

    Returns
    -------
    colors_rgba : (n, 4) ndarray
        RGBA colours in the original (pre-order) leaf index.
    order : (n,) ndarray
        Dendrogram leaf order (``leaves_list``).
    normed : (n,) ndarray
        Normalised cumulative cophenetic gap in leaf-order.
    """
    order = leaves_list(Z)
    coph = squareform(cophenet(Z))
    gaps = np.array([coph[order[i], order[i + 1]] for i in range(n - 1)])
    med = float(np.median(gaps)) or 1.0
    cumdist = np.concatenate([[0.0], np.cumsum(np.arcsinh(gaps / med))])
    normed = (cumdist - cumdist.min()) / (cumdist.max() - cumdist.min() + 1e-10)
    cmap = plt.get_cmap(cmap_name)
    colours_ordered = cmap(normed)
    colours_orig = np.zeros((n, 4))
    for leaf_pos, orig_idx in enumerate(order):
        colours_orig[orig_idx] = colours_ordered[leaf_pos]
    return colours_orig, order, normed
