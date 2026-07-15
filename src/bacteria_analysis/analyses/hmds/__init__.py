"""Hyperbolic multidimensional scaling (HMDS) analysis module.

Provides reusable HMDS fitting, distance utilities, geometry transforms,
Ward-dendrogram colormaps, and visualisation functions.

Typical usage::

    import pandas as pd
    from bacteria_analysis.analyses.hmds import (
        fit_hyperbolic_mds,
        chord_from_linear,
        clean_distance_matrix,
        plot_shepard,
        plot_poincare_2d,
    )

    rdm = pd.read_csv("neural_rdm.csv", index_col=0)
    distance = chord_from_linear(clean_distance_matrix(rdm))
    result = fit_hyperbolic_mds(distance, dim=2, seed=42)
    plot_shepard(result, distance, "shepard.png")
"""

from bacteria_analysis.analyses.hmds._colormap import leaf_order_colormap
from bacteria_analysis.analyses.hmds._distance import (
    chord_from_linear,
    clean_distance_matrix,
    normalize_to_max_two,
    preservation_metrics,
    upper_triangle,
)
from bacteria_analysis.analyses.hmds._geometry import (
    lorentz_distances,
    lorentz_to_poincare,
    recenter_poincare,
)
from bacteria_analysis.analyses.hmds._optimize import scipy_hyperbolic_mds
from bacteria_analysis.analyses.hmds._result import EmbeddingResult
from bacteria_analysis.analyses.hmds.core import (
    fit_hyperbolic_mds,
    radius_summary,
)
from bacteria_analysis.analyses.hmds.plots import (
    plot_poincare_2d,
    plot_poincare_2d_html,
    plot_poincare_3d_html,
    plot_shepard,
)

__all__ = [
    # result
    "EmbeddingResult",
    # geometry
    "lorentz_distances",
    "lorentz_to_poincare",
    "recenter_poincare",
    # optimisation
    "scipy_hyperbolic_mds",
    # distance
    "chord_from_linear",
    "clean_distance_matrix",
    "normalize_to_max_two",
    "upper_triangle",
    "preservation_metrics",
    # colormap
    "leaf_order_colormap",
    # pipeline
    "fit_hyperbolic_mds",
    "radius_summary",
    # plots
    "plot_shepard",
    "plot_poincare_2d",
    "plot_poincare_2d_html",
    "plot_poincare_3d_html",
]
