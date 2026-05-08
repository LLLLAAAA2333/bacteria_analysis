"""Compatibility wrapper for neural feature helpers."""

from bacteria_analysis.features.neural import (
    RdmResult,
    build_neural_rdm,
    build_stimulus_prototypes,
    build_trial_feature_matrix,
)

__all__ = [
    "RdmResult",
    "build_neural_rdm",
    "build_stimulus_prototypes",
    "build_trial_feature_matrix",
]
