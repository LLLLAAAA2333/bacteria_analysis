"""Public I/O API — dataset loading, result containers, and explicit saving."""

from bacteria_analysis._data_loaders import read_metabolite_matrix
from bacteria_analysis._analysis_dataset_impl import (
    AnalysisDataset,
    AnchorDataset,
    build_analysis_dataset,
    build_anchor_dataset,
)
from bacteria_analysis._analysis_results_impl import AnalysisResult, save_analysis_result

__all__ = [
    "AnalysisDataset",
    "AnalysisResult",
    "AnchorDataset",
    "build_analysis_dataset",
    "build_anchor_dataset",
    "read_metabolite_matrix",
    "save_analysis_result",
]
