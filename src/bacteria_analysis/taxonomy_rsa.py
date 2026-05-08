"""Compatibility wrapper for taxonomy-class RSA helpers."""

from bacteria_analysis.features.taxonomy import (
    ClassCandidate,
    avg_rank,
    build_final_shortlist,
    class_chemical_rdm_similarity,
    class_observed_scores,
    class_vs_full_chemical_similarity,
    fixed_class_permutations,
    full_search_permutation,
    pearson,
    reselection_stability,
    spearman,
    symmetric_rank_matrix,
)

__all__ = [
    "ClassCandidate",
    "avg_rank",
    "build_final_shortlist",
    "class_chemical_rdm_similarity",
    "class_observed_scores",
    "class_vs_full_chemical_similarity",
    "fixed_class_permutations",
    "full_search_permutation",
    "pearson",
    "reselection_stability",
    "spearman",
    "symmetric_rank_matrix",
]
