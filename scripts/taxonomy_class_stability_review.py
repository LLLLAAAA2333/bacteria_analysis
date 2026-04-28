"""Figure-first taxonomy class stability review for neural-chemical RSA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bacteria_analysis.model_space import read_metabolite_matrix
from plot_biological_subspace_rdm_panel import build_chemical_rdm, build_stimulus_mapping, load_taxonomy_qc


BASE_RESULTS = Path("results/202604_without_20260331")
DEFAULT_PREPROCESS_ROOT = Path("data/202604/202604_preprocess_without_20260331")
DEFAULT_MATRIX_PATH = Path("data/matrix.xlsx")
DEFAULT_RAW_METADATA_PATH = Path("data/metabolism_raw_data.xlsx")
DEFAULT_RESPONSE_WINDOW_RDM = (
    BASE_RESULTS / "neural_median_peak_review_lr_merge" / "neural_rdm__response_window__median_flattened.parquet"
)
DEFAULT_FULL_TRAJECTORY_RDM = (
    BASE_RESULTS / "neural_median_peak_review_lr_merge" / "neural_rdm__full_trajectory__median_flattened.parquet"
)
DEFAULT_OUTPUT_ROOT = BASE_RESULTS / "taxonomy_class_stability_review"
DEFAULT_DATE_CONTROLLED_FIGURES_DIR = BASE_RESULTS / "date_controlled_rsa_review" / "figures"
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown"}


@dataclass(frozen=True)
class ClassCandidate:
    model_id: str
    taxonomy_level: str
    category: str
    metabolites: tuple[str, ...]
    chemical: pd.DataFrame
    primary_rank_matrix: np.ndarray
    primary_observed_rsa: float
    full_trajectory_rsa: float
    n_pairs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--raw-metadata-path", type=Path, default=DEFAULT_RAW_METADATA_PATH)
    parser.add_argument("--response-window-rdm", type=Path, default=DEFAULT_RESPONSE_WINDOW_RDM)
    parser.add_argument("--full-trajectory-rdm", type=Path, default=DEFAULT_FULL_TRAJECTORY_RDM)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date-controlled-figures-dir", type=Path, default=DEFAULT_DATE_CONTROLLED_FIGURES_DIR)
    parser.add_argument("--taxonomy-level", default="Class", choices=("SuperClass", "Class", "SubClass"))
    parser.add_argument("--qc-threshold", type=float, default=0.2)
    parser.add_argument("--min-features", type=int, default=3)
    parser.add_argument("--fixed-permutations", type=int, default=2000)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--search-permutations", type=int, default=2000)
    parser.add_argument("--resample-fraction", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--figure-class-limit", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    diagnostics_dir = output_root / "diagnostics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    matrix = read_metabolite_matrix(args.matrix_path)
    stimulus_sample_map = build_stimulus_mapping(args.preprocess_root, matrix)
    date_map = load_date_map(args.preprocess_root)
    primary_neural = load_rdm(args.response_window_rdm)
    full_neural = load_rdm(args.full_trajectory_rdm)
    labels = shared_labels(primary_neural, stimulus_sample_map["stimulus"].astype(str).tolist())
    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    primary_values = matrix_values(primary_neural, labels)[upper_i, upper_j]
    primary_ranks = avg_rank(primary_values)
    full_values = matrix_values(full_neural, labels)[upper_i, upper_j]

    candidates = build_candidates(
        args=args,
        matrix=matrix,
        stimulus_sample_map=stimulus_sample_map,
        primary_neural=primary_neural,
        full_neural=full_neural,
        labels=labels,
        primary_ranks=primary_ranks,
        full_values=full_values,
        upper_i=upper_i,
        upper_j=upper_j,
    )
    if not candidates:
        raise ValueError(f"No eligible {args.taxonomy_level} candidates were found")
    full_chemical = build_full_chemical_rdm(args, matrix, stimulus_sample_map, labels)

    observed_scores = class_observed_scores(candidates)
    fixed_summary, fixed_null = fixed_class_permutations(
        candidates=candidates,
        labels=labels,
        neural_ranks=primary_ranks,
        upper_i=upper_i,
        upper_j=upper_j,
        n_permutations=int(args.fixed_permutations),
        rng=rng,
    )
    reselection_runs, reselection_summary = reselection_stability(
        candidates=candidates,
        primary_neural=primary_neural,
        labels=labels,
        date_map=date_map,
        n_resamples=int(args.resamples),
        resample_fraction=float(args.resample_fraction),
        top_k=int(args.top_k),
        rng=rng,
    )
    full_search_summary, full_search_null = full_search_permutation(
        candidates=candidates,
        labels=labels,
        neural_ranks=primary_ranks,
        upper_i=upper_i,
        upper_j=upper_j,
        n_permutations=int(args.search_permutations),
        rng=rng,
    )
    final_shortlist = build_final_shortlist(observed_scores, fixed_summary, reselection_summary, full_search_summary)
    class_similarity, class_similarity_summary, top_class_similarity = class_chemical_rdm_similarity(
        candidates=candidates,
        labels=labels,
        upper_i=upper_i,
        upper_j=upper_j,
    )
    class_full_similarity, class_full_similarity_summary = class_vs_full_chemical_similarity(
        candidates=candidates,
        full_chemical=full_chemical,
        labels=labels,
        upper_i=upper_i,
        upper_j=upper_j,
    )

    observed_scores.to_csv(tables_dir / "class_observed_scores.csv", index=False)
    class_similarity.to_csv(tables_dir / "class_chemical_rdm_pairwise_similarity.csv", index=False)
    class_similarity_summary.to_csv(tables_dir / "class_chemical_rdm_pairwise_similarity_summary.csv", index=False)
    top_class_similarity.to_csv(tables_dir / "top_class_vs_other_class_chemical_rdm_similarity.csv", index=False)
    class_full_similarity.to_csv(tables_dir / "class_vs_full_chemical_rdm_similarity.csv", index=False)
    class_full_similarity_summary.to_csv(tables_dir / "class_vs_full_chemical_rdm_similarity_summary.csv", index=False)
    fixed_summary.to_csv(tables_dir / "fixed_class_permutation_summary.csv", index=False)
    fixed_null.to_csv(tables_dir / "fixed_class_permutation_null_values.csv", index=False)
    reselection_runs.to_csv(tables_dir / "reselection_runs.csv", index=False)
    reselection_summary.to_csv(tables_dir / "reselection_stability_summary.csv", index=False)
    full_search_summary.to_csv(tables_dir / "full_search_permutation_summary.csv", index=False)
    full_search_null.to_csv(tables_dir / "full_search_permutation_null_values.csv", index=False)
    final_shortlist.to_csv(tables_dir / "final_class_shortlist.csv", index=False)

    plot_fixed_class_permutation(fixed_summary, figures_dir / "01_fixed_class_permutation.png", class_limit=int(args.figure_class_limit))
    plot_reselection_stability(reselection_summary, observed_scores, figures_dir / "02_reselection_stability.png", class_limit=int(args.figure_class_limit))
    for stale_figure in (
        figures_dir / "03_full_search_permutation.png",
        figures_dir / "03_taxonomy_class_stability_summary.png",
        figures_dir / "04_taxonomy_class_stability_summary.png",
        figures_dir / "05_class_chemical_rdm_similarity_distribution.png",
    ):
        stale_figure.unlink(missing_ok=True)
    plot_full_search_permutation(
        full_search_summary,
        full_search_null,
        reselection_summary,
        diagnostics_dir / "full_search_permutation_diagnostic.png",
    )
    rdm_comparison_path = figures_dir / "03_top_class_rdm_comparison.png"
    plot_top_class_rdm_comparison(
        primary_neural=primary_neural,
        full_chemical=full_chemical,
        candidate=candidates[0],
        labels=labels,
        stimulus_sample_map=stimulus_sample_map,
        output_path=rdm_comparison_path,
    )
    date_controlled_rdm_path = (
        Path(args.date_controlled_figures_dir)
        / "neural_chemical_rdm_foundation__response_window_full_vs_top_class_rdms.png"
    )
    date_controlled_rdm_path.parent.mkdir(parents=True, exist_ok=True)
    if rdm_comparison_path.resolve() != date_controlled_rdm_path.resolve():
        shutil.copyfile(rdm_comparison_path, date_controlled_rdm_path)
    plot_summary_scorecard(
        final_shortlist,
        figures_dir / "04_taxonomy_class_stability_summary.png",
        class_limit=min(int(args.figure_class_limit), 18),
    )
    plot_class_chemical_rdm_similarity(
        pairwise=class_similarity,
        top_class_similarity=top_class_similarity,
        output_path=figures_dir / "05_class_chemical_rdm_similarity_matrix.png",
    )
    plot_class_vs_full_chemical_similarity(
        similarity=class_full_similarity,
        output_path=figures_dir / "06_class_vs_full_chemical_rdm_similarity.png",
    )
    write_run_summary(
        output_root=output_root,
        args=args,
        labels=labels,
        observed_scores=observed_scores,
        fixed_summary=fixed_summary,
        reselection_summary=reselection_summary,
        full_search_summary=full_search_summary,
        final_shortlist=final_shortlist,
        date_controlled_rdm_path=date_controlled_rdm_path,
        class_similarity_summary=class_similarity_summary,
        class_full_similarity_summary=class_full_similarity_summary,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.min_features < 1:
        raise ValueError("min-features must be >= 1")
    if args.fixed_permutations < 1:
        raise ValueError("fixed-permutations must be >= 1")
    if args.resamples < 1:
        raise ValueError("resamples must be >= 1")
    if args.search_permutations < 1:
        raise ValueError("search-permutations must be >= 1")
    if not 0 < args.resample_fraction <= 1:
        raise ValueError("resample-fraction must be in (0, 1]")
    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")


def load_rdm(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "stimulus_row" not in frame.columns:
        raise ValueError(f"{path} is missing stimulus_row")
    matrix = frame.set_index("stimulus_row")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


def load_date_map(preprocess_root: Path) -> dict[str, str]:
    metadata = pd.read_parquet(preprocess_root / "trial_level" / "trial_metadata.parquet")
    support = metadata.loc[:, ["stimulus", "date"]].drop_duplicates().copy()
    counts = support.groupby("stimulus")["date"].nunique()
    repeated = counts.loc[counts > 1]
    if not repeated.empty:
        raise ValueError(f"Expected one date per stimulus in this filtered batch, got: {repeated.index.tolist()}")
    return dict(zip(support["stimulus"].astype(str), support["date"].astype(str)))


def shared_labels(neural: pd.DataFrame, candidate_labels: list[str]) -> list[str]:
    candidate_set = set(candidate_labels)
    labels = [label for label in neural.index.astype(str) if label in candidate_set and label in neural.columns]
    if len(labels) < 3:
        raise ValueError(f"Need at least 3 shared stimuli, found {len(labels)}")
    return labels


def build_candidates(
    *,
    args: argparse.Namespace,
    matrix: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame,
    primary_neural: pd.DataFrame,
    full_neural: pd.DataFrame,
    labels: list[str],
    primary_ranks: np.ndarray,
    full_values: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
) -> list[ClassCandidate]:
    taxonomy = load_taxonomy_qc(args.raw_metadata_path)
    retained = taxonomy.loc[
        taxonomy["normalized_name"].astype(str).isin(matrix.columns.astype(str))
        & taxonomy["QCRSD"].le(float(args.qc_threshold))
    ].copy()
    retained[args.taxonomy_level] = retained[args.taxonomy_level].astype(str).str.strip()
    retained = retained.loc[~retained[args.taxonomy_level].str.lower().isin(MISSING_TOKENS)].copy()

    candidates: list[ClassCandidate] = []
    for category, group in retained.groupby(args.taxonomy_level, sort=True):
        metabolites = tuple(group["normalized_name"].astype(str).drop_duplicates().tolist())
        if len(metabolites) < int(args.min_features):
            continue
        chemical = build_chemical_rdm(matrix, stimulus_sample_map, list(metabolites))
        chemical_square = square_rdm(chemical).loc[labels, labels]
        chemical_values = chemical_square.to_numpy(float)[upper_i, upper_j]
        chemical_ranks = avg_rank(chemical_values)
        primary_score = pearson(primary_ranks, chemical_ranks)
        full_score = spearman(full_values, chemical_values)
        rank_matrix = symmetric_rank_matrix(len(labels), upper_i, upper_j, chemical_ranks)
        candidates.append(
            ClassCandidate(
                model_id=f"{args.taxonomy_level}::{category}",
                taxonomy_level=str(args.taxonomy_level),
                category=str(category),
                metabolites=metabolites,
                chemical=chemical_square,
                primary_rank_matrix=rank_matrix,
                primary_observed_rsa=primary_score,
                full_trajectory_rsa=full_score,
                n_pairs=int(len(chemical_values)),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.primary_observed_rsa, reverse=True)


def build_full_chemical_rdm(
    args: argparse.Namespace,
    matrix: pd.DataFrame,
    stimulus_sample_map: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    taxonomy = load_taxonomy_qc(args.raw_metadata_path)
    metabolites = (
        taxonomy.loc[
            taxonomy["normalized_name"].astype(str).isin(matrix.columns.astype(str))
            & taxonomy["QCRSD"].le(float(args.qc_threshold)),
            "normalized_name",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if len(metabolites) < int(args.min_features):
        raise ValueError(f"Full chemical RDM has too few retained metabolites: {len(metabolites)}")
    return square_rdm(build_chemical_rdm(matrix, stimulus_sample_map, metabolites)).loc[labels, labels]


def square_rdm(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.set_index("stimulus_row") if "stimulus_row" in frame.columns else frame.copy()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


def matrix_values(matrix: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return matrix.loc[labels, labels].to_numpy(dtype=float)


def class_observed_scores(candidates: list[ClassCandidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "model_id": candidate.model_id,
                "taxonomy_level": candidate.taxonomy_level,
                "category": candidate.category,
                "n_features": len(candidate.metabolites),
                "n_pairs": candidate.n_pairs,
                "response_window_rsa": candidate.primary_observed_rsa,
                "full_trajectory_rsa": candidate.full_trajectory_rsa,
                "metabolites": " | ".join(candidate.metabolites),
            }
        )
    return pd.DataFrame(rows).sort_values("response_window_rsa", ascending=False).reset_index(drop=True)


def class_chemical_rdm_similarity(
    *,
    candidates: list[ClassCandidate],
    labels: list[str],
    upper_i: np.ndarray,
    upper_j: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectors = {
        candidate.model_id: candidate.chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
        for candidate in candidates
    }
    rows = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            rows.append(
                {
                    "left_model_id": left.model_id,
                    "left_category": left.category,
                    "left_response_window_rsa": left.primary_observed_rsa,
                    "right_model_id": right.model_id,
                    "right_category": right.category,
                    "right_response_window_rsa": right.primary_observed_rsa,
                    "chemical_rdm_rsa": spearman(vectors[left.model_id], vectors[right.model_id]),
                    "left_n_features": len(left.metabolites),
                    "right_n_features": len(right.metabolites),
                }
            )
    pairwise = pd.DataFrame(rows).sort_values("chemical_rdm_rsa", ascending=False).reset_index(drop=True)
    values = pairwise["chemical_rdm_rsa"].to_numpy(float)
    finite = values[np.isfinite(values)]
    summary = pd.DataFrame(
        [
            {
                "n_classes": len(candidates),
                "n_class_pairs": len(pairwise),
                "mean": nan_stat(np.mean, finite),
                "std": nan_stat(np.std, finite),
                "min": float(np.min(finite)) if finite.size else np.nan,
                "q05": nan_quantile(finite, 0.05),
                "q25": nan_quantile(finite, 0.25),
                "median": nan_quantile(finite, 0.50),
                "q75": nan_quantile(finite, 0.75),
                "q95": nan_quantile(finite, 0.95),
                "max": float(np.max(finite)) if finite.size else np.nan,
            }
        ]
    )
    top_model_id = candidates[0].model_id
    top_similarity = pairwise.loc[
        pairwise["left_model_id"].eq(top_model_id) | pairwise["right_model_id"].eq(top_model_id)
    ].copy()
    top_similarity["other_category"] = np.where(
        top_similarity["left_model_id"].eq(top_model_id),
        top_similarity["right_category"],
        top_similarity["left_category"],
    )
    top_similarity["other_response_window_rsa"] = np.where(
        top_similarity["left_model_id"].eq(top_model_id),
        top_similarity["right_response_window_rsa"],
        top_similarity["left_response_window_rsa"],
    )
    top_similarity = top_similarity.sort_values("chemical_rdm_rsa", ascending=False).reset_index(drop=True)
    return pairwise, summary, top_similarity


def class_vs_full_chemical_similarity(
    *,
    candidates: list[ClassCandidate],
    full_chemical: pd.DataFrame,
    labels: list[str],
    upper_i: np.ndarray,
    upper_j: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_values = full_chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
    rows = []
    for candidate in candidates:
        class_values = candidate.chemical.loc[labels, labels].to_numpy(float)[upper_i, upper_j]
        rows.append(
            {
                "model_id": candidate.model_id,
                "category": candidate.category,
                "n_features": len(candidate.metabolites),
                "class_vs_full_chemical_rdm_rsa": spearman(class_values, full_values),
                "response_window_rsa": candidate.primary_observed_rsa,
                "full_trajectory_rsa": candidate.full_trajectory_rsa,
            }
        )
    similarity = pd.DataFrame(rows).sort_values("class_vs_full_chemical_rdm_rsa", ascending=False).reset_index(drop=True)
    values = similarity["class_vs_full_chemical_rdm_rsa"].to_numpy(float)
    finite = values[np.isfinite(values)]
    summary = pd.DataFrame(
        [
            {
                "n_classes": len(similarity),
                "mean": nan_stat(np.mean, finite),
                "std": nan_stat(np.std, finite),
                "min": float(np.min(finite)) if finite.size else np.nan,
                "q05": nan_quantile(finite, 0.05),
                "q25": nan_quantile(finite, 0.25),
                "median": nan_quantile(finite, 0.50),
                "q75": nan_quantile(finite, 0.75),
                "q95": nan_quantile(finite, 0.95),
                "max": float(np.max(finite)) if finite.size else np.nan,
            }
        ]
    )
    return similarity, summary


def fixed_class_permutations(
    *,
    candidates: list[ClassCandidate],
    labels: list[str],
    neural_ranks: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    null_rows = []
    for candidate_index, candidate in enumerate(candidates):
        null_values = permuted_scores(
            rank_matrix=candidate.primary_rank_matrix,
            neural_ranks=neural_ranks,
            upper_i=upper_i,
            upper_j=upper_j,
            n_labels=len(labels),
            n_permutations=n_permutations,
            rng=rng,
        )
        finite = null_values[np.isfinite(null_values)]
        observed = candidate.primary_observed_rsa
        summary_rows.append(
            {
                "model_id": candidate.model_id,
                "taxonomy_level": candidate.taxonomy_level,
                "category": candidate.category,
                "n_features": len(candidate.metabolites),
                "observed_rsa": observed,
                "n_permutations": n_permutations,
                "null_mean": nan_stat(np.mean, finite),
                "null_q95": nan_quantile(finite, 0.95),
                "null_q99": nan_quantile(finite, 0.99),
                "null_max": float(np.max(finite)) if finite.size else np.nan,
                "n_ge_observed": int(np.sum(finite >= observed)) if np.isfinite(observed) else 0,
                "p_one_sided_ge": empirical_p(observed, finite),
                "observed_percentile": percentile(observed, finite),
            }
        )
        null_rows.append(
            pd.DataFrame(
                {
                    "model_id": candidate.model_id,
                    "category": candidate.category,
                    "candidate_index": candidate_index,
                    "iteration": np.arange(n_permutations),
                    "rsa_similarity": null_values,
                }
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary["p_fdr_bh"] = benjamini_hochberg(summary["p_one_sided_ge"].to_numpy(float))
    summary = summary.sort_values("observed_rsa", ascending=False).reset_index(drop=True)
    null = pd.concat(null_rows, ignore_index=True)
    return summary, null


def reselection_stability(
    *,
    candidates: list[ClassCandidate],
    primary_neural: pd.DataFrame,
    labels: list[str],
    date_map: dict[str, str],
    n_resamples: int,
    resample_fraction: float,
    top_k: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for resample_index in range(n_resamples):
        subset_labels = date_stratified_resample(labels, date_map, resample_fraction, rng)
        subset_scores = score_candidates_on_subset(candidates, primary_neural, subset_labels)
        subset_scores = subset_scores.sort_values("rsa_similarity", ascending=False).reset_index(drop=True)
        for rank_index, row in subset_scores.head(top_k).iterrows():
            run_rows.append(
                {
                    "resample_index": resample_index,
                    "rank": int(rank_index + 1),
                    "model_id": row["model_id"],
                    "category": row["category"],
                    "rsa_similarity": float(row["rsa_similarity"]),
                    "n_stimuli": len(subset_labels),
                    "n_pairs": int(row["n_pairs"]),
                    "stimuli": ";".join(subset_labels),
                }
            )
    runs = pd.DataFrame(run_rows)
    summary_rows = []
    total = max(1, n_resamples)
    for candidate in candidates:
        subset = runs.loc[runs["model_id"].eq(candidate.model_id)]
        ranks = subset["rank"].to_numpy(int) if not subset.empty else np.array([], dtype=int)
        summary_rows.append(
            {
                "model_id": candidate.model_id,
                "category": candidate.category,
                "n_features": len(candidate.metabolites),
                "top1_count": int(np.sum(ranks <= 1)),
                "top3_count": int(np.sum(ranks <= min(3, top_k))),
                "top5_count": int(np.sum(ranks <= min(5, top_k))),
                "top1_frequency": float(np.sum(ranks <= 1) / total),
                "top3_frequency": float(np.sum(ranks <= min(3, top_k)) / total),
                "top5_frequency": float(np.sum(ranks <= min(5, top_k)) / total),
                "mean_selected_rank": float(np.mean(ranks)) if ranks.size else np.nan,
                "mean_selected_rsa": float(subset["rsa_similarity"].mean()) if not subset.empty else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["top3_frequency", "top1_frequency", "mean_selected_rsa"],
        ascending=False,
    ).reset_index(drop=True)
    return runs, summary


def full_search_permutation(
    *,
    candidates: list[ClassCandidate],
    labels: list[str],
    neural_ranks: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_best = max(candidate.primary_observed_rsa for candidate in candidates)
    observed_best_candidate = max(candidates, key=lambda candidate: candidate.primary_observed_rsa)
    null_rows = []
    for iteration in range(n_permutations):
        permutation = rng.permutation(len(labels))
        best_score = -np.inf
        best_candidate = candidates[0]
        for candidate in candidates:
            values = candidate.primary_rank_matrix[np.ix_(permutation, permutation)][upper_i, upper_j]
            score = pearson(neural_ranks, values)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_candidate = candidate
        null_rows.append(
            {
                "iteration": iteration,
                "best_null_rsa": best_score,
                "null_winning_model_id": best_candidate.model_id,
                "null_winning_category": best_candidate.category,
            }
        )
    null = pd.DataFrame(null_rows)
    finite = null["best_null_rsa"].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    summary = pd.DataFrame(
        [
            {
                "observed_best_model_id": observed_best_candidate.model_id,
                "observed_best_category": observed_best_candidate.category,
                "observed_best_rsa": observed_best,
                "n_candidates": len(candidates),
                "n_permutations": n_permutations,
                "best_null_mean": nan_stat(np.mean, finite),
                "best_null_q95": nan_quantile(finite, 0.95),
                "best_null_q99": nan_quantile(finite, 0.99),
                "best_null_max": float(np.max(finite)) if finite.size else np.nan,
                "n_ge_observed": int(np.sum(finite >= observed_best)) if np.isfinite(observed_best) else 0,
                "search_corrected_p": empirical_p(observed_best, finite),
                "observed_percentile": percentile(observed_best, finite),
            }
        ]
    )
    return summary, null


def build_final_shortlist(
    observed: pd.DataFrame,
    fixed: pd.DataFrame,
    reselection: pd.DataFrame,
    search_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = observed.merge(
        fixed.loc[:, ["model_id", "observed_percentile", "p_one_sided_ge", "p_fdr_bh", "null_q95", "null_q99"]],
        on="model_id",
        how="left",
    ).merge(
        reselection.loc[:, ["model_id", "top1_frequency", "top3_frequency", "top5_frequency", "mean_selected_rank"]],
        on="model_id",
        how="left",
    )
    search_p = float(search_summary["search_corrected_p"].iloc[0])
    observed_best = str(search_summary["observed_best_model_id"].iloc[0])
    merged["fixed_signal_pass"] = merged["response_window_rsa"].to_numpy(float) > merged["null_q95"].to_numpy(float)
    merged["reselection_preferred"] = merged["top3_frequency"].fillna(0.0).ge(0.20)
    merged["is_observed_best"] = merged["model_id"].eq(observed_best)
    merged["search_corrected_p_for_best"] = np.where(merged["is_observed_best"], search_p, np.nan)
    shortlist = merged.loc[
        merged["response_window_rsa"].gt(0)
        & (merged["fixed_signal_pass"] | merged["reselection_preferred"] | merged["is_observed_best"])
    ].copy()
    if shortlist.empty:
        shortlist = merged.head(12).copy()
    return shortlist.sort_values(
        ["reselection_preferred", "fixed_signal_pass", "top3_frequency", "response_window_rsa"],
        ascending=False,
    ).reset_index(drop=True)


def score_candidates_on_subset(
    candidates: list[ClassCandidate],
    neural: pd.DataFrame,
    subset_labels: list[str],
) -> pd.DataFrame:
    rows = []
    upper_i, upper_j = np.triu_indices(len(subset_labels), k=1)
    neural_values = neural.loc[subset_labels, subset_labels].to_numpy(float)[upper_i, upper_j]
    for candidate in candidates:
        chemical_values = candidate.chemical.loc[subset_labels, subset_labels].to_numpy(float)[upper_i, upper_j]
        rows.append(
            {
                "model_id": candidate.model_id,
                "category": candidate.category,
                "rsa_similarity": spearman(neural_values, chemical_values),
                "n_pairs": len(neural_values),
            }
        )
    return pd.DataFrame(rows)


def date_stratified_resample(
    labels: list[str],
    date_map: dict[str, str],
    fraction_value: float,
    rng: np.random.Generator,
) -> list[str]:
    by_date: dict[str, list[str]] = {}
    for label in labels:
        by_date.setdefault(date_map.get(label, "unknown"), []).append(label)
    selected: list[str] = []
    for date in sorted(by_date):
        date_labels = sorted(by_date[date])
        size = max(1, int(round(len(date_labels) * fraction_value)))
        size = min(len(date_labels), size)
        selected.extend(rng.choice(date_labels, size=size, replace=False).astype(str).tolist())
    if len(selected) < 3:
        selected = rng.choice(labels, size=min(len(labels), 3), replace=False).astype(str).tolist()
    return sorted(selected)


def permuted_scores(
    *,
    rank_matrix: np.ndarray,
    neural_ranks: np.ndarray,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    n_labels: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    null_values = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        permutation = rng.permutation(n_labels)
        values = rank_matrix[np.ix_(permutation, permutation)][upper_i, upper_j]
        null_values[iteration] = pearson(neural_ranks, values)
    return null_values


def symmetric_rank_matrix(
    n_labels: int,
    upper_i: np.ndarray,
    upper_j: np.ndarray,
    ranks: np.ndarray,
) -> np.ndarray:
    matrix = np.full((n_labels, n_labels), np.nan, dtype=float)
    matrix[upper_i, upper_j] = ranks
    matrix[upper_j, upper_i] = ranks
    return matrix


def spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return np.nan
    return pearson(avg_rank(left[mask]), avg_rank(right[mask]))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return np.nan
    left = left[mask] - np.mean(left[mask])
    right = right[mask] - np.mean(right[mask])
    denominator = np.sqrt(np.sum(left * left) * np.sum(right * right))
    if denominator == 0:
        return np.nan
    return float(np.sum(left * right) / denominator)


def avg_rank(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float), copy=False).rank(method="average").to_numpy(float)


def empirical_p(observed: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        return np.nan
    return float((1 + np.sum(values >= observed)) / (values.size + 1))


def percentile(observed: float, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or values.size == 0:
        return np.nan
    return float(np.mean(values <= observed) * 100.0)


def nan_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else np.nan


def nan_stat(func: object, values: np.ndarray) -> float:
    return float(func(values)) if len(values) else np.nan


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return adjusted
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    scale = finite_values.size / np.arange(1, finite_values.size + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    finite_indices = np.flatnonzero(finite_mask)
    adjusted[finite_indices[order]] = ranked_adjusted
    return adjusted


def plot_fixed_class_permutation(summary: pd.DataFrame, output_path: Path, *, class_limit: int) -> None:
    plot_frame = summary.sort_values("observed_rsa", ascending=False).head(class_limit).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9.2, max(5.0, 0.28 * len(plot_frame) + 1.6)), constrained_layout=True)
    y = np.arange(len(plot_frame))
    ax.hlines(y, plot_frame["null_q95"], plot_frame["null_q99"], color="#8A8A8A", linewidth=2.0, label="fixed null q95-q99")
    points = ax.scatter(
        plot_frame["observed_rsa"],
        y,
        c=plot_frame["observed_percentile"],
        cmap="viridis",
        vmin=50,
        vmax=100,
        s=42,
        edgecolor="#222222",
        linewidth=0.35,
        label="observed RSA",
    )
    ax.axvline(0.0, color="#C9C9C9", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["category"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("response-window RSA")
    ax.set_title("Fixed-class permutation: conditional class signal", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("observed percentile within fixed-class null")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_reselection_stability(
    stability: pd.DataFrame,
    observed: pd.DataFrame,
    output_path: Path,
    *,
    class_limit: int,
) -> None:
    merged = stability.merge(observed.loc[:, ["model_id", "response_window_rsa"]], on="model_id", how="left")
    plot_frame = merged.sort_values(["top3_frequency", "response_window_rsa"], ascending=False).head(class_limit)
    plot_frame = plot_frame.iloc[::-1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, max(5.0, 0.30 * len(plot_frame) + 1.7)), constrained_layout=True)
    y = np.arange(len(plot_frame))
    y_labels = plot_frame["category"].astype(str).tolist()
    axes[0].barh(y - 0.18, plot_frame["top1_frequency"], height=0.16, color="#2F6B7E", label="top 1")
    axes[0].barh(y, plot_frame["top3_frequency"], height=0.16, color="#D98C35", label="top 3")
    axes[0].barh(y + 0.18, plot_frame["top5_frequency"], height=0.16, color="#759E54", label="top 5")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(y_labels, fontsize=8)
    axes[0].set_ylim(-0.5, len(plot_frame) - 0.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("reselection frequency")
    axes[0].set_title("Class recurrence after resampling")
    axes[0].legend(fontsize=8)

    heat_values = plot_frame.loc[:, ["top1_frequency", "top3_frequency", "top5_frequency"]].to_numpy(float)
    image = axes[1].imshow(
        heat_values,
        aspect="auto",
        cmap="YlGnBu",
        vmin=0,
        vmax=max(0.2, np.nanmax(heat_values)),
        origin="lower",
    )
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["top 1", "top 3", "top 5"])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(y_labels, fontsize=8)
    axes[1].set_ylim(-0.5, len(plot_frame) - 0.5)
    axes[1].tick_params(axis="y", length=0)
    axes[1].set_title("Reselection scorecard")
    for row in range(heat_values.shape[0]):
        for col in range(heat_values.shape[1]):
            axes[1].text(col, row, f"{heat_values[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[1], pad=0.02, label="frequency")
    fig.suptitle("Reselection stability: rerun class selection inside each resample", fontsize=12)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_full_search_permutation(
    summary: pd.DataFrame,
    null: pd.DataFrame,
    stability: pd.DataFrame,
    output_path: Path,
) -> None:
    observed = float(summary["observed_best_rsa"].iloc[0])
    search_p = float(summary["search_corrected_p"].iloc[0])
    q95 = float(summary["best_null_q95"].iloc[0])
    values = null["best_null_rsa"].to_numpy(float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(7.8, 5.1), constrained_layout=True)
    ax.hist(values, bins=60, color="#8064A2", edgecolor="#8064A2", linewidth=0.25, alpha=0.9)
    ax.axvline(observed, color="#F0A12B", linewidth=2.0, label=f"observed best = {observed:.3f}")
    ax.axvline(q95, color="#4E4E4E", linestyle="--", linewidth=1.5, label=f"null q95 = {q95:.3f}")
    if not stability.empty:
        stable = stability.sort_values(["top3_frequency", "top1_frequency"], ascending=False).iloc[0]
        ax.text(
            0.02,
            0.96,
            f"Most reselected class:\n{stable['category']}\ntop3 freq={float(stable['top3_frequency']):.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
        )
    ax.set_xlabel("best RSA after full class search under label shuffle")
    ax.set_ylabel("permutation count")
    ax.set_title(f"Full-search permutation: search-corrected null (p={search_p:.4f})")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_top_class_rdm_comparison(
    *,
    primary_neural: pd.DataFrame,
    full_chemical: pd.DataFrame,
    candidate: ClassCandidate,
    labels: list[str],
    stimulus_sample_map: pd.DataFrame,
    output_path: Path,
) -> None:
    display_order = clustered_order(primary_neural, labels)
    display_labels = stimulus_display_labels(display_order, stimulus_sample_map)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#FFFFFF")
    fig = plt.figure(figsize=(19.8, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 6, width_ratios=[1.0, 0.04, 1.0, 0.04, 1.0, 0.04])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[0, 4])]
    colorbar_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[0, 5])]

    panels = [
        (
            primary_neural.loc[display_order, display_order],
            "Neural RDM\nresponse window",
            "correlation distance",
        ),
        (
            full_chemical.loc[display_order, display_order],
            "Chemical RDM\nall QC20 metabolites",
            "log2 Euclidean distance",
        ),
        (
            candidate.chemical.loc[display_order, display_order],
            f"Chemical RDM\n{candidate.category}",
            "log2 Euclidean distance",
        ),
    ]

    for axis, colorbar_axis, (matrix, title, colorbar_label) in zip(axes, colorbar_axes, panels):
        display = mask_diagonal(matrix)
        values = display.to_numpy(float)
        finite = values[np.isfinite(values)]
        image = axis.imshow(
            values,
            cmap=cmap,
            vmin=float(np.min(finite)) if finite.size else None,
            vmax=float(np.max(finite)) if finite.size else None,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("stimuli")
        axis.set_ylabel("stimuli")
        apply_all_ticks(axis, display_labels)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Neural, full chemical, and top class RDMs\n"
        f"ordered by response-window neural clustering | top class: {candidate.category} "
        f"({len(candidate.metabolites)} metabolites)",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def clustered_order(matrix: pd.DataFrame, labels: list[str]) -> list[str]:
    values = matrix.loc[labels, labels].to_numpy(float)
    if len(labels) < 3:
        return labels
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return labels
    fill_value = float(np.nanmedian(finite))
    values = np.where(np.isfinite(values), values, fill_value)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    if np.allclose(values, 0.0):
        return labels
    tree = linkage(squareform(values, checks=False), method="average", optimal_ordering=True)
    return [labels[index] for index in leaves_list(tree)]


def mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.apply(pd.to_numeric, errors="coerce").copy()
    values = display.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


def stimulus_display_labels(labels: list[str], stimulus_sample_map: pd.DataFrame) -> list[str]:
    sample_map = dict(
        zip(
            stimulus_sample_map["stimulus"].astype(str),
            stimulus_sample_map["sample_id"].fillna("").astype(str),
        )
    )
    return [sample_map.get(label, label) or label for label in labels]


def apply_all_ticks(axis: plt.Axes, labels: list[str]) -> None:
    n_labels = len(labels)
    tick_indices = np.arange(n_labels)
    axis.set_xticks(tick_indices)
    axis.set_yticks(tick_indices)
    axis.set_xticklabels(labels, rotation=90, fontsize=4.5)
    axis.set_yticklabels(labels, fontsize=4.5)


def plot_summary_scorecard(shortlist: pd.DataFrame, output_path: Path, *, class_limit: int) -> None:
    plot_frame = shortlist.head(class_limit).iloc[::-1].copy()
    if plot_frame.empty:
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "No shortlisted classes", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return

    observed = plot_frame["response_window_rsa"].to_numpy(float)
    observed_scaled = (observed - np.nanmin(observed)) / max(1e-9, np.nanmax(observed) - np.nanmin(observed))
    values = np.column_stack(
        [
            observed_scaled,
            plot_frame["observed_percentile"].to_numpy(float) / 100.0,
            plot_frame["top3_frequency"].fillna(0.0).to_numpy(float),
            plot_frame["fixed_signal_pass"].astype(float).to_numpy(),
        ]
    )
    fig, ax = plt.subplots(figsize=(9.6, max(4.2, 0.36 * len(plot_frame) + 1.7)), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(plot_frame)))
    ax.set_yticklabels(plot_frame["category"].astype(str).tolist(), fontsize=8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["observed RSA", "fixed percentile", "top3 reselection", "fixed q95 pass"], rotation=25, ha="right")
    for row in range(len(plot_frame)):
        annotations = [
            f"{plot_frame['response_window_rsa'].iloc[row]:.3f}",
            f"{plot_frame['observed_percentile'].iloc[row]:.1f}",
            f"{plot_frame['top3_frequency'].iloc[row]:.2f}",
            "yes" if bool(plot_frame["fixed_signal_pass"].iloc[row]) else "no",
        ]
        for col, text in enumerate(annotations):
            color = "white" if values[row, col] < 0.55 else "black"
            ax.text(col, row, text, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, pad=0.02, label="scaled evidence")
    ax.set_title("Exploratory taxonomy class stability summary")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_class_chemical_rdm_similarity(
    *,
    pairwise: pd.DataFrame,
    top_class_similarity: pd.DataFrame,
    output_path: Path,
) -> None:
    matrix = class_similarity_matrix(pairwise)
    ordered_classes = clustered_similarity_order(matrix)
    display = matrix.loc[ordered_classes, ordered_classes]
    values = display.to_numpy(float)
    off_diagonal = values[~np.eye(len(display), dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    vmin = min(-0.10, float(np.nanmin(finite)) if finite.size else -0.10)
    vmax = max(0.85, float(np.nanmax(finite)) if finite.size else 0.85)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, max(6.4, 0.30 * len(display) + 1.2)),
        gridspec_kw={"width_ratios": [1.18, 0.82]},
        constrained_layout=True,
    )
    image = axes[0].imshow(values, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_xticks(np.arange(len(display)))
    axes[0].set_yticks(np.arange(len(display)))
    axes[0].set_xticklabels(ordered_classes, rotation=90, fontsize=6)
    axes[0].set_yticklabels(ordered_classes, fontsize=6)
    axes[0].set_xlabel("taxonomy class")
    axes[0].set_ylabel("taxonomy class")
    axes[0].set_title("Class-to-class chemical RDM RSA")
    axes[0].tick_params(length=0)
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    if "Purine nucleosides" in ordered_classes:
        index = ordered_classes.index("Purine nucleosides")
        for offset in (-0.5, 0.5):
            axes[0].axhline(index + offset, color="#00A6A6", linewidth=1.4)
            axes[0].axvline(index + offset, color="#00A6A6", linewidth=1.4)
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    colorbar.set_label("chemical RDM RSA", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    top = top_class_similarity.sort_values("chemical_rdm_rsa", ascending=True)
    y = np.arange(len(top))
    colors = np.where(top["other_category"].isin(["Purine nucleotides", "Pyrimidine nucleotides"]), "#00A6A6", "#D98C35")
    axes[1].barh(y, top["chemical_rdm_rsa"], color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(top["other_category"].astype(str).tolist(), fontsize=7)
    axes[1].set_xlim(vmin - 0.02, vmax + 0.05)
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_xlabel("chemical RDM RSA vs Purine nucleosides")
    axes[1].set_title("Purine nucleosides relation to other classes")
    for row, value in enumerate(top["chemical_rdm_rsa"].to_numpy(float)):
        pad = 0.012 if value >= 0 else -0.012
        ha = "left" if value >= 0 else "right"
        axes[1].text(value + pad, row, f"{value:.2f}", va="center", ha=ha, fontsize=6.5)
    fig.suptitle("Chemical RDM similarity among taxonomy classes", fontsize=12)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def class_similarity_matrix(pairwise: pd.DataFrame) -> pd.DataFrame:
    categories = sorted(
        set(pairwise["left_category"].astype(str).tolist())
        | set(pairwise["right_category"].astype(str).tolist())
    )
    matrix = pd.DataFrame(np.nan, index=categories, columns=categories, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)
    for row in pairwise.itertuples(index=False):
        left = str(row.left_category)
        right = str(row.right_category)
        value = float(row.chemical_rdm_rsa)
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def clustered_similarity_order(matrix: pd.DataFrame) -> list[str]:
    labels = matrix.index.astype(str).tolist()
    if len(labels) < 3:
        return labels
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return labels
    fill_value = float(np.nanmedian(finite))
    values = np.where(np.isfinite(values), values, fill_value)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 1.0)
    distances = np.clip(1.0 - values, 0.0, None)
    np.fill_diagonal(distances, 0.0)
    if np.allclose(distances, 0.0):
        return labels
    tree = linkage(squareform(distances, checks=False), method="average", optimal_ordering=True)
    return [labels[index] for index in leaves_list(tree)]


def plot_class_vs_full_chemical_similarity(*, similarity: pd.DataFrame, output_path: Path) -> None:
    plot_frame = similarity.sort_values("class_vs_full_chemical_rdm_rsa", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.28 * len(plot_frame) + 1.4)), constrained_layout=True)
    y = np.arange(len(plot_frame))
    colors = np.where(plot_frame["category"].eq("Purine nucleosides"), "#D55E00", "#4C78A8")
    ax.barh(y, plot_frame["class_vs_full_chemical_rdm_rsa"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_frame["category"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("chemical RDM RSA vs full QC20 chemical RDM")
    ax.set_title("Class RDM similarity to the full chemical RDM")
    median = float(np.nanmedian(plot_frame["class_vs_full_chemical_rdm_rsa"].to_numpy(float)))
    ax.axvline(median, color="#222222", linestyle="--", linewidth=1.1, label=f"median = {median:.2f}")
    ax.legend(fontsize=8)
    for row, value in enumerate(plot_frame["class_vs_full_chemical_rdm_rsa"].to_numpy(float)):
        ax.text(value + 0.01, row, f"{value:.2f}", va="center", fontsize=7)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_run_summary(
    *,
    output_root: Path,
    args: argparse.Namespace,
    labels: list[str],
    observed_scores: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    reselection_summary: pd.DataFrame,
    full_search_summary: pd.DataFrame,
    final_shortlist: pd.DataFrame,
    date_controlled_rdm_path: Path,
    class_similarity_summary: pd.DataFrame,
    class_full_similarity_summary: pd.DataFrame,
) -> None:
    top_observed = observed_scores.iloc[0]
    top_stable = reselection_summary.iloc[0]
    search = full_search_summary.iloc[0]
    lines = [
        "# Taxonomy Class Stability Review",
        "",
        "Exploratory figure-first review. These outputs are not paper-ready.",
        "",
        "## Inputs",
        "",
        f"- Preprocess root: `{args.preprocess_root}`",
        f"- Matrix path: `{args.matrix_path}`",
        f"- Raw metadata path: `{args.raw_metadata_path}`",
        f"- Taxonomy level: `{args.taxonomy_level}`",
        f"- Retained stimuli: `{len(labels)}`",
        f"- QC threshold: `{args.qc_threshold}`",
        "",
        "## Evidence Layers",
        "",
        f"- Fixed-class permutations: `{args.fixed_permutations}`",
        f"- Reselection resamples: `{args.resamples}`",
        f"- Full-search permutations: `{args.search_permutations}` internal diagnostic, not a main display panel",
        "",
        "## Quick Read",
        "",
        f"- Highest observed class: `{top_observed['category']}` RSA `{float(top_observed['response_window_rsa']):.4f}`",
        f"- Most reselected class: `{top_stable['category']}` top3 frequency `{float(top_stable['top3_frequency']):.3f}`",
        f"- Internal search-corrected best-class p: `{float(search['search_corrected_p']):.6f}`",
        "",
        "## Main Figures",
        "",
        "- `figures/01_fixed_class_permutation.png`",
        "- `figures/02_reselection_stability.png`",
        "- `figures/03_top_class_rdm_comparison.png`",
        "- `figures/04_taxonomy_class_stability_summary.png`",
        "- `figures/05_class_chemical_rdm_similarity_matrix.png`",
        "- `figures/06_class_vs_full_chemical_rdm_similarity.png`",
        f"- `{date_controlled_rdm_path}`",
        "",
        "## Class Chemical RDM Similarity",
        "",
        "```text",
        class_similarity_summary.to_string(index=False),
        "```",
        "",
        "## Class vs Full Chemical RDM Similarity",
        "",
        "```text",
        class_full_similarity_summary.to_string(index=False),
        "```",
        "",
        "## Diagnostic Only",
        "",
        "- `diagnostics/full_search_permutation_diagnostic.png`",
        "",
        "## Shortlist Preview",
        "",
        "```text",
        final_shortlist.head(12).loc[
            :,
            [
                "category",
                "response_window_rsa",
                "observed_percentile",
                "top3_frequency",
                "fixed_signal_pass",
                "search_corrected_p_for_best",
            ],
        ].to_string(index=False),
        "```",
    ]
    (output_root / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


if __name__ == "__main__":
    main()
