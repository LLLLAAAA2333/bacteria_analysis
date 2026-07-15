"""106bac RSA: neural response-window RDM vs chemical metabolite RDM.

Builds neural RDM (correlation distance, median prototype) and chemical RDM
(matrix.xlsx → QCRSD ≤ 0.20 → log2 → Euclidean distance), then computes
Spearman RSA with label-shuffle null, bootstrap CI, and stimulus-subset stability.

Uses matrix.xlsx directly with explicit QC filtering via QCRSD from raw metadata,
without pre-applied missing-to-1 or pre-filtering steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = str(PROJECT_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bacteria_analysis.analyses.rdm.builders import build_chemical_rdm, build_neural_rdm
from bacteria_analysis.analyses.rdm.core import align_square_rdms, spearman_similarity
from bacteria_analysis.analyses.rdm.stats import empirical_p_value, label_shuffle_null, stimulus_subset_rsa
from bacteria_analysis._analysis_dataset_impl import AnalysisDataset
from bacteria_analysis._data_loaders import build_stimulus_sample_map, read_metabolite_matrix

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sample_id(label: str) -> str:
    parts = str(label).strip().split()
    return parts[0] if parts else ""


def _sample_number(sample_id: str) -> int:
    import re
    match = re.search(r"(\d+)", str(sample_id))
    return int(match.group(1)) if match else -1


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xc, yc = x[mask] - np.mean(x[mask]), y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    return float(np.sum(xc * yc) / denom) if denom > 0 else float("nan")


def _spearman_vec(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return _pearson(rankdata(x[mask], method="average"), rankdata(y[mask], method="average"))


def _bootstrap_pair_rsa(
    neural_vals: np.ndarray, chemical_vals: np.ndarray, *, n_resamples: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_resamples):
        idx = rng.integers(0, len(neural_vals), size=len(neural_vals))
        rows.append({"iteration": i, "rsa": _spearman_vec(neural_vals[idx], chemical_vals[idx])})
    return pd.DataFrame(rows)


def _upper_tri_values(matrix: pd.DataFrame) -> np.ndarray:
    n = len(matrix)
    iu = np.triu_indices(n, k=1)
    return matrix.to_numpy(dtype=float)[iu]


def _neural_cluster_order(rdm: pd.DataFrame) -> list[str]:
    labels = rdm.index.astype(str).tolist()
    if len(labels) < 3:
        return labels
    values = rdm.to_numpy(dtype=float, copy=True)
    finite = values[np.isfinite(values)]
    fill = float(np.nanmedian(finite)) if finite.size else 0.0
    values = np.where(np.isfinite(values), values, fill)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    tree = linkage(squareform(values, checks=False), method="average", optimal_ordering=True)
    return [labels[i] for i in leaves_list(tree)]


def _mask_diagonal(matrix: pd.DataFrame) -> pd.DataFrame:
    display = matrix.apply(pd.to_numeric, errors="coerce").copy()
    values = display.to_numpy(dtype=float)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=display.index, columns=display.columns)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def plot_rdm_pair(neural: pd.DataFrame, chemical: pd.DataFrame, display_order: list[str],
                  rsa_value: float, output_path: Path) -> None:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#FFFFFF")
    n = len(display_order)
    fontsize = max(3.2, min(4.8, 240.0 / max(n, 1)))
    fig_w = max(13.4, 0.18 * n + 3.0)
    fig_h = max(5.7, 0.08 * n + 1.6)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.04, 1.0, 0.04])
    ax_n = fig.add_subplot(grid[0, 0])
    ax_c = fig.add_subplot(grid[0, 2])
    cax_n = fig.add_subplot(grid[0, 1])
    cax_c = fig.add_subplot(grid[0, 3])

    n_disp = _mask_diagonal(neural.loc[display_order, display_order])
    c_disp = _mask_diagonal(chemical.loc[display_order, display_order])

    for ax, cax, matrix, title, clabel in [
        (ax_n, cax_n, n_disp, "Neural RDM\nresponse window, correlation distance", "correlation distance"),
        (ax_c, cax_c, c_disp, f"Chemical RDM\nlog2 fc, euclidean\nRSA = {rsa_value:.4f}", "log2 Euclidean distance"),
    ]:
        vals = matrix.to_numpy(dtype=float)
        fin = vals[np.isfinite(vals)]
        im = ax.imshow(vals, cmap=cmap,
                       vmin=float(np.min(fin)) if fin.size else None,
                       vmax=float(np.max(fin)) if fin.size else None,
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_order, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(display_order, fontsize=fontsize)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(clabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle("106bac Neural vs Chemical RDM\nordered by neural clustering", fontsize=12)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_null_distribution(null: np.ndarray, observed: float, p_val: float,
                           output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    finite = null[np.isfinite(null)]
    q99 = float(np.quantile(finite, 0.99))
    ax.hist(finite, bins=80, color="#7E4CC2", edgecolor="#7E4CC2", linewidth=0.2, alpha=0.95)
    ax.axvline(observed, color="#F5A623", linewidth=2.0)
    ax.axvline(q99, color="#6E6E6E", linestyle="--", linewidth=1.4)
    ax.set_xlabel("shuffle RSA")
    ax.set_ylabel("permutations")
    ax.set_title(f"106bac label-shuffle null\n10,000 permutations; orange=observed; gray dashed=q99")
    ax.text(0.98, 0.92, f"obs={observed:.4f}\np={p_val:.6f}\nq99={q99:.4f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_bootstrap(bootstrap: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    vals = bootstrap["rsa"].dropna().to_numpy(dtype=float)
    ax.hist(vals, bins=50, color="#2563EB", alpha=0.78, edgecolor="white", linewidth=0.4)
    ci = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
    ax.axvline(ci[0], color="#C2410C", linestyle="--", linewidth=1.2)
    ax.axvline(ci[1], color="#C2410C", linestyle="--", linewidth=1.2)
    ax.axvline(float(np.mean(vals)), color="#C2410C", linewidth=2.0)
    ax.set_title(f"Pair bootstrap RSA\nmean={np.mean(vals):.4f}, 95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
    ax.set_xlabel("Spearman RSA")
    ax.set_ylabel("count")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_subset_stability(subset_results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    vals = subset_results["rsa_similarity"].dropna().to_numpy(dtype=float)
    axes[0].hist(vals, bins=50, color="#7E4CC2", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].axvline(float(np.median(vals)), color="#F5A623", linewidth=2.0)
    axes[0].set_title("Subset RSA distribution")
    axes[0].set_xlabel("Spearman RSA")
    axes[0].set_ylabel("count")

    pcts = subset_results["rsa_similarity"].dropna().to_numpy(dtype=float)
    axes[1].hist(pcts, bins=40, color="#F5A623", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].set_title(f"Subset RSA (n={len(subset_results)} subsets)")
    axes[1].set_xlabel("Spearman RSA")
    for ax in axes:
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="106bac neural-vs-chemical RSA")
    parser.add_argument("--neural-path", type=Path, default=Path("data/106bac.parquet"))
    parser.add_argument("--matrix-path", type=Path, default=Path("data/matrix.xlsx"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metabolism_raw_data.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/106bac_rsa"))
    parser.add_argument("--qc-threshold", type=float, default=0.2)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--subset-count", type=int, default=200)
    parser.add_argument("--subset-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    neural_raw = pd.read_parquet(args.neural_path)
    neural_raw["date"] = neural_raw["date"].fillna("").astype(str).str.strip()
    matrix = read_metabolite_matrix(args.matrix_path)
    metadata = pd.read_excel(args.metadata_path, sheet_name="all", engine="openpyxl")

    stimulus_sample_map = build_stimulus_sample_map(neural_raw, matrix_sample_ids=matrix.index)
    dates = tuple(sorted(neural_raw["date"].dropna().astype(str).unique()))

    ds = AnalysisDataset(
        neural=neural_raw.reset_index(drop=True),
        matrix=matrix,
        metadata=metadata.reset_index(drop=True),
        stimulus_sample_map=stimulus_sample_map,
        included_dates=dates,
        excluded_dates=(),
        parameters={"neural_path": str(args.neural_path), "matrix_path": str(args.matrix_path),
                     "metadata_path": str(args.metadata_path)},
    )

    # ---- build RDMs ----
    print("Building neural RDM (response_window)...")
    neural_result = build_neural_rdm(ds, view="response_window", aggregation="median",
                                     merge_lr=True, distance="correlation")
    neural_rdm = neural_result.matrix

    print("Building neural RDM (full_trajectory)...")
    neural_ft_result = build_neural_rdm(ds, view="full_trajectory", aggregation="median",
                                        merge_lr=True, distance="correlation")
    neural_ft_rdm = neural_ft_result.matrix

    print("Building chemical RDM...")
    chemical_result = build_chemical_rdm(ds, qc_threshold=args.qc_threshold,
                                         transform="log2", distance="euclidean")
    chemical_rdm = chemical_result.matrix

    # ---- align and compute RSA ----
    views = {
        "response_window": neural_rdm,
        "full_trajectory": neural_ft_rdm,
    }

    all_results: dict[str, dict] = {}

    for view_name, n_rdm in views.items():
        print(f"\n{'='*60}")
        print(f"RSA: {view_name} vs chemical")
        print(f"{'='*60}")

        aligned_n, aligned_c = align_square_rdms(n_rdm, chemical_rdm)
        n_shared = len(aligned_n)
        n_pairs = n_shared * (n_shared - 1) // 2
        print(f"  shared stimuli: {n_shared}, pairs: {n_pairs}")

        n_vals = _upper_tri_values(aligned_n)
        c_vals = _upper_tri_values(aligned_c)
        observed = _spearman_vec(n_vals, c_vals)
        observed_pearson = _pearson(n_vals, c_vals)
        print(f"  Spearman RSA = {observed:.6f}")
        print(f"  Pearson RSA  = {observed_pearson:.6f}")

        # ---- label-shuffle null ----
        print(f"  Running {args.permutations} label-shuffle permutations...")
        null = label_shuffle_null(aligned_n, aligned_c, n_permutations=args.permutations,
                                  seed=args.seed)
        p_val = empirical_p_value(observed, null, side="greater")
        finite_null = null[np.isfinite(null)]
        print(f"  p = {p_val:.6f}")
        print(f"  null mean = {float(np.mean(finite_null)):.6f}")
        print(f"  null q99  = {float(np.quantile(finite_null, 0.99)):.6f}")
        print(f"  null max  = {float(np.max(finite_null)):.6f}")

        # ---- bootstrap CI ----
        print(f"  Running {args.bootstrap_resamples} bootstrap resamples...")
        bootstrap = _bootstrap_pair_rsa(n_vals, c_vals, n_resamples=args.bootstrap_resamples,
                                        seed=args.seed + 1)
        bs_mean = float(bootstrap["rsa"].mean())
        bs_ci = (float(bootstrap["rsa"].quantile(0.025)), float(bootstrap["rsa"].quantile(0.975)))
        print(f"  bootstrap mean = {bs_mean:.6f}, 95% CI = [{bs_ci[0]:.6f}, {bs_ci[1]:.6f}]")

        # ---- stimulus subset stability ----
        print(f"  Running {args.subset_count} stimulus subsets...")
        subset_results = stimulus_subset_rsa(
            aligned_n, aligned_c,
            subset_count=args.subset_count,
            subset_fraction=args.subset_fraction,
            seed=args.seed + 2,
        )
        sub_vals = subset_results["rsa_similarity"].dropna().to_numpy(dtype=float)
        print(f"  subset RSA median = {float(np.median(sub_vals)):.6f}")
        print(f"  subset RSA range  = [{float(np.min(sub_vals)):.6f}, {float(np.max(sub_vals)):.6f}]")

        # ---- save tables ----
        prefix = view_name
        aligned_n.to_csv(tables_dir / f"neural_rdm__{prefix}.csv")
        aligned_c.to_csv(tables_dir / f"chemical_rdm__{prefix}.csv")

        null_df = pd.DataFrame({"iteration": range(args.permutations), "rsa_spearman": null})
        null_df.to_csv(tables_dir / f"label_shuffle_null__{prefix}.csv", index=False)

        bootstrap.to_csv(tables_dir / f"pair_bootstrap__{prefix}.csv", index=False)
        subset_results.to_csv(tables_dir / f"stimulus_subset_rsa__{prefix}.csv", index=False)

        # ---- plots ----
        display_order = _neural_cluster_order(aligned_n)
        plot_rdm_pair(aligned_n, aligned_c, display_order, observed,
                      figures_dir / f"rdm_heatmaps__{prefix}.png")
        plot_null_distribution(null, observed, p_val,
                               figures_dir / f"label_shuffle_null__{prefix}.png")
        plot_bootstrap(bootstrap, figures_dir / f"pair_bootstrap__{prefix}.png")
        plot_subset_stability(subset_results,
                              figures_dir / f"stimulus_subset_rsa__{prefix}.png")

        # ---- accumulate results ----
        all_results[view_name] = {
            "n_shared": n_shared,
            "n_pairs": n_pairs,
            "spearman_rsa": float(observed),
            "pearson_rsa": float(observed_pearson),
            "label_shuffle_permutations": args.permutations,
            "label_shuffle_p": float(p_val),
            "null_mean": float(np.mean(finite_null)),
            "null_std": float(np.std(finite_null, ddof=1)),
            "null_q99": float(np.quantile(finite_null, 0.99)),
            "null_max": float(np.max(finite_null)),
            "bootstrap_mean": bs_mean,
            "bootstrap_ci95": [bs_ci[0], bs_ci[1]],
            "subset_count": args.subset_count,
            "subset_fraction": args.subset_fraction,
            "subset_rsa_median": float(np.median(sub_vals)),
            "subset_rsa_q25": float(np.quantile(sub_vals, 0.25)),
            "subset_rsa_q75": float(np.quantile(sub_vals, 0.75)),
        }

    # ---- chemical metadata ----
    chemical_meta = chemical_result.metadata
    chem_info = {
        "matrix_path": str(args.matrix_path),
        "metadata_path": str(args.metadata_path),
        "qc_threshold": args.qc_threshold,
        "transform": "log2",
        "distance": "euclidean",
        "feature_count": chemical_meta.get("feature_count", "unknown"),
        "n_stimuli_chemical": int(len(chemical_rdm)),
    }

    # ---- summary JSON ----
    summary = {
        "batch": "106bac",
        "seed": args.seed,
        "neural": {
            "input": str(args.neural_path),
            "n_trials_lr_merged": neural_result.metadata.get("n_trials"),
            "n_features": neural_result.metadata.get("feature_count"),
            "view": "response_window + full_trajectory",
            "aggregation": "median",
            "merge_lr": True,
            "distance": "correlation",
        },
        "chemical": chem_info,
        "rsa": all_results,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- summary markdown ----
    md = [f"# 106bac RSA: Neural vs Chemical RDM", "",
          f"## 数据", "",
          f"- 神经: `{args.neural_path}`",
          f"- 化学: `{args.matrix_path}` (QCRSD ≤ {args.qc_threshold})",
          f"- 种子: {args.seed}", "",
          f"## 结果", ""]

    for view_name, r in all_results.items():
        md += [
            f"### {view_name}", "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 共享细菌数 | {r['n_shared']} |",
            f"| 细菌对数 | {r['n_pairs']} |",
            f"| **Spearman RSA** | **{r['spearman_rsa']:.4f}** |",
            f"| Pearson RSA | {r['pearson_rsa']:.4f} |",
            f"| 标签置换 p | {r['label_shuffle_p']:.6f} |",
            f"| 原假设均值 | {r['null_mean']:.4f} |",
            f"| 原假设 q99 | {r['null_q99']:.4f} |",
            f"| Bootstrap 均值 | {r['bootstrap_mean']:.4f} |",
            f"| Bootstrap 95% CI | [{r['bootstrap_ci95'][0]:.4f}, {r['bootstrap_ci95'][1]:.4f}] |",
            f"| 子集 RSA 中位数 | {r['subset_rsa_median']:.4f} |",
            f"| 子集 RSA IQR | [{r['subset_rsa_q25']:.4f}, {r['subset_rsa_q75']:.4f}] |",
            f"",
        ]

    md += [
        "## 与前批次对比", "",
        "| 批次 | 细菌数 | Spearman RSA | p 值 |",
        "|------|--------|-------------|------|",
        "| 76bac (响应窗口) | 26 | 0.343 | 0.001 |",
        "| 86bac (响应窗口) | 86 | 0.157 | 0.0002 |",
    ]
    if all_results:
        rw = all_results.get("response_window", {})
        md += [f"| **106bac (响应窗口)** | **{rw.get('n_shared', '?')}** | **{rw.get('spearman_rsa', '?'):.4f}** | **{rw.get('label_shuffle_p', '?'):.6f}** |"]

    output_dir.joinpath("run_summary.md").write_text("\n".join(md), encoding="utf-8")

    # ---- print final summary ----
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for view_name, r in all_results.items():
        print(f"\n  {view_name}:")
        print(f"    Spearman RSA = {r['spearman_rsa']:.4f}")
        print(f"    Pearson RSA  = {r['pearson_rsa']:.4f}")
        print(f"    p(label-shuffle) = {r['label_shuffle_p']:.6f}")
        print(f"    null q99 = {r['null_q99']:.4f}")
        print(f"    bootstrap 95% CI = [{r['bootstrap_ci95'][0]:.4f}, {r['bootstrap_ci95'][1]:.4f}]")
    print(f"\n  Output: {output_dir}")


if __name__ == "__main__":
    main()
