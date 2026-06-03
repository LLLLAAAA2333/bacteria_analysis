from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
NEURAL_PROTOTYPES = "active_scaled_flattened_neural_prototypes.csv"
CHEMICAL_MATRIX = "chemical_log2_zscore_matrix.csv"
RETAINED_CHEMICALS = "retained_chemical_features.csv"
CHEMICAL_AUDIT = "chemical_feature_type_audit/matrix_chemical_feature_type_audit.csv"

METRICS = ("stim_mean", "post_mean", "window_mean", "post_minus_stim", "signed_peak", "abs_peak")


def sample_number(sample_id: object) -> int:
    match = re.search(r"(\d+)", str(sample_id))
    return int(match.group(1)) if match else -1


def parse_neural_columns(columns: list[str]) -> tuple[list[str], list[int]]:
    neurons: list[str] = []
    timepoints: list[int] = []
    for column in columns:
        if "__t" not in column:
            continue
        neuron, time_text = column.split("__t", 1)
        try:
            timepoint = int(time_text)
        except ValueError:
            continue
        if neuron not in neurons:
            neurons.append(neuron)
        if timepoint not in timepoints:
            timepoints.append(timepoint)
    return neurons, sorted(timepoints)


def signed_peak(values: np.ndarray) -> np.ndarray:
    peaks = np.full(values.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(values):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        row_values = row[finite]
        peaks[row_index] = float(row_values[np.argmax(np.abs(row_values))])
    return peaks


def extract_neural_metrics(prototypes: pd.DataFrame) -> pd.DataFrame:
    neurons, timepoints = parse_neural_columns(prototypes.columns.astype(str).tolist())
    if not neurons or not timepoints:
        raise ValueError("No neural feature columns matching NEURON__tXX were found.")

    stim_timepoints = [timepoint for timepoint in timepoints if 5 <= timepoint < 15]
    post_timepoints = [timepoint for timepoint in timepoints if 15 <= timepoint < 25]
    if len(stim_timepoints) != 10 or len(post_timepoints) != 10:
        raise ValueError("Expected t05..t14 stimulus and t15..t24 post-stimulus windows.")

    rows: list[dict[str, object]] = []
    for neuron in neurons:
        all_columns = [f"{neuron}__t{timepoint:02d}" for timepoint in timepoints]
        stim_columns = [f"{neuron}__t{timepoint:02d}" for timepoint in stim_timepoints]
        post_columns = [f"{neuron}__t{timepoint:02d}" for timepoint in post_timepoints]

        all_values = prototypes.loc[:, all_columns].to_numpy(dtype=float, copy=False)
        stim_values = prototypes.loc[:, stim_columns].to_numpy(dtype=float, copy=False)
        post_values = prototypes.loc[:, post_columns].to_numpy(dtype=float, copy=False)
        peak = signed_peak(all_values)

        for row_index, sample_id in enumerate(prototypes["sample_id"].astype(str)):
            stim_mean = float(np.nanmean(stim_values[row_index]))
            post_mean = float(np.nanmean(post_values[row_index]))
            rows.extend(
                [
                    {
                        "sample_id": sample_id,
                        "neuron": neuron,
                        "metric": "stim_mean",
                        "response_value": stim_mean,
                    },
                    {
                        "sample_id": sample_id,
                        "neuron": neuron,
                        "metric": "post_mean",
                        "response_value": post_mean,
                    },
                    {
                        "sample_id": sample_id,
                        "neuron": neuron,
                        "metric": "window_mean",
                        "response_value": float(np.nanmean(all_values[row_index])),
                    },
                    {
                        "sample_id": sample_id,
                        "neuron": neuron,
                        "metric": "post_minus_stim",
                        "response_value": post_mean - stim_mean,
                    },
                    {
                        "sample_id": sample_id,
                        "neuron": neuron,
                        "metric": "signed_peak",
                        "response_value": float(peak[row_index]),
                    },
                    {
                        "sample_id": sample_id,
                        "neuron": neuron,
                        "metric": "abs_peak",
                        "response_value": float(abs(peak[row_index])),
                    },
                ]
            )
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q

    finite_indices = np.flatnonzero(finite)
    ordered_local = np.argsort(p[finite])
    ordered_indices = finite_indices[ordered_local]
    ordered_p = p[ordered_indices]

    n = len(ordered_p)
    ordered_q = ordered_p * n / np.arange(1, n + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    q[ordered_indices] = np.clip(ordered_q, 0.0, 1.0)
    return q


def safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> tuple[float, float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    n = int(valid.sum())
    if n < 4:
        return np.nan, np.nan, n
    x = left[valid]
    y = right[valid]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, n
    if method == "spearman":
        stat, p_value = spearmanr(x, y)
    elif method == "pearson":
        stat, p_value = pearsonr(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(stat), float(p_value), n


def build_feature_annotations(root: Path, chemical_features: list[str]) -> pd.DataFrame:
    retained = pd.read_csv(root / "tables" / RETAINED_CHEMICALS)
    audit = pd.read_csv(root / "tables" / CHEMICAL_AUDIT)

    retained_columns = [
        "_feature_name",
        "name",
        "Mass",
        "RT",
        "KEGG",
        "HMDB",
        "SuperClass",
        "Class",
        "SubClass",
        "DirectParent",
        "QCRSD",
        "_raw_missing_rate",
    ]
    retained = retained[[column for column in retained_columns if column in retained.columns]].copy()
    retained = retained.drop_duplicates("_feature_name", keep="first")
    retained = retained.rename(columns={"_feature_name": "chemical_feature", "name": "annotation_name"})

    audit_columns = [
        "matrix_feature",
        "mass_first",
        "mass_max",
        "broad_bucket",
        "SuperClass",
        "Class",
        "SubClass",
        "DirectParent",
    ]
    audit = audit[[column for column in audit_columns if column in audit.columns]].copy()
    audit = audit.drop_duplicates("matrix_feature", keep="first")
    audit = audit.rename(
        columns={
            "matrix_feature": "chemical_feature",
            "SuperClass": "audit_SuperClass",
            "Class": "audit_Class",
            "SubClass": "audit_SubClass",
            "DirectParent": "audit_DirectParent",
        }
    )

    annotations = pd.DataFrame({"chemical_feature": chemical_features})
    annotations = annotations.merge(retained, on="chemical_feature", how="left")
    annotations = annotations.merge(audit, on="chemical_feature", how="left")
    for column in ("SuperClass", "Class", "SubClass", "DirectParent"):
        audit_column = f"audit_{column}"
        if column in annotations and audit_column in annotations:
            annotations[column] = annotations[column].fillna(annotations[audit_column])
    return annotations


def compute_associations(neural_metrics: pd.DataFrame, chemical: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    sample_ids = [sample_id for sample_id in neural_metrics["sample_id"].unique() if sample_id in set(chemical["sample_id"])]
    chemical = chemical.set_index("sample_id").loc[sample_ids]
    chemical_features = chemical.columns.astype(str).tolist()

    rows: list[dict[str, object]] = []
    for (neuron, metric), group in neural_metrics.groupby(["neuron", "metric"], sort=True):
        neural_values = group.set_index("sample_id").loc[sample_ids, "response_value"].to_numpy(dtype=float)
        for chemical_feature in chemical_features:
            chemical_values = chemical[chemical_feature].to_numpy(dtype=float)
            spearman_r, spearman_p, n = safe_correlation(neural_values, chemical_values, "spearman")
            pearson_r, pearson_p, _ = safe_correlation(neural_values, chemical_values, "pearson")
            rows.append(
                {
                    "neuron": neuron,
                    "metric": metric,
                    "chemical_feature": chemical_feature,
                    "n_samples": n,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                }
            )

    result = pd.DataFrame(rows)
    result["spearman_q_global"] = benjamini_hochberg(result["spearman_p"].to_numpy(dtype=float))
    result["abs_spearman_r"] = result["spearman_r"].abs()
    result = result.merge(annotations, on="chemical_feature", how="left")
    return result.sort_values(["spearman_q_global", "abs_spearman_r"], ascending=[True, False]).reset_index(drop=True)


def top_per_neuron_metric(associations: pd.DataFrame, *, n: int) -> pd.DataFrame:
    ranked = associations.sort_values(["neuron", "metric", "abs_spearman_r"], ascending=[True, True, False])
    return ranked.groupby(["neuron", "metric"], sort=False).head(n).reset_index(drop=True)


def best_per_neuron_metric(associations: pd.DataFrame) -> pd.DataFrame:
    ranked = associations.sort_values(["neuron", "metric", "abs_spearman_r"], ascending=[True, True, False])
    return ranked.groupby(["neuron", "metric"], sort=False).head(1).reset_index(drop=True)


def top_bucket_counts(associations: pd.DataFrame, *, n: int = 100) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, group in associations.dropna(subset=["spearman_r"]).groupby("metric", sort=True):
        top = group.sort_values("abs_spearman_r", ascending=False).head(n).copy()
        counts = top["broad_bucket"].fillna("NA").value_counts()
        for bucket, count in counts.items():
            rows.append({"metric": metric, "top_n": n, "broad_bucket": bucket, "n_hits": int(count)})
    return pd.DataFrame(rows)


def summarize(associations: pd.DataFrame, neural_metrics: pd.DataFrame, chemical: pd.DataFrame) -> dict[str, object]:
    finite = associations.replace([np.inf, -np.inf], np.nan).dropna(subset=["spearman_r", "spearman_p"])
    summary: dict[str, object] = {
        "n_samples": int(chemical["sample_id"].nunique()),
        "n_neurons": int(neural_metrics["neuron"].nunique()),
        "n_metrics": int(neural_metrics["metric"].nunique()),
        "n_chemical_features": int(len([column for column in chemical.columns if column != "sample_id"])),
        "n_tests": int(len(finite)),
        "metric_definitions": {
            "stim_mean": "mean active-scaled response over t05..t14",
            "post_mean": "mean active-scaled response over t15..t24",
            "window_mean": "mean active-scaled response over t05..t24",
            "post_minus_stim": "post_mean - stim_mean",
            "signed_peak": "signed value with largest absolute magnitude over t05..t24",
            "abs_peak": "absolute signed_peak; response-strength screen",
        },
    }
    for threshold in (0.05, 0.10, 0.20):
        summary[f"n_q_le_{threshold:g}"] = int((finite["spearman_q_global"] <= threshold).sum())
    for threshold in (0.4, 0.5, 0.6, 0.7):
        summary[f"n_abs_r_ge_{threshold:g}"] = int((finite["abs_spearman_r"] >= threshold).sum())

    by_metric = (
        finite.groupby("metric")
        .agg(
            n_tests=("spearman_r", "size"),
            max_abs_r=("abs_spearman_r", "max"),
            n_q_le_0p1=("spearman_q_global", lambda x: int((x <= 0.10).sum())),
            n_abs_r_ge_0p5=("abs_spearman_r", lambda x: int((x >= 0.50).sum())),
        )
        .reset_index()
    )
    summary["by_metric"] = by_metric.to_dict(orient="records")
    return summary


def short_label(text: object, max_len: int = 42) -> str:
    value = str(text)
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def plot_top_associations(associations: pd.DataFrame, output_path: Path, *, n: int = 35) -> None:
    plot_frame = associations.dropna(subset=["spearman_r"]).sort_values(
        ["spearman_q_global", "abs_spearman_r"], ascending=[True, False]
    )
    plot_frame = plot_frame.head(n).iloc[::-1].copy()
    plot_frame["label"] = [
        f"{row.neuron} {row.metric} | {short_label(row.chemical_feature, 34)}"
        for row in plot_frame.itertuples(index=False)
    ]
    colors = np.where(plot_frame["spearman_r"] >= 0, "#c2410c", "#2563eb")

    fig_height = max(7.0, 0.28 * len(plot_frame) + 1.8)
    figure, axis = plt.subplots(figsize=(11.5, fig_height), constrained_layout=True)
    axis.barh(plot_frame["label"], plot_frame["spearman_r"], color=colors)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_xlabel("Spearman r")
    axis.set_title("Top neuron-chemical associations")
    axis.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_metric_heatmap(
    associations: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    n_chemicals: int = 40,
) -> None:
    metric_frame = associations.loc[associations["metric"].eq(metric)].dropna(subset=["spearman_r"]).copy()
    if metric_frame.empty:
        return
    selected = (
        metric_frame.groupby("chemical_feature")["abs_spearman_r"]
        .max()
        .sort_values(ascending=False)
        .head(n_chemicals)
        .index
    )
    plot_frame = metric_frame.loc[metric_frame["chemical_feature"].isin(selected)].copy()
    matrix = plot_frame.pivot_table(index="chemical_feature", columns="neuron", values="spearman_r", aggfunc="first")
    order = (
        plot_frame.groupby("chemical_feature")["abs_spearman_r"]
        .max()
        .sort_values(ascending=True)
        .index
    )
    matrix = matrix.loc[order]
    matrix.index = [short_label(index, 48) for index in matrix.index]

    figure, axis = plt.subplots(figsize=(11.5, max(8.0, 0.24 * len(matrix) + 2.0)), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="coolwarm",
        center=0,
        vmin=-0.75,
        vmax=0.75,
        linewidths=0.2,
        linecolor="#eeeeee",
        cbar_kws={"label": "Spearman r"},
    )
    axis.set_xlabel("Neuron")
    axis.set_ylabel("Chemical feature")
    axis.set_title(f"{metric}: strongest chemical associations by neuron")
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.tick_params(axis="y", labelsize=7)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_neuron_metric_summary(associations: pd.DataFrame, output_path: Path) -> None:
    finite = associations.dropna(subset=["spearman_r"]).copy()
    summary = (
        finite.groupby(["neuron", "metric"])["abs_spearman_r"]
        .max()
        .reset_index()
        .pivot(index="metric", columns="neuron", values="abs_spearman_r")
        .reindex(index=METRICS)
    )
    figure, axis = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    sns.heatmap(
        summary,
        ax=axis,
        cmap="magma",
        vmin=0,
        vmax=max(0.7, float(np.nanmax(summary.to_numpy()))),
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "max |Spearman r|"},
    )
    axis.set_xlabel("Neuron")
    axis.set_ylabel("Response metric")
    axis.set_title("Strongest chemical association per neuron and metric")
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run(root: Path) -> dict[str, object]:
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    output_dir = tables_dir / "single_neuron_chemical_associations"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prototypes = pd.read_csv(tables_dir / NEURAL_PROTOTYPES).sort_values("sample_id", key=lambda x: x.map(sample_number))
    chemical = pd.read_csv(tables_dir / CHEMICAL_MATRIX).sort_values("sample_id", key=lambda x: x.map(sample_number))
    chemical_features = [column for column in chemical.columns.astype(str).tolist() if column != "sample_id"]

    shared_samples = sorted(set(prototypes["sample_id"].astype(str)) & set(chemical["sample_id"].astype(str)), key=sample_number)
    prototypes = prototypes.loc[prototypes["sample_id"].astype(str).isin(shared_samples)].copy()
    chemical = chemical.loc[chemical["sample_id"].astype(str).isin(shared_samples)].copy()

    neural_metrics = extract_neural_metrics(prototypes)
    annotations = build_feature_annotations(root, chemical_features)
    associations = compute_associations(neural_metrics, chemical, annotations)
    top_table = top_per_neuron_metric(associations, n=10)
    best_table = best_per_neuron_metric(associations)
    bucket_counts = top_bucket_counts(associations, n=100)
    metric_summary = summarize(associations, neural_metrics, chemical)

    neural_metrics.to_csv(output_dir / "single_neuron_response_metrics.csv", index=False)
    associations.to_csv(output_dir / "single_neuron_chemical_associations.csv", index=False)
    top_table.to_csv(output_dir / "single_neuron_chemical_top10_per_neuron_metric.csv", index=False)
    best_table.to_csv(output_dir / "single_neuron_chemical_best_per_neuron_metric.csv", index=False)
    bucket_counts.to_csv(output_dir / "single_neuron_chemical_top100_bucket_counts.csv", index=False)
    (output_dir / "single_neuron_chemical_association_summary.json").write_text(
        json.dumps(metric_summary, indent=2), encoding="utf-8"
    )

    plot_top_associations(associations, figures_dir / "single_neuron_chemical_top_associations.png")
    plot_metric_heatmap(associations, figures_dir / "single_neuron_chemical_signed_peak_heatmap.png", metric="signed_peak")
    plot_metric_heatmap(associations, figures_dir / "single_neuron_chemical_abs_peak_heatmap.png", metric="abs_peak")
    plot_neuron_metric_summary(associations, figures_dir / "single_neuron_chemical_metric_summary.png")

    run_summary = {
        **metric_summary,
        "outputs": {
            "response_metrics": str(output_dir / "single_neuron_response_metrics.csv"),
            "all_associations": str(output_dir / "single_neuron_chemical_associations.csv"),
            "top10_per_neuron_metric": str(output_dir / "single_neuron_chemical_top10_per_neuron_metric.csv"),
            "best_per_neuron_metric": str(output_dir / "single_neuron_chemical_best_per_neuron_metric.csv"),
            "top100_bucket_counts": str(output_dir / "single_neuron_chemical_top100_bucket_counts.csv"),
            "summary": str(output_dir / "single_neuron_chemical_association_summary.json"),
            "top_associations_figure": str(figures_dir / "single_neuron_chemical_top_associations.png"),
            "signed_peak_heatmap": str(figures_dir / "single_neuron_chemical_signed_peak_heatmap.png"),
            "abs_peak_heatmap": str(figures_dir / "single_neuron_chemical_abs_peak_heatmap.png"),
            "metric_summary_figure": str(figures_dir / "single_neuron_chemical_metric_summary.png"),
        },
    }
    print(json.dumps(run_summary, indent=2))
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen single-neuron response metrics against chemical features.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
