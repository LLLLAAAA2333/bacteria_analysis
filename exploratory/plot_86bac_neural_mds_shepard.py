from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
NEURAL_RDM = "neural_shape_rdm__active_scaled_flattened_correlation.csv"
MDS_COORDINATES = "rdm_mds_3d_coordinates.csv"


def read_rdm(path: Path) -> pd.DataFrame:
    rdm = pd.read_csv(path, index_col=0)
    rdm.index = rdm.index.astype(str)
    rdm.columns = rdm.columns.astype(str)
    if not rdm.index.equals(rdm.columns):
        raise ValueError(f"RDM index/columns do not match: {path}")
    values = rdm.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"RDM contains NaN values: {path}")
    return values


def upper_triangle(values: np.ndarray) -> np.ndarray:
    return values[np.triu_indices_from(values, k=1)]


def load_pair_distances(root: Path) -> pd.DataFrame:
    rdm = read_rdm(root / "tables" / NEURAL_RDM)
    coordinates = pd.read_csv(root / "figures" / MDS_COORDINATES)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    coordinates = coordinates.set_index("sample_id", drop=False)
    shared = [sample for sample in rdm.index.astype(str) if sample in coordinates.index]
    if len(shared) < 4:
        raise ValueError(f"need at least 4 shared samples; found {len(shared)}")

    rdm_values = rdm.loc[shared, shared].to_numpy(float)
    rdm_values = (rdm_values + rdm_values.T) / 2.0
    np.fill_diagonal(rdm_values, 0.0)
    mds_values = coordinates.loc[shared, ["neural_mds1", "neural_mds2", "neural_mds3"]].to_numpy(float)
    original = upper_triangle(rdm_values)
    embedded = pdist(mds_values)
    left_idx, right_idx = np.triu_indices(len(shared), k=1)
    return pd.DataFrame(
        {
            "sample_left": np.asarray(shared, dtype=object)[left_idx],
            "sample_right": np.asarray(shared, dtype=object)[right_idx],
            "original_neural_rdm_distance": original,
            "neural_3d_mds_distance": embedded,
            "residual_mds_minus_original": embedded - original,
            "absolute_residual": np.abs(embedded - original),
        }
    )


def summarize_fit(pair_table: pd.DataFrame) -> dict[str, float]:
    original = pair_table["original_neural_rdm_distance"].to_numpy(float)
    embedded = pair_table["neural_3d_mds_distance"].to_numpy(float)
    slope_through_origin = float(np.sum(original * embedded) / np.sum(original**2))
    fitted = slope_through_origin * original
    return {
        "n_pairs": int(len(pair_table)),
        "distance_spearman": float(spearmanr(original, embedded).statistic),
        "distance_pearson": float(pearsonr(original, embedded).statistic),
        "normalized_raw_stress": float(np.sqrt(np.sum((original - embedded) ** 2) / np.sum(original**2))),
        "scaled_stress_through_origin": float(np.sqrt(np.sum((fitted - embedded) ** 2) / np.sum(embedded**2))),
        "slope_embedded_vs_original_through_origin": slope_through_origin,
        "mean_absolute_residual": float(np.mean(np.abs(embedded - original))),
        "median_absolute_residual": float(np.median(np.abs(embedded - original))),
        "q95_absolute_residual": float(np.quantile(np.abs(embedded - original), 0.95)),
    }


def binned_summary(pair_table: pd.DataFrame, n_bins: int = 24) -> pd.DataFrame:
    table = pair_table.copy()
    table["bin"] = pd.qcut(
        table["original_neural_rdm_distance"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bin_id, group in table.groupby("bin", sort=True):
        rows.append(
            {
                "bin": int(bin_id),
                "n_pairs": int(len(group)),
                "original_distance_mean": float(group["original_neural_rdm_distance"].mean()),
                "original_distance_median": float(group["original_neural_rdm_distance"].median()),
                "mds_distance_median": float(group["neural_3d_mds_distance"].median()),
                "mds_distance_q25": float(group["neural_3d_mds_distance"].quantile(0.25)),
                "mds_distance_q75": float(group["neural_3d_mds_distance"].quantile(0.75)),
                "absolute_residual_median": float(group["absolute_residual"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_shepard(pair_table: pd.DataFrame, bins: pd.DataFrame, metrics: dict[str, float], output_path: Path) -> None:
    original = pair_table["original_neural_rdm_distance"].to_numpy(float)
    embedded = pair_table["neural_3d_mds_distance"].to_numpy(float)
    limit = float(max(original.max(), embedded.max()) * 1.04)

    figure, axis = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    axis.scatter(
        original,
        embedded,
        s=10,
        color="#4a5568",
        alpha=0.24,
        linewidths=0,
        label="sample pairs",
    )
    axis.plot([0, limit], [0, limit], linestyle="--", color="#111827", linewidth=1.0, label="identity")
    axis.errorbar(
        bins["original_distance_median"],
        bins["mds_distance_median"],
        yerr=[
            bins["mds_distance_median"] - bins["mds_distance_q25"],
            bins["mds_distance_q75"] - bins["mds_distance_median"],
        ],
        fmt="o-",
        color="#dc2626",
        ecolor="#dc2626",
        elinewidth=1.0,
        capsize=2,
        markersize=4,
        label="binned median +/- IQR",
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Original neural RDM distance", fontsize=10)
    axis.set_ylabel("3D MDS distance", fontsize=10)
    axis.set_title(
        "Neural MDS Shepard diagram\n"
        f"rho={metrics['distance_spearman']:.3f}, r={metrics['distance_pearson']:.3f}, "
        f"stress={metrics['normalized_raw_stress']:.3f}",
        fontsize=12,
    )
    axis.grid(True, color="#e5e7eb", linewidth=0.6)
    axis.legend(loc="upper left", fontsize=8, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240)
    plt.close(figure)


def plot_shepard_html(
    pair_table: pd.DataFrame,
    bins: pd.DataFrame,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    original = pair_table["original_neural_rdm_distance"].to_numpy(float)
    embedded = pair_table["neural_3d_mds_distance"].to_numpy(float)
    limit = float(max(original.max(), embedded.max()) * 1.04)

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=original,
            y=embedded,
            mode="markers",
            name="sample pairs",
            marker={"size": 5, "color": "rgba(74,85,104,0.35)"},
            text=pair_table["sample_left"] + " - " + pair_table["sample_right"],
            customdata=np.column_stack([pair_table["residual_mds_minus_original"], pair_table["absolute_residual"]]),
            hovertemplate=(
                "%{text}<br>"
                "original=%{x:.4f}<br>"
                "MDS=%{y:.4f}<br>"
                "residual=%{customdata[0]:.4f}<br>"
                "abs residual=%{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            name="identity",
            line={"color": "#111827", "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bins["original_distance_median"],
            y=bins["mds_distance_median"],
            mode="lines+markers",
            name="binned median",
            line={"color": "#dc2626"},
            marker={"size": 7, "color": "#dc2626"},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": bins["mds_distance_q75"] - bins["mds_distance_median"],
                "arrayminus": bins["mds_distance_median"] - bins["mds_distance_q25"],
            },
            hovertemplate="original median=%{x:.4f}<br>MDS median=%{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            "Neural MDS Shepard diagram"
            f"<br>rho={metrics['distance_spearman']:.3f}, "
            f"r={metrics['distance_pearson']:.3f}, "
            f"stress={metrics['normalized_raw_stress']:.3f}"
        ),
        xaxis={"title": "Original neural RDM distance", "range": [0, limit]},
        yaxis={"title": "3D MDS distance", "range": [0, limit], "scaleanchor": "x", "scaleratio": 1},
        width=780,
        height=720,
        margin={"l": 70, "r": 25, "t": 80, "b": 65},
        legend={"font": {"size": 11}},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(root: Path, n_bins: int) -> dict[str, object]:
    pair_table = load_pair_distances(root)
    metrics = summarize_fit(pair_table)
    bins = binned_summary(pair_table, n_bins=n_bins)

    output_dir = root / "tables" / "mds_diagnostics"
    figures_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pair_path = output_dir / "neural_mds_3d_shepard_pair_distances.csv"
    bins_path = output_dir / "neural_mds_3d_shepard_binned_summary.csv"
    metrics_path = output_dir / "neural_mds_3d_shepard_summary.json"
    png_path = figures_dir / "neural_mds_3d_shepard_diagram.png"
    html_path = figures_dir / "neural_mds_3d_shepard_diagram.html"

    pair_table.to_csv(pair_path, index=False)
    bins.to_csv(bins_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_shepard(pair_table, bins, metrics, png_path)
    plot_shepard_html(pair_table, bins, metrics, html_path)

    run_summary = {
        "input": {
            "root": str(root),
            "neural_rdm": str(root / "tables" / NEURAL_RDM),
            "mds_coordinates": str(root / "figures" / MDS_COORDINATES),
        },
        "n_bins": n_bins,
        "metrics": metrics,
        "outputs": {
            "pair_distances": str(pair_path),
            "binned_summary": str(bins_path),
            "metrics": str(metrics_path),
            "static_png": str(png_path),
            "interactive_html": str(html_path),
        },
    }
    (output_dir / "neural_mds_3d_shepard_run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a Shepard diagram for the 86bac neural 3D MDS.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bins", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.bins), indent=2))


if __name__ == "__main__":
    main()
