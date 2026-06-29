from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
HMDS_COORDINATES = "hmds_comparison/tables/coordinates__chord_hmds_poincare_3d.csv"
GRAM_SAMPLE_MAPPING = "tables/taxonomy_from_GM300/gram_status_prior/gram_status_sample_mapping.csv"

GRAM_COLORS = {
    "Gram-negative": "#e11d48",
    "Gram-positive": "#6d28d9",
    "Unknown": "#9ca3af",
}
GRAM_ORDER = ["Gram-negative", "Gram-positive", "Unknown"]


def load_merged_table(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coordinates_path = root / HMDS_COORDINATES
    gram_path = root / GRAM_SAMPLE_MAPPING
    coordinates = pd.read_csv(coordinates_path)
    gram = pd.read_csv(gram_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    gram["AID"] = gram["AID"].astype(str)

    merged = coordinates.merge(
        gram[["AID", "gram_status", "gram_note"]],
        left_on="sample_id",
        right_on="AID",
        how="left",
    )
    if merged["gram_status"].isna().any():
        missing = sorted(merged.loc[merged["gram_status"].isna(), "sample_id"].astype(str))
        raise ValueError(f"Missing Gram-status labels for samples: {missing}")
    merged["gram_status"] = pd.Categorical(merged["gram_status"], GRAM_ORDER, ordered=True)
    merged = merged.sort_values("gram_status").reset_index(drop=True)
    merged["gram_status"] = merged["gram_status"].astype(str)

    summary = (
        merged.groupby("gram_status", sort=False)
        .agg(
            n_strains=("sample_id", "count"),
            n_genera=("genus", "nunique"),
            n_species=("species", "nunique"),
        )
        .reset_index()
    )
    return merged, summary


def add_poincare_wireframe(axis) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 32j, 0 : np.pi : 16j]
    sphere_x = np.cos(u) * np.sin(v)
    sphere_y = np.sin(u) * np.sin(v)
    sphere_z = np.cos(v)
    axis.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#6b7280",
        linewidth=0.45,
        alpha=0.22,
        rstride=2,
        cstride=2,
    )


def plot_static(merged: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    figure = plt.figure(figsize=(8.6, 8.0), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    add_poincare_wireframe(axis)

    for status in GRAM_ORDER:
        group = merged[merged["gram_status"].eq(status)]
        if group.empty:
            continue
        row = summary[summary["gram_status"].eq(status)].iloc[0]
        axis.scatter(
            group["poincare1"],
            group["poincare2"],
            group["poincare3"],
            s=52,
            color=GRAM_COLORS[status],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.55,
            label=f"{status} (n={int(row.n_strains)}, genera={int(row.n_genera)})",
        )

    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-1.02, 1.02)
    axis.set_zlim(-1.02, 1.02)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.set_zlabel("Poincare 3", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9, frameon=False)
    figure.suptitle("86bac chord HMDS Poincare ball colored by Gram status", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def add_wireframe_traces(figure) -> None:
    import plotly.graph_objects as go

    wire_line = {"color": "rgba(107,114,128,0.28)", "width": 1}
    theta = np.linspace(0.0, np.pi, 90)
    phi = np.linspace(0.0, 2.0 * np.pi, 140)
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        figure.add_trace(
            go.Scatter3d(
                x=np.cos(angle) * np.sin(theta),
                y=np.sin(angle) * np.sin(theta),
                z=np.cos(theta),
                mode="lines",
                line=wire_line,
                hoverinfo="skip",
                showlegend=False,
                name="Poincare ball boundary",
            )
        )
    for z_value in np.linspace(-0.8, 0.8, 5):
        radius = math.sqrt(max(1.0 - z_value**2, 0.0))
        figure.add_trace(
            go.Scatter3d(
                x=radius * np.cos(phi),
                y=radius * np.sin(phi),
                z=np.full_like(phi, z_value),
                mode="lines",
                line=wire_line,
                hoverinfo="skip",
                showlegend=False,
                name="Poincare ball boundary",
            )
        )


def plot_html(merged: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()
    add_wireframe_traces(figure)
    for status in GRAM_ORDER:
        group = merged[merged["gram_status"].eq(status)]
        if group.empty:
            continue
        row = summary[summary["gram_status"].eq(status)].iloc[0]
        figure.add_trace(
            go.Scatter3d(
                x=group["poincare1"],
                y=group["poincare2"],
                z=group["poincare3"],
                mode="markers",
                name=f"{status} (n={int(row.n_strains)}, genera={int(row.n_genera)})",
                marker={
                    "size": 5.8,
                    "color": GRAM_COLORS[status],
                    "line": {"width": 0.5, "color": "white"},
                },
                text=group["sample_id"],
                customdata=np.column_stack([group["genus"], group["species"], group["gram_note"]]),
                hovertemplate=(
                    "name=%{text}<br>"
                    "Gram=%{fullData.name}<br>"
                    "genus=%{customdata[0]}<br>"
                    "species=%{customdata[1]}<br>"
                    "%{customdata[2]}<br>"
                    "Poincare1=%{x:.4f}<br>"
                    "Poincare2=%{y:.4f}<br>"
                    "Poincare3=%{z:.4f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="86bac chord HMDS Poincare ball colored by Gram status",
        width=980,
        height=820,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"font": {"size": 11}},
        scene={
            "xaxis": {"title": "Poincare 1", "range": [-1.02, 1.02]},
            "yaxis": {"title": "Poincare 2", "range": [-1.02, 1.02]},
            "zaxis": {"title": "Poincare 3", "range": [-1.02, 1.02]},
            "aspectmode": "cube",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def run(root: Path) -> dict[str, object]:
    merged, summary = load_merged_table(root)
    figures_dir = root / "hmds_comparison" / "figures"
    tables_dir = root / "hmds_comparison" / "tables"
    png_path = figures_dir / "poincare_ball__chord_hmds_3d_gram_status.png"
    html_path = figures_dir / "poincare_ball__chord_hmds_3d_gram_status.html"
    table_path = tables_dir / "coordinates__chord_hmds_poincare_3d_gram_status.csv"
    summary_path = tables_dir / "gram_status_summary_for_chord_hmds.csv"

    plot_static(merged, summary, png_path)
    plot_html(merged, summary, html_path)
    merged.to_csv(table_path, index=False)
    summary.to_csv(summary_path, index=False)

    run_summary = {
        "input": {
            "root": str(root),
            "coordinates": str(root / HMDS_COORDINATES),
            "gram_sample_mapping": str(root / GRAM_SAMPLE_MAPPING),
        },
        "n_samples": int(len(merged)),
        "gram_status_counts": summary.to_dict(orient="records"),
        "outputs": {
            "png": str(png_path),
            "html": str(html_path),
            "coordinates_with_gram": str(table_path),
            "summary": str(summary_path),
        },
    }
    (tables_dir / "gram_status_hmds_run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 86bac chord HMDS Poincare coordinates by Gram status.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))


if __name__ == "__main__":
    main()
