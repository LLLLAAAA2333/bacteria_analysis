from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_86bac_chord_hmds as base


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
CHEMICAL_RDM = "chemical_rdm__qc20_missing50_log2_zscore_pca10_euclidean.csv"
REFERENCE_SAMPLE = "A001"


PLOT_SPECS = [
    {
        "key": "neural_chord_hmds_3d",
        "title": "Neural chord HMDS 3D",
        "coordinate_path": Path("hmds_comparison") / "tables" / "coordinates__chord_hmds_poincare_3d.csv",
        "dim": 3,
    },
    {
        "key": "chemical_euclidean_hmds_3d",
        "title": "Chemical Euclidean HMDS 3D",
        "coordinate_path": Path("chemical_hmds_comparison")
        / "tables"
        / "coordinates__chemical_hmds_poincare_3d.csv",
        "dim": 3,
    },
    {
        "key": "neural_chord_hmds_2d",
        "title": "Neural chord HMDS 2D",
        "coordinate_path": Path("hmds_2d_comparison")
        / "tables"
        / "coordinates__neural_chord_hmds_poincare_2d.csv",
        "dim": 2,
    },
    {
        "key": "chemical_euclidean_hmds_2d",
        "title": "Chemical Euclidean HMDS 2D",
        "coordinate_path": Path("hmds_2d_comparison")
        / "tables"
        / "coordinates__chemical_euclidean_hmds_poincare_2d.csv",
        "dim": 2,
    },
]


def chemical_distance_values(root: Path, reference_sample: str) -> pd.DataFrame:
    rdm = base.read_rdm(root / "tables" / CHEMICAL_RDM)
    if reference_sample not in set(rdm.index.astype(str)):
        raise ValueError(f"reference sample {reference_sample} not found in chemical RDM")
    samples = rdm.index.astype(str).tolist()
    distance = base.normalize_to_max_two(base.clean_distance_matrix(rdm))
    reference_index = samples.index(reference_sample)
    masked = distance.copy()
    np.fill_diagonal(masked, np.nan)
    nearest = distance.copy()
    np.fill_diagonal(nearest, np.inf)

    values = pd.DataFrame(
        {
            "sample_id": samples,
            f"chemical_distance_to_{reference_sample}": distance[:, reference_index],
            "chemical_nearest_distance": np.min(nearest, axis=1),
        }
    )
    distance_column = f"chemical_distance_to_{reference_sample}"
    values[f"{distance_column}_rank"] = values[distance_column].rank(method="average")
    values[f"{distance_column}_percentile"] = (
        (values[f"{distance_column}_rank"] - 1) / max(len(values) - 1, 1)
    )
    return values


def merge_values(coordinates_path: Path, values: pd.DataFrame) -> pd.DataFrame:
    coordinates = pd.read_csv(coordinates_path)
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    merged = coordinates.merge(values, on="sample_id", how="left")
    value_columns = [column for column in values.columns if column.startswith("chemical_distance_to_")]
    distance_column = [column for column in value_columns if not column.endswith("_rank") and not column.endswith("_percentile")][0]
    if merged[distance_column].isna().any():
        missing = sorted(merged.loc[merged[distance_column].isna(), "sample_id"].astype(str))
        raise ValueError(f"missing chemical distance values for samples: {missing}")
    return merged


def color_column(reference_sample: str) -> str:
    return f"chemical_distance_to_{reference_sample}"


def percentile_column(reference_sample: str) -> str:
    return f"{color_column(reference_sample)}_percentile"


def color_limits(values: pd.DataFrame, reference_sample: str) -> tuple[float, float]:
    column = color_column(reference_sample)
    return float(values[column].min()), float(values[column].max())


def add_poincare_wireframe(axis) -> None:
    u, v = np.mgrid[0 : 2 * np.pi : 32j, 0 : np.pi : 16j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    axis.plot_wireframe(
        x,
        y,
        z,
        color="#6b7280",
        linewidth=0.45,
        alpha=0.22,
        rstride=2,
        cstride=2,
    )


def set_poincare_3d_axis(axis) -> None:
    axis.set_xlim(-1.02, 1.02)
    axis.set_ylim(-1.02, 1.02)
    axis.set_zlim(-1.02, 1.02)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.set_zlabel("Poincare 3", fontsize=9)
    axis.tick_params(labelsize=8)


def set_poincare_2d_axis(axis) -> None:
    axis.set_xlim(-1.03, 1.03)
    axis.set_ylim(-1.03, 1.03)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.grid(True, color="#e5e7eb", linewidth=0.6)


def plot_static_3d(
    merged: pd.DataFrame,
    *,
    title: str,
    reference_sample: str,
    cmin: float,
    cmax: float,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(8.6, 8.0), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    add_poincare_wireframe(axis)
    scatter = axis.scatter(
        merged["poincare1"],
        merged["poincare2"],
        merged["poincare3"],
        c=merged[color_column(reference_sample)],
        cmap=plt.cm.viridis,
        vmin=cmin,
        vmax=cmax,
        s=48,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
    )
    set_poincare_3d_axis(axis)
    axis.set_title(title, fontsize=12)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label(f"Chemical distance to {reference_sample}", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(f"86bac HMDS colored by chemical distance to {reference_sample}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_static_2d(
    merged: pd.DataFrame,
    *,
    title: str,
    reference_sample: str,
    cmin: float,
    cmax: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, linewidth=1.0)
    axis.add_artist(circle)
    scatter = axis.scatter(
        merged["poincare1"],
        merged["poincare2"],
        c=merged[color_column(reference_sample)],
        cmap=plt.cm.viridis,
        vmin=cmin,
        vmax=cmax,
        s=44,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
    )
    set_poincare_2d_axis(axis)
    axis.set_title(title, fontsize=12)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label(f"Chemical distance to {reference_sample}", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(f"86bac 2D HMDS colored by chemical distance to {reference_sample}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def wireframe_traces() -> list[object]:
    import plotly.graph_objects as go

    traces = []
    wire_line = {"color": "rgba(107,114,128,0.28)", "width": 1}
    theta = np.linspace(0.0, np.pi, 90)
    phi = np.linspace(0.0, 2.0 * np.pi, 140)
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        traces.append(
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
        radius = float(np.sqrt(max(1.0 - z_value**2, 0.0)))
        traces.append(
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
    return traces


def hover_customdata(merged: pd.DataFrame, dim: int, reference_sample: str) -> np.ndarray:
    radius_cols = [f"poincare{i}" for i in range(1, dim + 1)]
    radius = np.linalg.norm(merged[radius_cols].to_numpy(float), axis=1)
    return np.column_stack(
        [
            merged["genus"].astype(str),
            merged["species"].astype(str),
            merged[color_column(reference_sample)],
            merged["chemical_nearest_distance"],
            merged[percentile_column(reference_sample)],
            radius,
        ]
    )


def plot_html_3d(
    merged: pd.DataFrame,
    *,
    title: str,
    reference_sample: str,
    cmin: float,
    cmax: float,
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()
    for trace in wireframe_traces():
        figure.add_trace(trace)
    figure.add_trace(
        go.Scatter3d(
            x=merged["poincare1"],
            y=merged["poincare2"],
            z=merged["poincare3"],
            mode="markers",
            marker={
                "size": 5.2,
                "color": merged[color_column(reference_sample)],
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "line": {"width": 0.5, "color": "white"},
                "colorbar": {"title": f"Distance to {reference_sample}"},
            },
            text=merged["sample_id"].astype(str),
            customdata=hover_customdata(merged, dim=3, reference_sample=reference_sample),
            hovertemplate=(
                "name=%{text}<br>"
                "genus=%{customdata[0]}<br>"
                "species=%{customdata[1]}<br>"
                f"chemical distance to {reference_sample}=%{{customdata[2]:.4f}}<br>"
                "nearest chemical distance=%{customdata[3]:.4f}<br>"
                f"distance-to-{reference_sample} percentile=%{{customdata[4]:.2f}}<br>"
                "Poincare radius=%{customdata[5]:.4f}<br>"
                "Poincare1=%{x:.4f}<br>"
                "Poincare2=%{y:.4f}<br>"
                "Poincare3=%{z:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=title,
        width=1040,
        height=820,
        margin={"l": 10, "r": 110, "t": 60, "b": 10},
        scene={
            "xaxis": {"title": "Poincare 1", "range": [-1.02, 1.02]},
            "yaxis": {"title": "Poincare 2", "range": [-1.02, 1.02]},
            "zaxis": {"title": "Poincare 3", "range": [-1.02, 1.02]},
            "aspectmode": "cube",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def plot_html_2d(
    merged: pd.DataFrame,
    *,
    title: str,
    reference_sample: str,
    cmin: float,
    cmax: float,
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line={"color": "rgba(75,85,99,0.55)", "width": 1.4},
            hoverinfo="skip",
            showlegend=False,
            name="Poincare disk boundary",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=merged["poincare1"],
            y=merged["poincare2"],
            mode="markers",
            marker={
                "size": 8,
                "color": merged[color_column(reference_sample)],
                "colorscale": "Viridis",
                "cmin": cmin,
                "cmax": cmax,
                "line": {"width": 0.6, "color": "white"},
                "colorbar": {"title": f"Distance to {reference_sample}"},
            },
            text=merged["sample_id"].astype(str),
            customdata=hover_customdata(merged, dim=2, reference_sample=reference_sample),
            hovertemplate=(
                "name=%{text}<br>"
                "genus=%{customdata[0]}<br>"
                "species=%{customdata[1]}<br>"
                f"chemical distance to {reference_sample}=%{{customdata[2]:.4f}}<br>"
                "nearest chemical distance=%{customdata[3]:.4f}<br>"
                f"distance-to-{reference_sample} percentile=%{{customdata[4]:.2f}}<br>"
                "Poincare radius=%{customdata[5]:.4f}<br>"
                "Poincare1=%{x:.4f}<br>"
                "Poincare2=%{y:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=title,
        width=820,
        height=760,
        margin={"l": 30, "r": 120, "t": 60, "b": 40},
        xaxis={"title": "Poincare 1", "range": [-1.03, 1.03], "zeroline": False},
        yaxis={
            "title": "Poincare 2",
            "range": [-1.03, 1.03],
            "scaleanchor": "x",
            "scaleratio": 1,
            "zeroline": False,
        },
        plot_bgcolor="white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def coordinate_radius_summary(merged: pd.DataFrame, dim: int, reference_sample: str) -> dict[str, float]:
    radius = np.linalg.norm(merged[[f"poincare{i}" for i in range(1, dim + 1)]].to_numpy(float), axis=1)
    column = color_column(reference_sample)
    return {
        "poincare_radius_min": float(radius.min()),
        "poincare_radius_median": float(np.median(radius)),
        "poincare_radius_mean": float(radius.mean()),
        "poincare_radius_max": float(radius.max()),
        f"spearman_radius_vs_chemical_distance_to_{reference_sample}": float(
            pd.Series(radius).corr(merged[column], method="spearman")
        ),
        f"pearson_radius_vs_chemical_distance_to_{reference_sample}": float(
            pd.Series(radius).corr(merged[column], method="pearson")
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.root
    reference_sample = args.reference_sample
    output_dir = root / f"hmds_chemical_distance_to_{reference_sample}_coloring"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    values = chemical_distance_values(root, reference_sample)
    cmin, cmax = color_limits(values, reference_sample)
    values.to_csv(table_dir / "chemical_distance_color_values.csv", index=False)

    summary_rows = []
    output_files: dict[str, str] = {}
    for spec in PLOT_SPECS:
        coordinate_path = root / spec["coordinate_path"]
        merged = merge_values(coordinate_path, values)
        merged.to_csv(table_dir / f"coordinates__{spec['key']}__chemical_distance_to_{reference_sample}_color.csv", index=False)
        title = f"{spec['title']} colored by chemical distance to {reference_sample}"

        png_path = figure_dir / f"{spec['key']}__chemical_distance_to_{reference_sample}.png"
        html_path = figure_dir / f"{spec['key']}__chemical_distance_to_{reference_sample}.html"
        if spec["dim"] == 3:
            plot_static_3d(merged, title=title, reference_sample=reference_sample, cmin=cmin, cmax=cmax, output_path=png_path)
            plot_html_3d(merged, title=title, reference_sample=reference_sample, cmin=cmin, cmax=cmax, output_path=html_path)
        else:
            plot_static_2d(merged, title=title, reference_sample=reference_sample, cmin=cmin, cmax=cmax, output_path=png_path)
            plot_html_2d(merged, title=title, reference_sample=reference_sample, cmin=cmin, cmax=cmax, output_path=html_path)

        output_files[f"{spec['key']}_png"] = str(png_path)
        output_files[f"{spec['key']}_html"] = str(html_path)
        summary_rows.append(
            {
                "embedding": spec["key"],
                "title": spec["title"],
                "dim": spec["dim"],
                **coordinate_radius_summary(merged, dim=int(spec["dim"]), reference_sample=reference_sample),
            }
        )

    pd.DataFrame(summary_rows).to_csv(table_dir / "chemical_distance_coloring_summary.csv", index=False)
    summary = {
        "input": {
            "root": str(root),
            "chemical_rdm": str(root / "tables" / CHEMICAL_RDM),
        },
        "color_definition": (
            f"Point color is chemical_distance_to_{reference_sample}: normalized chemical Euclidean RDM "
            f"distance from each strain to {reference_sample}. The chemical RDM is normalized to max=2 "
            "before this scalar is computed."
        ),
        "reference_sample": reference_sample,
        "color_range": {
            "min": cmin,
            "max": cmax,
        },
        "outputs": {
            "figures_dir": str(figure_dir),
            "tables_dir": str(table_dir),
            "chemical_distance_values": str(table_dir / "chemical_distance_color_values.csv"),
            "summary_csv": str(table_dir / "chemical_distance_coloring_summary.csv"),
            **output_files,
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Color existing HMDS embeddings by chemical distance to a reference sample.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference-sample", default=REFERENCE_SAMPLE)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
