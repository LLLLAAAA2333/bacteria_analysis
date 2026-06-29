from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

import compare_86bac_chord_hmds as base


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
DEFAULT_TABLES_DIR = DEFAULT_ROOT / "tables"
DEFAULT_FIGURES_DIR = DEFAULT_ROOT / "figures"
NEURAL_RDM = "neural_correlation_distance_matrix.csv"
CHEMICAL_RDM = "chemical_rdm__qc20_missing50_log2_zscore_pca10_euclidean.csv"
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"


def read_rdm(path: Path) -> pd.DataFrame:
    rdm = pd.read_csv(path, index_col=0)
    rdm.index = rdm.index.astype(str)
    rdm.columns = rdm.columns.astype(str)
    if not rdm.index.equals(rdm.columns):
        raise ValueError(f"RDM index/columns do not match: {path}")
    return rdm.apply(pd.to_numeric, errors="coerce")


def align_rdms(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = [sample for sample in left.index.astype(str) if sample in set(right.index.astype(str))]
    if len(shared) < 4:
        raise ValueError(f"need at least 4 shared samples for 3D MDS; found {len(shared)}")
    return left.loc[shared, shared], right.loc[shared, shared]


def clean_distance_matrix(rdm: pd.DataFrame) -> np.ndarray:
    values = rdm.to_numpy(dtype=float, copy=True)
    if np.isnan(values).any():
        raise ValueError("RDM contains NaN values")
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    if np.any(values < -1e-10):
        raise ValueError("RDM contains negative distances")
    values[values < 0] = 0.0
    return values


def upper_triangle(values: np.ndarray) -> np.ndarray:
    return values[np.triu_indices_from(values, k=1)]


def classical_mds(distance: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    n_samples = distance.shape[0]
    squared = distance**2
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues[:n_components], 0.0, None)
    coordinates = eigenvectors[:, :n_components] * np.sqrt(positive)
    return coordinates, eigenvalues


def preservation_summary(distance: np.ndarray, coordinates: np.ndarray, eigenvalues: np.ndarray) -> dict[str, float]:
    original = upper_triangle(distance)
    embedded = pdist(coordinates)
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    positive_total = float(positive_eigenvalues.sum()) if positive_eigenvalues.size else 0.0
    first3 = float(np.clip(eigenvalues[:3], 0.0, None).sum())
    normalized_stress = float(np.sqrt(np.sum((original - embedded) ** 2) / np.sum(original**2)))
    return {
        "positive_eigenvalue_fraction_3d": first3 / positive_total if positive_total else float("nan"),
        "normalized_raw_stress": normalized_stress,
        "distance_spearman": float(spearmanr(original, embedded).statistic),
        "distance_pearson": float(pearsonr(original, embedded).statistic),
        "eigenvalue_1": float(eigenvalues[0]),
        "eigenvalue_2": float(eigenvalues[1]),
        "eigenvalue_3": float(eigenvalues[2]),
    }


def centered_unit(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(centered))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("cannot normalize degenerate coordinates")
    return centered / norm


def procrustes_align(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    reference_unit = centered_unit(reference)
    moving_unit = centered_unit(moving)
    rotation, _ = orthogonal_procrustes(moving_unit, reference_unit)
    moving_aligned = moving_unit @ rotation
    deltas = reference_unit - moving_aligned
    summary = {
        "rmse": float(np.sqrt(np.mean(np.sum(deltas**2, axis=1)))),
        "mean_point_distance": float(np.mean(np.linalg.norm(deltas, axis=1))),
        "median_point_distance": float(np.median(np.linalg.norm(deltas, axis=1))),
    }
    return reference_unit, moving_aligned, summary


def sample_number(sample_id: str) -> float:
    match = re.search(r"(\d+)", sample_id)
    return float(match.group(1)) if match else float("nan")


def set_equal_axes(axis, coordinates: np.ndarray) -> None:
    center = coordinates.mean(axis=0)
    radius = float(np.max(np.ptp(coordinates, axis=0)) / 2.0)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_static(
    coordinates: pd.DataFrame,
    *,
    output_path: Path,
    summary: dict[str, object],
    color_table: pd.DataFrame,
) -> None:
    samples = coordinates["sample_id"].astype(str).to_numpy()
    species_indices = coordinates["species_index"].to_numpy(float)
    neural = coordinates[["neural_mds1", "neural_mds2", "neural_mds3"]].to_numpy(float)
    chemical = coordinates[["chemical_mds1", "chemical_mds2", "chemical_mds3"]].to_numpy(float)
    neural_unit = coordinates[["neural_unit1", "neural_unit2", "neural_unit3"]].to_numpy(float)
    chemical_aligned = coordinates[
        ["chemical_aligned_to_neural1", "chemical_aligned_to_neural2", "chemical_aligned_to_neural3"]
    ].to_numpy(float)

    n_species = len(color_table)
    norm = plt.Normalize(vmin=-0.5, vmax=n_species - 0.5)

    figure = plt.figure(figsize=(15, 5.2), constrained_layout=True)
    axes = [
        figure.add_subplot(1, 3, 1, projection="3d"),
        figure.add_subplot(1, 3, 2, projection="3d"),
        figure.add_subplot(1, 3, 3, projection="3d"),
    ]

    scatter = axes[0].scatter(
        neural[:, 0], neural[:, 1], neural[:, 2],
        c=species_indices, cmap=plt.cm.turbo, norm=norm, s=30,
    )
    axes[0].set_title(
        "Neural RDM MDS\n"
        f"stress={summary['neural']['normalized_raw_stress']:.3f}; "
        f"rho={summary['neural']['distance_spearman']:.3f}",
        fontsize=10,
    )
    set_equal_axes(axes[0], neural)

    axes[1].scatter(
        chemical[:, 0], chemical[:, 1], chemical[:, 2],
        c=species_indices, cmap=plt.cm.turbo, norm=norm, s=30,
    )
    axes[1].set_title(
        "Chemical RDM MDS\n"
        f"stress={summary['chemical']['normalized_raw_stress']:.3f}; "
        f"rho={summary['chemical']['distance_spearman']:.3f}",
        fontsize=10,
    )
    set_equal_axes(axes[1], chemical)

    axes[2].scatter(neural_unit[:, 0], neural_unit[:, 1], neural_unit[:, 2], color="#2b6cb0", s=25, label="neural")
    axes[2].scatter(
        chemical_aligned[:, 0],
        chemical_aligned[:, 1],
        chemical_aligned[:, 2],
        color="#dd6b20",
        marker="^",
        s=25,
        label="chemical aligned",
    )
    for index in range(len(samples)):
        axes[2].plot(
            [neural_unit[index, 0], chemical_aligned[index, 0]],
            [neural_unit[index, 1], chemical_aligned[index, 1]],
            [neural_unit[index, 2], chemical_aligned[index, 2]],
            color="#718096",
            alpha=0.22,
            linewidth=0.6,
        )
    axes[2].set_title(
        "Procrustes overlay\n"
        f"point RMSE={summary['procrustes']['rmse']:.3f}",
        fontsize=10,
    )
    axes[2].legend(loc="upper right", fontsize=8)
    set_equal_axes(axes[2], np.vstack([neural_unit, chemical_aligned]))

    for axis in axes:
        axis.set_xlabel("MDS1", fontsize=8)
        axis.set_ylabel("MDS2", fontsize=8)
        axis.set_zlabel("MDS3", fontsize=8)
        axis.tick_params(labelsize=7)

    colorbar = figure.colorbar(scatter, ax=axes[:2], fraction=0.03, pad=0.02)
    base.set_species_colorbar(colorbar, color_table, label_size=7)
    figure.suptitle("86bac neural and chemical RDMs projected with 3D classical MDS, colored by species", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_interactive(
    coordinates: pd.DataFrame,
    *,
    output_path: Path,
    summary: dict[str, object],
    color_table: pd.DataFrame,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    samples = coordinates["sample_id"].astype(str)
    species_indices = coordinates["species_index"].astype(int)
    n_species = len(color_table)
    species_colorscale = base.plotly_species_colorscale(color_table)
    customdata_species = np.column_stack(
        [
            coordinates["genus"].astype(str),
            coordinates["species"].astype(str),
            (species_indices + 1).astype(str),
        ]
    )
    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Neural RDM MDS", "Chemical RDM MDS", "Procrustes overlay"),
    )

    figure.add_trace(
        go.Scatter3d(
            x=coordinates["neural_mds1"],
            y=coordinates["neural_mds2"],
            z=coordinates["neural_mds3"],
            mode="markers",
            marker={
                "size": 4,
                "color": species_indices,
                "colorscale": species_colorscale,
                "cmin": -0.5,
                "cmax": n_species - 0.5,
                "line": {"width": 0.3, "color": "white"},
                "colorbar": {"title": "Species"},
            },
            text=samples,
            customdata=customdata_species,
            hovertemplate=(
                "%{text}<br>"
                "genus=%{customdata[0]}<br>"
                "species=%{customdata[1]}<br>"
                "species color code=S%{customdata[2]}<br>"
                "MDS1=%{x:.3f}<br>MDS2=%{y:.3f}<br>MDS3=%{z:.3f}<extra>neural</extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter3d(
            x=coordinates["chemical_mds1"],
            y=coordinates["chemical_mds2"],
            z=coordinates["chemical_mds3"],
            mode="markers",
            marker={
                "size": 4,
                "color": species_indices,
                "colorscale": species_colorscale,
                "cmin": -0.5,
                "cmax": n_species - 0.5,
                "line": {"width": 0.3, "color": "white"},
                "showscale": False,
            },
            text=samples,
            customdata=customdata_species,
            hovertemplate=(
                "%{text}<br>"
                "genus=%{customdata[0]}<br>"
                "species=%{customdata[1]}<br>"
                "species color code=S%{customdata[2]}<br>"
                "MDS1=%{x:.3f}<br>MDS2=%{y:.3f}<br>MDS3=%{z:.3f}<extra>chemical</extra>"
            ),
        ),
        row=1,
        col=2,
    )

    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    z_lines: list[float | None] = []
    for _, row in coordinates.iterrows():
        x_lines.extend([row["neural_unit1"], row["chemical_aligned_to_neural1"], None])
        y_lines.extend([row["neural_unit2"], row["chemical_aligned_to_neural2"], None])
        z_lines.extend([row["neural_unit3"], row["chemical_aligned_to_neural3"], None])
    figure.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line={"color": "rgba(113,128,150,0.35)", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    figure.add_trace(
        go.Scatter3d(
            x=coordinates["neural_unit1"],
            y=coordinates["neural_unit2"],
            z=coordinates["neural_unit3"],
            mode="markers",
            marker={"size": 4, "color": "#2b6cb0"},
            name="neural",
            text=samples,
            hovertemplate="%{text}<extra>neural aligned</extra>",
        ),
        row=1,
        col=3,
    )
    figure.add_trace(
        go.Scatter3d(
            x=coordinates["chemical_aligned_to_neural1"],
            y=coordinates["chemical_aligned_to_neural2"],
            z=coordinates["chemical_aligned_to_neural3"],
            mode="markers",
            marker={"size": 4, "color": "#dd6b20", "symbol": "diamond"},
            name="chemical aligned",
            text=samples,
            hovertemplate="%{text}<extra>chemical aligned</extra>",
        ),
        row=1,
        col=3,
    )

    figure.update_layout(
        title=(
            "86bac RDM 3D classical MDS colored by species | "
            f"neural stress={summary['neural']['normalized_raw_stress']:.3f}, "
            f"chemical stress={summary['chemical']['normalized_raw_stress']:.3f}"
        ),
        width=1450,
        height=560,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")


def build_outputs(
    tables_dir: Path, figures_dir: Path
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    neural_path = tables_dir / NEURAL_RDM
    chemical_path = tables_dir / CHEMICAL_RDM
    neural_rdm, chemical_rdm = align_rdms(read_rdm(neural_path), read_rdm(chemical_path))

    taxonomy = pd.read_csv(tables_dir / TAXONOMY)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    taxonomy = taxonomy.set_index("AID", drop=False)
    shared = [s for s in neural_rdm.index.astype(str) if s in taxonomy.index]
    if len(shared) < 4:
        raise ValueError(f"need at least 4 shared samples with taxonomy; found {len(shared)}")
    neural_rdm = neural_rdm.loc[shared, shared]
    chemical_rdm = chemical_rdm.loc[shared, shared]
    taxonomy_aligned = taxonomy.loc[shared].copy()
    sample_table = base.build_sample_table(shared, taxonomy_aligned)
    color_table = base.species_color_table(taxonomy_aligned)
    merged = sample_table.merge(
        color_table[["species", "species_index"]], on="species", how="left"
    )

    neural_distance = clean_distance_matrix(neural_rdm)
    chemical_distance = clean_distance_matrix(chemical_rdm)

    neural_coordinates, neural_eigenvalues = classical_mds(neural_distance)
    chemical_coordinates, chemical_eigenvalues = classical_mds(chemical_distance)
    neural_unit, chemical_aligned, procrustes_summary = procrustes_align(neural_coordinates, chemical_coordinates)

    samples = neural_rdm.index.astype(str).tolist()
    coordinates = pd.DataFrame(
        {
            "sample_id": samples,
            "sample_number": [sample_number(sample) for sample in samples],
            "genus": merged["genus"].astype(str).to_numpy(),
            "species": merged["species"].astype(str).to_numpy(),
            "species_index": merged["species_index"].astype(int).to_numpy(),
            "neural_mds1": neural_coordinates[:, 0],
            "neural_mds2": neural_coordinates[:, 1],
            "neural_mds3": neural_coordinates[:, 2],
            "chemical_mds1": chemical_coordinates[:, 0],
            "chemical_mds2": chemical_coordinates[:, 1],
            "chemical_mds3": chemical_coordinates[:, 2],
            "neural_unit1": neural_unit[:, 0],
            "neural_unit2": neural_unit[:, 1],
            "neural_unit3": neural_unit[:, 2],
            "chemical_aligned_to_neural1": chemical_aligned[:, 0],
            "chemical_aligned_to_neural2": chemical_aligned[:, 1],
            "chemical_aligned_to_neural3": chemical_aligned[:, 2],
        }
    )
    summary: dict[str, object] = {
        "input": {
            "tables_dir": str(tables_dir),
            "neural_rdm": str(neural_path),
            "chemical_rdm": str(chemical_path),
            "taxonomy": str(tables_dir / TAXONOMY),
        },
        "n_samples": len(samples),
        "n_species": int(taxonomy_aligned["species"].nunique()),
        "coloring": "Points colored by species (evenly-spaced turbo colors)",
        "method": "classical metric MDS from precomputed RDM distances",
        "neural": preservation_summary(neural_distance, neural_coordinates, neural_eigenvalues),
        "chemical": preservation_summary(chemical_distance, chemical_coordinates, chemical_eigenvalues),
        "procrustes": procrustes_summary,
        "outputs": {
            "coordinates_csv": str(figures_dir / "rdm_mds_3d_coordinates.csv"),
            "summary_json": str(figures_dir / "rdm_mds_3d_summary.json"),
            "static_png": str(figures_dir / "rdm_mds_3d_neural_chemical.png"),
            "interactive_html": str(figures_dir / "rdm_mds_3d_neural_chemical.html"),
            "species_color_map": str(figures_dir / "species_color_map__rdm_mds.csv"),
        },
    }
    return coordinates, summary, color_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Project neural and chemical RDMs to 3D with classical MDS.")
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--skip-html", action="store_true")
    args = parser.parse_args()

    coordinates, summary, color_table = build_outputs(args.tables_dir, args.figures_dir)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(args.figures_dir / "rdm_mds_3d_coordinates.csv", index=False)
    color_table.to_csv(args.figures_dir / "species_color_map__rdm_mds.csv", index=False)
    (args.figures_dir / "rdm_mds_3d_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_static(
        coordinates,
        output_path=args.figures_dir / "rdm_mds_3d_neural_chemical.png",
        summary=summary,
        color_table=color_table,
    )
    if not args.skip_html:
        plot_interactive(
            coordinates,
            output_path=args.figures_dir / "rdm_mds_3d_neural_chemical.html",
            summary=summary,
            color_table=color_table,
        )


if __name__ == "__main__":
    main()
