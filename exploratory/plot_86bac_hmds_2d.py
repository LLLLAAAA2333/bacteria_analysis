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
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"


def load_taxonomy(tables_dir: Path) -> pd.DataFrame:
    taxonomy = pd.read_csv(tables_dir / TAXONOMY)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    return taxonomy.set_index("AID", drop=False)


def aligned_sample_table(rdm: pd.DataFrame, taxonomy: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    samples = [sample for sample in rdm.index.astype(str) if sample in taxonomy.index]
    if len(samples) < 4:
        raise ValueError(f"need at least 4 samples with taxonomy; found {len(samples)}")
    taxonomy_aligned = taxonomy.loc[samples].copy()
    return samples, taxonomy_aligned, base.build_sample_table(samples, taxonomy_aligned)


def species_color_table(taxonomy: pd.DataFrame) -> pd.DataFrame:
    ordered = taxonomy.copy()
    ordered["_aid_sort_key"] = ordered["AID"].map(base.aid_sort_key)
    ordered = ordered.sort_values("_aid_sort_key")
    labels = list(dict.fromkeys(ordered["species"].astype(str)))
    first_aid = ordered.groupby("species", sort=False)["AID"].first()
    genus = ordered.groupby("species", sort=False)["genus"].first()
    counts = ordered["species"].value_counts()
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(labels)))
    return pd.DataFrame(
        {
            "species": labels,
            "species_index": np.arange(len(labels)),
            "genus": genus.loc[labels].to_numpy(str),
            "first_AID": first_aid.loc[labels].to_numpy(str),
            "n_strains": counts.loc[labels].to_numpy(int),
            "color_hex": [base.color_to_hex(color) for color in colors],
        }
    )


def plotly_discrete_colorscale(color_table: pd.DataFrame) -> list[list[float | str]]:
    n_colors = len(color_table)
    colorscale: list[list[float | str]] = []
    for index, color_hex in enumerate(color_table["color_hex"]):
        colorscale.append([index / n_colors, color_hex])
        colorscale.append([(index + 1) / n_colors, color_hex])
    return colorscale


def radius_summary(coords: np.ndarray) -> dict[str, float]:
    radius = np.linalg.norm(coords, axis=1)
    hyperbolic_radius = 2.0 * np.arctanh(np.clip(radius, 0.0, 0.999999))
    return {
        "poincare_radius_min": float(radius.min()),
        "poincare_radius_median": float(np.median(radius)),
        "poincare_radius_mean": float(radius.mean()),
        "poincare_radius_q90": float(np.quantile(radius, 0.90)),
        "poincare_radius_q95": float(np.quantile(radius, 0.95)),
        "poincare_radius_max": float(radius.max()),
        "hyperbolic_radius_median": float(np.median(hyperbolic_radius)),
        "fraction_radius_gt_0_90": float(np.mean(radius > 0.90)),
        "fraction_radius_gt_0_95": float(np.mean(radius > 0.95)),
    }


def fit_hmds_2d(
    distance: np.ndarray,
    *,
    method: str,
    input_distance_name: str,
    analysis_label: str,
    hmds_distance_label: str,
    args: argparse.Namespace,
) -> base.EmbeddingResult:
    bayesian = base.run_bayesian_hmds_if_available(
        distance,
        dim=2,
        trials=args.hmds_trials,
        bayesian_hmds=args.bayesian_hmds,
    )
    if bayesian is None:
        coords_lorentz, embedded_hyp, lambda_value, metadata = base.scipy_hyperbolic_mds(
            distance,
            dim=2,
            starts=args.hmds_starts,
            maxiter=args.hmds_maxiter,
            seed=args.seed,
        )
    else:
        coords_lorentz, embedded_hyp, lambda_value, metadata = bayesian

    poincare = base.recenter_poincare(base.lorentz_to_poincare(coords_lorentz))
    predicted = embedded_hyp / lambda_value
    n_params = distance.shape[0] * 2 + 1 - 2 * (2 - 1) / 2
    metrics = base.preservation_metrics(distance, embedded_hyp, predicted, n_params=n_params)
    metrics.update(
        {
            "lambda": float(lambda_value),
            "analysis_label": analysis_label,
            "hmds_distance_label": hmds_distance_label,
            **metadata,
            **radius_summary(poincare),
        }
    )
    return base.EmbeddingResult(
        method=method,
        input_distance_name=input_distance_name,
        coordinates=poincare,
        embedded_distance=embedded_hyp,
        predicted_distance=predicted,
        metrics=metrics,
    )


def save_coordinates(result: base.EmbeddingResult, sample_table: pd.DataFrame, output_path: Path) -> None:
    table = sample_table.copy()
    table["poincare1"] = result.coordinates[:, 0]
    table["poincare2"] = result.coordinates[:, 1]
    table["poincare_radius"] = np.linalg.norm(result.coordinates, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)


def set_disk_axis(axis) -> None:
    axis.set_xlim(-1.03, 1.03)
    axis.set_ylim(-1.03, 1.03)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Poincare 1", fontsize=9)
    axis.set_ylabel("Poincare 2", fontsize=9)
    axis.grid(True, color="#e5e7eb", linewidth=0.6)


def plot_static_disk(
    result: base.EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
) -> None:
    merged = sample_table.merge(color_table[["species", "species_index"]], on="species", how="left")
    figure, axis = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    circle = plt.Circle((0, 0), 1.0, color="#4b5563", fill=False, alpha=0.55, linewidth=1.0)
    axis.add_artist(circle)
    norm = plt.Normalize(vmin=-0.5, vmax=len(color_table) - 0.5)
    scatter = axis.scatter(
        result.coordinates[:, 0],
        result.coordinates[:, 1],
        c=merged["species_index"],
        cmap=plt.cm.turbo,
        norm=norm,
        s=44,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.45,
    )
    set_disk_axis(axis)
    axis.set_title(
        f"{base.analysis_label(result).title()} {base.hmds_distance_label(result)} HMDS 2D",
        fontsize=12,
    )
    figure.suptitle(
        f"86bac 2D Poincare disk colored by species\n"
        f"stress={result.metrics['normalized_raw_stress']:.3f}; "
        f"rho={result.metrics['distance_spearman']:.3f}; "
        f"lambda={result.metrics['lambda']:.3f}",
        fontsize=13,
    )
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.03)
    ticks = np.linspace(0, len(color_table) - 1, 6, dtype=int)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"S{tick + 1:02d}" for tick in ticks])
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Species color code", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_html_disk(
    result: base.EmbeddingResult,
    sample_table: pd.DataFrame,
    color_table: pd.DataFrame,
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    merged = sample_table.merge(
        color_table[["species", "species_index", "color_hex", "first_AID"]],
        on="species",
        how="left",
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    customdata = np.column_stack(
        [
            merged["genus"].astype(str),
            merged["species"].astype(str),
            (merged["species_index"].astype(int) + 1).astype(str),
            merged["first_AID"].astype(str),
            np.linalg.norm(result.coordinates, axis=1),
        ]
    )
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
            x=result.coordinates[:, 0],
            y=result.coordinates[:, 1],
            mode="markers",
            marker={
                "size": 8,
                "color": merged["species_index"],
                "colorscale": plotly_discrete_colorscale(color_table),
                "cmin": -0.5,
                "cmax": len(color_table) - 0.5,
                "line": {"width": 0.6, "color": "white"},
                "colorbar": {
                    "title": "Species",
                    "tickmode": "array",
                    "tickvals": np.linspace(0, len(color_table) - 1, 6, dtype=int),
                    "ticktext": [f"S{tick + 1:02d}" for tick in np.linspace(0, len(color_table) - 1, 6, dtype=int)],
                    "tickfont": {"size": 10},
                },
            },
            text=merged["sample_id"].astype(str),
            customdata=customdata,
            hovertemplate=(
                "name=%{text}<br>"
                "genus=%{customdata[0]}<br>"
                "species=%{customdata[1]}<br>"
                "species color code=S%{customdata[2]}<br>"
                "species first AID=%{customdata[3]}<br>"
                "Poincare radius=%{customdata[4]:.4f}<br>"
                "Poincare1=%{x:.4f}<br>"
                "Poincare2=%{y:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=(
            f"86bac {base.analysis_label(result)} {base.hmds_distance_label(result).lower()} HMDS 2D"
            f"<br>stress={result.metrics['normalized_raw_stress']:.3f}, "
            f"rho={result.metrics['distance_spearman']:.3f}, "
            f"lambda={result.metrics['lambda']:.3f}"
        ),
        width=820,
        height=760,
        margin={"l": 30, "r": 120, "t": 70, "b": 40},
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


def prepare_inputs(root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    tables_dir = root / "tables"
    taxonomy = load_taxonomy(tables_dir)

    neural_rdm, neural_path = base.read_neural_rdm(tables_dir)
    neural_samples, neural_taxonomy, neural_sample_table = aligned_sample_table(neural_rdm, taxonomy)
    neural_rdm = neural_rdm.loc[neural_samples, neural_samples]
    neural_distance = base.normalize_to_max_two(
        base.chord_from_linear(base.clean_distance_matrix(neural_rdm))
    )

    chemical_path = tables_dir / CHEMICAL_RDM
    chemical_rdm = base.read_rdm(chemical_path)
    chemical_samples, chemical_taxonomy, chemical_sample_table = aligned_sample_table(chemical_rdm, taxonomy)
    chemical_rdm = chemical_rdm.loc[chemical_samples, chemical_samples]
    chemical_distance = base.normalize_to_max_two(base.clean_distance_matrix(chemical_rdm))

    return (
        {
            "neural": neural_distance,
            "chemical": chemical_distance,
        },
        {
            "neural_path": neural_path,
            "chemical_path": chemical_path,
            "taxonomy_path": tables_dir / TAXONOMY,
            "neural_samples": neural_samples,
            "chemical_samples": chemical_samples,
            "neural_taxonomy": neural_taxonomy,
            "chemical_taxonomy": chemical_taxonomy,
            "neural_sample_table": neural_sample_table,
            "chemical_sample_table": chemical_sample_table,
        },
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.root
    output_dir = root / "hmds_2d_comparison"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    distances, metadata = prepare_inputs(root)
    datasets = [
        (
            "neural",
            "neural_chord_hmds_poincare_2d",
            "neural_chord_normalized",
            "neural",
            "Chord",
            distances["neural"],
            metadata["neural_samples"],
            metadata["neural_taxonomy"],
            metadata["neural_sample_table"],
        ),
        (
            "chemical",
            "chemical_euclidean_hmds_poincare_2d",
            "chemical_euclidean_normalized",
            "chemical",
            "Euclidean",
            distances["chemical"],
            metadata["chemical_samples"],
            metadata["chemical_taxonomy"],
            metadata["chemical_sample_table"],
        ),
    ]

    summary_rows = []
    backend_status: dict[str, str] = {}
    for prefix, method, input_name, analysis_label, distance_label, distance, samples, taxonomy, sample_table in datasets:
        color_table = species_color_table(taxonomy)
        color_table.to_csv(table_dir / f"species_color_map__{prefix}.csv", index=False)
        pd.DataFrame(distance, index=samples, columns=samples).to_csv(table_dir / f"distance__{input_name}.csv")

        try:
            result = fit_hmds_2d(
                distance,
                method=method,
                input_distance_name=input_name,
                analysis_label=analysis_label,
                hmds_distance_label=distance_label,
                args=args,
            )
            backend_status[prefix] = str(result.metrics.get("backend"))
        except Exception as exc:
            backend_status[prefix] = f"failed: {exc}"
            continue

        save_coordinates(result, sample_table, table_dir / f"coordinates__{method}.csv")
        base.save_pair_distances(result, distance, samples, table_dir / f"pair_distances__{method}.csv")
        base.plot_shepard(result, distance, figure_dir / f"shepard__{method}.png")
        plot_static_disk(result, sample_table, color_table, figure_dir / f"poincare_disk__{method}.png")
        plot_html_disk(result, sample_table, color_table, figure_dir / f"poincare_disk__{method}.html")
        summary_rows.append({"method": method, "input_distance": input_name, **result.metrics})

    summary = {
        "input": {
            "root": str(root),
            "neural_rdm": str(metadata["neural_path"]),
            "chemical_rdm": str(metadata["chemical_path"]),
            "taxonomy": str(metadata["taxonomy_path"]),
            "bayesian_hmds": str(args.bayesian_hmds),
        },
        "parameters": {
            "dim": 2,
            "seed": int(args.seed),
            "hmds_trials": int(args.hmds_trials),
            "hmds_starts": int(args.hmds_starts),
            "hmds_maxiter": int(args.hmds_maxiter),
        },
        "distance_inputs": {
            "neural": "sqrt(2 * neural correlation distance), normalized to max=2",
            "chemical": "chemical PCA Euclidean RDM, normalized to max=2; no chord transform",
        },
        "coloring": "Points are colored by species; full species color mappings are saved as species_color_map__*.csv.",
        "backend_status": backend_status,
        "outputs": {
            "summary_csv": str(table_dir / "embedding_comparison_summary.csv"),
            "summary_json": str(output_dir / "run_summary.json"),
            "figures_dir": str(figure_dir),
            "tables_dir": str(table_dir),
        },
    }
    pd.DataFrame(summary_rows).to_csv(table_dir / "embedding_comparison_summary.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit 2D HMDS embeddings for 86bac neural and chemical RDMs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bayesian-hmds", type=Path, default=base.DEFAULT_BAYESIAN_HMDS)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--hmds-trials", type=int, default=3)
    parser.add_argument("--hmds-starts", type=int, default=8)
    parser.add_argument("--hmds-maxiter", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
