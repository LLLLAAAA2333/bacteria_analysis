from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_86bac_chord_hmds as base


DEFAULT_ROOT = Path("results") / "86bac_shape_pca_rsa_t05_t24_silent_scale1"
CHEMICAL_RDM = "chemical_rdm__qc20_missing50_log2_zscore_pca10_euclidean.csv"
TAXONOMY = "taxonomy_from_GM300/86bac_sample_species_mapping.csv"


OBSOLETE_PATTERNS = (
    "*pseudo_chord*",
    "poincare_ball__chemical_pseudo_chord_hmds_3d.*",
    "poincare_projections__chemical_pseudo_chord_hmds.png",
)


def cleanup_obsolete_outputs(output_dir: Path) -> None:
    for folder_name in ("figures", "tables"):
        folder = output_dir / folder_name
        if not folder.exists():
            continue
        for pattern in OBSOLETE_PATTERNS:
            for path in folder.glob(pattern):
                if path.is_file():
                    path.unlink()


def result_metrics(
    original: np.ndarray,
    embedded: np.ndarray,
    predicted: np.ndarray,
    *,
    n_params: float,
) -> dict[str, float | int | str | None]:
    metrics = base.preservation_metrics(original, embedded, predicted, n_params=n_params)
    metrics.update(
        {
            "analysis_label": "chemical",
            "hmds_distance_label": "Euclidean",
        }
    )
    return metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.root
    tables_dir = root / "tables"
    output_dir = root / "chemical_hmds_comparison"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    cleanup_obsolete_outputs(output_dir)

    chemical_path = tables_dir / CHEMICAL_RDM
    chemical_rdm = base.read_rdm(chemical_path)
    taxonomy = pd.read_csv(tables_dir / TAXONOMY)
    taxonomy["AID"] = taxonomy["AID"].astype(str)
    taxonomy = taxonomy.set_index("AID", drop=False)
    samples = [sample for sample in chemical_rdm.index.astype(str) if sample in taxonomy.index]
    if len(samples) < 4:
        raise ValueError(f"need at least 4 samples with taxonomy; found {len(samples)}")

    chemical_rdm = chemical_rdm.loc[samples, samples]
    taxonomy = taxonomy.loc[samples].copy()
    sample_table = base.build_sample_table(samples, taxonomy)
    color_table = base.species_color_table(taxonomy)
    color_table.to_csv(table_dir / "species_color_map.csv", index=False)

    chemical_raw = base.clean_distance_matrix(chemical_rdm)
    euclidean = base.normalize_to_max_two(chemical_raw)
    pd.DataFrame(euclidean, index=samples, columns=samples).to_csv(
        table_dir / "chemical_distance__euclidean_normalized.csv"
    )

    results: list[base.EmbeddingResult] = []
    coords, embedded = base.euclidean_metric_mds(
        euclidean,
        dim=args.dim,
        seed=args.seed,
        n_init=args.mds_n_init,
    )
    metrics = result_metrics(
        euclidean,
        embedded,
        embedded,
        n_params=euclidean.shape[0] * args.dim + 1,
    )
    metrics.update({"lambda": None, "backend": "sklearn_metric_mds"})
    results.append(
        base.EmbeddingResult(
            method="chemical_euclidean_mds_3d",
            input_distance_name="chemical_euclidean_normalized",
            coordinates=coords,
            embedded_distance=embedded,
            predicted_distance=embedded,
            metrics=metrics,
        )
    )

    hmds_backend_status = "not_attempted"
    try:
        bayesian = base.run_bayesian_hmds_if_available(
            euclidean,
            dim=args.dim,
            trials=args.hmds_trials,
            bayesian_hmds=args.bayesian_hmds,
        )
        if bayesian is None:
            hmds_backend_status = "pystan_unavailable_or_bayesian_hmds_missing; used scipy fallback"
            coords_lorentz, embedded_hyp, lambda_value, metadata = base.scipy_hyperbolic_mds(
                euclidean,
                dim=args.dim,
                starts=args.hmds_starts,
                maxiter=args.hmds_maxiter,
                seed=args.seed,
            )
        else:
            hmds_backend_status = "BayesianHMDS_pystan"
            coords_lorentz, embedded_hyp, lambda_value, metadata = bayesian
        poincare = base.recenter_poincare(base.lorentz_to_poincare(coords_lorentz))
        predicted = embedded_hyp / lambda_value
        metrics = result_metrics(
            euclidean,
            embedded_hyp,
            predicted,
            n_params=euclidean.shape[0] * args.dim + 1 - args.dim * (args.dim - 1) / 2,
        )
        metrics.update({"lambda": float(lambda_value), **metadata})
        results.append(
            base.EmbeddingResult(
                method="chemical_hmds_poincare_3d",
                input_distance_name="chemical_euclidean_normalized",
                coordinates=poincare,
                embedded_distance=embedded_hyp,
                predicted_distance=predicted,
                metrics=metrics,
            )
        )
    except Exception as exc:
        hmds_backend_status = f"failed: {exc}"

    summary_rows = []
    for result in results:
        base.save_coordinates(result, sample_table, table_dir / f"coordinates__{result.method}.csv")
        base.save_pair_distances(result, euclidean, samples, table_dir / f"pair_distances__{result.method}.csv")
        base.plot_3d_embedding(result, sample_table, color_table, figure_dir / f"embedding__{result.method}.png")
        base.plot_shepard(result, euclidean, figure_dir / f"shepard__{result.method}.png")
        if "hmds" in result.method:
            base.plot_poincare_projections(
                result,
                sample_table,
                color_table,
                figure_dir / "poincare_projections__chemical_hmds.png",
            )
            base.plot_poincare_ball_3d(
                result,
                sample_table,
                color_table,
                figure_dir / "poincare_ball__chemical_hmds_3d.png",
            )
            base.plot_poincare_ball_3d_html(
                result,
                sample_table,
                color_table,
                figure_dir / "poincare_ball__chemical_hmds_3d.html",
            )
        summary_rows.append(
            {
                "method": result.method,
                "input_distance": result.input_distance_name,
                **result.metrics,
            }
        )

    summary = {
        "input": {
            "root": str(root),
            "chemical_rdm": str(chemical_path),
            "taxonomy": str(tables_dir / TAXONOMY),
            "bayesian_hmds": str(args.bayesian_hmds),
        },
        "parameters": {
            "dim": int(args.dim),
            "seed": int(args.seed),
            "mds_n_init": int(args.mds_n_init),
            "hmds_trials": int(args.hmds_trials),
            "hmds_starts": int(args.hmds_starts),
            "hmds_maxiter": int(args.hmds_maxiter),
        },
        "note": "The chemical RDM is a PCA Euclidean distance matrix; no chord transform is applied.",
        "n_samples": len(samples),
        "n_species": int(taxonomy["species"].nunique()),
        "hmds_backend_status": hmds_backend_status,
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
    parser = argparse.ArgumentParser(description="Compare 86bac chemical Euclidean RDM with Euclidean MDS and HMDS.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bayesian-hmds", type=Path, default=base.DEFAULT_BAYESIAN_HMDS)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--mds-n-init", type=int, default=8)
    parser.add_argument("--hmds-trials", type=int, default=3)
    parser.add_argument("--hmds-starts", type=int, default=6)
    parser.add_argument("--hmds-maxiter", type=int, default=700)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
