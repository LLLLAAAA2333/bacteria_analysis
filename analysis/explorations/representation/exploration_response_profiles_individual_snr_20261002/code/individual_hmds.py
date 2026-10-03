"""Checked 2D HMDS views for the individual-SNR response exploration.

Neural uncertainty is supplied by the new whole-animal bootstrap, not by an
older response representation. Chemical coordinates and chemical PCo1 scores
are reused only after numerical agreement of their input RDM is verified.
This module does not alter the source Notebook or its saved analyses.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components


CHEMICAL_SOURCE = "output/jupyter-notebook/chemical_hmds_20260928_230847_298095"
NEURAL_SETTINGS = dict(
    n_starts=8, seed=20260919, initial_max_iter=3000,
    initial_lambda_bounds=(0.05, 10.0), variance_floor=1e-10,
    max_blocks=30, iterations_per_block=2000, gradient_tolerance=1e-6,
    perturb_amplitudes=(0.0, 0.1, 0.3, 0.7), perturb_seed=20260930,
    ridge_fraction=1e-7, max_profile_steps=30,
)


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=str) + "\n", encoding="utf-8")


def _matrix(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if not frame.index.is_unique or not frame.index.equals(frame.columns):
        raise ValueError(f"Matrix needs identical unique row/column IDs: {path}")
    return frame


def _coordinates(frame, ids):
    """Check the native disk coordinates and export a uniform plotting schema."""
    if (not frame.index.is_unique or set(frame.index) != set(ids)
            or list(frame.columns) != ["Poincare1", "Poincare2"]):
        raise ValueError("Expected two Poincare columns and exactly the current sample IDs")
    result = frame.reindex(ids).astype(float).rename(
        columns={"Poincare1": "x", "Poincare2": "y"})
    values = result.to_numpy()
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) > 1 + 1e-12):
        raise ValueError("HMDS display coordinates are nonfinite or outside the unit disk")
    result.index.name = "sample_id"
    return result


def _component_coverage(distance, mask, coefficients=None):
    """Select a connected observed graph without inventing missing distances."""
    ids = distance.index
    graph = mask.to_numpy(bool)
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    included = (labels == largest) if sizes[largest] >= 3 else np.zeros(len(ids), dtype=bool)
    degrees = graph.sum(axis=1)
    observed = np.isfinite(distance.to_numpy(float))
    np.fill_diagonal(observed, False)
    zero = np.zeros(len(ids), dtype=bool)
    if coefficients is not None:
        if not coefficients.index.is_unique or set(coefficients.index) != set(ids):
            raise ValueError("Coefficient identities differ from HMDS inputs")
        values = coefficients.reindex(ids).to_numpy(float)
        zero = np.isfinite(values).any(axis=1) & np.all(~np.isfinite(values) | (values == 0), axis=1)
    reason = np.full(len(ids), "outside_largest_connected_component", dtype=object)
    reason[degrees == 0] = "below_bootstrap_coverage"
    reason[~np.isfinite(np.diag(distance))] = "undefined_full_profile"
    reason[zero] = "full_zero_profile"
    reason[included] = "included"
    return pd.DataFrame(dict(n_edges=degrees, n_full_rdm_edges=observed.sum(axis=1),
                             component=labels, component_size=sizes[labels],
                             included=included, reason=reason),
                        index=pd.Index(ids, name="sample_id"))


def _reuse_chemical(out, repo_root, ids, current_rdm):
    """Reuse a checked chemical fit; never substitute it for changed distances."""
    source = repo_root / CHEMICAL_SOURCE
    saved_pairs = pd.read_csv(source / "fitted_pairs_2d.csv")
    required = {"sample_i", "sample_j", "input_distance", "fitted_distance"}
    if not required <= set(saved_pairs):
        raise ValueError("Saved chemical fit has no auditable input/fitted pair distances")
    saved_ids = set(saved_pairs.sample_i) | set(saved_pairs.sample_j)
    n = len(current_rdm)
    pairs = [tuple(sorted((a, b))) for a, b in zip(saved_pairs.sample_i, saved_pairs.sample_j)]
    if (saved_ids != set(current_rdm.index) or len(pairs) != n * (n - 1) // 2
            or len(set(pairs)) != len(pairs) or any(a == b for a, b in pairs)):
        raise ValueError("Saved chemical pair coverage differs from the current RDM")
    values = current_rdm.to_numpy(float)
    if (not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values, values.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(values), 0, rtol=0, atol=1e-12)):
        raise ValueError("Current chemical RDM must be finite, symmetric and nonnegative")
    expected = np.array([current_rdm.loc[a, b]
                         for a, b in zip(saved_pairs.sample_i, saved_pairs.sample_j)])
    error = float(np.max(np.abs(expected - saved_pairs.input_distance.to_numpy(float))))
    if not np.allclose(expected, saved_pairs.input_distance, rtol=1e-12, atol=1e-12):
        raise ValueError("Current chemical RDM differs from the saved fit; cannot reuse coordinates")
    summary = pd.read_csv(source / "summary.csv")
    selected = summary.loc[summary.dimension.eq(2)]
    if len(selected) != 1:
        raise ValueError("Saved chemical diagnostics do not identify exactly one 2D result")
    diagnostic = selected.iloc[0].to_dict()
    for key in ("converged", "gradient_checks_passed"):
        if diagnostic[key] not in (True, "True", "true", 1):
            raise ValueError(f"Saved chemical 2D fit did not pass {key}")
    coordinates = pd.read_csv(source / "embedding_2d.csv", index_col=0)
    coordinates.index = coordinates.index.astype(str)
    coordinates = _coordinates(coordinates, ids)

    color_source = source / "color_reference"
    color_reference = _matrix(color_source / "chemical_reference_rdm.csv")
    if set(color_reference.index) != set(current_rdm.index):
        raise ValueError("Frozen chemical color reference has a different sample scope")
    color_reference = color_reference.reindex(index=current_rdm.index, columns=current_rdm.columns)
    if not np.allclose(color_reference, current_rdm, rtol=1e-12, atol=1e-12):
        raise ValueError("Frozen chemical PCo1 scores were computed from a different RDM")
    colors = pd.read_csv(color_source / "aid_to_chemical_color.csv", index_col=0)
    colors.index = colors.index.astype(str)
    if not colors.index.is_unique or set(colors.index) != set(ids):
        raise ValueError("Frozen chemical color scores do not cover current sample IDs")
    colors = colors.reindex(ids)[["chemical_PCo1"]].astype(float)
    if not np.isfinite(colors.to_numpy()).all():
        raise ValueError("Nonfinite frozen chemical PCo1 scores")
    colors.index.name = "sample_id"
    color_parameters = json.loads((color_source / "color_parameters.json").read_text())

    target = out / "chemical"
    target.mkdir(parents=True, exist_ok=False)
    names = ["embedding_2d.csv", "fitted_pairs_2d.csv", "summary.csv",
             "gradient_summary_2d.csv", "gradient_steps_2d.csv"]
    for name in names:
        shutil.copyfile(source / name, target / name)
    coordinates.to_csv(out / "chemical_coordinates.csv")
    colors.to_csv(out / "sample_colors.csv")
    sources = [source / name for name in names] + [
        color_source / name for name in ("aid_to_chemical_color.csv",
                                         "chemical_reference_rdm.csv", "color_parameters.json")]
    provenance = dict(
        status="reused_checked_fit", converged=True, source_directory=str(source),
        source_sha256={str(path): _hash(path) for path in sources},
        input_max_abs_difference=error, diagnostics=diagnostic,
        coordinates=str(out / "chemical_coordinates.csv"),
        shepard=str(target / "fitted_pairs_2d.csv"),
        limitation="2D lambda is fixed at 10; shared sigma is model residual, not LC-MS measurement error",
    )
    _json(target / "provenance.json", provenance)
    color_limit = float(np.max(np.abs(colors.chemical_PCo1.to_numpy())))
    _json(out / "sample_colors_parameters.json", dict(
        source_directory=str(color_source), scalar="chemical_PCo1", display_cmap="RdBu_r",
        normalization="TwoSlopeNorm", vmin=-color_limit, vcenter=0., vmax=color_limit,
        observed_min=float(colors.chemical_PCo1.min()), observed_max=float(colors.chemical_PCo1.max()),
        positive_inertia_fraction=color_parameters["positive_inertia_fraction"],
        source_reference_sha256=color_parameters["reference_sha256"],
        note="Frozen chemical PCo1 scores, not old palette colors. Similar colors do not imply similar full profiles.",
    ))
    return provenance


def run_hmds(output_dir, repo_root, progress=print):
    """Fit neural 2D HMDS and reuse verified chemical 2D HMDS inside this report.

    Only the largest eligible connected neural component is fitted. Coverage
    and every excluded ID remain explicit, and chemical coordinates stay on the
    full reference sample set. A failed neural convergence check returns
    status=diagnostic_only and leaves fitter checkpoints in place.
    Existing HMDS output directories are not overwritten or silently resumed.
    """
    output_dir, repo_root = Path(output_dir).resolve(), Path(repo_root).resolve()
    tables, out = output_dir / "tables", output_dir / "hmds"
    paths = {name: tables / filename for name, filename in dict(
        distance="hmds_neural_distance_chord.csv", variance="hmds_neural_variance.csv",
        valid_fraction="hmds_neural_valid_fraction.csv", pair_mask="hmds_neural_pair_mask.csv",
        bootstrap="hmds_bootstrap_parameters.json", chemical="rdm_chemical.csv",
    ).items()}
    distance, variance, valid_fraction, mask = [_matrix(paths[name]) for name in
                                                ("distance", "variance", "valid_fraction", "pair_mask")]
    ids = distance.index
    if any(not frame.index.equals(ids) for frame in (variance, valid_fraction, mask)):
        raise ValueError("Neural HMDS matrices must have identical sample ordering")
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes):
        raise ValueError("Neural HMDS pair mask must be explicitly boolean")
    if not np.array_equal(mask, mask.T) or np.diag(mask).any():
        raise ValueError("Neural HMDS pair mask must be symmetric with a false diagonal")
    d, v, fraction = [frame.to_numpy(float) for frame in (distance, variance, valid_fraction)]
    retained = mask.to_numpy(bool)
    if (not np.allclose(d, d.T, equal_nan=True) or not np.allclose(v, v.T, equal_nan=True)
            or not np.allclose(fraction, fraction.T, equal_nan=True)
            or not np.isfinite(d[retained]).all() or not np.isfinite(v[retained]).all()
            or np.any(d[retained] < 0) or np.any(d[retained] > 2 + 1e-12)
            or np.any(v[retained] < 0)):
        raise ValueError("Invalid neural chord distances or bootstrap variances")
    parameters = json.loads(paths["bootstrap"].read_text())
    minimum_fraction = float(parameters.get("min_valid_fraction", .8))
    if (not np.isfinite(fraction).all() or np.any(fraction < 0) or np.any(fraction > 1)
            or np.any(fraction[retained] < minimum_fraction - 1e-12)):
        raise ValueError("Retained HMDS pairs violate bootstrap coverage")
    chemical = _matrix(paths["chemical"])
    if set(chemical.index) != set(ids):
        raise ValueError("Neural and chemical views need the same sample IDs")
    coefficient_path = tables / "strain_coefficients.csv"
    coefficients = pd.read_csv(coefficient_path, index_col=0) if coefficient_path.exists() else None
    coverage = _component_coverage(distance, mask, coefficients)
    fit_ids = coverage.index[coverage.included]
    excluded_ids = coverage.index[~coverage.included].tolist()
    out.mkdir(parents=True, exist_ok=False)
    coverage_path = out / "neural_sample_coverage.csv"
    coverage.to_csv(coverage_path)
    shutil.copyfile(paths["pair_mask"], out / "neural_original_pair_mask.csv")
    if progress is not None:
        progress("Verifying and reusing the checked chemical HMDS and frozen chemical PCo1 scores")
    result = dict(chemical=_reuse_chemical(out, repo_root, ids, chemical),
                  sample_colors=str(out / "sample_colors.csv"),
                  neural_sample_coverage=str(coverage_path))
    source_code = [repo_root / "notebook" / name for name in
                   ("neural_hmds.py", "hmds_refinement.py")]
    input_paths = list(paths.values()) + ([coefficient_path] if coefficient_path.exists() else [])
    provenance = dict(input_sha256={str(path): _hash(path) for path in input_paths},
                      code_sha256={str(path): _hash(path) for path in source_code + [Path(__file__)]},
                      n_full_samples=len(ids), n_fitted_samples=len(fit_ids), excluded_sample_ids=excluded_ids,
                      n_full_eligible_pairs=int(np.triu(retained, 1).sum()),
                      n_fitted_pairs=int(np.triu(mask.loc[fit_ids, fit_ids].to_numpy(bool), 1).sum()),
                      neural_coverage_rule="Largest connected component of the original >=80%-bootstrap eligible graph; minimum 3 samples; no imputation or lowered threshold",
                      bootstrap=parameters, neural_settings=NEURAL_SETTINGS,
                      distance="sqrt(2 * (1 - cosine)); bootstrap variance ddof=1, not divided by draw count")
    _json(out / "input_provenance.json", provenance)
    neural_scope = dict(n_full_samples=len(ids), n_samples=len(fit_ids),
                        excluded_sample_ids=excluded_ids, coverage=str(coverage_path))
    if len(fit_ids) < 3:
        result["neural"] = dict(neural_scope, status="diagnostic_only", converged=False,
                                reason="No eligible connected neural component contains at least three samples",
                                coordinates=None, shepard=None)
        _json(out / "result.json", result)
        return result
    notebook_dir = str(repo_root / "notebook")
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    from neural_hmds import fit_neural_hmds_2d

    neural_dir = out / "neural"
    if progress is not None:
        progress(f"Fitting neural 2D HMDS on {len(fit_ids)}/{len(ids)} samples in the largest eligible connected component")
    fit_parameters = dict(parameters, fitted_sample_ids=fit_ids.tolist(),
                          excluded_sample_ids=excluded_ids,
                          sample_selection="Largest connected component at the original bootstrap coverage threshold")
    try:
        bundle = fit_neural_hmds_2d(distance.loc[fit_ids, fit_ids], variance.loc[fit_ids, fit_ids],
                                    mask.loc[fit_ids, fit_ids], fit_parameters, neural_dir,
                                    **NEURAL_SETTINGS)
        report, fit = bundle["report"], bundle["fit"]
        if (not fit["diagnostics"].get("joint_gradient_converged", False)
                or not report["gradient_checks"].get("all_passed", False)):
            raise RuntimeError("HMDS returned without confirmed convergence and numerical-gradient checks")
    except RuntimeError as error:
        result["neural"] = dict(neural_scope, status="diagnostic_only", converged=False,
                                reason=str(error), output_directory=str(neural_dir),
                                coordinates=None, shepard=None,
                                diagnostic_files=sorted(str(path) for path in neural_dir.glob("*")))
        _json(out / "result.json", result)
        if progress is not None:
            progress(f"Neural HMDS did not pass its checks; diagnostics retained: {error}")
        return result
    coordinates = _coordinates(fit["display_coordinates"], fit_ids)
    coordinates.to_csv(out / "neural_coordinates.csv")
    result["neural"] = dict(neural_scope, status="checked_fit", converged=True,
                            coordinates=str(out / "neural_coordinates.csv"),
                            shepard=str(neural_dir / "fitted_pairs.csv"),
                            report=str(neural_dir / "report.json"),
                            convergence=str(neural_dir / "convergence.json"),
                            diagnostics=fit["diagnostics"],
                            limitation="Checked local solution; display radius is not a biological hierarchy or confidence")
    _json(out / "result.json", result)
    if progress is not None:
        progress("Saved checked neural HMDS and matched chemical HMDS coordinates")
    return result
