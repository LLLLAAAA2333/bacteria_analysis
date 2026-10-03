"""Checked 3D HMDS on exactly the existing 2D neural sample/pair scope.

The existing 0.5-SNR distances, bootstrap variances and 2D fit are inputs, not
recomputed results. A 2D half-space solution is lifted to 3D and all coordinates,
positive lambda and nonnegative residual variances are jointly refined using
the repository's HMDS routines. Chemical 3D results are reused only after their
full 106-sample input distances and saved numerical checks are verified.
"""
from pathlib import Path
import copy
import hashlib
import json
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from individual_hmds import CHEMICAL_SOURCE, _hash, _json, _matrix


SETTINGS = dict(start_amplitudes=(.1, .3, .7), start_seed=20260919,
                perturb_amplitudes=(0., .1, .3, .7), perturb_seed=20260930,
                max_blocks=20, iterations_per_block=1000,
                gradient_tolerance=1e-6, perturb_gradient_tolerance=1e-7,
                ridge_fraction=1e-7, gradient_metric="local_geometric")
PAIR_COLUMNS = ["sample_i", "sample_j", "input_distance", "fitted_distance"]


def _relative_rmse(pairs):
    observed, fitted = pairs.input_distance.to_numpy(float), pairs.fitted_distance.to_numpy(float)
    return float(np.linalg.norm(fitted-observed) / np.linalg.norm(observed))


def _coordinate_table(frame, ids):
    if not frame.index.is_unique or not set(ids) <= set(frame.index):
        raise ValueError("3D coordinates do not cover unique selected sample IDs")
    if list(frame.columns) != ["Poincare1", "Poincare2", "Poincare3"]:
        raise ValueError("Expected exactly three Poincare coordinates")
    result = frame.reindex(ids).astype(float).rename(columns=dict(Poincare1="x", Poincare2="y", Poincare3="z"))
    values = result.to_numpy()
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) > 1 + 1e-12):
        raise ValueError("Nonfinite display coordinates or coordinates outside the unit ball")
    result.index.name = "sample_id"
    return result


def _verify_chemical_cache(out, repo_root, selected_ids, current_rdm):
    source = repo_root / CHEMICAL_SOURCE
    pairs = pd.read_csv(source / "fitted_pairs_3d.csv")
    if list(pairs.columns) != PAIR_COLUMNS:
        raise ValueError("Chemical cache does not have explicit input/fitted pair distances")
    full_ids = current_rdm.index
    pair_ids = [tuple(sorted(pair)) for pair in zip(pairs.sample_i, pairs.sample_j)]
    if (set(pairs.sample_i) | set(pairs.sample_j) != set(full_ids)
            or len(pair_ids) != len(full_ids)*(len(full_ids)-1)//2
            or len(set(pair_ids)) != len(pair_ids) or any(a == b for a, b in pair_ids)):
        raise ValueError("Chemical 3D cache and current RDM have different pair coverage")
    expected = np.array([current_rdm.loc[a, b] for a, b in zip(pairs.sample_i, pairs.sample_j)])
    error = float(np.max(np.abs(expected-pairs.input_distance.to_numpy(float))))
    if not np.allclose(expected, pairs.input_distance, rtol=1e-12, atol=1e-12):
        raise ValueError("Chemical 3D input distances differ from current RDM")
    summary = pd.read_csv(source / "summary.csv")
    row = summary.loc[summary.dimension.eq(3)]
    if len(row) != 1:
        raise ValueError("Exactly one checked chemical 3D summary is required")
    diagnostics = row.iloc[0].to_dict()
    for key in ("converged", "gradient_checks_passed"):
        if diagnostics[key] not in (True, "True", "true", 1):
            raise ValueError(f"Chemical 3D cache did not pass {key}")
    gradient = pd.read_csv(source / "gradient_summary_3d.csv")
    if len(gradient) != int(diagnostics["checked_parameters"]) or not gradient.adjacent_steps_agree.all():
        raise ValueError("Chemical 3D gradient summary does not substantiate its diagnostics")
    frame = pd.read_csv(source / "embedding_3d.csv", index_col=0)
    frame.index = frame.index.astype(str)
    if set(frame.index) != set(full_ids):
        raise ValueError("Chemical 3D coordinates have different full-fit sample scope")
    coordinates = _coordinate_table(frame, selected_ids)
    selected = pairs.loc[pairs.sample_i.isin(selected_ids) & pairs.sample_j.isin(selected_ids)].copy()
    if len(selected) != len(selected_ids)*(len(selected_ids)-1)//2:
        raise ValueError("Chemical 3D selected scope is not complete")
    out.mkdir(parents=True, exist_ok=False)
    coordinates.to_csv(out / "coordinates.csv")
    selected.to_csv(out / "pairs.csv", index=False)
    names = ["embedding_3d.csv", "fitted_pairs_3d.csv", "summary.csv", "gradient_summary_3d.csv", "gradient_steps_3d.csv"]
    for name in names:
        shutil.copyfile(source / name, out / name)
    report = dict(status="reused_checked_fit", converged=True, dimension=3,
                   n_full_samples=len(full_ids), n_samples=len(selected_ids), n_full_pairs=len(pairs), n_pairs=len(selected),
                   coordinates=str(out / "coordinates.csv"), shepard=str(out / "pairs.csv"),
                   input_max_abs_difference=error, diagnostics=diagnostics,
                   displayed_relative_rmse=_relative_rmse(selected),
                   full_relative_rmse=_relative_rmse(pairs), source_directory=str(source),
                   source_sha256={str(source / name): _hash(source / name) for name in names},
                   input_distance_unit="RMS log2 fold change", fitted_distance_unit="RMS log2 fold change; saved scaled hyperbolic distances",
                   sample_selection="Display exactly the existing neural 2D 81-sample scope; underlying chemical fit retains all 106 samples",
                   limitation="Reused checked local chemical 3D fit; shared residual sigma is not LC-MS measurement error")
    _json(out / "report.json", report)
    return report


def _checked_neural_3d(distance, variance, mask, baseline_bundle, out, progress):
    import neural_hmds
    from hmds_refinement import reanchor_hmds, refine_hmds

    baseline, base_reference = copy.deepcopy(baseline_bundle["fit"]), copy.deepcopy(baseline_bundle["reference"])
    if not baseline["diagnostics"].get("joint_gradient_converged", False):
        raise ValueError("The 2D warm start has not passed joint convergence")
    if not baseline_bundle["report"]["gradient_checks"].get("all_passed", False):
        raise ValueError("The 2D warm start has not passed numerical-gradient checks")
    ids = base_reference["coordinates"].index
    n = len(ids)
    if set(ids) != set(distance.index) or baseline["halfspace_coordinates"].shape != (n, 2):
        raise ValueError("The checked 2D solution has a different sample scope")
    pair_columns = ["sample_i", "sample_j", "input_distance", "bootstrap_variance"]
    pairs = base_reference["pairs"][pair_columns].copy()
    if len(pairs) != int(np.triu(mask.to_numpy(bool), 1).sum()):
        raise ValueError("Current and 2D saved pair masks have different sizes")
    pi, pj = distance.index.get_indexer(pairs.sample_i), distance.index.get_indexer(pairs.sample_j)
    if (pi < 0).any() or (pj < 0).any() or not mask.to_numpy(bool)[pi, pj].all():
        raise ValueError("The 2D warm start contains missing/excluded current pairs")
    observed = np.column_stack([distance.to_numpy()[pi, pj], variance.to_numpy()[pi, pj]])
    if not np.array_equal(observed, pairs[["input_distance", "bootstrap_variance"]].to_numpy()):
        raise ValueError("Current distances/variances differ from the checked 2D warm start")
    base_reference["pairs"] = pairs
    base_reference["parameters"].update(dimension=3, fixed_lambda=None, lambda_bounds=(0., np.inf))
    out.mkdir(parents=True, exist_ok=False)
    _json(out / "configuration.json", SETTINGS)
    for path in (Path(neural_hmds.__file__), Path(sys.modules["hmds_refinement"].__file__)):
        shutil.copyfile(path, out / (path.stem + "_used.py"))
    (out / "input_snapshot.pkl").write_bytes(pickle.dumps(dict(distance=distance, variance=variance, pair_mask=mask,
                                                               baseline_bundle=baseline_bundle)))

    def refine(reference, label, tolerance):
        if progress is not None:
            progress(label)
        return refine_hmds(reference, neural_hmds.hmds_geometry, neural_hmds.hmds_loss_gradient,
                            neural_hmds.hmds_projected_gradient, fixed_lambda=None,
                            max_blocks=SETTINGS["max_blocks"], iterations_per_block=SETTINGS["iterations_per_block"],
                            ridge_fraction=SETTINGS["ridge_fraction"], gradient_tolerance=tolerance,
                            gradient_metric=SETTINGS["gradient_metric"], label=label, verbose=progress is not None)

    runs = []
    with threadpool_limits(limits=1, user_api="blas"):
        for index, amplitude in enumerate(SETTINGS["start_amplitudes"]):
            seed = SETTINGS["start_seed"] + index
            rng = np.random.default_rng(seed)
            old_chart = baseline["halfspace_coordinates"]
            chart = np.column_stack([old_chart[:, 0], np.exp(old_chart[:, -1])*rng.normal(size=n)*amplitude,
                                     old_chart[:, -1]])
            chart[0] = 0.
            reference = copy.deepcopy(base_reference)
            reference["coordinates"] = pd.DataFrame(neural_hmds.hmds_chart_to_ball(chart), index=ids,
                                                       columns=["Poincare1", "Poincare2", "Poincare3"])
            reference["halfspace_coordinates"] = chart.copy()
            x = np.r_[chart[1:].ravel(), baseline["parameters_vector"][2*(n-1):]]
            reference["optimizer_result"].x = x
            reference = reanchor_hmds(reference, x, neural_hmds.hmds_geometry, neural_hmds.hmds_loss_gradient)
            fit = refine(reference, f"3D lift {index+1}/{len(SETTINGS['start_amplitudes'])}", SETTINGS["gradient_tolerance"])
            runs.append(dict(kind="lifted_2d", seed=seed, amplitude=amplitude, reference=reference, fit=fit))
            (out / "run_checkpoints.pkl").write_bytes(pickle.dumps(runs))
        best = min(runs, key=lambda record: record["fit"]["diagnostics"]["mean_nll"])
        for index, amplitude in enumerate(SETTINGS["perturb_amplitudes"]):
            seed = SETTINGS["perturb_seed"] + index
            rng = np.random.default_rng(seed)
            reference = copy.deepcopy(best["reference"])
            x, chart = best["fit"]["parameters_vector"].copy(), best["fit"]["halfspace_coordinates"].copy()
            tangent = rng.normal(size=(n-1, 3)) * amplitude
            chart[1:, :-1] += np.exp(chart[1:, -1, None]) * tangent[:, :-1]
            chart[1:, -1] += tangent[:, -1]
            x[:3*(n-1)] = chart[1:].ravel()
            reference["optimizer_result"].x = x
            reference = reanchor_hmds(reference, x, neural_hmds.hmds_geometry, neural_hmds.hmds_loss_gradient)
            fit = refine(reference, f"3D perturbation {index+1}/{len(SETTINGS['perturb_amplitudes'])}",
                          SETTINGS["perturb_gradient_tolerance"])
            runs.append(dict(kind="local_perturbation", seed=seed, amplitude=amplitude, reference=reference, fit=fit))
            (out / "run_checkpoints.pkl").write_bytes(pickle.dumps(runs))
        table = pd.DataFrame([dict(run=i, kind=r["kind"], seed=r["seed"], amplitude=r["amplitude"],
                                   **r["fit"]["diagnostics"]) for i, r in enumerate(runs)])
        table.to_csv(out / "all_runs.csv", index=False)
        eligible = table.loc[table.gradient_converged]
        if eligible.empty:
            raise RuntimeError("No 3D refinement met the convergence criterion; saved checkpoints remain diagnostic only")
        winner = int(eligible.mean_nll.idxmin())
        reference, fit = copy.deepcopy(runs[winner]["reference"]), copy.deepcopy(runs[winner]["fit"])
        checks = neural_hmds.check_neural_hmds_fit(reference, fit, out, SETTINGS["gradient_tolerance"])
        display, isometry_error = neural_hmds.center_hmds_for_display(fit["halfspace_coordinates"])
    fit_ids = reference["coordinates"].index
    fit["coordinates"] = pd.DataFrame(neural_hmds.hmds_chart_to_ball(fit["halfspace_coordinates"]),
                                       index=fit_ids, columns=["Poincare1", "Poincare2", "Poincare3"])
    fit["display_coordinates"] = pd.DataFrame(display, index=fit_ids, columns=["Poincare1", "Poincare2", "Poincare3"])
    fit["diagnostics"].update(joint_projected_grad_inf=checks["joint_projected_grad_inf"],
                               joint_geometric_grad_max=checks["joint_geometric_grad_max"],
                               joint_convergence_grad_norm=checks["joint_convergence_grad_norm"],
                               joint_gradient_converged=True, curvature_estimated=True)
    input_now = fit["pairs"][pair_columns].sort_values(["sample_i", "sample_j"]).reset_index(drop=True)
    old_pairs = pairs[pair_columns].copy()
    # Reanchoring can reverse a pair's endpoint ordering; verify by unordered labels.
    def canonical(frame):
        order = np.sort(frame[["sample_i", "sample_j"]].to_numpy(str), axis=1)
        return pd.DataFrame(dict(a=order[:, 0], b=order[:, 1],
                                 distance=frame.input_distance.to_numpy(), variance=frame.bootstrap_variance.to_numpy())).sort_values(["a", "b"]).reset_index(drop=True)
    if not canonical(input_now).equals(canonical(old_pairs)):
        raise ValueError("3D fitted pair inputs no longer equal the checked 2D inputs")
    fit["display_coordinates"].to_csv(out / "embedding_display_coordinates.csv")
    fit["coordinates"].to_csv(out / "embedding_coordinates.csv")
    _coordinate_table(fit["display_coordinates"], distance.index).to_csv(out / "coordinates.csv")
    fit["pairs"].to_csv(out / "fitted_pairs.csv", index=False)
    fit["pairs"][PAIR_COLUMNS].to_csv(out / "pairs.csv", index=False)
    fit["sigma"].rename("extra_residual_sigma").to_csv(out / "residual_sigma.csv")
    fit["history"].to_csv(out / "selected_history.csv", index=False)
    report = dict(status="checked_fit", converged=True, dimension=3, n_samples=n, n_pairs=len(pairs),
                   selected_run=winner, diagnostics=fit["diagnostics"], gradient_checks=checks,
                   display_isometry_max_distance_error=isometry_error, unconverged_runs=int((~table.gradient_converged).sum()),
                   same_observed_pairs_and_variances_as_2d=True, bootstrap_recomputed=False,
                   displayed_relative_rmse=_relative_rmse(fit["pairs"]),
                   relative_rmse_2d=float(baseline["diagnostics"]["relative_rmse"]),
                   relative_rmse_improvement_over_2d=float(1-fit["diagnostics"]["relative_rmse"]/baseline["diagnostics"]["relative_rmse"]),
                   coordinates=str(out / "coordinates.csv"), shepard=str(out / "pairs.csv"), configuration=SETTINGS,
                   input_distance_unit="Neural chord distance sqrt(2*(1-cosine))",
                   fitted_distance_unit="Hyperbolic geodesic distance divided by fitted lambda; neural chord-distance units",
                   limitation="Lowest-loss converged local solution among the starts checked here; not proof of global optimality. Ball Euclidean distances are not fitted distances")
    _json(out / "report.json", report)
    (out / "hmds_3d_result.pkl").write_bytes(pickle.dumps(dict(reference=reference, fit=fit, report=report, runs=table)))
    return report


def run_hmds_3d(output_dir, repo_root, progress=print):
    """Reuse existing 2D inputs and chemical 3D cache; fit/check neural 3D once."""
    output_dir, repo_root = Path(output_dir).resolve(), Path(repo_root).resolve()
    tables, previous, out = output_dir / "tables", output_dir / "hmds", output_dir / "hmds3d"
    coverage = pd.read_csv(previous / "neural_sample_coverage.csv", index_col=0)
    if not pd.api.types.is_bool_dtype(coverage.included):
        raise ValueError("Existing 2D coverage selection must be boolean")
    selected_ids = coverage.index[coverage.included].astype(str)
    if len(selected_ids) != 81:
        raise ValueError("This requested comparison must retain exactly the existing 81-sample 2D scope")
    baseline_path = previous / "neural/hmds_2d_result.pkl"
    baseline = pickle.loads(baseline_path.read_bytes())
    distance, variance, mask = [_matrix(tables / filename) for filename in (
        "hmds_neural_distance_chord.csv", "hmds_neural_variance.csv", "hmds_neural_pair_mask.csv")]
    if any(not value.index.equals(distance.index) for value in (variance, mask)):
        raise ValueError("Neural input matrices have different identity order")
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes) or np.diag(mask).any():
        raise ValueError("Neural pair mask must be boolean with false diagonal")
    old_provenance = json.loads((previous / "input_provenance.json").read_text())
    for filename in ("hmds_neural_distance_chord.csv", "hmds_neural_variance.csv", "hmds_neural_pair_mask.csv", "hmds_bootstrap_parameters.json"):
        path = tables / filename
        if old_provenance["input_sha256"].get(str(path)) != _hash(path):
            raise ValueError(f"Existing 2D input has changed: {filename}")
    input_paths = [baseline_path, previous / "neural_sample_coverage.csv", previous / "input_provenance.json"]
    input_paths += [tables / name for name in ("hmds_neural_distance_chord.csv", "hmds_neural_variance.csv",
                                              "hmds_neural_pair_mask.csv", "hmds_bootstrap_parameters.json", "rdm_chemical.csv")]
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [repo_root / "notebook" / name for name in ("neural_hmds.py", "hmds_refinement.py", "03_chemical_neuron_bacteria.ipynb")]
    _json(out / "input_provenance.json", dict(input_sha256={str(p): _hash(p) for p in input_paths},
                                              code_sha256={str(p): _hash(p) for p in source_paths + [Path(__file__)]},
                                              selected_sample_ids=selected_ids.tolist(), settings=SETTINGS,
                                              adaptation="Notebook03 cell31 2D-to-3D lift and multistarts; repository refinement, full geometric/numerical checks, and isometric display centering reused"))
    if progress is not None:
        progress("Verifying checked chemical 3D cache on the current full RDM")
    result = dict(chemical=_verify_chemical_cache(out / "chemical", repo_root, selected_ids, _matrix(tables / "rdm_chemical.csv")))
    if str(repo_root / "notebook") not in sys.path:
        sys.path.insert(0, str(repo_root / "notebook"))
    try:
        result["neural"] = _checked_neural_3d(distance.loc[selected_ids, selected_ids], variance.loc[selected_ids, selected_ids],
                                              mask.loc[selected_ids, selected_ids], baseline, out / "neural", progress)
    except RuntimeError as error:
        result["neural"] = dict(status="diagnostic_only", converged=False, dimension=3, n_samples=len(selected_ids),
                                 reason=str(error), output_directory=str(out / "neural"), coordinates=None, shepard=None)
        _json(out / "result.json", result)
        if progress is not None:
            progress(f"Neural 3D fit did not pass checks; diagnostics saved: {error}")
        return result
    _json(out / "result.json", result)
    return result
