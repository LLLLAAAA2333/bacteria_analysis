"""Checked full-sample 2D/3D HMDS without a bootstrap-coverage exclusion.

Reuse the measured 106-sample D and bootstrap V unchanged. All finite
off-diagonal pairs are fitted; the former 80% bootstrap cutoff is retained
as QC information, not an exclusion. V remains conditional on valid draws.
The previous 81-sample results, preprocessing and bootstrap are untouched.
"""
from pathlib import Path
import json
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

from individual_hmds import NEURAL_SETTINGS, _coordinates, _hash, _json, _matrix, _reuse_chemical
from individual_hmds_3d import _checked_neural_3d, _verify_chemical_cache


def _coverage_qc(distance, variance, fraction, former_mask):
    ids = distance.index
    upper = np.triu_indices(len(ids), 1)
    valid_fraction = fraction.to_numpy(float)
    pairs = pd.DataFrame(dict(sample_i=ids.to_numpy()[upper[0]], sample_j=ids.to_numpy()[upper[1]],
                               input_distance=distance.to_numpy(float)[upper],
                               bootstrap_variance=variance.to_numpy(float)[upper],
                               valid_fraction=valid_fraction[upper],
                               previously_above_80pct=former_mask.to_numpy(bool)[upper],
                               included=True))
    samples = []
    for i, sample in enumerate(ids):
        other = np.arange(len(ids)) != i
        v = valid_fraction[i, other]
        samples.append(dict(sample_id=sample, included=True, n_fitted_pairs=int(other.sum()),
                             min_pair_valid_fraction=float(v.min()), median_pair_valid_fraction=float(np.median(v)),
                             max_pair_valid_fraction=float(v.max()),
                             n_pairs_above_previous_cutoff=int(former_mask.to_numpy(bool)[i, other].sum())))
    return pd.DataFrame(samples).set_index("sample_id"), pairs


def run_full_hmds(output_dir, repo_root, progress=print):
    """Fit both dimensions using all existing finite samples/pairs; no resampling."""
    output_dir, repo_root = Path(output_dir).resolve(), Path(repo_root).resolve()
    tables, out = output_dir / "tables", output_dir / "hmds_full106"
    filenames = dict(distance="hmds_neural_distance_chord.csv", variance="hmds_neural_variance.csv",
                     fraction="hmds_neural_valid_fraction.csv", former_mask="hmds_neural_pair_mask.csv",
                     bootstrap="hmds_bootstrap_parameters.json", chemical="rdm_chemical.csv")
    paths = {key: tables / name for key, name in filenames.items()}
    distance, variance, fraction, former_mask, chemical = [_matrix(paths[key]) for key in
                                                          ("distance", "variance", "fraction", "former_mask", "chemical")]
    ids = distance.index
    if any(not matrix.index.equals(ids) for matrix in (variance, fraction, former_mask, chemical)):
        raise ValueError("All neural/chemical inputs must retain the original identical sample order")
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in former_mask.dtypes):
        raise ValueError("The former pair mask must be explicitly boolean")
    d, v, f = [matrix.to_numpy(float) for matrix in (distance, variance, fraction)]
    mask_array = ~np.eye(len(ids), dtype=bool)
    if (not np.isfinite(d).all() or not np.isfinite(v[mask_array]).all()
            or not np.isfinite(f).all() or np.any(v[mask_array] <= 0)
            or np.any(d < 0) or np.any(d > 2+1e-12) or np.any(f < 0) or np.any(f > 1)
            or not np.allclose(d, d.T) or not np.allclose(v, v.T) or not np.allclose(f, f.T)):
        raise ValueError("Full-sample fit requires finite symmetric chord D, positive off-diagonal V and valid coverage fractions")
    bootstrap = json.loads(paths["bootstrap"].read_text())
    draws = int(bootstrap["draws"])
    if np.any(f[mask_array]*draws < 2):
        raise ValueError("Every fitted variance requires at least two valid bootstrap draws")
    mask = pd.DataFrame(mask_array, index=ids, columns=ids)
    scope = dict(n_samples=len(ids), n_pairs=int(mask_array.sum()//2),
                  pair_selection="All finite off-diagonal pairs; no bootstrap valid-fraction exclusion",
                  former_valid_fraction_cutoff=float(bootstrap["min_valid_fraction"]),
                  minimum_valid_fraction=float(f[mask_array].min()), maximum_valid_fraction=float(f[mask_array].max()),
                  minimum_valid_draws=int(round(f[mask_array].min()*draws)), bootstrap_draws=draws,
                  uncertainty_limit="Bootstrap variance is estimated conditional on a draw retaining the fixed original cell/date support; it is not unconditional variance across all draws",
                  bootstrap_recomputed=False, snr_recomputed=False)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "individual_hmds_full_used.py")
    source_paths = [Path(__file__), Path(__file__).with_name("individual_hmds.py"),
                    Path(__file__).with_name("individual_hmds_3d.py")]
    source_paths += [repo_root / "notebook" / name for name in ("neural_hmds.py", "hmds_refinement.py")]
    _json(out / "input_provenance.json", dict(scope=scope, input_sha256={str(p): _hash(p) for p in paths.values()},
                                              code_sha256={str(p): _hash(p) for p in source_paths},
                                              neural_2d_settings=NEURAL_SETTINGS))
    mask.to_csv(out / "neural_full_pair_mask.csv")
    former_mask.to_csv(out / "neural_previous_80pct_pair_mask.csv")
    fraction.to_csv(out / "neural_valid_fraction_qc.csv")
    sample_qc, pair_qc = _coverage_qc(distance, variance, fraction, former_mask)
    sample_qc.to_csv(out / "neural_sample_coverage.csv")
    pair_qc.to_csv(out / "neural_pair_coverage_qc.csv", index=False)
    _json(out / "scope.json", scope)
    if str(repo_root / "notebook") not in sys.path:
        sys.path.insert(0, str(repo_root / "notebook"))
    from neural_hmds import fit_neural_hmds_2d

    out2d = out / "2d"
    out2d.mkdir()
    if progress is not None:
        progress(f"Full-sample HMDS: {len(ids)} samples, {scope['n_pairs']} pairs; reusing D/V, minimum valid draws {scope['minimum_valid_draws']}/{draws}")
    result2d = dict(chemical=_reuse_chemical(out2d, repo_root, ids, chemical))
    bootstrap_for_fit = dict(original_bootstrap_parameters=bootstrap, current_fit_scope=scope)
    try:
        if progress is not None:
            progress("Starting checked neural 2D HMDS on every sample and pair")
        bundle = fit_neural_hmds_2d(distance, variance, mask, bootstrap_for_fit,
                                    out2d / "neural", **NEURAL_SETTINGS)
        fit, report = bundle["fit"], bundle["report"]
        if (not fit["diagnostics"].get("joint_gradient_converged", False)
                or not report["gradient_checks"].get("all_passed", False)):
            raise RuntimeError("Full-sample neural 2D fit did not pass convergence and gradient checks")
    except RuntimeError as error:
        result2d["neural"] = dict(scope, status="diagnostic_only", converged=False, dimension=2,
                                   reason=str(error), coordinates=None, shepard=None,
                                   output_directory=str(out2d / "neural"))
        _json(out2d / "result.json", result2d)
        _json(out / "result.json", dict(scope=scope, **{"2d": result2d, "3d": None}))
        if progress is not None:
            progress(f"Full-sample 2D did not pass its checks; diagnostic checkpoints retained: {error}")
        return {"scope": scope, "2d": result2d, "3d": None}
    _coordinates(fit["display_coordinates"], ids).to_csv(out2d / "neural_coordinates.csv")
    result2d["neural"] = dict(scope, status="checked_fit", converged=True, dimension=2,
                               coordinates=str(out2d / "neural_coordinates.csv"),
                               shepard=str(out2d / "neural/fitted_pairs.csv"),
                               report=str(out2d / "neural/report.json"), diagnostics=fit["diagnostics"],
                               displayed_relative_rmse=float(fit["diagnostics"]["relative_rmse"]),
                               limitation="Checked local solution; low-coverage pairs retained, variance conditional on valid bootstrap draws")
    _json(out2d / "result.json", result2d)
    if progress is not None:
        progress(f"Full-sample neural 2D checked: relative RMSE {fit['diagnostics']['relative_rmse']:.6%}; starting 3D")

    out3d = out / "3d"
    out3d.mkdir()
    result3d = dict(chemical=_verify_chemical_cache(out3d / "chemical", repo_root, ids, chemical))
    try:
        result3d["neural"] = _checked_neural_3d(distance, variance, mask, bundle,
                                                out3d / "neural", progress)
        result3d["neural"].update(scope)
    except RuntimeError as error:
        result3d["neural"] = dict(scope, status="diagnostic_only", converged=False, dimension=3,
                                   reason=str(error), coordinates=None, shepard=None,
                                   output_directory=str(out3d / "neural"))
        if progress is not None:
            progress(f"Full-sample 3D did not pass its checks; diagnostic checkpoints retained: {error}")
    _json(out3d / "result.json", result3d)
    result = {"scope": scope, "2d": result2d, "3d": result3d}
    _json(out / "result.json", result)
    return result
