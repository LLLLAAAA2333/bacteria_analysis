"""Shared HMDS geometry, likelihood and numerical checks.

Geometry and derivative checks were extracted from the exploratory notebook.
The explicit 2D fitting entry point starts from current D/V inputs, without a
saved fit. Importing this module does not execute analysis. Coordinates use the
unit-curvature upper half-space; q = sigma**2 is additional model residual variance.
"""
import numpy as np

# Derivative-check tolerances; these are not optimizer convergence thresholds.
HMDS_DIAG_FD_ATOL = 1e-7
HMDS_DIAG_FD_RTOL = 1e-3
# Half-decades sample the narrow window between truncation and float64
# roundoff for small heights at large x. Keep the same range, tolerances,
# 32-ulp floor, and requirement for agreement at two adjacent step sizes.
HMDS_LOCAL_STEPS = 10.0 ** -np.linspace(1, 9, 17)
HMDS_LOCAL_MIN_ULPS = 32
# One bounded fallback for slow profile fits; affects optimizer units only.
HMDS_PROFILE_RIDGE_REDUCTION = 1e-3


def hmds_geometry(chart, pair_i, pair_j):
    """Stable hyperbolic distances and endpoint gradients in (x, log-height).

    sinh(d/2)^2 = sinh((h_i-h_j)/2)^2 + ||x_i-x_j||^2 * exp(-h_i-h_j)/4.
    Evaluate the sum in log space, without forming exp(h), cosh(d), or a difference
    of large Lorentz products. At coincident points choose the zero subgradient.
    """
    first, second = chart[pair_i], chart[pair_j]
    dx = first[:, :-1] - second[:, :-1]
    norm = np.hypot.reduce(dx, axis=1)
    delta = first[:, -1] - second[:, -1]
    absolute_delta = np.abs(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vertical = absolute_delta + 2 * np.log(-np.expm1(-absolute_delta)) - np.log(4)
        log_horizontal = 2 * np.log(norm) - first[:, -1] - second[:, -1] - np.log(4)
    log_sum = np.logaddexp(log_vertical, log_horizontal)
    distance = 2 * np.logaddexp(0.5 * log_sum, 0.5 * np.logaddexp(log_sum, 0))
    valid = np.isfinite(log_sum)
    root = np.zeros_like(log_sum)
    weight_vertical, weight_horizontal = np.zeros_like(root), np.zeros_like(root)
    root[valid] = np.exp(0.5 * (log_sum[valid] - np.logaddexp(log_sum[valid], 0)))
    weight_vertical[valid] = np.exp(log_vertical[valid] - log_sum[valid])
    weight_horizontal[valid] = np.exp(log_horizontal[valid] - log_sum[valid])
    horizontal = root * weight_horizontal
    direction = np.divide(dx, norm[:, None], out=np.zeros_like(dx), where=norm[:, None] > 0)
    magnitude = np.divide(2 * horizontal, norm, out=np.zeros_like(norm), where=norm > 0)
    gradient_x = magnitude[:, None] * direction
    vertical = np.divide(root * weight_vertical, np.tanh(delta / 2),
                         out=np.zeros_like(root), where=delta != 0)
    gradient_i = np.column_stack([gradient_x, vertical - horizontal])
    gradient_j = np.column_stack([-gradient_x, -vertical - horizontal])
    return distance, gradient_i, gradient_j


def hmds_loss_gradient(parameters, n_samples, dimension, pair_i, pair_j,
                       observed, variance, variance_scale=1.0):
    """Same Gaussian composite likelihood; q is numerically rescaled, not regularized."""
    n_coords = (n_samples - 1) * dimension
    chart = np.vstack([np.zeros(dimension), parameters[:n_coords].reshape(n_samples - 1, dimension)])
    scale = np.exp(parameters[n_coords])
    q = parameters[n_coords + 1:] * variance_scale
    distance, grad_i, grad_j = hmds_geometry(chart, pair_i, pair_j)
    predicted = distance / scale
    residual = predicted - observed
    total_variance = variance + q[pair_i] + q[pair_j]
    loss = 0.5 * np.mean(np.log(2 * np.pi * total_variance) + residual ** 2 / total_variance)
    coefficient = residual / total_variance / len(observed)
    grad_chart = np.zeros_like(chart)
    np.add.at(grad_chart, pair_i, (coefficient / scale)[:, None] * grad_i)
    np.add.at(grad_chart, pair_j, (coefficient / scale)[:, None] * grad_j)
    grad_log_scale = np.sum(-coefficient * predicted)
    grad_variance_pair = 0.5 * (1 / total_variance - residual ** 2 / total_variance ** 2) / len(observed)
    grad_q = np.zeros(n_samples)
    np.add.at(grad_q, pair_i, grad_variance_pair)
    np.add.at(grad_q, pair_j, grad_variance_pair)
    return float(loss), np.r_[grad_chart[1:].ravel(), grad_log_scale, grad_q * variance_scale]


def hmds_projected_gradient(parameters, gradient, lower, upper):
    """L-BFGS-B projected gradient, including feasible distance to an active bound."""
    return np.where(gradient >= 0, np.minimum(gradient, parameters - lower),
                    np.maximum(gradient, parameters - upper))


def hmds_chart_to_ball(chart):
    """Display-only Cayley transform, evaluated after scaling to avoid exponent overflow.

    Very remote points can round to the unit sphere in float64. Fit distances are
    always evaluated in the log-height chart, never reconstructed from this display.
    """
    x, h = chart[:, :-1], chart[:, -1]
    with np.errstate(divide="ignore"):
        log_x = np.log(np.max(np.abs(x), axis=1))
    log_scale = np.maximum.reduce([np.zeros(len(chart)), log_x, h])
    inverse = np.exp(-log_scale)
    scaled_x, height = x * inverse[:, None], np.exp(h - log_scale)
    x_squared = np.sum(scaled_x ** 2, axis=1)
    denominator = x_squared + (height + inverse) ** 2
    horizontal = 2 * scaled_x * inverse[:, None] / denominator[:, None]
    vertical = (x_squared + height ** 2 - inverse ** 2) / denominator
    return np.column_stack([horizontal, vertical])


def hmds_existing_problem(fit):
    """Recover the exact optimizer units, data and bounds without changing the fit."""
    settings, result = fit['parameters'], fit['optimizer_result']
    ids, pairs = fit['coordinates'].index, fit['pairs']
    dimension, n_samples = settings['dimension'], len(ids)
    n_coords = (n_samples - 1) * dimension
    x = np.asarray(result.x, dtype=float).copy()
    if x.shape != (n_coords + 1 + n_samples,) or not np.isfinite(x).all():
        raise ValueError('Stored optimizer parameters have an invalid shape or nonfinite values')
    if 'upper half-space' not in settings.get('coordinate_system', ''):
        raise ValueError('This diagnostic requires the updated upper half-space fit')
    pair_i, pair_j = ids.get_indexer(pairs['sample_i']), ids.get_indexer(pairs['sample_j'])
    if (pair_i < 0).any() or (pair_j < 0).any():
        raise ValueError('Stored pair labels do not match sample IDs')
    observed = pairs['input_distance'].to_numpy(float, copy=True)
    variance = np.maximum(pairs['bootstrap_variance'].to_numpy(float), settings['variance_floor'])
    variance_scale = settings['variance_parameter_scale']
    lower, upper = np.full(x.size, -np.inf), np.full(x.size, np.inf)
    fixed_lambda = settings.get('fixed_lambda')
    if fixed_lambda is None:
        with np.errstate(divide='ignore'):
            lower[n_coords], upper[n_coords] = np.log(settings.get('lambda_bounds') or (0., np.inf))
    else:
        # log(exp(log_lambda)) can differ by one ulp. Fix the recorded parameter
        # itself after checking agreement, rather than rejecting a valid fit.
        fixed_log = np.log(fixed_lambda)
        tolerance = 4 * np.finfo(float).eps * max(1.0, abs(fixed_log))
        if not np.isclose(x[n_coords], fixed_log, atol=tolerance, rtol=0):
            raise ValueError('Fixed lambda differs from the recorded optimizer parameter')
        lower[n_coords] = upper[n_coords] = x[n_coords]
    lower[n_coords + 1:] = 0
    if (x < lower).any() or (x > upper).any():
        raise ValueError('Stored optimizer parameters are outside their recorded bounds')

    def objective(values):
        return hmds_loss_gradient(values, n_samples, dimension, pair_i, pair_j,
                                  observed, variance, variance_scale)

    groups = {
        'horizontal_x': np.array([k for k in range(n_coords) if k % dimension != dimension - 1]),
        'log_height': np.arange(dimension - 1, n_coords, dimension),
        'log_lambda': np.array([n_coords]),
        'scaled_q': np.arange(n_coords + 1, x.size),
    }
    labels = [f'{ids[1 + k // dimension]}:x{k % dimension + 1}'
              if k % dimension != dimension - 1 else f'{ids[1 + k // dimension]}:log_height'
              for k in range(n_coords)]
    labels += ['log_lambda'] + [f'{sample}:scaled_q' for sample in ids]
    return x, lower, upper, objective, groups, labels


def hmds_finite_difference(objective, x, k, step, lower, upper, loss):
    """Central differences in the interior; second-order feasible one-sided at bounds."""
    left, right = x[k] - lower[k], upper[k] - x[k]
    if left == 0 and right == 0:
        return np.nan, 0.0, 'fixed: excluded'
    if min(left, right) >= step:
        offsets, weights, scheme = (-1, 1), (-0.5, 0.5), 'central'
    else:
        sign = 1 if right >= left else -1
        room = right if sign == 1 else left
        step = min(step, room / 2)
        offsets, weights = (0, sign, 2 * sign), (-1.5 * sign, 2 * sign, -0.5 * sign)
        scheme = 'forward' if sign == 1 else 'backward'
    values = []
    for offset in offsets:
        if offset == 0:
            values.append(loss)
            continue
        trial = x.copy()
        trial[k] += offset * step
        if step <= 0 or trial[k] == x[k] or not lower[k] <= trial[k] <= upper[k]:
            return np.nan, step, 'unrepresentable/infeasible step'
        values.append(objective(trial)[0])
    # Subtract the baseline to reduce cancellation of a constant objective offset.
    derivative = np.dot(weights, np.asarray(values) - loss) / step
    return float(derivative), step, scheme


def hmds_check_local_horizontal(fit, samples, lambda_value, start):
    """Geometry-scaled FD with pairwise loss differences and actual float64 steps.

    Fixed lambda and q: only incident pairs change when one x coordinate moves.
    Subtract squared residuals in factored form, avoiding subtraction of two
    almost identical total likelihoods. No optimization or fit modification.
    """
    x, lower, upper, objective, groups, labels = hmds_existing_problem(fit)
    loss, gradient = objective(x)
    result, settings = fit['optimizer_result'], fit['parameters']
    if not np.isclose(loss, result.fun, rtol=1e-9, atol=1e-10):
        raise ValueError('Current objective differs from the stored fit; resolve this before FD checks')
    finite = np.isfinite(result.jac)
    if not np.allclose(gradient[finite], np.asarray(result.jac)[finite], rtol=1e-7, atol=1e-9):
        raise ValueError('Current gradient differs from the stored fit')
    ids, pairs = fit['coordinates'].index, fit['pairs']
    dimension = settings['dimension']
    n_coords = (len(ids) - 1) * dimension
    chart = np.vstack([np.zeros(dimension), x[:n_coords].reshape(-1, dimension)])
    scale = np.exp(x[n_coords])
    q = x[n_coords + 1:] * settings['variance_parameter_scale']
    pi, pj = ids.get_indexer(pairs['sample_i']), ids.get_indexer(pairs['sample_j'])
    observed = pairs['input_distance'].to_numpy(float)
    variance = np.maximum(pairs['bootstrap_variance'].to_numpy(float), settings['variance_floor'])
    total_variance = variance + q[pi] + q[pj]
    geometry_rows, fd_rows, pair_rows = [], [], []
    for sample in samples:
        point = ids.get_indexer([sample])[0]
        if point < 1:
            raise ValueError(f'{sample} is absent or is the fixed anchor')
        incident = np.flatnonzero((pi == point) | (pj == point))
        if not len(incident):
            raise ValueError(f'{sample} has no retained pairs')
        ii, jj = pi[incident], pj[incident]
        distance, gradient_i, gradient_j = hmds_geometry(chart, ii, jj)
        residual = distance / scale - observed[incident]
        nearest = int(np.argmin(distance))
        other = np.where(ii == point, jj, ii)
        height = float(np.exp(chart[point, -1]))
        if not np.isfinite(height) or height <= 0:
            raise ValueError('Height is not representable in float64; a different chart is required')
        for axis in range(dimension - 1):
            k = (point - 1) * dimension + axis
            value = x[k]
            ulp = max(np.nextafter(value, np.inf) - value,
                      value - np.nextafter(value, -np.inf))
            endpoint_gradient = np.where(ii == point, gradient_i[:, axis], gradient_j[:, axis])
            contributions = residual / total_variance[incident] / scale / len(pairs) * endpoint_gradient
            for j in np.argsort(-np.abs(contributions))[:3]:
                pair_rows.append(dict(fixed_lambda=lambda_value, start=start, parameter=labels[k],
                                      other_sample=ids[other[j]], hyperbolic_distance=distance[j],
                                      fitted_distance=distance[j] / scale, observed_distance=observed[incident[j]],
                                      total_variance=total_variance[incident[j]],
                                      raw_gradient_contribution=contributions[j]))
            geometry_rows.append(dict(fixed_lambda=lambda_value, start=start, parameter=labels[k],
                                      x=value, log_height=chart[point, -1], height=height, float_spacing=ulp,
                                      old_min_step=1e-7 * max(1, abs(value)),
                                      old_min_step_over_height=1e-7 * max(1, abs(value)) / height,
                                      raw_gradient=gradient[k], local_tangent_gradient=height * gradient[k],
                                      nearest_retained_sample=ids[other[nearest]],
                                      min_retained_hyperbolic_distance=distance[nearest],
                                      pair_gradient_sum_delta=contributions.sum() - gradient[k]))

            def loss_change(new_value):
                trial = chart.copy()
                trial[point, axis] = new_value
                trial_distance = hmds_geometry(trial, ii, jj)[0]
                delta_residual = (trial_distance - distance) / scale
                # ((r + dr)**2 - r**2) / (2*S); constants and other pairs cancel.
                return np.sum(delta_residual * (2 * residual + delta_residual)
                              / (2 * total_variance[incident])) / len(pairs)

            for relative_step in HMDS_LOCAL_STEPS:
                requested = relative_step * height
                row = dict(fixed_lambda=lambda_value, start=start, parameter=labels[k],
                           relative_step=relative_step, requested_dx=requested,
                           analytic=gradient[k], numerical=np.nan, error_over_tolerance=np.nan,
                           agrees=False, usable=False, actual_dx_plus=np.nan, actual_dx_minus=np.nan)
                if requested < HMDS_LOCAL_MIN_ULPS * ulp:
                    row['note'] = 'skipped: step below float64 spacing margin'
                else:
                    plus, minus = value + requested, value - requested
                    a, b = plus - value, value - minus
                    # Unequal-step three-point derivative handles rounded coordinate updates.
                    numeric = (b / (a + b) * loss_change(plus) / a
                               - a / (a + b) * loss_change(minus) / b)
                    tolerance = HMDS_DIAG_FD_ATOL + HMDS_DIAG_FD_RTOL * max(abs(numeric), abs(gradient[k]))
                    ratio = abs(numeric - gradient[k]) / tolerance
                    row.update(numerical=numeric, error_over_tolerance=ratio,
                               agrees=bool(np.isfinite(ratio) and ratio <= 1),
                               usable=bool(np.isfinite(numeric)), actual_dx_plus=a, actual_dx_minus=b,
                               note='local-scale central difference')
                fd_rows.append(row)
    return geometry_rows, fd_rows, pair_rows


def initialize_neural_hmds(distance, variance, pair_mask, dimension, n_starts, seed,
                    max_iter, lambda_bounds, variance_floor, ftol=1e-10, gtol=1e-6,
                    history_every=25):
    """Random multistart initialization; finite lambda bounds apply only to this stage."""
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.sparse.csgraph import connected_components
    ids = distance.index
    for table in (distance, variance, pair_mask):
        if not ids.is_unique or not table.index.equals(ids) or not table.columns.equals(ids):
            raise ValueError("D, V and mask require identical unique sample IDs in the same order")
    if dimension not in (2, 3) or len(ids) < dimension + 1 or n_starts < 1:
        raise ValueError("Use dimension 2 or 3, enough samples, and at least one random start")
    if not 0 < lambda_bounds[0] < lambda_bounds[1] or variance_floor <= 0:
        raise ValueError("Invalid lambda bounds or variance floor")
    if max_iter < 1 or history_every < 1 or ftol <= 0 or gtol <= 0:
        raise ValueError("Optimizer tolerances and iteration settings must be positive")
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in pair_mask.dtypes):
        raise ValueError("pair_mask must contain booleans")
    d, v, mask = distance.to_numpy(float), variance.to_numpy(float), pair_mask.to_numpy(bool)
    if not np.array_equal(mask, mask.T) or np.diag(mask).any():
        raise ValueError("Pair mask must be symmetric with a false diagonal")
    if not np.allclose(d, d.T, equal_nan=True) or not np.allclose(v, v.T, equal_nan=True):
        raise ValueError("D and V must be symmetric")
    if not np.isfinite(d[mask]).all() or not np.isfinite(v[mask]).all() or (d[mask] < 0).any() or (v[mask] < 0).any():
        raise ValueError("Retained distances and variances must be finite and nonnegative")
    if connected_components(mask, directed=False, return_labels=False) != 1:
        raise ValueError("Retained pair graph is disconnected; inspect bootstrap coverage")
    pair_i, pair_j = np.where(np.triu(mask, k=1))
    observed = d[pair_i, pair_j]
    if not np.any(observed > 0):
        raise ValueError("All retained distances are zero")
    base_variance = np.maximum(v[pair_i, pair_j], variance_floor)
    # This is a change of optimizer units only: the likelihood still uses q_i=sigma_i**2.
    variance_scale = max(float(np.median(base_variance)), float(np.mean(observed ** 2)) * 1e-3)
    n_samples, n_coords = len(ids), (len(ids) - 1) * dimension
    bounds = ([(None, None)] * n_coords
              + [(np.log(lambda_bounds[0]), np.log(lambda_bounds[1]))]
              + [(0, None)] * n_samples)
    lower = np.array([-np.inf if lo is None else lo for lo, hi in bounds])
    upper = np.array([np.inf if hi is None else hi for lo, hi in bounds])
    rng = np.random.default_rng(seed)
    initializations, records, histories, results = [], [], [], []
    start_scales = np.geomspace(lambda_bounds[0], lambda_bounds[1], n_starts + 2)[1:-1]
    for initial_scale in start_scales:
        chart = rng.normal(size=(n_samples - 1, dimension)) * initial_scale * np.median(observed) / (2 * np.sqrt(dimension))
        initializations.append(("random", np.r_[chart.ravel(), np.log(initial_scale),
                                                 np.full(n_samples, np.median(base_variance) / variance_scale)]))
    for start, (kind, initial) in enumerate(initializations):
        cache, run_history = {}, []
        iteration = 0

        def objective(parameters):
            loss, gradient = hmds_loss_gradient(parameters, n_samples, dimension, pair_i, pair_j,
                                                 observed, base_variance, variance_scale)
            cache.update(x=parameters.copy(), loss=loss, gradient=gradient)
            return loss, gradient

        def record(parameters, step):
            if "x" not in cache or not np.array_equal(cache["x"], parameters):
                objective(parameters)
            pg = hmds_projected_gradient(parameters, cache["gradient"], lower, upper)
            run_history.append({"start": start, "kind": kind, "iteration": step,
                                "mean_nll": cache["loss"], "projected_grad_inf": np.max(np.abs(pg)),
                                "lambda": np.exp(parameters[n_coords])})

        def callback(parameters):
            nonlocal iteration
            iteration += 1
            if iteration % history_every == 0:
                record(parameters, iteration)

        record(initial, 0)
        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback,
            options={"maxiter": max_iter, "maxfun": 50 * max_iter + 1,
                     "ftol": ftol, "gtol": gtol, "maxls": 50, "maxcor": 20},
        )
        if run_history[-1]["iteration"] != result.nit:
            record(result.x, result.nit)
        final_loss, final_gradient = objective(result.x)
        projected = np.max(np.abs(hmds_projected_gradient(result.x, final_gradient, lower, upper)))
        results.append(result)
        histories.extend(run_history)
        records.append({"start": start, "kind": kind, "initial_mean_nll": run_history[0]["mean_nll"],
                        "mean_nll": final_loss, "success": bool(result.success),
                        "gradient_converged": bool(projected <= gtol), "projected_grad_inf": projected,
                        "n_iter": result.nit, "n_evaluations": result.nfev,
                        "lambda": np.exp(result.x[n_coords]), "message": str(result.message)})
        print(f"HMDS start {start+1}/{len(initializations)} ({kind}): loss={final_loss:.5f}, "
              f"projected gradient={projected:.2e}, success={result.success}", flush=True)
    finite_results = [k for k, result in enumerate(results) if np.isfinite(result.fun) and np.isfinite(result.x).all()]
    if not finite_results:
        raise RuntimeError("Every HMDS initialization failed numerically")
    best_index = min(finite_results, key=lambda k: results[k].fun)
    best = results[best_index]
    chart = np.vstack([np.zeros(dimension), best.x[:n_coords].reshape(n_samples - 1, dimension)])
    scale, q = np.exp(best.x[n_coords]), best.x[n_coords + 1:] * variance_scale
    ball = hmds_chart_to_ball(chart)
    predicted = hmds_geometry(chart, pair_i, pair_j)[0] / scale
    residual = predicted - observed
    total_variance = base_variance + q[pair_i] + q[pair_j]
    best_record = records[best_index]
    parameters = {"dimension": dimension, "seed": seed, "n_starts": n_starts,
                  "max_iter": max_iter, "lambda_bounds": lambda_bounds, "variance_floor": variance_floor,
                  "coordinate_system": "upper half-space: x and log-height; no coordinate bounds",
                  "variance_parameter_scale": variance_scale, "ftol": ftol, "gtol": gtol,
                  "history_every": history_every, "warm_start": "none; initialized from current inputs", "anchor_sample": ids[0],
                  "objective": "Gaussian negative composite log-likelihood; no priors"}
    diagnostics = pd.Series({
        "dimension": dimension, "n_samples": n_samples, "n_pairs": len(observed),
        "best_start": best_index, "optimizer_success": bool(best.success),
        "gradient_converged": best_record["gradient_converged"],
        "projected_grad_inf": best_record["projected_grad_inf"],
        "n_iter": best.nit, "termination_message": str(best.message),
        "lambda": scale, "raw_stress": residual @ residual,
        "relative_rmse": np.linalg.norm(residual) / np.linalg.norm(observed),
        "mean_nll": best.fun, "standardized_residual_rms": np.sqrt(np.mean(residual ** 2 / total_variance)),
        "variance_floor_pairs": int((v[pair_i, pair_j] < variance_floor).sum()),
        "lambda_at_bound": bool(np.isclose(scale, lambda_bounds, rtol=1e-3, atol=0).any()),
        "max_abs_log_height": float(np.max(np.abs(chart[:, -1]))),
        "display_boundary_points": int((np.linalg.norm(ball, axis=1) >= 1 - 1e-12).sum()),
        "minimum_pair_degree": int(mask.sum(axis=1).min()),
    })
    return {
        "coordinates": pd.DataFrame(ball, index=ids, columns=[f"Poincare{k+1}" for k in range(dimension)]),
        "halfspace_coordinates": chart, "lambda": scale,
        "sigma": pd.Series(np.sqrt(q), index=ids, name="extra_residual_sigma"),
        "pairs": pd.DataFrame({"sample_i": ids.to_numpy()[pair_i], "sample_j": ids.to_numpy()[pair_j],
                               "input_distance": observed, "fitted_distance": predicted,
                               "bootstrap_variance": v[pair_i, pair_j], "total_variance": total_variance,
                               "residual": residual}),
        "diagnostics": diagnostics, "starts": pd.DataFrame(records).set_index("start"),
        "history": pd.DataFrame(histories), "optimizer_result": best, "parameters": parameters,
    }


def check_neural_hmds_fit(reference, fit, output_directory, gradient_tolerance=1e-6):
    """Check free-lambda convergence and every parameter's numerical derivative.

    Same local-scale and adjacent-step checks as the current 3D fitting cell.
    Write diagnostics before raising; an unfinished fit is never labeled final.
    Use the fit's recorded gradient metric, always freeing lambda for this
    final check. Both raw and geometric joint norms are retained in the report.
    """
    import copy
    import json
    from pathlib import Path
    import pandas as pd
    from hmds_refinement import hmds_anchor_gradient, hmds_gradient_diagnostics

    output_directory = Path(output_directory)
    ids, pairs, settings = reference['coordinates'].index, reference['pairs'], reference['parameters']
    n, dimension = len(ids), settings['dimension']
    nc = (n - 1) * dimension
    x = fit['parameters_vector']
    pi, pj = ids.get_indexer(pairs.sample_i), ids.get_indexer(pairs.sample_j)

    def objective(values):
        return hmds_loss_gradient(
            values, n, dimension, pi, pj, pairs.input_distance.to_numpy(),
            np.maximum(pairs.bootstrap_variance.to_numpy(), settings['variance_floor']),
            settings['variance_parameter_scale'],
        )

    loss, gradient = objective(x)
    lower, upper = np.full(len(x), -np.inf), np.full(len(x), np.inf)
    lower[nc + 1:] = 0
    projected = hmds_projected_gradient(x, gradient, lower, upper)
    anchor_gradient = hmds_anchor_gradient(
        x, n, dimension, pi, pj, pairs.input_distance.to_numpy(),
        np.maximum(pairs.bootstrap_variance.to_numpy(), settings['variance_floor']),
        settings['variance_parameter_scale'], hmds_geometry)
    norms = hmds_gradient_diagnostics(x, gradient, projected, n, dimension, anchor_gradient)
    metric = fit['settings'].get('gradient_metric', 'original')
    if metric not in ('original', 'local_geometric'):
        raise ValueError('Unknown HMDS convergence metric')
    norm = norms['projected_grad_inf' if metric == 'original' else 'geometric_projected_grad_max']
    converged = bool(np.isfinite(loss) and np.isfinite(gradient).all()
                     and np.isfinite(norm) and norm <= gradient_tolerance)
    report = dict(mean_nll=loss, **norms, gradient_metric=metric,
                  joint_projected_grad_inf=norms['projected_grad_inf'],
                  joint_geometric_grad_max=norms['geometric_projected_grad_max'],
                  joint_convergence_grad_norm=norm,
                  joint_gradient_converged=converged, gradient_tolerance=gradient_tolerance)
    (output_directory / 'convergence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not converged:
        raise RuntimeError(f'Lowest-loss 2D run did not converge (gradient={norm:g}); '
                           f'inspect checkpoints in {output_directory}')

    checkref = copy.deepcopy(reference)
    checkref['optimizer_result'].x = x.copy()
    checkref['optimizer_result'].fun = loss
    checkref['optimizer_result'].jac = gradient.copy()
    checkref['parameters']['fixed_lambda'] = float(np.exp(x[nc]))
    _, rows, _ = hmds_check_local_horizontal(checkref, ids[1:], float(np.exp(x[nc])), 0)
    for row in rows:
        row['group'] = 'horizontal'
    for k in np.r_[np.arange(dimension - 1, nc, dimension), np.arange(nc, len(x))]:
        group = 'log_height' if k < nc else 'log_lambda' if k == nc else 'scaled_q'
        parameter = (f'{ids[1 + k // dimension]}:log_height' if k < nc else
                     'log_lambda' if k == nc else f'{ids[k - nc - 1]}:scaled_q')
        for relative_step in (1e-4, 1e-5, 1e-6, 1e-7):
            numeric, step, scheme = hmds_finite_difference(
                objective, x, k, relative_step * max(1, abs(x[k])), lower, upper, loss,
            )
            ratio = abs(numeric - gradient[k]) / (HMDS_DIAG_FD_ATOL + HMDS_DIAG_FD_RTOL
                                                 * max(abs(numeric), abs(gradient[k])))
            rows.append(dict(group=group, parameter=parameter, relative_step=relative_step,
                             analytic=gradient[k], numerical=numeric, step=step,
                             error_over_tolerance=ratio, agrees=bool(np.isfinite(ratio) and ratio <= 1),
                             usable=bool(np.isfinite(numeric)), note=scheme))
    steps = pd.DataFrame(rows)
    summaries = []
    for (group, parameter), values in steps.groupby(['group', 'parameter'], sort=False):
        flags = values.agrees.to_numpy(bool)
        summaries.append(dict(group=group, parameter=parameter, agreeing_steps=int(flags.sum()),
                              adjacent_steps_agree=bool(np.any(flags[:-1] & flags[1:])),
                              best_error_over_tolerance=values.error_over_tolerance.min()))
    summary = pd.DataFrame(summaries)
    steps.to_csv(output_directory / 'gradient_steps.csv', index=False)
    summary.to_csv(output_directory / 'gradient_summary.csv', index=False)
    passed = bool(len(summary) == len(x) and summary.adjacent_steps_agree.all())
    if not passed:
        raise RuntimeError(f'2D numerical gradient checks failed; inspect {output_directory / "gradient_summary.csv"}')
    report.update(checked_parameters=len(summary), all_gradient_checks_passed=passed)
    return report


def center_hmds_for_display(chart):
    """Center for display; allow only distance errors explained by roundoff.

    A translation/dilation is an exact isometry, but rounding the transformed
    coordinates can amplify errors for tight clusters at small heights. Check
    each pair against a forward-error bound, not just a fixed absolute cutoff.
    Quantitative fitted distances must still use the original log-height chart.
    """
    from scipy.optimize import minimize

    n, dimension = chart.shape
    origin = np.median(chart[:, :-1], axis=0)
    units = np.maximum(np.std(chart[:, :-1], axis=0), 1.)
    ci, cj = np.zeros(n, dtype=int), np.arange(1, n + 1)

    def objective(z):
        center = np.r_[origin + units * z[:-1], z[-1]]
        distance, gradient, _ = hmds_geometry(np.vstack([center, chart]), ci, cj)
        grad = np.mean(distance[:, None] * gradient, axis=0)
        grad[:-1] *= units
        return .5 * np.mean(distance**2), grad

    result = minimize(objective, np.r_[np.zeros(dimension - 1), np.log(np.linalg.norm(units))],
                      jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-9, 'ftol': 1e-14})
    center = np.r_[origin + units * result.x[:-1], result.x[-1]]
    display_chart = chart.copy()
    display_chart[:, :-1] = (chart[:, :-1] - center[:-1]) * np.exp(-center[-1])
    display_chart[:, -1] -= center[-1]
    ii, jj = np.triu_indices(n, 1)
    errors = np.abs(hmds_geometry(chart, ii, jj)[0] - hmds_geometry(display_chart, ii, jj)[0])
    # Conservative rounding bounds for subtract / exp / multiply (4 eps) and
    # log-height subtraction (2 eps). A horizontal error dx at height y has
    # distance 2*asinh(||dx||/(2*y)); add the height error by triangle inequality.
    # The pairwise distance error is at most the sum of endpoint displacements.
    eps = np.finfo(float).eps
    dx = 4 * eps * (np.abs(chart[:, :-1]) + np.abs(center[:-1])) * np.exp(-center[-1])
    dh = 2 * eps * (np.abs(chart[:, -1]) + abs(center[-1]))
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        displacement = dh + 2 * np.arcsinh(
            np.hypot.reduce(dx, axis=1) / (2 * np.exp(display_chart[:, -1] - dh)))
    tolerance = 1e-8 + displacement[ii] + displacement[jj]
    error = float(np.max(errors))
    if (not np.isfinite(error) or not np.isfinite(tolerance).all()
            or np.any(errors > tolerance)):
        raise RuntimeError(f'Display centering changed fitted hyperbolic distances '
                           f'beyond float64 roundoff (max error={error:g})')
    return hmds_chart_to_ball(display_chart), error


class _HMDSProfileError(RuntimeError):
    """A saved optimization attempt exhausted its numerical search budget."""


def _profile_neural_hmds_2d(reference, checkpoint, max_blocks, iterations_per_block,
                            gradient_tolerance, ridge_fraction, max_profile_steps=30):
    """Estimate lambda by a local profile, reanchoring each conditional fit.

    Only converged conditional fits supply profile scores. Bracketing is a
    numerical search budget, not a bound on curvature. The likelihood, variance
    model are unchanged. Convergence uses all-point geometric coordinate norms
    and the original lambda/scaled-q projected gradients; raw norms are retained.
    After a full chunk hits every iteration limit, profile fits can reduce the
    scaling ridge once, within the existing block budget. Initial conditioning
    retains its inner optimizer tolerances. Every chunk records ridge and anchor.
    Change a dominant unconverged anchor without waiting for other parameter
    groups to converge; all groups must still pass before accepting a fit.
    """
    import copy
    import pickle
    import pandas as pd
    from scipy.optimize import brentq
    import hmds_refinement as refinement

    cache, records = {}, []
    conditional_tolerance = gradient_tolerance / 10
    nc = 2 * (len(reference['coordinates']) - 1)

    def conditional(lam, base, first=False):
        ref = copy.deepcopy(base['reference'])
        x = base['parameters'].copy()
        histories, used = [], 0
        conditional_ridge = ridge_fraction
        change_anchor = False
        while used < max_blocks:
            if not first or used:
                ref = refinement.reanchor_hmds(
                    ref, x, hmds_geometry, hmds_loss_gradient, avoid_current_anchor=change_anchor)
            blocks = min(5, max_blocks - used)
            # Profile evaluations use tighter inner optimizer stopping tolerances
            # than the initial conditioning stage.
            fit = refinement.refine_hmds(
                ref, hmds_geometry, hmds_loss_gradient, hmds_projected_gradient,
                initial_parameters=x if first and not used else ref['optimizer_result'].x,
                fixed_lambda=lam, max_blocks=blocks,
                iterations_per_block=min(1000, iterations_per_block) if first else iterations_per_block,
                ridge_fraction=conditional_ridge, gradient_tolerance=conditional_tolerance,
                inner_ftol=1e-14 if first else 0., inner_gtol=1e-8 if first else 1e-12,
                gradient_metric='local_geometric',
                label=f'2D conditional lambda={lam:.8g}',
            )
            x = fit['parameters_vector']
            used += len(fit['history'])
            anchor_sample = str(ref['coordinates'].index[0])
            fit['history'] = fit['history'].assign(ridge_fraction=conditional_ridge, anchor_sample=anchor_sample)
            histories.append(fit['history'])
            records.append(dict(lambda_value=lam, ridge_fraction=conditional_ridge, anchor_sample=anchor_sample, **{
                k: v for k, v in fit['diagnostics'].items() if k != 'lambda_value'}))
            fit['history'] = pd.concat(histories, ignore_index=True)
            current = dict(reference=ref, fit=fit, parameters=x)
            checkpoint.write_bytes(pickle.dumps(dict(current=current, records=records)))
            if fit['diagnostics']['gradient_converged']:
                cache[lam] = current
                return current
            if not len(histories[-1]):
                break
            diagnostic = fit['diagnostics']
            # The pinned point has no independent coordinate in this chart.
            # If it carries the largest coordinate residual, make it free in
            # the next chart. Requiring q/free points to pass first can exhaust
            # the budget while this poorly conditioned direction barely moves.
            change_anchor = bool(diagnostic['anchor_geometric_grad_norm'] > conditional_tolerance
                                 and diagnostic['anchor_geometric_grad_norm']
                                 >= diagnostic['free_coordinate_geometric_grad_max'])
            if change_anchor and used < max_blocks:
                print(f'2D conditional lambda={lam:.8g}: changing anchor after {used} blocks '
                      'to optimize the dominant coordinate gradient', flush=True)
            if (not first and used < max_blocks and conditional_ridge == ridge_fraction
                    and fit['diagnostics']['stop_reason'] == 'block budget exhausted'
                    and (histories[-1]['n_iter'] >= iterations_per_block).all()):
                # A large Fisher ridge leaves weak directions poorly scaled.
                # Continue the saved parameters with less damping, not looser
                # convergence. refine_hmds retains its absolute ridge floor.
                conditional_ridge = ridge_fraction * HMDS_PROFILE_RIDGE_REDUCTION
                print(f'2D conditional lambda={lam:.8g}: reducing scaling ridge to '
                      f'{conditional_ridge:g} after {used} blocks', flush=True)
        gradient = fit['diagnostics']['convergence_grad_norm']
        raise _HMDSProfileError(
            f'Conditional fit at lambda={lam:g} did not converge: geometric gradient={gradient:.3g} '
            f'> {conditional_tolerance:g} after {used}/{max_blocks} blocks '
            f'(ridge_fraction={conditional_ridge:g}); inspect {checkpoint}')

    start = float(np.exp(reference['optimizer_result'].x[nc]))
    base = dict(reference=reference, parameters=reference['optimizer_result'].x)
    conditional(start, base, first=True)

    def score(lam):
        if lam not in cache:
            nearest = min(cache, key=lambda key: abs(np.log(lam / key)))
            conditional(lam, cache[nearest])
        return cache[lam]['fit']['diagnostics']['d_loss_d_log_lambda']

    previous, previous_score = start, score(start)
    root = start
    if abs(previous_score) > conditional_tolerance:
        # Close to a stationary point, a 10% jump needlessly enters a much
        # harder conditional problem. Start locally, then expand if necessary.
        step = start * min(.1, max(1e-4, abs(previous_score)))
        for _ in range(max_profile_steps):
            trial = previous + step if previous_score < 0 else previous / (1 + step / start)
            trial_score = score(trial)
            if abs(trial_score) <= conditional_tolerance:
                root = trial
                break
            if previous_score * trial_score < 0:
                root = brentq(score, *sorted((previous, trial)), xtol=1e-7, rtol=1e-10, maxiter=40)
                break
            previous, previous_score = trial, trial_score
            step = min(2 * step, start / 10)
        else:
            raise _HMDSProfileError(f'Lambda profile did not bracket a stationary point; inspect {checkpoint}')
    score(root)
    result = cache[root]
    fit, ref = result['fit'], result['reference']
    diagnostic = fit['diagnostics']
    raw_joint = float(np.max([diagnostic['projected_grad_inf'], abs(diagnostic['d_loss_d_log_lambda'])]))
    joint = float(np.max([diagnostic['geometric_projected_grad_max'], abs(diagnostic['d_loss_d_log_lambda'])]))
    diagnostic.update(conditional_projected_grad_inf=diagnostic['projected_grad_inf'],
                      conditional_geometric_grad_max=diagnostic['geometric_projected_grad_max'],
                      projected_grad_inf=raw_joint, joint_projected_grad_inf=raw_joint,
                      geometric_projected_grad_max=joint, joint_geometric_grad_max=joint,
                      convergence_grad_norm=joint, joint_convergence_grad_norm=joint,
                      gradient_metric='local_geometric',
                      gradient_converged=bool(np.isfinite(joint) and joint <= gradient_tolerance),
                      joint_gradient_converged=bool(np.isfinite(joint) and joint <= gradient_tolerance),
                      fixed_lambda=None, curvature_estimated=True,
                      stop_reason='profile stationary point; geometric coordinates, lambda and variance checked')
    ref['parameters'].update(fixed_lambda=None, lambda_bounds=(0., np.inf))
    fit['settings'].update(fixed_lambda=None, lambda_bounds=None,
                           gradient_metric='local_geometric',
                           conditional_ridge_reduction=HMDS_PROFILE_RIDGE_REDUCTION,
                           fit_strategy='adaptive local lambda profile with converged conditional fits')
    fit['profile_history'] = pd.DataFrame(records)
    return ref, fit


def fit_neural_hmds_2d(distance, variance, pair_mask, bootstrap_parameters, output_directory,
                       n_starts=8, seed=20260918, initial_max_iter=3000,
                       initial_lambda_bounds=(0.05, 10.0), variance_floor=1e-10,
                       max_blocks=30, iterations_per_block=2000, gradient_tolerance=1e-6,
                       perturb_amplitudes=(0.0, 0.1, 0.3, 0.7), perturb_seed=20260930,
                       ridge_fraction=1e-7, max_profile_steps=30):
    """Fit current neural D/V from scratch, refine, check and save a 2D bundle.

    The bounded first stage supplies random initial fits only. Final refinement
    profiles positive lambda and retains the original Gaussian composite likelihood
    and q >= 0 constraints. Coordinate convergence uses all-point geometric norms;
    raw norms are also recorded. No pickle/input cache is read. Perturbations check
    nearby solutions, not global optimality. All runs/checkpoints are retained.
    """
    import copy
    import hashlib
    import json
    from pathlib import Path
    import pickle
    import sys
    import pandas as pd
    from threadpoolctl import threadpool_limits
    import hmds_refinement

    if not perturb_amplitudes or any(not np.isfinite(a) or a < 0 for a in perturb_amplitudes):
        raise ValueError('Provide nonnegative perturbation amplitudes')
    if (max_blocks < 1 or iterations_per_block < 1 or max_profile_steps < 1
            or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0
            or not np.isfinite(ridge_fraction) or ridge_fraction <= 0):
        raise ValueError('Iteration budgets, tolerance and numerical ridge must be positive')
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    configuration = dict(dimension=2, n_starts=n_starts, seed=seed, initial_max_iter=initial_max_iter,
                         initial_lambda_bounds=initial_lambda_bounds, final_lambda_bounds='positive; no finite bound',
                         variance_floor=variance_floor, max_blocks=max_blocks,
                         gradient_metric='local_geometric',
                         coordinate_gradient_norm='maximum all-point orthonormal L2 norm, including anchor',
                         iterations_per_block=iterations_per_block, gradient_tolerance=gradient_tolerance,
                         perturb_amplitudes=perturb_amplitudes, perturb_seed=perturb_seed,
                         ridge_fraction=ridge_fraction, max_profile_steps=max_profile_steps,
                         conditional_ridge_reduction=HMDS_PROFILE_RIDGE_REDUCTION,
                         conditional_anchor_policy='change central anchor if its unconverged geometric gradient dominates coordinates',
                         fit_strategy='conditional initialization, adaptive lambda profile, local perturbations')
    (out / 'configuration.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    (out / 'input_snapshot.pkl').write_bytes(pickle.dumps(dict(
        distance=distance, variance=variance, pair_mask=pair_mask, bootstrap_parameters=bootstrap_parameters,
    )))
    for module in (sys.modules[__name__], hmds_refinement):
        path = Path(module.__file__)
        (out / (path.stem + '_used.py')).write_bytes(path.read_bytes())

    def refine(reference, label):
        print(f'2D {label}: conditioning and lambda profile', flush=True)
        return _profile_neural_hmds_2d(
            reference, out / f'{label}_profile_checkpoint.pkl', max_blocks,
            iterations_per_block, gradient_tolerance, ridge_fraction, max_profile_steps)

    def reanchor(reference, parameters):
        ref = hmds_refinement.reanchor_hmds(reference, parameters, hmds_geometry, hmds_loss_gradient)
        ref['parameters'].update(fixed_lambda=None, lambda_bounds=(0.0, np.inf))
        return ref

    with threadpool_limits(limits=1, user_api='blas'):
        initial = initialize_neural_hmds(
            distance, variance, pair_mask, dimension=2, n_starts=n_starts, seed=seed,
            max_iter=initial_max_iter, lambda_bounds=initial_lambda_bounds, variance_floor=variance_floor,
        )
        initial['parameters']['bootstrap'] = copy.deepcopy(bootstrap_parameters)
        (out / 'initial_fit.pkl').write_bytes(pickle.dumps(initial))
        initial['starts'].to_csv(out / 'initial_starts.csv')
        initial['history'].to_csv(out / 'initial_history.csv', index=False)
        ref = reanchor(initial, initial['optimizer_result'].x)
        try:
            ref, fitted = refine(ref, 'initial')
        except _HMDSProfileError as error:
            (out / 'convergence.json').write_text(json.dumps(dict(
                joint_gradient_converged=False, reason=str(error)), indent=2), encoding='utf-8')
            raise
        runs = [dict(kind='initial_refinement', seed=seed, amplitude=0.0, reference=ref, fit=fitted)]
        (out / 'refinement_checkpoints.pkl').write_bytes(pickle.dumps(runs))
        for i, amplitude in enumerate(perturb_amplitudes):
            converged = [run for run in runs if run['fit']['diagnostics']['gradient_converged']]
            best = min(converged or runs, key=lambda run: run['fit']['diagnostics']['mean_nll'])
            parameters = best['fit']['parameters_vector'].copy()
            chart = best['fit']['halfspace_coordinates'].copy()
            rng = np.random.default_rng(perturb_seed + i)
            tangent = rng.normal(size=(len(chart) - 1, 2)) * amplitude
            chart[1:, 0] += np.exp(chart[1:, -1]) * tangent[:, 0]
            chart[1:, -1] += tangent[:, -1]
            parameters[:2 * (len(chart) - 1)] = chart[1:].ravel()
            ref = reanchor(best['reference'], parameters)
            try:
                ref, fitted = refine(ref, f'perturb_{i + 1}')
            except _HMDSProfileError as error:
                # Preserve failed attempts; a valid converged run remains usable.
                # Never label an unfinished lower-loss attempt as the final fit.
                checkpoint = out / f'perturb_{i + 1}_profile_checkpoint.pkl'
                state = pickle.loads(checkpoint.read_bytes())['current']
                ref, fitted = state['reference'], state['fit']
                diagnostic = fitted['diagnostics']
                raw_joint = float(np.max([diagnostic['projected_grad_inf'], abs(diagnostic['d_loss_d_log_lambda'])]))
                joint = float(np.max([diagnostic['geometric_projected_grad_max'], abs(diagnostic['d_loss_d_log_lambda'])]))
                diagnostic.update(projected_grad_inf=raw_joint, joint_projected_grad_inf=raw_joint,
                                  geometric_projected_grad_max=joint, joint_geometric_grad_max=joint,
                                  convergence_grad_norm=joint, gradient_converged=False, stop_reason=str(error))
            runs.append(dict(kind='local_perturbation', seed=perturb_seed + i, amplitude=amplitude,
                             reference=ref, fit=fitted))
            (out / 'refinement_checkpoints.pkl').write_bytes(pickle.dumps(runs))

        table = pd.DataFrame([dict(run=i, kind=r['kind'], seed=r['seed'], amplitude=r['amplitude'],
                                   mean_nll=r['fit']['diagnostics']['mean_nll'],
                                   lambda_value=r['fit']['diagnostics']['lambda_value'],
                                   relative_rmse=r['fit']['diagnostics']['relative_rmse'],
                                   projected_gradient=r['fit']['diagnostics']['projected_grad_inf'],
                                   geometric_gradient=r['fit']['diagnostics']['geometric_projected_grad_max'],
                                   convergence_metric=r['fit']['diagnostics']['gradient_metric'],
                                   converged=r['fit']['diagnostics']['gradient_converged'],
                                   stop_reason=r['fit']['diagnostics']['stop_reason'])
                              for i, r in enumerate(runs)])
        table.to_csv(out / 'all_runs.csv', index=False)
        eligible = table.loc[table.converged]
        if eligible.empty:
            raise RuntimeError(f'No 2D run met the full convergence criterion; inspect {out}')
        winner = int(eligible.mean_nll.idxmin())
        reference = copy.deepcopy(runs[winner]['reference'])
        fit = copy.deepcopy(runs[winner]['fit'])
        checks = check_neural_hmds_fit(reference, fit, out, gradient_tolerance)
        display_points, display_error = center_hmds_for_display(fit['halfspace_coordinates'])

    ids = reference['coordinates'].index
    columns = ['Poincare1', 'Poincare2']
    fit['coordinates'] = pd.DataFrame(hmds_chart_to_ball(fit['halfspace_coordinates']), index=ids, columns=columns)
    fit['display_coordinates'] = pd.DataFrame(display_points, index=ids, columns=columns)
    fit['diagnostics'].update(joint_projected_grad_inf=checks['joint_projected_grad_inf'],
                              joint_geometric_grad_max=checks['joint_geometric_grad_max'],
                              joint_convergence_grad_norm=checks['joint_convergence_grad_norm'],
                              joint_gradient_converged=True, curvature_estimated=True)
    # Keep the reference ready for the existing 3D lift, using this run's row order.
    reference['coordinates'] = fit['coordinates'].copy()
    reference['halfspace_coordinates'] = fit['halfspace_coordinates'].copy()
    reference['pairs'] = fit['pairs'].copy()
    reference['sigma'] = fit['sigma'].copy()
    reference['lambda'] = fit['diagnostics']['lambda_value']
    reference['optimizer_result'].x = fit['parameters_vector'].copy()
    pair_i = ids.get_indexer(reference['pairs'].sample_i)
    pair_j = ids.get_indexer(reference['pairs'].sample_j)
    reference['optimizer_result'].fun, reference['optimizer_result'].jac = hmds_loss_gradient(
        fit['parameters_vector'], len(ids), 2, pair_i, pair_j, reference['pairs'].input_distance.to_numpy(),
        np.maximum(reference['pairs'].bootstrap_variance.to_numpy(), variance_floor),
        reference['parameters']['variance_parameter_scale'],
    )
    report = dict(n_samples=len(ids), n_pairs=len(fit['pairs']), dimension=2, selected_run=winner,
                  diagnostics=fit['diagnostics'], configuration=configuration,
                  gradient_checks=dict(checked_parameters=checks['checked_parameters'], all_passed=True),
                  unconverged_runs=int((~table.converged).sum()),
                  before=dict(relative_rmse=float(initial['diagnostics']['relative_rmse'])),
                  display_isometry_max_distance_error=display_error,
                  input_source='Current neural D, animal-bootstrap V and pair mask; no saved initial fit',
                  neural_hmds_code_sha256=hashlib.sha256((out / 'neural_hmds_used.py').read_bytes()).hexdigest(),
                  limitation='Lowest-loss checked local solution; not proof of a global optimum')
    bundle = dict(reference=reference, fit=fit, report=report, runs=table, output_directory=str(out))
    fit['coordinates'].to_csv(out / 'embedding_coordinates.csv')
    fit['display_coordinates'].to_csv(out / 'embedding_display_coordinates.csv')
    fit['pairs'].to_csv(out / 'fitted_pairs.csv', index=False)
    fit['sigma'].rename('extra_residual_sigma').to_csv(out / 'residual_sigma.csv')
    fit['history'].to_csv(out / 'selected_history.csv', index=False)
    fit['profile_history'].to_csv(out / 'selected_profile_history.csv', index=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    (out / 'hmds_2d_result.pkl').write_bytes(pickle.dumps(bundle))
    print('Saved checked 2D fit:', out / 'hmds_2d_result.pkl', flush=True)
    return bundle
