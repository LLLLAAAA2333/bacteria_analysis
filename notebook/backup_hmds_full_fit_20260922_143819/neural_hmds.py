"""Shared HMDS geometry, likelihood and numerical checks.

Extracted unchanged from the exploratory notebook. Importing this module does
not fit a model, bootstrap data, or read notebook cells. Coordinates use the
unit-curvature upper half-space; q = sigma**2 is additional model residual variance.
"""
import numpy as np

# Derivative-check tolerances; these are not optimizer convergence thresholds.
HMDS_DIAG_FD_ATOL = 1e-7
HMDS_DIAG_FD_RTOL = 1e-3
HMDS_LOCAL_STEPS = 10.0 ** -np.arange(1, 10)
HMDS_LOCAL_MIN_ULPS = 32


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
        lower[n_coords], upper[n_coords] = np.log(settings['lambda_bounds'])
    else:
        lower[n_coords] = upper[n_coords] = np.log(fixed_lambda)
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
