"""Refine existing HMDS fits with refreshed likelihood-based coordinate scaling.

The data, Gaussian composite likelihood, q >= 0 constraints and anchor are
unchanged. The positive-definite matrix only changes optimizer units; it is
not a prior or a penalty added to the scientific objective.
"""

import copy
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.linalg import null_space


def refine_hmds(reference, geometry, loss_gradient, projected_gradient,
                initial_parameters=None, fixed_lambda=None, max_blocks=30,
                iterations_per_block=1000, ridge_fraction=1e-5,
                gradient_tolerance=1e-6, label='', verbose=True,
                inner_ftol=1e-14, inner_gtol=1e-8):
    """Warm continuation; fixed_lambda=None frees lambda with no finite bound.

    Report both the ORIGINAL parameter-unit projected gradient and a local
    geometric gradient. Only the original threshold sets gradient_converged.
    Inner tolerances are tightened, not relaxed, to resolve previous stagnation.
    """
    settings = reference['parameters']
    ids, pairs = reference['coordinates'].index, reference['pairs']
    n, d, m = len(ids), settings['dimension'], len(pairs)
    nc = (n - 1) * d
    pi, pj = ids.get_indexer(pairs.sample_i), ids.get_indexer(pairs.sample_j)
    observed = pairs.input_distance.to_numpy(float)
    base_variance = np.maximum(pairs.bootstrap_variance.to_numpy(float), settings['variance_floor'])
    variance_scale = settings['variance_parameter_scale']
    x = np.asarray(reference['optimizer_result'].x if initial_parameters is None
                   else initial_parameters, dtype=float).copy()
    if x.shape != (nc + 1 + n,) or not np.isfinite(x).all() or (x[nc + 1:] < 0).any():
        raise ValueError('Invalid initial parameters')
    if (pi < 0).any() or (pj < 0).any() or d < 2:
        raise ValueError('Invalid sample labels or embedding dimension')
    lo, hi = np.full(x.size, -np.inf), np.full(x.size, np.inf)
    lo[nc + 1:] = 0
    if fixed_lambda is not None:
        if not np.isfinite(fixed_lambda) or fixed_lambda <= 0:
            raise ValueError('fixed_lambda must be positive and finite')
        x[nc] = lo[nc] = hi[nc] = np.log(fixed_lambda)
    mean_indices = np.arange(nc + int(fixed_lambda is None))
    horizontal = np.array([k for k in range(nc) if k % d != d - 1])
    heights = np.arange(d - 1, nc, d)
    history, last_result, stop = [], None, 'block budget exhausted'

    def objective(values):
        return loss_gradient(values, n, d, pi, pj, observed, base_variance, variance_scale)

    def diagnostics(values):
        value, grad = objective(values)
        pg = projected_gradient(values, grad, lo, hi)
        local = pg.copy()
        local[horizontal] *= np.exp(values[(horizontal // d) * d + d - 1])
        spacing = np.abs(np.spacing(values[horizontal]))
        resolution = np.exp(values[(horizontal // d) * d + d - 1]) / spacing
        return dict(mean_nll=value, lambda_value=float(np.exp(values[nc])),
                    projected_grad_inf=float(np.max(np.abs(pg))),
                    local_geometric_grad_inf=float(np.max(np.abs(local))),
                    d_loss_d_log_lambda=float(grad[nc]),
                    min_height_in_float_steps=float(np.min(resolution)),
                    min_log_height=float(values[heights].min()), max_log_height=float(values[heights].max()))

    initial_diagnostics = diagnostics(x)
    for block in range(max_blocks):
        started = perf_counter()
        before = objective(x)[0]
        chart = np.vstack([np.zeros(d), x[:nc].reshape(n-1, d)])
        scale, q = np.exp(x[nc]), x[nc + 1:] * variance_scale
        total_variance = base_variance + q[pi] + q[pj]
        weight = 1 / np.sqrt(m * total_variance)
        distance, gi, gj = geometry(chart, pi, pj)
        jac = np.zeros((m, len(mean_indices)))
        for endpoint, endpoint_gradient in ((pi, gi), (pj, gj)):
            valid = endpoint > 0
            rows = np.flatnonzero(valid)
            for axis in range(d):
                jac[rows, (endpoint[valid] - 1) * d + axis] += endpoint_gradient[valid, axis] / scale * weight[valid]
        if fixed_lambda is None:
            jac[:, -1] = -distance / scale * weight
        physical_units = np.ones(len(mean_indices))
        physical_units[horizontal] = np.exp(x[(horizontal // d) * d + d - 1])
        if not np.isfinite(physical_units).all() or (physical_units <= 0).any():
            stop = 'coordinate scaling exceeds float64 representation'
            break
        # The fixed anchor removes translations but leaves an unidentifiable
        # rotation. Remove its infinitesimal direction from each local chart.
        # This changes no pairwise distance and adds no scientific constraint.
        quotient = np.eye(len(mean_indices))
        height = np.exp(chart[1:, -1])
        horizontal_chart = chart[1:, :-1]
        rotations = []
        for axis in range(d-1):
            field = np.zeros((n-1, d))
            field[:, :-1] = horizontal_chart[:, axis, None]*horizontal_chart
            field[:, axis] += (1-np.sum(horizontal_chart**2, axis=1)-height**2)/2
            field[:, :-1] /= height[:, None]
            field[:, -1] = horizontal_chart[:, axis]
            rotations.append(np.r_[field.ravel(), np.zeros(len(mean_indices)-nc)])
        for first in range(d-1):
            for second in range(first+1, d-1):
                field = np.zeros((n-1, d))
                field[:, first] = -horizontal_chart[:, second]/height
                field[:, second] = horizontal_chart[:, first]/height
                rotations.append(np.r_[field.ravel(), np.zeros(len(mean_indices)-nc)])
        rotation_matrix = np.asarray(rotations)
        if np.isfinite(rotation_matrix).all() and np.linalg.norm(rotation_matrix) > 0:
            quotient = null_space(rotation_matrix)
        scaled_jac = (jac * physical_units) @ quotient
        fisher = scaled_jac.T @ scaled_jac
        eigenvalues, eigenvectors = np.linalg.eigh(fisher)
        ridge = max(float(eigenvalues[-1]) * ridge_fraction, 1e-12)
        transform = physical_units[:, None] * (quotient @ (eigenvectors / np.sqrt(np.maximum(eigenvalues, 0) + ridge)))
        qdiag = np.zeros(n)
        for endpoint in (pi, pj):
            np.add.at(qdiag, endpoint, 0.5 / m * (variance_scale / total_variance)**2)
        qunits = 1 / np.sqrt(np.maximum(qdiag, 1e-12))
        nm = transform.shape[1]

        def unpack(z):
            values = x.copy()
            values[mean_indices] += transform @ z[:nm]
            values[nc + 1:] = qunits * z[nm:]
            return values

        def transformed_objective(z):
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                value, grad = objective(unpack(z))
            if not np.isfinite(value) or not np.isfinite(grad).all():
                return np.inf, np.zeros_like(z)
            return value, np.r_[transform.T @ grad[mean_indices], qunits * grad[nc + 1:]]

        initial = np.r_[np.zeros(nm), x[nc + 1:] / qunits]
        last_result = minimize(
            transformed_objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(None, None)] * nm + [(0, None)] * n,
            options={'maxiter': iterations_per_block, 'maxfun': iterations_per_block * 50 + 1,
                     'ftol': inner_ftol, 'gtol': inner_gtol, 'maxls': 50, 'maxcor': 30},
        )
        candidate = unpack(last_result.x)
        candidate_loss = objective(candidate)[0]
        if np.isfinite(candidate_loss) and candidate_loss <= before:
            x = candidate
        row = diagnostics(x) | dict(block=block, loss_reduction=before - objective(x)[0],
                                    n_iter=last_result.nit, optimizer_success=bool(last_result.success),
                                    message=str(last_result.message), seconds=perf_counter() - started)
        history.append(row)
        if verbose:
            print(f'{label} block={block+1}: loss={row["mean_nll"]:.9f}, lambda={row["lambda_value"]:.5g}, '
                  f'original gradient={row["projected_grad_inf"]:.3g}, '
                  f'local gradient={row["local_geometric_grad_inf"]:.3g}', flush=True)
        if row['min_height_in_float_steps'] < 32:
            stop = 'local horizontal scale is below float64 resolution'
            break
        if row['projected_grad_inf'] <= gradient_tolerance:
            stop = 'original projected-gradient criterion met'
            break
        if len(history) >= 3 and all(h['loss_reduction'] < 1e-12 for h in history[-3:]):
            stop = 'objective stagnated for three refreshed scalings'
            break

    final = diagnostics(x)
    chart = np.vstack([np.zeros(d), x[:nc].reshape(n-1, d)])
    fitted_distance = geometry(chart, pi, pj)[0] / np.exp(x[nc])
    fitted_pairs = pairs.copy()
    fitted_pairs['fitted_distance'] = fitted_distance
    fitted_pairs['residual'] = fitted_distance-observed
    fitted_pairs['total_variance'] = base_variance + variance_scale * (x[nc + 1:][pi]+x[nc + 1:][pj])
    final.update(gradient_converged=final['projected_grad_inf'] <= gradient_tolerance,
                 relative_rmse=float(np.linalg.norm(fitted_distance-observed)/np.linalg.norm(observed)),
                 stop_reason=stop, blocks=len(history), fixed_lambda=fixed_lambda,
                 initial_mean_nll=initial_diagnostics['mean_nll'])
    return dict(parameters_vector=x, halfspace_coordinates=chart,
                sigma=pd.Series(np.sqrt(x[nc+1:]*variance_scale), index=ids),
                pairs=fitted_pairs, diagnostics=final, history=pd.DataFrame(history),
                settings=copy.deepcopy(settings) | dict(fixed_lambda=fixed_lambda, lambda_bounds=None,
                    ridge_fraction=ridge_fraction, gradient_tolerance=gradient_tolerance,
                    iterations_per_block=iterations_per_block, max_blocks=max_blocks,
                    inner_ftol=inner_ftol, inner_gtol=inner_gtol,
                    method='L-BFGS-B with refreshed Fisher scaling; unchanged Gaussian composite likelihood'))


def reanchor_hmds(reference, parameters_vector, geometry, loss_gradient):
    """Choose a central observed anchor and a well-scaled orientation.

    These are exact hyperbolic isometries, checked on ALL sample pairs. Sample
    IDs, observed pairs and variances are preserved; only parameter order changes.
    """
    result = copy.deepcopy(reference)
    ids = reference['coordinates'].index
    d = reference['parameters']['dimension']
    n, nc = len(ids), (len(ids)-1)*d
    values = np.asarray(parameters_vector, dtype=float)
    chart = np.vstack([np.zeros(d), values[:nc].reshape(n-1, d)])
    pi, pj = np.triu_indices(n, 1)
    distances = geometry(chart, pi, pj)[0]
    matrix = np.zeros((n, n))
    matrix[pi, pj] = matrix[pj, pi] = distances
    center = int(np.argmin(matrix.max(axis=1)))
    base = chart.copy()
    base[:, :-1] = (base[:, :-1]-chart[center, :-1])*np.exp(-chart[center, -1])
    base[:, -1] -= chart[center, -1]

    def rotate(theta, axis):
        a, b, c, den_constant = np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)
        with np.errstate(divide='ignore'):
            logscale = np.maximum.reduce([np.zeros(n), np.log(np.max(np.abs(base[:, :-1]), axis=1)), base[:, -1]])
        xx, yy = base[:, :-1]*np.exp(-logscale[:, None]), np.exp(base[:, -1]-logscale)
        one = np.exp(-logscale)
        others = np.sum(np.delete(xx, axis, axis=1)**2, axis=1)
        denominator = (c*xx[:, axis]+den_constant*one)**2+c**2*(others+yy**2)
        horizontal = xx*one[:, None]/denominator[:, None]
        horizontal[:, axis] = (a*c*(np.sum(xx**2, axis=1)+yy**2)
            +(a*den_constant+b*c)*xx[:, axis]*one+b*den_constant*one**2)/denominator
        height = base[:, -1]-2*logscale-np.log(denominator)
        return np.column_stack([horizontal, height])

    angles = np.linspace(0, np.pi, 2048, endpoint=False)
    chosen_angles = []
    for axis in range(d-1):
        theta = max(angles, key=lambda angle: rotate(angle, axis)[:, -1].min())
        base = rotate(theta, axis)
        chosen_angles.append(float(theta))
    rotated = base
    rotated[center] = 0
    error = float(np.max(np.abs(geometry(rotated, pi, pj)[0]-distances)))
    if error > 1e-6:
        raise FloatingPointError(f'Reanchoring changed pair distances by {error:g}')
    order = np.r_[center, np.delete(np.arange(n), center)]
    newids, newchart = ids[order], rotated[order]
    x = np.r_[newchart[1:].ravel(), values[nc], values[nc+1:][order]]
    with np.errstate(divide='ignore'):
        log_x = np.log(np.max(np.abs(newchart[:, :-1]), axis=1))
    display_scale = np.maximum.reduce([np.zeros(n), log_x, newchart[:, -1]])
    inverse = np.exp(-display_scale)
    scaled_x = newchart[:, :-1]*inverse[:, None]
    height = np.exp(newchart[:, -1]-display_scale)
    squared_x = np.sum(scaled_x**2, axis=1)
    denominator = squared_x+(height+inverse)**2
    ball = np.column_stack([2*scaled_x*inverse[:, None]/denominator[:, None],
                            (squared_x+height**2-inverse**2)/denominator])
    result['coordinates'] = pd.DataFrame(ball, index=newids, columns=reference['coordinates'].columns)
    result['halfspace_coordinates'] = newchart
    result['sigma'] = pd.Series(np.sqrt(x[nc+1:]*reference['parameters']['variance_parameter_scale']), index=newids)
    result['lambda'] = float(np.exp(x[nc]))
    result['parameters'].update(fixed_lambda=result['lambda'], anchor_sample=newids[0])
    result['optimizer_result'].x = x
    pairs, settings = result['pairs'], result['parameters']
    value, grad = loss_gradient(x, n, d, newids.get_indexer(pairs.sample_i), newids.get_indexer(pairs.sample_j),
                               pairs.input_distance.to_numpy(),
                               np.maximum(pairs.bootstrap_variance.to_numpy(), settings['variance_floor']),
                               settings['variance_parameter_scale'])
    result['optimizer_result'].fun, result['optimizer_result'].jac = value, grad
    result['reanchoring'] = dict(original_ids=list(ids), anchor_sample=ids[center],
                               original_anchor_x=chart[center, :-1].tolist(),
                               original_anchor_log_height=float(chart[center, -1]),
                               rotation_angles=chosen_angles, max_distance_change=error)
    return result


def profile_refine_hmds(reference, geometry, loss_gradient, projected_gradient,
                        lambda_bracket=(16., 17.), initial_parameters=None,
                        max_blocks=10, ridge_fraction=1e-7, verbose=True):
    """Solve a LOCAL curvature profile with converged nuisance-parameter fits.

    The bracket locates a zero of the profile score, not a bound imposed on
    lambda or a confidence interval. No claim of a global optimum is made.
    """
    lower, upper = lambda_bracket
    if not 0 < lower < upper:
        raise ValueError('Use a positive increasing lambda bracket')
    start = np.asarray(reference['optimizer_result'].x if initial_parameters is None
                       else initial_parameters, dtype=float).copy()
    nc = (len(reference['coordinates'])-1)*reference['parameters']['dimension']
    cache = {}
    records = []

    def score(log_lambda):
        initial = (cache[min(cache, key=lambda key: abs(key-log_lambda))]['parameters_vector']
                   if cache else start)
        fit = refine_hmds(reference, geometry, loss_gradient, projected_gradient,
                          initial_parameters=initial, fixed_lambda=float(np.exp(log_lambda)),
                          max_blocks=max_blocks, ridge_fraction=ridge_fraction,
                          gradient_tolerance=1e-7, verbose=False)
        cache[log_lambda] = fit
        records.append(fit['diagnostics'].copy())
        if not fit['diagnostics']['gradient_converged']:
            raise RuntimeError('Conditional fit failed to converge; profile score is not reliable')
        if verbose:
            print(f'lambda={np.exp(log_lambda):.8g}, loss={fit["diagnostics"]["mean_nll"]:.10f}, '
                  f'profile score={fit["diagnostics"]["d_loss_d_log_lambda"]:.3g}', flush=True)
        return fit['diagnostics']['d_loss_d_log_lambda']

    root = brentq(score, np.log(lower), np.log(upper), xtol=1e-8, rtol=1e-10, maxiter=30)
    score(root)
    best = cache[root]
    # Include the lambda direction explicitly when declaring joint convergence.
    joint_norm = max(best['diagnostics']['projected_grad_inf'],
                     abs(best['diagnostics']['d_loss_d_log_lambda']))
    best['diagnostics'].update(joint_projected_grad_inf=joint_norm,
                               joint_gradient_converged=joint_norm <= 1e-6,
                               conditional_projected_grad_inf=best['diagnostics']['projected_grad_inf'],
                               projected_grad_inf=joint_norm, gradient_converged=joint_norm <= 1e-6,
                               fixed_lambda=None, curvature_estimated=True,
                               stop_reason='joint projected-gradient criterion met' if joint_norm <= 1e-6
                               else 'profile root found; joint gradient criterion not met',
                               lambda_bracket=tuple(lambda_bracket))
    best['settings'].update(fixed_lambda=None, fit_strategy='local profile-score root with converged conditional fits')
    best['profile_history'] = pd.DataFrame(records)
    return best
