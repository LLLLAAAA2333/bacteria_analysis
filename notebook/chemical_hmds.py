"""Chemical HMDS with one shared residual variance and equal pair weights.

The shared pair variance is profiled analytically as mean squared residual.
There is no feature bootstrap, per-sample noise variance, or measurement-error claim.
"""
import copy
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def centered_ball(chart, geometry, chart_to_ball):
    n, dimension = chart.shape
    origin = np.median(chart[:, :-1], axis=0)
    units = np.maximum(np.std(chart[:, :-1], axis=0), 1)
    def objective(z):
        center = np.r_[origin + units*z[:-1], z[-1]]
        distance, derivative, _ = geometry(np.vstack([center, chart]), np.zeros(n, int), np.arange(1, n+1))
        gradient = np.mean(distance[:, None]*derivative, axis=0)
        gradient[:-1] *= units
        return .5*np.mean(distance**2), gradient
    result = minimize(objective, np.r_[np.zeros(dimension-1), np.log(np.linalg.norm(units))],
                      jac=True, method='L-BFGS-B', options=dict(maxiter=1000, ftol=1e-14, gtol=1e-9))
    center = np.r_[origin+units*result.x[:-1], result.x[-1]]
    transformed = chart.copy()
    transformed[:, :-1] = (chart[:, :-1]-center[:-1])*np.exp(-center[-1])
    transformed[:, -1] -= center[-1]
    pi, pj = np.triu_indices(n, 1)
    change = np.max(np.abs(geometry(chart, pi, pj)[0]-geometry(transformed, pi, pj)[0]))
    if not np.isfinite(change) or change > 1e-8:
        raise RuntimeError('Centering did not preserve hyperbolic distances')
    return chart_to_ball(transformed), float(change)


def fit_chemical_hmds(rdm, geometry, chart_to_ball, output_directory, seed=20260919,
                      initial_lambdas=(0.5, 2.0, 5.0), max_blocks=20,
                      iterations_per_block=1000, gradient_tolerance=1e-6, fixed_lambda_2d=10.0):
    """Generate both dimensions directly from the chemical RMS-distance DataFrame."""
    from chemical_hmds_polar import polar_start, refine_polar, polar_gradient_check
    from sklearn.manifold import MDS
    from threadpoolctl import threadpool_limits

    if not rdm.index.is_unique or not rdm.index.equals(rdm.columns):
        raise ValueError('RDM needs identical unique row/column IDs')
    values = rdm.to_numpy(float, copy=True)
    if len(values) < 4 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Need >=4 samples and finite nonnegative distances')
    if not np.allclose(values, values.T) or not np.allclose(np.diag(values), 0):
        raise ValueError('RDM must be symmetric with zero diagonal')
    maximum = values.max()
    if maximum <= 0:
        raise ValueError('All chemical distances are zero')
    # A central observed sample fixes translation without adding a distance constraint.
    anchor = int(np.argmin(values.max(axis=1)))
    order = np.r_[anchor, np.delete(np.arange(len(values)), anchor)]
    rdm = rdm.iloc[order, order].copy()
    values = rdm.to_numpy(float, copy=True)
    a = 2/maximum
    scaled = values*a
    n = len(values)
    pi, pj = np.triu_indices(n, 1)
    observed = scaled[pi, pj]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory/'chemical_input.pkl').write_bytes(pickle.dumps(rdm))
    results, run_rows = {}, []
    with threadpool_limits(limits=1, user_api='blas'):
        for dimension in (2, 3):
            model = MDS(n_components=dimension, metric_mds=True, metric='precomputed', init='random',
                        n_init=4, random_state=seed, max_iter=3000, eps=1e-10, n_jobs=1)
            euclidean = model.fit_transform(scaled)
            candidates = []
            for start, initial_lambda in enumerate(initial_lambdas):
                fixed_lambda = fixed_lambda_2d if dimension == 2 else None
                initial_coordinates = euclidean.copy()
                if fixed_lambda is not None:
                    initial_lambda = fixed_lambda
                    jitter = 0.05*start
                    initial_coordinates += np.random.default_rng(seed+start).normal(size=euclidean.shape)*jitter
                x = polar_start(initial_coordinates, initial_lambda)
                fit = refine_polar(x, n, dimension, observed, max_blocks,
                                    iterations_per_block, gradient_tolerance,
                                    label=f'Chemical {dimension}D start {start+1}/{len(initial_lambdas)}',
                                    fixed_lambda=fixed_lambda)
                candidates.append(fit)
                run_rows.append(dict(start=start, initial_lambda=initial_lambda, **fit['diagnostics']))
                (output_directory/f'chemical_{dimension}d_starts.pkl').write_bytes(pickle.dumps(candidates))
            best = copy.deepcopy(min(candidates, key=lambda fit: fit['diagnostics']['mean_nll']))
            steps, checked = polar_gradient_check(best, n, dimension, observed)
            steps.to_csv(output_directory/f'gradient_steps_{dimension}d.csv', index=False)
            checked.rename('adjacent_steps_agree').to_csv(output_directory/f'gradient_summary_{dimension}d.csv')
            ball, change = centered_ball(best['halfspace_coordinates'], geometry, chart_to_ball)
            best['coordinates'] = pd.DataFrame(ball, index=rdm.index,
                                               columns=[f'Poincare{k+1}' for k in range(dimension)])
            predicted = observed+best['residual']
            best['pairs'] = pd.DataFrame(dict(sample_i=rdm.index.to_numpy()[pi], sample_j=rdm.index.to_numpy()[pj],
                                              input_distance=values[pi, pj], fitted_distance=predicted/a))
            best['diagnostics'].update(gradient_checks_passed=bool(checked.all()), checked_parameters=len(checked),
                                       display_isometry_error=change, distance_scale_a=a,
                                       shared_sigma_original=best['diagnostics']['shared_sigma_scaled']/a,
                                       metric_mds_relative_rmse=float(np.linalg.norm(pdist(euclidean)-observed)/np.linalg.norm(observed)),
                                       lambda_initial_min=min(initial_lambdas))
            best['coordinates'].to_csv(output_directory/f'embedding_{dimension}d.csv')
            best['pairs'].to_csv(output_directory/f'fitted_pairs_{dimension}d.csv', index=False)
            results[dimension] = best
    summary = pd.DataFrame([fit['diagnostics'] for fit in results.values()]).set_index('dimension')
    runs = pd.DataFrame(run_rows)
    bundle = dict(results=results, summary=summary, runs=runs, input_rdm=rdm.copy(),
                  parameters=dict(seed=seed, initial_lambdas=initial_lambdas, max_blocks=max_blocks,
                                  iterations_per_block=iterations_per_block, gradient_tolerance=gradient_tolerance,
                                  distance_scale_a=a, residual_model='shared sigma; pair variance = 2*sigma**2',
                                  measurement_variance_available=False, bootstrap=False,
                                  likelihood='Gaussian mean NLL with analytically profiled shared variance',
                                  coordinate_system='log input-unit radius and spherical angles; free log lambda',
                                  fixed_lambda_2d=fixed_lambda_2d,
                                  input_provenance=copy.deepcopy(rdm.attrs)))
    summary.to_csv(output_directory/'summary.csv')
    runs.to_csv(output_directory/'runs.csv', index=False)
    (output_directory/'chemical_hmds_result.pkl').write_bytes(pickle.dumps(bundle))
    return bundle


def plot_chemical_hmds(bundle, color_mapping=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from chemical_colors import plotly_colors, color_provenance
    if color_mapping is None:
        raise ValueError('Pass the fixed chemical reference color_mapping from cell12 or cell42')
    fig = make_subplots(rows=2, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}], [{'type':'scene'}, {'type':'xy'}]],
                        column_widths=[.55,.45], horizontal_spacing=.12, vertical_spacing=.15,
                        subplot_titles=['Chemical HMDS: 2D', '2D Shepard', 'Chemical HMDS: 3D', '3D Shepard'])
    for row, dimension in enumerate((2,3), start=1):
        fit = bundle['results'][dimension]
        coordinates, pairs = fit['coordinates'], fit['pairs']
        marker = plotly_colors(color_mapping, coordinates.index)
        marker.update(size=6 if dimension==2 else 4, showscale=dimension==3)
        marker['colorbar'].update(x=.47, len=.36, y=.21, thickness=12)
        arguments = dict(x=coordinates.iloc[:,0], y=coordinates.iloc[:,1], mode='markers', marker=marker,
                         text=coordinates.index,
                         customdata=color_mapping['scores'].reindex(coordinates.index).to_numpy(),
                         hovertemplate='%{text}<br>Chemical PCo1: %{customdata:.3f}<extra></extra>', showlegend=False)
        if dimension == 2:
            fig.add_trace(go.Scatter(**arguments), row=row, col=1)
            theta = np.linspace(0,2*np.pi,200)
            fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='gray'),
                                    showlegend=False, hoverinfo='skip'), row=row,col=1)
            fig.update_xaxes(range=[-1.05,1.05], title_text='Poincare 1', row=row,col=1)
            fig.update_yaxes(range=[-1.05,1.05], scaleanchor='x', scaleratio=1, title_text='Poincare 2',row=row,col=1)
        else:
            fig.add_trace(go.Scatter3d(**arguments,z=coordinates.iloc[:,2]),row=row,col=1)
        maximum = max(pairs.input_distance.max(), pairs.fitted_distance.max())*1.03
        fig.add_trace(go.Scattergl(x=pairs.input_distance,y=pairs.fitted_distance,mode='markers',
                                   marker=dict(size=4,opacity=.3,color='#b76d2b'),showlegend=False),row=row,col=2)
        fig.add_trace(go.Scatter(x=[0,maximum],y=[0,maximum],mode='lines',line=dict(color='black',dash='dash'),
                                hoverinfo='skip',showlegend=False),row=row,col=2)
        fig.update_xaxes(range=[0,maximum],title_text='Observed RMS log2FC distance',row=row,col=2)
        fig.update_yaxes(range=[0,maximum],title_text='Fitted distance (original units)',
                         scaleanchor='x2' if dimension==2 else 'x3',scaleratio=1,row=row,col=2)
        rmse=fit['diagnostics']['relative_rmse']
        curvature = ('fitted' if fit['diagnostics']['lambda_fitted'] else 'fixed')
        status = '' if fit['diagnostics']['converged'] and fit['diagnostics']['gradient_checks_passed'] else ' | PROVISIONAL'
        fig.layout.annotations[0 if dimension==2 else 2].text += (
            f' | RMSE {rmse:.2%}<br>lambda {curvature}: {fit["diagnostics"]["lambda_value"]:.3f}{status}')
    fig.update_layout(height=1000,template='plotly_white',margin=dict(l=35,r=25,t=60,b=70),
                      meta=color_provenance(color_mapping),
                      scene=dict(xaxis=dict(title='Poincare 1',range=[-1.05,1.05]),
                                 yaxis=dict(title='Poincare 2',range=[-1.05,1.05]),
                                 zaxis=dict(title='Poincare 3',range=[-1.05,1.05]),aspectmode='cube'))
    fig.add_annotation(text=color_mapping['note'], x=.5, y=-.06, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=11))
    return fig
