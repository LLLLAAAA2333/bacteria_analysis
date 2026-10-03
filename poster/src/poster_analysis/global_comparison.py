"""Table-driven global comparisons for explicitly named chemical representations.

The main figure restores the original 380-feature reference-based log2FC view;
the 162-feature log-concentration view remains a separate supporting figure.
Functions receive numeric tables and never load inputs or refit embeddings.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


def distance_matrices(chemical_log2, neural_unit):
    """Return chemical RMS-log and neural cosine distances, aligned by strain ID.

    No chemical variance scaling, family reweighting, reference or imputation.
    The neural representation is already unit-normalized; cosine is calculated
    explicitly so small input rounding does not alter the distance definition.
    """
    for frame in (chemical_log2, neural_unit):
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError('Unique strain IDs and feature labels are required')
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Distances require finite, un-imputed input tables')
    if set(chemical_log2.index) != set(neural_unit.index):
        raise ValueError('Chemical and neural inputs must contain the same strains')
    if min(chemical_log2.shape) < 1 or len(chemical_log2) < 2:
        raise ValueError('At least two strains and one chemical annotation are required')
    ids = neural_unit.index
    x = chemical_log2.loc[ids].to_numpy(dtype=float)
    y = neural_unit.to_numpy(dtype=float)
    if np.any(np.linalg.norm(y, axis=1) <= 1e-12):
        raise ValueError('Cosine distance is undefined for a zero neural vector')
    chemical = squareform(pdist(x, metric='euclidean') / np.sqrt(x.shape[1]))
    neural = np.clip(squareform(pdist(y, metric='cosine')), 0, 2)
    return tuple(pd.DataFrame(v, index=ids, columns=ids) for v in (chemical, neural))


def paired_distances(chemical_distance, neural_distance, context=None):
    """One unordered off-diagonal pair per row; pairs share biological strains."""
    if not chemical_distance.index.equals(chemical_distance.columns):
        raise ValueError('Chemical distance matrix rows and columns must match')
    ids = chemical_distance.index
    if set(ids) != set(neural_distance.index) or set(ids) != set(neural_distance.columns):
        raise ValueError('Distance matrices must contain the same strain IDs')
    neural = neural_distance.loc[ids, ids]
    for frame in (chemical_distance, neural):
        if not np.isfinite(frame.to_numpy()).all() or not np.allclose(frame, frame.T):
            raise ValueError('Distances must be finite and symmetric')
        if not np.allclose(np.diag(frame), 0):
            raise ValueError('Distance diagonals must be zero')
    i, j = np.triu_indices(len(ids), k=1)
    pairs = pd.DataFrame({'strain_a': ids[i], 'strain_b': ids[j],
                          'chemical_distance': chemical_distance.to_numpy()[i,j],
                          'neural_distance': neural.to_numpy()[i,j]})
    if context is not None:
        if not context.index.is_unique or not set(ids).issubset(context.index):
            raise ValueError('Context must uniquely cover every strain')
        pairs['genus_a'] = context.loc[pairs.strain_a, 'genus'].to_numpy()
        pairs['genus_b'] = context.loc[pairs.strain_b, 'genus'].to_numpy()
        pairs['same_genus'] = pairs.genus_a.eq(pairs.genus_b)
    return pairs


def describe_correspondence(pairs):
    """Descriptive correlations only: deliberately no pair-independent p-value."""
    x, y = pairs[['chemical_distance', 'neural_distance']].to_numpy().T
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        raise ValueError('Two nonconstant distance vectors are required')
    return {'n_strains': len(set(pairs.strain_a) | set(pairs.strain_b)),
            'n_pairs': len(pairs),
            'pearson_r': float(np.corrcoef(x, y)[0,1]),
            'spearman_rho': float(np.corrcoef(rankdata(x), rankdata(y))[0,1])}


def plot_global_comparison(chemical_distance, neural_distance, order=None,
                           neural_limits=(0., 1.2)):
    """Two ordered RDMs and pair density; neural_limits controls display only.

    Values outside the heatmap limits saturate, with colorbar extensions marking
    the tails. Input distances and the pair-density coordinates are unchanged.
    """
    ids = chemical_distance.index.tolist() if order is None else list(order)
    if len(ids) != len(set(ids)) or set(ids) != set(chemical_distance.index):
        raise ValueError('Display order must contain every strain exactly once')
    pairs = paired_distances(chemical_distance, neural_distance)
    limits = np.asarray(neural_limits, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError('neural_limits must contain two finite increasing values')
    fig, axes = plt.subplots(1, 3, figsize=(13.6,4.6), layout='constrained')
    n = len(ids)
    for ax, values, title, (vmin, vmax), label in [
        (axes[0], neural_distance.loc[ids,ids].to_numpy(), 'Neural response patterns', limits, '1 − cosine'),
        (axes[1], chemical_distance.loc[ids,ids].to_numpy(), 'Chemical composition', (0., float(chemical_distance.max().max())), 'RMS difference in log2 concentration')]:
        mesh = ax.pcolormesh(np.arange(n+1), np.arange(n+1), values,
                            cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='flat')
        ax.set(xlim=(0,n), ylim=(n,0), aspect='equal', title=title,
               xlabel='Bacterial strains', ylabel='Bacterial strains')
        ax.set_xticks([]); ax.set_yticks([])
        below, above = np.any(values < vmin), np.any(values > vmax)
        extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        cb = fig.colorbar(mesh, ax=ax, fraction=.043, pad=.035, extend=extend)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=9)
        cb.solids.set_rasterized(False)
    ax = axes[2]
    density = ax.hexbin(pairs.chemical_distance, pairs.neural_distance,
                        gridsize=25, mincnt=1, cmap='Blues', linewidths=.1)
    ax.set(title='Matched strain pairs', xlabel='Chemical RMS log2 difference', ylabel='Neural 1 − cosine')
    ax.set_ylim(0, 2)
    cb = fig.colorbar(density, ax=ax, fraction=.043, pad=.035)
    cb.set_label('Pairs per bin', fontsize=9)
    cb.ax.tick_params(labelsize=9)
    cb.solids.set_rasterized(False)
    for letter, ax in zip('ABC', axes):
        ax.text(-.12, 1.07, letter, transform=ax.transAxes, weight='bold', fontsize=15)
    return fig


def plot_reference_fc_comparison(chemical_distance, neural_distance, order=None):
    """Restore the original 380-FC RDM layout, scales and pair-count colors.

    Layout follows the reviewed individual_comparison_display.plot_rdm without
    importing or executing that historical script. Distances are caller-supplied.
    """
    ids = chemical_distance.index.tolist() if order is None else list(order)
    if len(ids) != len(set(ids)) or set(ids) != set(chemical_distance.index):
        raise ValueError('Display order must contain every strain exactly once')
    paired_distances(chemical_distance, neural_distance)  # Validate, no transformation.
    neural = neural_distance.loc[ids, ids].to_numpy(float)
    chemical = chemical_distance.loc[ids, ids].to_numpy(float)
    style = {key: value for key, value in plt.rcParamsDefault.items() if key != 'backend'}
    style.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                  'axes.spines.top': False, 'axes.spines.right': False,
                  'svg.fonttype': 'none', 'pdf.fonttype': 42})
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#dddddd')
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(16.7, 5.2), layout='constrained')
        for ax, values, title, upper, label in (
                (axes[0], neural, 'Neural RDM', 2., '1 − cosine'),
                (axes[1], chemical, 'Chemical RDM', float(np.nanmax(chemical)), 'RMS Δlog₂FC')):
            im = ax.imshow(values, cmap=cmap, vmin=0, vmax=upper, interpolation='nearest')
            ax.set(title=title, xlabel='Sample', ylabel='Sample', xticks=[], yticks=[])
            fig.colorbar(im, ax=ax, shrink=.78, label=label)
        tri = np.triu_indices(len(ids), 1)
        xx, yy = chemical[tri], neural[tri]
        valid = np.isfinite(xx) & np.isfinite(yy)
        im = axes[2].hexbin(xx[valid], yy[valid], gridsize=27, mincnt=1,
                            cmap='RdBu_r', linewidths=0)
        axes[2].set(xlabel='Chemical RMS Δlog₂FC', ylabel='Neural 1 − cosine',
                    ylim=(0, 2), title='Matched sample pairs')
        fig.colorbar(im, ax=axes[2], shrink=.78, label='Pairs per bin')
    return fig


def plot_within_between(summary):
    """Cached genus-balanced comparisons; connecting lines are not intervals."""
    fig, axes = plt.subplots(1,2,figsize=(11,5.7),layout='constrained',sharey=True)
    summary = summary.sort_values(['n_strains','genus'], ascending=[False,True])
    yy = np.arange(len(summary))
    for ax, modality, title in zip(axes, ['chemical','neural'], ['Chemical distances','Neural distances']):
        within = summary[f'{modality}_within_median'].to_numpy()
        between = summary[f'{modality}_between_median'].to_numpy()
        ax.hlines(yy, np.minimum(within,between), np.maximum(within,between), color='#B9C1C5', lw=1.2)
        ax.scatter(within, yy, color='#367FA1', label='Within genus', s=32)
        ax.scatter(between, yy, color='#C96F3E', label='Between genera', s=32)
        ax.set(title=title, xlabel='Median distance')
        ax.grid(axis='x',color='#EEEEEE',lw=.6)
        ax.legend(frameon=False,fontsize=9)
    axes[0].set_yticks(yy, [f'{r.genus} (n={r.n_strains})' for r in summary.itertuples()])
    axes[0].invert_yaxis()
    return fig
