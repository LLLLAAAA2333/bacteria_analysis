"""Save full29 chemical-only descriptions; invoke fit_axes anew inside each CV fold."""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from local_chemical_axes import fit_axes, transform_axes


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_index(frame, metric):
    if len(frame) < 2:
        return frame.index.tolist()
    tree = linkage(pdist(frame.to_numpy(), metric=metric), method='average', optimal_ordering=True)
    return frame.index[leaves_list(tree)].tolist()


def run_analysis(repo_root, out=None):
    """Write full29 descriptive outputs to a new directory; refuse overwrites."""
    repo = Path(repo_root).resolve()
    canonical = repo / 'reports/exploration_bacteroides_local_model_20261003/chemical'
    output = Path(out).resolve() if out is not None else canonical
    if (output / 'manifest.json').exists() or any((output / 'tables').glob('*.csv')):
        raise FileExistsError(f'Refusing to overwrite existing local chemical results: {output}')
    source = repo / 'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    names = ['fresh_chemical_log2.csv', 'fresh_feature_metadata.csv', 'sample_context.csv']
    inputs = {name: source / name for name in names}
    all_log = pd.read_csv(inputs[names[0]], index_col='strain')
    metadata = pd.read_csv(inputs[names[1]], index_col='metabolite')
    context = pd.read_csv(inputs[names[2]], index_col='strain')
    assert all_log.index.is_unique and context.index.is_unique and metadata.index.is_unique
    assert set(all_log.index) == set(context.index)
    assert set(all_log.columns) == set(metadata.index)
    context = context.loc[all_log.index]
    context = context.loc[context.genus.eq('Bacteroides')].copy()
    context['taxonomy_flag'] = context.taxonomy_note.fillna('').astype(str).str.strip().ne('')
    x = all_log.loc[context.index]
    metadata = metadata.loc[x.columns]
    metadata.index.name = "metabolite"
    assert x.shape == (29, 162)
    fitted = fit_axes(x, metadata)
    z = (x[fitted['retained_features']] - fitted['means']) / fitted['scales']
    z = z[fitted['retained_features']]
    scores = fitted['train_scores']
    for folder in ['tables', 'figures']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    if output != canonical:
        shutil.copyfile(canonical / 'protocol.md', output / 'protocol.md')
    t = output / 'tables'
    context.to_csv(t / 'bacteroides_context_29.csv')
    x.to_csv(t / 'chemical_log2_29x162.csv')
    metadata.to_csv(t / 'feature_metadata_162.csv')
    z.to_csv(t / 'chemical_standardized_29x162.csv')
    info = pd.DataFrame({'local_mean_log2': fitted['means'], 'local_sd_log2': fitted['scales'],
                         'minimum_log2': x.min(), 'q25_log2': x.quantile(.25), 'median_log2': x.median(),
                         'q75_log2': x.quantile(.75), 'maximum_log2': x.max(), 'range_log2': x.max() - x.min(),
                         'included_standardized': x.columns.isin(fitted['retained_features'])})
    info.index.name = 'metabolite'
    info.join(metadata).to_csv(t / 'local_feature_scales_metadata.csv')
    scores.to_csv(t / 'local_axis_scores_29.csv')
    scores.join(context).to_csv(t / 'local_axis_scores_with_context.csv')
    excluded = fitted['excluded_features']
    metadata.loc[excluded].to_csv(t / 'constant_features.csv')
    grouped = {f for members in fitted['module_members'].values() for f in members}
    ungrouped = [f for f in x.columns if f not in grouped]
    metadata.loc[ungrouped].to_csv(t / 'ungrouped_features.csv')
    member_rows, representative_rows, pair_rows, summaries = [], [], [], []
    for module, features in fitted['module_members'].items():
        families = metadata.loc[features, 'family']
        corr = z[features].corr()
        for feature in features:
            member_rows.append({'module': module, 'metabolite': feature, 'family': metadata.loc[feature, 'family'],
                                'score_weight': fitted['score_weights'][module][feature],
                                'local_mean_log2': info.loc[feature, 'local_mean_log2'],
                                'local_sd_log2': info.loc[feature, 'local_sd_log2'],
                                'range_log2': info.loc[feature, 'range_log2'],
                                'SuperClass': metadata.loc[feature, 'SuperClass'], 'Class': metadata.loc[feature, 'Class'],
                                'SubClass': metadata.loc[feature, 'SubClass']})
        reps = sorted([(f, z[f].corr(scores[module])) for f in features], key=lambda v: (-v[1], v[0]))
        for rank, (feature, r) in enumerate(reps, 1):
            representative_rows.append({'module': module, 'rank': rank, 'metabolite': feature,
                                        'member_axis_correlation': r, 'representative_top3': rank <= 3})
        local_pairs = []
        for i, a in enumerate(features):
            for b in features[i + 1:]:
                record = {'module': module, 'feature_1': a, 'feature_2': b,
                          'cross_family': families[a] != families[b], 'pearson_r': corr.loc[a, b]}
                pair_rows.append(record); local_pairs.append(record)
        for scope in ['all_pairs', 'cross_family_pairs']:
            values = np.array([row['pearson_r'] for row in local_pairs if scope == 'all_pairs' or row['cross_family']])
            summaries.append({'module': module, 'n_annotations': len(features), 'n_families': families.nunique(),
                              'pair_scope': scope, 'n_pairs': len(values), 'r_min': values.min(), 'r_q25': np.quantile(values, .25),
                              'r_median': np.median(values), 'r_q75': np.quantile(values, .75), 'r_max': values.max(),
                              'fraction_positive': np.mean(values > 0),
                              'member_raw_log2_sd_median': info.loc[features, 'local_sd_log2'].median(),
                              'member_raw_log2_range_median': info.loc[features, 'range_log2'].median(),
                              'axis_sd': scores[module].std(ddof=1), 'axis_min': scores[module].min(), 'axis_max': scores[module].max()})
    members = pd.DataFrame(member_rows, columns=['module', 'metabolite', 'family', 'score_weight', 'local_mean_log2', 'local_sd_log2', 'range_log2', 'SuperClass', 'Class', 'SubClass'])
    members.to_csv(t / 'local_module_members.csv', index=False)
    pd.DataFrame(representative_rows, columns=['module', 'rank', 'metabolite', 'member_axis_correlation', 'representative_top3']).to_csv(t / 'local_module_representatives.csv', index=False)
    pd.DataFrame(pair_rows, columns=['module', 'feature_1', 'feature_2', 'cross_family', 'pearson_r']).to_csv(t / 'local_module_pair_correlations.csv', index=False)
    pd.DataFrame(summaries).to_csv(t / 'local_module_summary.csv', index=False)
    members.groupby(['module', 'SuperClass'], dropna=False).agg(n_annotations=('metabolite', 'size'), score_weight=('score_weight', 'sum')).reset_index().to_csv(t / 'local_module_annotation_composition.csv', index=False)
    strain_order = ordered_index(z, 'euclidean') if z.shape[1] else x.index.tolist()
    module_order = ordered_index(scores.T, 'correlation') if scores.shape[1] else []
    display = scores.loc[strain_order, module_order].T
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(13.5, max(4, .4 * len(module_order) + 1.8)))
    vlim = float(np.ceil(np.abs(display.to_numpy()).max() * 2) / 2) if display.size else 1
    if display.size:
        im = ax.imshow(display, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='nearest')
        ax.set_xticks(range(len(strain_order)), strain_order, rotation=90, fontsize=9)
        labels = [f'{m} ({len(fitted["module_members"][m])} annotations)' for m in module_order]
        ax.set_yticks(range(len(module_order)), labels, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=.021, pad=.022, label='Family-weighted standardized score')
        ax.set_xlabel('Bacteroides strain')
    else:
        ax.text(.5, .5, 'No eligible chemical axes under the fixed rule', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title('Local chemical combinations within Bacteroides', loc='left', fontsize=14, pad=12)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(output / 'figures/01_local_chemical_axes.png', dpi=180)
    fig.savefig(output / 'figures/01_local_chemical_axes.svg')
    plt.close(fig)
    (output / 'figures/01_local_chemical_axes_caption.txt').write_text(
        'All eligible local chemical axes and all 29 Bacteroides strains. Modules are newly defined only within these 29 chemical profiles using average-linkage Pearson distance and a fixed 0.5 cut; at least three annotations and three Mass-column families per module. '
        'Each annotation uses the local training mean and sample SD; scores equal-weight families, then annotations within families. Colors are dimensionless, not concentration or log-fold change. '
        'Both row and column order use chemistry only. Full29 groups are descriptive and must not be reused in held-out model evaluation: each training fold must independently refit the axes. All ungrouped annotations remain in the exported full162 tables.\n')
    orders = {'strain_order': strain_order, 'module_order': module_order, 'feature_order': fitted['feature_order'],
              'module_members': fitted['module_members'], 'ungrouped_features': ungrouped, 'constant_features': excluded}
    (output / 'orders.json').write_text(json.dumps(orders, indent=2) + '\n')
    api_file = Path(__file__).with_name('local_chemical_axes.py')
    manifest = {'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in inputs.items()},
                'protocol_sha256': sha256(output / 'protocol.md'), 'code_sha256': {api_file.name: sha256(api_file), Path(__file__).name: sha256(__file__)},
                'parameters': fitted['parameters'], 'dimensions': {'strains': len(x), 'annotations': len(x.columns),
                    'retained_standardized': len(fitted['retained_features']), 'axes': len(fitted['module_members']),
                    'grouped_annotations': len(grouped), 'ungrouped_annotations': len(ungrouped), 'taxonomy_flags': int(context.taxonomy_flag.sum())},
                'figure_symmetric_limit': vlim, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
                'isolation': 'Only listed chemical/context files read; neural analysis, CV outcome and previous modules not read.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['dimensions']))
    return output


if __name__ == '__main__':
    run_analysis(Path(__file__).resolve().parents[4])
