"""Run the authorized top-two sensitivity analysis into a fresh directory."""
from pathlib import Path
import hashlib
import json
import platform
import shutil
import numpy as np
import pandas as pd
import scipy
from neural_subspace import fit_pca, compare_subspaces, spectrum_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_analysis(repo_root, out=None):
    repo = Path(repo_root).resolve()
    canonical = repo / 'reports/exploration_bacteroides_neural_reliability_20261003/subspace'
    output = Path(out).resolve() if out is not None else canonical
    if (output / 'manifest.json').exists() or any((output / 'tables').glob('*.csv')):
        raise FileExistsError(f'Refusing to overwrite existing subspace results: {output}')
    source = repo / 'reports/exploration_bacteroides_local_model_20261003/neural/tables'
    names = ['neural_unit_profiles.csv', 'strain_metadata.csv', 'pre_gate_unit_profiles.csv']
    inputs = {name: source / name for name in names}
    x = pd.read_csv(inputs[names[0]], index_col='strain').sort_index()
    metadata = pd.read_csv(inputs[names[1]], index_col='strain', dtype={'dates': str})
    pre = pd.read_csv(inputs[names[2]], index_col='strain')
    assert x.index.is_unique and metadata.index.is_unique and pre.index.is_unique
    assert set(x.index) == set(metadata.index) == set(pre.index)
    assert set(x.columns) == set(pre.columns)
    metadata = metadata.loc[x.index]
    pre = pre.loc[x.index, x.columns]
    assert x.shape == pre.shape == (29, 13) and metadata.genus.eq('Bacteroides').all()
    assert metadata.species.nunique() == 16
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), np.ones(len(x)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(pre, axis=1), np.ones(len(pre)), atol=1e-12)
    full, prefit = fit_pca(x), fit_pca(pre)
    tests = [('leave_one_strain_out', f'strain:{s}', s, [s]) for s in x.index]
    tests += [('leave_one_recorded_species_out', f'species:{species}', species,
               metadata.index[metadata.species.eq(species)].tolist()) for species in sorted(metadata.species.unique())]
    rows, fold_ids, fits = [], [], {'full_gated': (full, x, []), 'full_pre_gate': (prefit, pre, [])}
    for scheme, fit_id, label, omitted in tests:
        training = x.drop(index=omitted)
        fit = fit_pca(training)
        fits[fit_id] = (fit, x, omitted)
        rows.append({'fit_id': fit_id, 'scheme': scheme, 'omitted_label': label,
                     'n_train': len(training), 'n_omitted': len(omitted), 'omitted_strains': ';'.join(omitted),
                     **compare_subspaces(full['loadings'], fit['loadings']), **spectrum_metrics(fit)})
        fold_ids.append({'fit_id': fit_id, 'scheme': scheme, 'train_ids': training.index.tolist(), 'omitted_ids': omitted})
    results = pd.DataFrame(rows)
    pre_comparison = {'comparison': 'full_pre_gate_vs_full_gated', 'n_strains': len(x),
                      **compare_subspaces(full['loadings'], prefit['loadings']), **spectrum_metrics(prefit)}
    for folder in ['tables', 'figures']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    if output != canonical:
        shutil.copyfile(canonical / 'protocol.md', output / 'protocol.md')
    t = output / 'tables'
    x.to_csv(t / 'gated_unit_profiles_29x13.csv')
    pre.to_csv(t / 'pre_gate_unit_profiles_29x13.csv')
    metadata.to_csv(t / 'strain_metadata_29.csv')
    pd.DataFrame(full['centered'], index=x.index, columns=x.columns).to_csv(t / 'full_centered_profiles_29x13.csv')
    components = [f'PC{i}' for i in range(1, 14)]
    mean_rows, loading_rows, spectrum_rows, score_rows = [], [], [], []
    for fit_id, (fit, values, omitted) in fits.items():
        mean_rows.append({'fit_id': fit_id, **dict(zip(x.columns, fit['mean']))})
        scores = (values.to_numpy() - fit['mean']) @ fit['loadings']
        for index, neuron in enumerate(x.columns):
            loading_rows.append({'fit_id': fit_id, 'neuron': neuron, **dict(zip(components, fit['loadings'][index]))})
        for index, component in enumerate(components):
            spectrum_rows.append({'fit_id': fit_id, 'component': component, 'singular_value': fit['singular_values'][index],
                                  'sample_eigenvalue': fit['variance'][index], 'evr': fit['ratio'][index],
                                  'cumulative_evr': fit['ratio'][:index+1].sum()})
        for index, strain in enumerate(x.index):
            score_rows.append({'fit_id': fit_id, 'strain': strain, 'role': 'omitted_projection' if strain in omitted else 'training',
                               **dict(zip(components, scores[index]))})
    pd.DataFrame(mean_rows).to_csv(t / 'all_fit_means.csv', index=False)
    pd.DataFrame(loading_rows).to_csv(t / 'all_fit_loadings.csv', index=False)
    pd.DataFrame(spectrum_rows).to_csv(t / 'all_fit_spectra.csv', index=False)
    pd.DataFrame(score_rows).to_csv(t / 'all_fit_strain_scores.csv', index=False)
    full_scores = pd.DataFrame(full['scores'], index=x.index, columns=components)
    full_scores.join(metadata).to_csv(t / 'full_gated_scores_with_metadata.csv')
    pd.DataFrame(full['loadings'], index=x.columns, columns=components).rename_axis('neuron').to_csv(t / 'full_gated_loadings.csv')
    pd.DataFrame(prefit['scores'], index=x.index, columns=components).to_csv(t / 'full_pre_gate_scores.csv')
    pd.DataFrame(prefit['loadings'], index=x.columns, columns=components).rename_axis('neuron').to_csv(t / 'full_pre_gate_loadings.csv')
    results.to_csv(t / 'deletion_subspace_metrics.csv', index=False)
    pd.DataFrame([pre_comparison]).to_csv(t / 'pre_gate_subspace_comparison.csv', index=False)
    summaries = []
    metrics = ['principal_angle_1_deg', 'max_principal_angle_deg', 'projector_frobenius_over_sqrt2',
               'pc1_absolute_cosine', 'pc1_acute_angle_deg', 'top2_evr', 'lambda2_minus_lambda3',
               'relative_gap_2_vs_3', 'lambda2_over_lambda3']
    for scheme, group in results.groupby('scheme', sort=False):
        for metric in metrics:
            values = group[metric]
            summaries.append({'scheme': scheme, 'metric': metric, 'n': len(values), 'minimum': values.min(),
                              'q25': values.quantile(.25), 'median': values.median(), 'q75': values.quantile(.75), 'maximum': values.max()})
    pd.DataFrame(summaries).to_csv(t / 'deletion_metric_summary.csv', index=False)
    worst_rows = []
    for scheme, group in results.groupby('scheme', sort=False):
        for metric in ['max_principal_angle_deg', 'pc1_acute_angle_deg']:
            row = group.loc[group[metric].idxmax()].to_dict()
            worst_rows.append({'selected_by': metric, **row})
    pd.DataFrame(worst_rows).to_csv(t / 'worst_deletion_cases.csv', index=False)
    (output / 'fold_ids.json').write_text(json.dumps(fold_ids, indent=2) + '\n')
    summary = {'dimensions': {'strains': len(x), 'neurons': len(x.columns), 'strain_deletions': 29, 'recorded_species_deletions': 16, 'all_fits_including_two_full': len(fits)},
               'full_gated_spectrum': spectrum_metrics(full), 'pre_gate_comparison': pre_comparison,
               'deletion_metric_summaries': summaries, 'worst_cases': worst_rows,
               'interpretation': 'Descriptive deletion/representation sensitivity against an overlapping full29 reference; not independent replication.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    code_files = [Path(__file__), Path(__file__).with_name('neural_subspace.py')]
    manifest = {'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name,path in inputs.items()},
                'protocol_sha256': sha256(output / 'protocol.md'), 'code_sha256': {p.name: sha256(p) for p in code_files},
                'parameters': {'k': 2, 'center': 'training mean', 'coordinate_sd_scaling': False,
                    'canonical_sign': 'largest-absolute loading positive; ties by source neuron order',
                    'projector_metric': 'Frobenius norm of projector difference divided by sqrt(2)',
                    'eigenvalue_ddof': 1, 'neuron_order': x.columns.tolist(), 'strain_order': x.index.tolist()},
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary))
    return output


if __name__ == '__main__':
    run_analysis(Path(__file__).resolve().parents[4])
