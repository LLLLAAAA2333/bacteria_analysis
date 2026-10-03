"""Independently recompute exported chemical results from source files.

Does not import the analysis module. Source data are never changed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform


def verify(repo_root):
    repo = Path(repo_root)
    out = repo / 'reports/exploration_genus_patterns_independent_20261003/chemical'
    source = repo / 'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    tables = out / 'tables'
    x0 = pd.read_csv(source / 'fresh_chemical_log2.csv', index_col='strain')
    c0 = pd.read_csv(source / 'sample_context.csv', index_col='strain')
    meta = pd.read_csv(source / 'fresh_feature_metadata.csv', index_col='metabolite')
    ids = x0.index[c0.loc[x0.index, 'genus'].map(c0.genus.value_counts()) >= 2]
    x = x0.loc[ids]
    genus = c0.loc[ids, 'genus']
    names = sorted(genus.unique())
    center = pd.DataFrame([x.loc[genus == g].to_numpy().mean(axis=0) for g in names], index=names, columns=x.columns)
    ref = center.to_numpy().mean(axis=0)
    sd = np.sqrt(((center.to_numpy() - ref) ** 2).sum(axis=0) / (len(names) - 1))
    z = pd.DataFrame((x.to_numpy() - ref) / sd, index=x.index, columns=x.columns)
    cz = pd.DataFrame((center.to_numpy() - ref) / sd, index=names, columns=x.columns)
    checks = {}

    def close(name, actual, expected):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-10, equal_nan=True)
        checks[name] = float(np.nanmax(np.abs(a - e)))

    close('original_strain_log2', pd.read_csv(tables / 'strain_log2_90x162.csv', index_col=0).loc[ids, x.columns], x)
    close('genus_log2_centers', pd.read_csv(tables / 'genus_mean_log2_13x162.csv', index_col=0).loc[names, x.columns], center)
    close('genus_log2_differences', pd.read_csv(tables / 'feature_genus_log2_difference_162x13.csv', index_col=0).loc[x.columns, names].T, center - ref)
    close('strain_standardized', pd.read_csv(tables / 'strain_standardized_90x162.csv', index_col=0).loc[ids, x.columns], z)
    close('genus_standardized', pd.read_csv(tables / 'feature_genus_standardized_162x13.csv', index_col=0).loc[x.columns, names].T, cz)
    info = pd.read_csv(tables / 'feature_reference_scale_metadata.csv', index_col=0).loc[x.columns]
    close('reference', info.equal_genus_reference_log2, ref)
    close('scale', info.between_genus_sd_log2, sd)
    close('equal_genus_center_is_zero', cz.mean(), np.zeros(len(x.columns)))
    assert (sd > 1e-12).all()
    orders = json.loads((out / 'orders.json').read_text())
    members = pd.read_csv(tables / 'module_members.csv')
    new_scores = {}
    for module, block in members.groupby('module'):
        features = block.metabolite.tolist()
        families = meta.loc[features, 'family']
        assert len(features) >= 3 and families.nunique() >= 3
        family_scores = [z[families.index[families == family]].mean(axis=1) for family in sorted(families.unique())]
        new_scores[module] = np.mean(family_scores, axis=0)
        weights = np.array([1 / families.nunique() / (families == families[f]).sum() for f in features])
        close(f'{module}_weights', block.score_weight, weights)
        assert set(features) == set(orders['modules'][module])
    score = pd.DataFrame(new_scores, index=ids)
    mc = score.groupby(genus).mean()
    close('all_strain_module_scores', pd.read_csv(tables / 'strain_module_scores.csv', index_col=0).loc[ids, score.columns], score)
    close('all_genus_module_centers', pd.read_csv(tables / 'genus_module_centers.csv', index_col=0).loc[names, score.columns], mc)
    # Pearson distances recomputed by explicitly centered dot products.
    profile = center.to_numpy() - ref
    lengths = np.sqrt((profile ** 2).sum(axis=0))
    corr = profile.T @ profile / np.outer(lengths, lengths)
    dist = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(dist, 0)
    tree = linkage(squareform(dist, checks=False), method='average', optimal_ordering=True)
    cl = fcluster(tree, t=.5, criterion='distance')
    valid_groups = []
    for label in set(cl):
        feats = x.columns[cl == label].tolist()
        if len(feats) >= 3 and meta.loc[feats, 'family'].nunique() >= 3:
            valid_groups.append(frozenset(feats))
    assert set(valid_groups) == {frozenset(v) for v in orders['modules'].values()}
    assert len(members) == 139 and len(set(orders['ungrouped'])) == 23
    assert set(members.metabolite) | set(orders['ungrouped']) == set(x.columns)
    assert not set(members.metabolite) & set(orders['ungrouped'])
    assert center.index[leaves_list(linkage(pdist(cz), method='average', optimal_ordering=True))].tolist() == orders['genus_order']
    assert mc.columns[leaves_list(linkage(pdist(mc.T, metric='correlation'), method='average', optimal_ordering=True))].tolist() == orders['module_order']
    assert x.columns[leaves_list(tree)].tolist() == orders['feature_order']

    # Explicitly delete each strain, recompute all genus centers and reference.
    loo = pd.read_csv(tables / 'module_leave_one_strain_out.csv').set_index(['omitted_strain', 'module'])
    loo_error = []
    for strain in ids:
        g = genus[strain]
        changed = x.drop(index=strain).groupby(genus.drop(index=strain)).mean()
        target_z = (changed.loc[g] - changed.mean()) / sd
        for m, block in members.groupby('module'):
            features = block.metabolite.tolist()
            family = meta.loc[features, 'family']
            val = np.mean([target_z[family.index[family == f]].mean() for f in family.unique()])
            loo_error.append(val - loo.loc[(strain, m), 'loo_center'])
    close('all_1350_explicit_leave_one_out_centers', loo_error, np.zeros(len(loo_error)))
    consistency = pd.read_csv(tables / 'module_direction_consistency.csv')
    for row in consistency.itertuples():
        values = score.loc[genus == row.genus, row.module]
        sign = np.sign(mc.loc[row.genus, row.module])
        frac = ((values * sign) > 1e-12).mean()
        assert np.isclose(frac, row.strain_direction_fraction)
        assert int(((values * sign) > 1e-12).sum()) == row.n_strains_same_direction
    # np.cov(aweights=...) independently recomputes equal-genus correlations.
    weights = np.array([1 / (13 * (genus == g).sum()) for g in genus])
    residual = x.to_numpy() - center.loc[genus].to_numpy()
    def weighted_r(a):
        cov = np.cov(a, rowvar=False, aweights=weights, ddof=0)
        return cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    individual, within = weighted_r(x.to_numpy()), weighted_r(residual)
    pairs = pd.read_csv(tables / 'module_pair_correlations.csv')
    idx = {f: i for i, f in enumerate(x.columns)}
    for key, expected in [('between_genus_r', corr), ('individual_equal_genus_r', individual), ('within_genus_equal_genus_r', within)]:
        close(key, pairs[key], [expected[idx[a], idx[b]] for a, b in zip(pairs.feature_1, pairs.feature_2)])
    manifest = json.loads((out / 'manifest.json').read_text())
    for item in manifest['inputs'].values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    assert hashlib.sha256((out / 'protocol.md').read_bytes()).hexdigest() == manifest['protocol_sha256']
    assert hashlib.sha256((out / 'code/analyze_chemical_patterns.py').read_bytes()).hexdigest() == manifest['code_sha256']
    report = {'status': 'PASS', 'dimensions': {'strains': len(x), 'genera': len(names), 'features': x.shape[1], 'modules': len(new_scores), 'assigned_annotations': len(members), 'ungrouped_annotations': 23}, 'max_abs_errors': checks,
              'discrete_checks': ['input IDs unique/aligned', 'eligible module partitions reproduced', 'exact family weights', 'all feature coverage', 'feature/genus/module orders', 'all strain-direction counts', '1350 explicit deletions', 'source/protocol/code hashes'],
              'scope': 'Numerical and export checks only; does not independently validate biological findings.'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'largest_abs_error': max(checks.values()), 'discrete_checks': len(report['discrete_checks'])}))
    return report


if __name__ == '__main__':
    verify(Path(__file__).resolve().parents[4])
