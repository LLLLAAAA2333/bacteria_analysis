"""Independent source recomputation plus API-contract checks; no neural input."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform, pdist
from local_chemical_axes import fit_axes, transform_axes


def verify(repo_root):
    repo = Path(repo_root).resolve()
    out = repo / 'reports/exploration_bacteroides_local_model_20261003/chemical'
    source = repo / 'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    x0 = pd.read_csv(source / 'fresh_chemical_log2.csv', index_col='strain')
    context = pd.read_csv(source / 'sample_context.csv', index_col='strain')
    metadata = pd.read_csv(source / 'fresh_feature_metadata.csv', index_col='metabolite')
    x = x0.loc[context.index[context.genus.eq('Bacteroides')]]
    a = x.to_numpy()
    mean = a.sum(axis=0) / len(a)
    sd = np.sqrt(((a - mean) ** 2).sum(axis=0) / (len(a) - 1))
    z = pd.DataFrame((a - mean) / sd, index=x.index, columns=x.columns)
    errors = {}
    def close(name, actual, expected):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        errors[name] = float(np.max(np.abs(actual - expected))) if actual.size else 0.
    t = out / 'tables'
    saved_x = pd.read_csv(t / 'chemical_log2_29x162.csv', index_col=0).loc[x.index, x.columns]
    close('all_raw_log2_values', saved_x, x)
    scales = pd.read_csv(t / 'local_feature_scales_metadata.csv', index_col=0).loc[x.columns]
    close('local_means', scales.local_mean_log2, mean)
    close('local_sample_sd', scales.local_sd_log2, sd)
    close('raw_ranges', scales.range_log2, a.max(axis=0)-a.min(axis=0))
    close('all_standardized_values', pd.read_csv(t / 'chemical_standardized_29x162.csv', index_col=0).loc[x.index, x.columns], z)
    members = pd.read_csv(t / 'local_module_members.csv')
    orders = json.loads((out / 'orders.json').read_text())
    scores = pd.DataFrame(index=x.index)
    for module, block in members.groupby('module'):
        ff = block.metabolite.tolist()
        families = metadata.loc[ff, 'family']
        assert len(ff) >= 3 and families.nunique() >= 3
        weights = np.array([1 / families.nunique() / (families == families[f]).sum() for f in ff])
        close(module+'_weights', block.score_weight, weights)
        scores[module] = pd.concat([z[families.index[families == family]].mean(axis=1) for family in families.unique()], axis=1).mean(axis=1)
        assert ff == orders['module_members'][module]
    close('all_axis_scores', pd.read_csv(t / 'local_axis_scores_29.csv', index_col=0).loc[x.index, scores.columns], scores)
    close('axes_centered', scores.mean(), np.zeros(len(scores.columns)))
    centered = a - mean
    corr = centered.T @ centered / np.sqrt(np.outer((centered**2).sum(axis=0), (centered**2).sum(axis=0)))
    distance = np.clip(1-corr, 0, 2); np.fill_diagonal(distance,0)
    tree = linkage(squareform(distance,checks=False),method='average',optimal_ordering=True)
    clusters = fcluster(tree,t=.5,criterion='distance')
    groups=[]
    for label in set(clusters):
        ff=x.columns[clusters == label].tolist()
        if len(ff)>=3 and metadata.loc[ff,'family'].nunique()>=3: groups.append(ff)
    groups.sort(key=lambda ff:(-len(ff),min(ff)))
    assert {f'L{i:02d}':ff for i,ff in enumerate(groups,1)} == orders['module_members']
    assert x.columns[leaves_list(tree)].tolist() == orders['feature_order']
    assert x.index[leaves_list(linkage(pdist(z),method='average',optimal_ordering=True))].tolist() == orders['strain_order']
    assert scores.columns[leaves_list(linkage(pdist(scores.T,metric='correlation'),method='average',optimal_ordering=True))].tolist() == orders['module_order']
    pairs=pd.read_csv(t/'local_module_pair_correlations.csv');loc={f:i for i,f in enumerate(x.columns)}
    close('all_member_pair_correlations',pairs.pearson_r,[corr[loc[f],loc[g]] for f,g in zip(pairs.feature_1,pairs.feature_2)])
    ungrouped = set(pd.read_csv(t/'ungrouped_features.csv').metabolite)
    assert not ungrouped & set(members.metabolite)
    assert ungrouped | set(members.metabolite) == set(x.columns)
    assert len(members)==128 and len(ungrouped)==34 and len(scores.columns)==12
    saved_context=pd.read_csv(t/'bacteroides_context_29.csv',index_col=0).loc[x.index]
    pd.testing.assert_frame_equal(saved_context[context.columns],context.loc[x.index],check_dtype=False)
    assert saved_context.taxonomy_flag.equals(context.loc[x.index,'taxonomy_note'].fillna('').astype(str).str.strip().ne('').rename('taxonomy_flag'))
    # Public API trained on actual 28 rows; validation row never enters discovery.
    before_x=x.copy(deep=True);before_meta=metadata.copy(deep=True)
    train=x.iloc[:-1];heldout=x.iloc[-1:]
    fit=fit_axes(train,metadata)
    close('fold_training_mean',fit['means'],train.to_numpy().mean(axis=0))
    close('fold_training_sd',fit['scales'],train.to_numpy().std(axis=0,ddof=1))
    close('transform_train_identity',transform_axes(train,fit),fit['train_scores'])
    predicted=transform_axes(heldout,fit)
    manual=pd.DataFrame(index=heldout.index)
    for m,ff in fit['module_members'].items():
        manual[m]=((heldout[ff]-fit['means'][ff])/fit['scales'][ff]) @ fit['score_weights'][m]
    close('heldout_fixed_training_parameters',predicted,manual)
    close('column_permutation_transform',transform_axes(heldout[heldout.columns[::-1]],fit),predicted)
    reordered=transform_axes(train.iloc[::-1],fit)
    close('row_permutation_transform',reordered,fit['train_scores'].iloc[::-1])
    old_means=fit['means'].copy();old_scales=fit['scales'].copy();old_scores=fit['train_scores'].copy()
    transform_axes(heldout+100,fit)
    pd.testing.assert_series_equal(fit['means'],old_means);pd.testing.assert_series_equal(fit['scales'],old_scales)
    pd.testing.assert_frame_equal(fit['train_scores'],old_scores)
    pd.testing.assert_frame_equal(x,before_x);pd.testing.assert_frame_equal(metadata,before_meta)
    synthetic=pd.DataFrame({'a':[0.,1.,2.,3.],'b':[0.,2.,4.,6.],'c':[0.,3.,6.,9.],'constant':[1.,1.,1.,1.]},index=['s1','s2','s3','s4'])
    synthetic_meta=pd.DataFrame({'family':['f1','f2','f3','f4']},index=synthetic.columns)
    sf=fit_axes(synthetic.iloc[:3],synthetic_meta)
    assert sf['excluded_features']==['constant'] and list(sf['module_members'])==['L01']
    close('synthetic_heldout_training_z',transform_axes(synthetic.iloc[3:],sf),[[2.]])
    no_axes=fit_axes(synthetic[['a','constant']],synthetic_meta)
    assert no_axes['train_scores'].shape==(4,0) and transform_axes(synthetic[['a','constant']],no_axes).shape==(4,0)
    constants_only=fit_axes(synthetic[['constant']],synthetic_meta)
    assert constants_only['retained_features']==[] and constants_only['train_scores'].shape==(4,0)
    same_family=synthetic_meta.copy();same_family['family']='same'
    assert fit_axes(synthetic,same_family)['train_scores'].shape==(4,0)
    try: transform_axes(heldout.drop(columns=x.columns[0]),fit)
    except ValueError: pass
    else: raise AssertionError('Missing input column was not rejected')
    manifest=json.loads((out/'manifest.json').read_text())
    for item in manifest['inputs'].values(): assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    assert hashlib.sha256((out/'protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    for name,h in manifest['code_sha256'].items(): assert hashlib.sha256((out/'code'/name).read_bytes()).hexdigest()==h
    report={'status':'PASS','max_abs_errors':errors,'coverage':{'strains':29,'annotations':162,'axes':12,'grouped':128,'ungrouped':34,'taxonomy_flags':6},
            'checks':['raw/scaled values and raw ranges','exact module partition and IDs','equal-family weights','all scores and member correlations','all feature coverage','three chemical-only orders','species/dates/taxonomy flags preserved','28-strain fit parameters','heldout transform freezing','train transform equality','row/column permutation','input/fitted-object immutability','constants and empty-axis shape','minimum family restriction','missing column error','source/protocol/code hashes'],
            'scope':'Numerical/API checks only, not neural model evaluation or biological validation.'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','largest_abs_error':max(errors.values()),'checks':len(report['checks'])}))
    return report


if __name__=='__main__':
    verify(Path(__file__).resolve().parents[4])
