"""Independent audit of saved results; no scientific output/model files modified.

Rebuilds all 46 fits from original inputs using independent NumPy least squares
and SciPy correlation-distance clustering. Never imports the analysis code.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model'
TABLES = MODEL / 'tables'
PRIMARY = 'primary_unit_adf_minus_ash'
ERRORS = {}
CHECKS = []


def close(label, actual, saved, atol=1e-10):
    actual, saved = np.asarray(actual, dtype=float), np.asarray(saved, dtype=float)
    assert actual.shape == saved.shape, (label, actual.shape, saved.shape)
    np.testing.assert_allclose(actual, saved, rtol=1e-10, atol=atol, err_msg=label)
    error = float(np.nanmax(np.abs(actual - saved), initial=0))
    ERRORS[label] = max(ERRORS.get(label, 0), error)


def read_table(name, index=None):
    return pd.read_csv(TABLES / (name + '.csv'), index_col=index)


def independent_ols(x, y):
    x, y = np.asarray(x), np.asarray(y)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    sse, sst = ((y - predicted) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    r = np.corrcoef(x, y)[0, 1]
    return dict(intercept=intercept, slope=slope, training_mean=y.mean(),
                training_predictor_mean=x.mean(), pearson_r=r, r_squared=r*r,
                sse_model=sse, sse_mean=sst, rmse_model=np.sqrt(sse / len(y)),
                rmse_mean=np.sqrt(sst / len(y)))


def audit():
    manifest = json.loads((MODEL / 'manifest.json').read_text())
    for record in list(manifest['inputs'].values()) + [manifest['frozen_axes_api'], manifest['root_protocol']]:
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
    assert hashlib.sha256((MODEL/'protocol_snapshot.md').read_bytes()).hexdigest() == manifest['protocol_snapshot_sha256']
    assert hashlib.sha256((MODEL/'code/fit_fixed_contrast.py').read_bytes()).hexdigest() == manifest['model_code_sha256']
    CHECKS.append('All six original input hashes, frozen axes API, protocol snapshot, and model code match manifest')
    paths = {k: v['path'] for k, v in manifest['inputs'].items()}
    context = pd.read_csv(paths['sample_context.csv'], index_col='strain', dtype={'dates': str})
    ids = sorted(context.index[context.genus.eq('Bacteroides')])
    assert len(ids) == 29 and context.loc[ids, 'species'].nunique() == 16
    x = pd.read_csv(paths['fresh_chemical_log2.csv'], index_col='strain').loc[ids]
    metadata = pd.read_csv(paths['fresh_feature_metadata.csv'], index_col='metabolite').loc[x.columns]
    unit = pd.read_csv(paths['neural_unit_coefficients.csv'], index_col='strain').loc[ids]
    raw = pd.read_csv(paths['strain_coefficients.csv'], index_col='strain').loc[ids, unit.columns]
    pre = pd.read_csv(paths['neural_pre_gate_unit_coefficients.csv'], index_col='strain').loc[ids, unit.columns]
    assert x.shape == (29, 162) and raw.shape == unit.shape == pre.shape == (29, 13)
    assert all(frame.index.is_unique and frame.columns.is_unique for frame in [x, metadata, raw, unit, pre])
    norm = np.sqrt((raw.to_numpy()**2).sum(axis=1))
    close('source_raw_to_unit_identity', raw.to_numpy()/norm[:, None], unit)
    y = pd.DataFrame({PRIMARY: unit.ADF-unit.ASH, 'raw_adf_minus_ash': raw.ADF-raw.ASH,
                      'pre_gate_unit_adf_minus_ash': pre.ADF-pre.ASH}, index=ids)
    for name in unit: y['unit_'+name] = unit[name]
    cohort = read_table('cohort_targets', 'strain')
    assert cohort.index.tolist() == ids == manifest['strain_order']
    assert manifest['target_order'] == y.columns.tolist()
    close('cohort_targets', y, cohort[y.columns])
    close('cohort_norm', norm, cohort.coefficient_l2_norm)
    close('contrast_normalization_identity', y.raw_adf_minus_ash/norm, y[PRIMARY])
    assert int(cohort.taxonomy_flag.sum()) == 6
    for c in ['genus', 'species', 'taxonomy_note', 'dates']:
        assert cohort[c].fillna('').astype(str).tolist() == context.loc[ids, c].fillna('').astype(str).tolist()
    for name, original in [('chemical_log2_29x162', x), ('unit_profiles_29x13', unit),
                           ('raw_coefficients_29x13', raw), ('pre_gate_unit_profiles_29x13', pre)]:
        saved = read_table(name, 'strain')
        assert saved.index.equals(original.index) and saved.columns.equals(original.columns)
        close('export_'+name, original, saved)
    CHECKS.append('All29 cohort, 16 species, six flags, original metadata, input matrices and raw/unit contrast identity')

    expected_folds = [('full_cohort', 'full_apparent', [])]
    expected_folds += [('strain_'+s, 'leave_one_strain_out', [s]) for s in ids]
    expected_folds += [('species_%02d'%i, 'leave_one_recorded_species_out',
                       context.loc[ids].index[context.loc[ids, 'species'].eq(species)].tolist())
                      for i, species in enumerate(sorted(context.loc[ids, 'species'].unique()), 1)]
    fold_manifest = json.loads((MODEL/'fold_manifest.json').read_text())
    assert [(f['fold_id'],f['scheme'],f['omitted_ids']) for f in fold_manifest] == expected_folds[1:]
    all_errors, all_fit_rows, selected_sets = [], [], {}
    heldout = read_table('heldout_predictions').set_index(['fold_id','strain'])
    full_output = read_table('selected_full_state_scores', 'strain')
    for fold, scheme, omitted in expected_folds:
        train = [s for s in ids if s not in omitted]
        test = omitted or ids
        saved = json.loads((MODEL/'parameters'/f'{fold}.json').read_text())
        assert saved['train_ids'] == train and saved['omitted_ids'] == omitted
        assert saved['scheme'] == scheme and saved['source_features'] == x.columns.tolist()
        assert not (set(train) & set(omitted))
        a = x.loc[train].to_numpy()
        means, sds = a.mean(axis=0), a.std(axis=0, ddof=1)
        close('all46_training_means', means, [saved['training_log2_means'][c] for c in x])
        close('all46_training_sds', sds, [saved['training_log2_sample_sds'][c] for c in x])
        retained = np.flatnonzero(sds > 1e-12)
        assert saved['retained_features'] == x.columns[retained].tolist()
        assert saved['excluded_features'] == x.columns[sds <= 1e-12].tolist()
        z = (a[:, retained]-means[retained])/sds[retained]
        tree = linkage(np.clip(pdist(z.T, metric='correlation'), 0, 2), method='average', optimal_ordering=True)
        labels = fcluster(tree, 0.5, criterion='distance')
        groups = []
        for label in sorted(set(labels)):
            members = x.columns[retained[labels == label]].tolist()
            if len(members) >= 3 and metadata.loc[members, 'family'].nunique() >= 3: groups.append(members)
        groups.sort(key=lambda m: (-len(m), min(m)))
        modules = {'L%02d'%i: members for i,members in enumerate(groups, 1)}
        assert modules == saved['module_members'], fold
        assert x.columns[retained[leaves_list(tree)]].tolist() == saved['feature_order']
        scores, candidates, weights = {}, {}, {}
        for state, members in modules.items():
            counts = metadata.loc[members, 'family'].value_counts()
            weights[state] = np.array([1/(len(counts)*counts[f]) for f in metadata.loc[members,'family']])
            close('all46_weights', weights[state], [saved['score_weights'][state][m] for m in members])
            ix = x.columns.get_indexer(members)
            scores[state] = ((a[:, ix]-means[ix])/sds[ix]) @ weights[state]
            candidates[state] = independent_ols(scores[state], y.loc[train, PRIMARY].to_numpy())
        selected = min(modules, key=lambda s: (-candidates[s]['r_squared'], tuple(sorted(modules[s]))))
        assert selected == saved['selected_state_id'] and not saved['no_state_fallback']
        assert modules[selected] == saved['selected_members']
        selected_sets[fold] = set(modules[selected])
        for c in saved['candidates']:
            state = c['state_id']; calc = candidates[state]
            assert json.loads(c['members_json']) == modules[state]
            assert c['n_annotations'] == len(modules[state])
            assert c['n_families'] == metadata.loc[modules[state],'family'].nunique()
            for key in ['pearson_r','r_squared','intercept','slope','sse_model','sse_mean']:
                close('all46_candidate_'+key, calc[key], c['primary_'+key])
        train_scores = pd.DataFrame(scores, index=train)
        recorded_scores = pd.DataFrame.from_dict(saved['training_all_state_scores'], orient='index').loc[train, train_scores.columns]
        close('all46_all_candidate_train_scores', train_scores, recorded_scores)
        ix = x.columns.get_indexer(modules[selected])
        test_score = ((x.loc[test].to_numpy()[:, ix]-means[ix])/sds[ix]) @ weights[selected]
        output = full_output.loc[test] if not omitted else heldout.loc[fold].loc[test]
        close('all46_selected_transformed_scores', test_score, output.chemical_state_score)
        predictions = {}
        for target in y:
            fit = independent_ols(scores[selected], y.loc[train,target].to_numpy())
            for k,v in fit.items(): close('all46_secondary_and_primary_ols_'+k, v, saved['response_models'][target][k])
            prediction = fit['intercept']+fit['slope']*test_score
            predictions[target] = prediction
            close('all46_predictions', prediction, output['pred_'+target])
            close('all46_training_mean_baselines', np.repeat(y.loc[train,target].mean(),len(test)), output['baseline_'+target])
            for s,pred in zip(test,prediction):
                obs=float(y.loc[s,target]);base=float(y.loc[train,target].mean())
                all_errors.append(dict(scheme=scheme,fold_id=fold,strain=s,target=target,observed=obs,predicted=float(pred),
                    training_mean_baseline=base,error_model=float(pred-obs),error_training_mean_baseline=base-obs,
                    squared_error_model=float((pred-obs)**2),squared_error_training_mean_baseline=(base-obs)**2))
        close('all46_prediction_ADF_minus_ASH_identity', predictions['unit_ADF']-predictions['unit_ASH'], predictions[PRIMARY])
        all_fit_rows.append(dict(fold_id=fold,scheme=scheme,n_train=len(train),n_test=len(test),selected_state_id=selected,
                                 n_states=len(modules),n_selected_members=len(modules[selected])))
    full_parameters=json.loads((MODEL/'parameters/full_cohort.json').read_text())
    expected_candidates=pd.DataFrame(full_parameters['candidates'])
    pd.testing.assert_frame_equal(expected_candidates,read_table('full_candidate_results'),check_dtype=False,check_exact=False,atol=1e-12)
    exported_models=read_table('full_selected_response_models').set_index('target')
    for target,fit in full_parameters['response_models'].items():
        cols=[k for k in fit if not isinstance(fit[k],bool)]
        close('full_response_model_table',[fit[k] for k in cols],exported_models.loc[target,cols].to_numpy())
    expected_scores=pd.DataFrame.from_dict(full_parameters['training_all_state_scores'],orient='index').loc[ids]
    close('full_candidate_scores_export',expected_scores,read_table('full_candidate_scores','strain')[expected_scores.columns])
    CHECKS.append('Independent reconstruction of full fit + all45 fold scales, correlations, clusters, weights, candidates, primary-only selection, all16 response fits, heldout transforms and baselines; full candidate/model exports consistent')
    errors = pd.DataFrame(all_errors)
    for name, expected in [('full_fit_target_errors', errors[errors.scheme.eq('full_apparent')]),
                            ('heldout_target_errors', errors[~errors.scheme.eq('full_apparent')])]:
        keys=['scheme','fold_id','strain','target']; cols=[c for c in expected if c not in keys]
        e=expected.set_index(keys).sort_index();s=read_table(name).set_index(keys).sort_index()
        assert e.index.equals(s.index);close(name,e[cols],s[cols])
    unit_targets=['unit_'+c for c in unit]
    vectors=errors[errors.target.isin(unit_targets)].groupby(['scheme','fold_id','strain'])[['squared_error_model','squared_error_training_mean_baseline']].sum().reset_index()
    for name, expected in [('full_fit_vector_errors',vectors[vectors.scheme.eq('full_apparent')]),('heldout_vector_errors',vectors[~vectors.scheme.eq('full_apparent')])]:
        keys=['scheme','fold_id','strain'];cols=['squared_error_model','squared_error_training_mean_baseline']
        e=expected.set_index(keys).sort_index();s=read_table(name).set_index(keys).sort_index()
        assert e.index.equals(s.index);close(name,e[cols],s[cols])
    vectors['target']='unit_profile_13d'
    for filename in ['full_fit_performance','pooled_performance','fold_performance']:
        saved=read_table(filename)
        for _,row in saved.iterrows():
            pool=vectors if row.target=='unit_profile_13d' else errors
            g=pool[pool.scheme.eq(row.scheme)&pool.target.eq(row.target)]
            if 'fold_id' in saved: g=g[g.fold_id.eq(row.fold_id)]
            n=len(g);dim=13 if row.target=='unit_profile_13d' else 1
            sm=g.squared_error_model.sum();sb=g.squared_error_training_mean_baseline.sum()
            calc=[n,dim,sm,sb,np.sqrt(sm/n),np.sqrt(sb/n),np.sqrt(sm/(n*dim)),np.sqrt(sb/(n*dim)),1-sm/sb,
                  int((g.squared_error_model<g.squared_error_training_mean_baseline).sum())]
            cols=['n_predictions','n_coordinates','sse_model','sse_training_mean_baseline','rmse_model','rmse_training_mean_baseline',
                  'rmse_per_coordinate_model','rmse_per_coordinate_training_mean_baseline','error_improvement','n_strains_better_than_mean']
            close(filename,calc,row[cols].to_numpy())
    selections=read_table('fold_selection_stability')
    for _,r in selections.iterrows():
        a,b=selected_sets[r.fold_id],selected_sets['full_cohort']
        assert set(json.loads(r.members_json))==a
        close('selection_jaccard_to_full',len(a&b)/len(a|b),r.jaccard_to_full_selected)
        assert r.exact_full_member_match==(a==b)
    for _,r in read_table('fold_selected_member_jaccard').iterrows():
        a,b=selected_sets[r.fold_1],selected_sets[r.fold_2]
        close('pairwise_selected_jaccard',len(a&b)/len(a|b),r.member_jaccard)
    for _,r in read_table('fold_selected_member_frequency').iterrows():
        folds=[f for f,s,_ in expected_folds if s==r.scheme]
        count=sum(r.metabolite in selected_sets[f] for f in folds)
        close('selection_frequency',[len(folds),count,count/len(folds)],[r.n_folds,r.selected_count,r.selected_fraction])
    CHECKS.append('Full/heldout target and13D errors, all pooled and fold metrics, member overlap/frequency')
    performance=read_table('pooled_performance')
    result={'status':'PASS','verification_design':'independent source-input recomputation; no imported analysis functions',
            'n_fits':46,'n_heldout_folds':45,'n_heldout_strain_predictions':58,'n_targets':16,
            'maximum_absolute_error':max(ERRORS.values()),'error_maxima':ERRORS,'checks':CHECKS,
            'primary_heldout_performance':performance[performance.target.eq(PRIMARY)].to_dict('records'),
            'scope':'No added scientific models, selection, thresholds or search; only verification/ written.'}
    (ROOT/'verification/independent_model_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(all_fit_rows).to_csv(ROOT/'verification/independently_verified_fits.csv',index=False)
    print(json.dumps({k:result[k] for k in ['status','n_fits','maximum_absolute_error','primary_heldout_performance']},indent=2))


if __name__ == '__main__':
    audit()
