"""Fixed ADF-ASH response, train-local chemical states and one-slope OLS.

No neural target selection, prediction renormalization, or secondary state search.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import platform
import shutil
import numpy as np
import pandas as pd
import scipy

PRIMARY = 'primary_unit_adf_minus_ash'
RAW = 'raw_adf_minus_ash'
PRE = 'pre_gate_unit_adf_minus_ash'
SCHEMES = ['leave_one_strain_out', 'leave_one_recorded_species_out']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_axes_api(repo):
    path = Path(repo) / 'reports/exploration_bacteroides_local_model_20261003/chemical/code/local_chemical_axes.py'
    spec = importlib.util.spec_from_file_location('frozen_local_chemical_axes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def fit_ols(x, y):
    """OLS with an intercept; return scalar parameters and training diagnostics."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = float(x.mean()), float(y.mean())
    dx, dy = x - xm, y - ym
    xx, yy = float(dx @ dx), float(dy @ dy)
    slope = float(dx @ dy / xx) if xx > 1e-24 else 0.
    intercept = ym - slope * xm
    pred = intercept + slope * x
    sse = float(np.sum((y-pred)**2))
    r = float(dx @ dy / np.sqrt(xx*yy)) if xx > 1e-24 and yy > 1e-24 else 0.
    return {'intercept': intercept, 'slope': slope, 'training_mean': ym, 'training_predictor_mean': xm,
            'pearson_r': r, 'r_squared': r*r, 'sse_model': sse, 'sse_mean': yy,
            'rmse_model': float(np.sqrt(sse/len(y))), 'rmse_mean': float(np.sqrt(yy/len(y))),
            'response_constant': bool(yy <= 1e-24), 'predictor_constant': bool(xx <= 1e-24)}


def fit_selected_state(log_frame, feature_metadata, targets, axes_api):
    """Fit chemistry on train only; select once on primary r^2, fit all targets."""
    if not log_frame.index.equals(targets.index):
        raise ValueError('Training chemistry and targets must have identical strain order')
    fitted_axes = axes_api.fit_axes(log_frame, feature_metadata)
    candidate_rows = []
    for state, members in fitted_axes['module_members'].items():
        primary_fit = fit_ols(fitted_axes['train_scores'][state], targets[PRIMARY])
        family_count = feature_metadata.loc[members, 'family'].nunique()
        candidate_rows.append({'state_id': state, 'n_annotations': len(members), 'n_families': int(family_count),
                               'primary_pearson_r': primary_fit['pearson_r'], 'primary_r_squared': primary_fit['r_squared'],
                               'primary_intercept': primary_fit['intercept'], 'primary_slope': primary_fit['slope'],
                               'primary_sse_model': primary_fit['sse_model'], 'primary_sse_mean': primary_fit['sse_mean'],
                               'members': members, 'member_tie_key': tuple(sorted(members))})
    ranked = sorted(candidate_rows, key=lambda row: (-row['primary_r_squared'], row['member_tie_key']))
    selected = ranked[0]['state_id'] if ranked else None
    score = fitted_axes['train_scores'][selected] if selected is not None else pd.Series(0., index=log_frame.index)
    response_models = {target: fit_ols(score, targets[target]) for target in targets}
    return {'axes': fitted_axes, 'selected_state_id': selected, 'candidate_rows': candidate_rows,
            'response_models': response_models, 'no_state_fallback': selected is None}


def predict_state(log_frame, fitted, axes_api):
    transformed = axes_api.transform_axes(log_frame, fitted['axes'])
    state = fitted['selected_state_id']
    score = transformed[state] if state is not None else pd.Series(0., index=log_frame.index)
    predictions = pd.DataFrame({target: fit['intercept'] + fit['slope'] * score
                                for target, fit in fitted['response_models'].items()}, index=log_frame.index)
    return score, predictions


def candidate_frame(fitted):
    columns = ['state_id', 'n_annotations', 'n_families', 'primary_pearson_r', 'primary_r_squared',
               'primary_intercept', 'primary_slope', 'primary_sse_model', 'primary_sse_mean', 'members_json', 'selected']
    records = []
    for row in fitted['candidate_rows']:
        record = {k:v for k,v in row.items() if k not in ['members','member_tie_key']}
        record['members_json'] = json.dumps(row['members'])
        record['selected'] = row['state_id'] == fitted['selected_state_id']
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def parameter_record(fitted, train_ids, omitted_ids, fold_id, scheme):
    axes = fitted['axes'];state = fitted['selected_state_id']
    return {'fold_id': fold_id, 'scheme': scheme, 'train_ids': list(train_ids), 'omitted_ids': list(omitted_ids),
            'selected_state_id': state, 'selected_members': axes['module_members'].get(state, []),
            'no_state_fallback': fitted['no_state_fallback'],
            'primary_target': PRIMARY, 'selection': 'maximum training Pearson r squared; ties by sorted member tuple',
            'source_features': axes['source_features'], 'retained_features': axes['retained_features'],
            'excluded_features': axes['excluded_features'], 'training_log2_means': axes['means'].to_dict(),
            'training_log2_sample_sds': axes['scales'].to_dict(), 'feature_order': axes['feature_order'],
            'module_members': axes['module_members'],
            'score_weights': {m:w.to_dict() for m,w in axes['score_weights'].items()},
            'chemical_parameters': axes['parameters'], 'candidates': candidate_frame(fitted).to_dict('records'),
            'training_all_state_scores': axes['train_scores'].to_dict(orient='index'),
            'response_models': fitted['response_models']}


def errors_for_predictions(observed, predicted, fitted, fold_id, scheme):
    records=[]
    for strain in observed.index:
        for target in observed:
            actual=float(observed.loc[strain,target]);prediction=float(predicted.loc[strain,target])
            baseline=fitted['response_models'][target]['training_mean']
            records.append({'scheme':scheme,'fold_id':fold_id,'strain':strain,'target':target,
                            'observed':actual,'predicted':prediction,'training_mean_baseline':baseline,
                            'error_model':prediction-actual,'error_training_mean_baseline':baseline-actual,
                            'squared_error_model':(prediction-actual)**2,
                            'squared_error_training_mean_baseline':(baseline-actual)**2})
    return pd.DataFrame(records)


def performance_table(errors, unit_targets):
    records=[]
    def summarize(block, scheme, target, n_coordinates):
        n=len(block);model=float(block.squared_error_model.sum());baseline=float(block.squared_error_training_mean_baseline.sum())
        return {'scheme':scheme,'target':target,'n_predictions':n,'n_coordinates':n_coordinates,
                'sse_model':model,'sse_training_mean_baseline':baseline,
                'rmse_model':float(np.sqrt(model/n)), 'rmse_training_mean_baseline':float(np.sqrt(baseline/n)),
                'rmse_per_coordinate_model':float(np.sqrt(model/(n*n_coordinates))),
                'rmse_per_coordinate_training_mean_baseline':float(np.sqrt(baseline/(n*n_coordinates))),
                'error_improvement':float(1-model/baseline) if baseline>0 else np.nan,
                'n_strains_better_than_mean':int((block.squared_error_model < block.squared_error_training_mean_baseline).sum())}
    for (scheme,target),block in errors.groupby(['scheme','target'],sort=False):
        records.append(summarize(block,scheme,target,1))
    vectors=errors[errors.target.isin(unit_targets)].groupby(['scheme','fold_id','strain'],sort=False)[['squared_error_model','squared_error_training_mean_baseline']].sum().reset_index()
    for scheme,block in vectors.groupby('scheme',sort=False):
        records.append(summarize(block,scheme,'unit_profile_13d',len(unit_targets)))
    return pd.DataFrame(records),vectors


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run_analysis(repo_root, out=None):
    repo=Path(repo_root).resolve()
    canonical=repo/'reports/exploration_bacteroides_adf_ash_chemical_20261003/model'
    output=Path(out).resolve() if out is not None else canonical
    if (output/'manifest.json').exists() or any((output/'tables').glob('*.csv')):
        raise FileExistsError(f'Refusing to overwrite existing fixed-contrast model results: {output}')
    d=repo/'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    source_names=['fresh_chemical_log2.csv','fresh_feature_metadata.csv','neural_unit_coefficients.csv',
                  'neural_pre_gate_unit_coefficients.csv','sample_context.csv']
    inputs={name:d/name for name in source_names}
    inputs['strain_coefficients.csv']=repo/'reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv'
    x0=pd.read_csv(inputs[source_names[0]],index_col='strain')
    feature_metadata=pd.read_csv(inputs[source_names[1]],index_col='metabolite')
    unit0=pd.read_csv(inputs[source_names[2]],index_col='strain')
    pre0=pd.read_csv(inputs[source_names[3]],index_col='strain')
    context=pd.read_csv(inputs[source_names[4]],index_col='strain',dtype={'dates':str})
    raw0=pd.read_csv(inputs['strain_coefficients.csv'],index_col='strain')
    for frame in [x0,unit0,pre0,context,raw0]: assert frame.index.is_unique
    assert x0.columns.is_unique and feature_metadata.index.is_unique
    assert set(x0.index)==set(unit0.index)==set(pre0.index)==set(context.index)
    assert set(x0.columns)==set(feature_metadata.index)
    assert set(unit0.columns)==set(pre0.columns)==set(raw0.columns)
    ids=sorted(context.index[context.genus.eq('Bacteroides')])
    assert len(ids)==29 and set(ids).issubset(raw0.index)
    x=x0.loc[ids];unit=unit0.loc[ids];pre=pre0.loc[ids,unit.columns];raw=raw0.loc[ids,unit.columns]
    metadata=feature_metadata.loc[x.columns].copy();metadata.index.name='metabolite'
    context=context.loc[ids].copy();context['taxonomy_flag']=context.taxonomy_note.fillna('').astype(str).str.strip().ne('')
    assert context.species.nunique()==16 and int(context.taxonomy_flag.sum())==6
    for frame in [x,unit,pre,raw]: assert np.isfinite(frame.to_numpy()).all()
    norm=np.linalg.norm(raw,axis=1);assert (norm>0).all()
    normalized_raw=raw.to_numpy()/norm[:,None]
    identity_error=float(np.abs(normalized_raw-unit.to_numpy()).max())
    np.testing.assert_allclose(normalized_raw,unit,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(unit,axis=1),np.ones(29),atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(pre,axis=1),np.ones(29),atol=1e-12)
    cohort=context.copy();cohort['coefficient_l2_norm']=norm
    cohort[PRIMARY]=unit.ADF-unit.ASH
    cohort[RAW]=raw.ADF-raw.ASH
    cohort[PRE]=pre.ADF-pre.ASH
    cohort['raw_ADF']=raw.ADF;cohort['raw_ASH']=raw.ASH
    for neuron in unit: cohort['unit_'+neuron]=unit[neuron]
    unit_targets=['unit_'+n for n in unit.columns]
    target_names=[PRIMARY,RAW,PRE]+unit_targets
    targets=cohort[target_names]
    np.testing.assert_allclose(cohort[PRIMARY],cohort[RAW]/norm,atol=1e-12)
    axes_api,api_path=load_axes_api(repo)
    for folder in ['tables','parameters']:(output/folder).mkdir(parents=True,exist_ok=True)
    protocol=repo/'reports/exploration_bacteroides_adf_ash_chemical_20261003/PROTOCOL.md'
    shutil.copyfile(protocol,output/'protocol_snapshot.md')
    t=output/'tables'
    cohort.to_csv(t/'cohort_targets.csv')
    x.to_csv(t/'chemical_log2_29x162.csv');metadata.to_csv(t/'feature_metadata_162.csv')
    unit.to_csv(t/'unit_profiles_29x13.csv');pre.to_csv(t/'pre_gate_unit_profiles_29x13.csv');raw.to_csv(t/'raw_coefficients_29x13.csv')
    full=fit_selected_state(x,metadata,targets,axes_api)
    full_score,full_predictions=predict_state(x,full,axes_api)
    candidate_frame(full).to_csv(t/'full_candidate_results.csv',index=False)
    full['axes']['train_scores'].to_csv(t/'full_candidate_scores.csv')
    full_state=cohort.copy();full_state.insert(0,'chemical_state_score',full_score)
    full_state.insert(1,'selected_state_id',full['selected_state_id'])
    for target in target_names:
        full_state['pred_'+target]=full_predictions[target]
        full_state['baseline_'+target]=full['response_models'][target]['training_mean']
    full_state.to_csv(t/'selected_full_state_scores.csv')
    selected_members=full['axes']['module_members'].get(full['selected_state_id'],[])
    member_table=metadata.loc[selected_members].copy()
    member_table['training_log2_mean']=full['axes']['means'].loc[selected_members]
    member_table['training_log2_sample_sd']=full['axes']['scales'].loc[selected_members]
    member_table['score_weight']=full['axes']['score_weights'].get(full['selected_state_id'],pd.Series(dtype=float))
    member_table.to_csv(t/'selected_full_state_members.csv')
    pd.DataFrame([{'target':target,**model} for target,model in full['response_models'].items()]).to_csv(t/'full_selected_response_models.csv',index=False)
    full_params=parameter_record(full,ids,[],'full_cohort','full_apparent')
    save_json(output/'parameters/full_cohort.json',full_params)
    full_errors=errors_for_predictions(targets,full_predictions,full,'full_cohort','full_apparent')
    full_errors.to_csv(t/'full_fit_target_errors.csv',index=False)
    full_performance,full_vectors=performance_table(full_errors,unit_targets)
    full_performance.to_csv(t/'full_fit_performance.csv',index=False)
    full_vectors.to_csv(t/'full_fit_vector_errors.csv',index=False)
    algebra_error=float(np.abs(full_predictions['unit_ADF']-full_predictions['unit_ASH']-full_predictions[PRIMARY]).max())
    assert algebra_error<1e-12
    print(json.dumps({'event':'full_fit_tables_ready','selected_state_id':full['selected_state_id'],
                      'n_candidates':len(full['candidate_rows']),'members':selected_members,
                      'primary_pearson_r':full['response_models'][PRIMARY]['pearson_r']}),flush=True)
    tests=[(SCHEMES[0],f'strain_{s}',s,[s]) for s in ids]
    tests += [(SCHEMES[1],f'species_{i:02d}',species,context.index[context.species.eq(species)].tolist())
              for i,species in enumerate(sorted(context.species.unique()),1)]
    heldout_rows=[];all_errors=[];selection_rows=[];fold_metric_rows=[];fold_records=[]
    full_member_set=set(selected_members)
    for scheme,fold_id,label,omitted in tests:
        train_ids=[s for s in ids if s not in omitted]
        fitted=fit_selected_state(x.loc[train_ids],metadata,targets.loc[train_ids],axes_api)
        score,predictions=predict_state(x.loc[omitted],fitted,axes_api)
        params=parameter_record(fitted,train_ids,omitted,fold_id,scheme);params['omitted_label']=label
        save_json(output/'parameters'/f'{fold_id}.json',params)
        fold_records.append({'fold_id':fold_id,'scheme':scheme,'omitted_label':label,'n_train':len(train_ids),'n_test':len(omitted),
                             'parameters_file':f'parameters/{fold_id}.json','train_ids':train_ids,'omitted_ids':omitted})
        local_members=fitted['axes']['module_members'].get(fitted['selected_state_id'],[])
        local_set=set(local_members);union=full_member_set|local_set
        jaccard=len(full_member_set&local_set)/len(union) if union else 1.
        selection_rows.append({'fold_id':fold_id,'scheme':scheme,'omitted_label':label,'n_train':len(train_ids),
                               'n_candidates':len(fitted['candidate_rows']),'selected_state_id':fitted['selected_state_id'],
                               'n_selected_members':len(local_members),'members_json':json.dumps(local_members),
                               'jaccard_to_full_selected':jaccard,'exact_full_member_match':local_set==full_member_set,
                               'no_state_fallback':fitted['no_state_fallback']})
        for strain in omitted:
            row={'scheme':scheme,'fold_id':fold_id,'strain':strain,'omitted_label':label,'n_train':len(train_ids),
                 'selected_state_id':fitted['selected_state_id'],'n_candidates':len(fitted['candidate_rows']),
                 'chemical_state_score':float(score.loc[strain]) if not fitted['no_state_fallback'] else np.nan,
                 'no_state_fallback':fitted['no_state_fallback'],**cohort.loc[strain].to_dict()}
            for target in target_names:
                row['pred_'+target]=float(predictions.loc[strain,target])
                row['baseline_'+target]=fitted['response_models'][target]['training_mean']
            heldout_rows.append(row)
        errors=errors_for_predictions(targets.loc[omitted],predictions,fitted,fold_id,scheme)
        all_errors.append(errors)
        fold_perf,_=performance_table(errors,unit_targets)
        fold_perf.insert(1,'fold_id',fold_id);fold_perf.insert(2,'n_train',len(train_ids));fold_metric_rows.append(fold_perf)
        error=float(np.abs(predictions['unit_ADF']-predictions['unit_ASH']-predictions[PRIMARY]).max())
        algebra_error=max(algebra_error,error);assert error<1e-12
    heldout=pd.DataFrame(heldout_rows);errors=pd.concat(all_errors,ignore_index=True)
    pooled,vectors=performance_table(errors,unit_targets)
    selections=pd.DataFrame(selection_rows)
    heldout.to_csv(t/'heldout_predictions.csv',index=False)
    errors.to_csv(t/'heldout_target_errors.csv',index=False)
    vectors.to_csv(t/'heldout_vector_errors.csv',index=False)
    pooled.to_csv(t/'pooled_performance.csv',index=False)
    pd.concat(fold_metric_rows,ignore_index=True).to_csv(t/'fold_performance.csv',index=False)
    selections.to_csv(t/'fold_selection_stability.csv',index=False)
    save_json(output/'fold_manifest.json',fold_records)
    pair_rows=[]
    selected_sets={row['fold_id']:set(json.loads(row['members_json'])) for row in selection_rows}
    for i,a in enumerate(selection_rows):
        for b in selection_rows[i:]:
            aa,bb=selected_sets[a['fold_id']],selected_sets[b['fold_id']];union=aa|bb
            pair_rows.append({'fold_1':a['fold_id'],'scheme_1':a['scheme'],'fold_2':b['fold_id'],'scheme_2':b['scheme'],
                              'member_jaccard':len(aa&bb)/len(union) if union else 1.})
    pd.DataFrame(pair_rows).to_csv(t/'fold_selected_member_jaccard.csv',index=False)
    frequency_rows=[]
    for scheme in SCHEMES:
        scheme_selections=[row for row in selection_rows if row['scheme']==scheme]
        for feature in x.columns:
            count=sum(feature in selected_sets[row['fold_id']] for row in scheme_selections)
            frequency_rows.append({'scheme':scheme,'metabolite':feature,'n_folds':len(scheme_selections),
                                   'selected_count':count,'selected_fraction':count/len(scheme_selections),
                                   'in_full_selected_state':feature in full_member_set})
    pd.DataFrame(frequency_rows).to_csv(t/'fold_selected_member_frequency.csv',index=False)
    summary={'cohort':{'n_strains':29,'n_recorded_species':16,'n_chemical_features':162,'n_taxonomy_flags':6},
             'primary_target':PRIMARY,'target_definition':'unit_ADF - unit_ASH = (raw_coefficient_ADF - raw_coefficient_ASH)/L2_norm_13',
             'full_n_candidates':len(full['candidate_rows']),'full_selected_state_id':full['selected_state_id'],
             'full_selected_members':selected_members,'full_primary_response_model':full['response_models'][PRIMARY],
             'full_performance':full_performance.where(pd.notna(full_performance),None).to_dict('records'),
             'heldout_performance':pooled.where(pd.notna(pooled),None).to_dict('records'),
             'selection_stability':{scheme:{'n_folds':int(selections.scheme.eq(scheme).sum()),
                 'no_state_fallback_count':int(selections.loc[selections.scheme.eq(scheme),'no_state_fallback'].sum()),
                 'exact_full_member_match_count':int(selections.loc[selections.scheme.eq(scheme),'exact_full_member_match'].sum()),
                 'jaccard_to_full_min':float(selections.loc[selections.scheme.eq(scheme),'jaccard_to_full_selected'].min()),
                 'jaccard_to_full_median':float(selections.loc[selections.scheme.eq(scheme),'jaccard_to_full_selected'].median()),
                 'jaccard_to_full_max':float(selections.loc[selections.scheme.eq(scheme),'jaccard_to_full_selected'].max())} for scheme in SCHEMES},
             'checks':{'raw_to_unit_max_abs_error':identity_error,'primary_prediction_adf_minus_ash_max_abs_error':algebra_error},
             'metric':'Held-out error improvement = 1 - SSE_model/SSE_training_mean_baseline; not ordinary in-sample R squared.',
             'all13_architecture':'Same primary-selected chemical state, separately fitted intercept/slope for each unit coordinate; predictions are not renormalized.',
             'no_state_rule':'Use each response training mean if chemical fit yields no eligible state.',
             'limits':'Target prioritized from prior reliability in this cohort; frozen global neural templates; overlapping folds; no independent culture/animal confirmation.'}
    save_json(output/'summary.json',summary)
    manifest={'inputs':{name:{'path':str(path),'sha256':sha256(path)} for name,path in inputs.items()},
              'frozen_axes_api':{'path':str(api_path),'sha256':sha256(api_path)},
              'root_protocol':{'path':str(protocol),'sha256':sha256(protocol)},'protocol_snapshot_sha256':sha256(output/'protocol_snapshot.md'),
              'model_code_sha256':sha256(__file__), 'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
              'neuron_order':unit.columns.tolist(),'target_order':target_names,'strain_order':ids,
              'parameters':{'chemical':full['axes']['parameters'],'model':'OLS intercept plus one selected chemical state',
                            'selection':'training primary Pearson r squared; sorted member tuple tie','no_secondary_selection':True,'predicted_vector_renormalization':False}}
    save_json(output/'manifest.json',manifest)
    print(json.dumps({'event':'complete','folds':len(tests),'predictions':len(heldout),'primary_performance':pooled[pooled.target.eq(PRIMARY)].to_dict('records')}),flush=True)
    return output


if __name__=='__main__':
    run_analysis(Path(__file__).resolve().parents[4])
