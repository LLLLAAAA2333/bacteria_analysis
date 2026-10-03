"""Independently replay saved fits from source data using numpy least squares.

This sanity check does not call the model's fit/predict helpers for real data.
Full independent re-clustering is performed by the separate reviewer branch.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd


def verify(repo_root):
    repo=Path(repo_root).resolve();out=repo/'reports/exploration_bacteroides_adf_ash_chemical_20261003/model';t=out/'tables'
    source=repo/'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    x=pd.read_csv(source/'fresh_chemical_log2.csv',index_col='strain')
    metadata=pd.read_csv(source/'fresh_feature_metadata.csv',index_col='metabolite')
    unit=pd.read_csv(source/'neural_unit_coefficients.csv',index_col='strain')
    pre=pd.read_csv(source/'neural_pre_gate_unit_coefficients.csv',index_col='strain')
    context=pd.read_csv(source/'sample_context.csv',index_col='strain',dtype={'dates':str})
    raw=pd.read_csv(repo/'reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv',index_col='strain')
    ids=sorted(context.index[context.genus.eq('Bacteroides')]);x=x.loc[ids];unit=unit.loc[ids];pre=pre.loc[ids,unit.columns];raw=raw.loc[ids,unit.columns]
    primary='primary_unit_adf_minus_ash';raw_name='raw_adf_minus_ash';pre_name='pre_gate_unit_adf_minus_ash'
    y=pd.DataFrame({primary:unit.ADF-unit.ASH,raw_name:raw.ADF-raw.ASH,pre_name:pre.ADF-pre.ASH},index=ids)
    for neuron in unit:y['unit_'+neuron]=unit[neuron]
    targets=y.columns.tolist();unit_targets=['unit_'+c for c in unit]
    saved_cohort=pd.read_csv(t/'cohort_targets.csv',index_col='strain')
    saved_full=pd.read_csv(t/'selected_full_state_scores.csv',index_col='strain')
    held=pd.read_csv(t/'heldout_predictions.csv').set_index(['fold_id','strain'])
    folds=json.loads((out/'fold_manifest.json').read_text())
    assert len(folds)==45 and len(held)==58
    errors={}
    def close(name,actual,expected):
        aa,ee=np.asarray(actual,float),np.asarray(expected,float)
        np.testing.assert_allclose(aa,ee,rtol=1e-9,atol=1e-10)
        errors[name]=max(errors.get(name,0.),float(np.max(np.abs(aa-ee))) if aa.size else 0.)
    close('cohort_targets_from_source',saved_cohort.loc[ids,targets],y)
    close('raw_to_saved_unit',raw.to_numpy()/np.linalg.norm(raw,axis=1)[:,None],unit)
    unit_sse=[]
    parameter_files=[out/'parameters/full_cohort.json']+[out/fold['parameters_file'] for fold in folds]
    primary_selection_count=0
    for path in parameter_files:
        params=json.loads(path.read_text());train=params['train_ids'];test=params['omitted_ids'] or ids
        xt=x.loc[train];yt=y.loc[train];means=xt.to_numpy().mean(axis=0);sds=xt.to_numpy().std(axis=0,ddof=1)
        close('all46_training_means',[params['training_log2_means'][f] for f in x.columns],means)
        close('all46_training_sample_sd',[params['training_log2_sample_sds'][f] for f in x.columns],sds)
        train_scores={};test_scores={};candidates=[]
        for module,members in params['module_members'].items():
            families=metadata.loc[members,'family'];weights=np.array([1/families.nunique()/(families==families[f]).sum() for f in members])
            close('all46_exact_family_weights',[params['score_weights'][module][f] for f in members],weights)
            m=xt[members].to_numpy().mean(axis=0);s=xt[members].to_numpy().std(axis=0,ddof=1)
            train_scores[module]=(xt[members].to_numpy()-m)/s@weights
            test_scores[module]=(x.loc[test,members].to_numpy()-m)/s@weights
            r=float(np.corrcoef(train_scores[module],yt[primary])[0,1])
            candidates.append((-(r*r),tuple(sorted(members)),module))
        expected_selected=sorted(candidates)[0][2] if candidates else None
        assert params['selected_state_id']==expected_selected
        primary_selection_count+=1
        if expected_selected is not None:
            sx=train_scores[expected_selected];tx=test_scores[expected_selected]
        else:sx=np.zeros(len(train));tx=np.zeros(len(test))
        beta=np.linalg.lstsq(np.column_stack([np.ones(len(train)),sx]),yt.to_numpy(),rcond=None)[0]
        prediction=np.column_stack([np.ones(len(test)),tx])@beta
        baseline=np.tile(yt.to_numpy().mean(axis=0),(len(test),1))
        close('all46_intercepts',[params['response_models'][target]['intercept'] for target in targets],beta[0])
        close('all46_slopes',[params['response_models'][target]['slope'] for target in targets],beta[1])
        source=(saved_full.loc[test] if not params['omitted_ids'] else held.loc[[(params['fold_id'],s) for s in test]])
        close('all46_selected_scores',source.chemical_state_score,tx)
        close('all46_predictions',[source['pred_'+target].to_numpy() for target in targets],prediction.T)
        close('all46_training_mean_baselines',[source['baseline_'+target].to_numpy() for target in targets],baseline.T)
        close('all46_primary_pred_adf_minus_ash',prediction[:,targets.index('unit_ADF')]-prediction[:,targets.index('unit_ASH')],prediction[:,targets.index(primary)])
        assert params['no_state_fallback']==(expected_selected is None)
        if params['omitted_ids']:
            assert not set(train)&set(test) and set(train)|set(test)==set(ids)
            if params['scheme']=='leave_one_strain_out':assert len(test)==1
            else:
                assert test==context.loc[ids].index[context.loc[ids,'species'].eq(params['omitted_label'])].tolist()
    # Rebuild all pooled metrics directly from observations and saved predictions.
    for filename,wide,scheme_values in [('full_fit_performance.csv',saved_full.reset_index(),['full_apparent']),
                                        ('pooled_performance.csv',held.reset_index(),['leave_one_strain_out','leave_one_recorded_species_out'])]:
        table=pd.read_csv(t/filename)
        for row in table.itertuples():
            block=wide if row.scheme=='full_apparent' else wide[wide.scheme.eq(row.scheme)]
            cols=unit_targets if row.target=='unit_profile_13d' else [row.target]
            observed=block[cols].to_numpy();prediction=block[['pred_'+c for c in cols]].to_numpy();baseline=block[['baseline_'+c for c in cols]].to_numpy()
            em=np.sum((prediction-observed)**2,axis=1);eb=np.sum((baseline-observed)**2,axis=1)
            close('all51_performance_sse',[row.sse_model,row.sse_training_mean_baseline],[em.sum(),eb.sum()])
            close('all51_performance_rmse',[row.rmse_model,row.rmse_training_mean_baseline],[np.sqrt(em.mean()),np.sqrt(eb.mean())])
            close('all51_error_improvement',row.error_improvement,1-em.sum()/eb.sum())
            assert row.n_predictions==len(block) and row.n_coordinates==len(cols)
            assert row.n_strains_better_than_mean==int((em<eb).sum())
    selections=pd.read_csv(t/'fold_selection_stability.csv');full_members=set(json.loads((out/'parameters/full_cohort.json').read_text())['selected_members'])
    selected_sets={}
    for row in selections.itertuples():
        member_set=set(json.loads(row.members_json));selected_sets[row.fold_id]=member_set
        close('jaccard_to_full',row.jaccard_to_full_selected,len(member_set&full_members)/len(member_set|full_members))
        assert row.exact_full_member_match==(member_set==full_members)
    pairs=pd.read_csv(t/'fold_selected_member_jaccard.csv')
    assert len(pairs)==45*46//2
    for row in pairs.itertuples():
        a,b=selected_sets[row.fold_1],selected_sets[row.fold_2]
        close('all1035_member_jaccards',row.member_jaccard,len(a&b)/len(a|b) if a|b else 1.)
    manifest=json.loads((out/'manifest.json').read_text())
    for item in list(manifest['inputs'].values())+[manifest['frozen_axes_api'],manifest['root_protocol']]:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    assert hashlib.sha256((out/'code/fit_fixed_contrast.py').read_bytes()).hexdigest()==manifest['model_code_sha256']
    # Focused edge-case contract: no chemical state gives each response mean.
    from fit_fixed_contrast import load_axes_api,fit_selected_state,predict_state,run_analysis
    axes_api,_=load_axes_api(repo)
    xx=pd.DataFrame({'constant':[1.,1.,1.,1.]},index=['a','b','c','d'])
    meta=pd.DataFrame({'family':['one']},index=['constant'])
    yy=pd.DataFrame({primary:[-2.,0.,1.,3.],raw_name:[1.,4.,7.,10.]},index=xx.index)
    empty=fit_selected_state(xx.iloc[:3],meta,yy.iloc[:3],axes_api)
    _,pp=predict_state(xx.iloc[3:],empty,axes_api)
    assert empty['no_state_fallback'] and empty['selected_state_id'] is None
    close('empty_state_training_mean_fallback',pp.to_numpy()[0],yy.iloc[:3].mean())
    try:run_analysis(repo)
    except FileExistsError:pass
    else:raise AssertionError('Output guard failed')
    report={'status':'PASS','fits_replayed':46,'full_candidates':12,'heldout_rows':58,'candidate_selection_rechecks':primary_selection_count,
            'max_abs_errors':errors,'checks':['source target and raw-normalization identities','all46 training means/sampleSDs','family weights','candidate primary r squared selection',
                                            'numpy least-squares intercepts/slopes','all46 primary/aux/13-coordinate predictions','training-mean baselines','primary predicted contrast identity',
                                            '45 fold train/heldout IDs','all51 performance rows','1035 membership Jaccards','source/API/code hashes','empty-state mean fallback','existing-output refusal'],
            'scope':'Independent replay of saved module memberships and linear fits; separate reviewer independently re-clusters all training folds.'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','fits':46,'max_abs_error':max(errors.values())}))
    return report


if __name__=='__main__':verify(Path(__file__).resolve().parents[4])
