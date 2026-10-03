"""Focused independent consistency checks for this scientific analysis round."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
T=OUT/'tables'


def main():
    checks={}
    manifest=json.loads((OUT/'logs/input_manifest.json').read_text())
    for row in manifest:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    checks['unchanged_input_files']=len(manifest)
    codefiles=sorted((OUT/'code').glob('*.py'))
    for path in codefiles:
        ast.parse(path.read_text())
    checks['parsed_python_files']=len(codefiles)
    a=pd.read_csv(T/'animal_metrics.csv',dtype={'date':str})
    curves=pd.read_parquet(T/'animal_curves.parquet')
    trials=pd.read_parquet(T/'trial_curves.parquet')
    assert a.shape[0]==len(curves)==7063
    assert a[['date','worm_key']].drop_duplicates().shape[0]==49
    assert a.sample_id.nunique()==106
    assert a.groupby('neuron_class').sample_id.nunique().eq(106).all()
    checks['animal_curve_rows']=len(a)
    vals=curves[[str(i) for i in range(10)]].mean(axis=1)
    ix=['sample_id','date','worm_key','neuron_class']
    err=float(np.max(np.abs(vals-a.set_index(ix).stim.reindex(vals.index))))
    assert err<1e-12
    checks['curve_scalar_max_error']=err
    # Independently rederive sampled curves from stored raw channels, including
    # singleton/bilateral classes and repeated/single-date strains.
    raw=pd.read_parquet(ROOT/'data/106bac.parquet')
    raw['sample_id']=raw.stim_name.str.extract(r'^(A\d{3})',expand=False)
    sampled=a.sample(18,random_state=20260929)
    forced=a[(a.sample_id.isin(['A024','A247','A249'])) & a.neuron_class.isin(['ADF','AWCON'])]
    sampled=pd.concat([sampled,forced]).drop_duplicates(ix)
    errors=[]
    bilateral=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ']
    for row in sampled.itertuples():
        nn=[row.neuron_class+s for s in ['L','R']] if row.neuron_class in bilateral else [row.neuron_class]
        sub=raw[raw.sample_id.eq(row.sample_id)&raw.date.eq(row.date)&raw.worm_key.eq(row.worm_key)&raw.neuron.isin(nn)]
        r=sub.groupby(['segment_index','time_point']).delta_F_over_F0.mean().groupby('time_point').mean().to_numpy()
        saved=curves.loc[(row.sample_id,row.date,row.worm_key,row.neuron_class)].to_numpy()
        errors.append(float(np.max(np.abs(r-saved))))
    assert max(errors)<1e-12
    checks['independent_raw_curve_checks']=len(errors)
    checks['independent_raw_curve_max_error']=max(errors)
    ph=pd.read_csv(T/'phenotype_animal_sensitivity.csv',dtype={'date':str}).set_index(ix)
    err=float(np.max(np.abs(ph['mean']-a.set_index(ix).stim.reindex(ph.index))))
    assert err<1e-12
    checks['independent_trial_metric_max_error']=err
    spec=importlib.util.spec_from_file_location('ct',OUT/'code/05_chemical_tests.py')
    ct=importlib.util.module_from_spec(spec);spec.loader.exec_module(ct)
    d=a[a.neuron_class.eq('ADF')].reset_index(drop=True)
    tax=pd.read_csv(T/'taxonomy.csv').set_index('sample_id')
    ref=pd.read_csv(T/'chemical_reference_groups.csv').set_index('sample_id')
    log=pd.read_csv(T/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(T/'chemical_feature_metadata.csv',index_col=0)
    names=meta.index[meta.complete_eligible]
    assert len(names)==162 and log[names].notna().all().all()
    d=d.join(tax[['genus_clean']],on='sample_id').join(ref[['reference_group']],on='sample_id')
    d['total_log']=d.sample_id.map(log[names].median(axis=1))
    X=log.loc[d.sample_id,['Asparagine']].to_numpy()
    f=ct.fit(d,X,'full');sw=np.sqrt(f['w']);zw=ct.design(d,'full')*sw[:,None]
    orthog=float(np.linalg.norm(zw.T@(f['yr']*sw))/(np.linalg.norm(zw)*np.linalg.norm(d.stim.to_numpy()*sw)))
    assert orthog<1e-12
    whole=np.column_stack([zw,X*sw[:,None]])
    independent_beta=np.linalg.lstsq(whole,d.stim.to_numpy()*sw,rcond=None)[0][-1]
    assert abs(independent_beta-f['beta'][0])<1e-10
    checks['weighted_projection_relative_orthogonality_error']=orthog
    checks['partial_vs_joint_fit_slope_error']=float(abs(independent_beta-f['beta'][0]))
    cv=pd.read_csv(T/'chem_cv_predictions.csv',dtype={'date':str})
    repeats=set(a.groupby('sample_id').date.nunique().loc[lambda x:x>1].index)
    for _,g in cv.groupby(['target','panel']):
        assert len(g)==g.sample_id.nunique()==100 and g.date.nunique()==9
        assert not set(g.sample_id)&repeats
    checks['cv_unique_strains_per_target_panel']=100
    # Independently recompute one feature-selection fold, rather than checking
    # only saved prediction shapes.
    single=d[~d.sample_id.isin(repeats)]
    train=single[single.date.ne('20260601')].reset_index(drop=True)
    ff=ct.fit(train,log.loc[train.sample_id,names].to_numpy())
    winner=names[np.argmax(abs(ff['r']))]
    fold=pd.read_csv(T/'chem_cv_folds.csv',dtype={'date':str}).query("target=='ADF' and panel=='complete' and date=='20260601'").iloc[0]
    assert winner==fold.selected_feature
    checks['independent_adf_fold_selection']=winner
    # Reconcile numerical summary with prediction tables.
    summary=json.loads((OUT/'logs/chemical_results.json').read_text())
    for item in summary['cv']:
        g=cv[(cv.target==item['target'])&(cv.panel==item['panel'])]
        r2=1-np.sum((g.observed_relative-g.predicted_relative)**2)/np.sum(g.observed_relative**2)
        assert abs(r2-item['relative_R2'])<1e-12
    checks['cv_summary_reconciled']=True
    for path in (OUT/'figures').glob('*.png'):
        assert path.stat().st_size>10000
    core=['02_adf_crossdate','03_awcon_response','05_asparagine_refutation']
    for stem in core:
        assert (OUT/'figures'/f'{stem}.png').exists(),stem
        assert (OUT/'figures'/f'{stem}.pdf').exists(),stem
    for name in ['REVIEW.md','REPORT.md','RESEARCH_LOG.md','README.md']:
        assert (OUT/name).exists()
    checks['core_figures']=core
    checks['source_raw_F_or_exposure_metadata_recovered']=False
    checks['confirmatory_or_independent_validation_performed']=False
    checks['status']='passed'
    (OUT/'logs/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
