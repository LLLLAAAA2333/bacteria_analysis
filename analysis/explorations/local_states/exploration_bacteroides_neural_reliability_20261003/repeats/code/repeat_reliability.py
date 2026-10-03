"""Fixed-template Bacteroides repeat checks. No chemical input or model."""
from pathlib import Path
import hashlib
import json
import platform
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REL=Path('reports/exploration_bacteroides_neural_reliability_20261003/repeats')
EPS=1e-12


def frozen_condition_fit(raw, templates):
    """Animal x condition x neuron x 40; recompute SNR, never refit templates."""
    present=np.isfinite(raw).all(axis=-1)
    n=present.sum(axis=0)
    total=np.where(present[...,None],raw,0.).sum(axis=0)
    mu=np.divide(total,n[...,None],out=np.full(raw.shape[1:],np.nan),where=n[...,None]>0)
    deviation=np.where(present[...,None],raw-mu[None],0.)
    variance=np.divide(np.sum(deviation**2,axis=0),(n-1)[...,None],
                       out=np.full(raw.shape[1:],np.nan),where=n[...,None]>=2)
    p=np.mean(mu**2,axis=-1);v=np.mean(variance,axis=-1)
    coherent=p-np.divide(v,n,out=np.full(n.shape,np.nan),where=n>0)
    numerator=np.maximum(coherent,0.)
    ratio=np.divide(numerator,v,out=np.full(n.shape,np.nan),where=v>0)
    ratio[(v==0)&(numerator>0)]=np.inf
    ratio[(v==0)&(numerator==0)]=0.
    snr=np.sqrt(ratio)
    bins=mu.reshape(*mu.shape[:-1],8,5).mean(axis=-1)
    projection=np.einsum('kcb,cb->kc',bins,templates)/np.sum(templates**2,axis=1)
    eligible=n>=2;passed=eligible&(snr>=.5)
    status=np.full(n.shape,'missing',dtype='U16')
    status[(n>0)&~eligible]='limited_n';status[eligible&~passed]='below_snr';status[passed]='retained'
    pre=np.where(eligible,projection,np.nan)
    gated=np.where(passed,projection,np.where(eligible,0.,np.nan))
    return {'n':n,'eligible':eligible,'snr':snr,'passed':passed,'status':status,
            'projection':projection,'gated':gated,'pre_gate':pre}


def shared_date_aggregate(values, shared, conditions, strains):
    """Both halves call with identical date x neuron support; unsupported stays NaN."""
    result=np.full((len(strains),values.shape[1]),np.nan)
    date_counts=np.zeros(result.shape,dtype=int)
    original_complete=np.zeros(len(strains),dtype=bool)
    for i,strain in enumerate(strains):
        rows=np.array([s==strain for s,d in conditions])
        support=shared[rows];count=support.sum(axis=0)
        sums=np.where(support,values[rows],0.).sum(axis=0)
        result[i]=np.divide(sums,count,out=np.full(values.shape[1],np.nan),where=count>0)
        date_counts[i]=count;original_complete[i]=bool(support.all())
    return result,date_counts,original_complete


def unit_profiles(coefficients):
    complete=np.isfinite(coefficients).all(axis=1)
    norms=np.full(len(coefficients),np.nan)
    norms[complete]=np.linalg.norm(coefficients[complete],axis=1)
    valid=complete&(norms>EPS)
    units=np.full_like(coefficients,np.nan)
    units[valid]=coefficients[valid]/norms[valid,None]
    return units,norms,valid


def metric_arrays(unit, mean, plane, cells):
    scores=(unit-mean)@plane
    return {'unit_13D':unit,'fixed_PC12':scores,'fixed_PC1':scores[:,[0]],
            'fixed_PC2':scores[:,[1]],
            'unit_ADF_minus_ASH':(unit[:,cells.index('ADF')]-unit[:,cells.index('ASH')])[:,None],
            'unit_AWB':unit[:,[cells.index('AWB')]]}


def comparison_metrics(a,b,date_sets):
    """Same cohort, all ordered i != j pairs; no zero filling or rescaling."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    n=len(a);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    same=np.sum((a-b)**2,axis=1)
    cross=np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)
    off=~np.eye(n,dtype=bool)
    matched=off&(np.asarray(date_sets)[:,None]==np.asarray(date_sets)[None,:])
    s=float(np.sqrt(np.mean(same))) if n else np.nan
    between=float(np.sqrt(np.mean(cross[off]))) if off.any() else np.nan
    matched_rms=float(np.sqrt(np.mean(cross[matched]))) if matched.any() else np.nan
    pearson=spearman=np.nan
    if a.shape[1]==1 and n>=3 and np.std(a[:,0])>EPS and np.std(b[:,0])>EPS:
        pearson=float(np.corrcoef(a[:,0],b[:,0])[0,1])
        spearman=float(spearmanr(a[:,0],b[:,0]).statistic)
    return {'n_strains':n,'n_ordered_between_pairs':int(off.sum()),'same_RMS':s,
            'between_RMS':between,'same_to_between_ratio':s/between if between>EPS else np.nan,
            'same_original_date_label_between_RMS':matched_rms,
            'n_same_original_date_label_ordered_pairs':int(matched.sum()),
            'pearson':pearson,'spearman':spearman}


def plane_comparison(a,b):
    """Center each half on the same complete strain cohort; no column SD scaling."""
    if len(a)<3:
        return {'angle_small_deg':np.nan,'angle_large_deg':np.nan,'half_A_PC12_variance_ratio':np.nan,'half_B_PC12_variance_ratio':np.nan}
    _,sa,va=np.linalg.svd(a-a.mean(axis=0),full_matrices=False)
    _,sb,vb=np.linalg.svd(b-b.mean(axis=0),full_matrices=False)
    if min(sa[1],sb[1])<=EPS:
        return {'angle_small_deg':np.nan,'angle_large_deg':np.nan,'half_A_PC12_variance_ratio':np.nan,'half_B_PC12_variance_ratio':np.nan}
    singular=np.linalg.svd(va[:2]@vb[:2].T,compute_uv=False)
    angles=np.degrees(np.arccos(np.clip(singular,0,1)))
    return {'angle_small_deg':float(angles[0]),'angle_large_deg':float(angles[1]),
            'half_A_PC12_variance_ratio':float(np.sum(sa[:2]**2)/np.sum(sa**2)),
            'half_B_PC12_variance_ratio':float(np.sum(sb[:2]**2)/np.sum(sb**2))}


def _quantiles(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return {'n_valid_splits':int(len(x)),'median':float(np.median(x)) if len(x) else None,
            'q05':float(np.quantile(x,.05)) if len(x) else None,
            'q95':float(np.quantile(x,.95)) if len(x) else None}


def run_analysis(repo_root,out=None):
    root=Path(repo_root).resolve();output=root/REL if out is None else Path(out)
    if not output.is_absolute():output=(root/output).resolve()
    if (output/'summary.json').exists() or ((output/'tables').is_dir() and any((output/'tables').iterdir())):
        raise FileExistsError(f'Saved results exist: {output}; choose a fresh output directory.')
    src=root/'reports/exploration_chemical_pattern_direct_report_20261003/tables'
    old=root/'reports/exploration_response_profiles_individual_snr_20261002/tables'
    local=root/'reports/exploration_bacteroides_local_model_20261003/neural/tables'
    paths={'context':src/'sample_context.csv','full_unit':src/'neural_unit_coefficients.csv',
           'conditions':old/'condition_metrics.csv','templates':old/'templates.csv',
           'assignments':old/'split_animal_assignments.csv',
           'animal_curves':root/'reports/exploration_20260929/tables/animal_curves.parquet',
           'full29_mean':local/'mean_unit_profile.csv','full29_loadings':local/'neural_loadings.csv'}
    manifest={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for k,p in paths.items()}
    metadata=pd.read_csv(paths['context'],index_col='strain',dtype={'dates':str})
    metadata=metadata.loc[metadata.genus.eq('Bacteroides')].sort_index();strains=metadata.index.tolist()
    full_unit=pd.read_csv(paths['full_unit'],index_col='strain').loc[strains]
    cells=full_unit.columns.tolist();assert len(strains)==29 and len(cells)==13
    loading=pd.read_csv(paths['full29_loadings'],index_col='neuron').loc[cells,['PC1','PC2']].to_numpy()
    mean=pd.read_csv(paths['full29_mean'],index_col='neuron').loc[cells,'mean_unit'].to_numpy()
    assert np.allclose(mean,full_unit.mean(axis=0),atol=1e-12)
    template_table=pd.read_csv(paths['templates'])
    templates=template_table.pivot(index='cell',columns='bin_index',values='template').loc[cells].to_numpy()
    assert np.isfinite(templates).all() and np.allclose(np.mean(templates**2,axis=1),1)
    curves=pd.read_parquet(paths['animal_curves'],filters=[('sample_id','in',strains)]).reset_index()
    for col in ['sample_id','date','worm_key','neuron_class']:curves[col]=curves[col].astype(str)
    assert not curves.duplicated(['sample_id','date','worm_key','neuron_class']).any()
    curves['animal_id']=curves.date+'|'+curves.worm_key
    animals=sorted(curves.animal_id.unique());conditions=sorted(set(zip(curves.sample_id,curves.date)))
    assert len(animals)==30 and len(conditions)==35
    ai={a:i for i,a in enumerate(animals)};ci={c:i for i,c in enumerate(conditions)};ni={n:i for i,n in enumerate(cells)}
    raw=np.full((len(animals),len(conditions),13,40),np.nan)
    raw[[ai[v] for v in curves.animal_id],[ci[v] for v in zip(curves.sample_id,curves.date)],
        [ni[v] for v in curves.neuron_class],:]=curves[[str(t) for t in range(40)]].to_numpy()
    existing=pd.read_csv(paths['conditions'],dtype={'block':str})
    existing=existing.loc[existing.strain.isin(strains)].set_index(['strain','block','cell'])
    fullfit=frozen_condition_fit(raw,templates)
    expected_gated=np.array([[existing.loc[(s,d,c),'coefficient'] for c in cells] for s,d in conditions])
    expected_raw=np.array([[existing.loc[(s,d,c),'raw_coefficient'] for c in cells] for s,d in conditions])
    expected_n=np.array([[existing.loc[(s,d,c),'n_animals'] for c in cells] for s,d in conditions])
    full_errors={'gated_coefficient_max_abs_error':float(np.max(np.abs(expected_gated-fullfit['gated']))),
                 'raw_coefficient_max_abs_error':float(np.max(np.abs(expected_raw-fullfit['pre_gate']))),
                 'animal_count_max_abs_error':int(np.max(np.abs(expected_n-fullfit['n'])))}
    assert max(full_errors.values())<1e-10,full_errors
    date_rows=[];date_pair_rows=[];date_summaries=[]
    repeat_ids=[s for s in strains if len(metadata.loc[s,'dates'].split(';'))==2]
    assert len(repeat_ids)==6
    for representation,column in [('gated','coefficient'),('pre_gate','raw_coefficient')]:
        date_units={};date_metrics={}
        for strain in repeat_ids:
            dates=sorted(metadata.loc[strain,'dates'].split(';'))
            for date in dates:
                coeff=np.array([existing.loc[(strain,date,c),column] for c in cells])
                unit,norm,valid=unit_profiles(coeff[None,:]);assert valid[0]
                measures=metric_arrays(unit,mean,loading,cells)
                date_units[(strain,date)]=unit[0]
                date_metrics[(strain,date)]={k:v[0] for k,v in measures.items()}
                date_rows.append({'representation':representation,'strain':strain,'date':date,
                     'species':metadata.loc[strain,'species'],'taxonomy_note':metadata.loc[strain,'taxonomy_note'],
                     'coefficient_norm':norm[0],'raw_ADF_minus_ASH':coeff[ni['ADF']]-coeff[ni['ASH']],
                     'raw_AWB':coeff[ni['AWB']],**{f'coefficient_{c}':coeff[j] for j,c in enumerate(cells)},
                     **{f'unit_{c}':unit[0,j] for j,c in enumerate(cells)},
                     'fixed_PC1':float(measures['fixed_PC1'][0,0]),'fixed_PC2':float(measures['fixed_PC2'][0,0]),
                     'unit_ADF_minus_ASH':float(measures['unit_ADF_minus_ASH'][0,0]),
                     'unit_AWB':float(measures['unit_AWB'][0,0])})
        for metric in ['unit_13D','fixed_PC12','fixed_PC1','fixed_PC2','unit_ADF_minus_ASH','unit_AWB']:
            early=[];late=[]
            for strain in repeat_ids:
                d1,d2=sorted(metadata.loc[strain,'dates'].split(';'))
                a=date_metrics[(strain,d1)][metric];b=date_metrics[(strain,d2)][metric]
                early.append(a);late.append(b)
                row={'representation':representation,'metric':metric,'strain':strain,'earlier_date':d1,'later_date':d2,
                     'species':metadata.loc[strain,'species'],'taxonomy_note':metadata.loc[strain,'taxonomy_note'],
                     'change_l2':float(np.linalg.norm(b-a)), 'earlier_value':float(a[0]) if len(a)==1 else np.nan,
                     'later_value':float(b[0]) if len(b)==1 else np.nan,
                     'signed_change':float(b[0]-a[0]) if len(a)==1 else np.nan}
                date_pair_rows.append(row)
            early=np.array(early);late=np.array(late);pairmeans=(early+late)/2
            pairs=np.triu_indices(6,1);differences=pairmeans[:,None,:]-pairmeans[None,:,:]
            same=float(np.sqrt(np.mean(np.sum((early-late)**2,axis=1))))
            between=float(np.sqrt(np.mean(np.sum(differences[pairs]**2,axis=1))))
            date_summaries.append({'representation':representation,'metric':metric,'n_repeat_strains':6,
                                   'n_unique_between_mean_pairs':15,'same_RMS':same,
                                   'between_mean_profile_RMS':between,'descriptive_same_to_between_mean_ratio':same/between})
    assignment=pd.read_csv(paths['assignments']);split_ids=sorted(assignment.split.unique());assert len(split_ids)==100
    metrics=[];plane_rows=[];coverages=[];condition_rows=[];half_profiles=[];strain_differences=[]
    for split in split_ids:
        group=assignment[assignment.split.eq(split)].set_index('animal_id')
        assert group.index.is_unique and set(animals).issubset(group.index)
        labels=group.loc[animals,'half'].to_numpy();fits={h:frozen_condition_fit(raw[labels==h],templates) for h in ['A','B']}
        shared=fits['A']['eligible']&fits['B']['eligible']
        aggregate={};units={};norms={};valids={}
        for representation in ['gated','pre_gate']:
            aggregate[representation]={};units[representation]={};norms[representation]={};valids[representation]={}
            for half in ['A','B']:
                coefficients,dates_count,all_original=shared_date_aggregate(fits[half][representation],shared,conditions,strains)
                aggregate[representation][half]=coefficients
                units[representation][half],norms[representation][half],valids[representation][half]=unit_profiles(coefficients)
        joint=np.logical_and.reduce([valids[r][h] for r in ['gated','pre_gate'] for h in ['A','B']])
        joint_ids=np.array(strains)[joint];date_sets=metadata.loc[joint_ids,'dates'].to_numpy()
        gate_disagreement=(fits['A']['passed']!=fits['B']['passed'])&shared
        gate_fraction=float(gate_disagreement.sum()/shared.sum()) if shared.any() else np.nan
        for i,strain in enumerate(strains):
            indices=np.array([s==strain for s,d in conditions])
            n_shared=int(shared[indices].sum());gated_diff=int(gate_disagreement[indices].sum())
            coverages.append({'split':int(split),'strain':strain,'species':metadata.loc[strain,'species'],
                'original_dates':metadata.loc[strain,'dates'],'n_common_observed_cells':int(np.sum(dates_count[i]>0)),
                'all_original_dates_complete13':bool(all_original[i]),'joint_complete13_nonzero':bool(joint[i]),
                'n_shared_date_cell_conditions':n_shared,'n_gate_disagreements':gated_diff,
                'gate_disagreement_fraction':gated_diff/n_shared if n_shared else np.nan,
                **{f'common_date_count_{c}':int(dates_count[i,j]) for j,c in enumerate(cells)},
                **{f'{r}_{h}_norm':float(norms[r][h][i]) for r in ['gated','pre_gate'] for h in ['A','B']}})
        for k,(strain,date) in enumerate(conditions):
            for j,cell in enumerate(cells):
                condition_rows.append({'split':int(split),'strain':strain,'date':date,'cell':cell,
                    'shared_n_ge2':bool(shared[k,j]),'gate_disagrees':bool(gate_disagreement[k,j]),
                    **{f'{h}_{key}':fits[h][key][k,j].item() for h in ['A','B'] for key in ['n','snr','status','projection','gated','pre_gate']}})
        for representation in ['gated','pre_gate']:
            arrays={h:metric_arrays(units[representation][h],mean,loading,cells) for h in ['A','B']}
            for metric in arrays['A']:
                a=arrays['A'][metric][joint];b=arrays['B'][metric][joint]
                result=comparison_metrics(a,b,date_sets)
                metrics.append({'split':int(split),'representation':representation,'metric':metric,
                                'complete_strains':';'.join(joint_ids),
                                'n_all_original_dates_complete13':int(all_original[joint].sum()),
                                'gate_disagreement_fraction':gate_fraction,**result})
                for i,strain in enumerate(joint_ids):
                    strain_differences.append({'split':int(split),'representation':representation,'metric':metric,
                        'strain':strain,'difference_l2':float(np.linalg.norm(a[i]-b[i])),
                        'half_A_value':float(a[i,0]) if a.shape[1]==1 else np.nan,
                        'half_B_value':float(b[i,0]) if b.shape[1]==1 else np.nan})
            plane_rows.append({'split':int(split),'representation':representation,'n_strains':int(joint.sum()),
                 **plane_comparison(units[representation]['A'][joint],units[representation]['B'][joint])})
            for half in ['A','B']:
                for i,strain in enumerate(strains):
                    coef=aggregate[representation][half][i];unit=units[representation][half][i]
                    half_profiles.append({'split':int(split),'representation':representation,'half':half,'strain':strain,
                        'joint_complete13_nonzero':bool(joint[i]),'coefficient_norm':float(norms[representation][half][i]),
                        'raw_ADF_minus_ASH':float(coef[ni['ADF']]-coef[ni['ASH']]),'raw_AWB':float(coef[ni['AWB']]),
                        **{f'coefficient_{c}':float(coef[j]) for j,c in enumerate(cells)},
                        **{f'unit_{c}':float(unit[j]) for j,c in enumerate(cells)},
                        'fixed_PC1':float(arrays[half]['fixed_PC1'][i,0]),'fixed_PC2':float(arrays[half]['fixed_PC2'][i,0]),
                        'unit_ADF_minus_ASH':float(arrays[half]['unit_ADF_minus_ASH'][i,0]),
                        'unit_AWB':float(arrays[half]['unit_AWB'][i,0])})
    metrics=pd.DataFrame(metrics);planes=pd.DataFrame(plane_rows);coverage=pd.DataFrame(coverages)
    differences=pd.DataFrame(strain_differences);half_profiles=pd.DataFrame(half_profiles)
    per_strain=[]
    for (representation,metric,strain),g in differences.groupby(['representation','metric','strain']):
        vals=g.difference_l2.to_numpy()
        per_strain.append({'representation':representation,'metric':metric,'strain':strain,
                           'n_valid_splits':len(vals),'paired_difference_RMS':float(np.sqrt(np.mean(vals**2))),
                           'difference_median':float(np.median(vals)),'difference_q05':float(np.quantile(vals,.05)),
                           'difference_q95':float(np.quantile(vals,.95))})
    summaries=[]
    for (representation,metric),g in metrics.groupby(['representation','metric'],sort=False):
        for field in ['same_RMS','between_RMS','same_to_between_ratio','same_original_date_label_between_RMS','pearson','spearman']:
            summaries.append({'representation':representation,'metric':metric,'quantity':field,
                              **_quantiles(g[field]),'n_strains_min':int(g.n_strains.min()),
                              'n_strains_median':float(g.n_strains.median()),'n_strains_max':int(g.n_strains.max())})
    plane_summary=[]
    for representation,g in planes.groupby('representation'):
        for field in ['angle_small_deg','angle_large_deg','half_A_PC12_variance_ratio','half_B_PC12_variance_ratio']:
            plane_summary.append({'representation':representation,'quantity':field,**_quantiles(g[field])})
    unique_coverage=coverage.groupby('split').agg(n_complete13=('n_common_observed_cells',lambda x:int((x==13).sum())),
        n_joint=('joint_complete13_nonzero','sum'),n_all_original_dates=('all_original_dates_complete13','sum'),
        n_shared_date_cells=('n_shared_date_cell_conditions','sum'),n_gate_disagreements=('n_gate_disagreements','sum'))
    unique_coverage['gate_disagreement_fraction']=unique_coverage.n_gate_disagreements/unique_coverage.n_shared_date_cells
    summary={'analysis_date':'2026-10-03','n_Bacteroides_strains':29,'n_record_dates':5,'n_animals_by_date_worm_id':30,
        'n_strain_date_conditions':35,'n_recorded_animal_strain_exposures':len(curves[['sample_id','date','worm_key']].drop_duplicates()),
        'n_saved_splits':100,'split_quantiles_are':'descriptive sensitivity, not confidence intervals or independent experiments',
        'primary_support':'same date x neuron n>=2 in both halves, common-date equal mean, all13 and nonzero in both representations',
        'fixed_templates':True,'end_to_end_independent_validation':False,'no_chemical_input':True,
        'same_RMS_definition':'sqrt(mean_i ||A_i-B_i||^2)',
        'between_RMS_definition':'sqrt(mean_ordered_i!=j ||A_i-B_j||^2)',
        'support':{col:_quantiles(unique_coverage[col]) for col in unique_coverage},
        'animal_metrics':summaries,'half_plane_metrics':plane_summary,'date_pair_metrics':date_summaries,
        'full_condition_reconstruction_check':full_errors,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    for directory in ['tables','verification']:(output/directory).mkdir(parents=True,exist_ok=True)
    tables=output/'tables'
    frames={'animal_split_metrics':metrics,'animal_half_plane_angles':planes,'animal_strain_coverage':coverage,
        'animal_split_condition_support':pd.DataFrame(condition_rows),'animal_half_profiles':half_profiles,
        'animal_paired_strain_differences':differences,'animal_per_strain_summary':pd.DataFrame(per_strain),
        'animal_metric_summary':pd.DataFrame(summaries),'animal_plane_summary':pd.DataFrame(plane_summary),
        'date_profiles':pd.DataFrame(date_rows),'date_pair_changes':pd.DataFrame(date_pair_rows),
        'date_summary':pd.DataFrame(date_summaries)}
    for name,frame in frames.items():frame.to_csv(tables/f'{name}.csv',index=False)
    unique_coverage.to_csv(tables/'animal_split_coverage_summary.csv')
    metadata.to_csv(tables/'strain_metadata.csv')
    pd.DataFrame({'animal_id':animals}).to_csv(tables/'animal_ids.csv',index=False)
    pd.DataFrame(conditions,columns=['strain','date']).to_csv(tables/'conditions.csv',index=False)
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'verification/full_condition_reconstruction.json').write_text(json.dumps(full_errors,indent=2)+'\n')
    return add_auxiliary_descriptors(output)

def add_auxiliary_descriptors(result_dir):
    """Add the pre-agreed unnormalized descriptors from saved primary profiles only."""
    output=Path(result_dir);t=output/'tables'
    if (t/'animal_auxiliary_metric_summary.csv').exists():
        raise FileExistsError('Auxiliary descriptors already saved; refuse overwrite.')
    half=pd.read_csv(t/'animal_half_profiles.csv')
    dates=pd.read_csv(t/'date_profiles.csv',dtype={'date':str})
    metadata=pd.read_csv(t/'strain_metadata.csv',index_col='strain',dtype={'dates':str})
    lookup={'coefficient_ADF_minus_ASH':'raw_ADF_minus_ASH','coefficient_AWB':'raw_AWB','coefficient_norm':'coefficient_norm'}
    rows=[];date_summary=[];date_changes=[]
    for (split,representation),g in half.groupby(['split','representation']):
        selected=g[g.joint_complete13_nonzero]
        a=selected[selected.half.eq('A')].set_index('strain').sort_index()
        b=selected[selected.half.eq('B')].set_index('strain').loc[a.index]
        assert a.index.equals(b.index)
        sets=metadata.loc[a.index,'dates'].to_numpy()
        for metric,column in lookup.items():
            rows.append({'split':int(split),'representation':representation,'metric':metric,
                         'units':'delta_F_over_F0 coefficient space',
                         **comparison_metrics(a[[column]].to_numpy(),b[[column]].to_numpy(),sets)})
    for representation,g in dates.groupby('representation'):
        for metric,column in lookup.items():
            early=[];late=[]
            for strain,pair in g.groupby('strain'):
                pair=pair.sort_values('date');assert len(pair)==2
                a,b=pair[column].to_numpy();early.append(a);late.append(b)
                date_changes.append({'representation':representation,'metric':metric,'strain':strain,
                                     'earlier_date':pair.date.iloc[0],'later_date':pair.date.iloc[1],
                                     'earlier_value':a,'later_value':b,'signed_change':b-a})
            early,late=np.array(early),np.array(late);means=(early+late)/2
            pairs=np.triu_indices(6,1)
            same=float(np.sqrt(np.mean((late-early)**2)))
            between=float(np.sqrt(np.mean((means[:,None]-means[None,:])[pairs]**2)))
            date_summary.append({'representation':representation,'metric':metric,'n_repeat_strains':6,
                                 'units':'delta_F_over_F0 coefficient space','n_unique_between_mean_pairs':15,
                                 'same_RMS':same,'between_mean_profile_RMS':between,
                                 'descriptive_same_to_between_mean_ratio':same/between,
                                 'pearson':float(np.corrcoef(early,late)[0,1]),
                                 'spearman':float(spearmanr(early,late).statistic)})
    frame=pd.DataFrame(rows);summaries=[]
    for (representation,metric),g in frame.groupby(['representation','metric']):
        for field in ['same_RMS','between_RMS','same_to_between_ratio','same_original_date_label_between_RMS','pearson','spearman']:
            summaries.append({'representation':representation,'metric':metric,'quantity':field,**_quantiles(g[field]),
                              'n_strains_min':int(g.n_strains.min()),'n_strains_median':float(g.n_strains.median()),
                              'n_strains_max':int(g.n_strains.max())})
    frame.to_csv(t/'animal_auxiliary_split_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(t/'animal_auxiliary_metric_summary.csv',index=False)
    pd.DataFrame(date_summary).to_csv(t/'date_auxiliary_summary.csv',index=False)
    pd.DataFrame(date_changes).to_csv(t/'date_auxiliary_pair_changes.csv',index=False)
    summary=json.loads((output/'summary.json').read_text())
    summary['animal_unnormalized_auxiliary_metrics']=summaries
    summary['date_unnormalized_auxiliary_metrics']=date_summary
    counts=half[half.representation.eq('gated')&half.half.eq('A')].groupby('strain').joint_complete13_nonzero.sum()
    metadata.join(counts.rename('n_valid_splits')).to_csv(t/'animal_per_strain_coverage_summary.csv')
    summary['strains_never_joint_complete13']=counts.index[counts.eq(0)].tolist()
    summary['n_unique_strains_ever_joint_complete13']=int(counts.gt(0).sum())
    summary['auxiliary_interpretation']='Unnormalized descriptors use the identical complete13 cohort. Do not compare their RMS magnitudes numerically with unit-coordinate RMS; compare patterns, same/between ratios and correlations.'
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def load_saved_results(result_dir):
    """Read-only default for an existing Notebook."""
    result_dir=Path(result_dir)
    return {'summary':json.loads((result_dir/'summary.json').read_text()),
            'metrics':pd.read_csv(result_dir/'tables/animal_metric_summary.csv'),
            'date_summary':pd.read_csv(result_dir/'tables/date_summary.csv'),
            'plane_summary':pd.read_csv(result_dir/'tables/animal_plane_summary.csv')}


if __name__=='__main__':
    result=run_analysis(Path(__file__).resolve().parents[4])
    print(json.dumps({'support':result['support'],'plane':result['half_plane_metrics'],
                      'date':result['date_pair_metrics'],'full_condition_check':result['full_condition_reconstruction_check']},indent=2))
