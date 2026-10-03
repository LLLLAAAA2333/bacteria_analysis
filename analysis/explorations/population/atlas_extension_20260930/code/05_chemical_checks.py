"""Stress tests of held-out chemical predictions; no further model search."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SOURCE=ROOT/'reports/exploration_20260929/tables'


def scores(p, scope):
    p=p.copy()
    # Each measured strain contributes once after averaging biological animals.
    q=p.groupby(['model','neuron_class','window','date','sample_id'],as_index=False)[['observed','predicted']].mean()
    q['loss']=(q.observed-q.predicted)**2;q['null']=q.observed**2
    rows=[]
    for key,d in q.groupby(['model','neuron_class','window']):
        total_l=d.loss.sum();total_n=d['null'].sum()
        b=d.groupby('date')[['loss','null']].sum()
        reduced=1-(total_l-b.loss)/(total_n-b['null'])
        rows.append(dict(zip(['model','neuron_class','window'],key),scope=scope,
                    r2=1-total_l/total_n,n_strains=d.sample_id.nunique(),n_blocks=len(b),
                    min_leave_block_r2=reduced.min(),max_leave_block_r2=reduced.max(),
                    positive_blocks=int((b.loss<b['null']).sum()),
                    min_leave_strain_r2=(1-(total_l-d.loss)/(total_n-d['null'])).min(),
                    max_leave_strain_r2=(1-(total_l-d.loss)/(total_n-d['null'])).max()))
    return pd.DataFrame(rows),q


def main():
    pred=pd.read_csv(OUT/'tables/targeted_chemistry_predictions.csv')
    tax=pd.read_csv(SOURCE/'taxonomy.csv')[['sample_id','genus_clean']]
    pred=pred.merge(tax,on='sample_id',validate='many_to_one')
    # Audit centering and biological weights independently of training code.
    keys=['model','window','neuron_class','date','worm_key']
    assert pred.groupby(keys)[['observed','predicted']].mean().abs().max().max()<1e-12
    assert np.max(np.abs(pred.groupby(['model','window','neuron_class','date','sample_id']).weight.sum()-1))<1e-12
    summary,means=scores(pred,'all_strains')
    # Competing explanation: within the same animal and genus, chemistry must
    # rank strains beyond the shared genus. Groups of one have no comparison.
    key=keys+['genus_clean']
    within=pred[pred.groupby(key).sample_id.transform('nunique')>=2].copy()
    for col in ['observed','predicted']:
        within[col]-=within.groupby(key)[col].transform('mean')
    sw,mw=scores(within,'within_animal_genus')
    pd.concat([summary,sw]).to_csv(OUT/'tables/chemical_checks_scores.csv',index=False)
    mw.to_csv(OUT/'tables/chemical_checks_within_genus_predictions.csv',index=False)
    # A data-discovered co-modulation phenotype, selected from neural evidence:
    # equal physical-unit mean of ASI and ASJ, only co-measured animal-strains.
    ix=['model','window','date','worm_key','sample_id','genus_clean']
    joint=pred[pred.neuron_class.isin(['ASI','ASJ'])].pivot(index=ix,columns='neuron_class',values=['observed','predicted']).dropna()
    axis=pd.DataFrame({'observed':joint['observed'].mean(axis=1),'predicted':joint['predicted'].mean(axis=1)}).reset_index()
    axis['neuron_class']='ASI_ASJ_mean'
    sa,ma=scores(axis,'co_measured_ASI_ASJ')
    axis.to_csv(OUT/'tables/chemical_checks_axis_predictions.csv',index=False)
    ka=['model','window','date','worm_key','genus_clean']
    aw=axis[axis.groupby(ka).sample_id.transform('nunique')>=2].copy()
    for col in ['observed','predicted']:
        aw[col]-=aw.groupby(ka)[col].transform('mean')
    saw,maw=scores(aw,'co_measured_ASI_ASJ_within_genus')
    pd.concat([sa,saw]).to_csv(OUT/'tables/chemical_checks_axis_scores.csv',index=False)
    # Trial repeat falsification: use unchanged out-of-fold predictions against
    # first and later trials. Not an independent experiment or new tuning set.
    trials=pd.read_parquet(SOURCE/'trial_curves.parquet').reset_index()
    trials['date']=trials.date.astype(int)
    trialkey=['date','worm_key','sample_id','neuron_class']
    trials['first']=trials.segment_index.eq(trials.groupby(trialkey).segment_index.transform('min'))
    for phase,lo,hi in [('stim',0,10),('post',10,30)]:
        trials[phase]=trials[[str(t) for t in range(lo,hi)]].mean(axis=1)
    checks=[]
    # Retain only stimulus sets with both first and later for all their strains.
    common=trials.groupby(trialkey)['first'].agg(lambda v: (~v).any()).rename('has_later').reset_index()
    common=common[common.groupby(['date','worm_key','neuron_class']).has_later.transform('all')]
    for variant,mask in [('first',trials['first']),('later',~trials['first'])]:
        target=trials[mask].groupby(trialkey)[['stim','post']].mean().reset_index().merge(common[trialkey],on=trialkey)
        target=target.melt(id_vars=trialkey,value_vars=['stim','post'],var_name='window',value_name='target')
        q=pred.merge(target,on=trialkey+['window'],how='inner',validate='many_to_one')
        # Restrict the source-prediction model to the same candidate set if
        # coverage excludes anything, then center both on that measured set.
        q['observed']=q.target-q.groupby(keys).target.transform('mean')
        q['predicted']-=q.groupby(keys).predicted.transform('mean')
        s,_=scores(q,variant+'_trial_target');checks.append(s)
    pd.concat(checks).to_csv(OUT/'tables/chemical_checks_trial_scores.csv',index=False)
    (OUT/'logs/chemical_checks_methods.json').write_text(json.dumps(dict(
        scopes=['all-strain held-out response','within animal and genus','co-measured ASI+ASJ mean'],
        sensitivity='drop each evaluation block or strain without refitting; fixed predictions vs first/later targets',
        inference='deletion ranges are sensitivity ranges, not confidence intervals; target axis is exploratory',
        n_models=pred.model.nunique(),checks_passed=['animal centering','strain-cell weight sums']),indent=2))
    print(pd.concat([summary,sw]).query("neuron_class in ['ASK','ASJ','AWA','ASI','AWCON','ASH'] and model in ['panel162','panel248','taxonomy']").round(3).to_string(index=False))
    print(sa.round(3).to_string(index=False));print(saw.round(3).to_string(index=False))


if __name__=='__main__':
    main()
