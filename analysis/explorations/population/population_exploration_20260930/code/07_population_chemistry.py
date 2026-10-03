"""Can chemical composition predict cell-selective response patterns?

Strain-linked chemistry is tested against animal-centered neural responses.
Outer and inner splits hold out entire sampling blocks, excluding all six
cross-block strains globally. Scaling, group coherence selection and ridge
penalties are learned within training splits. No distance-matrix correlation,
no trial-level biological replication, no molecular causal inference.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
OLD=ROOT/'reports/exploration_20260929/tables'
SEED=2026093007
LAMBDAS=[.01,.1,1.,10.]
WINDOWS=['stim','full']
NEURONS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
MODELS=['level','taxonomy','groups','taxonomy_groups','panel','taxonomy_panel']


def chemical_design(log,meta,tax,ref,members,train_ids):
    names=meta.index[meta.complete_eligible].tolist()
    train=log.loc[train_ids,names]
    z=(log[names]-train.mean())/train.std(ddof=1).clip(lower=1e-6)
    # In each fold repeat chemical-only coherence selection from annotation
    # memberships, not the seven groups selected using all 106 strains.
    selected=[]
    scores={}
    for gid,g in members[members.kind.eq('annotation')].groupby('group_id'):
        ff=g.feature.tolist()
        if not 3<=len(ff)<=12:
            continue
        c=train[ff].corr(method='spearman').to_numpy()
        pairs=c[np.triu_indices(len(ff),1)]
        rest=[spearmanr(z.loc[train_ids,f],z.loc[train_ids,[c for c in ff if c!=f]].median(axis=1)).statistic for f in ff]
        if np.median(pairs)>=.20 and np.mean(pairs>0)>=.85 and min(rest)>=.10:
            selected.append(gid)
            scores[gid]=z[ff].median(axis=1)
    groups=pd.DataFrame(scores,index=log.index)
    # Include both report-unit and feature-standardized overall levels: these
    # are panel summaries, not total molecular concentrations.
    level=pd.DataFrame({'median_log':log[names].median(axis=1),'median_z':z.median(axis=1)})
    level=(level-level.loc[train_ids].mean())/level.loc[train_ids].std(ddof=1).clip(lower=1e-6)
    references=pd.get_dummies(ref.reference_group,prefix='ref',dtype=float)
    genera=pd.get_dummies(tax.genus_clean,prefix='genus',dtype=float)
    # Remove unseen category columns: no test response or coefficient is used.
    references=references.loc[:,references.loc[train_ids].sum().gt(0)]
    genera=genera.loc[:,genera.loc[train_ids].sum().gt(0)]
    base=pd.concat([level,references],axis=1)
    return dict(level=base,taxonomy=pd.concat([base,genera],axis=1),
        groups=pd.concat([base,groups],axis=1),taxonomy_groups=pd.concat([base,genera,groups],axis=1),
        panel=pd.concat([base,z],axis=1),taxonomy_panel=pd.concat([base,genera,z],axis=1)),selected


def centered_rows(data,x):
    """Retain one row per measured animal-strain-cell, with both phase means."""
    xx=x.loc[data.sample_id].reset_index(drop=True)
    block=data[['date','worm_key']].astype(str).agg('/'.join,axis=1).to_numpy()
    yy=data[WINDOWS].reset_index(drop=True)
    xx=xx-xx.groupby(block).transform('mean')
    yy=yy-yy.groupby(block).transform('mean')
    weight=1/data.groupby(['sample_id','date']).sample_id.transform('size').to_numpy()
    return xx.to_numpy(),yy.to_numpy(),weight


def predict_fold(train,test,designs,lambdas):
    predictions=[]
    for neuron in NEURONS:
        tr=train[train.neuron_class.eq(neuron)].reset_index(drop=True)
        te=test[test.neuron_class.eq(neuron)].reset_index(drop=True)
        assert len(tr)>0 and len(te)>0
        for name,x in designs.items():
            X,Y,w=centered_rows(tr,x)
            V,Z,v=centered_rows(te,x)
            sw=np.sqrt(w)
            gram=(X*sw[:,None]).T@(X*sw[:,None])/w.sum()
            xy=(X*w[:,None]).T@Y/w.sum()
            ev,U=np.linalg.eigh(gram)
            ev=np.maximum(ev,0)
            proj=U.T@xy
            yscale=np.sqrt(np.sum(w[:,None]*Y**2,axis=0)/w.sum()).clip(1e-6)
            for lam in lambdas:
                coef=U@(proj/(ev[:,None]+lam))
                pred=V@coef
                for j,window in enumerate(WINDOWS):
                    frame=te[['sample_id','date','worm_key']].copy()
                    frame['neuron_class']=neuron;frame['window']=window;frame['model']=name;frame['lambda']=lam
                    frame['observed']=Z[:,j];frame['predicted']=pred[:,j];frame['weight']=v
                    frame['train_scale']=yscale[j]
                    predictions.append(frame)
    return pd.concat(predictions,ignore_index=True)


def losses(pred):
    p=pred.copy()
    p['loss']=p.weight*((p.observed-p.predicted)/p.train_scale)**2
    p['null']=p.weight*(p.observed/p.train_scale)**2
    # First average each held-out block x cell x phase; each component gets
    # equal weight when selecting a shared ridge penalty across the population.
    per=p.groupby(['model','lambda','date','neuron_class','window']).agg(
        loss=('loss','sum'),null=('null','sum'),weight=('weight','sum'))
    per['mse']=per.loss/per.weight
    return per.reset_index()


def main():
    a=pd.read_csv(OLD/'animal_metrics.csv')
    log=pd.read_csv(OLD/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(OLD/'chemical_feature_metadata.csv',index_col=0)
    tax=pd.read_csv(OLD/'taxonomy.csv',index_col=0)
    ref=pd.read_csv(OLD/'chemical_reference_groups.csv',index_col=0)
    members=pd.read_csv(OUT/'tables/chemical_groups_members.csv')
    repeat=a.groupby('sample_id').date.nunique()
    excluded=repeat[repeat>1].index.tolist()
    a=a[~a.sample_id.isin(excluded)].copy()
    dates=sorted(a.date.unique())
    # Cache training/test partitions and transforms shared between outer folds.
    inner_cache={};group_records=[]
    for excluded_date in dates:
        for val_date in dates:
            if excluded_date==val_date:
                continue
            tr=a[~a.date.isin([excluded_date,val_date])];te=a[a.date.eq(val_date)]
            ids=sorted(tr.sample_id.unique())
            designs,selected=chemical_design(log,meta,tax,ref,members,ids)
            pred=predict_fold(tr,te,designs,LAMBDAS)
            inner_cache[(excluded_date,val_date)]=losses(pred)
            group_records.append(dict(outer_test=excluded_date,inner_test=val_date,selected_groups=';'.join(selected),n_train_strains=len(ids)))
        print(f'Inner folds complete for held-out block {excluded_date}',flush=True)
    final=[];choices=[]
    for test_date in dates:
        inner=pd.concat([inner_cache[(test_date,v)] for v in dates if v!=test_date])
        risks=inner.groupby(['model','lambda']).mse.mean()
        chosen={m:float(risks.loc[m].idxmin()) for m in MODELS}
        tr=a[~a.date.eq(test_date)];te=a[a.date.eq(test_date)]
        assert not(set(tr.sample_id)&set(te.sample_id))
        assert not(set(zip(tr.date,tr.worm_key))&set(zip(te.date,te.worm_key)))
        designs,selected=chemical_design(log,meta,tax,ref,members,sorted(tr.sample_id.unique()))
        for model in MODELS:
            pred=predict_fold(tr,te,{model:designs[model]},[chosen[model]])
            final.append(pred)
            choices.append(dict(test_date=test_date,model=model,lambda_chosen=chosen[model],
                n_train_strains=tr.sample_id.nunique(),n_test_strains=te.sample_id.nunique(),
                n_features=designs[model].shape[1],selected_groups=';'.join(selected)))
    predictions=pd.concat(final,ignore_index=True)
    predictions.to_csv(OUT/'tables/chemistry_population_predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(OUT/'tables/chemistry_population_choices.csv',index=False)
    pd.DataFrame(group_records).to_csv(OUT/'tables/chemistry_population_inner_groups.csv',index=False)
    per=losses(predictions)
    per['relative_r2']=1-per.loss/per['null']
    per.to_csv(OUT/'tables/chemistry_population_block_results.csv',index=False)
    result=per.groupby(['model','neuron_class','window']).agg(loss=('loss','sum'),null=('null','sum'),
        improving_blocks=('relative_r2',lambda x:int((x>0).sum())),median_block_r2=('relative_r2','median')).reset_index()
    result['relative_r2']=1-result.loss/result['null']
    result.to_csv(OUT/'tables/chemistry_population_neuron_results.csv',index=False)
    # These scores use per-training-fold response scales, including the
    # per-cell scores above. Raw-unit and strain-mean scores are exported by
    # 09_synthesis.py, so low overall R2 is not mistaken for the proportion of
    # stable strain-mean variance explained.
    overall=per.groupby(['model','window']).agg(loss=('loss','sum'),null=('null','sum')).reset_index()
    overall['relative_r2']=1-overall.loss/overall['null']
    overall.to_csv(OUT/'tables/chemistry_population_summary.csv',index=False)
    (OUT/'logs/chemistry_population_methods.json').write_text(json.dumps(dict(
        seed=SEED,n_strains=a.sample_id.nunique(),n_animals=a[['date','worm_key']].drop_duplicates().shape[0],
        excluded_repeated_strains=excluded,outer_folds=len(dates),inner_folds=len(dates)-1,
        lambdas=LAMBDAS,windows=WINDOWS,targets=NEURONS,
        units='observed/predicted in animal-centered stored delta_F/F0',
        weights='each measured strain per neuron per block has total weight one',
        reference='zero relative response within each tested animal',
        evaluation='all test stimuli used only to center measured/predicted values; relative-set prediction, not absolute single-exposure prediction',
        inference='exploratory blocked prediction; no independent experiment and no confirmatory p values',
        limitations=['report profiles paired by strain ID only','QC panel fixed before neural modeling',
            'scheduling/carryover not identified','13 equal standardized response dimensions include weak signals']),indent=2))
    print(overall.round(4).to_string(index=False))
    print(result.pivot(index=['neuron_class','window'],columns='model',values='relative_r2').round(3).to_string())


if __name__=='__main__':
    main()
