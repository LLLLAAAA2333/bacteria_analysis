"""Adversarial checks for population identification, not independent validation."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
T=ROOT/'reports/exploration_20260929/tables'
spec=importlib.util.spec_from_file_location('information',OUT/'code/05_population_information.py')
info=importlib.util.module_from_spec(spec)
spec.loader.exec_module(info)
NEURONS=info.NEURONS


def array(d,animals,strains,features):
    result=np.full((len(animals),len(strains),len(features)),np.nan)
    for i,animal in enumerate(animals):
        s=d[d.worm_key.eq(animal)].set_index(['sample_id','neuron_class'])
        for j,(w,n) in enumerate(features):
            result[i,:,j]=s[w].reindex(pd.MultiIndex.from_product([strains,[n]])).to_numpy()
    return result


def main():
    a=pd.read_csv(T/'animal_metrics.csv')
    tr=pd.read_parquet(T/'trial_curves.parquet').reset_index()
    tr['date']=tr.date.astype(int)
    key=['sample_id','date','worm_key','neuron_class']
    tr['rank']=tr.groupby(key).segment_index.rank(method='first')
    for w,(lo,hi) in {'stim':(0,10),'post':(10,30),'full':(0,40)}.items():
        tr[w]=tr[[str(t) for t in range(lo,hi)]].mean(axis=1)
    first=tr[tr['rank'].eq(1)].groupby(key,as_index=False)[['stim','post','full']].mean()
    later=tr[tr['rank'].gt(1)].groupby(key,as_index=False)[['stim','post','full']].mean()
    # Remove a linear association with the mean retained-trial position within
    # each animal and neuron. This can remove true signal correlated with order
    # and cannot identify arbitrary carryover or nonlinear order effects.
    order=tr.groupby(key,as_index=False).segment_index.mean()
    detrended=a.merge(order,on=key,validate='one_to_one')
    for _,indices in detrended.groupby(['date','worm_key','neuron_class']).groups.items():
        sub=detrended.loc[indices]
        x=sub.segment_index.to_numpy();x=x-x.mean()
        z=sub[['stim','post','full']].to_numpy()
        if x@x>0:
            detrended.loc[indices,['stim','post','full']]=z-x[:,None]*(x@z/(x@x))
    modes=[('first_to_later',first,later),('later_to_first',later,first),
           ('linear_order_removed',detrended,detrended),('common_cells',a,a),('without_AWCON',a,a)]
    records=[]
    for variant,train_data,test_data in modes:
        for w in ['stim','both','full']:
            windows=['stim','post'] if w=='both' else [w]
            neurons=[n for n in NEURONS if variant!='without_AWCON' or n!='AWCON']
            features=[(p,n) for p in windows for n in neurons]
            for date,d in train_data.groupby('date'):
                dt=test_data[test_data.date.eq(date)]
                animals=sorted(set(d.worm_key)&set(dt.worm_key));strains=sorted(set(d.sample_id)&set(dt.sample_id))
                train_values=array(d,animals,strains,features);test_values=array(dt,animals,strains,features)
                for i,animal in enumerate(animals):
                    train=np.delete(train_values,i,axis=0);test=test_values[i]
                    # Some animals have only one retained trial for a strain.
                    # Later-trial evaluation cannot invent that missing row.
                    row_mask=np.isfinite(test).any(axis=1)
                    train=train[:,row_mask,:];test=test[row_mask]
                    fold_strains=np.array(strains)[row_mask]
                    mask=np.isfinite(test).all(axis=0)&(np.isfinite(train).sum(axis=0)>=2).all(axis=0)
                    if variant=='common_cells':
                        mask &= np.isfinite(train).all(axis=(0,1))
                    cols=np.flatnonzero(mask)
                    assert len(cols)>0, (variant,w,date,animal,len(fold_strains))
                    for mode in ['raw','scaled','shape','raw_shape','magnitude','scaled_magnitude']:
                        pred=info.predict(train,test,fold_strains,cols,mode)
                        for s,p in zip(fold_strains,pred):
                            records.append(dict(variant=variant,window=w,date=date,worm_key=animal,
                                mode=mode,sample_id=s,prediction=p,correct=int(p==s),n_features=len(cols),
                                n_stimuli=len(fold_strains),chance=1/len(fold_strains)))
    results=pd.DataFrame(records)
    results.to_csv(OUT/'tables/information_checks_predictions.csv',index=False)
    animal=results.groupby(['variant','window','mode','date','worm_key'],as_index=False).agg(
        accuracy=('correct','mean'),n_features=('n_features','first'),n_stimuli=('n_stimuli','first'),chance=('chance','first'))
    animal.to_csv(OUT/'tables/information_checks_animals.csv',index=False)
    summary=animal.groupby(['variant','window','mode'],as_index=False).agg(
        accuracy=('accuracy','mean'),n_animals=('accuracy','size'),min_features=('n_features','min'),chance=('chance','mean'),
        min_stimuli=('n_stimuli','min'),max_stimuli=('n_stimuli','max'))
    summary.to_csv(OUT/'tables/information_checks_summary.csv',index=False)
    # Describe order identity directly; no test of randomization is inferred.
    first_order=tr.groupby(['date','worm_key','sample_id']).segment_index.min().rename('first_segment').reset_index()
    first_order['stimulus_rank']=first_order.groupby(['date','worm_key']).first_segment.rank(method='dense')
    first_order.to_csv(OUT/'tables/information_stimulus_order.csv',index=False)
    repeated_rank=first_order.groupby(['date','sample_id']).stimulus_rank.nunique()
    (OUT/'logs/information_checks.json').write_text(json.dumps(dict(
        strain_blocks_with_same_first_rank=int(repeated_rank.eq(1).sum()),n_strain_blocks=len(repeated_rank),
        limit='Shared stimulus schedules/carryover remain confounded; linear detrending is not identification.',
        tested='cross-animal first/later transfer, linear position removal, common cell coverage, AWCON removal'),indent=2))
    print(summary.round(4).to_string(index=False))


if __name__=='__main__':
    main()
