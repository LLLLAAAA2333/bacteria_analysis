"""Bounded follow-up: learn chemical effects from within-genus contrasts only.

Triggered by failure of across-genus-trained models on within-genus outcomes.
Targets locked to ASK and the ASI/ASJ mean; no per-molecule selection.
"""
from pathlib import Path
import importlib.util
import json
import pandas as pd
import numpy as np

OUT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('targeted',OUT/'code/04_targeted_chemistry.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
model.CELLS=['ASK','ASI_ASJ_mean']


def arrays_within(rows,dm):
    group=rows[['date','worm_key','genus_clean']].astype(str).agg('/'.join,axis=1).to_numpy()
    x=dm.loc[rows.sample_id].reset_index(drop=True);y=rows[model.WINDOWS].reset_index(drop=True)
    x-=x.groupby(group).transform('mean');y-=y.groupby(group).transform('mean')
    weight=1/rows.groupby(['date','sample_id']).sample_id.transform('size').to_numpy()
    return x.to_numpy(),y.to_numpy(),weight


def main():
    a=pd.read_csv(model.SOURCE/'animal_metrics.csv')
    repeat=a.groupby('sample_id').date.nunique();a=a[~a.sample_id.isin(repeat[repeat>1].index)]
    tax=pd.read_csv(model.SOURCE/'taxonomy.csv',index_col=0)
    idx=['date','worm_key','sample_id']
    paired=a[a.neuron_class.isin(['ASI','ASJ'])].pivot(index=idx,columns='neuron_class',values=['stim','post']).dropna()
    axis=pd.DataFrame({w:paired[w].mean(axis=1) for w in model.WINDOWS}).reset_index()
    axis['neuron_class']='ASI_ASJ_mean'
    a=pd.concat([a[a.neuron_class.eq('ASK')][idx+['neuron_class','stim','post']],axis],ignore_index=True)
    a=a.merge(tax[['genus_clean']],left_on='sample_id',right_index=True,validate='many_to_one')
    group=['date','worm_key','neuron_class','genus_clean']
    a=a[a.groupby(group).sample_id.transform('nunique')>=2].copy()
    chem=pd.read_csv(model.SOURCE/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(model.SOURCE/'chemical_feature_metadata.csv',index_col=0)
    ref=pd.read_csv(model.SOURCE/'chemical_reference_groups.csv',index_col=0)
    model.arrays=arrays_within
    blocks=sorted(a.date.unique());predictions=[];choices=[]
    for held in blocks:
        inner=[]
        for val in blocks:
            if val==held:continue
            tr=a[~a.date.isin([held,val])];te=a[a.date.eq(val)]
            dm=model.design(chem,meta,tax,ref,sorted(tr.sample_id.unique()))
            dm={k:dm[k] for k in ['level','panel162','relative162']}
            risk,_=model.fit_predict(tr,te,dm);risk['inner_block']=val;inner.append(risk)
        risks=pd.concat(inner).groupby(['model','neuron_class','window','penalty']).mse.mean().reset_index()
        choice=risks.loc[risks.groupby(['model','neuron_class','window']).mse.idxmin()].copy()
        choice['outer_block']=held;choices.append(choice)
        tr=a[~a.date.eq(held)];te=a[a.date.eq(held)]
        assert not(set(tr.sample_id)&set(te.sample_id))
        dm=model.design(chem,meta,tax,ref,sorted(tr.sample_id.unique()))
        dm={k:dm[k] for k in ['level','panel162','relative162']}
        _,pred=model.fit_predict(tr,te,dm,output=True)
        pred=pred.merge(choice[['model','neuron_class','window','penalty']],on=['model','neuron_class','window','penalty'])
        predictions.append(pred)
        print('within-genus holdout',held,flush=True)
    pred=pd.concat(predictions)
    pred.to_csv(OUT/'tables/within_genus_chemistry_predictions.csv',index=False)
    pd.concat(choices).to_csv(OUT/'tables/within_genus_chemistry_choices.csv',index=False)
    block,summary,mean=model.summarize(pred)
    summary.to_csv(OUT/'tables/within_genus_chemistry_scores.csv',index=False)
    block.to_csv(OUT/'tables/within_genus_chemistry_block_scores.csv',index=False)
    mean.to_csv(OUT/'tables/within_genus_chemistry_strain_predictions.csv',index=False)
    (OUT/'logs/within_genus_chemistry_methods.json').write_text(json.dumps(dict(
        rationale='Across-genus-trained predictions failed within-genus contrasts; train directly on those contrasts.',
        targets=model.CELLS,windows=model.WINDOWS,models=['level','panel162','relative162'],
        n_strains=a.sample_id.nunique(),n_animals=len(a[idx[:2]].drop_duplicates()),n_blocks=len(blocks),
        centering='within each animal x genus x neuron; require >=2 strains, no between-genus response variation used',
        tuning='nested blocked per-target ridge as in 04; exploratory follow-up, no independent validation'),indent=2))
    print(summary.round(4).to_string(index=False))


if __name__=='__main__':main()
