"""Nested blocked prediction of 13 response channels, with target-specific tuning.

Observation: animal x strain x cell. Trials already averaged; all test animals
and strains excluded from training. Predict within-animal contrasts, not an
absolute response to an isolated exposure. No confirmatory significance tests.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT / 'reports/exploration_20260929/tables'
CELLS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
WINDOWS = ['stim','post']
PENALTIES = np.array([.01,.1,1.,10.,100.])
MODELS = ['level','taxonomy','panel162','relative162','panel248','taxonomy_panel162']
SEED = 2026093014


def design(chem, meta, tax, ref, ids):
    panels = {}
    for key, flag in [('162','complete_eligible'),('248','primary_eligible')]:
        features = meta.index[meta[flag]].tolist()
        raw = chem[features].fillna(chem.loc[ids,features].median())
        panels[key] = (raw-raw.loc[ids].mean())/raw.loc[ids].std().clip(lower=1e-6)
    z = panels['162']
    lev = pd.DataFrame({'median_log':chem[z.columns].median(axis=1),
                        'median_z':z.median(axis=1)})
    lev = (lev-lev.loc[ids].mean())/lev.loc[ids].std().clip(lower=1e-6)
    r = pd.get_dummies(ref.reference_group,prefix='ref',dtype=float)
    g = pd.get_dummies(tax.genus_clean,prefix='genus',dtype=float)
    r = r.loc[:,r.loc[ids].sum()>0]
    g = g.loc[:,g.loc[ids].sum()>0]
    base = pd.concat([lev,r],axis=1)
    return {'level':base,'taxonomy':pd.concat([base,g],axis=1),
            'panel162':pd.concat([base,z],axis=1),
            'relative162':pd.concat([base,z.sub(z.median(axis=1),axis=0)],axis=1),
            'panel248':pd.concat([base,panels['248']],axis=1),
            'taxonomy_panel162':pd.concat([base,g,z],axis=1)}


def arrays(rows, design_matrix):
    group = rows[['date','worm_key']].astype(str).agg('/'.join,axis=1).to_numpy()
    x = design_matrix.loc[rows.sample_id].reset_index(drop=True)
    y = rows[WINDOWS].reset_index(drop=True)
    x = x-x.groupby(group).transform('mean')
    y = y-y.groupby(group).transform('mean')
    w = 1/rows.groupby(['date','sample_id']).sample_id.transform('size').to_numpy()
    return x.to_numpy(),y.to_numpy(),w


def fit_predict(train, test, designs, penalties=PENALTIES, output=False):
    scores, predictions = [], []
    for cell in CELLS:
        tr = train[train.neuron_class.eq(cell)].reset_index(drop=True)
        te = test[test.neuron_class.eq(cell)].reset_index(drop=True)
        for model,dm in designs.items():
            x,y,w = arrays(tr,dm)
            v,z,q = arrays(te,dm)
            gram = (x*w[:,None]).T@x/w.sum()
            xy = (x*w[:,None]).T@y/w.sum()
            ev,u = np.linalg.eigh(gram)
            ev = np.maximum(ev,0)
            uy = u.T@xy
            for penalty in penalties:
                coef = u@(uy/(ev[:,None]+penalty))
                pred = v@coef
                for j,window in enumerate(WINDOWS):
                    scores.append(dict(model=model,neuron_class=cell,window=window,penalty=penalty,
                                       mse=np.sum(q*(z[:,j]-pred[:,j])**2)/q.sum()))
                    if output:
                        f = te[['date','worm_key','sample_id']].copy()
                        f['model']=model;f['neuron_class']=cell;f['window']=window;f['penalty']=penalty
                        f['observed']=z[:,j];f['predicted']=pred[:,j];f['weight']=q
                        predictions.append(f)
    return pd.DataFrame(scores), pd.concat(predictions,ignore_index=True) if output else None


def summarize(pred):
    a = pred.copy()
    a['loss'] = a.weight*(a.observed-a.predicted)**2
    a['null'] = a.weight*a.observed**2
    block = a.groupby(['model','neuron_class','window','date']).agg(loss=('loss','sum'),null=('null','sum')).reset_index()
    block['r2'] = 1-block.loss/block['null']
    summary = block.groupby(['model','neuron_class','window']).agg(loss=('loss','sum'),null=('null','sum'),
                        improving_blocks=('r2',lambda v:int((v>0).sum())),median_block_r2=('r2','median')).reset_index()
    summary['r2_animal'] = 1-summary.loss/summary['null']
    mean = a.groupby(['model','neuron_class','window','date','sample_id']).agg(observed=('observed','mean'),predicted=('predicted','mean'),n_animals=('worm_key','nunique')).reset_index()
    mean['loss']=(mean.observed-mean.predicted)**2;mean['null']=mean.observed**2
    s = mean.groupby(['model','neuron_class','window']).agg(loss=('loss','sum'),null=('null','sum')).reset_index()
    s['r2_strain_mean']=1-s.loss/s['null']
    summary=summary.merge(s.drop(columns=['loss','null']),on=['model','neuron_class','window'])
    return block,summary,mean


def main():
    a=pd.read_csv(SOURCE/'animal_metrics.csv')
    chem=pd.read_csv(SOURCE/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(SOURCE/'chemical_feature_metadata.csv',index_col=0)
    tax=pd.read_csv(SOURCE/'taxonomy.csv',index_col=0)
    ref=pd.read_csv(SOURCE/'chemical_reference_groups.csv',index_col=0)
    reps=a.groupby('sample_id').date.nunique();excluded=reps[reps>1].index.tolist()
    a=a[~a.sample_id.isin(excluded)].copy()
    blocks=sorted(a.date.unique())
    cache={}
    for i,d1 in enumerate(blocks):
        for d2 in blocks[i+1:]:
            tr=a[~a.date.isin([d1,d2])]
            dm=design(chem,meta,tax,ref,sorted(tr.sample_id.unique()))
            for val,outer in [(d1,d2),(d2,d1)]:
                sc,_=fit_predict(tr,a[a.date.eq(val)],dm)
                sc['inner_block']=val
                cache[(outer,val)]=sc
        print('inner training blocks',i+1,'/',len(blocks),flush=True)
    predictions=[];choices=[];inner=[]
    for held in blocks:
        risks=pd.concat([cache[(held,v)] for v in blocks if v!=held])
        risks['outer_block']=held;inner.append(risks)
        # Each cell x phase has its own training-only penalty; equal weight to
        # inner blocks prevents one large block dominating model selection.
        risk=risks.groupby(['model','neuron_class','window','penalty']).mse.mean().reset_index()
        choice=risk.loc[risk.groupby(['model','neuron_class','window']).mse.idxmin()].copy()
        choice['outer_block']=held;choices.append(choice)
        tr=a[~a.date.eq(held)];te=a[a.date.eq(held)]
        assert not(set(tr.sample_id)&set(te.sample_id))
        dm=design(chem,meta,tax,ref,sorted(tr.sample_id.unique()))
        _,pr=fit_predict(tr,te,dm,output=True)
        pr=pr.merge(choice[['model','neuron_class','window','penalty']],on=['model','neuron_class','window','penalty'])
        predictions.append(pr)
        print('outer prediction',held,flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(OUT/'tables/targeted_chemistry_predictions.csv',index=False)
    pd.concat(choices).to_csv(OUT/'tables/targeted_chemistry_choices.csv',index=False)
    pd.concat(inner).to_csv(OUT/'tables/targeted_chemistry_inner_risks.csv',index=False)
    block,summary,mean=summarize(pred)
    block.to_csv(OUT/'tables/targeted_chemistry_block_scores.csv',index=False)
    summary.to_csv(OUT/'tables/targeted_chemistry_scores.csv',index=False)
    mean.to_csv(OUT/'tables/targeted_chemistry_strain_predictions.csv',index=False)
    # Genuine fragility check: delete each held-out strain's contribution from
    # evaluation, without refitting or selecting the favourable deletion.
    deletions=[]
    for key,d in mean.groupby(['model','neuron_class','window']):
        total_l=d.loss.sum();total_n=d['null'].sum()
        for r in d.itertuples():
            deletions.append(dict(zip(['model','neuron_class','window'],key),omitted_strain=r.sample_id,
                r2_without=1-(total_l-r.loss)/(total_n-r.null)))
    pd.DataFrame(deletions).to_csv(OUT/'tables/targeted_chemistry_delete_one.csv',index=False)
    methods=dict(seed=SEED,excluded_repeated_strains=excluded,n_strains=a.sample_id.nunique(),
                 n_animals=len(a[['date','worm_key']].drop_duplicates()),windows=WINDOWS,models=MODELS,
                 penalties=PENALTIES.tolist(),outer_blocks=len(blocks),inner_blocks=len(blocks)-1,
                 target='animal-centered response relative to the measured stimulus set, delta F/F0',
                 observation='animal-strain-cell; trials averaged beforehand; each strain-cell has total weight 1',
                 preprocessing='training-only z scores and median imputation; full-panel QC eligibility reused',
                 selection='per cell and phase nested blocked tuning; all six predeclared representations reported',
                 inference='exploratory reuse of this dataset, no independent experimental validation; no p values',
                 limitation='cross-block prediction limits memorization but does not identify strain versus block confounding')
    (OUT/'logs/targeted_chemistry_methods.json').write_text(json.dumps(methods,indent=2))
    print(summary.pivot(index=['neuron_class','window'],columns='model',values='r2_strain_mean').round(3).to_string())


if __name__=='__main__':
    main()
