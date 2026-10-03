"""Predict the entire 13 x 5 response vector from notebook-matched log2FC.

No neuron is a baseline or preselected target. All 106 matched strains are
retained. Entire acquisition blocks are held out, and every test strain ID is
purged from training, including its observations on another block.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
T=OUT/'tables'
CELLS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
TARGETS=['bin0','bin1','bin2','bin3','bin4']
PENALTIES=np.array([.001,.01,.1,1.,10.,100.])
MODELS=['reference','level_reference','genus_reference','fc380','genus_fc380']


def load():
    wide=pd.read_parquet(T/'aligned_neural_animal_5bins.parquet').reset_index()
    wide['date']=wide.date.astype(str)
    frames=[]
    for n in CELLS:
        cols=[c for c in wide if c.startswith(n+'__')]
        assert len(cols)==5
        f=wide[['sample_id','date','worm_key']+cols].dropna().rename(columns=dict(zip(cols,TARGETS)))
        f['neuron_class']=n;frames.append(f)
    a=pd.concat(frames,ignore_index=True)
    chem=pd.read_csv(T/'aligned_chemical_log2fc_paired.csv',index_col=0)
    # All 380 notebook features, no added pseudocount, QC filter or z score.
    assert chem.shape==(106,380) and np.isfinite(chem).all().all()
    tax=pd.read_csv(T/'aligned_taxonomy_paired.csv',index_col=0)
    ref=pd.read_csv(T/'aligned_chemical_reference_groups_paired.csv',index_col=0)
    return a,chem,tax,ref


def designs(chem,tax,ref,ids):
    r=pd.get_dummies(ref.reference_group,prefix='ref',dtype=float)
    g=pd.get_dummies(tax.genus_clean,prefix='genus',dtype=float)
    r=r.loc[:,r.loc[ids].sum()>0];g=g.loc[:,g.loc[ids].sum()>0]
    # A single constant rescales the penalty and preserves the notebook RMS
    # geometry exactly. Individual chemical features are NOT standardized.
    x=chem/np.sqrt(chem.shape[1])
    level=pd.DataFrame({'mean_log2fc':chem.mean(axis=1),'rms_log2fc':np.sqrt((chem**2).mean(axis=1))})
    level=(level-level.loc[ids].mean())/level.loc[ids].std().clip(lower=1e-8)
    return dict(reference=r,level_reference=pd.concat([r,level],axis=1),
                genus_reference=pd.concat([r,g],axis=1),fc380=pd.concat([r,x],axis=1),
                genus_fc380=pd.concat([r,g,x],axis=1))


def arrays(d,x):
    group=d[['date','worm_key']].astype(str).agg('/'.join,axis=1).to_numpy()
    xx=x.loc[d.sample_id].reset_index(drop=True)
    yy=d[TARGETS].reset_index(drop=True)
    xx-=xx.groupby(group).transform('mean');yy-=yy.groupby(group).transform('mean')
    # Animal averaging within block, equal block weight for repeated strains.
    n_animals=d.groupby(['date','sample_id']).sample_id.transform('size').to_numpy()
    n_blocks=d.groupby('sample_id').date.transform('nunique').to_numpy()
    w=1/(n_animals*n_blocks)
    return xx.to_numpy(),yy.to_numpy(),w


def partition(a,held):
    test=a[a.date.eq(held)].copy()
    ids=set(test.sample_id)
    train=a[~a.sample_id.isin(ids)].copy()
    assert not(set(train.sample_id)&ids)
    assert not(set(zip(train.date,train.worm_key))&set(zip(test.date,test.worm_key)))
    return train,test


def fit(train,test,dms,output=False):
    risks=[];predictions=[]
    for cell in CELLS:
        tr=train[train.neuron_class.eq(cell)].reset_index(drop=True)
        te=test[test.neuron_class.eq(cell)].reset_index(drop=True)
        if len(te)==0:continue
        assert len(tr)>0
        for name,x in dms.items():
            X,Y,w=arrays(tr,x);V,Z,v=arrays(te,x)
            gram=(X*w[:,None]).T@X/w.sum();xy=(X*w[:,None]).T@Y/w.sum()
            ev,u=np.linalg.eigh(gram);ev=np.maximum(ev,0);uy=u.T@xy
            # One scale per cell, shared across its five bins. This balances
            # cells in model selection without selecting particular cells.
            scale=max(float(np.sqrt(np.sum(w[:,None]*Y**2)/(w.sum()*5))),1e-8)
            for penalty in PENALTIES:
                pred=V@(u@(uy/(ev[:,None]+penalty)))
                risks.append(dict(model=name,penalty=penalty,neuron_class=cell,
                    scaled_mse=np.sum(v[:,None]*((Z-pred)/scale)**2)/(v.sum()*5)))
                if output:
                    for j,col in enumerate(TARGETS):
                        f=te[['sample_id','date','worm_key']].copy()
                        f['model']=name;f['penalty']=penalty;f['neuron_class']=cell;f['bin']=j
                        f['observed']=Z[:,j];f['predicted']=pred[:,j];f['weight']=v;f['train_cell_scale']=scale
                        predictions.append(f)
    return pd.DataFrame(risks),pd.concat(predictions,ignore_index=True) if output else None


def summarize(pred):
    rows=[]
    # Different cells share animals: components are descriptive, not n=65
    # independent biological repetitions. Aggregate to strain x block first.
    means=pred.groupby(['model','neuron_class','bin','date','sample_id']).agg(
        observed=('observed','mean'),predicted=('predicted','mean'),train_cell_scale=('train_cell_scale','first')).reset_index()
    means['block_weight']=1/means.groupby(['model','neuron_class','bin','sample_id']).date.transform('nunique')
    for scope,groups in [('population',['model']),('cell',['model','neuron_class']),('block',['model','date'])]:
        for key,d in means.groupby(groups):
            if not isinstance(key,tuple):key=(key,)
            for units,scale in [('raw',np.ones(len(d))),('cell_balanced',d.train_cell_scale.to_numpy())]:
                w=d.block_weight.to_numpy();err=(d.observed-d.predicted).to_numpy()/scale;null=d.observed.to_numpy()/scale
                rows.append(dict(scope=scope,**dict(zip(groups,key)),units=units,
                    r2=1-np.sum(w*err**2)/np.sum(w*null**2),loss=np.sum(w*err**2),null=np.sum(w*null**2),
                    n_strains=d.sample_id.nunique(),n_blocks=d.date.nunique()))
    return pd.DataFrame(rows),means


def deletion_scores(means):
    """Fixed predictions, but restore equal strain weights after each deletion."""
    deletion=[]
    for model,d in means.groupby('model'):
        for unit in ['sample_id','date']:
            for omitted in d[unit].unique():
                z=d[d[unit]!=omitted]
                w=1/z.groupby(['neuron_class','bin','sample_id']).date.transform('nunique').to_numpy()
                for units,scale in [('raw',np.ones(len(z))),('cell_balanced',z.train_cell_scale.to_numpy())]:
                    r2=1-np.sum(w*((z.observed-z.predicted)/scale)**2)/np.sum(w*(z.observed/scale)**2)
                    deletion.append(dict(model=model,units=units,omission_unit=unit,omitted=omitted,r2=r2))
    return pd.DataFrame(deletion)


def main():
    a,chem,tax,ref=load();blocks=sorted(a.date.unique())
    all_predictions=[];choices=[];risks=[];audit=[]
    for held in blocks:
        tr,te=partition(a,held)
        inner=[]
        for val in sorted(tr.date.unique()):
            it,iv=partition(tr,val)
            dm=designs(chem,tax,ref,sorted(it.sample_id.unique()))
            loss,_=fit(it,iv,dm)
            loss['inner_block']=val;loss['outer_block']=held;inner.append(loss)
            audit.append(dict(outer_block=held,inner_block=val,n_train_strains=it.sample_id.nunique(),n_test_strains=iv.sample_id.nunique(),
                train_strains=';'.join(sorted(it.sample_id.unique())),test_strains=';'.join(sorted(iv.sample_id.unique())),
                train_animals=';'.join(sorted(set(it.date+'/'+it.worm_key))),test_animals=';'.join(sorted(set(iv.date+'/'+iv.worm_key)))))
        inner=pd.concat(inner);risks.append(inner)
        risk=inner.groupby(['model','penalty']).scaled_mse.mean()
        chosen={m:float(risk.loc[m].idxmin()) for m in MODELS}
        dm=designs(chem,tax,ref,sorted(tr.sample_id.unique()))
        _,pred=fit(tr,te,dm,output=True)
        choice=pd.DataFrame([dict(model=m,penalty=chosen[m],outer_block=held) for m in MODELS])
        pred=pred.merge(choice[['model','penalty']],on=['model','penalty']);all_predictions.append(pred);choices.append(choice)
        audit.append(dict(outer_block=held,inner_block='outer',n_train_strains=tr.sample_id.nunique(),n_test_strains=te.sample_id.nunique(),
            train_strains=';'.join(sorted(tr.sample_id.unique())),test_strains=';'.join(sorted(te.sample_id.unique())),
            train_animals=';'.join(sorted(set(tr.date+'/'+tr.worm_key))),test_animals=';'.join(sorted(set(te.date+'/'+te.worm_key)))))
        print('Completed population chemical holdout',held,flush=True)
    pred=pd.concat(all_predictions,ignore_index=True)
    # Across outer folds repeated strains still contribute total weight one.
    pred['weight']/=pred.groupby(['model','neuron_class','bin','sample_id']).date.transform('nunique')
    pred.to_csv(T/'population_chemistry_predictions.csv',index=False)
    pd.concat(choices).to_csv(T/'population_chemistry_choices.csv',index=False)
    pd.concat(risks).to_csv(T/'population_chemistry_inner_risks.csv',index=False)
    pd.DataFrame(audit).to_csv(T/'population_chemistry_fold_audit.csv',index=False)
    scores,means=summarize(pred)
    scores.to_csv(T/'population_chemistry_scores.csv',index=False)
    means.to_csv(T/'population_chemistry_strain_block_predictions.csv',index=False)
    # Leave-out evaluation-block and strain influence, fixed predictions.
    deletion_scores(means).to_csv(T/'population_chemistry_deletion.csv',index=False)
    (OUT/'logs/population_chemistry_methods.json').write_text(json.dumps(dict(
        status='success',neurons=CELLS,bins='5 equal 5-second bins [0,25), all neurons considered from outset',
        n_strains=106,n_chemical_features=380,n_animal_strain=607,outer_folds=9,
        chemical_input='exact notebook log2FC; equal features; /sqrt(380) is a common constant, no feature z score or QC removal',
        neural_target='animal-centered calcium differences over its measured stimulus set; missing cells omitted, not zero filled',
        unit='animal-strain-neuron; trial averaged upstream; each strain has total weight one including multiple blocks',
        models=MODELS,penalties=PENALTIES.tolist(),selection='one penalty per population/model/outer fold, chosen using all 13 neurons and 5 bins equally after train cell scaling',
        holdout='entire block plus purge every test strain ID from all training blocks; same for inner folds',
        evaluation='strain-block mean targets; repeated strains get equal blocks then total strain weight one; R2 relative zero within-animal response; deletion checks re-normalize remaining strain weights without refitting',
        limitation='No independent experiment. Strain/block confounding remains. This test covers the whole response vector rather than optimizing selected cells.'),indent=2))
    print(scores[scores.scope=='population'].round(4).to_string(index=False))


if __name__=='__main__':main()
