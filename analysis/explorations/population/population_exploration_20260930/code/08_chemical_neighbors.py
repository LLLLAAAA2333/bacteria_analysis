"""Fixed local nonlinear check of chemical explanation: k=3 nearest neighbors.

Chemical distances only (not neural/chemical distance-matrix correlations).
Exclude the same six repeated strains as 07, hold out whole experiment blocks,
fit 162-feature chemical scaling in the training strains only. Predict a neural
profile as the unweighted mean of the three chemical neighbors' animal-centered
profiles. The second, prespecified sensitivity removes each sample's median z
level before measuring distance. No k tuning, target-based feature selection,
or use of test-neural intercepts for prediction. Exploratory blocked prediction.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
OLD=ROOT/'reports/exploration_20260929/tables'
T=OUT/'tables'
K=3
WINDOWS=['stim','full']
NEURONS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
MODELS=['panel162_knn3','panel162_relative_knn3']


def summarize_losses(pred):
    p=pred.copy()
    p['loss']=p.weight*((p.observed-p.predicted)/p.train_scale)**2
    p['null']=p.weight*(p.observed/p.train_scale)**2
    b=p.groupby(['model','date','neuron_class','window']).agg(loss=('loss','sum'),null=('null','sum'),weight=('weight','sum')).reset_index()
    b['relative_r2']=1-b.loss/b['null']
    n=b.groupby(['model','neuron_class','window']).agg(loss=('loss','sum'),null=('null','sum'),
        improving_blocks=('relative_r2',lambda x:int(x.gt(0).sum())),median_block_r2=('relative_r2','median')).reset_index()
    n['relative_r2']=1-n.loss/n['null']
    s=b.groupby(['model','window']).agg(loss=('loss','sum'),null=('null','sum')).reset_index()
    s['relative_r2']=1-s.loss/s['null']
    return b,n,s


def targeted_contrasts(pred,tax):
    p=pred.merge(tax[['genus_clean']],left_on='sample_id',right_index=True,validate='many_to_one')
    keys=['model','window','date','worm_key','neuron_class']
    rows=[]
    specs=[('Escherichia_minus_Lactobacillus','genus_clean','Escherichia','Lactobacillus'),
           ('A189_minus_A178','sample_id','A189','A178')]
    for label,column,left,right in specs:
        q=p[p[column].isin([left,right])].groupby(keys+[column])[['observed','predicted']].mean().unstack(column)
        needed=[(v,s) for v in ['observed','predicted'] for s in [left,right]]
        q=q.dropna(subset=needed)
        z=q.index.to_frame(index=False)
        z['contrast_name']=label
        for v in ['observed','predicted']:z[v]=q[(v,left)].to_numpy()-q[(v,right)].to_numpy()
        rows.append(z)
    pairs=pd.concat(rows,ignore_index=True)
    pairs['same_direction']=np.sign(pairs.observed)==np.sign(pairs.predicted)
    # Cell relationship needs both classes in the same animal, not a difference
    # of averages from potentially different animal subsets.
    r=pairs[pairs.neuron_class.isin(['ADF','ASH'])].pivot(index=keys[:-1]+['contrast_name'],columns='neuron_class',values=['observed','predicted'])
    r=r.dropna()
    rr=r.index.to_frame(index=False)
    rr['neuron_class']='ADF_minus_ASH'
    for v in ['observed','predicted']:rr[v]=r[(v,'ADF')].to_numpy()-r[(v,'ASH')].to_numpy()
    rr['same_direction']=np.sign(rr.observed)==np.sign(rr.predicted)
    pairs=pd.concat([pairs,rr],ignore_index=True)
    summary=pairs.groupby(['contrast_name','model','window','neuron_class']).agg(
        n_animal_pairs=('observed','size'),n_blocks=('date','nunique'),
        observed_mean=('observed','mean'),predicted_mean=('predicted','mean'),
        n_correct_direction=('same_direction','sum'),observed_min=('observed','min'),observed_max=('observed','max')).reset_index()
    return pairs,summary


def main():
    a=pd.read_csv(OLD/'animal_metrics.csv')
    log=pd.read_csv(OLD/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(OLD/'chemical_feature_metadata.csv',index_col=0)
    tax=pd.read_csv(OLD/'taxonomy.csv',index_col=0)
    repeated=a.groupby('sample_id').date.nunique().loc[lambda x:x.gt(1)].index.tolist()
    a=a[~a.sample_id.isin(repeated)].copy()
    features=meta.index[meta.complete_eligible].tolist()
    assert len(features)==162 and a.sample_id.nunique()==100
    assert np.isfinite(log.loc[a.sample_id.unique(),features]).all().all()
    neighbors=[];predictions=[];scalings=[];folds=[]
    for held in sorted(a.date.unique()):
        tr=a[a.date.ne(held)].copy();te=a[a.date.eq(held)].copy()
        train_ids=sorted(tr.sample_id.unique());test_ids=sorted(te.sample_id.unique())
        assert not(set(train_ids)&set(test_ids))
        assert not(set(zip(tr.date,tr.worm_key))&set(zip(te.date,te.worm_key)))
        mu=log.loc[train_ids,features].mean();sd=log.loc[train_ids,features].std(ddof=1).clip(lower=1e-6)
        z=(log.loc[train_ids+test_ids,features]-mu)/sd
        zr=z.sub(z.median(axis=1),axis=0)
        scale=pd.DataFrame({'feature':features,'training_mean_log':mu,'training_sd_log':sd})
        scale['held_block']=held;scalings.append(scale)
        # Center targets separately within each animal and neuron, over the
        # retained candidate stimuli. Training profiles give animals equal
        # weight within each strain; test rows keep actual missing-cell masks.
        for frame in [tr,te]:
            for window in WINDOWS:
                frame[f'centered_{window}']=frame[window]-frame.groupby(['date','worm_key','neuron_class'])[window].transform('mean')
        profiles=tr.groupby(['sample_id','neuron_class'])[[f'centered_{w}' for w in WINDOWS]].mean()
        for model,xx in zip(MODELS,[z,zr]):
            xtrain=xx.loc[train_ids].to_numpy();xtest=xx.loc[test_ids].to_numpy()
            d=np.sqrt(((xtest[:,None,:]-xtrain[None,:,:])**2).mean(axis=2))
            ranks=np.argsort(d,axis=1,kind='stable')[:,:K]
            chosen={s:[train_ids[i] for i in ranks[j]] for j,s in enumerate(test_ids)}
            for j,s in enumerate(test_ids):
                for rank,i in enumerate(ranks[j],start=1):
                    n=train_ids[i]
                    neighbors.append(dict(model=model,held_block=held,sample_id=s,rank=rank,neighbor=n,
                        distance_rms_z=d[j,i],target_genus=tax.loc[s,'genus_clean'],neighbor_genus=tax.loc[n,'genus_clean'],
                        neighbor_block=int(tr.loc[tr.sample_id.eq(n),'date'].iloc[0]),
                        target_median_z=float(z.loc[s].median()),neighbor_median_z=float(z.loc[n].median())))
            for neuron in NEURONS:
                train_cell=tr[tr.neuron_class.eq(neuron)]
                test_cell=te[te.neuron_class.eq(neuron)].copy()
                wtrain=1/train_cell.groupby('sample_id').sample_id.transform('size').to_numpy()
                for window in WINDOWS:
                    target=f'centered_{window}'
                    profile=profiles[target].xs(neuron,level='neuron_class')
                    assert set(train_ids).issubset(profile.index)
                    pred={s:float(profile.loc[ids].mean()) for s,ids in chosen.items()}
                    q=test_cell[['sample_id','date','worm_key','neuron_class']].copy()
                    q['model']=model;q['window']=window
                    q['observed']=test_cell[target].to_numpy()
                    q['predicted_uncentered']=q.sample_id.map(pred)
                    # Use only predictions to set prediction origin, never the
                    # observed held-block neural mean. Relative-set evaluation.
                    q['predicted']=q.predicted_uncentered-q.groupby(['date','worm_key']).predicted_uncentered.transform('mean')
                    q['weight']=1/q.groupby('sample_id').sample_id.transform('size')
                    q['train_scale']=max(1e-6,float(np.sqrt(np.sum(wtrain*train_cell[target].to_numpy()**2)/wtrain.sum())))
                    predictions.append(q)
        folds.append(dict(held_block=held,n_train_strains=len(train_ids),n_test_strains=len(test_ids),
                          n_train_animals=len(tr[['date','worm_key']].drop_duplicates()),
                          n_test_animals=len(te[['date','worm_key']].drop_duplicates()),n_features=len(features)))
    pred=pd.concat(predictions,ignore_index=True)
    near=pd.DataFrame(neighbors)
    pred.to_csv(T/'chem_neighbors_predictions.csv',index=False)
    near.to_csv(T/'chem_neighbors_members.csv',index=False)
    pd.concat(scalings,ignore_index=True).to_csv(T/'chem_neighbors_training_scaling.csv',index=False)
    pd.DataFrame(folds).to_csv(T/'chem_neighbors_coverage.csv',index=False)
    b,n,s=summarize_losses(pred)
    b.to_csv(T/'chem_neighbors_block_results.csv',index=False)
    n.to_csv(T/'chem_neighbors_neuron_results.csv',index=False)
    s.to_csv(T/'chem_neighbors_summary.csv',index=False)
    pairs,summary=targeted_contrasts(pred,tax)
    pairs.to_csv(T/'chem_neighbors_targeted_animal_contrasts.csv',index=False)
    summary.to_csv(T/'chem_neighbors_targeted_summary.csv',index=False)
    # Each held-out observed value and training scale must agree with 07.
    ridge=pd.read_csv(T/'chemistry_population_predictions.csv')
    ridge=ridge[ridge.model.eq('panel')]
    keys=['sample_id','date','worm_key','neuron_class','window']
    compare=pred[pred.model.eq(MODELS[0])].merge(ridge[keys+['observed','weight','train_scale']],on=keys,
                                               validate='one_to_one',suffixes=('_knn','_ridge'))
    errors={v:float(abs(compare[v+'_knn']-compare[v+'_ridge']).max()) for v in ['observed','weight','train_scale']}
    assert len(compare)==len(ridge) and max(errors.values())<1e-12
    assert near.groupby(['model','sample_id']).size().eq(K).all()
    assert near.groupby(['model','sample_id']).neighbor.nunique().eq(K).all()
    assert near.held_block.ne(near.neighbor_block).all()
    input_names=['animal_metrics.csv','chemical_log.csv','chemical_feature_metadata.csv','taxonomy.csv']
    manifest=[dict(path=str((OLD/name).relative_to(ROOT)),sha256=hashlib.sha256((OLD/name).read_bytes()).hexdigest()) for name in input_names]
    methods=dict(k=K,models=MODELS,n_strains=100,excluded_strains=repeated,held_blocks=len(folds),
        n_features=len(features),features=features,windows=WINDOWS,neurons=NEURONS,
        fitting='Training-only sample mean/sd in log chemistry; no k tuning; equal neighbor weights.',
        sensitivity='Subtract each sample median over 162 training-standardized values before distance.',
        neural_target='Animal-centered stored delta F/F0, after excluding repeated strains; predictions separately centered from predicted values only.',
        evaluation='Identical observed rows, animal weights, training response scales and zero-relative-response comparator to 07.',
        validation_errors_vs_07=errors,
        scope='Fixed simple local nonlinear comparator; failure does not rule out all nonlinear mappings or absent/unmeasured chemistry.',
        annotation_groups='Not used: avoids chemical group selection on held-out profiles.',
        inference='Exploratory blocked prediction; no p values, no independent experiment, no causal molecule claims.',
        inputs=manifest,run_status='success')
    (OUT/'logs/chem_neighbors_methods.json').write_text(json.dumps(methods,indent=2))
    print(s.round(4).to_string(index=False))
    print('\nPer-neuron held-block prediction:\n'+n.pivot(index=['neuron_class','window'],columns='model',values='relative_r2').round(3).to_string())
    print('\nSelected contrasts:\n'+summary[summary.neuron_class.isin(['ADF','ASH','AWA','ADF_minus_ASH'])].round(4).to_string(index=False))
    print('\nBifidobacterium neighbor identities:\n'+near[near.sample_id.isin(['A178','A179','A189'])].to_string(index=False))
    print('\nValidation vs 07: '+json.dumps(errors))


if __name__=='__main__':main()
