"""Bounded chemical-composition alternatives to whole-panel regression.

Two representations only: two within-annotation-class PCs, and the 25/50/75%
quantiles of class-wise relative feature z scores. Class labels with >=3
complete-QC features are fixed from metadata, not neural outcomes. Neither is
assumed a biochemical pathway or chemically coherent activity score. All axes,
scaling and per-cell/window ridge penalties are fitted within nested whole
experiment-block splits. One chemical reference profile per strain is paired
by strain ID, not a verified culture/exposure concentration.

Run: .pixi/envs/default/bin/python reports/atlas_extension_20260930/code/03_chemical_profiles.py
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
SOURCE = ROOT/'reports/exploration_20260929/tables'
T = OUT/'tables'
L = OUT/'logs'
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
WINDOWS = ['stim','post','full']
LAMBDAS = [.01, .1, 1., 10., 100.]
MODELS = ['level','taxonomy','class_pcs','taxonomy_class_pcs','class_quantiles','taxonomy_class_quantiles']


def design(log, meta, tax, ref, train_ids, outer_label=None):
    names = meta.index[meta.complete_eligible].tolist()
    z = (log[names]-log.loc[train_ids,names].mean())/log.loc[train_ids,names].std(ddof=1).clip(lower=1e-6)
    level = pd.DataFrame({'median_log': log[names].median(axis=1), 'median_z': z.median(axis=1)})
    level = (level-level.loc[train_ids].mean())/level.loc[train_ids].std(ddof=1).clip(lower=1e-6)
    refs = pd.get_dummies(ref.reference_group, prefix='ref',dtype=float)
    refs = refs.loc[:,refs.loc[train_ids].sum().gt(0)]
    genus = pd.get_dummies(tax.genus_clean,prefix='genus',dtype=float)
    genus = genus.loc[:,genus.loc[train_ids].sum().gt(0)]
    pcs, quant, loadings, coverage = {}, {}, [], []
    # Relative per-feature level is a standardized profile contrast, not a
    # molecular concentration ratio; subtract the 162-feature median.
    relative = z.sub(z.median(axis=1),axis=0)
    classes = meta.loc[names].groupby('Class',dropna=True)
    for label,rows in classes:
        ff=rows.index.tolist()
        if len(ff)<3 or str(label).lower()=='na': continue
        train=z.loc[train_ids,ff].to_numpy()
        _,s,v=np.linalg.svd(train,full_matrices=False)
        for j in range(2):
            axis=v[j].copy()
            # Fix arbitrary SVD orientation by largest absolute loading.
            axis *= 1 if axis[np.argmax(abs(axis))]>=0 else -1
            score=pd.Series(z[ff].to_numpy()@axis,index=z.index)
            score /= score.loc[train_ids].std(ddof=1)
            key=f'{label}:PC{j+1}'
            pcs[key]=score
            if outer_label is not None:
                for feature,loading in zip(ff,axis):
                    loadings.append(dict(held_block=outer_label,chemical_class=label,axis=f'PC{j+1}',feature=feature,loading=loading,explained_class_fraction=s[j]**2/(s**2).sum()))
        for q in [.25,.5,.75]:
            key=f'{label}:Q{int(q*100)}'
            score=relative[ff].quantile(q,axis=1)
            quant[key]=(score-score.loc[train_ids].mean())/max(1e-6,score.loc[train_ids].std(ddof=1))
        coverage.append(dict(chemical_class=label,n_features=len(ff),features=';'.join(ff)))
    pca=pd.DataFrame(pcs,index=log.index);q=pd.DataFrame(quant,index=log.index)
    base=pd.concat([level,refs],axis=1)
    out={'level':base, 'taxonomy':pd.concat([base,genus],axis=1),
         'class_pcs':pd.concat([base,pca],axis=1), 'taxonomy_class_pcs':pd.concat([base,genus,pca],axis=1),
         'class_quantiles':pd.concat([base,q],axis=1), 'taxonomy_class_quantiles':pd.concat([base,genus,q],axis=1)}
    return out,loadings,coverage


def centered_rows(rows, x):
    block=rows[['date','worm_key']].astype(str).agg('/'.join,axis=1).to_numpy()
    xx=x.loc[rows.sample_id].reset_index(drop=True)
    yy=rows[WINDOWS].reset_index(drop=True)
    xx=xx-xx.groupby(block).transform('mean')
    yy=yy-yy.groupby(block).transform('mean')
    weight=1/rows.groupby(['sample_id','date']).sample_id.transform('size').to_numpy()
    return xx.to_numpy(),yy.to_numpy(),weight


def fitted(train,test,designs,lambdas,save_coef=False):
    preds=[];coefs=[]
    for cell in NEURONS:
        tr=train[train.neuron_class.eq(cell)].reset_index(drop=True)
        te=test[test.neuron_class.eq(cell)].reset_index(drop=True)
        for model,x in designs.items():
            X,Y,w=centered_rows(tr,x);V,Z,v=centered_rows(te,x)
            gram=(X*w[:,None]).T@X/w.sum();cross=(X*w[:,None]).T@Y/w.sum()
            ev,u=np.linalg.eigh(gram);ev=np.maximum(ev,0);proj=u.T@cross
            scale=np.sqrt((w[:,None]*Y**2).sum(axis=0)/w.sum()).clip(1e-6)
            for penalty in lambdas:
                coef=u@(proj/(ev[:,None]+penalty));p=V@coef
                for j,window in enumerate(WINDOWS):
                    q=te[['sample_id','date','worm_key']].copy()
                    q['neuron_class']=cell;q['window']=window;q['model']=model;q['lambda']=penalty
                    q['observed']=Z[:,j];q['predicted']=p[:,j];q['weight']=v;q['train_scale']=scale[j]
                    preds.append(q)
                    if save_coef:
                        for feature,value in zip(x.columns,coef[:,j]):
                            coefs.append(dict(held_block=int(test.date.iloc[0]),model=model,neuron_class=cell,window=window,lambda_chosen=penalty,feature=feature,coefficient=value))
    return pd.concat(preds,ignore_index=True),coefs


def loss(pred):
    p=pred.copy()
    p['loss']=p.weight*(p.observed-p.predicted)**2
    p['null']=p.weight*p.observed**2
    per=p.groupby(['model','lambda','date','neuron_class','window']).agg(loss=('loss','sum'),null=('null','sum'),weight=('weight','sum')).reset_index()
    per['mse']=per.loss/per.weight
    return per


def scores(q, keys):
    data=q.copy();data['loss']=(data.observed-data.predicted)**2;data['null']=data.observed**2
    data['sign_agreement']=(np.sign(data.observed)==np.sign(data.predicted)).astype(float)
    s=data.groupby(keys).agg(loss=('loss','sum'),null=('null','sum'),n=('loss','size'),sign_agreement=('sign_agreement','mean')).reset_index()
    s['r2']=1-s.loss/s['null']
    return s


def checks(pred,tax):
    # Equal weight per held-out strain for scientific interpretation. The
    # animal rows remain available, so this does not inflate biological n.
    ss=pred.groupby(['model','neuron_class','window','date','sample_id'])[['observed','predicted']].mean().reset_index()
    ss['genus']=ss.sample_id.map(tax.genus_clean)
    ss.to_csv(T/'chem_profiles_strain_predictions.csv',index=False)
    keys=['model','neuron_class','window']
    summary=scores(ss,keys);summary.to_csv(T/'chem_profiles_strain_scores.csv',index=False)
    block=scores(ss,keys+['date']);block.to_csv(T/'chem_profiles_block_scores.csv',index=False)
    # Countercheck: can the chemistry distinguish strains within the same
    # genus AND held-out block, beyond between-genus differences? The same
    # predicted values are centered independently of neural test responses.
    paired=pred.copy()
    paired['genus']=paired.sample_id.map(tax.genus_clean)
    gkeys=keys+['date','worm_key','genus']
    paired=paired[paired.groupby(gkeys).sample_id.transform('size')>=2].copy()
    paired[['observed','predicted']]-=paired.groupby(gkeys)[['observed','predicted']].transform('mean')
    paired.to_csv(T/'chem_profiles_within_genus_animal_predictions.csv',index=False)
    within=paired.groupby(keys+['date','genus','sample_id'])[['observed','predicted']].mean().reset_index()
    within.to_csv(T/'chem_profiles_within_genus_predictions.csv',index=False)
    scores(within,keys).to_csv(T/'chem_profiles_within_genus_scores.csv',index=False)
    scores(within,keys+['genus']).to_csv(T/'chem_profiles_within_genus_by_genus.csv',index=False)
    # Fixed-prediction stress test; no re-fit and no independent validation.
    # Remove each test strain in turn and recompute gain vs taxonomy.
    taxpred=ss[ss.model.eq('taxonomy')][['neuron_class','window','date','sample_id','predicted']].rename(columns={'predicted':'taxonomy_pred'})
    z=ss.merge(taxpred,on=['neuron_class','window','date','sample_id'],validate='many_to_one')
    z['incremental_gain']=(z.observed-z.taxonomy_pred)**2-(z.observed-z.predicted)**2
    z.to_csv(T/'chem_profiles_strain_gain.csv',index=False)
    deletes=[]
    for key,g in z.groupby(keys):
        null=(g.observed**2).sum();gain=g.incremental_gain.sum()
        for row in g.itertuples(index=False):
            deletes.append(dict(zip(keys,key),deleted_strain=row.sample_id,delta_r2_vs_taxonomy=(gain-row.incremental_gain)/(null-row.observed**2)))
    pd.DataFrame(deletes).to_csv(T/'chem_profiles_delete_one_strain.csv',index=False)
    axis_checks(pred,tax)
    return summary,within,z



def axis_checks(pred,tax):
    """Neurally nominated ASI/ASJ mean; common animals only, no chemistry selection."""
    keys=['model','window','sample_id','date','worm_key']
    p=pred[pred.neuron_class.isin(['ASI','ASJ'])].pivot(index=keys,columns='neuron_class',values=['observed','predicted']).dropna()
    a=p.index.to_frame(index=False)
    for name in ['observed','predicted']:a[name]=p[name].mean(axis=1).to_numpy()
    a['genus']=a.sample_id.map(tax.genus_clean)
    a.to_csv(T/'chem_profiles_axis_animal_predictions.csv',index=False)
    summary=[]
    for scope in ['co_measured_ASI_ASJ','co_measured_ASI_ASJ_within_genus']:
        q=a.copy()
        if scope.endswith('within_genus'):
            group=['model','window','date','worm_key','genus']
            q=q[q.groupby(group).sample_id.transform('size')>=2].copy()
            q[['observed','predicted']]-=q.groupby(group)[['observed','predicted']].transform('mean')
        q=q.groupby(['model','window','date','sample_id','genus'])[['observed','predicted']].mean().reset_index()
        q['scope']=scope
        q.to_csv(T/f'chem_profiles_axis_{scope}_predictions.csv',index=False)
        result=scores(q,['model','window'])
        result['scope']=scope
        result['n_blocks']=q.groupby(['model','window']).date.nunique().to_numpy()
        summary.append(result)
    pd.concat(summary,ignore_index=True).to_csv(T/'chem_profiles_axis_scores.csv',index=False)



def within_training_check(a,log,meta,tax,ref):
    """Bounded follow-up: train on within-genus contrasts, not genus separation.

    Neurally nominated targets only: ASK stim/post and the physical-unit mean
    of co-measured ASI/ASJ post. Chemistry representations and penalties remain
    unchanged. This follow-up is exploratory and is logged as such.
    """
    keys=['sample_id','date','worm_key']
    ask=a[a.neuron_class.eq('ASK')][keys+['stim','post']].melt(id_vars=keys,var_name='window',value_name='response')
    ask['target']='ASK'
    both=a[a.neuron_class.isin(['ASI','ASJ'])].pivot(index=keys,columns='neuron_class',values='post').dropna()
    axis=both.index.to_frame(index=False);axis['response']=both.mean(axis=1).to_numpy()
    axis['window']='post';axis['target']='ASI_ASJ_mean'
    data=pd.concat([ask,axis],ignore_index=True);data['genus']=data.sample_id.map(tax.genus_clean)
    g=['date','worm_key','genus','target','window']
    data=data[data.groupby(g).sample_id.transform('size')>=2].copy()
    blocks=sorted(data.date.unique());models=['level','class_pcs','class_quantiles']
    pred=[];choices=[];coefficients=[]

    def rows(d,x):
        d=d.reset_index(drop=True)
        group=d[['date','worm_key','genus']].astype(str).agg('/'.join,axis=1).to_numpy()
        xx=x.loc[d.sample_id].reset_index(drop=True);y=d.response.copy()
        xx-=xx.groupby(group).transform('mean');y-=y.groupby(group).transform('mean')
        w=1/d.groupby(['sample_id','date']).sample_id.transform('size').to_numpy()
        return xx.to_numpy(),y.to_numpy(),w

    def solve(tr,te,designs):
        result=[];coefs=[]
        for (target,window),trg in tr.groupby(['target','window']):
            teg=te[te.target.eq(target)&te.window.eq(window)]
            for model in models:
                x=designs[model];X,y,w=rows(trg,x);V,z,v=rows(teg,x)
                gram=(X*w[:,None]).T@X/w.sum();cross=(X*w[:,None]).T@y/w.sum()
                ev,u=np.linalg.eigh(gram);ev=np.maximum(ev,0)
                scale=max(1e-6,np.sqrt(np.sum(w*y*y)/w.sum()))
                for penalty in LAMBDAS:
                    coef=u@((u.T@cross)/(ev+penalty));prediction=V@coef
                    frame=teg[keys+['genus']].copy()
                    frame['target']=target;frame['window']=window;frame['model']=model;frame['lambda']=penalty
                    frame['observed']=z;frame['predicted']=prediction;frame['weight']=v;frame['train_scale']=scale
                    result.append(frame)
                    coefs.extend(dict(target=target,window=window,model=model,lambda_chosen=penalty,feature=f,coefficient=c) for f,c in zip(x.columns,coef))
        return pd.concat(result,ignore_index=True),coefs

    for held in blocks:
        inner=[]
        for val in blocks:
            if val==held:continue
            tr=data[~data.date.isin([held,val])];te=data[data.date.eq(val)]
            designs,_,_=design(log,meta,tax,ref,sorted(tr.sample_id.unique()))
            p,_=solve(tr,te,designs)
            p['loss']=p.weight*(p.observed-p.predicted)**2
            risk=p.groupby(['model','target','window','lambda']).agg(loss=('loss','sum'),weight=('weight','sum')).reset_index()
            risk['mse']=risk.loss/risk.weight;inner.append(risk)
        risk=pd.concat(inner).groupby(['model','target','window','lambda']).mse.mean()
        best=risk.groupby(level=[0,1,2]).idxmin().map(lambda x:x[-1])
        tr=data[data.date.ne(held)];te=data[data.date.eq(held)]
        designs,_,_=design(log,meta,tax,ref,sorted(tr.sample_id.unique()))
        p,cf=solve(tr,te,designs)
        selected=[]
        for (model,target,window),penalty in best.items():
            pred.append(p[p.model.eq(model)&p.target.eq(target)&p.window.eq(window)&p['lambda'].eq(penalty)])
            selected.append(dict(held_block=held,model=model,target=target,window=window,lambda_chosen=penalty,n_training_strains=tr.sample_id.nunique(),n_test_strains=te.sample_id.nunique()))
        choices.extend(selected)
        cf=pd.DataFrame(cf);cf['held_block']=held
        coefficients.append(cf.merge(pd.DataFrame(selected)[['held_block','model','target','window','lambda_chosen']],validate='many_to_one'))
        print(f'Finished within-genus training check block {held}',flush=True)
    p=pd.concat(pred,ignore_index=True)
    p.to_csv(T/'chem_profiles_within_fit_predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(T/'chem_profiles_within_fit_choices.csv',index=False)
    pd.concat(coefficients,ignore_index=True).to_csv(T/'chem_profiles_within_fit_coefficients.csv',index=False)
    data.to_csv(T/'chem_profiles_within_fit_targets.csv',index=False)
    q=p.groupby(['model','target','window','date','sample_id','genus'])[['observed','predicted']].mean().reset_index()
    q.to_csv(T/'chem_profiles_within_fit_strain_predictions.csv',index=False)
    result=scores(q,['model','target','window'])
    result.to_csv(T/'chem_profiles_within_fit_scores.csv',index=False)
    scores(q,['model','target','window','date']).to_csv(T/'chem_profiles_within_fit_block_scores.csv',index=False)
    deletions=[]
    for key,rows_ in q.groupby(['model','target','window']):
        for sample in rows_.sample_id:
            z=rows_[rows_.sample_id.ne(sample)]
            deletions.append(dict(model=key[0],target=key[1],window=key[2],deleted_strain=sample,r2=1-np.sum((z.observed-z.predicted)**2)/np.sum(z.observed**2)))
    pd.DataFrame(deletions).to_csv(T/'chem_profiles_within_fit_delete_one.csv',index=False)
    methods=dict(status='success',timing='Exploratory follow-up after held-out genus-centered evaluation failed; not an independent validation.',
        question='Does training specifically on genus-internal contrasts recover a transferable chemical explanation masked by between-genus signal?',
        targets=['ASK stim','ASK post','co-measured ASI/ASJ physical-unit mean post'],models=models,lambdas=LAMBDAS,n_outer_blocks=len(blocks),n_strains=data.sample_id.nunique(),n_animals=data[['date','worm_key']].drop_duplicates().shape[0],
        centering='Both X and Y within each animal x genus x target/window; require >=2 distinct strains per group; never subtract test responses from predictions.',
        evaluation='Equal-strain mean prediction R2 over within-genus-centered animal responses; inner folds select ridge penalty by raw-unit per-target MSE.',
        figures='None: a descriptive or positive held-out score alone is not sufficient for a mechanistic chemical main figure.')
    (L/'chem_profiles_within_fit_methods.json').write_text(json.dumps(methods,indent=2))
    print('Within-genus training check:')
    print(result.round(4).to_string(index=False))
    return result



def interpret_bridge():
    """Decompose existing fits on the exact neural-defined five-cell contrast.

    Run after 07_population_bridge.py. No model or feature selection is added.
    PC axes are summed within class, avoiding arbitrary PC sign/rotation in
    cross-fold coefficient rankings. Correlated classes still make attribution
    non-unique; this is a fitted contribution, not an intervention effect.
    """
    membership=pd.read_csv(T/'population_bridge_membership.csv')
    pred=pd.read_csv(T/'chem_profiles_predictions.csv')
    coefs=pd.read_csv(T/'chem_profiles_coefficients.csv')
    log=pd.read_csv(SOURCE/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(SOURCE/'chemical_feature_metadata.csv',index_col=0)
    tax=pd.read_csv(SOURCE/'taxonomy.csv',index_col=0)
    ref=pd.read_csv(SOURCE/'chemical_reference_groups.csv',index_col=0)
    a=pd.read_csv(SOURCE/'animal_metrics.csv')
    ex=a.groupby('sample_id').date.nunique().loc[lambda q:q>1].index
    a=a[~a.sample_id.isin(ex)]
    records=[];verification=[]
    for held,group in membership.groupby('date'):
        designs,_,_=design(log,meta,tax,ref,sorted(a.loc[a.date.ne(held),'sample_id'].unique()))
        for model in ['class_pcs','class_quantiles']:
            x=designs[model]
            for cell in ['ASI','ASJ','ASK','ADF','ASH']:
                beta=coefs[coefs.held_block.eq(held)&coefs.model.eq(model)&coefs.neuron_class.eq(cell)&coefs.window.eq('post')].set_index('feature').coefficient
                assert len(beta)==x.shape[1]
                for worm,g in group.groupby('worm_key'):
                    high=g.loc[g.group.eq('higher'),'sample_id'];low=g.loc[g.group.eq('lower'),'sample_id']
                    dx=x.loc[high].mean()-x.loc[low].mean()
                    terms=dx*beta
                    # Both PCs or all three class quantiles become one class contribution.
                    labels=[f.split(':PC')[0].split(':Q')[0] if (':PC' in f or ':Q' in f) else 'Panel level/reference' for f in terms.index]
                    for label,value in terms.groupby(labels).sum().items():
                        records.append(dict(model=model,date=held,worm_key=worm,neuron_class=cell,chemical_class=label,contribution=value))
                    q=pred[pred.model.eq(model)&pred.date.eq(held)&pred.worm_key.eq(worm)&pred.neuron_class.eq(cell)&pred.window.eq('post')].set_index('sample_id')
                    delta=q.loc[high,'predicted'].mean()-q.loc[low,'predicted'].mean()
                    verification.append(abs(delta-terms.sum()))
    out=pd.DataFrame(records)
    out.to_csv(T/'chem_profiles_bridge_class_contributions.csv',index=False)
    summary=out.groupby(['model','neuron_class','chemical_class']).agg(mean_contribution=('contribution','mean'),min_animal=('contribution','min'),max_animal=('contribution','max'),positive_animals=('contribution',lambda v:int((v>0).sum())),n_animals=('contribution','size'),n_blocks=('date','nunique')).reset_index()
    summary.to_csv(T/'chem_profiles_bridge_class_summary.csv',index=False)
    blocks=out.groupby(['model','neuron_class','chemical_class','date']).contribution.mean().reset_index()
    blocks.to_csv(T/'chem_profiles_bridge_class_blocks.csv',index=False)
    assert max(verification)<1e-12
    (L/'chem_profiles_bridge_methods.json').write_text(json.dumps(dict(status='success',no_refit=True,scope='Exact five-cell co-measured animal membership selected using training-animal ASI/ASJ responses; post window.',
        interpretation='Existing model contribution dx_class @ beta_class; sum all within-class axes. Correlated chemical classes prevent unique causal attribution.',
        n_animals=membership[['date','worm_key']].drop_duplicates().shape[0],n_blocks=membership.date.nunique(),max_reconstruction_error=max(verification),
        run='Import code/03_chemical_profiles.py and call interpret_bridge() after code/07_population_bridge.py.'),indent=2))
    return summary



def verify_outputs():
    """Check units/weighting, independent aggregates, and an independent baseline implementation."""
    p=pd.read_csv(T/'chem_profiles_predictions.csv')
    w=pd.read_csv(T/'chem_profiles_within_fit_predictions.csv')
    md=json.loads((L/'chem_profiles_methods.json').read_text())
    assert all(hashlib.sha256((SOURCE/k).read_bytes()).hexdigest()==v for k,v in md['input_sha256'].items())
    assert not p.duplicated(['model','neuron_class','window','sample_id','date','worm_key']).any()
    assert not w.duplicated(['model','target','window','sample_id','date','worm_key']).any()
    assert p.groupby(['model','neuron_class','window','sample_id']).weight.sum().sub(1).abs().max()<1e-12
    assert w.groupby(['model','target','window','sample_id']).weight.sum().sub(1).abs().max()<1e-12
    means=w.groupby(['model','target','window','date','worm_key','genus'])[['observed','predicted']].mean()
    assert means.abs().max().max()<1e-12
    for row in pd.read_csv(T/'chem_profiles_within_fit_scores.csv').itertuples(index=False):
        q=w[w.model.eq(row.model)&w.target.eq(row.target)&w.window.eq(row.window)].groupby('sample_id')[['observed','predicted']].mean()
        r2=1-np.sum((q.observed-q.predicted)**2)/np.sum(q.observed**2)
        assert abs(r2-row.r2)<1e-12
    baseline_errors=None
    baseline=T/'targeted_chemistry_predictions.csv'
    if baseline.exists():
        b=pd.read_csv(baseline)
        keys=['model','neuron_class','window','date','worm_key','sample_id']
        z=p[p.model.isin(['level','taxonomy'])&p.window.isin(['stim','post'])].merge(b[b.model.isin(['level','taxonomy'])],on=keys,suffixes=('_class','_panel'),validate='one_to_one')
        baseline_errors={f:float(abs(z[f+'_class']-z[f+'_panel']).max()) for f in ['observed','predicted','weight']}
        assert max(baseline_errors.values())<1e-12
    status=dict(status='passed',source_hashes_unchanged=True,prediction_rows=len(p),within_fit_rows=len(w),within_fit_groups=len(means),within_fit_max_abs_group_mean=float(means.abs().max().max()),
        strain_weights_sum_one=True,duplicate_rows=False,within_fit_r2_independently_recomputed=True,baseline_errors_vs_04=baseline_errors,
        baseline_check_note='Comparison executed if code/04 outputs exist; null otherwise.')
    (L/'chem_profiles_verification.json').write_text(json.dumps(status,indent=2))
    return status


def main():
    T.mkdir(exist_ok=True,parents=True);L.mkdir(exist_ok=True,parents=True)
    names=['animal_metrics.csv','chemical_log.csv','chemical_feature_metadata.csv','taxonomy.csv','chemical_reference_groups.csv']
    hashes={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in names}
    a=pd.read_csv(SOURCE/'animal_metrics.csv');log=pd.read_csv(SOURCE/'chemical_log.csv',index_col=0)
    meta=pd.read_csv(SOURCE/'chemical_feature_metadata.csv',index_col=0);tax=pd.read_csv(SOURCE/'taxonomy.csv',index_col=0);ref=pd.read_csv(SOURCE/'chemical_reference_groups.csv',index_col=0)
    repeated=a.groupby('sample_id').date.nunique().loc[lambda s:s>1].index.tolist()
    a=a[~a.sample_id.isin(repeated)].copy();blocks=sorted(a.date.unique())
    assert a.sample_id.nunique()==100 and meta.complete_eligible.sum()==162
    predictions=[];choices=[];all_loadings=[];all_coefs=[]
    for held in blocks:
        inner=[]
        for val in blocks:
            if val==held:continue
            train=a[~a.date.isin([held,val])];test=a[a.date.eq(val)]
            designs,_,_=design(log,meta,tax,ref,sorted(train.sample_id.unique()))
            p,_=fitted(train,test,designs,LAMBDAS);inner.append(loss(p))
        # Select penalty separately for each target; evaluation remains outer
        # block held-out. No choice among chemical representations is made.
        risks=pd.concat(inner).groupby(['model','neuron_class','window','lambda']).mse.mean()
        best=risks.groupby(level=[0,1,2]).idxmin().map(lambda x:x[-1])
        train=a[a.date.ne(held)];test=a[a.date.eq(held)]
        assert not(set(train.sample_id)&set(test.sample_id))
        designs,loads,coverage=design(log,meta,tax,ref,sorted(train.sample_id.unique()),held)
        p,cf=fitted(train,test,designs,LAMBDAS,save_coef=True)
        for key,penalty in best.items():
            model,cell,window=key
            q=p[p.model.eq(model)&p.neuron_class.eq(cell)&p.window.eq(window)&p['lambda'].eq(penalty)]
            predictions.append(q)
            choices.append(dict(held_block=held,model=model,neuron_class=cell,window=window,lambda_chosen=penalty,n_train_strains=train.sample_id.nunique(),n_test_strains=test.sample_id.nunique(),n_features=designs[model].shape[1]))
        cf=pd.DataFrame(cf)
        choice=pd.DataFrame(choices).query('held_block==@held')
        all_coefs.append(cf.merge(choice[['held_block','model','neuron_class','window','lambda_chosen']],validate='many_to_one'))
        all_loadings.extend(loads)
        print(f'Finished chemical-class outer block {held}',flush=True)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(T/'chem_profiles_predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(T/'chem_profiles_choices.csv',index=False)
    pd.DataFrame(all_loadings).to_csv(T/'chem_profiles_pc_loadings.csv',index=False)
    pd.concat(all_coefs,ignore_index=True).to_csv(T/'chem_profiles_coefficients.csv',index=False)
    pd.DataFrame(coverage).to_csv(T/'chem_profiles_members.csv',index=False)
    summary,within,gain=checks(pred,tax)
    assert hashes=={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in names}
    # Comparison against the previous round verifies observed targets and
    # weights; penalty choice differs intentionally.
    old=pd.read_csv(ROOT/'reports/population_exploration_20260930/tables/chemistry_population_predictions.csv')
    keys=['sample_id','date','worm_key','neuron_class','window']
    v=pred[pred.model.eq('level')].merge(old[old.model.eq('level')][keys+['observed','weight','train_scale']],on=keys,validate='one_to_one',suffixes=('_new','_old'))
    maxerr={c:float(abs(v[c+'_new']-v[c+'_old']).max()) for c in ['observed','weight','train_scale']}
    assert max(maxerr.values())<1e-12
    methods=dict(status='success',windows=WINDOWS,neurons=NEURONS,lambdas=LAMBDAS,models=MODELS,n_strains=100,n_animals=a[['date','worm_key']].drop_duplicates().shape[0],excluded_strains=repeated,
        n_features=162,n_class_features=sum(c['n_features'] for c in coverage),n_classes=len(coverage),outer_folds=len(blocks),inner_folds=len(blocks)-1,
        transformations='Training-fold log chemistry feature mean/SD; 2 PC axes per metadata Class with >=3 features; alternative 25/50/75% quantiles of z minus per-strain median across 162 features. All derived columns standardized using training strains.',
        targets='Mean stimulus 0-10 s, post 10-30 s and full 0-40 s stored delta F/F0, centered within each observed animal and cell over retained strains.',
        weights='Each strain per cell per block total weight 1; trials first averaged within animal.',
        selection='5 ridge penalties nested separately using raw-unit per-target MSE by model, cell and window; no best representation selected for final metrics; exploratory evaluation, no p-values.',
        interpretation='Class axes/quantiles are report-profile descriptors, not pathways, molecular concentrations, or causal mechanisms.',
        counterchecks='Held-out within-animal-and-genus centered predictions; fixed-prediction remove-one-strain gain vs taxonomy.',
        input_sha256=hashes,previous_target_comparison=maxerr,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (L/'chem_profiles_methods.json').write_text(json.dumps(methods,indent=2))
    print(summary.pivot(index=['neuron_class','window'],columns='model',values='r2').round(3).to_string())
    within_training_check(a,log,meta,tax,ref)
    print('Within genus/block:')
    print(pd.read_csv(T/'chem_profiles_within_genus_scores.csv').pivot(index=['neuron_class','window'],columns='model',values='r2').round(3).to_string())
    verify_outputs()


if __name__=='__main__':main()
