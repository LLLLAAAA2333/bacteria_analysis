"""Atlas contributions beyond ADF/ASH, conditional on sampled stimulus sets.

Animal x strain x (phase, neuron), dF/F0; all trials averaged per animal before
any classification. Whole-animal outer holdout; templates and scales are
training-only. Main comparisons use identical heldout animals/candidates.
Coarse encodings are deliberately distinct from relative-amplitude vectors.
No cross-date or new-strain validation; no confirmatory significance claims.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SRC=ROOT/'reports/exploration_20260929/tables'
NEURONS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
BASE=['ADF','ASH']
FEATURES=[(w,n) for w in ['stim','post'] for n in NEURONS]
SEED=2026093001


def array(d,animals,strains):
    out=np.full((len(animals),len(strains),len(FEATURES)),np.nan)
    for i,animal in enumerate(animals):
        sub=d[d.worm_key.eq(animal)].set_index(['sample_id','neuron_class'])
        for j,(window,neuron) in enumerate(FEATURES):
            out[i,:,j]=sub[window].reindex(pd.MultiIndex.from_product([strains,[neuron]])).to_numpy()
    return out


def classify(train,test,cols,mode,candidate_groups=None):
    """Expected accuracy under uniform tie breaking, one per candidate strain."""
    tr=train[:,:,cols].copy();te=test[:,cols].copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if mode=='scaled':
            mean=np.nanmean(tr,axis=(0,1));scale=np.maximum(np.nanstd(tr,axis=(0,1)),1e-6)
            tr=(tr-mean)/scale;te=(te-mean)/scale
        elif mode.startswith('state'):
            threshold=float(mode[5:])
            tr=np.where(np.isnan(tr),np.nan,(tr>threshold).astype(float)-(tr< -threshold).astype(float))
            te=(te>threshold).astype(float)-(te< -threshold).astype(float)
        centroid=np.nanmean(tr,axis=0)
    assert np.isfinite(centroid).all() and np.isfinite(te).all()
    distances=np.mean((te[:,None,:]-centroid[None,:,:])**2,axis=2)
    if candidate_groups is not None:
        groups=np.asarray(candidate_groups)
        distances=np.where(groups[:,None]==groups[None,:],distances,np.inf)
    tied=np.isclose(distances,distances.min(axis=1,keepdims=True),rtol=0,atol=1e-12)
    return np.diag(tied)/tied.sum(axis=1),tied.sum(axis=1)


def evaluate(train_data,test_data,variant,modes,genus_map=None):
    records=[]
    for date,d in train_data.groupby('date'):
        dt=test_data[test_data.date.eq(date)]
        animals=sorted(set(d.worm_key)&set(dt.worm_key));strains=sorted(set(d.sample_id)&set(dt.sample_id))
        values=array(d,animals,strains);vtest=array(dt,animals,strains)
        for i,animal in enumerate(animals):
            tr=np.delete(values,i,axis=0);te=vtest[i]
            rows=np.isfinite(te).any(axis=1);tr=tr[:,rows];te=te[rows];labels=np.array(strains)[rows]
            valid=np.isfinite(te).all(axis=0)&(np.isfinite(tr).sum(axis=0)>=2).all(axis=0)
            if variant=='common_cells':valid &= np.isfinite(tr).all(axis=(0,1))
            cells=[n for n in NEURONS if all(valid[j] for j,f in enumerate(FEATURES) if f[1]==n)]
            if not all(n in cells for n in BASE):continue
            sets={'ADF_ASH':BASE,'all_available':cells}
            # These extensions were chosen after the all-cell marginal audit;
            # report them explicitly as same-data discoveries, not prespecified
            # or independently validated panels.
            for name,extension in [('four_cell',['AWA','ASK']),('six_cell',['AWA','ASK','AWB','ASJ'])]:
                if all(n in cells for n in extension):sets[name]=BASE+extension
            for n in cells:
                if n not in BASE:sets['add_'+n]=BASE+[n]
                sets['omit_'+n]=[q for q in cells if q!=n]
                sets['only_'+n]=[n]
            for mode in modes:
                for name,neurons in sets.items():
                    cols=[j for j,(_,n) in enumerate(FEATURES) if n in neurons]
                    groups=None if genus_map is None else [genus_map[label] for label in labels]
                    score,ties=classify(tr,te,cols,mode,groups)
                    for row,(label,correct,tie) in enumerate(zip(labels,score,ties)):
                        candidate_count=len(labels) if groups is None else sum(g==groups[row] for g in groups)
                        if groups is not None and candidate_count<3:continue
                        records.append(dict(variant=variant,date=date,worm_key=animal,sample_id=label,
                            mode=mode,panel=name,correct=correct,n_ties=int(tie),n_cells=len(neurons),
                            available_cells=';'.join(cells),n_candidates=candidate_count))
    return pd.DataFrame(records)


def summarize(pred):
    animal=pred.groupby(['variant','mode','panel','date','worm_key'],as_index=False).agg(
        accuracy=('correct','mean'),n_strains=('sample_id','nunique'),n_cells=('n_cells','first'),
        tie_fraction=('n_ties',lambda x:np.mean(x>1)))
    animal.to_csv(OUT/'tables/atlas_roles_animal_accuracy.csv',index=False)
    rng=np.random.default_rng(SEED)
    rows=[]
    for (variant,mode),d in animal.groupby(['variant','mode']):
        p=d.pivot(index=['date','worm_key'],columns='panel',values='accuracy')
        comparisons=[('all_available','ADF_ASH','all_minus_base','all')]
        if 'four_cell' in p:
            comparisons.extend([('four_cell','ADF_ASH','four_minus_base','four'),
                                ('all_available','four_cell','all_minus_four','all')])
        if 'six_cell' in p:
            comparisons.extend([('six_cell','ADF_ASH','six_minus_base','six'),
                                ('all_available','six_cell','all_minus_six','all')])
        for n in NEURONS:
            if 'add_'+n in p: comparisons.append(('add_'+n,'ADF_ASH','addition',n))
            if 'omit_'+n in p:comparisons.append(('all_available','omit_'+n,'ablation',n))
            if 'only_'+n in p:comparisons.append(('only_'+n,'ADF_ASH','single_minus_base',n))
        for lhs,rhs,kind,cell in comparisons:
            pair=p[[lhs,rhs]].dropna();v=pair[lhs].to_numpy()-pair[rhs].to_numpy()
            b=np.zeros(2000)
            dates=pair.index.get_level_values(0).to_numpy()
            for block in np.unique(dates):
                x=v[dates==block];b+=rng.choice(x,(2000,len(x)),replace=True).sum(axis=1)
            b/=len(v);lo,hi=np.quantile(b,[.025,.975])
            support=d[d.panel.eq(lhs)].set_index(['date','worm_key']).reindex(pair.index)
            source=pred[(pred.variant.eq(variant))&(pred['mode'].eq(mode))&(pred.panel.eq(lhs))]
            source=source.set_index(['date','worm_key']).loc[pair.index].reset_index()
            rows.append(dict(variant=variant,mode=mode,kind=kind,cell=cell,lhs=lhs,rhs=rhs,
                lhs_accuracy=pair[lhs].mean(),rhs_accuracy=pair[rhs].mean(),delta=v.mean(),
                descriptive_low=lo,descriptive_high=hi,n_animals=len(v),n_strains=source.sample_id.nunique(),
                n_animal_strain=int(support.n_strains.sum()),n_blocks=len(np.unique(dates)),
                improved=int((v>1e-12).sum()),worse=int((v< -1e-12).sum()),ties=int((np.abs(v)<=1e-12).sum()),
                positive_blocks=int(pd.Series(v,index=pair.index).groupby(level=0).mean().gt(0).sum())))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'tables/atlas_roles_matched_comparisons.csv',index=False)
    return animal,result


def main():
    a=pd.read_csv(SRC/'animal_metrics.csv')
    results=[evaluate(a,a,'primary',['raw','scaled','state0','state0.05','state0.1'])]
    taxonomy=pd.read_csv(SRC/'taxonomy.csv').set_index('sample_id').genus_clean.to_dict()
    results.append(evaluate(a,a,'within_genus',['raw','state0.05'],taxonomy))
    trials=pd.read_parquet(SRC/'trial_curves.parquet').reset_index();trials['date']=trials.date.astype(int)
    key=['sample_id','date','worm_key','neuron_class']
    trials['rank']=trials.groupby(key).segment_index.rank(method='first')
    trials['stim']=trials[[str(x) for x in range(0,10)]].mean(axis=1)
    trials['post']=trials[[str(x) for x in range(10,30)]].mean(axis=1)
    first=trials[trials['rank'].eq(1)].groupby(key,as_index=False)[['stim','post']].mean()
    later=trials[trials['rank'].gt(1)].groupby(key,as_index=False)[['stim','post']].mean()
    order=trials.groupby(key,as_index=False).segment_index.mean()
    detrended=a.merge(order,on=key,validate='one_to_one')
    for _,indices in detrended.groupby(['date','worm_key','neuron_class']).groups.items():
        sub=detrended.loc[indices];x=sub.segment_index.to_numpy();x=x-x.mean()
        values=sub[['stim','post']].to_numpy()
        if x@x>0:detrended.loc[indices,['stim','post']]=values-x[:,None]*(x@values/(x@x))
    for variant,train,test in [('first_to_later',first,later),('later_to_first',later,first),
                             ('common_cells',a,a),('linear_order_removed',detrended,detrended)]:
        results.append(evaluate(train,test,variant,['raw','scaled','state0.05']))
    pred=pd.concat(results,ignore_index=True)
    pred.to_csv(OUT/'tables/atlas_roles_predictions.csv',index=False)
    within=pred[pred.variant.eq('within_genus')&pred.panel.isin(['four_cell','ADF_ASH'])].copy()
    within['genus_clean']=within.sample_id.map(taxonomy)
    within=within.pivot(index=['mode','date','worm_key','sample_id','genus_clean'],columns='panel',values='correct').dropna().reset_index()
    within['delta']=within.four_cell-within.ADF_ASH
    within.to_csv(OUT/'tables/atlas_roles_within_genus_paired.csv',index=False)
    animal,result=summarize(pred)
    # Figure is an exploratory audit of marginal contribution, never a claim
    # that identification accuracy itself explains a biological mechanism.
    fig,axes=plt.subplots(1,2,figsize=(10.5,5.0),sharey=True)
    order=[n for n in NEURONS if n not in BASE]
    colors={'raw':'#395d92','scaled':'#c17b34','state0.05':'#43896c'}
    jitter=np.random.default_rng(SEED)
    raw_animal=animal[animal.variant.eq('primary')&animal['mode'].eq('raw')].pivot(
        index=['date','worm_key'],columns='panel',values='accuracy')
    for ax,kind,title in zip(axes,['addition','ablation'],['Add one neuron to ADF + ASH','Remove one neuron from the atlas']):
        for mi,(mode,color) in enumerate(colors.items()):
            s=result[(result.variant.eq('primary'))&(result['mode'].eq(mode))&(result.kind.eq(kind))].set_index('cell')
            for yi,n in enumerate(order):
                if n not in s.index:continue
                row=s.loc[n];y=yi+(mi-1)*.18
                if mode=='raw':
                    lhs,rhs=('add_'+n,'ADF_ASH') if kind=='addition' else ('all_available','omit_'+n)
                    individual=(raw_animal[lhs]-raw_animal[rhs]).dropna().to_numpy()*100
                    ax.scatter(individual,y+jitter.uniform(-.08,.08,len(individual)),s=8,color='.6',alpha=.4,zorder=1)
                ax.plot([row.descriptive_low*100,row.descriptive_high*100],[y,y],color=color,lw=1)
                ax.plot(row.delta*100,y,'o',color=color,ms=4,label={'raw':'Raw response vector','scaled':'Training-scaled','state0.05':'Three states (±0.05)'}[mode] if yi==0 else None)
        ax.axvline(0,color='.65',lw=.7);ax.set_title(title,fontsize=11)
        ax.set_xlabel('Change in correct identification (percentage points)')
        ax.set_xlim(-40,40)
        ax.spines[['top','right']].set_visible(False)
    raw_support=result[result.variant.eq('primary')&result['mode'].eq('raw')&result.kind.eq('addition')].set_index('cell')
    axes[0].set_yticks(range(len(order)),[f'{n} (n={int(raw_support.loc[n,"n_animals"])})' for n in order]);axes[0].invert_yaxis();axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    fig.suptitle('Which neurons complement ADF and ASH?',fontsize=13)
    fig.text(.02,.015,'Whole-animal holdout. Gray dots: animals, raw vector. Colored points: means; bars: descriptive 95% resampling ranges.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.96));fig.savefig(OUT/'figures/atlas_roles_incremental.png',dpi=220);fig.savefig(OUT/'figures/atlas_roles_incremental.pdf');plt.close(fig)
    (OUT/'logs/atlas_roles_methods.json').write_text(json.dumps(dict(seed=SEED,
        phases={'stim':'0–9 s mean','post':'10–29 s mean'},
        units='dF/F0; each biological animal is (date,worm_key); trial means before model',
        support='Every pair uses identical candidates and heldout animals; ADF and ASH must both be observed; each cell has >=2 training animals per candidate.',
        state_encoding='-1 below -threshold; 0 within threshold; +1 above threshold. Encode each training animal before calculating the soft template. No molecular or electrophysiological meaning is asserted.',
        tie_break='Expected accuracy under uniform random choice among equal closest templates.',
        resampling='2000 fixed-prediction whole-animal resamples stratified by block; descriptive ranges, not algorithm-refitted confidence intervals or independent validation.',
        selection='All 13 cells assessed; addition and ablation exploratory. Four/six-cell panels chosen after examining these same data. No p values or FDR. 0, 0.05, 0.1 thresholds all reported, no best threshold selected.',
        within_genus='Candidate pool restricted to true genus, at least 3 candidate strains in that block; training templates/scales unchanged. Descriptive test of discrimination beyond genus, not an independently validated genus classifier.',
        limitations='Stimulus identities and many presentation schedules repeat across training/test animals. This asks sampled-set animal transfer, not new date/strain generalization; cannot distinguish every carryover explanation.'),ensure_ascii=False,indent=2))
    print(result[(result.variant.eq('primary'))&(result.kind.isin(['all_minus_base','addition']))].round(4).to_string(index=False))
    assert not pred.duplicated(['variant','mode','panel','date','worm_key','sample_id']).any()
    assert pred.correct.between(0,1).all() and pred.n_ties.ge(1).all()
    new=pred[pred.variant.eq('primary')&pred['mode'].eq('raw')&pred.panel.eq('all_available')]
    assert new.n_candidates.between(11,13).all()
    old=pd.read_csv(ROOT/'reports/population_exploration_20260930/tables/information_predictions.csv')
    old=old[old.window.eq('both')&old['mode'].eq('raw')]
    matched=new.merge(old,on=['date','worm_key','sample_id'],validate='one_to_one',suffixes=('_new','_old'))
    assert len(matched)==499 and (matched.correct_new==matched.correct_old).all()
    (OUT/'logs/atlas_roles_verification.json').write_text(json.dumps(dict(status='passed',duplicate_rows=0,
        probability_bounds='passed',raw_atlas_agrees_with_previous_heldout_predictions=len(matched),
        main_candidate_range=[int(new.n_candidates.min()),int(new.n_candidates.max())],
        matched_support_animals=len(new[['date','worm_key']].drop_duplicates()),
        main_shapes=[len(pred),len(result)],verification_limits='Arithmetic/support checks; not biological identifiability.'),indent=2))


if __name__=='__main__':main()
