"""Population-first calcium response axes; no neuron or taxon preselection.

Unit: animal x strain x cell means of trials. All neurons/time bins stay together
in 200 synchronized animal split-halves. Axis/scale and strain coordinates are
fit in one half; predicting the other half measures animal repeatability for the
measured catalogue, not independent strain/date validation. Missing cells remain
missing; primary evaluation requires all 13 classes in BOTH animal-half means.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SOURCE=ROOT/'reports/exploration_20260929/tables'
T=OUT/'tables'; F=OUT/'figures'; L=OUT/'logs'
NEURONS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
SEED=2026093001
NSPLIT=200
RANKS=[1,2,3,4,5,8,13,20,40,65]

def scale_cells(x, variant, nbin):
    if variant=='raw': return np.ones(x.shape[1])
    sc=np.sqrt(np.mean(np.var(x.reshape(len(x),13,nbin),axis=0,ddof=1),axis=1))
    return np.repeat(np.maximum(sc,1e-8),nbin)

def orient(v):
    return v*np.where(v[np.arange(len(v)),np.abs(v).argmax(axis=1)]<0,-1,1)[:,None]

def fit(x):
    mu=x.mean(axis=0);u,s,v=np.linalg.svd(x-mu,full_matrices=False)
    return mu,s,orient(v)

def transform(x, scale, direction=False):
    z=x/scale
    if direction:
        norm=np.linalg.norm(z,axis=1,keepdims=True)
        z=z/np.maximum(norm,1e-12)
    return z

def main():
    rng=np.random.default_rng(SEED)
    c=pd.read_parquet(SOURCE/'animal_curves.parquet').reset_index()
    c['date']=c.date.astype(str)
    tax=pd.read_csv(SOURCE/'taxonomy.csv').set_index('sample_id')
    animals=c[['date','worm_key']].drop_duplicates().sort_values(['date','worm_key']).reset_index(drop=True)
    blocks=c[['sample_id','date']].drop_duplicates().sort_values(['date','sample_id']).reset_index(drop=True)
    ai={v:i for i,v in enumerate(animals.itertuples(index=False,name=None))}
    bi={v:i for i,v in enumerate(blocks.itertuples(index=False,name=None))}
    dimap={dt:np.flatnonzero(animals.date.eq(dt).to_numpy()) for dt in animals.date.unique()}
    matrices={}; rawmatrices={}
    for nbin in [5,8]:
        vals=np.column_stack([c[[str(t) for t in range(5*b,5*b+5)]].mean(axis=1) for b in range(nbin)])
        w=c[['sample_id','date','worm_key','neuron_class']].copy()
        for b in range(nbin):w[f'bin{b}']=vals[:,b]
        # Remove animal/cell offsets using the strains that animal actually saw.
        means=w.groupby(['date','worm_key','neuron_class'])[[f'bin{b}' for b in range(nbin)]].transform('mean')
        centered=vals-means.to_numpy()
        mat=np.full((len(blocks),13*nbin,len(animals)),np.nan); raw=mat.copy()
        for row,vr,vc in zip(c.itertuples(),vals,centered):
            i=bi[(row.sample_id,row.date)];j=NEURONS.index(row.neuron_class)*nbin;k=ai[(row.date,row.worm_key)]
            mat[i,j:j+nbin,k]=vc;raw[i,j:j+nbin,k]=vr
        matrices[nbin]=mat;rawmatrices[nbin]=raw
        if nbin==5:
            cols=[f'{n}_bin{b}' for n in NEURONS for b in range(nbin)]
            export=[]
            for k,a in animals.iterrows():
                use=np.isfinite(mat[:,:,k]).any(axis=1)
                z=pd.DataFrame(mat[use,:,k],columns=cols)
                z.insert(0,'sample_id',blocks.loc[use,'sample_id'].to_numpy());z.insert(1,'date',a.date);z.insert(2,'worm_key',a.worm_key);export.append(z)
            pd.concat(export,ignore_index=True).to_csv(T/'population_structure_animal_centered_65.csv',index=False)
    # Full-data axes are descriptive; never reused in held-out reconstruction.
    loading=[]; scores=[]; variances=[]; ref={}; contributions=[]
    for nbin,mat in matrices.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);mean=np.nanmean(mat,axis=2)
        valid=np.isfinite(mean).all(axis=1)
        for variant in ['raw','cell_scaled','direction']:
            sc=scale_cells(mean[valid],variant,nbin)
            x=transform(mean[valid],sc,variant=='direction');mu,s,v=fit(x);coord=(x-mu)@v.T
            ref[(nbin,variant)]=(sc,mu,v)
            for k in range(min(8,len(s))):
                en=coord[:,k]**2;top=np.argsort(en)[-3:]
                variances.append(dict(n_bins=nbin,variant=variant,axis=k+1,variance_fraction=s[k]**2/(s*s).sum(),
                    top3_score_energy_fraction=en[top].sum()/en.sum(),top3_strain_dates=';'.join(blocks.loc[valid].iloc[top].apply(lambda r:f'{r.sample_id}@{r.date}',axis=1))))
                for j,n in enumerate(NEURONS):
                    contributions.append(dict(n_bins=nbin,variant=variant,axis=k+1,neuron_class=n,loading_energy=float((v[k,j*nbin:(j+1)*nbin]**2).sum())))
                    for b in range(nbin):loading.append(dict(n_bins=nbin,variant=variant,axis=k+1,neuron_class=n,bin=b,start_sec=5*b,end_sec=5*b+5,loading=v[k,j*nbin+b],cell_scale=sc[j*nbin+b],raw_unit_direction=v[k,j*nbin+b]*sc[j*nbin+b]))
            q=blocks.loc[valid].copy();q['n_bins']=nbin;q['variant']=variant
            for k in range(8):q[f'axis{k+1}']=coord[:,k]
            scores.append(q)
    pd.DataFrame(loading).to_csv(T/'population_structure_loadings.csv',index=False)
    pd.DataFrame(contributions).to_csv(T/'population_structure_cell_contributions.csv',index=False)
    pd.DataFrame(variances).to_csv(T/'population_structure_variance.csv',index=False)
    pd.concat(scores,ignore_index=True).join(tax[['genus_clean','species_clean']],on='sample_id').to_csv(T/'population_structure_scores.csv',index=False)
    records=[]; stability=[]; axisrep=[]; outliers=[]; coverage=[]
    for split in range(NSPLIT):
        half=np.zeros(len(animals),bool)
        for ix in dimap.values():half[rng.choice(ix,len(ix)//2,replace=False)]=True
        for nbin,mat in matrices.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                h=[np.nanmean(mat[:,:,a],axis=2) for a in [half,~half]]
            valid=np.isfinite(h[0]).all(axis=1)&np.isfinite(h[1]).all(axis=1)
            h=[a[valid] for a in h];n=len(h[0]); kept=blocks[valid].copy()
            if nbin==5:
                coverage.append(dict(split=split,n_strain_dates=n,n_strains=kept.sample_id.nunique(),n_blocks=kept.date.nunique(),n_train_animals=int(half.sum()),n_test_animals=int((~half).sum())))
            for variant in ['raw','cell_scaled','direction']:
                errors={k:0. for k in RANKS if k<=13*nbin};null=0.; corr=[]
                for side,(a,b) in enumerate([h,h[::-1]]):
                    sc=scale_cells(a,variant,nbin);x=transform(a,sc,variant=='direction');y=transform(b,sc,variant=='direction')
                    mu,s,v=fit(x);null+=float(np.sum((y-mu)**2));xc=x-mu
                    for k in errors:
                        pred=mu+(xc@v[:k].T)@v[:k]
                        errors[k]+=float(np.sum((y-pred)**2))
                    for k in range(5):
                        xx=xc@v[k];yy=(y-mu)@v[k]
                        cr=np.corrcoef(xx,yy)[0,1]
                        keep_axis=np.ones(len(xx),bool);keep_axis[np.argsort(np.abs(xx))[-3:]]=False
                        cr_robust=np.corrcoef(xx[keep_axis],yy[keep_axis])[0,1]
                        axisrep.append(dict(split=split,side=side,n_bins=nbin,variant=variant,axis=k+1,n_strain_dates=n,score_correlation=cr,score_correlation_without_training_top3=cr_robust,training_variance_fraction=s[k]**2/(s*s).sum()))
                    if nbin==5:
                        # Compare axis subspaces in a common full-data scaling only
                        # for descriptive fit-stability, not predictive validation.
                        refsc,_,refv=ref[(nbin,variant)]
                        _,_,independent_v=fit(y)
                        independent_scale=scale_cells(b,variant,nbin)
                        _,_,independent_own_v=fit(transform(b,independent_scale,variant=='direction'))
                        for k in [1,2,3,5]:
                            physical=v[:k]*sc[None,:]/refsc[None,:]
                            q,_=np.linalg.qr(physical.T)
                            cos=np.linalg.svd(refv[:k]@q[:,:k],compute_uv=False)
                            independent_cos=np.linalg.svd(v[:k]@independent_v[:k].T,compute_uv=False)
                            own_physical=independent_own_v[:k]*independent_scale[None,:]/refsc[None,:]
                            own_q,_=np.linalg.qr(own_physical.T)
                            own_cos=np.linalg.svd(q[:,:k].T@own_q[:,:k],compute_uv=False)
                            stability.append(dict(split=split,side=side,variant=variant,rank=k,mean_squared_principal_cosine=np.mean(cos**2),worst_principal_cosine=cos.min(),independent_half_mean_squared_principal_cosine=np.mean(independent_cos**2),independent_half_worst_principal_cosine=independent_cos.min(),independent_own_scale_mean_squared_principal_cosine=np.mean(own_cos**2),independent_own_scale_worst_principal_cosine=own_cos.min()))
                        # Refutation: exclude the three largest TRAIN response
                        # contrasts; refit EVERYTHING, evaluate retained strains.
                        remove=np.argsort(np.linalg.norm(xc,axis=1))[-3:];keep=np.ones(n,bool);keep[remove]=False
                        sc2=scale_cells(a[keep],variant,nbin);xx=transform(a[keep],sc2,variant=='direction');yy=transform(b[keep],sc2,variant=='direction')
                        mm,ss,vv=fit(xx);den=float(np.sum((yy-mm)**2))
                        for k in [1,3,5,13,65]:
                            pred=mm+((xx-mm)@vv[:k].T)@vv[:k]
                            # Matched retained-set ordinary fit, in the robust
                            # fit's scales, avoids attributing new denominators
                            # or excluded hard cases to stability.
                            po=mu+(xc@v[:k].T)@v[:k]
                            po=po[keep]*sc[None,:]/sc2[None,:] if variant!='direction' else po[keep]
                            outliers.append(dict(split=split,side=side,variant=variant,rank=k,n_strain_dates=int(keep.sum()),
                                drop_top3_refit_R2=1-float(np.sum((yy-pred)**2))/den,
                                ordinary_same_subset_R2=1-float(np.sum((yy-po)**2))/den if variant!='direction' else np.nan,
                                excluded=';'.join(kept.iloc[remove].sample_id)))
                for k,sse in errors.items():records.append(dict(split=split,n_bins=nbin,variant=variant,rank=k,n_strain_dates=n,test_sse=sse,null_sse=null,cross_animal_R2=1-sse/null))
    cv=pd.DataFrame(records); cv.to_csv(T/'population_structure_reconstruction.csv',index=False)
    def summary(df,keys,col,prefix):
        return df.groupby(keys)[col].agg(median='median',p05=lambda x:x.quantile(.05),p95=lambda x:x.quantile(.95)).rename(columns=lambda x:prefix+'_'+x).reset_index()
    cs=summary(cv,['n_bins','variant','rank'],'cross_animal_R2','R2');cs.to_csv(T/'population_structure_reconstruction_summary.csv',index=False)
    ap=pd.DataFrame(axisrep);ap.to_csv(T/'population_structure_axis_reproducibility.csv',index=False)
    aps=summary(ap,['n_bins','variant','axis'],'score_correlation','correlation');aps.to_csv(T/'population_structure_axis_reproducibility_summary.csv',index=False)
    stability_df=pd.DataFrame(stability)
    stability_df.to_csv(T/'population_structure_subspace_stability.csv',index=False)
    stability_df.query('variant=="cell_scaled" and side==0')[['split','rank','independent_own_scale_mean_squared_principal_cosine','independent_own_scale_worst_principal_cosine']].rename(columns={'independent_own_scale_mean_squared_principal_cosine':'mean_sq_cos','independent_own_scale_worst_principal_cosine':'worst_cos'}).to_csv(T/'population_structure_independent_scales_diagnostic.csv',index=False)
    pd.DataFrame(outliers).to_csv(T/'population_structure_outlier_checks.csv',index=False)
    pd.DataFrame(coverage).to_csv(T/'population_structure_split_coverage.csv',index=False)
    # Focused figure: what are the leading whole-population directions? Color
    # gives weight, not measured dF/F0. All 13 x 5 features are shown.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    ld=pd.DataFrame(loading);vr=pd.DataFrame(variances)
    fig,axes=plt.subplots(1,2,figsize=(7.5,5.6),sharey=True,layout='constrained')
    q=ld[(ld.n_bins==5)&ld.variant.eq('cell_scaled')&ld.axis.le(2)]
    vmax=q.loading.abs().max()
    for k,ax in enumerate(axes,1):
        z=q[q.axis.eq(k)].pivot(index='neuron_class',columns='bin',values='loading').reindex(NEURONS)
        im=ax.imshow(z,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        vf=vr.query('n_bins==5 and variant=="cell_scaled" and axis==@k').variance_fraction.iloc[0]
        cr=aps.query('n_bins==5 and variant=="cell_scaled" and axis==@k').correlation_median.iloc[0]
        ax.set_title(f'Population axis {k}\n{vf:.0%} of full-data variance',fontsize=11)
        ax.set_xticks(range(5),['0–5','5–10','10–15','15–20','20–25'],rotation=45,ha='right');ax.set_xlabel('Time after stimulus onset (s)')
        ax.set_yticks(range(13),NEURONS);ax.axvline(1.5,color='k',lw=1)
    fig.colorbar(im,ax=axes,label='Axis weight (cell-scaled response)',shrink=.65)
    fig.suptitle('Whole-population response contrasts: 13 neurons × 5 time bins',fontsize=13)
    fig.supxlabel('Descriptive full-data axes; independently fitted axes can rotate/exchange',fontsize=8)
    fig.savefig(F/'population_structure_axes.png',dpi=220);fig.savefig(F/'population_structure_axes.pdf');plt.close(fig)
    # One question per figure: how many degrees of freedom survive animals?
    fig,ax=plt.subplots(figsize=(6.9,4.6),layout='constrained')
    for variant,color,label in [('raw','#676767','Measured calcium units'),('cell_scaled','#2171b5','Equal cell scaling'),('direction','#d95f02','Cell-scaled contrast direction')]:
        q=cs[(cs.n_bins==5)&cs.variant.eq(variant)&cs['rank'].le(13)]
        ax.plot(q['rank'],q.R2_median,'o-',color=color,label=label)
        ax.fill_between(q['rank'],q.R2_p05,q.R2_p95,color=color,alpha=.12)
    ax.axhline(0,color='.65',lw=.7);ax.set_xticks([1,2,3,4,5,8,13]);ax.set_xlabel('Number of population axes fitted in training animals')
    ax.set_ylabel('Cross-animal reconstruction $R^2$');ax.legend(frameon=False,fontsize=9)
    ax.set_title('A single response direction does not exhaust repeatable variation')
    fig.savefig(F/'population_structure_reconstruction.png',dpi=220);fig.savefig(F/'population_structure_reconstruction.pdf');plt.close(fig)
    # Direct trace evidence: entirely omit one animal, learn axes from other
    # animals, then select its top/bottom 3 strains using TRAIN scores only.
    # No reference axis or cell list guides selection. Largest absolute loading
    # is oriented positive; axis identity can exchange (recorded in the fits).
    selected=[];traces=[];looaxes=[];testcontrasts=[];direct_coverage=[]
    mat=matrices[5]; timecols=[str(t) for t in range(-5,40)]
    for held,animal in animals.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            train=np.nanmean(np.delete(mat,held,axis=2),axis=2)
        valid=np.isfinite(train).all(axis=1)
        sc=scale_cells(train[valid],'cell_scaled',5);mu,ss,v=fit(train[valid]/sc)
        candidates=np.flatnonzero(np.isfinite(mat[:,:,held]).any(axis=1)&valid)
        if len(candidates)<6:continue
        candidate_scores=(train[candidates]/sc-mu)@v[:2].T
        for axis in [1,2]:
            ix=axis-1
            for nidx,n in enumerate(NEURONS):
                looaxes.append(dict(held_animal=f'{animal.date}__{animal.worm_key}',axis=axis,neuron_class=n,loading_energy=float((v[ix,nidx*5:(nidx+1)*5]**2).sum())))
            order=np.argsort(candidate_scores[:,ix])
            sides={'Low':candidates[order[:3]],'High':candidates[order[-3:]]}
            for side,indices in sides.items():
                for ind in indices:
                    selected.append(dict(held_animal=f'{animal.date}__{animal.worm_key}',sample_id=blocks.iloc[ind].sample_id,date=animal.date,worm_key=animal.worm_key,axis=axis,side=side,training_score=candidate_scores[np.flatnonzero(candidates==ind)[0],ix],n_candidates=len(candidates)))
                strains=blocks.iloc[indices].sample_id.tolist()
                cc=c[c.date.eq(animal.date)&c.worm_key.eq(animal.worm_key)&c.sample_id.isin(strains)]
                counts=cc.groupby('neuron_class').sample_id.nunique()
                for neuron in NEURONS:
                    direct_coverage.append(dict(held_animal=f'{animal.date}__{animal.worm_key}',axis=axis,side=side,neuron_class=neuron,n_selected=3,n_observed=int(counts.get(neuron,0)),retained=counts.get(neuron,0)==3))
                complete_cells=counts[counts.eq(3)].index
                cc=cc[cc.neuron_class.isin(complete_cells)].groupby('neuron_class')[timecols].mean()
                for neuron,row in cc.iterrows():
                    for t in range(-5,40):traces.append(dict(held_animal=f'{animal.date}__{animal.worm_key}',axis=axis,side=side,neuron_class=neuron,time_sec=t,mean_dff=row[str(t)]))
    pd.DataFrame(direct_coverage).to_csv(T/'population_structure_held_animal_coverage.csv',index=False)
    tr=pd.DataFrame(traces)
    pairkeys=['held_animal','axis','neuron_class']
    paired_keys=tr.groupby(pairkeys).side.nunique().loc[lambda x:x.eq(2)].reset_index()[pairkeys]
    tr=tr.merge(paired_keys,on=pairkeys,how='inner')
    tr.to_csv(T/'population_structure_held_animal_curves.csv',index=False)
    pd.DataFrame(selected).to_csv(T/'population_structure_held_animal_selections.csv',index=False)
    pd.DataFrame(looaxes).to_csv(T/'population_structure_held_animal_loadings.csv',index=False)
    q=tr.pivot(index=['held_animal','axis','neuron_class','time_sec'],columns='side',values='mean_dff').dropna()
    q['difference']=q.High-q.Low;q=q.reset_index()
    for (animal,axis,n),group in q.groupby(['held_animal','axis','neuron_class']):
        for label,lo,hi in [('stim',0,10),('post',10,25),('full25',0,25)]:
            use=group.time_sec.ge(lo)&group.time_sec.lt(hi)
            testcontrasts.append(dict(held_animal=animal,axis=axis,neuron_class=n,window=label,difference=group.loc[use,'difference'].mean()))
    contrast=pd.DataFrame(testcontrasts);contrast.to_csv(T/'population_structure_held_animal_contrasts.csv',index=False)
    contrast.groupby(['axis','neuron_class','window']).difference.agg(n='size',mean='mean',median='median',n_positive=lambda x:x.gt(0).sum(),n_negative=lambda x:x.lt(0).sum()).to_csv(T/'population_structure_held_animal_contrast_summary.csv')
    # A direct evidence figure, with no acquisition dates and no named-cell
    # preselection: actual held-animal high-minus-low responses in all 13 cells.
    bins=q[q.time_sec.ge(0)&q.time_sec.lt(25)].copy();bins['bin']=bins.time_sec//5
    paired=bins.groupby(['held_animal','axis','neuron_class','bin']).difference.mean().reset_index()
    paired.to_csv(T/'population_structure_held_animal_bin_contrasts.csv',index=False)
    bin_summary=paired.groupby(['axis','neuron_class','bin']).difference.agg(n='size',mean='mean',median='median',p05=lambda x:x.quantile(.05),p95=lambda x:x.quantile(.95),n_positive=lambda x:x.gt(0).sum(),n_negative=lambda x:x.lt(0).sum()).reset_index()
    bin_summary.to_csv(T/'population_structure_held_animal_bin_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.8,6.8),sharey=True,layout='constrained')
    vmax=bin_summary['mean'].abs().max()
    counts=bin_summary.groupby('neuron_class')['n'].min()
    for axis,ax in enumerate(axes,1):
        heat=bin_summary[bin_summary.axis.eq(axis)].pivot(index='neuron_class',columns='bin',values='mean').reindex(NEURONS)
        im=ax.imshow(heat,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        for ni in range(13):
            for bb in range(5):
                val=heat.iloc[ni,bb];ax.text(bb,ni,f'{val:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(val)>.60*vmax else '#222222')
        ax.set_xticks(range(5),['0–5','5–10','10–15','15–20','20–25'],rotation=40,ha='right')
        ax.set_yticks(range(13),[f'{n} (n={counts[n]})' for n in NEURONS]);ax.axvline(1.5,color='#222222',lw=1)
        ax.set_xlabel('Time after stimulus onset (s)');ax.set_title(f'Population axis {axis}: high − low',fontsize=11)
    fig.colorbar(im,ax=axes,label='Held-animal mean difference (ΔF/F₀)',shrink=.7)
    fig.suptitle('All neurons, selected by population response in other animals',fontsize=12)
    fig.supxlabel('Within each test animal: mean of 3 high-score strains − mean of 3 low-score strains.\n'
                  'Average across test animals; n = animals; vertical line = stimulus end (10 s).',fontsize=8)
    fig.savefig(F/'population_structure_held_responses.png',dpi=220);fig.savefig(F/'population_structure_held_responses.pdf');plt.close(fig)
    # One focused diagnostic of adaptation: refit the complete population after
    # replacing trial means by first presentations or subsequent presentations.
    trial=pd.read_parquet(SOURCE/'trial_curves.parquet').reset_index()
    trial['date']=trial.date.astype(str)
    seq=trial.groupby(['sample_id','date','worm_key']).segment_index.transform('min')
    trialchecks=[];trialaxes=[]
    for variant,mask in [('first',trial.segment_index.eq(seq)),('later',trial.segment_index.gt(seq))]:
        z=trial.loc[mask].groupby(['sample_id','date','worm_key','neuron_class'])[timecols].mean().reset_index()
        for b in range(5):z[f'bin{b}']=z[[str(t) for t in range(5*b,5*b+5)]].mean(axis=1)
        bincols=[f'bin{b}' for b in range(5)]
        z[bincols]-=z.groupby(['date','worm_key','neuron_class'])[bincols].transform('mean')
        d=z.groupby(['sample_id','date','neuron_class'])[bincols].mean().unstack('neuron_class').swaplevel(0,1,axis=1).reindex(columns=pd.MultiIndex.from_product([NEURONS,bincols]))
        d=d.dropna();xx=d.to_numpy();sc=scale_cells(xx,'cell_scaled',5);mu,ss,v=fit(xx/sc)
        refsc,_,refv=ref[(5,'cell_scaled')]
        for k in [1,2,3,5]:
            phys=v[:k]*sc[None,:]/refsc[None,:];basis,_=np.linalg.qr(phys.T);cos=np.linalg.svd(refv[:k]@basis[:,:k],compute_uv=False)
            trialchecks.append(dict(trial_variant=variant,rank=k,n_complete_strain_blocks=len(d),mean_squared_principal_cosine=np.mean(cos**2),worst_principal_cosine=cos.min()))
        for axis in [1,2]:
            for ni,n in enumerate(NEURONS):trialaxes.append(dict(trial_variant=variant,axis=axis,neuron_class=n,loading_energy=(v[axis-1,ni*5:(ni+1)*5]**2).sum()))
    pd.DataFrame(trialchecks).to_csv(T/'population_structure_first_later_subspaces.csv',index=False)
    pd.DataFrame(trialaxes).to_csv(T/'population_structure_first_later_loadings.csv',index=False)
    meta={'seed':SEED,'splits':NSPLIT,'n_animals':len(animals),'n_strain_date_blocks':len(blocks),'n_strains':blocks.sample_id.nunique(),
          'source':str(SOURCE/'animal_curves.parquet'),'bins':'primary [0,25), five non-overlapping 5-second means; sensitivity [0,40), eight bins',
          'independence':'split whole animals within acquisition blocks; every strain/cell/time bin moves together',
          'centering':'each animal/cell/bin mean across available stimuli removed before split',
          'primary_scaling':'training strain-block SD pooled over five bins for each cell, one scale per cell',
          'missing':'no imputation; strain-block included only when all13 cells represented in each animal-half mean',
          'scope':'repeatable animal-level strain contrasts within measured stimulus sets, not independent strain/date replication',
          'selection':'2 axes shown for descriptive parsimony; rank1..65 assessed, no optimal rank significance claim; full-data axes not used in predictive evaluation',
          'direct_evidence':'require exactly 3 measured strains per side per neuron, and both sides present for contrasts; leave one whole animal out; learn cell scales and ordered PCA directions on other animals; select 3 high/3 low candidate strains by training scores; record held-animal original curves, no held-response selection',
          'first_later':'recompute population axes with first presentations or later presentations; same data, a sensitivity check, not independent validation',
          'axis_stability':'report both independently refitted test PCA in train scales and independently scaled halves mapped back to common full-data scales; neither establishes a unique biological module',
          'uncertainty':'5th–95th percentiles over synchronized random splits; descriptive sensitivity, not confidence intervals',
          'limitations':'animal centering refers to each animal stimulus set; axes are continuous combinations, not neuron interactions or discrete classes; cell scaling gives noisy small cells influence; arbitrary axis signs; time is calcium, not spikes'}
    (L/'population_structure_methods.json').write_text(json.dumps(meta,indent=2))
    aligned=OUT/'tables/aligned_neural_animal_5bins.parquet'
    check={'status':'passed','n_held_animals':tr.held_animal.nunique(),'n_exact_three_per_side_cells':int(pd.DataFrame(direct_coverage).retained.sum()),'partial_selected_cell_groups':int(pd.DataFrame(direct_coverage).n_observed.isin([1,2]).sum()),'matched_curve_sides':bool(tr.groupby(pairkeys).side.nunique().eq(2).all()),'independent_strain_validation':False}
    centered_export=pd.read_csv(T/'population_structure_animal_centered_65.csv',dtype={'date':str})
    feature_cols=[col for col in centered_export if '_bin' in col]
    max_centered=float(centered_export.groupby(['date','worm_key'])[feature_cols].mean().abs().max().max())
    assert max_centered<1e-10;check['max_abs_animal_center']=max_centered
    if aligned.exists():
        align=pd.read_parquet(aligned).reset_index();align['date']=align.date.astype(str)
        align_cols=[col for col in align if '__' in col]
        align[align_cols]-=align.groupby(['date','worm_key'])[align_cols].transform('mean')
        aa=align.set_index(['sample_id','date','worm_key'])[align_cols].sort_index().to_numpy()
        bb=centered_export.set_index(['sample_id','date','worm_key'])[feature_cols].sort_index().to_numpy()
        assert np.array_equal(np.isnan(aa),np.isnan(bb))
        err=float(np.nanmax(np.abs(aa-bb)));assert err<1e-10;check['aligned_raw_pipeline_max_abs_difference']=err
    (L/'population_structure_verification.json').write_text(json.dumps(check,indent=2))
    print(cs.to_string(index=False));print('\nAXIS REPEATABILITY\n'+aps.query('n_bins==5').to_string(index=False))

if __name__=='__main__':main()
