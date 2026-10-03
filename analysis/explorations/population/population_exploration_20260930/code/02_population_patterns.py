"""Exploratory population response composition, in measured calcium units.

Read-only inputs from the verified previous round. No p values. Candidate
features/taxa are selected after examining all 13 neuron classes. Animal splits
keep each (date,worm_key) together across strains, neurons and time summaries.
"""
from pathlib import Path
import json,hashlib,warnings
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SOURCE=ROOT/'reports/exploration_20260929/tables'
T=OUT/'tables'
SEED=2026093002
NSPLIT=300
NBOOT=4000
NEURONS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
METRICS=['stim','post','full']
GROUPS=['Bacteroides','Bifidobacterium','Escherichia','Lactobacillus']
PAIRS=list(combinations(GROUPS,2))


def pca(x):
    xc=x-x.mean(axis=0)
    _,s,v=np.linalg.svd(xc,full_matrices=False)
    return s*s/(s*s).sum(),v,xc@v.T


def main():
    rng=np.random.default_rng(SEED)
    a=pd.read_csv(SOURCE/'animal_metrics.csv',dtype={'date':str})
    tax=pd.read_csv(SOURCE/'taxonomy.csv').set_index('sample_id')
    tax['species_clean']=tax.species_clean.str.strip()
    curves=pd.read_parquet(SOURCE/'animal_curves.parquet').reset_index()
    a=a.join(tax[['genus_clean','species_clean']],on='sample_id')
    curves=curves.join(tax[['genus_clean','species_clean']],on='sample_id')
    date=a.groupby(['sample_id','date','neuron_class'])[METRICS].mean()
    strain=date.groupby(['sample_id','neuron_class']).mean()
    population=pd.concat({w:strain[w].unstack('neuron_class').reindex(columns=NEURONS) for w in METRICS},axis=1)
    population.columns=[f'{w}_{n}' for w,n in population.columns]
    population.join(tax[['genus_clean','species_clean']]).to_csv(T/'population_strain_profiles.csv')
    correlations=[]
    for w in METRICS:
        cc=strain[w].unstack('neuron_class')[NEURONS].corr()
        for n1,n2 in combinations(NEURONS,2):
            correlations.append(dict(metric=w,neuron_a=n1,neuron_b=n2,strain_mean_pearson=cc.loc[n1,n2],n_strains=len(population)))
    pd.DataFrame(correlations).to_csv(T/'population_neuron_correlations.csv',index=False)
    animal=a.pivot(index=['sample_id','date','worm_key','genus_clean','species_clean'],columns='neuron_class',values=METRICS)
    animal.columns=[f'{w}_{n}' for w,n in animal.columns]
    animal.to_csv(T/'population_animal_profiles.csv')
    # Raw-unit PCA is primary. SD and animal-residual scaling are diagnostic,
    # since unit variance can magnify low-amplitude/noisy classes.
    variance=[];loadings=[];score_tables=[]
    residual=a[METRICS]-a.groupby(['sample_id','date','neuron_class'])[METRICS].transform('mean')
    residual['neuron_class']=a.neuron_class
    noise=residual.groupby('neuron_class')[METRICS].std().reindex(NEURONS)
    for metric in METRICS:
        x=population[[f'{metric}_{n}' for n in NEURONS]].to_numpy()
        for scaling in ['raw','unit_sd','animal_residual_sd']:
            scale=np.ones(13) if scaling=='raw' else (x.std(axis=0,ddof=1) if scaling=='unit_sd' else noise[metric].to_numpy())
            vr,v,scores=pca(x/scale)
            for k in range(13):
                top=np.argsort(np.abs(scores[:,k]))[-3:]
                variance.append(dict(metric=metric,scaling=scaling,component=k+1,variance_fraction=vr[k],
                    top3_strain_score_variance_fraction=float((scores[top,k]**2).sum()/(scores[:,k]**2).sum()),
                    top3_strains=';'.join(population.index[top])))
                for j,n in enumerate(NEURONS):
                    loadings.append(dict(metric=metric,scaling=scaling,component=k+1,neuron_class=n,loading=v[k,j],scale=scale[j]))
            if scaling=='raw':
                pp=pd.DataFrame(scores,index=population.index,columns=[f'PC{k+1}' for k in range(13)])
                pp['metric']=metric;score_tables.append(pp.reset_index())
    pd.DataFrame(variance).to_csv(T/'population_pca_variance.csv',index=False)
    pd.DataFrame(loadings).to_csv(T/'population_pca_loadings.csv',index=False)
    pd.concat(score_tables,ignore_index=True).to_csv(T/'population_pca_scores.csv',index=False)

    # Cross-animal reconstruction for the same measured strain catalogue.
    # A common-profile gain model predicts alpha_strain * mean_response_profile.
    # PCA rank k allows k independent response-composition axes around the mean.
    # Every model is fitted to one animal half, evaluated in the other, then
    # directions are swapped. No axis or rank is reselected using held-out y.
    sd=a[['sample_id','date']].drop_duplicates().sort_values(['date','sample_id']).reset_index(drop=True)
    keys=list(sd.itertuples(index=False,name=None));key_lookup={k:i for i,k in enumerate(keys)}
    ids=a[['date','worm_key']].drop_duplicates().sort_values(['date','worm_key']).reset_index(drop=True)
    lookup={k:i for i,k in enumerate(ids.itertuples(index=False,name=None))}
    matrices={w:np.full((len(sd),13,len(ids)),np.nan) for w in METRICS}
    for row in a.itertuples():
        i=key_lookup[(row.sample_id,row.date)];j=NEURONS.index(row.neuron_class);k=lookup[(row.date,row.worm_key)]
        for w in METRICS:matrices[w][i,j,k]=getattr(row,w)
    date_indices={dt:np.flatnonzero(ids.date.eq(dt).to_numpy()) for dt in ids.date.unique()}
    cv=[]
    for split in range(NSPLIT):
        half=np.zeros(len(ids),bool)
        for ix in date_indices.values():half[rng.choice(ix,size=len(ix)//2,replace=False)]=True
        for metric,mat in matrices.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                halves=[np.nanmean(mat[:,:,mask],axis=2) for mask in [half,~half]]
            valid=np.isfinite(halves[0]).all(axis=1)&np.isfinite(halves[1]).all(axis=1)
            x,y=[z[valid] for z in halves];observed_dates=sd.date.to_numpy()[valid]
            for remove_date_mean in [False,True]:
                xx=x.copy();yy=y.copy()
                if remove_date_mean:
                    for dt in np.unique(observed_dates):
                        ix=observed_dates==dt;xx[ix]-=xx[ix].mean(axis=0);yy[ix]-=yy[ix].mean(axis=0)
                errors={f'rank_{k}':0. for k in range(1,14)}
                if not remove_date_mean:
                    errors['common_profile_gain']=0.
                    errors['best_single_profile']=0.
                null=0.
                for train,test in [(xx,yy),(yy,xx)]:
                    mean=train.mean(axis=0);center=train-mean
                    _,_,v=np.linalg.svd(center,full_matrices=False)
                    null+=float(np.sum((test-mean)**2))
                    for k in range(1,14):
                        pred=mean+(center@v[:k].T)@v[:k]
                        if k==13:assert np.allclose(pred,train,atol=1e-12,rtol=1e-12)
                        errors[f'rank_{k}']+=float(np.sum((test-pred)**2))
                    if not remove_date_mean:
                        pred=(train@mean/(mean@mean))[:,None]*mean
                        errors['common_profile_gain']+=float(np.sum((test-pred)**2))
                        # More generous null: estimate the common profile by
                        # uncentered SVD rather than forcing it to the mean.
                        _,_,vraw=np.linalg.svd(train,full_matrices=False)
                        pred=(train@vraw[0])[:,None]*vraw[0]
                        errors['best_single_profile']+=float(np.sum((test-pred)**2))
                for model,sse in errors.items():
                    cv.append(dict(split=split,metric=metric,date_centered_diagnostic=remove_date_mean,model=model,
                        n_strain_dates=int(valid.sum()),n_dates=len(np.unique(observed_dates)),test_sse=sse,null_sse=null,
                        cross_animal_R2=1-sse/null))
    cv=pd.DataFrame(cv);cv.to_csv(T/'population_reconstruction_draws.csv',index=False)
    assert len(cv)==NSPLIT*len(METRICS)*(13+13+2)
    cv_summary=cv.groupby(['metric','date_centered_diagnostic','model']).agg(
        R2_median=('cross_animal_R2','median'),R2_p05=('cross_animal_R2',lambda x:x.quantile(.05)),
        R2_p95=('cross_animal_R2',lambda x:x.quantile(.95)),n_strain_dates_min=('n_strain_dates','min'),
        n_strain_dates_median=('n_strain_dates','median'),n_strain_dates_max=('n_strain_dates','max')).reset_index()
    cv_summary.to_csv(T/'population_reconstruction_summary.csv',index=False)

    # All genera with >=3 measured strains were inspected. Four groups below
    # were chosen because they provide distinct, interpretable ADF/ASH patterns
    # and some same-animal comparisons; choices are explicitly post hoc.
    slong=strain.reset_index().join(tax[['genus_clean','species_clean']],on='sample_id')
    genus=slong.groupby(['genus_clean','neuron_class'])[METRICS].agg(['mean','median','min','max','size'])
    genus.to_csv(T/'population_all_genus_profiles.csv')
    supports=[]
    for group in GROUPS:
        for neuron in ['ADF','ASH','AWA','AWCON']:
            q=slong[slong.genus_clean.eq(group)&slong.neuron_class.eq(neuron)]
            obs=a[a.genus_clean.eq(group)&a.neuron_class.eq(neuron)]
            for w in METRICS:
                v=q[w].to_numpy()
                loo=np.array([np.delete(v,i).mean() for i in range(len(v))])
                supports.append(dict(genus_clean=group,neuron_class=neuron,metric=w,n_strains=len(q),n_species=q.species_clean.nunique(),
                    n_dates=obs.date.nunique(),n_true_animals=len(obs[['date','worm_key']].drop_duplicates()),n_animal_strain_rows=len(obs),
                    mean=v.mean(),median=np.median(v),min=v.min(),max=v.max(),n_positive_strains=int((v>0).sum()),n_negative_strains=int((v<0).sum()),
                    leave_one_strain_mean_min=loo.min(),leave_one_strain_mean_max=loo.max(),
                    n_positive_animal_strain_rows=int(obs[w].gt(0).sum()),n_negative_animal_strain_rows=int(obs[w].lt(0).sum())))
    support=pd.DataFrame(supports);support.to_csv(T/'population_selected_group_support.csv',index=False)

    # Contrasts are computed within exactly the same animals, averaging strains
    # within each genus first; neurons/trials do not provide independent n.
    groupmeans=a[a.genus_clean.isin(GROUPS)].groupby(['date','worm_key','genus_clean','neuron_class'])[METRICS].mean()
    contrast=[]
    for g1,g2 in PAIRS:
        x=groupmeans.xs(g1,level='genus_clean');y=groupmeans.xs(g2,level='genus_clean')
        both=x.join(y,how='inner',lsuffix='_a',rsuffix='_b')
        for (dt,worm,neuron),row in both.iterrows():
            for metric in METRICS:
                contrast.append(dict(group_a=g1,group_b=g2,date=dt,worm_key=worm,neuron_class=neuron,metric=metric,
                    mean_a=row[metric+'_a'],mean_b=row[metric+'_b'],difference=row[metric+'_a']-row[metric+'_b']))
    contrasts=pd.DataFrame(contrast);contrasts.to_csv(T/'population_paired_animal_contrasts.csv',index=False)
    summaries=[]
    # One global set of within-date animal weights is shared by all contrasts.
    boot_weights=np.zeros((NBOOT,len(ids)),int)
    for ix in date_indices.values():
        draws=rng.integers(0,len(ix),size=(NBOOT,len(ix)))
        for b in range(NBOOT):np.add.at(boot_weights[b],ix[draws[b]],1)
    for key,q in contrasts.groupby(['group_a','group_b','neuron_class','metric']):
        date_results=[]
        for dt,g in q.groupby('date'):
            ix=np.array([lookup[(dt,worm)] for worm in g.worm_key]);ww=boot_weights[:,ix]
            numerator=ww@g.difference.to_numpy();denominator=ww.sum(axis=1)
            with np.errstate(divide='ignore',invalid='ignore'):date_results.append(numerator/denominator)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);draw=np.nanmean(date_results,axis=0)
        finite=np.isfinite(draw);ci=np.quantile(draw[finite],[.025,.975])
        summaries.append(dict(zip(['group_a','group_b','neuron_class','metric'],key),
            equal_date_mean_difference=q.groupby('date').difference.mean().mean(),low=ci[0],high=ci[1],
            n_dates=q.date.nunique(),n_true_animals=len(q[['date','worm_key']].drop_duplicates()),n_boot_finite=int(finite.sum())))
    pd.DataFrame(summaries).to_csv(T/'population_paired_contrast_summary.csv',index=False)
    coverage=a[a.genus_clean.isin(GROUPS)][['sample_id','date','worm_key','genus_clean','species_clean']].drop_duplicates()
    coverage.to_csv(T/'population_group_design.csv',index=False)

    # Check whether the selected stimulus sign difference survives already
    # computed trial median, first/later, order and trace-QC variants.
    ph=pd.read_csv(SOURCE/'phenotype_animal_sensitivity.csv',dtype={'date':str})
    variants=['mean','median','first','later','qc_exclude_below_minus1','order_adjusted']
    check=[]
    for group in ['Escherichia','Lactobacillus','Bacteroides','Bifidobacterium']:
        for neuron in ['ADF','ASH']:
            q=ph[ph.genus_clean.eq(group)&ph.neuron_class.eq(neuron)]
            for variant in variants:
                bystrain=q.groupby(['sample_id','date'])[variant].mean().groupby('sample_id').mean()
                check.append(dict(genus_clean=group,neuron_class=neuron,variant=variant,n_strains=len(bystrain),
                    mean=bystrain.mean(),median=bystrain.median(),n_positive=int(bystrain.gt(0).sum()),n_negative=int(bystrain.lt(0).sum()),
                    min=bystrain.min(),max=bystrain.max()))
    pd.DataFrame(check).to_csv(T/'population_variant_checks.csv',index=False)
    # Store close-to-raw shared-animal curves for the diagnostic E/L contrast.
    joint=curves[curves.genus_clean.isin(['Escherichia','Lactobacillus'])].groupby(['date','worm_key','genus_clean','neuron_class'])[[str(t) for t in range(-5,40)]].mean()
    ec=joint.xs('Escherichia',level='genus_clean');lc=joint.xs('Lactobacillus',level='genus_clean');common=ec.index.intersection(lc.index)
    paired_curves=pd.concat({'Escherichia':ec.loc[common],'Lactobacillus':lc.loc[common]},names=['genus_clean']).reset_index()
    paired_curves.to_csv(T/'population_escher_lacto_shared_curves.csv',index=False)
    # The composition figure uses the identical animals for BOTH neurons.
    adfkeys=set(map(tuple,paired_curves[paired_curves.neuron_class.eq('ADF')][['date','worm_key']].to_numpy()))
    ashkeys=set(map(tuple,paired_curves[paired_curves.neuron_class.eq('ASH')][['date','worm_key']].to_numpy()))
    jointkeys=adfkeys&ashkeys
    joint_curves=paired_curves[paired_curves.neuron_class.isin(['ADF','ASH']) &
        pd.MultiIndex.from_frame(paired_curves[['date','worm_key']]).isin(jointkeys)]
    joint_curves.to_csv(T/'population_escher_lacto_joint_curves.csv',index=False)
    assert joint_curves.groupby(['genus_clean','neuron_class']).size().nunique()==1
    joint_contrasts=contrasts[contrasts.group_a.eq('Escherichia')&contrasts.group_b.eq('Lactobacillus')&
        contrasts.neuron_class.isin(['ADF','ASH']) & pd.MultiIndex.from_frame(contrasts[['date','worm_key']]).isin(jointkeys)]
    joint_contrasts.to_csv(T/'population_joint_adf_ash_contrasts.csv',index=False)
    # The same joint-animal comparison under trial/order/QC response summaries.
    joint_variant=[]
    pg=ph[ph.genus_clean.isin(['Escherichia','Lactobacillus'])&ph.neuron_class.isin(['ADF','ASH'])]
    pg=pg[pd.MultiIndex.from_frame(pg[['date','worm_key']]).isin(jointkeys)]
    pv=pg.groupby(['date','worm_key','neuron_class','genus_clean'])[variants].mean()
    ve=pv.xs('Escherichia',level='genus_clean');vl=pv.xs('Lactobacillus',level='genus_clean')
    common=ve.index.intersection(vl.index)
    for (dt,worm,neuron) in common:
        for variant in variants:
            joint_variant.append(dict(date=dt,worm_key=worm,neuron_class=neuron,variant=variant,
                escherichia_mean=ve.loc[(dt,worm,neuron),variant],lactobacillus_mean=vl.loc[(dt,worm,neuron),variant],
                difference=ve.loc[(dt,worm,neuron),variant]-vl.loc[(dt,worm,neuron),variant]))
    pd.DataFrame(joint_variant).to_csv(T/'population_joint_variant_contrasts.csv',index=False)
    plot(population.join(tax[['genus_clean']]),joint_curves)
    inputs=[SOURCE/n for n in ['animal_metrics.csv','animal_curves.parquet','taxonomy.csv','phenotype_animal_sensitivity.csv']]
    report=dict(seed=SEED,n_splits=NSPLIT,n_boot=NBOOT,windows={'stim':[0,10],'post':[10,30],'full':[0,40]},
        examined='All 13 neurons; raw, unit-SD and animal-residual-SD PCA; all genera with >=3 strains inspected.',
        chosen_groups=GROUPS,selection='Post hoc phenotype prioritization after full-population inspection; ADF and ASH are separable response axes, not claimed synergistic.',
        scope='Cross-animal agreement conditional on measured strains and cohorts; not independent validation or new-date prediction.',
        bootstrap='Whole animals resampled within observed dates with shared weights across all contrasts; 95% percentile intervals conditional on dates and selected groups.',
        split_intervals='5–95% percentiles over animal partitions, not confidence intervals.',
        unknown='No calcium-to-spiking conversion; taxonomy cannot establish a causal stimulus property; shared stimulus order and nonlinear carryover remain alternatives despite trial/order sensitivity checks.',
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'logs/population_methods.json').write_text(json.dumps(report,indent=2))
    print('Cross-animal reconstruction, raw units:')
    print(cv_summary[(~cv_summary.date_centered_diagnostic)&cv_summary.model.isin(['common_profile_gain','best_single_profile','rank_1','rank_2','rank_3','rank_5','rank_13'])].round(4).to_string(index=False))
    print('\nSelected stimulus group support:')
    print(support[support.metric.eq('stim')&support.neuron_class.isin(['ADF','ASH'])].round(4).to_string(index=False))
    print('\nWithin-animal Escherichia minus Lactobacillus:')
    ss=pd.DataFrame(summaries);print(ss[ss.group_a.eq('Escherichia')&ss.group_b.eq('Lactobacillus')&ss.neuron_class.isin(['ADF','ASH'])].round(4).to_string(index=False))
    print('\nAll population analyses completed successfully.')


def plot(profiles,paired_curves):
    colors={'Bacteroides':'#177c68','Bifidobacterium':'#9556a4','Escherichia':'#2668a3','Lactobacillus':'#d57a20'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={'width_ratios':[1.25,1,1]},layout='constrained')
    ax=axes[0];other=profiles[~profiles.genus_clean.isin(GROUPS)]
    ax.scatter(other.stim_ADF,other.stim_ASH,color='#bbbbbb',s=21,alpha=.6,label=f'Other genera ({len(other)} strains)')
    for group in GROUPS:
        q=profiles[profiles.genus_clean.eq(group)]
        ax.scatter(q.stim_ADF,q.stim_ASH,s=35,color=colors[group],alpha=.86,label=f'{group} ({len(q)})',edgecolor='white',lw=.35)
    for sid in ['A077','A065','A138']:
        q=profiles.loc[sid];ax.annotate(sid,(q.stim_ADF,q.stim_ASH),xytext=(4,-11 if sid=='A065' else 5),textcoords='offset points',fontsize=8)
    ax.axhline(0,color='.65',lw=.75);ax.axvline(0,color='.65',lw=.75)
    ax.set(xlabel=r'ADF stimulus mean, $\Delta F/F_0$',ylabel=r'ASH stimulus mean, $\Delta F/F_0$',title='A  Two separable response axes')
    ax.legend(frameon=False,fontsize=7.8,loc='upper right')
    time=np.arange(-5,40)
    for ax,neuron in zip(axes[1:],['ADF','ASH']):
        for group in ['Escherichia','Lactobacillus']:
            q=paired_curves[paired_curves.genus_clean.eq(group)&paired_curves.neuron_class.eq(neuron)]
            values=q[[str(t) for t in time]].to_numpy()
            for v in values:ax.plot(time,v,color=colors[group],lw=.8,alpha=.3)
            ax.plot(time,values.mean(axis=0),color=colors[group],lw=2,label=f'{group}: {len(q)} shared animals')
        ax.axvspan(0,10,color='.82',alpha=.35);ax.axhline(0,color='.65',lw=.75)
        ax.set(xlabel='Seconds after stimulus onset',ylabel=neuron+r' $\Delta F/F_0$',title=f'{"B" if neuron=="ADF" else "C"}  Same animals: {neuron}',xlim=(-5,39),ylim=(-.3,1.35))
        ax.legend(frameon=False,fontsize=7.5,loc='upper right')
    fig.suptitle('A positive ADF response can accompany opposite ASH responses',fontsize=14)
    fig.savefig(OUT/'figures/explore_population_combinations.png',dpi=190)
    fig.savefig(OUT/'figures/explore_population_combinations.pdf')
    plt.close(fig)


if __name__=='__main__':main()
