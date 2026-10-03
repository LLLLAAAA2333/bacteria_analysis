"""Bounded, second-stage Asparagine checks after the 162-feature ADF screen.

Asparagine was selected for its largest absolute full-adjusted ADF correlation,
not prespecified. These are potential refutations, with no p values. Chemical
values are strain-linked collaborator reports, not verified neural exposures.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
T=OUT/'tables'
spec=importlib.util.spec_from_file_location('chemical_tests',OUT/'code/05_chemical_tests.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
FEATURE='Asparagine'
REPEATS=['A011','A013','A014','A024','A025','A044']
ASNHIGH=['A009','A013','A014','A019','A020','A044']
SELECTED_SPECIES=['Bacteroides uniformis','Bacteroides fluxus']


def fit(d,x,mode):
    if mode!='species':
        return m.fit(d,x,mode)
    # Add species to the full adjustment; genus columns become redundant. Use
    # the same rank-truncated weighted SVD as 05 to handle exact dependencies.
    z=np.column_stack([m.design(d,'full'),pd.get_dummies(d.species_clean,dtype=float).to_numpy()])
    w=1/d.groupby(['sample_id','date']).sample_id.transform('size').to_numpy(float)
    sw=np.sqrt(w);zw=z*sw[:,None]
    y=d.stim.to_numpy(float);x=np.asarray(x,float)
    u,s,_=np.linalg.svd(zw,full_matrices=False)
    keep=s>np.finfo(float).eps*max(zw.shape)*s[0];basis=u[:,keep]
    xw=x*sw[:,None];yw=y*sw
    xrw=xw-basis@(basis.T@xw);yrw=yw-basis@(basis.T@yw)
    xr=xrw/sw[:,None];yr=yrw/sw
    assert np.linalg.norm(zw.T@xrw) <= 1e-12*max(1,np.linalg.norm(zw)*np.linalg.norm(xw))
    assert np.linalg.norm(zw.T@yrw) <= 1e-12*max(1,np.linalg.norm(zw)*np.linalg.norm(yw))
    xx=np.sum(w[:,None]*xr*xr,axis=0);xy=np.sum(w[:,None]*xr*yr[:,None],axis=0);yy=np.sum(w*yr*yr)
    return dict(beta=np.divide(xy,xx,out=np.zeros_like(xy),where=xx>1e-12),
                r=np.divide(xy,np.sqrt(xx*yy),out=np.zeros_like(xy),where=xx*yy>1e-12),
                xr=xr,yr=yr,w=w,rank=int(keep.sum()))


def main():
    a=pd.read_csv(T/'animal_metrics.csv',dtype={'date':str})
    tax=pd.read_csv(T/'taxonomy.csv').set_index('sample_id')
    tax['species_clean']=tax.species_clean.str.strip()
    ref=pd.read_csv(T/'chemical_reference_groups.csv').set_index('sample_id')
    raw=pd.read_csv(T/'chemical_raw.csv',index_col=0)
    log=pd.read_csv(T/'chemical_log.csv',index_col=0)
    legacy=pd.read_csv(T/'chemical_legacy_logfc.csv',index_col=0)
    meta=pd.read_csv(T/'chemical_feature_metadata.csv',index_col=0)
    names=meta.index[meta.complete_eligible].tolist()
    d=a[a.neuron_class.eq('ADF')].join(tax[['genus_clean','species_clean']],on='sample_id').join(ref[['reference_group']],on='sample_id')
    d['total_log']=d.sample_id.map(log[names].median(axis=1))
    d=d.reset_index(drop=True)
    ph=pd.read_csv(T/'phenotype_animal_sensitivity.csv',dtype={'date':str})
    variants=['median','first','later','qc_exclude_below_minus1','order_adjusted']
    d=d.merge(ph[['sample_id','date','worm_key','neuron_class',*variants]],on=['sample_id','date','worm_key','neuron_class'],validate='one_to_one')
    assert d.stim.notna().all() and log[FEATURE].notna().all()
    records=[]
    points=[]

    def evaluate(label,dd,profile=log,variant='stim',modes=('animal','full','species')):
        dd=dd.copy().reset_index(drop=True)
        if variant!='stim':
            dd['stim']=dd[variant]
        dd=dd.dropna(subset=['stim']).reset_index(drop=True)
        x=profile.loc[dd.sample_id,[FEATURE]].to_numpy()
        base=fit(dd,x,'animal')
        basesd=float(np.sqrt(np.sum(base['w']*base['xr'][:,0]**2)/sum(base['w'])))
        strains=dd[['sample_id','species_clean']].drop_duplicates()
        for mode in modes:
            f=base if mode=='animal' else fit(dd,x,mode)
            rsd=float(np.sqrt(np.sum(f['w']*f['xr'][:,0]**2)/sum(f['w'])))
            rec=dict(scenario=label,adjustment=mode,response_variant=variant,partial_r=float(f['r'][0]),slope=float(f['beta'][0]),
                n_strains=dd.sample_id.nunique(),n_dates=dd.date.nunique(),n_animals=len(dd[['date','worm_key']].drop_duplicates()),n_rows=len(dd),
                n_species=dd.species_clean.nunique(),n_species_with_multiple_strains=int((strains.groupby('species_clean').size()>1).sum()),
                report_min=float(raw.loc[dd.sample_id,FEATURE].min()),report_max=float(raw.loc[dd.sample_id,FEATURE].max()),
                residual_x_sd=rsd,residual_x_sd_fraction_of_animal=rsd/basesd if basesd>0 else np.nan,nuisance_rank=f['rank'])
            records.append(rec)
            if label in ['all','Bacteroides','non_Bacteroides'] and mode=='animal':
                p=dd[['sample_id','date','worm_key','species_clean','genus_clean','stim']].copy()
                p['scenario']=label;p['x_report']=raw.loc[dd.sample_id,FEATURE].to_numpy();p['x_log']=log.loc[dd.sample_id,FEATURE].to_numpy()
                p['x_residual']=f['xr'][:,0];p['y_residual']=f['yr'];p['weight']=f['w'];points.append(p)

    scenarios=[('all',d),('without_A024_A025',d[~d.sample_id.isin(['A024','A025'])]),
               ('without_crossdate6',d[~d.sample_id.isin(REPEATS)]),
               ('without_highAsn6',d[~d.sample_id.isin(ASNHIGH)]),
               ('without_uniformis_fluxus',d[~d.species_clean.isin(SELECTED_SPECIES)])]
    bact=d[d.genus_clean.eq('Bacteroides')]
    other=d[d.genus_clean.ne('Bacteroides')]
    scenarios += [('Bacteroides',bact),('non_Bacteroides',other),
                  ('Bact_without_A024_A025',bact[~bact.sample_id.isin(['A024','A025'])]),
                  ('Bact_without_crossdate6',bact[~bact.sample_id.isin(REPEATS)]),
                  ('Bact_without_highAsn6',bact[~bact.sample_id.isin(ASNHIGH)]),
                  ('Bact_without_uniformis_fluxus',bact[~bact.species_clean.isin(SELECTED_SPECIES)])]
    for label,dd in scenarios:
        evaluate(label,dd)
    for date in sorted(d.date.unique()):
        evaluate('omit_date_'+date,d[d.date.ne(date)],modes=('animal','full'))
        bb=bact[bact.date.ne(date)]
        if date in set(bact.date):evaluate('Bact_omit_date_'+date,bb,modes=('animal','full'))
    for sample in sorted(d.sample_id.unique()):
        evaluate('omit_strain_'+sample,d[d.sample_id.ne(sample)],modes=('animal','full'))
    for sp in sorted(d.species_clean.unique()):
        evaluate('omit_species_'+sp,d[d.species_clean.ne(sp)],modes=('animal','full'))
    for sp in sorted(bact.species_clean.unique()):
        evaluate('Bact_omit_species_'+sp,bact[bact.species_clean.ne(sp)],modes=('animal','full'))
    for label,dd in [('all',d),('Bacteroides',bact),('non_Bacteroides',other)]:
        for variant in variants:
            evaluate(label+'_'+variant,dd,variant=variant,modes=('animal','full'))
        evaluate(label+'_legacy_FC',dd,profile=legacy,modes=('animal','full'))
        evaluate(label+'_report_linear',dd,profile=raw,modes=('animal','full'))
    sensitivity=pd.DataFrame(records)
    for group in ['all','Bacteroides','non_Bacteroides']:
        base_r=sensitivity.loc[sensitivity.scenario.eq(group)&sensitivity.adjustment.eq('full'),'partial_r'].iloc[0]
        legacy_r=sensitivity.loc[sensitivity.scenario.eq(group+'_legacy_FC')&sensitivity.adjustment.eq('full'),'partial_r'].iloc[0]
        # log legacy FC differs from log(report+1) only by known reference-group
        # offsets, so projecting those offsets out should give the same result.
        assert np.isclose(base_r,legacy_r,atol=1e-10,rtol=1e-10)
    sensitivity.to_csv(T/'asn_sensitivity.csv',index=False)
    pp=pd.concat(points,ignore_index=True)
    pp.to_csv(T/'asn_plot_points.csv',index=False)
    strain=d[['sample_id','genus_clean','species_clean']].drop_duplicates().set_index('sample_id')
    strain['asparagine_report_ng_mL']=raw.loc[strain.index,FEATURE]
    strain['asparagine_log2_report_plus1']=log.loc[strain.index,FEATURE]
    strain['posthoc_highAsn6']=strain.index.isin(ASNHIGH)
    strain['n_dates']=d.groupby('sample_id').date.nunique()
    strain['adf_mean_equal_dates']=d.groupby(['sample_id','date']).stim.mean().groupby('sample_id').mean()
    strain.to_csv(T/'asn_strain_data.csv')
    covary=[]
    for label,ids in [('all',d.sample_id.unique()),('Bacteroides',bact.sample_id.unique()),('non_Bacteroides',other.sample_id.unique())]:
        rr=log.loc[ids,names].corr(method='spearman')[FEATURE]
        for feature,rho in rr.items():
            covary.append(dict(group=label,feature=feature,spearman_rho=rho,n_strains=len(ids)))
    covary=pd.DataFrame(covary)
    covary.to_csv(T/'asn_covary.csv',index=False)

    # Sensitivity to the second-stage selection rule: on each training fold,
    # choose the largest |full-adjusted r| among the same 162 complete-QC features,
    # then fit only its animal-adjusted training slope. All six repeated strains
    # are omitted globally to avoid any strain crossing train/test dates.
    cvd=d[~d.sample_id.isin(REPEATS)].reset_index(drop=True)
    folds=[];predictions=[]
    for date in sorted(cvd.date.unique()):
        train=cvd[cvd.date.ne(date)].reset_index(drop=True);test=cvd[cvd.date.eq(date)].reset_index(drop=True)
        assert not set(train.sample_id)&set(test.sample_id)
        assert train.sample_id.nunique()+test.sample_id.nunique()==100
        full=fit(train,log.loc[train.sample_id,names].to_numpy(),'full')
        j=int(np.argmax(np.abs(full['r'])));feature=names[j]
        animal=fit(train,log.loc[train.sample_id,[feature]].to_numpy(),'animal')
        xp=m.center_animal(test,log.loc[test.sample_id,[feature]].to_numpy())[:,0]
        yp=m.center_animal(test,test[['stim']].to_numpy())[:,0]
        prediction=xp*animal['beta'][0]
        pred=test[['sample_id','date','worm_key']].copy()
        pred['observed_relative']=yp;pred['predicted_relative']=prediction
        pred=pred.groupby(['sample_id','date'],as_index=False)[['observed_relative','predicted_relative']].mean()
        pred['selected_feature']=feature;predictions.append(pred)
        sse=float(np.sum((pred.observed_relative-pred.predicted_relative)**2));null=float(np.sum(pred.observed_relative**2))
        folds.append(dict(date=date,selected_feature=feature,training_full_r=full['r'][j],training_animal_r=animal['r'][0],
                          training_animal_slope=animal['beta'][0],n_train_strains=train.sample_id.nunique(),n_test_strains=test.sample_id.nunique(),
                          test_sse=sse,null_sse=null,test_relative_R2=1-sse/null))
    folds=pd.DataFrame(folds);predictions=pd.concat(predictions,ignore_index=True)
    assert len(predictions)==predictions.sample_id.nunique()==100
    assert len(folds)==9 and not set(predictions.sample_id)&set(REPEATS)
    folds.to_csv(T/'asn_selection_rule_cv_folds.csv',index=False)
    predictions.to_csv(T/'asn_selection_rule_cv_predictions.csv',index=False)
    cvsummary=dict(n_strains=len(predictions),n_dates=len(folds),relative_R2=float(1-folds.test_sse.sum()/folds.null_sse.sum()),
                   days_improved=int(folds.test_relative_R2.gt(0).sum()),selected_features=folds.selected_feature.value_counts().to_dict(),
                   rmse=float(np.sqrt(np.mean((predictions.observed_relative-predictions.predicted_relative)**2))),
                   null_rmse=float(np.sqrt(np.mean(predictions.observed_relative**2))))
    summary=dict(feature=FEATURE,selection='Second-stage conditional selection: maximum absolute full-adjusted ADF correlation among 162 complete-QC features after broad animal-only screen; post hoc.',
                 no_inferential_tests=True,metadata=meta.loc[FEATURE].to_dict(),
                 posthoc_highAsn6=ASNHIGH,posthoc_highAsn6_rationale='Six high-report Bacteroides strains separated by observed chemical-value gap; feature-distribution subgroup, not prespecified.',
                 cv=cvsummary,
                 caveats=['Shared-animal fixed intercepts and equal strain-date weights preserve repeated-stimulus dependence for descriptive projection, not inferential randomization.',
                          'No tested feature is a verified actual exposure concentration; QC repeatability is not structural identification confidence.',
                          'Species-adjusted residual chemical variation is reported; species strata often have only one or two strains.',
                          'Leave-one-date-out reuses the existing discovery data; relative responses require test-animal centering and are not independent validation.',
                          'Removing selected species or high-Asn strains narrows chemical support; attenuation is not proof of a molecule mechanism or its absence.'],
                 input_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T/'animal_metrics.csv',T/'phenotype_animal_sensitivity.csv',T/'chemical_raw.csv',T/'chemical_log.csv',T/'chemical_legacy_logfc.csv',T/'chemical_feature_metadata.csv',T/'taxonomy.csv',OUT/'code/05_chemical_tests.py']})
    (OUT/'logs/06_asparagine_summary.json').write_text(json.dumps(summary,indent=2))
    plot(pp,sensitivity)
    print(sensitivity[sensitivity.scenario.isin([x[0] for x in scenarios])][['scenario','adjustment','partial_r','n_strains','n_species','residual_x_sd_fraction_of_animal','report_min','report_max']].round(4).to_string(index=False))
    print('\nAsparagine / Aspartic acid co-variation:')
    print(covary[covary.feature.eq('Aspartic acid')].to_string(index=False))
    print('\nLeave-one-date-out second-stage selection sensitivity:')
    print(folds.round(4).to_string(index=False));print(json.dumps(cvsummary,indent=2))
    print('\nAll requested Asparagine checks ran successfully.')


def plot(points,sens):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(12.8,10.5),layout='constrained')
    dates=sorted(points.date.unique());colors=dict(zip(dates,plt.get_cmap('tab10').colors))
    b=points[points.scenario.eq('Bacteroides')]
    ax=axs[0,0]
    for dt,g in b.groupby('date'):
        ax.scatter(g.x_log,g.stim,s=15,c=[colors[dt]],alpha=.38,label=dt)
        means=g.groupby('sample_id')[['x_log','stim']].mean()
        ax.scatter(means.x_log,means.stim,s=38,facecolors='none',edgecolors=[colors[dt]],lw=1.1)
    labels={'A013':(6,18),'A014':(6,-20),'A024':(-45,20),'A025':(8,-23)}
    for sid,offset in labels.items():
        q=b[b.sample_id.eq(sid)][['x_log','stim']].mean()
        ax.annotate(sid,(q.x_log,q.stim),xytext=offset,textcoords='offset points',fontsize=9,
                    arrowprops={'arrowstyle':'-','lw':.7,'color':'.25'})
    ax.set(xlabel='Asparagine: log2(report ng/mL + 1)',ylabel=r'ADF stimulus mean [0,10) s, $\Delta F/F_0$',
           title='A  Bacteroides: 29 strains; dots are animals')
    ax.text(.02,.96,'Open circles: strain/date means\nColours: imaging dates',transform=ax.transAxes,va='top',fontsize=9)
    for ax,group,title in [(axs[0,1],'Bacteroides','B  Within the same animals: Bacteroides'),(axs[1,0],'non_Bacteroides','C  Outside Bacteroides: 77 strains')]:
        g=points[points.scenario.eq(group)]
        for dt,dd in g.groupby('date'):
            ax.scatter(dd.x_residual,dd.y_residual,s=15,c=[colors[dt]],alpha=.35)
        row=sens[sens.scenario.eq(group)&sens.adjustment.eq('animal')].iloc[0]
        xx=np.array([g.x_residual.min(),g.x_residual.max()]);ax.plot(xx,xx*row.slope,c='.2',lw=1.3)
        ax.axhline(0,color='.6',lw=.7);ax.axvline(0,color='.6',lw=.7)
        ax.set(xlabel='Animal-centered log2(report + 1)',ylabel=r'Animal-centered ADF, $\Delta F/F_0$',title=title)
        ax.set_xlim(-6.8,6.4);ax.set_ylim(-.72,1.4)
        ax.text(.03,.96,f'Descriptive partial r = {row.partial_r:.3f}',transform=ax.transAxes,va='top')
    ax=axs[1,1]
    scenarios=['Bacteroides','Bact_without_A024_A025','Bact_without_crossdate6','Bact_without_uniformis_fluxus','Bact_without_highAsn6']
    labels=['All Bacteroides (29)','Drop A024/A025 (27)','Drop six repeat strains (23)','Drop uniformis + fluxus (25)','Drop six high-report strains (23)']
    for mode,color,offset,label in [('animal','#2166ac',-.15,'Animal'),('full','#b2182b',0,'+ reference / total / genus'),('species','#56644c',.15,'+ species')]:
        q=sens[sens.adjustment.eq(mode)].set_index('scenario').loc[scenarios]
        ax.scatter(q.partial_r,np.arange(len(q))+offset,s=32,color=color,label=label)
    ax.axvline(0,color='.6',lw=.7);ax.set(yticks=np.arange(5),yticklabels=labels,xlabel='Descriptive partial correlation (no uncertainty bars)',title='D  Can selected strains or species explain it?',xlim=(-.8,.25))
    ax.invert_yaxis();ax.legend(loc='lower left',fontsize=8,frameon=False)
    # All nine dates are represented in B/C, including dates absent from panel A.
    handles=[plt.Line2D([],[],marker='o',ls='',color=colors[dt],alpha=.75,ms=5) for dt in dates]
    fig.legend(handles,[x[4:] for x in dates],loc='outside lower center',ncol=len(handles),frameon=False,title='Imaging date in 2026 (MMDD)')
    fig.suptitle('ADF–Asparagine: a post hoc chemical association with limited scope',fontsize=14)
    fig.savefig(OUT/'figures/05_asparagine_refutation.png',dpi=180)
    fig.savefig(OUT/'figures/05_asparagine_refutation.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
