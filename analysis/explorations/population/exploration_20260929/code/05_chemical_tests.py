"""Limited chemical explanations of two neurally selected scalar phenotypes.

No p values: strains were not randomized across date/order/chemical backgrounds.
Shared animals are retained as intercept blocks, not independent strain replicates.
Within-date leave-out prediction evaluates relative strain responses, not absolute
responses on a new day. Feature selection is rerun inside every training fold.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
T = OUT/'tables'
TARGETS = ['ADF','AWCON']
TOP_AWCON = ['A247','A296','A290','A288','A289','A302','A300']


def design(d,mode):
    z = pd.get_dummies(d.date+'_'+d.worm_key,dtype=float)
    if mode in ['reference','reference_total','full']:
        z = pd.concat([z,pd.get_dummies(d.reference_group,dtype=float)],axis=1)
    if mode in ['reference_total','full']:
        z['total'] = d.total_log
    if mode == 'full':
        z = pd.concat([z,pd.get_dummies(d.genus_clean,dtype=float)],axis=1)
    return z.to_numpy(float)


def fit(d,x,mode='animal'):
    """Equal strain-date weight; residual projection has animal intercepts.

    Every column of X shares rows here because primary features are complete.
    The correlation is a descriptive weighted partial Pearson correlation.
    """
    d = d.reset_index(drop=True)
    x = np.asarray(x,float)
    y = d.stim.to_numpy(float)
    w = 1/d.groupby(['sample_id','date']).sample_id.transform('size').to_numpy(float)
    z = design(d,mode)
    sw = np.sqrt(w)
    zw = z*sw[:,None]
    # Animal, genus and reference indicators contain exact dependencies. An
    # explicit rank cutoff avoids unstable coefficients on near-null columns.
    u,s,_ = np.linalg.svd(zw,full_matrices=False)
    keep = s > np.finfo(float).eps*max(zw.shape)*s[0]
    basis = u[:,keep]
    xw,yw = x*sw[:,None],y*sw
    xrw = xw-basis@(basis.T@xw)
    yrw = yw-basis@(basis.T@yw)
    xr,yr = xrw/sw[:,None],yrw/sw
    xx = (w[:,None]*xr*xr).sum(axis=0)
    xy = (w[:,None]*xr*yr[:,None]).sum(axis=0)
    yy = np.sum(w*yr*yr)
    beta = np.divide(xy,xx,out=np.zeros_like(xy),where=xx>1e-12)
    r = np.divide(xy,np.sqrt(xx*yy),out=np.zeros_like(xy),where=xx*yy>1e-12)
    return dict(beta=beta,r=r,xr=xr,yr=yr,w=w,rank=int(keep.sum()))


def center_animal(d, values):
    f = pd.DataFrame(values).set_axis(d.index)
    return (f-f.groupby([d.date,d.worm_key]).transform('mean')).to_numpy()


def main():
    a = pd.read_csv(T/'animal_metrics.csv',dtype={'date':str})
    tax = pd.read_csv(T/'taxonomy.csv').set_index('sample_id')
    ref = pd.read_csv(T/'chemical_reference_groups.csv').set_index('sample_id')
    log = pd.read_csv(T/'chemical_log.csv',index_col=0)
    legacy = pd.read_csv(T/'chemical_legacy_logfc.csv',index_col=0)
    raw = pd.read_csv(T/'chemical_raw.csv',index_col=0)
    meta = pd.read_csv(T/'chemical_feature_metadata.csv',index_col=0)
    names = meta.index[meta.complete_eligible].tolist()
    broad = meta.index[meta.primary_eligible].tolist()
    x = log[names]
    assert x.notna().all().all()
    a = a.join(tax[['genus_clean','species_clean']],on='sample_id').join(ref[['reference_group']],on='sample_id')
    # Median across the fixed complete-QC panel, avoiding missingness-dependent totals.
    a['total_log'] = a.sample_id.map(x.median(axis=1))
    repeats = a.groupby('sample_id').date.nunique()
    single_ids = repeats[repeats.eq(1)].index
    screen, sensitivity, points, covary = [],[],[],[]
    selected = {}
    for target in TARGETS:
        d = a[a.neuron_class.eq(target)].reset_index(drop=True)
        X = x.loc[d.sample_id].to_numpy()
        primary = fit(d,X)
        best = int(np.argmax(np.abs(primary['r'])))
        feature = names[best]
        selected[target] = feature
        iqr = float(x[feature].quantile(.75)-x[feature].quantile(.25))
        for mode in ['animal','reference','reference_total','full']:
            f = fit(d,X,mode)
            for j,name in enumerate(names):
                screen.append(dict(target=target,feature=name,adjustment=mode,r=f['r'][j],
                    slope=f['beta'][j],effect_per_IQR=f['beta'][j]*(x[name].quantile(.75)-x[name].quantile(.25)),
                    n_strains=d.sample_id.nunique(),n_dates=d.date.nunique(),
                    n_animals=d[['date','worm_key']].drop_duplicates().shape[0],n_animal_strain_rows=len(d),
                    nuisance_rank=f['rank'],
                    residual_x_sd=float(np.sqrt(np.sum(f['w']*f['xr'][:,j]**2)/sum(f['w']))),
                    residual_x_sd_fraction=float(np.sqrt(np.sum(f['w']*f['xr'][:,j]**2)/np.sum(primary['w']*primary['xr'][:,j]**2)))))
        scenarios = [('all',d,x),('legacy_FC',d,legacy[names]),
                     ('without_top7_AWCON',d[~d.sample_id.isin(TOP_AWCON)],x),
                     ('without_Bacteroides',d[d.genus_clean.ne('Bacteroides')],x)]
        # Quantile transform checks leverage of chemical magnitude outliers.
        scenarios.append(('chemical_ranks',d,x.rank()))
        for date in sorted(d.date.unique()):
            scenarios.append(('omit_date_'+date,d[d.date.ne(date)],x))
        for sample in sorted(d.sample_id.unique()):
            scenarios.append(('omit_strain_'+sample,d[d.sample_id.ne(sample)],x))
        for label,dd,profile in scenarios:
            for mode in ['animal','full']:
                f = fit(dd,profile.loc[dd.sample_id,[feature]].to_numpy(),mode)
                sensitivity.append(dict(target=target,feature=feature,scenario=label,adjustment=mode,
                    r=f['r'][0],slope=f['beta'][0],effect_per_original_IQR=f['beta'][0]*iqr if label!='chemical_ranks' else np.nan,
                    n_strains=dd.sample_id.nunique(),n_dates=dd.date.nunique()))
        for mode in ['animal','full']:
            f=fit(d,X[:,[best]],mode)
            pp=d[['sample_id','date','worm_key','genus_clean','reference_group','stim']].copy()
            pp['x_report_log']=X[:,best]
            pp['x_residual']=f['xr'][:,0]
            pp['y_residual']=f['yr']
            pp['target']=target;pp['feature']=feature;pp['adjustment']=mode
            points.append(pp)
        c=x.corr(method='spearman')[feature].drop(feature)
        for name,value in c[abs(c).ge(.65)].sort_values(key=abs,ascending=False).items():
            covary.append(dict(target=target,selected_feature=feature,cofeature=name,spearman=value))
    screening=pd.DataFrame(screen)
    screening.to_csv(T/'chem_screen.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(T/'chem_sensitivity.csv',index=False)
    pd.concat(points).to_csv(T/'chem_plot_points.csv',index=False)
    pd.DataFrame(covary).to_csv(T/'chem_covary.csv',index=False)

    # Nested feature selection; no test-day outcome participates in selection/fitting.
    # All six repeated strains are omitted, so no strain can appear on both sides.
    folds,predictions=[],[]
    for target in TARGETS:
        d=a[a.neuron_class.eq(target)&a.sample_id.isin(single_ids)].reset_index(drop=True)
        for panel in ['complete','broad']:
            nn=names if panel=='complete' else broad
            for date in sorted(d.date.unique()):
                train=d[d.date.ne(date)].reset_index(drop=True)
                test=d[d.date.eq(date)].reset_index(drop=True)
                # Imputation uses distinct training strains only; no outcome or test-day fit.
                med=log.loc[train.sample_id.unique(),nn].median()
                profile=log[nn].fillna(med)
                f=fit(train,profile.loc[train.sample_id].to_numpy())
                j=int(np.argmax(np.abs(f['r'])))
                feature=nn[j]
                xp=center_animal(test,profile.loc[test.sample_id,[feature]].to_numpy())[:,0]
                yp=center_animal(test,test[['stim']].to_numpy())[:,0]
                pred=xp*f['beta'][j]
                pp=test[['sample_id','date','worm_key']].copy()
                pp['observed_relative']=yp;pp['predicted_relative']=pred
                pp=pp.groupby(['sample_id','date'],as_index=False)[['observed_relative','predicted_relative']].mean()
                pp['target']=target;pp['panel']=panel;pp['selected_feature']=feature
                predictions.append(pp)
                sse=np.sum((pp.observed_relative-pp.predicted_relative)**2)
                null=np.sum(pp.observed_relative**2)
                folds.append(dict(target=target,panel=panel,date=date,selected_feature=feature,
                    slope=f['beta'][j],training_r=f['r'][j],n_train_strains=train.sample_id.nunique(),
                    n_test_strains=test.sample_id.nunique(),test_sse=sse,null_sse=null,
                    test_relative_R2=1-sse/null if null>0 else np.nan,
                    train_x_min=float(profile.loc[train.sample_id,feature].min()),
                    train_x_max=float(profile.loc[train.sample_id,feature].max()),
                    test_x_min=float(profile.loc[test.sample_id,feature].min()),
                    test_x_max=float(profile.loc[test.sample_id,feature].max())))
    folds=pd.DataFrame(folds);predictions=pd.concat(predictions,ignore_index=True)
    folds.to_csv(T/'chem_cv_folds.csv',index=False)
    predictions.to_csv(T/'chem_cv_predictions.csv',index=False)
    cv=[]
    for (target,panel),dd in predictions.groupby(['target','panel']):
        sub=folds[folds.target.eq(target)&folds.panel.eq(panel)]
        cv.append(dict(target=target,panel=panel,n_strains=len(dd),n_dates=dd.date.nunique(),
            relative_R2=1-sub.test_sse.sum()/sub.null_sse.sum(),
            RMSE=float(np.sqrt(np.mean((dd.observed_relative-dd.predicted_relative)**2))),
            null_RMSE=float(np.sqrt(np.mean(dd.observed_relative**2))),
            n_days_improved=int((sub.test_relative_R2>0).sum()),
            selected_features=sub.selected_feature.value_counts().to_dict()))
    (OUT/'logs/chemical_results.json').write_text(json.dumps(dict(selected=selected,cv=cv,
        primary_n_features=len(names),broad_n_features=len(broad),
        scope='Exploratory associations; no confirmatory p values, no independent validation.'),indent=2))
    plot(pd.concat(points),screening,folds,predictions,selected)
    print(json.dumps(dict(selected=selected,cv=cv),indent=2))
    print(screening[screening.apply(lambda r:r.feature==selected[r.target],axis=1)].round(4).to_string(index=False))


def plot(points,screen,folds,predictions,selected):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dates=sorted(points.date.unique()); colors=dict(zip(dates,plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(2,3,figsize=(12.7,7.6),layout='constrained')
    for row,target in enumerate(TARGETS):
        pp=points[points.target.eq(target)&points.adjustment.eq('animal')]
        pp=pp.groupby(['sample_id','date'],as_index=False)[['x_residual','y_residual']].mean()
        ax=axes[row,0]
        for date,g in pp.groupby('date'):
            ax.scatter(g.x_residual,g.y_residual,s=22,color=colors[date],alpha=.8,label=date[4:])
        if target=='AWCON':
            for sample in ['A247','A249','A296']:
                q=pp[pp.sample_id.eq(sample)].iloc[0]
                ax.annotate(sample,(q.x_residual,q.y_residual),xytext=(4,5),textcoords='offset points',fontsize=8)
        f=screen[screen.target.eq(target)&screen.feature.eq(selected[target])&screen.adjustment.eq('animal')].iloc[0]
        xx=np.array([pp.x_residual.min(),pp.x_residual.max()]);ax.plot(xx,xx*f.slope,c='black',lw=1)
        ax.axhline(0,color='.7',lw=.7);ax.axvline(0,color='.7',lw=.7)
        ax.set(xlabel=f'{selected[target]}\nanimal-centered log2(report + 1)',
               ylabel=f'{target}: animal-centered stimulus mean\n'+r'$\Delta F/F_0$',title=f'{chr(65+row*3)}  Observed strain/date differences')
        ax=axes[row,1]
        modes=['animal','reference','reference_total','full']
        labels=['Animal','+ reference group','+ chemical overall level','+ genus']
        g=screen[screen.target.eq(target)&screen.feature.eq(selected[target])].set_index('adjustment').loc[modes]
        ax.barh(labels,g.r,color=['#487aa1','#7794a9','#a5b2bd','#626669'])
        ax.invert_yaxis();ax.axvline(0,c='.4',lw=.8);ax.set(xlim=(-.65,.65),xlabel='Descriptive partial correlation',title=f'{chr(66+row*3)}  Competing explanations')
        ax=axes[row,2]
        g=predictions[predictions.target.eq(target)&predictions.panel.eq('complete')]
        for date,dd in g.groupby('date'):
            ax.scatter(dd.observed_relative,dd.predicted_relative,s=22,color=colors[date],alpha=.8)
        lim=[min(g.observed_relative.min(),g.predicted_relative.min()),max(g.observed_relative.max(),g.predicted_relative.max())]
        ax.plot(lim,lim,c='.5',ls='--',lw=1);ax.axhline(0,c='.7',lw=.7)
        ax.set(xlabel=r'Observed relative $\Delta F/F_0$',ylabel=r'Predicted relative $\Delta F/F_0$',title=f'{chr(67+row*3)}  Leave-one-date-out selection')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=9,title='Imaging date in 2026 (MMDD)',frameon=False)
    fig.savefig(OUT/'figures/04_chemical_explanations.png',dpi=180)
    fig.savefig(OUT/'figures/04_chemical_explanations.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
