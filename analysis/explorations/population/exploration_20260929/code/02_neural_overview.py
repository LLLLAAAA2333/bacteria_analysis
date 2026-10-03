"""Broad-window neural overview with animal-preserving split-half checks.

Independent biological identity is (date, worm_key). All strains from an animal
stay in the same half / bootstrap draw. No p values, no independent validation.
The full 40-s window is a sensitivity summary, not 40 independent observations.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

OUT = Path(__file__).resolve().parents[1]
TABLES = OUT/'tables'
SEED = 20260929
N_SPLITS = 500
N_BOOT = 1000
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
METRICS = ['stim','post','full']
UNITS = r'$\Delta F/F_0$'


def pearson(x,y):
    if len(x)<5 or np.std(x)==0 or np.std(y)==0:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])


def main():
    rng = np.random.default_rng(SEED)
    a = pd.read_csv(TABLES/'animal_metrics.csv')
    d = pd.read_csv(TABLES/'date_metrics.csv')
    curves = pd.read_parquet(TABLES/'animal_curves.parquet')
    curve_metrics=curves.index.to_frame(index=False)
    curve_metrics['date']=curve_metrics.date.astype(int)
    for w,(lo,hi) in {'stim':(0,10),'post':(10,30),'full':(0,40)}.items():
        curve_metrics[w]=curves[[str(t) for t in range(lo,hi)]].mean(axis=1).to_numpy()
    aligned=a.merge(curve_metrics,on=['sample_id','date','worm_key','neuron_class'],suffixes=('_table','_curve'),validate='one_to_one')
    assert len(aligned)==len(a)==len(curves)
    for w in METRICS:
        assert np.allclose(aligned[f'{w}_table'],aligned[f'{w}_curve'],rtol=1e-12,atol=1e-12)
    animal_ids = a[['date','worm_key']].drop_duplicates().sort_values(['date','worm_key'])
    animal_keys = list(animal_ids.itertuples(index=False,name=None))
    animal_lookup = {x:i for i,x in enumerate(animal_keys)}
    sd = d[['sample_id','date']].drop_duplicates().sort_values(['date','sample_id']).reset_index(drop=True)
    sd_keys = list(sd.itertuples(index=False,name=None))
    sd_lookup = {x:i for i,x in enumerate(sd_keys)}
    dates = np.sort(a.date.unique())
    date_animals = {dt:np.flatnonzero(animal_ids.date.to_numpy()==dt) for dt in dates}
    date_rows = {dt:np.flatnonzero(sd.date.to_numpy()==dt) for dt in dates}
    values = {w:np.full((len(sd),len(NEURONS),len(animal_keys)),np.nan) for w in METRICS}
    for row in a.itertuples():
        i = sd_lookup[(row.sample_id,row.date)]
        j = NEURONS.index(row.neuron_class)
        k = animal_lookup[(row.date,row.worm_key)]
        for w in METRICS:
            values[w][i,j,k]=getattr(row,w)
    means = {w:np.nanmean(v,axis=2) for w,v in values.items()}
    centered = {}
    top3 = {}
    for w in METRICS:
        centered[w] = means[w].copy()
        for dt,ix in date_rows.items():
            centered[w][ix] -= np.nanmean(means[w][ix],axis=0)
        for j,n in enumerate(NEURONS):
            salience = pd.Series(np.abs(centered[w][:,j]),index=sd.sample_id).groupby(level=0).max()
            top3[(w,n)] = set(salience.nlargest(3).index)

    # Every split is assigned once per true animal, jointly across all neurons
    # and strains. Conditional correlations do not remove strain-date confounding.
    split_records=[]
    for b in range(N_SPLITS):
        half=np.zeros(len(animal_keys),bool)
        for ix in date_animals.values():
            selected=rng.permutation(ix)[:len(ix)//2]
            half[selected]=True
        for w,V in values.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                x=np.nanmean(V[:,:,half],axis=2)
                y=np.nanmean(V[:,:,~half],axis=2)
            nx=np.isfinite(V[:,:,half]).sum(axis=2)
            ny=np.isfinite(V[:,:,~half]).sum(axis=2)
            for j,n in enumerate(NEURONS):
                valid=np.isfinite(x[:,j]) & np.isfinite(y[:,j])
                for mode,mask in [('all',valid),('drop_top3',valid & ~sd.sample_id.isin(top3[(w,n)]).to_numpy()),('at_least_2_per_half',valid & (nx[:,j]>=2)&(ny[:,j]>=2))]:
                    # Re-center after each exclusion. Otherwise excluded strong
                    # strains could leave correlated date offsets in the remainder.
                    xc=x[:,j].copy();yc=y[:,j].copy()
                    for ix in date_rows.values():
                        vix=ix[mask[ix]]
                        if len(vix)<2:
                            mask[vix]=False
                        else:
                            xc[vix]-=xc[vix].mean();yc[vix]-=yc[vix].mean()
                    xx,yy=xc[mask],yc[mask]
                    split_records.append(dict(split=b,metric=w,neuron_class=n,mode=mode,n_strain_dates=int(mask.sum()),
                        raw_pearson=pearson(x[mask,j],y[mask,j]),centered_pearson=pearson(xx,yy),
                        centered_spearman=pearson(rankdata(xx),rankdata(yy))))
    splits=pd.DataFrame(split_records)
    splits.to_csv(TABLES/'overview_split_half_draws.csv',index=False)
    summary=splits.groupby(['metric','neuron_class','mode']).agg(
        n_min=('n_strain_dates','min'),n_median=('n_strain_dates','median'),
        raw_r_median=('raw_pearson','median'),r_median=('centered_pearson','median'),
        r_p05=('centered_pearson',lambda x:x.quantile(.05)),r_p95=('centered_pearson',lambda x:x.quantile(.95)),
        spearman_median=('centered_spearman','median')).reset_index()
    summary['omitted_top3']=[';'.join(sorted(top3[(r.metric,r.neuron_class)])) if r.mode=='drop_top3' else '' for r in summary.itertuples()]
    summary.to_csv(TABLES/'overview_split_half_summary.csv',index=False)

    diag=[]
    for w in METRICS:
        for j,n in enumerate(NEURONS):
            idx=(a.neuron_class==n)
            obs=a[idx]
            residuals=obs[w]-obs.groupby(['sample_id','date'])[w].transform('mean')
            total=np.sum((means[w][:,j]-np.mean(means[w][:,j]))**2)
            within=np.sum(centered[w][:,j]**2)
            row=dict(metric=w,neuron_class=n,n_strains=obs.sample_id.nunique(),n_strain_dates=len(means[w]),
                n_animal_curves=len(obs),n_true_animals=len(obs[['date','worm_key']].drop_duplicates()),
                response_p10=np.quantile(means[w][:,j],.1),response_median=np.median(means[w][:,j]),response_p90=np.quantile(means[w][:,j],.9),
                response_min=np.min(means[w][:,j]),response_max=np.max(means[w][:,j]),
                within_strain_date_animal_rms=float(np.sqrt(np.mean(residuals**2))),
                date_grouping_fraction=1-within/total)
            for mode,prefix in [('all','split'),('drop_top3','drop3'),('at_least_2_per_half','min2')]:
                s=summary[(summary.metric==w)&(summary.neuron_class==n)&(summary['mode']==mode)].iloc[0]
                for c in ['r_median','r_p05','r_p95','spearman_median','n_median']:
                    row[f'{prefix}_{c}']=s[c]
            row['top3_strains']=';'.join(sorted(top3[(w,n)]))
            diag.append(row)
    diagnostic=pd.DataFrame(diag)
    diagnostic.to_csv(TABLES/'overview_diagnostic.csv',index=False)

    # Resampling true animals within each date; a single weight vector is shared
    # across every strain-neuron cell, respecting the repeated-stimulus design.
    boots={w:np.full((N_BOOT,len(sd),len(NEURONS)),np.nan) for w in METRICS}
    for b in range(N_BOOT):
        weights=np.zeros(len(animal_keys),int)
        for ix in date_animals.values():
            draw=rng.choice(ix,size=len(ix),replace=True)
            np.add.at(weights,draw,1)
        for w,V in values.items():
            numerator=np.einsum('ijk,k->ij',np.nan_to_num(V),weights)
            denominator=np.einsum('ijk,k->ij',np.isfinite(V).astype(float),weights)
            with np.errstate(invalid='ignore',divide='ignore'):
                boots[w][b]=numerator/denominator
    repeats=sd.groupby('sample_id').date.nunique().loc[lambda s:s>1].index
    repeat_rows=[]
    for strain in repeats:
        rows=sd.index[sd.sample_id.eq(strain)].tolist()
        assert len(rows)==2
        i,k=rows
        for j,n in enumerate(NEURONS):
            for w in METRICS:
                diff=boots[w][:,k,j]-boots[w][:,i,j]
                valid=np.isfinite(diff)
                lo,hi=np.quantile(diff[valid],[.025,.975])
                repeat_rows.append(dict(sample_id=strain,neuron_class=n,metric=w,
                    first_date=int(sd.loc[i,'date']),second_date=int(sd.loc[k,'date']),
                    first_n=int(np.isfinite(values[w][i,j]).sum()),second_n=int(np.isfinite(values[w][k,j]).sum()),
                    first_mean=means[w][i,j],second_mean=means[w][k,j],difference=means[w][k,j]-means[w][i,j],
                    difference_boot_p025=lo,difference_boot_p975=hi,n_valid_boot=int(valid.sum()),
                    same_sign=bool(means[w][i,j]*means[w][k,j]>0),
                    first_date_centered=centered[w][i,j],second_date_centered=centered[w][k,j]))
    pd.DataFrame(repeat_rows).to_csv(TABLES/'overview_cross_date.csv',index=False)

    # Main overview deliberately retains both dates for the six repeat strains.
    # One common linear color scale in original dF/F units, no z-score per neuron.
    vmin,vmax=-.65,3.0
    cmap=LinearSegmentedColormap.from_list('response',[(0,'#174c79'),((-vmin)/(vmax-vmin),'#fbfaf6'),(1,'#a51b22')])
    fig,axes=plt.subplots(1,2,figsize=(13.4,20.4),sharey=True,gridspec_kw={'wspace':.08})
    labels=[f'{r.sample_id}{"*" if r.sample_id in repeats else ""}' for r in sd.itertuples()]
    for ax,w,title in zip(axes,['stim','post'],['Stimulus present: 0–10 s','After stimulus: 10–30 s']):
        im=ax.imshow(means[w],aspect='auto',cmap=cmap,norm=Normalize(vmin,vmax),interpolation='none')
        ax.set_xticks(range(len(NEURONS)),NEURONS,rotation=60,ha='left',fontsize=10)
        ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0)
        ax.set_yticks(range(len(sd)),labels,fontsize=6.9)
        ax.set_title(title,pad=64,fontsize=13)
        for dt,ix in date_rows.items():
            ax.axhline(ix[-1]+.5,color='#444444',lw=.8)
        ax.spines[['right','top','left','bottom']].set_visible(False)
    # Date labels on right make shared batches evident without collapsing strains.
    for dt,ix in date_rows.items():
        axes[1].text(len(NEURONS)-.25,np.mean(ix),str(dt),va='center',fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.89,top=.875,bottom=.08)
    cax=fig.add_axes([.31,.035,.4,.013])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.5,0,.5,1,2,3])
    cb.set_label('Mean calcium response, '+UNITS+' (shared linear scale)',fontsize=10)
    fig.suptitle('Broad-window calcium response overview',x=.48,y=.985,fontsize=15)
    fig.text(.07,.95,'112 strain–date rows; 106 strains, 9 dates, 49 animals. * = strain measured on two dates.\nEach cell: equal-weight animal mean after averaging repeated trials and available bilateral channels.',fontsize=10)
    fig.savefig(OUT/'figures/01_neural_overview.png',dpi=170,facecolor='white')
    fig.savefig(OUT/'figures/01_neural_overview.pdf',facecolor='white')
    plt.close(fig)
    heat=d[['sample_id','date','neuron_class','n_animals','n_trials','stim','post','full']].copy()
    heat['cross_date_repeat']=heat.sample_id.isin(repeats)
    heat.to_csv(TABLES/'overview_heatmap_data.csv',index=False)

    # Close-to-raw curves for targeted inspection; no selected trials are dropped.
    # The deterministic list is chosen from broad-window evidence (not peak timing).
    selected=['A024','A025','A044']
    awtop=sd.loc[np.argmax(means['stim'][:,NEURONS.index('AWCON')]),'sample_id']
    selected += [awtop]
    nearest=curves.reset_index()
    nearest=nearest[nearest.sample_id.isin(selected)&nearest.neuron_class.isin(['ADF','AWB','AWCON'])]
    nearest.to_csv(TABLES/'overview_selected_animal_curves.csv',index=False)
    fig,axes=plt.subplots(len(selected),3,figsize=(12.8,10.6),sharex=True,sharey='col')
    for i,strain in enumerate(selected):
        for j,n in enumerate(['ADF','AWB','AWCON']):
            ax=axes[i,j]
            subset=curves.xs((strain,n),level=('sample_id','neuron_class'))
            for k,(dt,g) in enumerate(subset.groupby(level='date')):
                color=['#156a70','#9b446f'][k]
                t=np.arange(-5,40)
                ax.plot(t,g.to_numpy().T,color=color,lw=.65,alpha=.4)
                ax.plot(t,g.mean(axis=0),color=color,lw=2,label=f'{dt}; n={len(g)} animals')
            ax.axvspan(0,10,color='#b7b2a9',alpha=.17,zorder=-2)
            ax.axhline(0,color='#888888',lw=.6,zorder=-1)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(-5,39)
            ax.legend(loc='upper right',fontsize=7,frameon=False)
            if i==0:
                ax.set_title(n,fontsize=12)
            if j==0:
                ax.set_ylabel(f'{strain}\n'+UNITS,fontsize=11)
            if i==len(selected)-1:
                ax.set_xlabel('Seconds after stimulus onset')
    fig.suptitle('Animal curves: two cross-date examples, a failed repeat, and strong AWCON activation',fontsize=13,y=.99)
    fig.text(.06,.952,'Thin lines: animals, after averaging trials/channels; thick lines: date means. Grey: stimulus present (0–10 s).\nCurves use measured calcium '+UNITS+' with upstream fitted baseline; selected examples, not a random validation set.',fontsize=10,va='top')
    fig.tight_layout(rect=(0,0,1,.935))
    fig.savefig(OUT/'figures/01_selected_animal_curves.png',dpi=170,facecolor='white')
    fig.savefig(OUT/'figures/01_selected_animal_curves.pdf',facecolor='white')
    plt.close(fig)
    curve_rows=[]
    for (strain,date,n),g in curves.groupby(level=['sample_id','date','neuron_class']):
        if strain not in selected or n not in ['ADF','AWB','AWCON']:
            continue
        # Diagnostics derived from all time points; raw fitted-baseline curves remain
        # in the companion table. They are not used to infer firing or sharp latency.
        m=g.mean(axis=0)
        curve_rows.append(dict(sample_id=strain,date=int(date),neuron_class=n,n_animals=len(g),
            stim_curve_mean=float(m[[str(t) for t in range(0,10)]].mean()),
            post_curve_mean=float(m[[str(t) for t in range(10,30)]].mean()),
            max_animal_timepoint=float(g.to_numpy().max()),min_animal_timepoint=float(g.to_numpy().min())))
    pd.DataFrame(curve_rows).to_csv(TABLES/'overview_selected_curve_checks.csv',index=False)
    meta=dict(seed=SEED,n_splits=N_SPLITS,n_bootstraps=N_BOOT,neurons=NEURONS,windows={'stim':[0,10],'post':[10,30],'full':[0,40]},
        analysis_units='Animal means within strain-date; true animal=(date,worm_key); stimulus samples from the same animal remain grouped.',
        split_intervals='5–95 percentiles across random partitions; NOT confidence intervals and NOT independent validation.',
        cross_date_bootstrap='Percentile 95% intervals, resample animals within each date jointly over all strains; conditional on 2 observed dates, not a population-of-dates CI.',
        date_centering='subtract date mean separately in each half using identical observed strain rows; removes common date offsets but cannot identify a strain effect separately from date.',
        selected_curve_strains=selected,
        input_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TABLES/'animal_metrics.csv',TABLES/'date_metrics.csv',TABLES/'animal_curves.parquet']},
        multiple_testing='No significance tests or p values. Data-driven ranking/feature selection is exploratory.',
        completed=['all 13 neuron broad-window overview','shared-animal split half','drop top 3 strains sensitivity','at least 2 animals per half sensitivity','six two-date comparisons','selected near-raw animal curve export'])
    (OUT/'logs/02_overview_metadata.json').write_text(json.dumps(meta,indent=2))
    print(diagnostic[['neuron_class','metric','split_r_median','split_r_p05','split_r_p95','split_spearman_median','drop3_r_median','date_grouping_fraction']].round(3).to_string(index=False))
    print('\nCross-date ADF:')
    print(pd.DataFrame(repeat_rows).query("neuron_class=='ADF'")[['sample_id','metric','first_n','second_n','first_mean','second_mean','difference','difference_boot_p025','difference_boot_p975']].round(3).to_string(index=False))
    print('\nStrongest AWCON stimulus strain:',awtop)
    print('Successful overview outputs written to',OUT)


if __name__=='__main__':
    main()
