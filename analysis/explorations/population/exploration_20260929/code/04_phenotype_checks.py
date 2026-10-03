"""Animal-level, design-aware checks of two discovered response phenotypes.

Input: 01_prepare.py outputs and original 106bac.parquet (read only).
All response numbers are stored delta F/F0; t=0 is stimulus onset, stimulus
lasts [0,10) s. No inferential p values: phenotypes were selected from these
data. Each marginal bootstrap interval resamples its animal-level means or
already paired contrasts within a date; these are not simultaneous intervals.
They describe animal uncertainty conditional on these dates,
not independent validation, uncertainty about selected groups, or date effects.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
T = OUT / 'tables'
SEED = 2026092904
NBOOT = 10000
KEY = ['sample_id', 'date', 'worm_key', 'neuron_class']
TRIALKEY = KEY + ['segment_index']
TIME = np.arange(-5,40)
REPEATED = ['A011','A013','A014','A024','A025','A044']
HIGH = ['A247','A296','A290','A288','A289','A302','A300']
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
COL = ['#2166ac','#b2182b']


def boot(x, seed=SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    return np.quantile(x[rng.integers(0,len(x),(NBOOT,len(x)))].mean(axis=1), [.025,.975])


def summarize(df, groups, value):
    rows=[]
    for key,g in df.groupby(groups,observed=True):
        if not isinstance(key,tuple): key=(key,)
        lo,hi=boot(g[value])
        rows.append(dict(zip(groups,key), n_animals=len(g), mean=g[value].mean(),
                         low=lo,high=hi,min=g[value].min(),max=g[value].max()))
    return pd.DataFrame(rows)


def paired_contrasts(df, variants):
    """High minus other strains within each animal/date/class; equal strains."""
    dates=sorted(df.loc[df.sample_id.isin(HIGH),'date'].unique())
    rows=[]
    for (date,worm,neuron),g in df[df.date.isin(dates)].groupby(['date','worm_key','neuron_class']):
        for variant in variants:
            h=g[g.sample_id.isin(HIGH)][variant].dropna()
            l=g[~g.sample_id.isin(HIGH)][variant].dropna()
            if len(h) and len(l):
                rows.append(dict(date=date,worm_key=worm,neuron_class=neuron,variant=variant,
                                 n_high_strains=len(h),n_other_strains=len(l),
                                 high_mean=h.mean(),other_mean=l.mean(),contrast=h.mean()-l.mean()))
    return pd.DataFrame(rows)


def main():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180, 'font.family':'DejaVu Sans'})
    trial=pd.read_parquet(T/'trial_curves.parquet').reset_index()
    animal_curves=pd.read_parquet(T/'animal_curves.parquet').reset_index()
    taxonomy=pd.read_excel(ROOT/'data/GM300_bacteria_species_summary.xlsx').rename(columns={'AID':'sample_id'})
    taxonomy=taxonomy[['sample_id','genus_clean','species_clean']]
    trial['stim']=trial[[str(i) for i in range(10)]].mean(axis=1)
    trial['post']=trial[[str(i) for i in range(10,30)]].mean(axis=1)
    trial=trial.sort_values(TRIALKEY)
    base=trial.groupby(KEY).agg(mean=('stim','mean'),median=('stim','median'),first=('stim','first'),
                               n_trials=('stim','size'),mean_segment=('segment_index','mean'))
    # First retained trial for each strain/animal, not necessarily first exposure
    # in the experiment: upstream exclusions may have removed earlier trials.
    is_first=trial.groupby(KEY).cumcount().eq(0)
    later=trial[~is_first].groupby(KEY).stim.mean().rename('later')
    base=base.join(later)

    # QC refutation: remove an entire neuron-class trial if any original side
    # has any stored value below -1. This is a flag, not proof of raw corruption:
    # upstream baseline handling can yield such values, as discussed in report.
    raw=pd.read_parquet(ROOT/'data/106bac.parquet',columns=['date','worm_key','segment_index','neuron','delta_F_over_F0','stim_name'])
    raw['sample_id']=raw.stim_name.str.extract(r'^(A\d{3})',expand=False)
    bilateral=NEURONS[:9]
    raw['neuron_class']=raw.neuron.replace({n+s:n for n in bilateral for s in ['L','R']})
    invalid=(raw[raw.neuron_class.isin(NEURONS)].groupby(TRIALKEY).delta_F_over_F0.min().lt(-1).rename('below_minus1'))
    trial=trial.merge(invalid.reset_index(),on=TRIALKEY,how='left',validate='one_to_one')
    assert trial.below_minus1.notna().all()
    base=base.join(trial[~trial.below_minus1].groupby(KEY).stim.mean().rename('qc_exclude_below_minus1'))

    # A strain intercept for every animal and a shared linear segment trend
    # within each animal/class. Fit the trend only to within-strain changes,
    # then align animal means to their own mean segment. No trial-level test.
    centers=trial.groupby(KEY)[['stim','segment_index']].transform('mean')
    trial['dx']=trial.segment_index-centers.segment_index
    trial['dy']=trial.stim-centers.stim
    trial['dxdy']=trial.dx*trial.dy
    trial['dx2']=trial.dx**2
    order=trial.groupby(['date','worm_key','neuron_class']).agg(dxdy=('dxdy','sum'),dx2=('dx2','sum'),
                                        reference_segment=('segment_index','mean'),n_trials=('stim','size'))
    order['slope_per_segment']=order.dxdy/order.dx2.replace(0,np.nan)
    trial=trial.merge(order[['reference_segment','slope_per_segment']].reset_index(),on=['date','worm_key','neuron_class'],validate='many_to_one')
    trial['order_adjusted']=trial.stim-trial.slope_per_segment*(trial.segment_index-trial.reference_segment)
    base=base.join(trial.groupby(KEY).order_adjusted.mean()).reset_index()
    base=base.merge(taxonomy,on='sample_id',how='left',validate='many_to_one')
    base['later_minus_first']=base.later-base['first']
    base.to_csv(T/'phenotype_animal_sensitivity.csv',index=False)
    trial[TRIALKEY+['stim','post','below_minus1','slope_per_segment','order_adjusted']].to_csv(T/'phenotype_trial_checks.csv',index=False)
    order.to_csv(T/'phenotype_order_slopes.csv')
    variants=['mean','median','first','later','qc_exclude_below_minus1','order_adjusted']
    strain_long=base.melt(id_vars=KEY,value_vars=variants,var_name='variant',value_name='value')
    date_summ=summarize(strain_long.dropna(subset=['value']),['sample_id','date','neuron_class','variant'],'value')
    date_summ.to_csv(T/'phenotype_strain_date_sensitivity.csv',index=False)
    paired=paired_contrasts(base,variants)
    paired.to_csv(T/'phenotype_high_other_paired_animals.csv',index=False)
    paired_summ=summarize(paired,['date','neuron_class','variant'],'contrast')
    paired_summ.to_csv(T/'phenotype_high_other_contrasts.csv',index=False)
    a247_a249=[]
    for (date,worm),g in base[(base.neuron_class=='AWCON')&base.sample_id.isin(['A247','A249'])].groupby(['date','worm_key']):
        g=g.set_index('sample_id')
        if {'A247','A249'}.issubset(g.index):
            for variant in variants:
                a247_a249.append(dict(date=date,worm_key=worm,variant=variant,
                    A247=g.loc['A247',variant],A249=g.loc['A249',variant],
                    contrast=g.loc['A247',variant]-g.loc['A249',variant]))
    a247_a249=pd.DataFrame(a247_a249)
    a247_a249.to_csv(T/'phenotype_a247_a249_paired_animals.csv',index=False)
    summarize(a247_a249,['date','variant'],'contrast').to_csv(T/'phenotype_a247_a249_contrasts.csv',index=False)
    paired_curve=[]
    for (date,worm,neuron),g in animal_curves[animal_curves.date.isin(paired.date)].groupby(['date','worm_key','neuron_class']):
        h=g[g.sample_id.isin(HIGH)];l=g[~g.sample_id.isin(HIGH)]
        if len(h) and len(l):
            for t in TIME:
                paired_curve.append(dict(date=date,worm_key=worm,neuron_class=neuron,time_s=t,
                    high_mean=h[str(t)].mean(),other_mean=l[str(t)].mean()))
    pd.DataFrame(paired_curve).to_csv(T/'phenotype_high_other_curves.csv',index=False)

    # Targeted contrast available with identical strain composition on both
    # dates: A024 minus average(A013,A014), requiring all three in each animal.
    adf=[]
    for (date,worm),g in base[(base.neuron_class=='ADF')&base.sample_id.isin(['A024','A013','A014'])].groupby(['date','worm_key']):
        g=g.set_index('sample_id')
        if set(['A024','A013','A014']).issubset(g.index):
            for variant in variants:
                vals=g.loc[['A024','A013','A014'],variant]
                if vals.notna().all():
                    adf.append(dict(date=date,worm_key=worm,variant=variant,
                          contrast=vals.loc['A024']-vals.loc[['A013','A014']].mean()))
    adf=pd.DataFrame(adf)
    adf.to_csv(T/'phenotype_adf_paired_animals.csv',index=False)
    adf_summ=summarize(adf,['date','variant'],'contrast')
    adf_summ.to_csv(T/'phenotype_adf_paired_contrasts.csv',index=False)

    # Figure 02: all six repeated strains. Thin curves/points are animals,
    # thick curves/diamonds equal-animal means; individual trial noise was
    # averaged before display. Identical limits prevent implicit exaggeration.
    fig,axs=plt.subplots(3,4,figsize=(14,9),gridspec_kw={'width_ratios':[3,1.2,3,1.2]},constrained_layout=True)
    for i,s in enumerate(REPEATED):
        row=i//2;col=(i%2)*2;ax=axs[row,col];bx=axs[row,col+1]
        g=animal_curves[(animal_curves.neuron_class=='ADF')&animal_curves.sample_id.eq(s)]
        dates=sorted(g.date.unique())
        for j,date in enumerate(dates):
            q=g[g.date.eq(date)]
            vals=q[[str(t) for t in TIME]].to_numpy()
            for curve in vals: ax.plot(TIME,curve,color=COL[j],alpha=.25,lw=.8)
            ax.plot(TIME,vals.mean(axis=0),color=COL[j],lw=2,label=f'{date}, n={len(q)}')
            b=base[(base.neuron_class=='ADF')&base.sample_id.eq(s)&base.date.eq(date)]['mean']
            bx.scatter(j+np.linspace(-.12,.12,len(b)),b,color=COL[j],s=20,alpha=.7)
            lo,hi=boot(b);bx.errorbar(j,b.mean(),yerr=[[b.mean()-lo],[hi-b.mean()]],fmt='D',color=COL[j],capsize=3,ms=5)
        ax.axvspan(0,10,color='.8',alpha=.35);ax.axhline(0,color='.65',lw=.6)
        ax.set(xlim=(-5,39),ylim=(-.3,2.8),title=s,xlabel='Seconds from stimulus onset',ylabel=r'ADF $\Delta F/F_0$')
        ax.legend(fontsize=7,loc='upper right',frameon=False)
        bx.axhline(0,color='.7',lw=.6);bx.set(xticks=[0,1],xticklabels=['Date 1','Date 2'],ylim=(-.1,1.5),ylabel='Mean [0,10) s')
    fig.suptitle('ADF: six repeated strains, two dates each; animal curves and stimulus-period means',fontsize=13)
    fig.savefig(OUT/'figures/02_adf_crossdate.png');fig.savefig(OUT/'figures/02_adf_crossdate.pdf');plt.close(fig)
    ac=animal_curves[(animal_curves.neuron_class=='ADF')&animal_curves.sample_id.isin(REPEATED)]
    ac.to_csv(T/'phenotype_adf_repeated_curves.csv',index=False)

    # Figure 03: discovery, same-animal contrasts, class specificity, raw curves.
    fig=plt.figure(figsize=(15,12),constrained_layout=True)
    gs=fig.add_gridspec(4,2,height_ratios=[1.25,1,1,1])
    ax=fig.add_subplot(gs[0,:])
    aw=base[base.neuron_class.eq('AWCON')]
    rank=aw.groupby(['sample_id','date'])['mean'].mean().groupby('sample_id').mean().sort_values()
    rank.to_csv(T/'phenotype_awcon_strain_rank.csv',header=['mean_equal_dates'])
    rng=np.random.default_rng(SEED)
    for i,s in enumerate(rank.index):
        q=aw[aw.sample_id.eq(s)]
        ax.scatter(i+rng.uniform(-.23,.23,len(q)),q['mean'],s=9,alpha=.48,color='#b2182b' if s in HIGH else '#52626b')
    ax.plot(range(len(rank)),rank.to_numpy(),color='k',lw=.8,zorder=3)
    ax.text(.03,.88,'Dots: animal means; line: equal-date strain mean\nRed: seven selected strains, named in panels D–G',
            transform=ax.transAxes,ha='left',va='top',fontsize=9)
    ax.axhline(0,color='.6',lw=.7)
    ax.set(xlabel='All 106 strains, ranked by equal-date mean (selected in these data)',ylabel=r'AWCON mean [0,10) s, $\Delta F/F_0$',title='A  Strong positive responses are concentrated in seven selected strains')
    ax=fig.add_subplot(gs[1,0])
    dates=sorted(aw[aw.sample_id.isin(HIGH)].date.unique())
    for i,date in enumerate(dates):
        q=paired[(paired.neuron_class=='AWCON')&paired.date.eq(date)&paired.variant.eq('mean')]
        lo,hi=boot(q.contrast);v=q.contrast.mean()
        ax.scatter(i+np.linspace(-.12,.12,len(q)),q.contrast,color='#2166ac',s=23,alpha=.7)
        ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],color='k',fmt='D',capsize=4,ms=6)
    ax.axhline(0,color='.5',lw=.7);ax.set(xticks=range(4),xticklabels=[f'{d}\nn={len(paired[(paired.neuron_class=="AWCON")&paired.date.eq(d)&paired.variant.eq("mean")])}' for d in dates],ylabel=r'High minus other strains, $\Delta F/F_0$',title='B  Same-animal contrasts remain positive on all four dates')
    ax=fig.add_subplot(gs[1,1])
    for j,date in enumerate(dates):
        q=paired_summ[(paired_summ.variant=='mean')&paired_summ.date.eq(date)].set_index('neuron_class').reindex(NEURONS)
        ax.plot(range(13),q['mean'],marker='o',ms=3,lw=1,alpha=.8,label=str(date))
    ax.axhline(0,color='.5',lw=.7);ax.set(xticks=range(13),xticklabels=NEURONS,ylabel=r'Within-animal contrast, $\Delta F/F_0$',title='C  Contrasts across the recorded neuron classes')
    ax.tick_params(axis='x',labelrotation=65);ax.legend(fontsize=7,frameon=False,ncol=2)
    for k,date in enumerate(dates):
        ax=fig.add_subplot(gs[2+k//2,k%2]);q=animal_curves[(animal_curves.neuron_class=='AWCON')&animal_curves.date.eq(date)]
        hs=[s for s in HIGH if s in set(q.sample_id)]
        colors=['#b2182b','#ef8a62']
        for j,s in enumerate(hs):
            vals=q[q.sample_id.eq(s)][[str(t) for t in TIME]].to_numpy()
            for v in vals:ax.plot(TIME,v,color=colors[j],alpha=.2,lw=.7)
            ax.plot(TIME,vals.mean(axis=0),color=colors[j],lw=2,label=f'{s}, n={len(vals)}')
        low=q[~q.sample_id.isin(HIGH)].groupby('worm_key')[[str(t) for t in TIME]].mean()
        low_label=f'Other {q[~q.sample_id.isin(HIGH)].sample_id.nunique()} strains / animal'
        if str(date)=='20260313':
            low=q[q.sample_id.eq('A249')].set_index('worm_key')[[str(t) for t in TIME]]
            low_label=f'A249, n={len(low)} (same animals)'
        for v in low.to_numpy():ax.plot(TIME,v,color='#52626b',alpha=.25,lw=.8)
        ax.plot(TIME,low.mean().to_numpy(),color='#52626b',lw=2,ls='--',label=low_label)
        ax.axvspan(0,10,color='.8',alpha=.35);ax.axhline(0,color='.6',lw=.6)
        title=f'{chr(68+k)}  {date}'
        if str(date)=='20260313':
            delta=a247_a249[a247_a249.variant.eq('mean')].contrast.mean()
            title+=f' | A247 minus A249: {delta:.3f}'
        ax.set(xlim=(-5,39),ylim=(-1,9.3),xlabel='Seconds from stimulus onset',ylabel=r'AWCON $\Delta F/F_0$',title=title)
        ax.legend(fontsize=8,frameon=False)
    fig.suptitle('AWCON: selective calcium responses in the measured sessions, without cross-date repeats of selected strains',fontsize=13)
    fig.savefig(OUT/'figures/03_awcon_response.png');fig.savefig(OUT/'figures/03_awcon_response.pdf');plt.close(fig)

    # Same species counterexamples and full per-animal first-vs-later behavior.
    species=aw[aw.sample_id.isin(HIGH)].species_clean.unique()
    peers=aw[aw.species_clean.isin(species)].copy()
    peers.to_csv(T/'phenotype_awcon_taxonomic_peers.csv',index=False)
    invalid_counts=(trial.groupby('neuron_class').below_minus1.agg(['sum','count']))
    invalid_counts.to_csv(T/'phenotype_qc_counts.csv')
    high_animals=aw[aw.sample_id.isin(HIGH)]
    report={
        'seed':SEED,'bootstrap_draws':NBOOT,'selected_high_strains':HIGH,
        'selection':'Data-driven gap in ranked stimulus-period AWCON means; seven >1.8. Descriptive, not a prespecified test.',
        'awcon_high_dates':list(map(int,dates)),
        'awcon_high_animal_strain_pairs':len(high_animals),
        'awcon_high_distinct_animals':len(high_animals[['date','worm_key']].drop_duplicates()),
        'awcon_high_animal_mean_range':[float(high_animals['mean'].min()),float(high_animals['mean'].max())],
        'awcon_high_positive_animal_strain_pairs':int(high_animals['mean'].gt(0).sum()),
        'awcon_high_later_lower_than_first':int(high_animals.later_minus_first.lt(0).sum()),
        'awcon_high_first_later_pairs':int(high_animals.later.notna().sum()),
        'awcon_high_median_later_minus_first':float(high_animals.later_minus_first.median()),
        'awcon_qc_dropped_trials':int(invalid_counts.loc['AWCON','sum']),
        'adf_qc_dropped_trials':int(invalid_counts.loc['ADF','sum']),
        'checks_successful':['trial median','first retained trial versus later trials','exclude class trial with original trace value below -1','animal-by-strain intercept plus animal-specific linear segment trend','same-animal same-date strain contrasts','same-species descriptive counterexamples','all six repeated strains shown'],
        'not_identifiable':['Strain versus date for the seven AWCON high-response strains','Causal effects of stimulus order, carryover, and elapsed time separately','Generality beyond the measured strain/date combinations','Independent validation after phenotype selection'],
    }
    (OUT/'logs/04_phenotype_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('\nADF paired contrast:\n'+adf_summ.round(4).to_string(index=False))
    print('\nAWCON paired contrasts:\n'+paired_summ[paired_summ.neuron_class.eq('AWCON')].round(4).to_string(index=False))
    print('\nAll class date contrasts:\n'+paired_summ[paired_summ.variant.eq('mean')].pivot(index='neuron_class',columns='date',values='mean').round(3).to_string())
    print('\nHigh strain sensitivities:\n'+date_summ[date_summ.neuron_class.eq('AWCON')&date_summ.sample_id.isin(HIGH)].pivot(index='sample_id',columns='variant',values='mean').round(3).to_string())


if __name__=='__main__':
    main()
