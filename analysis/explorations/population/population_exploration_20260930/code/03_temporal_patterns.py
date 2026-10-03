"""Broad-phase population calcium patterns; no spike timing inference.

Run from repo root with .pixi/envs/default/bin/python. Uses verified, read-only
previous-round trial curves. Unit: stored delta F/F0. The trial is aggregated
within animals; biological observations are (date,worm_key). Stage changes and
neuron relationships were selected after inspecting these data. No p values
or independent validation are claimed. Dates are matching blocks, not figures.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
SOURCE=ROOT/'reports/exploration_20260929/tables'
T=OUT/'tables'
KEY=['sample_id','date','worm_key','neuron_class']
NEURONS=['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']
PHASES={'stim':range(0,10),'post':range(10,30),'pooled':range(0,30),'full':range(0,40)}
SELECTED=['A178','A179','A189']
SEED=2026093003
NBOOT=10000
MIN_BALANCE_MASS=.1


def ci(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan
    rng=np.random.default_rng(SEED)
    return np.quantile(x[rng.integers(len(x),size=(NBOOT,len(x)))].mean(axis=1),[.025,.975])


def features(animal):
    w=animal.pivot(index=['sample_id','date','worm_key','variant'],columns='neuron_class',values=[*PHASES,'delta'])
    f=pd.DataFrame(index=w.index)
    for phase in [*PHASES,'delta']:
        for n in NEURONS:
            f[f'{phase}_{n}']=w[(phase,n)]
    for left,right in [('ADF','ASH'),('AWB','ASH'),('ADF','AWA')]:
        name=f'{left.lower()}_minus_{right.lower()}'
        for phase in PHASES:
            f[f'{phase}_{name}']=w[(phase,left)]-w[(phase,right)]
            mass=abs(w[(phase,left)])+abs(w[(phase,right)])
            f[f'{phase}_{name}_normalized']=f[f'{phase}_{name}'].div(mass).where(mass>=MIN_BALANCE_MASS)
        f[f'reorganization_{name}']=f[f'post_{name}']-f[f'stim_{name}']
        f[f'reorganization_{name}_normalized']=f[f'post_{name}_normalized']-f[f'stim_{name}_normalized']
    return f.reset_index()


def main():
    trial=pd.read_parquet(SOURCE/'trial_curves.parquet').reset_index().sort_values(KEY+['segment_index'])
    for name,window in PHASES.items():trial[name]=trial[[str(t) for t in window]].mean(axis=1)
    trial['first_retained']=trial.groupby(KEY).cumcount().eq(0)
    order=trial[trial.sample_id.isin(SELECTED)].groupby(KEY).segment_index.agg(
        first_segment='min',last_segment='max',n_trials='size',
        retained_segments=lambda x:';'.join(map(str,x)))
    order.to_csv(T/'temporal_selected_trial_order_internal.csv')
    animals=[]
    for variant,g,agg in [('mean',trial,'mean'),('median',trial,'median'),
                          ('first',trial[trial.first_retained],'mean'),('later',trial[~trial.first_retained],'mean')]:
        a=g.groupby(KEY)[list(PHASES)].agg(agg).reset_index()
        a['delta']=a.post-a.stim;a['variant']=variant
        animals.append(a)
    animals=pd.concat(animals,ignore_index=True)
    animals['delta_centered_within_animal']=animals.delta-animals.groupby(['date','worm_key','neuron_class','variant']).delta.transform('mean')
    animals.to_csv(T/'temporal_phase_animal.csv',index=False)
    f=features(animals)
    f.to_csv(T/'temporal_feature_animal.csv',index=False)
    numeric=[c for c in f if c not in ['sample_id','date','worm_key','variant']]
    df=f.groupby(['sample_id','date','variant'])[numeric].mean()
    df.to_csv(T/'temporal_feature_date_internal.csv')
    sf=df.groupby(['sample_id','variant']).mean().reset_index()
    taxonomy=pd.read_csv(SOURCE/'taxonomy.csv')
    sf=sf.merge(taxonomy[['sample_id','genus_clean','species_clean']],on='sample_id',validate='many_to_one')
    sf.to_csv(T/'temporal_feature_strain.csv',index=False)
    coverage=f.groupby(['sample_id','date','variant'])[numeric].count()
    coverage.to_csv(T/'temporal_feature_coverage.csv')
    common=animals.groupby(['sample_id','date','neuron_class','variant'])[['stim','post','delta','delta_centered_within_animal']].mean()
    common=common.groupby(['sample_id','neuron_class','variant']).mean().reset_index()
    common.to_csv(T/'temporal_phase_strain.csv',index=False)
    general=[]
    for n in NEURONS:
        a=animals[(animals.neuron_class==n)&(animals.variant=='mean')]
        s=common[(common.neuron_class==n)&(common.variant=='mean')]
        general.append(dict(neuron_class=n,n_strains=len(s),n_animal_strain=len(a),
             strain_negative_to_positive=int(((s.stim<0)&(s.post>0)).sum()),
             animal_negative_to_positive_fraction=float(((a.stim<0)&(a.post>0)).mean()),
             animal_post_greater_fraction=float((a.post>a.stim).mean()),
             stim_post_strain_correlation=float(s.stim.corr(s.post)),
             mean_delta_equal_strains=float(s.delta.mean())))
    pd.DataFrame(general).to_csv(T/'temporal_general_phase_summary.csv',index=False)

    # The selected contrasts explicitly use the same animal for both strains;
    # subtracting each animal's common phase effect cancels algebraically.
    pairs=[]
    for left,right in [('A189','A178'),('A189','A179'),('A179','A178')]:
        l=f[f.sample_id.eq(left)].set_index(['date','worm_key','variant'])
        r=f[f.sample_id.eq(right)].set_index(['date','worm_key','variant'])
        common_idx=l.index.intersection(r.index)
        for idx in common_idx:
            for metric in numeric:
                lv=l.loc[idx,metric];rv=r.loc[idx,metric]
                if np.isfinite(lv) and np.isfinite(rv):
                    pairs.append(dict(left=left,right=right,date=idx[0],worm_key=idx[1],variant=idx[2],
                                      metric=metric,left_value=lv,right_value=rv,contrast=lv-rv))
    pairs=pd.DataFrame(pairs)
    pairs.to_csv(T/'temporal_selected_paired_animals.csv',index=False)
    summary=[]
    for key,g in pairs.groupby(['left','right','variant','metric']):
        lo,hi=ci(g.contrast)
        summary.append(dict(zip(['left','right','variant','metric'],key),n_animals=len(g),
            mean=g.contrast.mean(),median=g.contrast.median(),low=lo,high=hi,
            minimum=g.contrast.min(),maximum=g.contrast.max(),n_positive=int(g.contrast.gt(0).sum())))
    summary=pd.DataFrame(summary)
    summary.to_csv(T/'temporal_selected_paired_summary.csv',index=False)

    # A scalar gain is fitted using 13-class vectors where available. This is
    # descriptive and does not establish calibrated gain across neuron classes.
    gain=[]
    for (date,worm,variant),g in f[f.sample_id.isin(SELECTED)].groupby(['date','worm_key','variant']):
        for target,ref in [('A189','A178'),('A189','A179'),('A179','A178')]:
            if not {target,ref}.issubset(set(g.sample_id)):continue
            q=g.set_index('sample_id')
            for phase in PHASES:
                cols=[f'{phase}_{n}' for n in NEURONS]
                x=q.loc[ref,cols].to_numpy(float);y=q.loc[target,cols].to_numpy(float)
                keep=np.isfinite(x)&np.isfinite(y);x=x[keep];y=y[keep]
                if len(x)<6 or np.linalg.norm(x)==0 or np.linalg.norm(y)==0:continue
                beta=max(0,float(x@y/(x@x)))
                gain.append(dict(target=target,reference=ref,date=date,worm_key=worm,variant=variant,
                    phase=phase,n_neurons=len(x),gain=beta,
                    relative_residual_norm=float(np.linalg.norm(y-beta*x)/np.linalg.norm(y)),
                    cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))))
    pd.DataFrame(gain).to_csv(T/'temporal_population_gain_check.csv',index=False)

    # One discovery figure, grouped by strains. Missing neuron classes are not
    # imputed. Heatmaps use available-animal means, while matching tests above
    # require the needed classes jointly in each animal.
    curves=pd.read_parquet(SOURCE/'animal_curves.parquet').reset_index()
    curves=curves[curves.sample_id.isin(SELECTED)]
    curves.to_csv(T/'temporal_selected_curves.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'font.family':'DejaVu Sans'})
    fig=plt.figure(figsize=(13,8),constrained_layout=True)
    gs=fig.add_gridspec(2,3,height_ratios=[1,1.05])
    displays=['ADF','ASH','AWA','AWB'];colors=['#cc4c02','#756bb1','#238b45','#2171b5']
    times=np.arange(-5,40)
    shown=curves[curves.neuron_class.isin(displays)][[str(t) for t in times]].to_numpy()
    ymin=min(-.4,float(np.nanmin(shown))-.1);ymax=float(np.nanmax(shown))+.15
    cm=common[(common.variant=='mean')&common.sample_id.isin(SELECTED)]
    clim=max(abs(cm[['stim','post']].to_numpy()).max(),.1)
    for i,strain in enumerate(SELECTED):
        ax=fig.add_subplot(gs[0,i]);q=cm[cm.sample_id.eq(strain)].set_index('neuron_class').reindex(NEURONS)
        h=ax.imshow(q[['stim','post']],cmap='RdBu_r',vmin=-clim,vmax=clim,aspect='auto')
        ax.set(yticks=np.arange(13),yticklabels=NEURONS,xticks=[0,1],xticklabels=['Stimulus\n[0,10) s','After removal\n[10,30) s'])
        species=taxonomy.set_index('sample_id').loc[strain,'species_clean'].replace('Bifidobacterium','B.')
        ax.set_title(f'{strain}: {species.strip()}')
        if i==2:fig.colorbar(h,ax=ax,label=r'Mean $\Delta F/F_0$',fraction=.045,pad=.035)
        bx=fig.add_subplot(gs[1,i])
        for n,col in zip(displays,colors):
            q=curves[(curves.sample_id==strain)&(curves.neuron_class==n)]
            vals=q[[str(t) for t in times]].to_numpy()
            for v in vals:bx.plot(times,v,color=col,alpha=.2,lw=.7)
            bx.plot(times,vals.mean(axis=0),color=col,lw=2,label=f'{n}, n={len(q)}')
        bx.axvspan(0,10,color='.8',alpha=.35);bx.axhline(0,color='.6',lw=.6)
        bx.set(xlim=(-5,39),ylim=(ymin,ymax),xlabel='Seconds from stimulus onset',ylabel=r'$\Delta F/F_0$')
        bx.legend(loc='upper right',frameon=False,fontsize=8)
    fig.suptitle('Different population calcium trajectories among three Bifidobacterium stimuli\nSame four animals; thin curves are animals, thick curves are means',fontsize=12)
    fig.savefig(OUT/'figures/explore_temporal_bifidobacterium.png')
    fig.savefig(OUT/'figures/explore_temporal_bifidobacterium.pdf')
    plt.close(fig)
    # Assert that the mean stage metrics reproduce verified preparation output.
    reference=pd.read_csv(SOURCE/'animal_metrics.csv',dtype={'date':str})
    m=animals[animals.variant.eq('mean')].merge(reference,on=KEY,suffixes=('_new','_reference'),validate='one_to_one')
    error=float(max(abs(m.stim_new-m.stim_reference).max(),abs(m.post_new-m.post_reference).max()))
    assert error<1e-12
    retained=['reorganization_adf_minus_ash','reorganization_adf_minus_ash_normalized','post_adf_minus_ash_normalized','full_adf_minus_ash_normalized','pooled_adf_minus_ash_normalized','stim_ADF','post_ADF','stim_AWA','post_AWA']
    result=dict(seed=SEED,bootstrap_replicates=NBOOT,normalization_min_mass=MIN_BALANCE_MASS,
                selected_strains=SELECTED,selection='After 13-class broad-phase screening; same-animal same-genus triad; not confirmatory.',
                phase_consistency_max_error=error,plot_curve_min=ymin,plot_curve_max=ymax,
                first_trial_definition='First retained segment within animal and strain, not proven first lifetime exposure.',
                interpretation='Stored calcium response stages only; not firing, latency, or a causal removal response.',
                decoding='Not executed here: root agent owns shared-animal population decoding.',
                run_status='success')
    (OUT/'logs/temporal_run_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    print(summary[(summary.left=='A189')&(summary.right=='A178')&summary.metric.isin(retained)].round(4).to_string(index=False))


if __name__=='__main__':main()
