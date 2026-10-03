"""Export interpretable contrasts and a small set of evidence displays.

No additional feature screening occurs here. Chosen examples and conditions
are documented in the exploration logs. All uncertainty displays are internal,
conditional descriptions; they do not adjust for exploratory selection.
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
OLD=ROOT/'reports/exploration_20260929/tables'
SEED=2026093009


def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    p=pd.read_csv(OUT/'tables/chemistry_population_predictions.csv')
    raw=p.copy()
    raw['sse']=raw.weight*(raw.observed-raw.predicted)**2
    raw['sst']=raw.weight*raw.observed**2
    scores=raw.groupby(['model','window','neuron_class'])[['sse','sst']].sum().reset_index()
    scores['relative_r2']=1-scores.sse/scores.sst
    scores['unit']='animal_strain'
    means=p.groupby(['model','sample_id','window','neuron_class']).agg(
        observed=('observed','mean'),predicted=('predicted','mean')).reset_index()
    means.to_csv(OUT/'tables/synthesis_chem_strain_predictions.csv',index=False)
    means['sse']=(means.observed-means.predicted)**2;means['sst']=means.observed**2
    ms=means.groupby(['model','window','neuron_class'])[['sse','sst']].sum().reset_index()
    ms['relative_r2']=1-ms.sse/ms.sst;ms['unit']='strain_mean'
    pd.concat([scores,ms]).to_csv(OUT/'tables/synthesis_chem_raw_scores.csv',index=False)
    metadata=pd.read_csv(OLD/'chemical_feature_metadata.csv')
    exact={str(x).strip().casefold() for x in metadata.name}
    pd.DataFrame([{'literature_compound':c,'exact_report_annotation_present':c.casefold() in exact,
                   'related_annotation_not_a_substitute':r} for c,r in [
        ('Cadaverine','N-Acetylcadaverine'),('Putrescine','N-Acetylputrescine'),
        ('Spermidine',''),('Spermine','')]]).to_csv(OUT/'tables/literature_candidate_coverage.csv',index=False)

    # Exact same-animal contrast examples. Animal centering cancels in each
    # contrast; these predictions can thus be compared in original dF/F units.
    joint=pd.read_csv(OUT/'tables/population_escher_lacto_joint_curves.csv')
    shared=joint[['date','worm_key']].drop_duplicates()
    cases=[('Escherichia_minus_Lactobacillus',['A065','A084','A215'],['A138','A206'],shared),
           ('A189_minus_A178',['A189'],['A178'],pd.DataFrame({'date':[20260313]*4,'worm_key':['w1','w3','w4','w5']}))]
    contrasts=[]
    for name,plus,minus,animals in cases:
        rows=p[p.sample_id.isin(plus+minus)].merge(animals,on=['date','worm_key'],validate='many_to_one')
        rows['side']=np.where(rows.sample_id.isin(plus),'plus','minus')
        g=rows.groupby(['model','window','neuron_class','date','worm_key','side'])[['observed','predicted']].mean().unstack('side')
        diff=g.xs('plus',axis=1,level=1)-g.xs('minus',axis=1,level=1)
        diff=diff.dropna().reset_index();diff['contrast']=name
        contrasts.append(diff)
    contrasts=pd.concat(contrasts,ignore_index=True)
    contrasts.to_csv(OUT/'tables/synthesis_contrast_predictions_animals.csv',index=False)
    csummary=contrasts.groupby(['contrast','model','window','neuron_class']).agg(
        observed=('observed','mean'),predicted=('predicted','mean'),n_animals=('observed','size'),
        observed_min=('observed','min'),observed_max=('observed','max')).reset_index()
    csummary.to_csv(OUT/'tables/synthesis_contrast_predictions.csv',index=False)

    # A focused within-genus example in near-source calcium curves: reciprocal
    # ADF/AWA differences in the same four animals, without a date legend.
    curves=pd.read_parquet(OLD/'animal_curves.parquet').reset_index()
    subset=curves[curves.sample_id.isin(['A178','A189'])&curves.neuron_class.isin(['ADF','AWA'])].copy()
    assert subset.groupby(['sample_id','neuron_class']).size().eq(4).all()
    subset.to_csv(OUT/'tables/figure_bifidobacterium_curves.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharex=True,sharey=True)
    colors={'A178':'#0072B2','A189':'#D55E00'}
    labels={'A178':'A178 · B. longum','A189':'A189 · B. stercoris'}
    time=np.arange(-5,40)
    for ax,n in zip(axes,['ADF','AWA']):
        ax.axvspan(0,10,color='#dddddd',alpha=.6,zorder=0)
        ax.axhline(0,color='.6',lw=.6)
        for strain in ['A178','A189']:
            values=subset[subset.sample_id.eq(strain)&subset.neuron_class.eq(n)][[str(t) for t in time]].to_numpy()
            for row in values:
                ax.plot(time,row,color=colors[strain],alpha=.28,lw=.9)
            ax.plot(time,values.mean(axis=0),color=colors[strain],lw=2.3,label=labels[strain])
        ax.set(title=n,xlabel='Seconds after stimulus onset',xlim=(-5,39))
    axes[0].set_ylabel(r'Calcium response, $\Delta F/F_0$')
    axes[1].legend(frameon=False,loc='upper right',fontsize=9)
    fig.suptitle('Within Bifidobacterium: stronger ADF accompanies weaker AWA',fontsize=13)
    fig.text(.5,-.025,'Same 4 animals · thin lines: animals; thick lines: means · grey: stimulus',ha='center',fontsize=10)
    fig.tight_layout()
    save(fig,'02_within_genus_composition')

    f=pd.read_csv(OUT/'tables/temporal_feature_animal.csv')
    pairs=[]
    for variant,d in f[f.sample_id.isin(['A178','A189'])].groupby('variant'):
        d=d.set_index(['date','worm_key','sample_id'])
        for metric in ['stim_ADF','stim_AWA','full_ADF','full_AWA','full_adf_minus_ash_normalized','full_adf_minus_awa_normalized']:
            delta=d[metric].unstack('sample_id').diff(axis=1)['A189'].dropna()
            for (date,worm),v in delta.items():
                pairs.append(dict(variant=variant,metric=metric,date=date,worm_key=worm,delta=v))
    pd.DataFrame(pairs).to_csv(OUT/'tables/synthesis_bifido_paired_effects.csv',index=False)

    # Supporting information display, not a substitute for the response figures.
    a=pd.read_csv(OUT/'tables/information_animal_accuracy.csv')
    modes=['magnitude','best_cell','raw','raw_shape']
    labels=['Total magnitude','Training-selected\nsingle cell','Population\nresponse','Population\ndirection only']
    sub=a[a.window.eq('stim')&a['mode'].isin(modes)].copy()
    sub.to_csv(OUT/'tables/figure_information_points.csv',index=False)
    rng=np.random.default_rng(SEED)
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    wide=sub.pivot(index=['date','worm_key'],columns='mode',values='accuracy').reindex(columns=modes)
    jitter=rng.uniform(-.075,.075,len(wide))
    palette=['#999999','#777777','#0072B2','#009E73']
    for i,m in enumerate(modes):
        vals=wide[m].to_numpy()*100
        ax.scatter(i+jitter,vals,s=14,color=palette[i],alpha=.65,zorder=2)
        ax.plot([i-.18,i+.18],[vals.mean()]*2,color='black',lw=2,zorder=3)
    chance=sub.chance.mean()*100
    ax.axhline(chance,ls='--',color='.4',lw=1,label=f'Average chance: {chance:.1f}%')
    ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='Correct strain identification (%)',ylim=(0,85))
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax.set_title('Cell composition carries information beyond response magnitude',fontsize=12,pad=13)
    fig.text(.5,-.015,'Stimulus [0,10) s · each point: one held-out animal (n=49)\n11–13 candidate strains per animal; all available cells held out together',ha='center',fontsize=9)
    fig.tight_layout()
    save(fig,'S01_population_information')
    (OUT/'logs/synthesis_summary.json').write_text(json.dumps(dict(
        seed=SEED,main_figures=['explore_population_combinations','02_within_genus_composition'],
        support_figures=['S01_population_information','explore_temporal_bifidobacterium'],
        selection='neural examples chosen during all-cell exploration; not confirmatory tests',
        chemical_score_units='synthesis_chem_raw_scores exports raw-unit SSE ratios separately for animal observations and strain means'),indent=2))
    print('Saved focused within-genus figure, supporting population figure and interpretable prediction contrasts.')


if __name__=='__main__':
    main()
