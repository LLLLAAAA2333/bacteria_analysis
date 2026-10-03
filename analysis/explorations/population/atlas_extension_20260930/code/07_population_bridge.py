"""Can held-out chemical predictions reconstruct a neural-defined population contrast?

The high/low groups were selected by other animals' ASI/ASJ post responses,
never by chemistry. Chemical models exclude the entire test block. This is
an exploratory bridge, not independent validation of the selected phenotype.
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
CELLS=['ASI','ASJ','ASK','ADF','ASH']
SEED=2026093017


def main():
    assignment=pd.read_csv(OUT/'tables/response_patterns_held_assignments.csv')
    pred=pd.read_csv(OUT/'tables/targeted_chemistry_predictions.csv')
    pred=pred[pred.window.eq('post') & pred.neuron_class.isin(CELLS)]
    joined=pred.merge(assignment[['date','worm_key','sample_id','group']],on=['date','worm_key','sample_id'],validate='many_to_one')
    records=[]
    for (model,date,worm),d in joined.groupby(['model','date','worm_key']):
        counts=d.groupby(['neuron_class','group']).sample_id.nunique()
        if len(counts)!=2*len(CELLS) or not counts.eq(3).all():continue
        mean=d.groupby(['neuron_class','group'])[['observed','predicted']].mean()
        for n in CELLS:
            delta=mean.loc[(n,'higher')]-mean.loc[(n,'lower')]
            records.append(dict(model=model,date=date,worm_key=worm,neuron_class=n,
                                observed=delta.observed,predicted=delta.predicted))
    contrasts=pd.DataFrame(records)
    assert len(contrasts)>0
    contrasts.to_csv(OUT/'tables/population_bridge_animal_contrasts.csv',index=False)
    rng=np.random.default_rng(SEED);summaries=[]
    for (model,cell),g in contrasts.groupby(['model','neuron_class']):
        boot=np.zeros((2000,2))
        for _,q in g.groupby('date'):
            arr=q[['observed','predicted']].to_numpy()
            boot+=arr[rng.integers(0,len(q),(2000,len(q)))].sum(axis=1)
        boot/=len(g)
        lo,hi=np.quantile(boot,[.025,.975],axis=0)
        summaries.append(dict(model=model,neuron_class=cell,n_animals=len(g),n_blocks=g.date.nunique(),
            observed=g.observed.mean(),predicted=g.predicted.mean(),
            observed_low=lo[0],observed_high=hi[0],predicted_low=lo[1],predicted_high=hi[1],
            observed_positive=int(g.observed.gt(0).sum()),predicted_positive=int(g.predicted.gt(0).sum()),
            signs_agree=int((g.observed*g.predicted>0).sum())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'tables/population_bridge_summary.csv',index=False)
    # Actual membership and taxonomic concentration are part of the evidence.
    tax=pd.read_csv(ROOT/'reports/exploration_20260929/tables/taxonomy.csv')
    eligible=contrasts[['date','worm_key']].drop_duplicates()
    support=assignment.merge(eligible,on=['date','worm_key']).merge(tax[['sample_id','genus_clean','species_clean']],on='sample_id')
    support.to_csv(OUT/'tables/population_bridge_membership.csv',index=False)
    support.groupby(['group','genus_clean']).agg(n_assignments=('sample_id','size'),n_strains=('sample_id','nunique')).to_csv(OUT/'tables/population_bridge_genus_coverage.csv')
    # One question, one panel: response redistribution and its reconstruction.
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    obs=contrasts[contrasts.model.eq('panel162')]
    for i,n in enumerate(CELLS):
        g=obs[obs.neuron_class.eq(n)]
        ax.scatter(g.observed,np.full(len(g),i)+rng.uniform(-.13,.13,len(g)),s=14,c='.6',alpha=.55,zorder=1)
    configurations=[('observed','panel162','#242424',-.18,'Measured calcium'),
                    ('predicted','panel162','#b96c20',0,'Chemical panel (162 features)'),
                    ('predicted','taxonomy','#39729a',.18,'Genus + chemical level')]
    for field,model,color,offset,label in configurations:
        s=summary[summary.model.eq(model)].set_index('neuron_class').reindex(CELLS)
        ax.errorbar(s[field],np.arange(len(CELLS))+offset,
                    xerr=np.array([s[field]-s[field+'_low'],s[field+'_high']-s[field]]),
                    fmt='o',ms=5,color=color,elinewidth=1.3,capsize=2,label=label)
    ax.axvline(0,color='.6',lw=.8);ax.set_yticks(range(len(CELLS)),CELLS);ax.invert_yaxis()
    ax.set_xlabel(r'Higher minus lower ASI/ASJ group: 10–30 s response ($\Delta F/F_0$)')
    ax.set_title('Which parts of the population shift does chemistry reconstruct?',fontsize=12)
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8,loc='upper left')
    n=obs[['date','worm_key']].drop_duplicates().shape[0]
    fig.text(.02,.015,f'Identical {n} animals with all five neurons and six selected strains. Gray points: measured animal contrasts.\nBars: descriptive 95% animal-resampling ranges; fixed selections and predictions, not independent validation.',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(OUT/'figures/population_bridge.png',dpi=220);fig.savefig(OUT/'figures/population_bridge.pdf');plt.close(fig)
    (OUT/'logs/population_bridge_methods.json').write_text(json.dumps(dict(seed=SEED,
        selection='Each held animal: high/low three strains selected using ASI/ASJ post responses in >=2 other animals.',
        coverage='Require both groups of three strains and all five cells observed/predicted in the same animal; globally repeated strains excluded by chemical training.',
        chemical_split='Nested leave-block; no same-block responses used in chemical model fit.',
        uncertainty='2000 animal resamples within blocks conditional on fixed selections and predictions; descriptive ranges only.',
        limitation='Data-discovered neural target; species/genus composition can explain apparent system-wide agreement. Models selected within training, phenotype selected across this dataset.'),indent=2))
    print(summary.round(4).to_string(index=False));print(support.groupby(['group','genus_clean']).size().to_string())


if __name__=='__main__':main()
