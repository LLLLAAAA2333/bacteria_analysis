"""Three saved-table diagnostic figures; no model fitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

PRIMARY='primary_unit_adf_minus_ash'


def _save(fig,out,name):
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def _style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.16)


def _date_label(label):
    return '; '.join(d[:4]+'-'+d[4:6]+'-'+d[6:8] for d in str(label).split(';'))


def plot_saved(result_dir,out=None):
    result=Path(result_dir).resolve();out=result/'figures' if out is None else Path(out).resolve()
    if out.is_dir() and any(out.iterdir()):raise FileExistsError('Choose a new empty figure directory.')
    tables=result/'tables'
    data=pd.read_csv(tables/'ordered_strains_and_thirds.csv',dtype={'dates':str})
    heldout=pd.read_csv(tables/'heldout_predictions.csv')
    assoc=pd.read_csv(tables/'fixed_axis_associations.csv').set_index('response')
    performance=pd.read_csv(tables/'pooled_performance.csv')
    leverage=pd.read_csv(tables/'strain_leverage_and_residual.csv')
    summary=json.loads((result/'summary.json').read_text())
    assert len(data)==29
    dates=sorted(data.dates.unique());palette=plt.get_cmap('tab10')
    colors={d:palette(i) for i,d in enumerate(dates)}
    heldout=heldout.drop(columns=[c for c in ['dates','taxonomy_note'] if c in heldout]).merge(
        data[['strain','dates','taxonomy_note']],on='strain',validate='many_to_one')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    out.mkdir(parents=True,exist_ok=True)
    handles=[Line2D([0],[0],marker='o',ls='',color=colors[d],label=_date_label(d)) for d in dates]

    def scatter(ax,g,x,y):
        ax.scatter(g[x],g[y],c=[colors[d] for d in g.dates],s=34,edgecolors='white',linewidths=.5,zorder=3)
        flagged=g.taxonomy_note.notna()
        ax.scatter(g.loc[flagged,x],g.loc[flagged,y],s=57,facecolors='none',edgecolors='.25',linewidths=.8,zorder=4)

    fig,axes=plt.subplots(1,3,figsize=(15.0,5.6),constrained_layout=True)
    ax=axes[0];scatter(ax,data,'chemical_state_score',PRIMARY)
    ax.plot(data.chemical_state_score,data['pred_'+PRIMARY],color='.2',lw=1.4,zorder=2)
    # Identity labels follow fixed x leverage, not response fit quality.
    top=leverage.nlargest(3,'OLS_leverage').strain.tolist()
    for _,r in data[data.strain.isin(top)].iterrows():
        ax.annotate(r.strain,(r.chemical_state_score,r[PRIMARY]),xytext=(5,7),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Selected chemical-state score',ylabel='ADF − ASH (unit coordinates)',title='Full cohort: 29 strains')
    _style(ax)
    schemes=[('leave_one_strain_out','Held-out strains'),('leave_one_recorded_species_out','Held-out recorded species')]
    limvals=np.r_[heldout[PRIMARY],heldout['pred_'+PRIMARY]]
    lo,hi=limvals.min(),limvals.max();pad=(hi-lo)*.08
    for ax,(scheme,title) in zip(axes[1:],schemes):
        g=heldout[heldout.scheme.eq(scheme)];assert len(g)==29
        scatter(ax,g,PRIMARY,'pred_'+PRIMARY)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='.55',ls='--',lw=1,zorder=1)
        ax.set(xlabel='Observed ADF − ASH',ylabel='Held-out prediction',title=title,
               xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad))
        ax.set_aspect('equal',adjustable='box');_style(ax)
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,
               title='Original recorded-date set',fontsize=8,title_fontsize=9)
    fig.suptitle('One selected chemical state and the fixed neural contrast',fontsize=14,fontweight='bold')
    _save(fig,out,'01_chemical_relationship_and_heldout')

    fig,axes=plt.subplots(2,2,figsize=(12.8,9.2),constrained_layout=True)
    limits=np.r_[data.unit_ADF,data.unit_ASH];low,high=limits.min(),limits.max();pad=(high-low)*.12
    for ax,cell in zip(axes[0],['ADF','ASH']):
        target='unit_'+cell;scatter(ax,data,'chemical_state_score',target)
        ax.plot(data.chemical_state_score,data['pred_'+target],color='.2',lw=1.4)
        ax.set(xlabel='Selected chemical-state score',ylabel=f'{cell} unit coordinate',
               title=f'{cell} separately',ylim=(low-pad,high+pad));_style(ax)
    ax=axes[1,0]
    for i,tier in enumerate(['LOW','MID','HIGH']):
        g=data[data.chemical_third.eq(tier)].copy()
        g['plot_x']=i+np.linspace(-.18,.18,len(g))
        scatter(ax,g,'plot_x',PRIMARY)
        ax.plot([i-.22,i+.22],[g[PRIMARY].median()]*2,color='.2',lw=2,zorder=2)
    ax.set_xticks([0,1,2],['Low (10)','Middle (10)','High (9)'])
    ax.set(xlabel='Chemical-score rank thirds',ylabel='ADF − ASH (unit coordinates)',title='Individual support and overlap')
    _style(ax)
    ax=axes[1,1];cells=summary['neural_coordinate_order']
    effects=assoc.loc[['unit_'+c for c in cells],'fitted_observed_iqr_effect'].to_numpy()
    barcolors=['#ba6734' if c=='ADF' else '#377ba8' if c=='ASH' else '#a8afb5' for c in cells]
    ax.barh(np.arange(13),effects,color=barcolors,height=.7)
    ax.set_yticks(np.arange(13),cells);ax.invert_yaxis();ax.axvline(0,color='.3',lw=.8)
    ax.set(xlabel='Fitted unit-coordinate change over chemical IQR',title='Same chemical axis; all 13 coordinates')
    ax.grid(axis='x',alpha=.16);ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,
               title='Original recorded-date set',fontsize=8,title_fontsize=9)
    fig.suptitle('Coordinate changes and the observed chemical range',fontsize=14,fontweight='bold')
    _save(fig,out,'02_adf_ash_thirds_and_all13_effects')

    fig,axes=plt.subplots(1,3,figsize=(13.2,11.6),constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.15,5.1,5.2]},sharey=True)
    ys=np.arange(29);ax=axes[0];x=data.chemical_state_score.to_numpy()
    ax.barh(ys,x,color=[colors[d] for d in data.dates],height=.72)
    ax.axvline(0,color='.3',lw=.8);ax.set_title('Chemical\nstate',fontsize=11)
    ax.set_yticks(ys,[s+('*' if pd.notna(f) else '') for s,f in zip(data.strain,data.taxonomy_note)],fontsize=9)
    ax.set_ylim(28.5,-.5);ax.set_xlabel('Score');ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',alpha=.15)
    values=data[['unit_'+c for c in cells]].to_numpy();vmax=float(np.max(np.abs(values)))
    im=axes[1].imshow(values,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
    axes[1].set_xticks(np.arange(13),cells,rotation=55,ha='right');axes[1].set_title('Complete neural unit profile',fontsize=11)
    axes[1].tick_params(axis='y',left=False,labelleft=False)
    axes[2].set_xlim(0,1);axes[2].set_title('Recorded species and full date set',loc='left',fontsize=11)
    axes[2].axis('off')
    for i,row in data.iterrows():
        species=row.species.replace('Bacteroides ','B. ')
        axes[2].text(0,i,f'{species}  |  {row.dates}',va='center',fontsize=8.5)
    for ax in axes[:2]:
        for boundary in [9.5,19.5]:ax.axhline(boundary,color='.35',lw=.8)
    cb=fig.colorbar(im,ax=axes[1],location='bottom',shrink=.65,pad=.02)
    cb.set_label('Signed unit coordinate; no coordinate-wise SD scaling',fontsize=9)
    fig.suptitle('All 29 strains in chemical-score order',fontsize=14,fontweight='bold')
    _save(fig,out,'03_chemical_ordered_all13_context')
    return [str(out/f'{name}.png') for name in ['01_chemical_relationship_and_heldout',
            '02_adf_ash_thirds_and_all13_effects','03_chemical_ordered_all13_context']]


if __name__=='__main__':
    for path in plot_saved(Path(__file__).resolve().parents[1]):print(path)
