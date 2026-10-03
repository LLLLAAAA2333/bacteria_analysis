"""Read saved repeat tables and draw three English figures; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'gated':'#367da8','pre_gate':'#c77d3f'}
LABELS={'gated':'SNR-gated','pre_gate':'Same-template pre-gate'}


def _save(fig,out,name):
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def _points(ax,x,values,color):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    jitter=np.linspace(-.035,.035,len(values))
    ax.scatter(x+jitter,values,s=10,color=color,alpha=.28,edgecolors='none')
    lo,med,hi=np.quantile(values,[.05,.5,.95])
    ax.vlines(x,lo,hi,color=color,lw=2)
    ax.plot([x-.075,x+.075],[med,med],color=color,lw=2.4)


def plot_saved(result_dir,out=None):
    result=Path(result_dir).resolve();out=result/'figures' if out is None else Path(out).resolve()
    if out.is_dir() and any(out.iterdir()):raise FileExistsError('Choose a new empty figure directory.')
    out.mkdir(parents=True,exist_ok=True)
    t=result/'tables';metrics=pd.read_csv(t/'animal_split_metrics.csv')
    dates=pd.read_csv(t/'date_profiles.csv',dtype={'date':str});ds=pd.read_csv(t/'date_summary.csv')
    planes=pd.read_csv(t/'animal_half_plane_angles.csv');coverage=pd.read_csv(t/'animal_strain_coverage.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.7),constrained_layout=True)
    names=[('unit_ADF_minus_ASH','ADF − ASH (unit coordinates)'),('unit_AWB','AWB (unit coordinate)'),
           ('fixed_PC12','Positions in the fixed neural plane'),('unit_13D','Complete 13-neuron unit profiles')]
    for ax,(metric,title) in zip(axes.flat,names):
        for j,rep in enumerate(['gated','pre_gate']):
            data=metrics[(metrics.metric==metric)&(metrics.representation==rep)]
            offset=(-.12 if j==0 else .12)
            for x,field in enumerate(['same_RMS','between_RMS']):_points(ax,x+offset,data[field],COLORS[rep])
        ax.set_xticks([0,1],['Same strain','Different strains'])
        ax.set_xlim(-.4,1.4);ax.set_ylim(bottom=0);ax.set_ylabel('RMS difference')
        ax.set_title(title,loc='left',pad=10);ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    handles=[plt.Line2D([0],[0],color=COLORS[r],marker='o',ls='',label=LABELS[r]) for r in COLORS]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Animal splits: repeat scatter and between-strain differences',fontsize=13,fontweight='bold')
    _save(fig,out,'01_animal_repeat_vs_between')

    fig,axes=plt.subplots(2,2,figsize=(11.1,8.0),constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.15,1]})
    for row,(metric,title) in enumerate(names[:2]):
        ax=axes[row,0];g=dates[dates.representation.eq('gated')]
        early=[];late=[]
        for strain,pair in g.groupby('strain'):
            pair=pair.sort_values('date');a,b=pair[metric].to_numpy();early.append(a);late.append(b)
            flag=bool(pair.taxonomy_note.notna().any())
            ax.scatter(a,b,color=COLORS['gated'],s=30)
            offset=(5,5)
            if strain=='A013':offset=(5,-13)
            if strain=='A014':offset=(5,6)
            ax.annotate(strain+('*' if flag else ''),(a,b),xytext=offset,textcoords='offset points',fontsize=8)
        low=min(early+late);high=max(early+late);pad=max(.07,(high-low)*.15)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],ls='--',lw=.9,color='.55')
        ax.set_xlim(low-pad,high+pad);ax.set_ylim(low-pad,high+pad)
        ax.set_xlabel('Earlier recording');ax.set_ylabel('Later recording')
        ax.set_title(title,loc='left',pad=10);ax.spines[['top','right']].set_visible(False)
        ax=axes[row,1]
        for j,rep in enumerate(['gated','pre_gate']):
            data=ds[(ds.metric==metric)&(ds.representation==rep)].iloc[0]
            offset=(-.11 if j==0 else .11)
            vals=[data.same_RMS,data.between_mean_profile_RMS]
            ax.plot(np.array([0,1])+offset,vals,'o-',color=COLORS[rep],lw=1.2,ms=6)
        ax.set_xticks([0,1],['Cross-date change','Between strain means'])
        ax.set_ylim(bottom=0);ax.set_ylabel('RMS difference');ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title('Same six strains; descriptive reference',loc='left',pad=10)
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Six strains recorded on two dates',fontsize=13,fontweight='bold')
    _save(fig,out,'02_cross_date_pairs')

    fig,axes=plt.subplots(1,2,figsize=(12.0,8.2),constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.1,1]})
    ax=axes[0]
    for j,rep in enumerate(['gated','pre_gate']):
        data=planes[planes.representation.eq(rep)];offset=-.12 if j==0 else .12
        for x,col in enumerate(['angle_small_deg','angle_large_deg']):_points(ax,x+offset,data[col],COLORS[rep])
    ax.set_xticks([0,1],['Smaller angle','Larger angle']);ax.set_xlim(-.4,1.4);ax.set_ylim(0,92)
    ax.set_ylabel('Angle between half-sample planes (degrees)')
    ax.set_title('Planes refitted within each animal half',loc='left',pad=12)
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax=axes[1];counts=coverage.groupby('strain').joint_complete13_nonzero.sum().sort_index()
    ax.barh(counts.index,counts.to_numpy(),color=['#397da6' if n else '#bfc7cd' for n in counts],height=.72)
    ax.invert_yaxis();ax.set_xlim(0,112);ax.tick_params(axis='y',length=0,labelsize=8)
    for i,n in enumerate(counts):ax.text(n+1.5,i,str(int(n)),va='center',fontsize=8)
    ax.set_xlabel('Valid split pairings (of 100)');ax.set_title('Complete 13-neuron common support',loc='left',pad=12)
    ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Animal-half plane orientation and observation coverage',fontsize=13,fontweight='bold')
    _save(fig,out,'03_half_planes_and_coverage')
    return out


if __name__=='__main__':plot_saved(Path(__file__).resolve().parents[1])
