"""Supporting context checks, with fixed original 380-feature pair categories."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_context(out):
    out = Path(out)
    t, dest = out/'tables', out/'figures'
    dest.mkdir(exist_ok=True)
    order = ['Cnear_Nnear','Cnear_Nfar','Cfar_Nnear','Cfar_Nfar']
    labels = ['Chemical near · Neural near','Chemical near · Neural far',
              'Chemical far · Neural near','Chemical far · Neural far']
    quality = pd.read_csv(t/'chemical_distance_quality.csv').pivot(
        index='category',columns='feature_scope',values='mean_distance_share_pct').loc[order]
    support = pd.read_csv(t/'category_summary.csv').set_index('category').loc[order]
    features = pd.read_csv(t/'selected_features.csv').feature.tolist()
    f = pd.read_csv(t/'feature_summary.csv')
    abs_delta = f.pivot(index='feature',columns='category',values='mean_abs_log2fc_difference').loc[features,order]
    ctx = pd.read_csv(t/'feature_context_checks.csv')
    near = ctx[ctx.chemical_band.eq('near') & ctx.reference_scope.ne('leave_one_sample_range')]
    scopes = ['all','same_reference','A250','A306','ref12']
    ratios = near.pivot(index='feature',columns='reference_scope',values='log2_mean_absolute_difference_ratio').loc[features,scopes]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(15,11),layout='constrained')
    gs = fig.add_gridspec(2,2,height_ratios=[1,2.3],wspace=.13,hspace=.15)
    ax=fig.add_subplot(gs[0,0]); left=np.zeros(4)
    for col,label,color in [('both_reported','Both reported','#3e8989'),
                            ('one_sided_report_missing','One missing','#d98d47'),
                            ('both_report_missing','Both missing','#85858c')]:
        vals=quality[col].to_numpy();ax.barh(np.arange(4),vals,left=left,color=color,label=label)
        for k,v in enumerate(vals):
            if v>=5: ax.text(left[k]+v/2,k,f'{v:.0f}%',ha='center',va='center',fontsize=8,color='white' if col=='both_reported' else '#222')
        left+=vals
    ax.set(yticks=np.arange(4),yticklabels=labels,xlim=(0,100),xlabel='Mean share of squared chemical distance (%)')
    ax.invert_yaxis();ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.25),fontsize=8)
    ax.set_title('A   Original-report coverage',loc='left',fontweight='bold')
    ax=fig.add_subplot(gs[0,1]);vals=support.same_reference_fraction.to_numpy()*100
    ax.barh(np.arange(4),vals,color='#718da0');ax.set(yticks=np.arange(4),yticklabels=labels,xlim=(0,108),xlabel='Pairs sharing a chemical reference (%)')
    for k,v in enumerate(vals):ax.text(v+1.5,k,f'{v:.1f}%',va='center')
    ax.invert_yaxis();ax.set_title('B   Chemical reference composition',loc='left',fontweight='bold')
    ax=fig.add_subplot(gs[1,0]);im=ax.imshow(abs_delta,aspect='auto',cmap='YlOrBr',vmin=0,vmax=float(abs_delta.to_numpy().max()))
    ax.set(yticks=np.arange(len(features)),yticklabels=features,
           xticks=np.arange(4),xticklabels=['C near\nN near','C near\nN far','C far\nN near','C far\nN far'])
    ax.set_title('C   Complete-report feature differences',loc='left',fontweight='bold')
    for (i,j),v in np.ndenumerate(abs_delta.to_numpy()):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if v>abs_delta.to_numpy().max()*.6 else '#222')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.025,label='Mean |Δlog₂FC|')
    ax=fig.add_subplot(gs[1,1]);limit=float(np.nanmax(np.abs(ratios)))
    im=ax.imshow(ratios,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    ax.set(yticks=np.arange(len(features)),yticklabels=features,xticks=np.arange(5),
           xticklabels=['All\n545 / 221','Same ref.\n461 / 201','A250\n33 / 14','A306\n70 / 71','ref12\n123 / 115'])
    ax.set_title('D   Chemical-near: neural-far / neural-near',loc='left',fontweight='bold')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.025,label='log₂ ratio of mean |Δlog₂FC|')
    for ext in ['png','svg']:
        fig.savefig(dest/f'04_context_checks.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (dest/'context_caption.txt').write_text(
        'Supporting context checks. All panels preserve the original full-380-feature distance categories. '
        'A: per-pair squared-distance shares, averaged over pairs; missing means original report NaN, not confirmed chemical absence or below-LOD status. '
        'B: shared numerical reference group; nearly all chemical-far pairs cross reference groups. '
        'C: absolute log2FC differences for the same 15 outcome-selected complete-report/QC RSD <= 0.30 features highlighted in Figure 02. '
        'D: the ratio of mean absolute feature differences in neural-far versus neural-near pairs, restricted to chemical-near pairs; positive values mean larger differences in neural-far pairs. '
        'Labels give neural-near / neural-far pair counts. A050-only has 235 / 1 pairs and is omitted from this heatmap because of its single far-neural pair; its values remain in the tables. '
        'Reference groups are numerical normalization groups, not biological labels. Pairs share samples; no independent-sample inference, confidence intervals, or causal interpretation is intended. '
        'Named features remain report annotations.\n')


if __name__ == '__main__':
    plot_context(Path(__file__).resolve().parents[1])
