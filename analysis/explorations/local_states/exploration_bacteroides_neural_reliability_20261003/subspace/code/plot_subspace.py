"""Save neural-only projection/sensitivity figures from frozen result tables."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OFFSETS = {s: (5, 7, 'left') for s in ['A001','A002','A005','A006','A007','A008','A009','A010','A011','A013','A014','A015','A016','A017','A019','A020','A021','A022','A023','A024','A025','A026','A038','A040','A041','A044','A045','A048','A049']}
OFFSETS.update({'A007':(-5,10,'right'),'A008':(-5,-10,'right'),'A010':(-5,-10,'right'),
                'A013':(-5,9,'right'),'A014':(5,-10,'left'),'A017':(5,-10,'left'),
                'A022':(5,-11,'left'),'A023':(-5,7,'right'),'A038':(5,-4,'left'),
                'A041':(-5,12,'right'),'A044':(5,-9,'left')})
SCHEMES = ['leave_one_strain_out','leave_one_recorded_species_out']
COLORS = ['#417DA3','#C0784E']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def date_label(value):
    return ' + '.join(f'{part[:4]}-{part[4:6]}-{part[6:]}' for part in str(value).split(';'))


def styles(values, species=False):
    names=sorted(values.unique())
    if species:
        palette=plt.get_cmap('tab20').colors
        order=[0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11]
        markers=['o','s','^','D']
        return {g:{'color':palette[order[i]],'marker':markers[i%4]} for i,g in enumerate(names)}
    palette=plt.get_cmap('tab10').colors;markers=['o','s','^','D','v','P','X','<']
    return {g:{'color':palette[i],'marker':markers[i]} for i,g in enumerate(names)}


def plot_projection(ax,legend_ax,data,grouping,title,summary):
    mapping=styles(data[grouping],species=grouping=='species');handles=[]
    for name,style in mapping.items():
        rows=data[data[grouping].eq(name)]
        ax.scatter(rows.PC1,rows.PC2,s=64,marker=style['marker'],color=style['color'],edgecolors='white',linewidths=.7,zorder=3)
        label=name.replace('Bacteroides ','B. ') if grouping=='species' else date_label(name)
        label+=f' (n={len(rows)})'
        handles.append(Line2D([0],[0],marker=style['marker'],color='none',markerfacecolor=style['color'],markeredgecolor='white',markersize=7.5,label=label))
    for row in data.itertuples():
        label=row.strain+('*' if bool(row.taxonomy_flag) else '')
        dx,dy,ha=OFFSETS[row.strain]
        ax.annotate(label,(row.PC1,row.PC2),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=8.3,color='#252525',zorder=4)
    ax.set_xlim(-.71,.56);ax.set_ylim(-.61,.61);ax.set_aspect('equal',adjustable='box')
    ax.set_xticks([-.6,-.4,-.2,0,.2,.4]);ax.set_yticks([-.6,-.4,-.2,0,.2,.4,.6])
    spec=summary['full_gated_spectrum']
    ax.set_xlabel(f"PC1 projection ({100*spec['pc1_evr']:.1f}% of variance)",fontsize=11)
    ax.set_ylabel(f"PC2 projection ({100*spec['pc2_evr']:.1f}% of variance)",fontsize=11)
    ax.set_title(title,loc='left',fontsize=12,pad=10)
    ax.axhline(0,color='#E2E2E2',lw=.7,zorder=0);ax.axvline(0,color='#E2E2E2',lw=.7,zorder=0)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    for side in ['left','bottom']: ax.spines[side].set_color('#999999')
    ax.tick_params(labelsize=9,color='#999999')
    legend_ax.set_axis_off()
    legend_ax.legend(handles=handles,loc='upper left',ncol=2,frameon=False,fontsize=8.8,handletextpad=.4,columnspacing=1.15,labelspacing=.58,borderaxespad=0)
    return mapping


def save_plots(output):
    output=Path(output).resolve();t=output/'tables';f=output/'figures'
    paths={'scores':t/'full_gated_scores_with_metadata.csv','deletions':t/'deletion_subspace_metrics.csv',
           'summary':output/'summary.json','worst':t/'worst_deletion_cases.csv'}
    data=pd.read_csv(paths['scores'],dtype={'strain':str,'dates':str})
    deletions=pd.read_csv(paths['deletions']);summary=json.loads(paths['summary'].read_text())
    worst=pd.read_csv(paths['worst'])
    assert len(data)==29 and len(deletions)==45 and set(data.strain)==set(OFFSETS)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig=plt.figure(figsize=(14.6,10.0))
    grid=fig.add_gridspec(2,2,height_ratios=[5.6,2.45],hspace=.21,wspace=.18,left=.065,right=.99,bottom=.065,top=.91)
    a=fig.add_subplot(grid[0,0]);b=fig.add_subplot(grid[0,1]);la=fig.add_subplot(grid[1,0]);lb=fig.add_subplot(grid[1,1])
    species_styles=plot_projection(a,la,data,'species','A  Recorded species',summary)
    date_styles=plot_projection(b,lb,data,'dates','B  Recorded date set',summary)
    fig.suptitle('Two-dimensional neural structure within Bacteroides',x=.065,ha='left',y=.973,fontsize=16)
    fig.text(.065,.025,'* Source taxonomy-note flag',fontsize=9,color='#555555')
    for ext in ['png','svg']: fig.savefig(f/f'01_neural_pc1_pc2.{ext}',dpi=180)
    plt.close(fig)

    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.8,5.0),gridspec_kw={'width_ratios':[1,1.15]})
    rng=np.random.default_rng(20261003);jitter_records=[]
    for i,scheme in enumerate(SCHEMES):
        block=deletions[deletions.scheme.eq(scheme)].sort_values('fit_id')
        jitter=rng.uniform(-.115,.115,len(block));angle=block.max_principal_angle_deg.to_numpy()
        ax.scatter(angle,i+jitter,s=45,color=COLORS[i],marker='o' if i==0 else '^',edgecolors='white',linewidths=.65,zorder=3)
        q1,median,q3=np.quantile(angle,[.25,.5,.75])
        ax.plot([q1,q3],[i,i],color='#333333',lw=2.5,zorder=2)
        ax.plot([median,median],[i-.18,i+.18],color='#333333',lw=1.4,zorder=4)
        worst_row=block.loc[block.max_principal_angle_deg.idxmax()]
        label=worst_row.omitted_label.replace('Bacteroides ','B. ')
        ax.annotate(label,(worst_row.max_principal_angle_deg,i),xytext=(0,25),textcoords='offset points',ha='center',va='center',fontsize=9)
        for fit_id,jit in zip(block.fit_id,jitter): jitter_records.append({'fit_id':fit_id,'y_jitter':float(jit)})
        if i==0:
            bx.scatter(block.pc1_acute_angle_deg,angle,s=44,color=COLORS[i],edgecolors='white',linewidths=.6,label='Leave one strain',zorder=3)
        else:
            bx.scatter(block.pc1_acute_angle_deg,angle,s=70,facecolors='none',edgecolors=COLORS[i],linewidths=1.2,marker='^',label='Leave recorded species',zorder=4)
    ax.set_yticks([0,1],['Leave one strain\n(n=29)','Leave recorded species\n(n=16)'],fontsize=10)
    ax.set_ylim(-.45,1.52);ax.set_xlim(-.4,14.1)
    ax.set_xticks([0,3,6,9,12]);ax.set_xlabel('Maximum principal angle (degrees)',fontsize=11)
    ax.set_title('A  Change in the top-two plane',loc='left',fontsize=13,pad=13)
    ax.grid(axis='x',color='#E7E7E7',lw=.65,zorder=0);ax.tick_params(axis='y',length=0)
    for row in worst[worst.selected_by.eq('pc1_acute_angle_deg')].itertuples():
        label=row.omitted_label.replace('Bacteroides ','B. ')
        bx.annotate(label,(row.pc1_acute_angle_deg,row.max_principal_angle_deg),xytext=(-4,10),textcoords='offset points',ha='right',va='bottom',fontsize=9)
    bx.plot([0,14],[0,14],color='#A0A0A0',ls='--',lw=.9,zorder=1)
    bx.set_xlim(-2,91);bx.set_ylim(-.25,14.1)
    bx.set_xticks([0,15,30,45,60,75,90]);bx.set_yticks([0,3,6,9,12])
    bx.set_xlabel('Single-PC1 angle (degrees)',fontsize=11)
    bx.set_ylabel('Top-two plane maximum angle (degrees)',fontsize=11)
    bx.set_title('B  Single-axis and plane sensitivity',loc='left',fontsize=13,pad=13)
    bx.legend(loc='upper right',frameon=False,fontsize=9)
    bx.grid(color='#EEEEEE',lw=.6,zorder=0)
    for axis in [ax,bx]:
        for side in ['top','right']: axis.spines[side].set_visible(False)
        for side in ['left','bottom']: axis.spines[side].set_color('#999999')
        axis.tick_params(labelsize=9,color='#999999')
    fig.tight_layout(w_pad=3)
    for ext in ['png','svg']: fig.savefig(f/f'02_subspace_deletion_sensitivity.{ext}',dpi=180)
    plt.close(fig)
    notes='''# Figure captions

## 01 — Two-dimensional neural structure within Bacteroides

Both panels show all 29 strains at identical PC1/PC2 coordinates from centered PCA of the full29 gated 13-coordinate unit profiles; no neuron SD scaling or re-normalization is applied. The two axes have the same geometric scale. Panel A uses recorded species, abbreviated B. for Bacteroides; panel B uses complete recorded date-sets and never assigns multi-date strains to one date. All strain IDs are shown. Asterisks indicate source taxonomy-note flags, not statistical significance. Species/date fields are annotation, not PCA predictors or filters; dates are not assumed chemical/experimental batches. PC coordinates are projections in unit-profile space, not SD or neural response amplitude. PC1 and PC2 explain 33.60% and 31.24% of within-cohort variance, together 64.84%; this is not model explanatory power.

## 02 — Deletion sensitivity of the top-two plane and PC1

Panel A shows every one of 29 leave-one-strain and 16 leave-one-recorded-species maximum principal angles, comparing each training subset's re-centered PCA's top-two span to the full29 top-two span. Jitter affects vertical display only; dark horizontal segments show the middle 50% and vertical ticks the median, not confidence intervals. Labels identify the maximum-angle omission in each scheme: A015 and recorded B. fluxus. Panel B shows the same 45 fits, comparing the acute single-PC1 angle with maximum top-two principal angle; open triangles allow coincident species/strain fits to remain distinguishable. The dashed line marks equal angles. Labels identify the largest PC1-angle cases in each scheme: A048 and recorded B. salyersiae. Some points coincide because omitting a singleton species equals omitting its only strain.

All comparisons use an overlapping full29 reference that contains each training subset. They measure sensitivity to these prescribed deletions, not independent experimental repeatability. Rotations or swaps inside the two-dimensional span can move PC1 substantially without comparably changing that span. No stable/unstable threshold is drawn and K=2 was fixed before this analysis. The separate all29 pre-gate comparison is in the tables/README; it does not select the displayed dimensionality.
'''
    (f/'figure_notes.md').write_text(notes)
    generated=['01_neural_pc1_pc2.png','01_neural_pc1_pc2.svg','02_subspace_deletion_sensitivity.png','02_subspace_deletion_sensitivity.svg','figure_notes.md']
    params={'inputs':{name:{'path':str(path),'sha256':digest(path)} for name,path in paths.items()},
            'code_sha256':digest(__file__),'dpi':180,'species_styles':species_styles,'date_styles':date_styles,
            'projection_offsets_points':OFFSETS,'projection_limits':{'x':[-.71,.56],'y':[-.61,.61]},
            'projection_aspect':'equal','jitter_seed':20261003,'angle_plot_jitter':jitter_records,
            'extreme_example_rule':'maximum plane angle or maximum PC1 angle separately within each deletion scheme',
            'output_sha256':{name:digest(f/name) for name in generated}}
    (output/'plot_parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    return [f/name for name in generated]


if __name__=='__main__': save_plots(Path(__file__).resolve().parents[1])
