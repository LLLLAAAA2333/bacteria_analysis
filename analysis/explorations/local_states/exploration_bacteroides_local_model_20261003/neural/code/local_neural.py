"""Neural-only PCA within Bacteroides. Saved results are read-only by default."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REL = Path('reports/exploration_bacteroides_local_model_20261003/neural')
SRC = Path('reports/exploration_chemical_pattern_direct_report_20261003/tables')
EPS = 1e-12


def fit_pca(values):
    """Column-mean-centered SVD; no SD scaling. Rows=strains, columns=neurons."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    centered = values - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])
    signs[signs == 0] = 1
    loadings = (vt * signs[:, None]).T
    scores = centered @ loadings
    variance = singular ** 2 / (len(values) - 1)
    ratio = variance / variance.sum()
    return dict(mean=mean, centered=centered, loadings=loadings, scores=scores,
                singular_values=singular, variance=variance, ratio=ratio)


def stability_checks(values, metadata, full):
    """Fixed leave-strain / leave-recorded-species refits; no chemical input."""
    rows, heldout, loadings = [], [], []
    tests = [('strain', s, [s]) for s in values.index]
    tests += [('species', species, metadata.index[metadata.species.eq(species)].tolist())
              for species in sorted(metadata.species.unique())]
    direction = full['loadings'][:, 0]
    for kind, name, omitted in tests:
        train = values.drop(index=omitted)
        fit = fit_pca(train.to_numpy())
        raw_dot = float(direction @ fit['loadings'][:, 0])
        sign = 1 if raw_dot >= 0 else -1
        aligned = fit['loadings'][:, 0] * sign
        identifier = f'{kind}:{name}'
        rows.append(dict(holdout_id=identifier, holdout_type=kind, omitted_label=name,
                         omitted_strains=';'.join(omitted), n_omitted=len(omitted),
                         train_n=len(train), pc1_absolute_loading_cosine=abs(raw_dot),
                         pc1_variance_ratio=fit['ratio'][0], pc2_variance_ratio=fit['ratio'][1],
                         pc1_minus_pc2_variance_ratio=fit['ratio'][0]-fit['ratio'][1],
                         pc1_to_pc2_eigenvalue_ratio=fit['variance'][0]/fit['variance'][1],
                         aligned_to_full_pc1=True))
        loadings.append(dict(holdout_id=identifier, **dict(zip(values.columns, aligned))))
        for strain in omitted:
            index = values.index.get_loc(strain)
            score = float((values.loc[strain].to_numpy() - fit['mean']) @ aligned)
            heldout.append(dict(holdout_id=identifier, holdout_type=kind, strain=strain,
                                training_centered_pc1_score=score,
                                full_centered_pc1_score=float(full['scores'][index,0]),
                                score_difference=score-full['scores'][index,0]))
    return pd.DataFrame(rows), pd.DataFrame(heldout), pd.DataFrame(loadings)


def savefig(fig, output, name):
    for ext in ['png', 'svg']:
        fig.savefig(output / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def draw_figures(result_dir, figure_dir):
    """Read saved neural results only. Figure output must be empty or new."""
    source, output = Path(result_dir).resolve(), Path(figure_dir).resolve()
    if output.is_dir() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite saved figures: {output}')
    output.mkdir(parents=True, exist_ok=True)
    load = pd.read_csv(source/'tables/neural_loadings.csv', index_col='neuron')
    spectrum = pd.read_csv(source/'tables/pca_spectrum.csv')
    strains = pd.read_csv(source/'tables/strain_scores.csv', index_col='strain', dtype={'dates':str})
    centered = pd.read_csv(source/'tables/neural_centered_profiles.csv', index_col='strain')
    params = json.loads((source/'parameters.json').read_text())
    neurons = load.index.tolist()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1,2,figsize=(11.6,5.6),constrained_layout=True,
                             gridspec_kw={'width_ratios':[1,1.1]})
    locations=np.arange(len(neurons))
    axes[0].barh(locations-.18, load.PC1, color='#397da6', height=.34, label='PC1')
    axes[0].barh(locations+.18, load.PC2, color='#c78344', height=.34, label='PC2')
    axes[0].set_yticks(locations,neurons)
    axes[0].legend(frameon=False,loc='lower left')
    axes[0].invert_yaxis();axes[0].axvline(0,color='.35',lw=.8)
    maximum=np.ceil(np.abs(load[['PC1','PC2']]).to_numpy().max()*10)/10
    axes[0].set_xlim(-maximum,maximum)
    axes[0].set_xlabel('Signed loading')
    axes[0].set_title('Two leading neural directions',loc='left',pad=12)
    axes[1].bar(np.arange(1,14),100*spectrum.variance_ratio,
                color=['#397da6']+['#b8c5cc']*12,label='Per component')
    axes[1].plot(np.arange(1,14),100*spectrum.cumulative_variance_ratio,
                 'o-',color='#414141',ms=4,lw=1.2,label='Cumulative')
    axes[1].set_xticks(np.arange(1,14));axes[1].set_ylim(0,104)
    axes[1].set_xlabel('Neural principal component')
    axes[1].set_ylabel('Within-genus variance (%)')
    axes[1].set_title('Variance in the 29 unit profiles',loc='left',pad=12)
    axes[1].legend(frameon=False,loc='center right')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.14)
        ax.set_axisbelow(True)
    fig.suptitle(f'Bacteroides neural variation: PC1 {100*spectrum.variance_ratio.iloc[0]:.1f}%; PC2 {100*spectrum.variance_ratio.iloc[1]:.1f}%',
                 fontsize=14,fontweight='bold')
    savefig(fig,output,'01_neural_pc1_and_variance')

    order=params['pc1_strain_order'];meta=strains.loc[order];matrix=centered.loc[order]
    limit=np.ceil(np.abs(matrix.to_numpy()).max()*10)/10
    fig=plt.figure(figsize=(15.1,10.2),constrained_layout=True)
    grid=fig.add_gridspec(1,3,width_ratios=[8.2,4.9,.3])
    ax=fig.add_subplot(grid[0,0]);ann=fig.add_subplot(grid[0,1]);cbax=fig.add_subplot(grid[0,2])
    im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,
                  interpolation='nearest')
    labels=[s+('*' if bool(meta.loc[s,'taxonomy_flag']) else '') for s in order]
    ax.set_yticks(range(29),labels);ax.tick_params(axis='both',length=0)
    ax.set_xticks(range(13),neurons,rotation=45,ha='right')
    ax.set_title('Neural profiles ordered by increasing PC1',loc='left',pad=14,fontweight='bold')
    for spine in ax.spines.values():spine.set_visible(False)
    ann.set_xlim(0,1);ann.set_ylim(28.5,-.5);ann.axis('off')
    ann.text(.02,-1.25,'Species',fontsize=10,fontweight='bold')
    ann.text(.62,-1.25,'Dates (2026)',fontsize=10,fontweight='bold')
    for i,(_,row) in enumerate(meta.iterrows()):
        dates=' / '.join(d[4:6]+'-'+d[6:8] for d in row.dates.split(';'))
        ann.text(.02,i,row.species.replace('Bacteroides ', 'B. '),va='center',fontsize=9)
        ann.text(.62,i,dates,va='center',fontsize=9)
    bar=fig.colorbar(im,cax=cbax);bar.set_label('Unit coefficient − Bacteroides mean')
    fig.supxlabel('* Taxonomy note retained',fontsize=9)
    savefig(fig,output,'02_neural_strains_by_pc1')
    return {'centered_heatmap_color_limit':float(limit)}


def load_saved_results(result_dir):
    """Read core saved results for an existing Notebook; no computation or writes."""
    source=Path(result_dir).resolve()
    return {
        'parameters':json.loads((source/'parameters.json').read_text()),
        'spectrum':pd.read_csv(source/'tables/pca_spectrum.csv'),
        'loadings':pd.read_csv(source/'tables/neural_loadings.csv',index_col='neuron'),
        'strain_scores':pd.read_csv(source/'tables/strain_scores.csv',index_col='strain',dtype={'dates':str}),
        'stability':pd.read_csv(source/'tables/pc1_stability.csv'),
        'pre_gate_summary':pd.read_csv(source/'tables/pre_gate_sensitivity_summary.csv'),
    }


def run_analysis(repo_root, out=None):
    """Compute fixed neural analysis into a fresh directory; no overwrite permitted."""
    root=Path(repo_root).resolve()
    output=root/REL if out is None else Path(out)
    output=(root/output).resolve() if not output.is_absolute() else output.resolve()
    if ((output/'tables').is_dir() and any((output/'tables').iterdir())) or any(
            (output/name).exists() for name in ['parameters.json','source_manifest.json']):
        raise FileExistsError(f'Saved scientific results already exist: {output}; choose a fresh out directory.')
    paths={
        'neural_unit':root/SRC/'neural_unit_coefficients.csv',
        'same_template_pre_gate_unit':root/SRC/'neural_pre_gate_unit_coefficients.csv',
        'sample_context':root/SRC/'sample_context.csv',
        'gated_coefficients':root/'reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv',
        'saved_gain_context':root/'reports/exploration_matched_pair_context_20261003/tables/sample_context.csv',
    }
    sources={k:{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for k,p in paths.items()}
    context=pd.read_csv(paths['sample_context'],index_col='strain',dtype={'dates':str})
    metadata=context.loc[context.genus.eq('Bacteroides')].sort_index().copy()
    assert len(metadata)==29 and metadata.species.nunique()==16 and metadata.index.is_unique
    x=pd.read_csv(paths['neural_unit'],index_col='strain').loc[metadata.index]
    pre=pd.read_csv(paths['same_template_pre_gate_unit'],index_col='strain').loc[x.index,x.columns]
    coeff=pd.read_csv(paths['gated_coefficients'],index_col='strain').loc[x.index,x.columns]
    saved_gain=pd.read_csv(paths['saved_gain_context'],usecols=['strain','coefficient_norm'],index_col='strain').loc[x.index,'coefficient_norm']
    norms=np.linalg.norm(coeff.to_numpy(),axis=1)
    assert x.shape==(29,13) and np.isfinite(x.to_numpy()).all() and np.isfinite(pre.to_numpy()).all()
    assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-10)
    assert np.allclose(np.linalg.norm(pre,axis=1),1,atol=1e-10)
    assert np.allclose(norms,saved_gain,atol=1e-10)
    assert np.allclose(coeff.to_numpy()/norms[:,None],x,atol=1e-10)
    species_names=sorted(metadata.species.unique())
    species_codes={name:f'S{i+1:02d}' for i,name in enumerate(species_names)}
    metadata['species_code']=metadata.species.map(species_codes)
    metadata['taxonomy_flag']=metadata.taxonomy_note.fillna('').ne('')
    metadata['n_dates']=metadata.dates.str.split(';').map(len)
    metadata['coefficient_norm']=norms
    main=fit_pca(x.to_numpy());prefit=fit_pca(pre.to_numpy())
    components=[f'PC{i+1}' for i in range(13)]
    scores=pd.DataFrame(main['scores'],index=x.index,columns=components)
    loadings=pd.DataFrame(main['loadings'],index=pd.Index(x.columns,name='neuron'),columns=components)
    spectrum=pd.DataFrame({'component':components,'singular_value':main['singular_values'],
                           'sample_variance':main['variance'],'variance_ratio':main['ratio'],
                           'cumulative_variance_ratio':np.cumsum(main['ratio'])})
    reconstruction=main['mean']+main['scores'][:,[0]]@main['loadings'][:,[0]].T
    residual=x.to_numpy()-reconstruction
    strain_scores=metadata.join(scores)
    strain_scores['pc1_residual_l2']=np.linalg.norm(residual,axis=1)
    strain_scores['pc1_residual_rms']=np.sqrt(np.mean(residual**2,axis=1))
    strain_scores['centered_profile_l2']=np.linalg.norm(main['centered'],axis=1)
    stability,heldout,stability_loadings=stability_checks(x,metadata,main)
    pc1_dot=float(main['loadings'][:,0]@prefit['loadings'][:,0])
    pre_sign=1 if pc1_dot>=0 else -1
    aligned_pre_loading=prefit['loadings'][:,0]*pre_sign
    aligned_pre_scores=prefit['scores'][:,0]*pre_sign
    pre_scores=pd.DataFrame({'PC1_gated':scores.PC1,'PC1_pre_gate_aligned':aligned_pre_scores,
                             'pre_gate_minus_gated_score':aligned_pre_scores-scores.PC1},index=x.index)
    pre_summary={'pc1_absolute_loading_cosine':abs(pc1_dot),
                 'aligned_pc1_score_pearson':float(np.corrcoef(scores.PC1,aligned_pre_scores)[0,1]),
                 'gated_pc1_variance_ratio':float(main['ratio'][0]),
                 'pre_gate_pc1_variance_ratio':float(prefit['ratio'][0]),
                 'score_rmse':float(np.sqrt(np.mean((aligned_pre_scores-scores.PC1)**2))),
                 'same_template':True,'refit_raw_templates':False}
    for dirname in ['tables','verification']:(output/dirname).mkdir(parents=True,exist_ok=True)
    table=output/'tables'
    metadata.to_csv(table/'strain_metadata.csv')
    strain_scores.to_csv(table/'strain_scores.csv')
    loadings.to_csv(table/'neural_loadings.csv')
    spectrum.to_csv(table/'pca_spectrum.csv',index=False)
    x.to_csv(table/'neural_unit_profiles.csv')
    pd.DataFrame(main['centered'],index=x.index,columns=x.columns).to_csv(table/'neural_centered_profiles.csv')
    pd.DataFrame({'mean_unit':main['mean'],'pre_gate_mean_unit':prefit['mean']},index=pd.Index(x.columns,name='neuron')).to_csv(table/'mean_unit_profile.csv')
    pd.DataFrame(reconstruction,index=x.index,columns=x.columns).to_csv(table/'pc1_reconstructed_profiles.csv')
    pd.DataFrame(residual,index=x.index,columns=x.columns).to_csv(table/'pc1_residual_profiles.csv')
    coeff.to_csv(table/'gated_coefficients_descriptor.csv')
    stability.to_csv(table/'pc1_stability.csv',index=False)
    heldout.to_csv(table/'heldout_pc1_projections.csv',index=False)
    stability_loadings.to_csv(table/'stability_aligned_loadings.csv',index=False)
    pre.to_csv(table/'pre_gate_unit_profiles.csv')
    pd.DataFrame({'component':components,'singular_value':prefit['singular_values'],
                  'sample_variance':prefit['variance'],'variance_ratio':prefit['ratio'],
                  'cumulative_variance_ratio':np.cumsum(prefit['ratio'])}).to_csv(table/'pre_gate_pca_spectrum.csv',index=False)
    pd.DataFrame({'PC1_gated':main['loadings'][:,0],'PC1_pre_gate_aligned':aligned_pre_loading},
                 index=pd.Index(x.columns,name='neuron')).to_csv(table/'pre_gate_pc1_loadings.csv')
    pre_scores.to_csv(table/'pre_gate_pc1_scores.csv')
    pd.DataFrame([pre_summary]).to_csv(table/'pre_gate_sensitivity_summary.csv',index=False)
    mapping=[]
    for species in species_names:
        group=metadata.loc[metadata.species.eq(species)]
        dates=sorted(set(d for ds in group.dates for d in ds.split(';')))
        mapping.append(dict(species_code=species_codes[species],species=species,n_strains=len(group),
                            strains=';'.join(group.index),dates=';'.join(dates),
                            n_taxonomy_flagged=int(group.taxonomy_flag.sum())))
    pd.DataFrame(mapping).to_csv(table/'species_mapping.csv',index=False)
    date_members=[]
    for strain,row in metadata.iterrows():
        for date in row.dates.split(';'):
            date_members.append(dict(strain=strain,date=date,species=row.species,
                                     species_code=row.species_code,PC1=float(scores.loc[strain,'PC1'])))
    pd.DataFrame(date_members).to_csv(table/'strain_date_membership.csv',index=False)
    order=scores.PC1.sort_values(kind='stable').index.tolist()
    params={'analysis_date':'2026-10-03','genus':'Bacteroides','n_strains':29,'n_neurons':13,
            'n_species_labels':16,'n_record_dates':len(set(d for ds in metadata.dates for d in ds.split(';'))),
            'primary_direction':'PC1 only, fixed before chemical review',
            'centering':'within-Bacteroides neuron means','column_sd_scaling':False,
            'centered_rows_renormalized':False,'reconstruction_renormalized':False,
            'orientation':'largest-absolute loading positive; first input neuron breaks ties',
            'neuron_order':x.columns.tolist(),'input_strain_order':x.index.tolist(),
            'pc1_strain_order':order,'n_flagged_strains':int(metadata.taxonomy_flag.sum()),
            'strains_with_taxonomy_note':metadata.index[metadata.taxonomy_flag].tolist(),
            'input_norm_max_abs_error':float(np.max(np.abs(norms-saved_gain))),
            'pc1_variance_ratio':float(main['ratio'][0]),'pc2_variance_ratio':float(main['ratio'][1]),
            'pc1_pc2_cumulative_variance_ratio':float(main['ratio'][:2].sum()),
            'rank_tolerance':EPS,'centered_matrix_rank':int(np.sum(main['singular_values']>EPS)),
            'no_chemical_input':True,'no_raw_template_refit':True,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,
                        'pandas':pd.__version__,'matplotlib':matplotlib.__version__}}
    (output/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    (output/'parameters.json').write_text(json.dumps(params,indent=2)+'\n')
    plot_params=draw_figures(output,output/'figures')
    (output/'plot_parameters.json').write_text(json.dumps(plot_params,indent=2)+'\n')
    return {'output':output,'spectrum':spectrum,'loadings':loadings,
            'strain_scores':strain_scores,'stability':stability,'pre_gate_summary':pre_summary}


if __name__=='__main__':
    result=run_analysis(Path(__file__).resolve().parents[4])
    print(result['spectrum'].to_string(index=False))
    print(result['loadings']['PC1'].to_string())
    print(result['stability'].groupby('holdout_type').pc1_absolute_loading_cosine.agg(['min','median','max']).to_string())
    print(result['pre_gate_summary'])
