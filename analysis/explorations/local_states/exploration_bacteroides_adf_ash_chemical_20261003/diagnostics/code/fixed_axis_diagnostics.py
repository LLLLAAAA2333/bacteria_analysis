"""Read the selected model outputs; describe one frozen chemical axis.

No chemical grouping, state selection, or held-out model fitting occurs here.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REL = Path('reports/exploration_bacteroides_adf_ash_chemical_20261003')
PRIMARY = 'primary_unit_adf_minus_ash'
SENSITIVITY = ['raw_adf_minus_ash', 'pre_gate_unit_adf_minus_ash']
TIERS = ['LOW', 'MID', 'HIGH']


def describe_xy(x, y, reference_iqr=None):
    """Descriptive OLS/correlation, no p-values, intervals, or model selection."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert x.ndim == y.ndim == 1 and len(x) == len(y)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    xc, yc = x-x.mean(), y-y.mean()
    sx, sy = np.dot(xc, xc), np.dot(yc, yc)
    slope = float(np.dot(xc, yc)/sx) if sx > 1e-20 else np.nan
    q25, q75 = np.quantile(x, [.25, .75])
    iqr = q75-q25
    pearson = float(np.dot(xc, yc)/np.sqrt(sx*sy)) if sx*sy > 1e-20 else np.nan
    spearman = float(spearmanr(x, y).statistic) if sx*sy > 1e-20 else np.nan
    return {'n_strains':len(x), 'pearson':pearson, 'spearman':spearman,
            'slope':slope, 'intercept':float(y.mean()-slope*x.mean()),
            'x_min':float(x.min()), 'x_max':float(x.max()),
            'x_q25':float(q25), 'x_q75':float(q75), 'x_iqr':float(iqr),
            'fitted_observed_iqr_effect':float(slope*iqr),
            'fitted_full_iqr_effect':float(slope*(iqr if reference_iqr is None else reference_iqr)),
            'y_mean':float(y.mean()), 'y_sd':float(y.std(ddof=1))}


def rank_thirds(frame):
    """Rank-only bins conditional on a response-selected x; exactly 10/10/9."""
    data=frame.reset_index().sort_values(['chemical_state_score','strain'],kind='stable').reset_index(drop=True)
    assert len(data)==29
    data['chemical_rank']=np.arange(1,30)
    data['chemical_third']=pd.Categorical(['LOW']*10+['MID']*10+['HIGH']*9,TIERS,ordered=True)
    return data


def demean_groups(frame, group_field, response=PRIMARY):
    """Retain repeated label groups and pool centered observations descriptively."""
    counts=frame.groupby(group_field).size()
    labels=counts[counts>=2].index
    sub=frame[frame[group_field].isin(labels)].copy()
    sub['x_group_mean']=sub.groupby(group_field).chemical_state_score.transform('mean')
    sub['y_group_mean']=sub.groupby(group_field)[response].transform('mean')
    sub['x_demeaned']=sub.chemical_state_score-sub.x_group_mean
    sub['y_demeaned']=sub[response]-sub.y_group_mean
    sub['n_in_group']=sub.groupby(group_field)[response].transform('size')
    stats=describe_xy(sub.x_demeaned,sub.y_demeaned)
    stats.update(group_field=group_field,n_groups=len(labels),
                 n_singleton_groups_omitted=int(counts.eq(1).sum()),
                 within_centering_df=int(len(sub)-len(labels)))
    grouped=[]
    for label,g in sub.groupby(group_field):
        grouped.append({'group_field':group_field,'group_label':label,'n_strains':len(g),
                        'strains':';'.join(g.index),'x_min':g.chemical_state_score.min(),
                        'x_max':g.chemical_state_score.max(),'y_min':g[response].min(),'y_max':g[response].max()})
    return stats,sub.reset_index(),pd.DataFrame(grouped)


def _clean_json(value):
    if isinstance(value,dict):return {k:_clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [_clean_json(v) for v in value]
    if isinstance(value,(np.integer,np.floating)):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def run_diagnostics(repo_root, out=None):
    """Read fitted outputs and save diagnostics to a fresh directory."""
    root=Path(repo_root).resolve()
    result=root/REL/'diagnostics' if out is None else Path(out)
    if not result.is_absolute():result=(root/result).resolve()
    if (result/'summary.json').exists() or ((result/'tables').is_dir() and any((result/'tables').iterdir())):
        raise FileExistsError(f'Existing diagnostic results: {result}; use a fresh directory.')
    model=root/REL/'model'
    paths={name:model/'tables'/f'{name}.csv' for name in ['selected_full_state_scores',
        'selected_full_state_members','cohort_targets','heldout_predictions','pooled_performance']}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    manifest={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                 'bytes':p.stat().st_size} for k,p in paths.items()}
    data=pd.read_csv(paths['selected_full_state_scores'],index_col='strain',dtype={'dates':str})
    cohort=pd.read_csv(paths['cohort_targets'],index_col='strain',dtype={'dates':str})
    members=pd.read_csv(paths['selected_full_state_members'])
    heldout=pd.read_csv(paths['heldout_predictions'],dtype={'dates':str})
    performance=pd.read_csv(paths['pooled_performance'])
    assert len(data)==29 and data.index.is_unique and data.index.equals(cohort.index)
    for col in cohort.columns:
        if col in data:
            pd.testing.assert_series_equal(data[col],cohort[col],check_names=False)
    assert data.genus.eq('Bacteroides').all() and data.species.nunique()==16
    cells=[col[5:] for col in data if col.startswith('unit_')]
    assert len(cells)==13 and {'ADF','ASH'}.issubset(cells)
    targets=[PRIMARY,*SENSITIVITY,*['unit_'+cell for cell in cells]]
    assert np.isfinite(data[['chemical_state_score',*targets]].to_numpy()).all()
    assert data.selected_state_id.nunique()==1
    assert np.allclose(data[PRIMARY],data.unit_ADF-data.unit_ASH)
    assert np.allclose(data.raw_adf_minus_ash,data.raw_ADF-data.raw_ASH)
    assert np.allclose(data[PRIMARY]*data.coefficient_l2_norm,data.raw_adf_minus_ash)
    assert len(heldout)==58 and heldout.groupby('scheme').strain.nunique().eq(29).all()
    assert not heldout.duplicated(['scheme','strain']).any()
    ordered=rank_thirds(data)
    q25,q75=np.quantile(data.chemical_state_score,[.25,.75]);full_iqr=float(q75-q25)
    associations=pd.DataFrame([{'response':t,**describe_xy(data.chemical_state_score,data[t])} for t in targets])
    fullstats=associations[associations.response.eq(PRIMARY)].iloc[0].to_dict()
    expected=fullstats['intercept']+fullstats['slope']*data.chemical_state_score
    prediction_error=float(np.max(np.abs(expected-data['pred_'+PRIMARY])))
    assert prediction_error<1e-10
    group_stats=[];composition=[]
    for tier,g in ordered.groupby('chemical_third',observed=True):
        for col in ['chemical_state_score',*targets]:
            group_stats.append({'chemical_third':tier,'variable':col,'n_strains':len(g),
                'mean':g[col].mean(),'median':g[col].median(),'min':g[col].min(),'max':g[col].max(),
                'sd':g[col].std(ddof=1),'n_species':g.species.nunique(),'n_original_date_sets':g.dates.nunique(),
                'n_taxonomy_flags':int(g.taxonomy_note.notna().sum())})
        for field in ['species','dates']:
            for label,h in g.groupby(field):
                composition.append({'chemical_third':tier,'label_field':field,'label':label,
                                    'n_strains':len(h),'strains':';'.join(h.strain)})
    influence=[]
    for mode,groups in [('one_strain',[(s,[s]) for s in data.index]),
                        ('one_species',[(s,g.index.tolist()) for s,g in data.groupby('species')])]:
        for label,ids in groups:
            g=data.drop(index=ids)
            stats=describe_xy(g.chemical_state_score,g[PRIMARY],reference_iqr=full_iqr)
            influence.append({'deletion':mode,'deleted_label':label,'deleted_strains':';'.join(ids),
                              'n_deleted':len(ids),**stats,
                              'slope_change_from_full':stats['slope']-fullstats['slope']})
    influence=pd.DataFrame(influence)
    x=data.chemical_state_score.to_numpy();xc=x-x.mean()
    leverage=data[['chemical_state_score',PRIMARY,'species','dates','taxonomy_note']].copy()
    leverage['OLS_leverage']=1/len(data)+xc**2/np.sum(xc**2)
    leverage['squared_x_deviation_share']=xc**2/np.sum(xc**2)
    leverage['residual']=data[PRIMARY]-data['pred_'+PRIMARY]
    demean=[];demean_individual=[];demean_group_rows=[]
    for field in ['species','dates']:
        stats,individual,groups=demean_groups(data,field)
        demean.append(stats);individual['group_field']=field
        demean_individual.append(individual);demean_group_rows.append(groups)
    composition_rows=[]
    assert 'metabolite' in members and 'family' in members
    for field in ['SuperClass','Class','SubClass','column']:
        if field in members:
            for label,g in members.fillna({field:'Unannotated'}).groupby(field):
                row={'annotation_level':field,'annotation':label,'n_members':len(g),
                     'n_families':g.family.nunique(),'members':';'.join(g.metabolite)}
                if 'score_weight' in g:row['summed_score_weight']=g.score_weight.sum()
                composition_rows.append(row)
    ordered_unit=ordered[['strain','chemical_rank','chemical_third','chemical_state_score',
                         'species','dates','taxonomy_note',*['unit_'+c for c in cells]]]
    chosen_id=str(data.selected_state_id.iloc[0])
    summary={'selected_state_id':chosen_id,'n_strains':29,'n_species':16,
        'n_original_date_sets':int(data.dates.nunique()),'n_taxonomy_flags':int(data.taxonomy_note.notna().sum()),
        'member_count':len(members),'Mass_column_family_count':int(members.family.nunique()),
        'neural_coordinate_order':cells,'primary_association':fullstats,
        'sensitivity_associations':associations[associations.response.isin(SENSITIVITY)].to_dict('records'),
        'ADF_and_ASH':associations[associations.response.isin(['unit_ADF','unit_ASH'])].to_dict('records'),
        'third_sizes':ordered.chemical_third.value_counts(sort=False).astype(int).to_dict(),
        'within_group_descriptions':demean,'influence_ranges':[],
        'highest_leverage_strains':leverage.sort_values('OLS_leverage',ascending=False).head(5).reset_index().to_dict('records'),
        'source_model_prediction_reconstruction_max_abs_error':prediction_error,
        'inference_scope':'Exploratory, fixed full-cohort state. No causal/dose interpretation, no significance tests, no fold-based confidence intervals.',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    for mode,g in influence.groupby('deletion'):
        summary['influence_ranges'].append({'deletion':mode,'n_deletions':len(g),
            'pearson_min':g.pearson.min(),'pearson_max':g.pearson.max(),
            'spearman_min':g.spearman.min(),'spearman_max':g.spearman.max(),
            'slope_min':g.slope.min(),'slope_max':g.slope.max(),
            'full_iqr_effect_min':g.fitted_full_iqr_effect.min(),'full_iqr_effect_max':g.fitted_full_iqr_effect.max()})
    for folder in ['tables','verification']:(result/folder).mkdir(parents=True,exist_ok=True)
    tables={'ordered_strains_and_thirds':ordered,'fixed_axis_associations':associations,
        'rank_third_summary':pd.DataFrame(group_stats),'rank_third_species_date_composition':pd.DataFrame(composition),
        'fixed_axis_deletion_influence':influence,'within_group_demeaned_associations':pd.DataFrame(demean),
        'within_group_individual_residuals':pd.concat(demean_individual,ignore_index=True),
        'within_group_support':pd.concat(demean_group_rows,ignore_index=True),
        'selected_state_annotation_composition':pd.DataFrame(composition_rows),
        'selected_state_members':members,'chemical_ordered_unit13':ordered_unit,
        'heldout_predictions':heldout,'pooled_performance':performance}
    for name,frame in tables.items():frame.to_csv(result/'tables'/f'{name}.csv',index=False)
    leverage.to_csv(result/'tables/strain_leverage_and_residual.csv')
    (result/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (result/'summary.json').write_text(json.dumps(_clean_json(summary),indent=2,allow_nan=False)+'\n')
    return add_unnormalized_coordinate_descriptions(result)


def add_unnormalized_coordinate_descriptions(result_dir):
    """Two prespecified coordinate descriptions clarify the L2 denominator."""
    result=Path(result_dir);target=result/'tables/unnormalized_adf_ash_descriptions.csv'
    if target.exists():raise FileExistsError('Unnormalized coordinate descriptions already exist.')
    data=pd.read_csv(result/'tables/ordered_strains_and_thirds.csv')
    rows=[{'response':t,**describe_xy(data.chemical_state_score,data[t])} for t in ['raw_ADF','raw_ASH']]
    pd.DataFrame(rows).to_csv(target,index=False)
    summary=json.loads((result/'summary.json').read_text())
    summary['unnormalized_ADF_and_ASH']=rows
    (result/'summary.json').write_text(json.dumps(_clean_json(summary),indent=2,allow_nan=False)+'\n')
    return summary


def load_saved(result_dir):
    """Read-only Notebook entry point; no calculations or writes."""
    result=Path(result_dir)
    return {'summary':json.loads((result/'summary.json').read_text()),
            'ordered_strains':pd.read_csv(result/'tables/ordered_strains_and_thirds.csv',dtype={'dates':str}),
            'associations':pd.read_csv(result/'tables/fixed_axis_associations.csv')}


if __name__=='__main__':
    result=run_diagnostics(Path(__file__).resolve().parents[4])
    print(json.dumps(_clean_json(result),indent=2))
