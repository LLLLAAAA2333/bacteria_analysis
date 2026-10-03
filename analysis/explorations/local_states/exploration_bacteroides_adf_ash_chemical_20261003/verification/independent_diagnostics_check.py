"""Independently recompute saved descriptive diagnostics; no new model search."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
DIAG=ROOT/'diagnostics'
PRIMARY='primary_unit_adf_minus_ash'
ERRORS={}


def close(label,a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    assert a.shape==b.shape,(label,a.shape,b.shape)
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10,err_msg=label)
    ERRORS[label]=max(ERRORS.get(label,0),float(np.nanmax(np.abs(a-b),initial=0)))


def stats(x,y,ref_iqr=None):
    x,y=np.asarray(x),np.asarray(y)
    intercept,slope=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    q1,q3=np.quantile(x,[.25,.75]);iqr=q3-q1
    return dict(n_strains=len(x),pearson=np.corrcoef(x,y)[0,1],spearman=spearmanr(x,y).statistic,
      slope=slope,intercept=intercept,x_min=x.min(),x_max=x.max(),x_q25=q1,x_q75=q3,x_iqr=iqr,
      fitted_observed_iqr_effect=slope*iqr,fitted_full_iqr_effect=slope*(iqr if ref_iqr is None else ref_iqr),
      y_mean=y.mean(),y_sd=y.std(ddof=1))


def check_stats(label,calculated,row):
    close(label,list(calculated.values()),row[list(calculated)].to_numpy())


def table(name):
    return pd.read_csv(DIAG/'tables'/f'{name}.csv',dtype={'dates':str})


def audit():
    manifest=json.loads((DIAG/'source_manifest.json').read_text())
    for record in manifest.values():
        assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
    data=pd.read_csv(ROOT/'model/tables/selected_full_state_scores.csv',index_col='strain',dtype={'dates':str})
    members=pd.read_csv(ROOT/'model/tables/selected_full_state_members.csv')
    ordered=data.reset_index().sort_values(['chemical_state_score','strain'],kind='stable').reset_index(drop=True)
    ordered['chemical_rank']=range(1,30);ordered['chemical_third']=['LOW']*10+['MID']*10+['HIGH']*9
    saved=table('ordered_strains_and_thirds')
    pd.testing.assert_frame_equal(ordered,saved,check_dtype=False,check_exact=False,atol=1e-12)
    frame13=table('chemical_ordered_unit13')
    pd.testing.assert_frame_equal(ordered[frame13.columns],frame13,check_dtype=False,check_exact=False,atol=1e-12)
    for filename in ['fixed_axis_associations','unnormalized_adf_ash_descriptions']:
        for _,row in table(filename).iterrows():
            check_stats(filename,stats(data.chemical_state_score,data[row.response]),row)
    for _,row in table('rank_third_summary').iterrows():
        g=ordered[ordered.chemical_third.eq(row.chemical_third)];v=g[row.variable]
        calc=[len(g),v.mean(),v.median(),v.min(),v.max(),v.std(ddof=1),g.species.nunique(),g.dates.nunique(),g.taxonomy_flag.sum()]
        cols=['n_strains','mean','median','min','max','sd','n_species','n_original_date_sets','n_taxonomy_flags']
        close('rank_third_summary',calc,row[cols].to_numpy())
    composition=table('rank_third_species_date_composition')
    count=0
    for tier,g in ordered.groupby('chemical_third'):
        for field in ['species','dates']:
            for label,h in g.groupby(field):
                row=composition[composition.chemical_third.eq(tier)&composition.label_field.eq(field)&composition.label.astype(str).eq(str(label))]
                assert len(row)==1 and row.iloc[0].n_strains==len(h) and row.iloc[0].strains==';'.join(h.strain)
                count+=1
    assert count==len(composition)
    full=stats(data.chemical_state_score,data[PRIMARY]);iqr=full['x_iqr']
    deletion=table('fixed_axis_deletion_influence')
    for _,row in deletion.iterrows():
        ids=[row.deleted_label] if row.deletion=='one_strain' else data.index[data.species.eq(row.deleted_label)].tolist()
        assert row.deleted_strains==';'.join(ids) and row.n_deleted==len(ids)
        g=data.drop(index=ids);calc=stats(g.chemical_state_score,g[PRIMARY],iqr)
        check_stats('fixed_axis_deletion_influence',calc,row)
        close('deletion_slope_change',calc['slope']-full['slope'],row.slope_change_from_full)
    leverage=table('strain_leverage_and_residual').set_index('strain').loc[data.index]
    centered=data.chemical_state_score-data.chemical_state_score.mean();shares=centered**2/(centered**2).sum()
    close('leverage',shares+1/29,leverage.OLS_leverage)
    close('x_deviation_share',shares,leverage.squared_x_deviation_share)
    close('residual',data[PRIMARY]-(full['intercept']+full['slope']*data.chemical_state_score),leverage.residual)
    individual=table('within_group_individual_residuals');support=table('within_group_support')
    for _,row in table('within_group_demeaned_associations').iterrows():
        field=row.group_field;counts=data.groupby(field).size();labels=counts.index[counts>=2]
        sub=data[data[field].isin(labels)].copy();x=sub.chemical_state_score;y=sub[PRIMARY]
        xm=np.zeros(len(sub));ym=xm.copy();n=xm.copy()
        for label in labels:
            mask=sub[field].eq(label).to_numpy();xm[mask]=x[mask].mean();ym[mask]=y[mask].mean();n[mask]=mask.sum()
            gs=sub.loc[mask];sr=support[support.group_field.eq(field)&support.group_label.astype(str).eq(str(label))].iloc[0]
            assert sr.strains==';'.join(gs.index)
            close('within_group_support',[len(gs),x[mask].min(),x[mask].max(),y[mask].min(),y[mask].max()],sr[['n_strains','x_min','x_max','y_min','y_max']])
        xx=x.to_numpy()-xm;yy=y.to_numpy()-ym
        check_stats('within_group_association',stats(xx,yy),row)
        close('within_group_counts',[len(labels),int((counts==1).sum()),len(sub)-len(labels)],[row.n_groups,row.n_singleton_groups_omitted,row.within_centering_df])
        sr=individual[individual.group_field.eq(field)].set_index('strain').loc[sub.index]
        close('within_group_individual',np.column_stack([xm,ym,xx,yy,n]),sr[['x_group_mean','y_group_mean','x_demeaned','y_demeaned','n_in_group']])
    comp=table('selected_state_annotation_composition')
    for _,row in comp.iterrows():
        g=members[members[row.annotation_level].fillna('Unannotated').eq(row.annotation)]
        assert row.members==';'.join(g.metabolite)
        close('annotation_composition',[len(g),g.family.nunique(),g.score_weight.sum()],[row.n_members,row.n_families,row.summed_score_weight])
    pd.testing.assert_frame_equal(members,table('selected_state_members'),check_exact=False,atol=1e-12)
    for name in ['heldout_predictions','pooled_performance']:
        original=pd.read_csv(ROOT/'model/tables'/f'{name}.csv',dtype={'dates':str})
        pd.testing.assert_frame_equal(original,table(name),check_exact=False,atol=1e-12)
    result={'status':'PASS','maximum_absolute_error':max(ERRORS.values()),'error_maxima':ERRORS,
            'scope':'All18 associations,45 fixed-axis deletions,51 rank-third summaries,40 composition rows,29 leverage rows,48 demeaned residual rows,14 group supports,25 annotation composition rows, input hashes and copied model outputs',
            'interpretation_notes':['Rank thirds are chemical-score-only conditional on a response-selected full-data state, not independent evidence.',
             'Fixed-axis deletion and group demeaning are descriptive; they do not repeat discovery within training folds.',
             'Unit and unnormalized coordinate signs are descriptive regression trends, not neuron-level causality.']}
    (ROOT/'verification/independent_diagnostics_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
