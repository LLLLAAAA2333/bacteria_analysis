"""Sample-level descriptive neural/compound associations from existing caches.

No independent-pair tests, refitting of neural templates, imputation of absent
chemical reports, mechanistic claim, or predictive-validation claim is made.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_residuals(values, design):
    ranks = rankdata(np.asarray(values,float),axis=0)
    return ranks - design @ np.linalg.lstsq(design,ranks,rcond=None)[0]


def association(x, y, design):
    """Pearson correlation of rank residuals (partial Spearman convention)."""
    a,b=rank_residuals(x,design),rank_residuals(y,design)
    denominator=np.outer(np.linalg.norm(a,axis=0),np.linalg.norm(b,axis=0))
    return np.divide(a.T@b,denominator,out=np.full(denominator.shape,np.nan),where=denominator>1e-10)


def unit(values):
    norm=np.linalg.norm(values,axis=1)
    assert np.all(norm>0)
    return values/norm[:,None]


def run_exploration(repo,out):
    repo,out=Path(repo).resolve(),Path(out).resolve()
    tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    neural=repo/'reports/exploration_response_profiles_individual_snr_20261002/tables'
    chemistry=repo/'reports/population_first_20260930/tables'
    paths=[neural/'strain_coefficients.csv',neural/'condition_metrics.csv',
           neural/'sensitivity_condition_metrics.csv',neural/'sensitivity_templates.csv',neural/'templates.csv',
           chemistry/'aligned_chemical_log2fc_paired.csv',chemistry/'aligned_chemical_metadata.csv',
           chemistry/'aligned_chemical_report_values_all.csv',chemistry/'aligned_chemical_reference_groups_paired.csv',
           chemistry/'aligned_taxonomy_paired.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+list((repo/'notebook').glob('*.ipynb'))}
    coeff=pd.read_csv(paths[0],index_col=0)
    metrics=pd.read_csv(paths[1])
    sensitivity=pd.read_csv(paths[2])
    tmpl=pd.read_csv(paths[3]);main_t=pd.read_csv(paths[4]).pivot(index='cell',columns='bin_index',values='template').loc[coeff.columns]
    chem=pd.read_csv(paths[5],index_col=0).loc[coeff.index]
    meta=pd.read_csv(paths[6],index_col=0).loc[chem.columns]
    raw=pd.read_csv(paths[7],index_col=0).loc[coeff.index,chem.columns]
    refs=pd.read_csv(paths[8],index_col=0).loc[coeff.index].reference_group
    taxonomy=pd.read_csv(paths[9],index_col=0).loc[coeff.index]
    assert coeff.shape==(106,13) and chem.shape==(106,380)
    assert np.isfinite(coeff.to_numpy()).all()
    primary=meta.previous_complete_162_eligible.to_numpy(bool)
    assert primary.sum()==162 and raw.loc[:,primary].notna().all().all()
    assert (meta.loc[primary,'QCRSD']<=.30).all()
    dates=metrics[['strain','block']].drop_duplicates()
    date_weights=pd.crosstab(dates.strain,dates.block).reindex(coeff.index)
    date_weights=date_weights.div(date_weights.sum(axis=1),axis=0)
    sample=pd.DataFrame({'sample_id':coeff.index,'reference_group':refs.to_numpy(),
                         'genus':taxonomy.genus_clean.to_numpy(),'species':taxonomy.species_clean.to_numpy()})
    sample['neural_dates']=[';'.join(str(c) for c in date_weights.columns[row>0]) for row in date_weights.to_numpy()]
    sample['neural_nonzero_cells']=(coeff.to_numpy()!=0).sum(axis=1)
    sample.to_csv(tables/'sample_context.csv',index=False)
    date_weights.to_csv(tables/'neural_date_weights.csv')
    z=np.c_[np.ones(len(coeff)),pd.get_dummies(refs).to_numpy(float),date_weights.to_numpy(float)]
    zg=np.c_[z,pd.get_dummies(taxonomy.genus_clean).to_numpy(float)]
    designs={'pooled':np.ones((len(coeff),1)),'reference_date':z,'reference_date_genus':zg}
    y=coeff.to_numpy();representations={'coefficient':y,'unit':unit(y)}
    pre=metrics.groupby(['strain','cell']).raw_coefficient.mean().unstack('cell').loc[coeff.index,coeff.columns].to_numpy()
    representations['pre_gate_same_template']=pre
    template_cos={}
    for label in ['unfiltered','0.25','0.75']:
        df=sensitivity[sensitivity.threshold.eq(label)]
        values=df.groupby(['strain','cell']).coefficient.mean().unstack('cell').loc[coeff.index,coeff.columns].to_numpy()
        t=tmpl[tmpl.threshold.eq(label)].pivot(index='cell',columns='bin_index',values='template').loc[coeff.columns]
        dot=np.sum(t.to_numpy()*main_t.to_numpy(),axis=1)
        cosine=dot/(np.linalg.norm(t,axis=1)*np.linalg.norm(main_t,axis=1))
        assert np.isfinite(cosine).all() and (np.abs(cosine)>.8).all()
        # Keep the primary template sign convention if another fit flipped sign.
        values*=np.where(dot>=0,1,-1)[None,:]
        name='unfiltered' if label=='unfiltered' else 'snr_'+label
        representations[name]=values
        template_cos[name]={cell:float(v) for cell,v in zip(coeff.columns,cosine)}
    features=chem.columns[primary].tolist();x=chem.loc[:,primary].to_numpy()
    rows=[];matrices={}
    for rep,values in representations.items():
        for adjust,design in designs.items():
            if rep not in ['coefficient','unit'] and adjust!='reference_date':continue
            r=association(x,values,design);matrices[(rep,adjust)]=r
            for j,f in enumerate(features):
                for k,cell in enumerate(coeff.columns):
                    rows.append(dict(feature=f,cell=cell,representation=rep,adjustment=adjust,
                                     rho=float(r[j,k]),n_samples=len(x),design_rank=int(np.linalg.matrix_rank(design))))
    scan=pd.DataFrame(rows);scan.to_csv(tables/'primary_associations.csv',index=False)
    wide=scan.assign(metric=scan.representation+'__'+scan.adjustment).pivot(index=['feature','cell'],columns='metric',values='rho')
    scoring=['coefficient__reference_date','unit__reference_date',
             'coefficient__reference_date_genus','unit__reference_date_genus',
             'pre_gate_same_template__reference_date','unfiltered__reference_date']
    a=wide[scoring].to_numpy();same=(np.sign(a)==np.sign(a[:,[0]])).all(axis=1)
    wide['consistent_direction_across_six_checks']=same
    wide['descriptive_screen_score']=np.where(same,np.min(np.abs(a),axis=1),0)
    wide=wide.sort_values('descriptive_screen_score',ascending=False).reset_index()
    wide.to_csv(tables/'candidate_ranking.csv',index=False)
    # Show twelve distinct compound/cell pairs, not the strongest pooled-only hits.
    selected=wide.head(12).copy();selected.insert(0,'display_rank',range(1,len(selected)+1))
    selected.to_csv(tables/'selected_candidates.csv',index=False)
    # Restrict/rerank/refit within each subset; deletion intervals are not CIs.
    robustness=[];within=[]
    for label,labels in [('reference',refs.to_numpy()),('genus',taxonomy.genus_clean.to_numpy())]:
        for group in sorted(set(labels)):
            keep=labels!=group
            for rep in ['coefficient','unit']:
                r=association(x[keep],representations[rep][keep],z[keep])
                for candidate in selected.itertuples():
                    j,k=features.index(candidate.feature),coeff.columns.get_loc(candidate.cell)
                    robustness.append(dict(feature=candidate.feature,cell=candidate.cell,representation=rep,
                        omitted_type=label,omitted_group=group,n_retained=int(keep.sum()),rho=float(r[j,k])))
    for date in date_weights.columns:
        keep=date_weights[date].eq(0).to_numpy()
        for rep in ['coefficient','unit']:
            r=association(x[keep],representations[rep][keep],z[keep])
            for candidate in selected.itertuples():
                j,k=features.index(candidate.feature),coeff.columns.get_loc(candidate.cell)
                robustness.append(dict(feature=candidate.feature,cell=candidate.cell,representation=rep,
                    omitted_type='neural_date',omitted_group=str(date),n_retained=int(keep.sum()),rho=float(r[j,k])))
    for group in sorted(refs.unique()):
        keep=refs.eq(group).to_numpy()
        for rep in ['coefficient','unit']:
            r=association(x[keep],representations[rep][keep],z[keep])
            for candidate in selected.itertuples():
                j,k=features.index(candidate.feature),coeff.columns.get_loc(candidate.cell)
                within.append(dict(feature=candidate.feature,cell=candidate.cell,representation=rep,
                    reference_group=group,n_samples=int(keep.sum()),nuisance_rank=int(np.linalg.matrix_rank(z[keep])),rho=float(r[j,k])))
    pd.DataFrame(robustness).to_csv(tables/'leave_one_group_out.csv',index=False)
    pd.DataFrame(within).to_csv(tables/'within_reference_associations.csv',index=False)
    # Alternative ranks computed separately within each chemical reference.
    xr,yr=np.empty_like(x),np.empty_like(y)
    for group in refs.unique():
        take=refs.eq(group).to_numpy()
        xr[take]=(rankdata(x[take],axis=0)-.5)/take.sum()
        yr[take]=(rankdata(y[take],axis=0)-.5)/take.sum()
    xr-=z@np.linalg.lstsq(z,xr,rcond=None)[0];yr-=z@np.linalg.lstsq(z,yr,rcond=None)[0]
    r=(xr.T@yr)/np.outer(np.linalg.norm(xr,axis=0),np.linalg.norm(yr,axis=0))
    pd.DataFrame([dict(feature=f,cell=cell,rho=float(r[j,k])) for j,f in enumerate(features)
                  for k,cell in enumerate(coeff.columns)]).to_csv(tables/'within_reference_rank_sensitivity.csv',index=False)
    # Other features: use actual reported samples only, no raw-fill values.
    # Small-support residual correlations are listed as unavailable.
    secondary=[]
    for f in chem.columns[~primary]:
        keep=raw[f].notna().to_numpy();nn=int(keep.sum());rank=int(np.linalg.matrix_rank(z[keep]))
        ok=nn>=40 and nn-rank>=25
        xx=chem.loc[keep,[f]].to_numpy()
        r=association(xx,y[keep],z[keep])[0] if ok else np.full(len(coeff.columns),np.nan)
        controls={}
        if ok:
            for name,values in representations.items():
                controls[name]=association(xx,values[keep],z[keep])[0]
            controls['coefficient_genus']=association(xx,y[keep],zg[keep])[0]
            controls['unit_genus']=association(xx,representations['unit'][keep],zg[keep])[0]
        genus_rank=int(np.linalg.matrix_rank(zg[keep]))
        for k,cell in enumerate(coeff.columns):
            secondary.append(dict(feature=f,cell=cell,rho=float(r[k]),n_reported=nn,nuisance_rank=rank,
                genus_residual_dimensions=nn-genus_rank,
                qc_rsd=float(meta.loc[f,'QCRSD']),screenable=ok,
                **{name:float(val[k]) for name,val in controls.items()}))
    pd.DataFrame(secondary).to_csv(tables/'secondary_reported_only_associations.csv',index=False)
    # Full 106 samples retained in each diagnostic scatter.
    residual_x=rank_residuals(x,z);residual_y=rank_residuals(y,z)
    point_rows=[]
    for candidate in selected.itertuples():
        j,k=features.index(candidate.feature),coeff.columns.get_loc(candidate.cell)
        for i,s in enumerate(coeff.index):
            point_rows.append(dict(feature=candidate.feature,cell=candidate.cell,sample_id=s,
                reference_group=refs.iloc[i],genus=taxonomy.genus_clean.iloc[i],
                chemical_log2fc=float(x[i,j]),coefficient=float(y[i,k]),unit_coefficient=float(unit(y)[i,k]),
                chemical_rank_residual=float(residual_x[i,j]),coefficient_rank_residual=float(residual_y[i,k])))
    pd.DataFrame(point_rows).to_csv(tables/'candidate_points.csv',index=False)
    rx=rank_residuals(x,z);chemical_r=(rx.T@rx)/np.outer(np.linalg.norm(rx,axis=0),np.linalg.norm(rx,axis=0))
    linked=[]
    for f in selected.feature.unique():
        j=features.index(f)
        other=np.argsort(-np.abs(chemical_r[j]))
        for k in [k for k in other if k!=j][:5]:
            linked.append(dict(feature=f,other_feature=features[k],rho_reference_date=float(chemical_r[j,k])))
    pd.DataFrame(linked).to_csv(tables/'candidate_chemical_covariation.csv',index=False)
    # Explicitly retain contextual results for the earlier pair-level Arginine example.
    wide[wide.feature.eq('Arginine')].to_csv(tables/'arginine_context.csv',index=False)
    diagnostics={}
    for name,design in designs.items():
        u,s,_=np.linalg.svd(design,full_matrices=False)
        rank=int((s>np.finfo(float).eps*max(design.shape)*s[0]).sum())
        basis=u[:,:rank]
        leverage=(basis*basis).sum(axis=1)
        residual=rank_residuals(x,design)
        diagnostics[name]=dict(rank=int(np.linalg.matrix_rank(design)),
            residual_dimensions=int(len(x)-np.linalg.matrix_rank(design)),
            samples_with_no_residual_information=int((leverage>1-1e-8).sum()),
            max_design_residual_crossproduct=float(np.max(np.abs(design.T@residual))))
        assert np.max(np.abs(design.T@residual))<1e-8
    parameters=dict(n_samples=106,n_primary_features=162,n_cells=13,
        primary_selection='All 106 raw report values present and report QC RSD <=0.30; pre-existing list',
        association='Pearson correlation of residuals of global average ranks after OLS on stated nuisance design',
        covariates='Chemical reference dummies plus per-strain neural recording-date membership weighted equally over available dates; intercept',
        genus_sensitivity='Additional genus dummies; changes estimand to available within-genus variation, eliminates singleton-genus information',
        main_representations=['SNR>=0.5 signed coefficients','L2-normalized coefficients'],
        template_alignment_cosines=template_cos,design_diagnostics=diagnostics,
        candidate_selection='Top12 by minimum absolute association across coefficient, unit, coefficient+genus, unit+genus, pre-gate same-template and true-unfiltered ref/date checks, only if all six signs agree',
        deletion_checks='Remove every sample in group, rerank and refit nuisance design; ranges are not confidence intervals or held-out prediction',
        secondary_selection='Remaining features use original-report observed samples only; >=40 samples and >=25 residual dimensions, list QC separately',
        interpretation='Same-data exploratory screening; no multiplicity-adjusted significance, independent validation, causal attribution, or technology-performance claim',
        source_sha256=hashes)
    (out/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    verification=dict(inputs_and_notebooks_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items()),
        n_primary_compound_cell_pairs=162*13,n_primary_samples=106,n_missing_primary_raw_values=int(raw.loc[:,primary].isna().sum().sum()),
        n_selected=len(selected),n_candidate_points=len(point_rows),all_unit_norms_one=bool(np.allclose(np.linalg.norm(unit(y),axis=1),1)),
        no_new_templates_fitted=True,no_missing_values_imputed=True)
    assert verification['inputs_and_notebooks_unchanged']
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(selected[['feature','cell','descriptive_screen_score','coefficient__reference_date','coefficient__reference_date_genus']].to_string(index=False))
    return verification


if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]
    run_exploration(out.parents[1],out)
