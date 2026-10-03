"""Context checks for secondary associations with incomplete chemical reports.

Uses only observed chemical values. These candidates are separate from the
complete162 primary screen and are not added to its diagnostic PDF.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from explore_relations import association


def check_secondary(repo,out):
    repo,out=Path(repo),Path(out)
    table=out/'tables'
    scan=pd.read_csv(table/'secondary_reported_only_associations.csv')
    checks=['coefficient','unit','coefficient_genus','unit_genus','pre_gate_same_template','unfiltered']
    values=scan[checks].to_numpy()
    finite=np.isfinite(values).all(axis=1)
    same=(np.sign(values)==np.sign(values[:,[0]])).all(axis=1)
    score=np.zeros(len(scan));use=finite&same
    score[use]=np.abs(values[use]).min(axis=1)
    scan['descriptive_screen_score']=score
    pool=scan[(scan.n_reported>=80)&(scan.qc_rsd<=.30)&(scan.genus_residual_dimensions>=25)]
    selected=pool.nlargest(3,'descriptive_screen_score').copy()
    selected['selection_reason']='Top3 same six-check score; >=80 original reports; QC<=.30; >=25 genus-adjusted residual dimensions'
    counter=scan[scan.feature.eq('Thymidine')&scan.cell.eq('ASH')].copy()
    counter['selection_reason']='High-coverage strongest adjusted coefficient-only secondary example; counterexample to pooled interpretation'
    selected=pd.concat([selected,counter],ignore_index=True)
    selected.to_csv(table/'secondary_context_candidates.csv',index=False)
    source=repo/'reports/population_first_20260930/tables'
    neural=repo/'reports/exploration_response_profiles_individual_snr_20261002/tables'
    y=pd.read_csv(neural/'strain_coefficients.csv',index_col=0)
    x=pd.read_csv(source/'aligned_chemical_log2fc_paired.csv',index_col=0).loc[y.index]
    raw=pd.read_csv(source/'aligned_chemical_report_values_all.csv',index_col=0).loc[y.index]
    ctx=pd.read_csv(table/'sample_context.csv',index_col=0).loc[y.index]
    dates=pd.read_csv(table/'neural_date_weights.csv',index_col=0).loc[y.index]
    z=np.c_[np.ones(len(y)),pd.get_dummies(ctx.reference_group).to_numpy(float),dates]
    rows=[]
    for row in selected.itertuples():
        reported=raw[row.feature].notna().to_numpy()
        for ref in sorted(ctx.reference_group.unique()):
            keep=reported&ctx.reference_group.eq(ref).to_numpy()
            r=association(x.loc[keep,[row.feature]],y.loc[keep,[row.cell]],z[keep])[0,0]
            rows.append(dict(feature=row.feature,cell=row.cell,check='within_reference',group=ref,
                n_samples=int(keep.sum()),nuisance_rank=int(np.linalg.matrix_rank(z[keep])),rho=float(r)))
        for name in ['reference_group','genus']:
            for group in sorted(ctx[name].unique()):
                keep=reported&ctx[name].ne(group).to_numpy()
                r=association(x.loc[keep,[row.feature]],y.loc[keep,[row.cell]],z[keep])[0,0]
                rows.append(dict(feature=row.feature,cell=row.cell,check='omit_'+name,group=group,
                    n_samples=int(keep.sum()),nuisance_rank=int(np.linalg.matrix_rank(z[keep])),rho=float(r)))
    pd.DataFrame(rows).to_csv(table/'secondary_context_checks.csv',index=False)
    return selected[['feature','cell','n_reported','rho','coefficient_genus','descriptive_screen_score']]


if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]
    print(check_secondary(out.parents[1],out).to_string(index=False))
