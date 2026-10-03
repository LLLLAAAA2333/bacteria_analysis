"""Independent covariance/angle recomputation plus geometry identities."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from neural_subspace import compare_subspaces
from run_subspace_analysis import run_analysis


def covariance_pca(values):
    a=np.asarray(values,float);mean=np.mean(a,axis=0)
    covariance=np.cov(a,rowvar=False,ddof=1)
    eigenvalues,loadings=np.linalg.eigh(covariance)
    order=np.argsort(eigenvalues)[::-1];eigenvalues=eigenvalues[order];loadings=loadings[:,order]
    for k in range(loadings.shape[1]):
        j=np.argmax(np.abs(loadings[:,k]))
        if loadings[j,k]<0: loadings[:,k]*=-1
    return mean,eigenvalues,loadings


def verify(repo_root):
    repo=Path(repo_root).resolve();out=repo/'reports/exploration_bacteroides_neural_reliability_20261003/subspace';t=out/'tables'
    source=repo/'reports/exploration_bacteroides_local_model_20261003/neural/tables'
    x=pd.read_csv(source/'neural_unit_profiles.csv',index_col='strain').sort_index()
    pre=pd.read_csv(source/'pre_gate_unit_profiles.csv',index_col='strain').loc[x.index,x.columns]
    metadata=pd.read_csv(source/'strain_metadata.csv',index_col='strain',dtype={'dates':str}).loc[x.index]
    results=pd.read_csv(t/'deletion_subspace_metrics.csv').set_index('fit_id')
    mean_table=pd.read_csv(t/'all_fit_means.csv').set_index('fit_id')
    loading_table=pd.read_csv(t/'all_fit_loadings.csv')
    spectra=pd.read_csv(t/'all_fit_spectra.csv')
    scores=pd.read_csv(t/'all_fit_strain_scores.csv')
    folds=json.loads((out/'fold_ids.json').read_text())
    assert len(folds)==45 and len(results)==45
    assert sum(row['scheme']=='leave_one_strain_out' for row in folds)==29
    assert sum(row['scheme']=='leave_one_recorded_species_out' for row in folds)==16
    errors={};components=[f'PC{i}' for i in range(1,14)]
    def close(name,actual,expected,atol=1e-9):
        aa,ee=np.asarray(actual,float),np.asarray(expected,float)
        np.testing.assert_allclose(aa,ee,atol=atol,rtol=1e-9)
        errors[name]=max(errors.get(name,0),float(np.max(np.abs(aa-ee))))
    full_mean,full_eigen,full_loading=covariance_pca(x)
    all_fits=[('full_gated',x,[],x),('full_pre_gate',pre,[],pre)]
    for fold in folds:
        omitted=fold['omitted_ids'];training=x.loc[fold['train_ids']]
        assert not set(omitted)&set(training.index)
        assert set(omitted)|set(training.index)==set(x.index)
        if fold['scheme']=='leave_one_strain_out': assert len(omitted)==1 and fold['fit_id']=='strain:'+omitted[0]
        else:
            species=fold['fit_id'].split(':',1)[1]
            assert omitted==metadata.index[metadata.species.eq(species)].tolist()
        all_fits.append((fold['fit_id'],training,omitted,x))
    angles_rows=[]
    for fit_id,training,omitted,all_values in all_fits:
        mean,eigen,loading=covariance_pca(training)
        close('all_fit_means',mean_table.loc[fit_id,x.columns],mean)
        saved_loadings=loading_table[loading_table.fit_id.eq(fit_id)].set_index('neuron').loc[x.columns,components]
        close('all_fit_canonical_loadings',saved_loadings,loading)
        saved_spectrum=spectra[spectra.fit_id.eq(fit_id)].set_index('component').loc[components]
        close('all_fit_eigenvalues',saved_spectrum.sample_eigenvalue,eigen)
        close('all_fit_evr',saved_spectrum.evr,eigen/eigen.sum())
        saved_scores=scores[scores.fit_id.eq(fit_id)].set_index('strain').loc[x.index]
        close('all_training_centered_projections',saved_scores[components],(all_values-mean)@loading)
        assert saved_scores.index[saved_scores.role.eq('omitted_projection')].tolist()==omitted
        if fit_id=='full_gated': continue
        angles=np.sort(np.degrees(subspace_angles(full_loading[:,:2],loading[:,:2])))
        v,u=full_loading[:,:2],loading[:,:2]
        projector=np.sqrt(((v@v.T-u@u.T)**2).sum()/2)
        trig=np.sqrt(np.sin(np.radians(angles))@np.sin(np.radians(angles)))
        close('projector_trig_identity',projector,trig)
        single_cos=abs(full_loading[:,0]@loading[:,0]);single_angle=np.degrees(np.arccos(np.clip(single_cos,0,1)))
        if fit_id=='full_pre_gate': row=pd.read_csv(t/'pre_gate_subspace_comparison.csv').iloc[0]
        else: row=results.loc[fit_id]
        close('all_principal_angles_scipy',row[['principal_angle_1_deg','principal_angle_2_deg']],angles,atol=1e-7)
        close('all_max_principal_angles',row.max_principal_angle_deg,angles[-1],atol=1e-7)
        close('all_projector_distances',row.projector_frobenius_over_sqrt2,projector)
        close('all_pc1_absolute_cosines',row.pc1_absolute_cosine,single_cos)
        close('all_pc1_acute_angles',row.pc1_acute_angle_deg,single_angle,atol=1e-7)
        close('all_top2_evr',row.top2_evr,eigen[:2].sum()/eigen.sum())
        close('all_lambda2_lambda3_gap',row.lambda2_minus_lambda3,eigen[1]-eigen[2])
        close('all_relative_gap',row.relative_gap_2_vs_3,(eigen[1]-eigen[2])/eigen[1])
        close('all_lambda2_lambda3_ratio',row.lambda2_over_lambda3,eigen[1]/eigen[2])
        angles_rows.append({'fit_id':fit_id,'angle1_scipy_deg':angles[0],'angle2_scipy_deg':angles[1],
                            'projector_distance_explicit':projector,'projector_distance_trig':trig})
    # Rotation/sign invariance and non-invariance of a single PC direction.
    radians=np.radians(68);rotation=np.array([[np.cos(radians),-np.sin(radians)],[np.sin(radians),np.cos(radians)]])
    rotated=full_loading.copy();rotated[:,:2]=full_loading[:,:2]@rotation
    geometry=compare_subspaces(full_loading,rotated)
    assert geometry['projector_frobenius_over_sqrt2']<1e-12 and geometry['max_principal_angle_deg']<3e-6
    close('rotation_single_pc1_angle',geometry['pc1_acute_angle_deg'],68.)
    flipped=full_loading.copy();flipped[:,0]*=-1
    flipped_metrics=compare_subspaces(full_loading,flipped)
    assert flipped_metrics['projector_frobenius_over_sqrt2']<1e-12 and flipped_metrics['max_principal_angle_deg']<3e-6
    close('sign_flip_pc1_absolute_cosine',flipped_metrics['pc1_absolute_cosine'],1.)
    # Recompute all exported metric summaries and selected extrema directly.
    summary_table=pd.read_csv(t/'deletion_metric_summary.csv')
    for row in summary_table.itertuples():
        vals=results.loc[results.scheme.eq(row.scheme),row.metric]
        close('metric_summary_quantiles',[row.minimum,row.q25,row.median,row.q75,row.maximum],np.quantile(vals,[0,.25,.5,.75,1]))
        assert row.n==len(vals)
    worst=pd.read_csv(t/'worst_deletion_cases.csv')
    for row in worst.itertuples():
        block=results[results.scheme.eq(row.scheme)]
        assert row.fit_id==block[row.selected_by].idxmax()
    manifest=json.loads((out/'manifest.json').read_text())
    for item in manifest['inputs'].values(): assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    for name,value in manifest['code_sha256'].items(): assert hashlib.sha256((out/'code'/name).read_bytes()).hexdigest()==value
    assert hashlib.sha256((out/'protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    pd.testing.assert_frame_equal(pd.read_csv(t/'strain_metadata_29.csv',index_col='strain',dtype={'dates':str}),metadata)
    try: run_analysis(repo)
    except FileExistsError: pass
    else: raise AssertionError('Overwrite guard failed')
    pd.DataFrame(angles_rows).to_csv(out/'independent_angle_verification.csv',index=False)
    verification={'status':'PASS','fits_verified':47,'deletion_comparisons':45,'pre_gate_comparisons':1,
                  'max_abs_errors':errors,'geometry_rotation_test_degrees':68,
                  'checks':['covariance eigendecomposition of all47 fits','canonical signs','all13 spectra/loadings/means','all29 training-centered projections per fit','all45 train/omitted ID sets',
                            'SciPy principal angles','explicit projector distance','sine identity','PC1 absolute cosine and acute angle','lambda2-lambda3 gap and ratio',
                            'sign and plane rotation invariance','all summaries and extrema','metadata and source/code hashes','existing-output refusal'],
                  'scope':'Computational verification only; no independent experimental validation.'}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({'status':'PASS','largest_abs_error':max(errors.values()),'fits':47}))
    return verification


if __name__=='__main__': verify(Path(__file__).resolve().parents[4])
