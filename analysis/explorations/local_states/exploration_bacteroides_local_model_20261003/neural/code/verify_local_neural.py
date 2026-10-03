"""Independent verification via covariance eigenvectors; no analysis import."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def independent_pca(x):
    mean=np.mean(x,axis=0)
    centered=x-mean
    covariance=centered.T@centered/(len(x)-1)
    ev,vec=np.linalg.eigh(covariance)
    ev,vec=ev[::-1],vec[:,::-1]
    for j in range(vec.shape[1]):
        if vec[np.argmax(np.abs(vec[:,j])),j]<0:vec[:,j]*=-1
    return mean,vec,ev,centered@vec


def verify(result_dir):
    out=Path(result_dir).resolve();t=out/'tables';errors={}
    def check(a,b,label):
        a,b=np.asarray(a,float),np.asarray(b,float)
        error=float(np.max(np.abs(a-b)))
        assert np.allclose(a,b,atol=1e-10,rtol=1e-10), (label,error)
        errors[label]=max(errors.get(label,0),error)
    manifest=json.loads((out/'source_manifest.json').read_text())
    for item in manifest.values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    source_context=pd.read_csv(manifest['sample_context']['path'],index_col='strain',dtype={'dates':str})
    metadata=pd.read_csv(t/'strain_metadata.csv',index_col='strain',dtype={'dates':str})
    expected=source_context[source_context.genus.eq('Bacteroides')].sort_index()
    assert metadata.index.equals(expected.index) and len(metadata)==29
    for field in ['species','taxonomy_note','dates']:
        assert metadata[field].fillna('').equals(expected[field].fillna(''))
    assert metadata.taxonomy_flag.sum()==6
    check(metadata.n_dates,expected.dates.str.split(';').map(len),'complete date counts')
    x=pd.read_csv(t/'neural_unit_profiles.csv',index_col='strain')
    raw_source=pd.read_csv(manifest['neural_unit']['path'],index_col='strain').loc[metadata.index,x.columns]
    check(x,raw_source,'source strain alignment')
    check(np.linalg.norm(x,axis=1),1,'unit vectors')
    mu,v,e,s=independent_pca(x.to_numpy())
    ratio=e/e.sum()
    savedload=pd.read_csv(t/'neural_loadings.csv',index_col='neuron')
    savedscore=pd.read_csv(t/'strain_scores.csv',index_col='strain')
    spectrum=pd.read_csv(t/'pca_spectrum.csv')
    savedmean=pd.read_csv(t/'mean_unit_profile.csv',index_col='neuron')
    check(savedmean.mean_unit,mu,'mean no scaling')
    check(savedload,v,'all 13 loading vectors from eigendecomposition')
    check(savedload.T@savedload,np.eye(13),'orthonormal loadings')
    check(savedscore[[f'PC{i}' for i in range(1,14)]],s,'all strain scores')
    check(spectrum.sample_variance,e,'all eigenvalues')
    check(spectrum.variance_ratio,ratio,'all variance ratios')
    check(spectrum.cumulative_variance_ratio,np.cumsum(ratio),'cumulative variance')
    check((x-mu).to_numpy(),s@v.T,'full reconstruction identity')
    reconstruction=pd.read_csv(t/'pc1_reconstructed_profiles.csv',index_col='strain')
    residual=pd.read_csv(t/'pc1_residual_profiles.csv',index_col='strain')
    check(reconstruction,mu+np.outer(s[:,0],v[:,0]),'PC1 reconstruction')
    check(residual,x-reconstruction,'PC1 residual')
    check(residual.to_numpy()@v[:,0],0,'residual orthogonal to PC1')
    check(np.square(residual).to_numpy().sum()/np.square(x-mu).to_numpy().sum(),1-ratio[0],'variance residual identity')
    check(savedscore.pc1_residual_l2,np.linalg.norm(residual,axis=1),'strain residual L2')
    check(savedscore.pc1_residual_rms,np.sqrt(np.mean(np.square(residual),axis=1)),'strain residual RMS')
    stability=pd.read_csv(t/'pc1_stability.csv')
    savedstabilityload=pd.read_csv(t/'stability_aligned_loadings.csv',index_col='holdout_id')
    heldout=pd.read_csv(t/'heldout_pc1_projections.csv')
    assert len(stability)==45 and stability.holdout_type.value_counts().to_dict()=={'strain':29,'species':16}
    for r in stability.itertuples():
        omitted=r.omitted_strains.split(';');train=x.drop(index=omitted)
        if r.holdout_type=='species':
            assert set(omitted)==set(metadata.index[metadata.species.eq(r.omitted_label)])
        tm,tv,te,ts=independent_pca(train.to_numpy())
        dot=float(tv[:,0]@v[:,0]);aligned=tv[:,0]*(1 if dot>=0 else -1)
        check(r.pc1_absolute_loading_cosine,abs(dot),'45 stability cosines')
        check([r.pc1_variance_ratio,r.pc2_variance_ratio],te[:2]/te.sum(),'45 training variance ratios')
        check(savedstabilityload.loc[r.holdout_id],aligned,'45 aligned loading vectors')
        hs=heldout[heldout.holdout_id.eq(r.holdout_id)].set_index('strain').loc[omitted]
        check(hs.training_centered_pc1_score,(x.loc[omitted]-tm).to_numpy()@aligned,'all heldout projections')
    pre=pd.read_csv(t/'pre_gate_unit_profiles.csv',index_col='strain')
    pm,pv,pe,ps=independent_pca(pre.to_numpy())
    sign=1 if pv[:,0]@v[:,0]>=0 else -1
    precheck=pd.read_csv(t/'pre_gate_sensitivity_summary.csv').iloc[0]
    check(precheck.pc1_absolute_loading_cosine,abs(pv[:,0]@v[:,0]),'pre-gate loading cosine')
    check(precheck.aligned_pc1_score_pearson,np.corrcoef(s[:,0],ps[:,0]*sign)[0,1],'pre-gate score correlation')
    check(pd.read_csv(t/'pre_gate_pc1_scores.csv',index_col='strain').PC1_pre_gate_aligned,ps[:,0]*sign,'pre-gate aligned scores')
    params=json.loads((out/'parameters.json').read_text())
    assert params['pc1_strain_order']==savedscore.PC1.sort_values(kind='stable').index.tolist()
    date_rows=pd.read_csv(t/'strain_date_membership.csv',dtype={'date':str})
    actual={(r.strain,r.date) for r in date_rows.itertuples()}
    expect={(strain,date) for strain,row in metadata.iterrows() for date in row.dates.split(';')}
    assert actual==expect and len(date_rows)==35
    result={'status':'PASS','independent_method':'covariance eigendecomposition plus direct projection/reconstruction checks',
            'n_strains':29,'n_neurons':13,'n_species_holdouts':16,'n_strain_holdouts':29,
            'n_strain_date_memberships':35,'metadata_and_taxonomy_flags_match_source':True,
            'source_hashes_match':True,'max_absolute_errors':errors,
            'maximum_absolute_error':max(errors.values())}
    (out/'verification/numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    verify(Path(__file__).resolve().parents[1])
