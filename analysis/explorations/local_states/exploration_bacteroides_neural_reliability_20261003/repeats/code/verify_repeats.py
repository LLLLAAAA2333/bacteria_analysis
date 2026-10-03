"""Independent checks from source records and saved tables; no analysis import."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


def verify(result_dir):
    out=Path(result_dir).resolve();t=out/'tables';errors={}
    def eq(a,b,label):
        a,b=np.asarray(a,float),np.asarray(b,float)
        assert a.shape==b.shape or b.shape==(),(label,a.shape,b.shape)
        assert np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True),label
        finite=np.isfinite(a)&np.isfinite(b)
        err=float(np.max(np.abs(a-b)[finite])) if np.any(finite) else 0.
        errors[label]=max(errors.get(label,0),err)
    src=json.loads((out/'source_manifest.json').read_text())
    for f in src.values():assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
    meta=pd.read_csv(t/'strain_metadata.csv',index_col='strain',dtype={'dates':str});ids=meta.index.tolist()
    source_meta=pd.read_csv(src['context']['path'],index_col='strain',dtype={'dates':str}).loc[ids]
    for col in ['dates','species','taxonomy_note']:assert meta[col].fillna('').equals(source_meta[col].fillna(''))
    assert len(ids)==29
    x=pd.read_csv(src['full_unit']['path'],index_col='strain').loc[ids];cells=x.columns.tolist()
    mean=pd.read_csv(src['full29_mean']['path'],index_col='neuron').loc[cells,'mean_unit'].to_numpy()
    plane=pd.read_csv(src['full29_loadings']['path'],index_col='neuron').loc[cells,['PC1','PC2']].to_numpy()
    templates=pd.read_csv(src['templates']['path']).pivot(index='cell',columns='bin_index',values='template').loc[cells]
    curves=pd.read_parquet(src['animal_curves']['path'],filters=[('sample_id','in',ids)]).reset_index()
    for col in ['sample_id','date','worm_key','neuron_class']:curves[col]=curves[col].astype(str)
    curves['animal_id']=curves.date+'|'+curves.worm_key
    grouped={key:(g.animal_id.to_numpy(),g[[str(v) for v in range(40)]].to_numpy()) for key,g in curves.groupby(['sample_id','date','neuron_class'])}
    assignment=pd.read_csv(src['assignments']['path']);support=pd.read_csv(t/'animal_split_condition_support.csv',dtype={'date':str})
    half=pd.read_csv(t/'animal_half_profiles.csv');coverage=pd.read_csv(t/'animal_strain_coverage.csv')
    metrics=pd.read_csv(t/'animal_split_metrics.csv');aux=pd.read_csv(t/'animal_auxiliary_split_metrics.csv')
    planes=pd.read_csv(t/'animal_half_plane_angles.csv')
    assert len(support)==45500 and len(coverage)==2900 and len(half)==11600
    for split in [0,99]:
        mapping=assignment[assignment.split.eq(split)].set_index('animal_id').half.to_dict()
        saved=support[support.split.eq(split)].set_index(['strain','date','cell'])
        derived={}
        for key,(animals,values) in grouped.items():
            cell=key[2];h=templates.loc[cell].to_numpy()
            for label in ['A','B']:
                vals=values[np.array([mapping[a]==label for a in animals])]
                vals=vals[np.isfinite(vals).all(axis=1)];n=len(vals)
                eq(n,saved.loc[key,f'{label}_n'],'independent split cell n')
                if not n:derived[(*key,label)]=(n,np.nan,np.nan);continue
                mu=np.mean(vals,axis=0);projection=float(np.mean(mu.reshape(8,5),axis=1)@h/(h@h))
                if n>=2:
                    v=float(np.var(vals,axis=0,ddof=1).mean());signal=float(np.mean(mu**2));coherent=max(signal-v/n,0.)
                    snr=np.sqrt(coherent/v) if v>0 else (np.inf if coherent>0 else 0.)
                    gated=projection if snr>=.5 else 0.
                    eq(snr,saved.loc[key,f'{label}_snr'],'independent SNR')
                    eq(gated,saved.loc[key,f'{label}_gated'],'independent gated coefficient')
                    eq(projection,saved.loc[key,f'{label}_pre_gate'],'independent pre-gate coefficient')
                else:gated=np.nan
                eq(projection,saved.loc[key,f'{label}_projection'],'independent fixed template projection')
                derived[(*key,label)]=(n,gated,projection if n>=2 else np.nan)
        for representation,position in [('gated',1),('pre_gate',2)]:
            for label in ['A','B']:
                savedhalf=half[(half.split==split)&(half.representation==representation)&(half.half==label)].set_index('strain').loc[ids]
                expected=np.full((29,13),np.nan)
                for i,strain in enumerate(ids):
                    for j,cell in enumerate(cells):
                        shared_dates=[d for d in meta.loc[strain,'dates'].split(';') if derived[(strain,d,cell,'A')][0]>=2 and derived[(strain,d,cell,'B')][0]>=2]
                        if shared_dates:expected[i,j]=np.mean([derived[(strain,d,cell,label)][position] for d in shared_dates])
                eq(savedhalf[[f'coefficient_{c}' for c in cells]],expected,'common-date coefficient aggregation')
                complete=np.isfinite(expected).all(axis=1);norm=np.full(29,np.nan);norm[complete]=np.linalg.norm(expected[complete],axis=1)
                unit=np.full_like(expected,np.nan);valid=complete&(norm>1e-12);unit[valid]=expected[valid]/norm[valid,None]
                eq(savedhalf[[f'unit_{c}' for c in cells]],unit,'complete13 unit normalization')
                eq(savedhalf[['fixed_PC1','fixed_PC2']],(unit-mean)@plane,'fixed full29 projection')
            group=half[(half.split==split)&(half.representation==representation)&half.joint_complete13_nonzero]
            a=group[group.half.eq('A')].set_index('strain').sort_index();b=group[group.half.eq('B')].set_index('strain').loc[a.index]
            n=len(a);sets=meta.loc[a.index,'dates'].to_numpy()
            cols={'unit_13D':[f'unit_{c}' for c in cells],'fixed_PC12':['fixed_PC1','fixed_PC2'],
                  'fixed_PC1':['fixed_PC1'],'fixed_PC2':['fixed_PC2'],'unit_ADF_minus_ASH':['unit_ADF_minus_ASH'],'unit_AWB':['unit_AWB'],
                  'coefficient_ADF_minus_ASH':['raw_ADF_minus_ASH'],'coefficient_AWB':['raw_AWB'],'coefficient_norm':['coefficient_norm']}
            for name,columns in cols.items():
                av,bv=a[columns].to_numpy(),b[columns].to_numpy()
                same=np.sqrt(sum(np.sum((av[i]-bv[i])**2) for i in range(n))/n)
                between=np.sqrt(sum(np.sum((av[i]-bv[j])**2) for i in range(n) for j in range(n) if i!=j)/(n*(n-1)))
                pairs=[np.sum((av[i]-bv[j])**2) for i in range(n) for j in range(n) if i!=j and sets[i]==sets[j]]
                table=aux if name.startswith('coefficient_') else metrics
                row=table[(table.split==split)&(table.representation==representation)&(table.metric==name)].iloc[0]
                eq(row.same_RMS,same,'selected split same RMS')
                eq(row.between_RMS,between,'selected split all ordered between RMS')
                eq(row.n_ordered_between_pairs,n*(n-1),'ordered-pair denominator')
                eq(row.n_same_original_date_label_ordered_pairs,len(pairs),'same-original-label denominator')
                eq(row.same_original_date_label_between_RMS,np.sqrt(np.mean(pairs)) if pairs else np.nan,'same-original-label RMS')
            aa=a[[f'unit_{c}' for c in cells]].to_numpy();bb=b[[f'unit_{c}' for c in cells]].to_numpy()
            ca=np.cov(aa,rowvar=False);cb=np.cov(bb,rowvar=False)
            ea,va=np.linalg.eigh(ca);eb,vb=np.linalg.eigh(cb)
            angles=np.degrees(subspace_angles(va[:,-2:],vb[:,-2:]))
            row=planes[(planes.split==split)&(planes.representation==representation)].iloc[0]
            eq([row.angle_large_deg,row.angle_small_deg],angles,'independent half-plane principal angles')
    date=pd.read_csv(t/'date_profiles.csv',dtype={'date':str});dsummary=pd.read_csv(t/'date_summary.csv')
    existing=pd.read_csv(src['conditions']['path'],dtype={'block':str}).set_index(['strain','block','cell'])
    for row in date.itertuples():
        col='coefficient' if row.representation=='gated' else 'raw_coefficient'
        coef=np.array([existing.loc[(row.strain,row.date,c),col] for c in cells]);unit=coef/np.linalg.norm(coef)
        eq(np.array([getattr(row,f'unit_{c}') for c in cells]),unit,'date source coefficient normalization')
    for representation,g in date.groupby('representation'):
        av=[];bv=[]
        for strain,pair in g.groupby('strain'):
            pair=pair.sort_values('date');av.append(pair.iloc[0][[f'unit_{c}' for c in cells]].to_numpy(float));bv.append(pair.iloc[1][[f'unit_{c}' for c in cells]].to_numpy(float))
        av,bv=np.array(av),np.array(bv);m=(av+bv)/2
        same=np.sqrt(np.mean(np.sum((av-bv)**2,axis=1)))
        between=np.sqrt(np.mean([np.sum((m[i]-m[j])**2) for i in range(6) for j in range(i+1,6)]))
        row=dsummary[(dsummary.representation==representation)&dsummary.metric.eq('unit_13D')].iloc[0]
        eq(row.same_RMS,same,'six-date pair same RMS');eq(row.between_mean_profile_RMS,between,'six-date mean profile reference')
    result={'status':'PASS','checked_saved_splits':[0,99],'method':'independent per-cell source means/sample variances and fixed-template projection; direct pair loops; covariance eigenvectors plus scipy subspace_angles',
            'source_hashes_match':True,'all_metadata_labels_match':True,'max_absolute_error_by_check':errors,
            'maximum_absolute_error':max(errors.values())}
    (out/'verification/independent_numerical_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
