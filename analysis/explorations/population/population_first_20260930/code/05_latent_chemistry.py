"""Fixed rank-3/rank-5 countercheck, with a population basis learned per fold.

Keep 106 strains, all 13 neurons x 5 bins, and notebook's 380 unscaled log2FCs.
Split/purge strain IDs BEFORE animal centering: purged responses never enter a
training animal mean. A PCA captures variance, not demonstrated repeatability.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[1]; T=OUT/'tables'; L=OUT/'logs'
RANKS=[3,5]; PENALTIES=np.array([.001,.01,.1,1.,10.,100.])
MODELS=['fc380_reference','genus_reference']


def split(design,block):
    test=design.loc[design.date.eq(block)].copy()
    train=design.loc[~design.sample_id.isin(test.sample_id)].copy()
    assert not set(train.sample_id)&set(test.sample_id)
    assert not set(train.date)&set(test.date)
    return train,test


def strain_weights(design):
    return 1/design.groupby('sample_id').sample_id.transform('size').to_numpy()


def targets(animal,design):
    keys=pd.MultiIndex.from_frame(design[['sample_id','date']])
    keep=animal.index.droplevel('worm_key').isin(keys)
    a=animal.loc[keep]
    centered=a-a.groupby(level=['date','worm_key']).transform('mean')
    result=centered.groupby(level=['sample_id','date']).mean().reindex(keys).to_numpy()
    assert np.isfinite(result).all() and result.shape==(len(design),65)
    return result


def population_basis(train_y,weights):
    mean=weights@train_y/weights.sum()
    center=train_y-mean
    variance=weights@center**2/weights.sum()
    scale=np.repeat(np.maximum(np.sqrt(variance.reshape(13,5).mean(axis=1)),1e-8),5)
    scaled=center/scale
    eigenvalues,vectors=np.linalg.eigh((scaled*weights[:,None]).T@scaled/weights.sum())
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[order]; vectors=vectors[:,order[:5]]
    assert np.allclose(vectors.T@vectors,np.eye(5),atol=1e-10)
    return mean,scale,vectors,eigenvalues


def centered_design(chem,reference,genus,train,test,model):
    ids=train.sample_id
    r=reference.loc[:,reference.loc[ids].sum()>0]
    if model=='fc380_reference':x=pd.concat([r,chem/np.sqrt(380)],axis=1)
    else:x=pd.concat([r,genus.loc[:,genus.loc[ids].sum()>0]],axis=1)
    outputs=[]
    for d in [train,test]:
        a=x.loc[d.sample_id].reset_index(drop=True)
        a-=a.groupby(d.date.to_numpy()).transform('mean')
        outputs.append(a.to_numpy())
    return outputs


def predictions(x,y,weights,v,penalty):
    w=weights/weights.sum(); xm=w@x; ym=w@y
    a=(x-xm)*np.sqrt(w[:,None]); b=(y-ym)*np.sqrt(w[:,None])
    coefficient=a.T@np.linalg.solve(a@a.T+penalty*np.eye(len(a)),b)
    return (v-xm)@coefficient+ym


def main():
    animal=pd.read_parquet(T/'aligned_neural_animal_5bins.parquet')
    design=animal.index.to_frame(index=False)[['sample_id','date']].drop_duplicates().sort_values(['sample_id','date']).reset_index(drop=True)
    assert len(design)==112 and design.sample_id.nunique()==106
    chem=pd.read_csv(T/'aligned_chemical_log2fc_paired.csv',index_col=0)
    ref=pd.get_dummies(pd.read_csv(T/'aligned_chemical_reference_groups_paired.csv',index_col=0).reference_group,prefix='ref',dtype=float)
    genus=pd.get_dummies(pd.read_csv(T/'aligned_taxonomy_paired.csv',index_col=0).genus_clean,prefix='genus',dtype=float)
    assert chem.shape==(106,380) and np.isfinite(chem).all().all()
    output=[]; choices=[]; risks=[]; audits=[]; loadings=[]; basis_records=[]
    for held in sorted(design.date.unique()):
        train,test=split(design,held)
        inner=[]
        for val in sorted(train.date.unique()):
            it,iv=split(train,val); w=strain_weights(it)
            u=1/iv.sample_id.map(train.groupby('sample_id').size()).to_numpy()
            yi,yv=targets(animal,it),targets(animal,iv)
            mean,scale,basis,eigen=population_basis(yi,w)
            zi,zv=(yi-mean)/scale@basis,(yv-mean)/scale@basis
            for model in MODELS:
                x,v=centered_design(chem,ref,genus,it,iv,model)
                for penalty in PENALTIES:
                    pred=predictions(x,zi,w,v,penalty)
                    for rank in RANKS:
                        inner.append(dict(outer_block=held,inner_block=val,model=model,rank=rank,penalty=penalty,
                            loss=float(np.sum(u[:,None]*(zv[:,:rank]-pred[:,:rank])**2)),
                            null=float(np.sum(u[:,None]*zv[:,:rank]**2))))
            audits.append(dict(outer_block=held,inner_block=val,train_ids=';'.join(sorted(it.sample_id.unique())),test_ids=';'.join(sorted(iv.sample_id.unique())),
                               train_blocks=';'.join(sorted(it.date.unique())),test_blocks=val,n_train_strains=it.sample_id.nunique(),n_test_strains=iv.sample_id.nunique()))
        inner=pd.DataFrame(inner);risks.append(inner)
        risk=inner.groupby(['model','rank','penalty'])[['loss','null']].sum()
        risk['relative_loss']=risk.loss/risk.null
        yi,yv=targets(animal,train),targets(animal,test);w=strain_weights(train)
        u=1/test.sample_id.map(design.groupby('sample_id').size()).to_numpy()
        mean,scale,basis,eigen=population_basis(yi,w)
        zi,zv=(yi-mean)/scale@basis,(yv-mean)/scale@basis
        for rank in RANKS:
            basis_records.append(dict(outer_block=held,rank=rank,training_variance_fraction=float(eigen[:rank].sum()/eigen.sum()),n_train_strains=train.sample_id.nunique()))
        for j,feature in enumerate(animal.columns):
            for axis in range(5):loadings.append(dict(outer_block=held,feature=feature,axis=axis,loading=basis[j,axis],train_feature_mean=mean[j],train_neuron_scale=scale[j]))
        for model in MODELS:
            x,v=centered_design(chem,ref,genus,train,test,model)
            for rank in RANKS:
                penalty=float(risk.loc[(model,rank)].relative_loss.idxmin())
                choices.append(dict(outer_block=held,model=model,rank=rank,penalty=penalty))
                pred=predictions(x,zi[:,:rank],w,v,penalty)
                for i,row in enumerate(test.itertuples(index=False)):
                    for axis in range(rank):output.append(dict(outer_block=held,sample_id=row.sample_id,date=row.date,model=model,rank=rank,axis=axis,
                        observed=zv[i,axis],predicted=pred[i,axis],weight=u[i],penalty=penalty))
        audits.append(dict(outer_block=held,inner_block='outer',train_ids=';'.join(sorted(train.sample_id.unique())),test_ids=';'.join(sorted(test.sample_id.unique())),
                           train_blocks=';'.join(sorted(train.date.unique())),test_blocks=held,n_train_strains=train.sample_id.nunique(),n_test_strains=test.sample_id.nunique()))
    pred=pd.DataFrame(output);summaries=[];blocks=[]
    assert np.allclose(pred.groupby(['model','rank','axis','sample_id']).weight.sum(),1)
    for (model,rank),d in pred.groupby(['model','rank']):
        loss=np.sum(d.weight*(d.observed-d.predicted)**2); null=np.sum(d.weight*d.observed**2)
        summaries.append(dict(model=model,rank=rank,r2=1-loss/null,loss=loss,null=null,n_strains=d.sample_id.nunique(),n_blocks=d.date.nunique()))
        for block,z in d.groupby('date'):blocks.append(dict(model=model,rank=rank,block=block,r2=1-np.sum(z.weight*(z.observed-z.predicted)**2)/np.sum(z.weight*z.observed**2)))
    for name,frame in [('predictions',pred),('choices',pd.DataFrame(choices)),('inner_risks',pd.concat(risks)),('fold_audit',pd.DataFrame(audits)),
                       ('loadings',pd.DataFrame(loadings)),('training_basis',pd.DataFrame(basis_records)),('summary',pd.DataFrame(summaries)),('block_scores',pd.DataFrame(blocks))]:
        frame.to_csv(T/f'latent_chemistry_{name}.csv',index=False)
    parameters=dict(status='success',ranks=RANKS,penalties=PENALTIES.tolist(),models=MODELS,neural_features=65,chemical_features=380,n_strains=106,n_strain_blocks=112,
        basis='Training-only strain-weighted PCA, after fold-specific animal centering and per-neuron five-bin pooled training SD; rank fixed, not optimized',
        split='Whole acquisition block held out, purge all test strain IDs; repeat same process in inner CV before any neural centering, scaling, or PCA',
        chemical='All unstandardized notebook log2FC/sqrt(380) plus reference; no QC removal or imputation',
        reference_comparison='Genus+reference, same target coordinates and folds',
        x_centering='Each fold partition separately centered over its actual strain directory within each block; final ridge intercept fitted only on training',
        evaluation='Sum strain-equal weighted coordinate SSE/null across distinct fold-learned subspaces; zero coordinate predicts training feature mean; axes not aligned across folds',
        limitation='PCA retains variance, not independently validated repeatability. No comparison to a differently split neural-repeatability R2 establishes modality superiority.')
    (L/'latent_chemistry_methods.json').write_text(json.dumps(parameters,indent=2))
    findings='''# 降到训练内主要群体坐标，化学预测是否明显改善？\n\n该反证固定 rank 3 和 rank 5，不选择其中表现较好的 rank。所有 13 神经元及 5 个时间窗从起点进入，不设置 ADF/ASH 基座。每个内外层划分先清除测试菌株，再在允许的动物刺激集合内中心化并汇总到菌株–采集块；训练内计算神经元尺度和加权 PCA，随后仅用训练参数投影测试目标。化学保持完整 380 个 log₂FC 特征。每折只选一个跨全部保留坐标的 λ。\n\n观测为 112 个菌株–采集块均值，推广对象为本数据设计支持的新菌株/留出采集块；106 个菌株每个总权重为 1，重复块等分。折内坐标可能改变，不跨折命名或强行对齐。零预测指训练群体均值在该折坐标中的原点，R² 按各折子空间内的加权 SSE/null 汇总。\n\n| 模型 | 固定维数 | 外层 R² |\n| --- | --- | --- |\n'''
    for record in summaries:findings+=f"| {record['model']} | {record['rank']} | {record['r2']:.5f} |\n"
    findings+='''\n完整化学谱在两个固定低维子空间中仍只获得较小的汇总解释比例；这次直接降低群体目标维度未解决解释不足。该结果削弱了“只需去掉 65 维目标中的低方差维度，化学就能充分解释群体差异”的简单解释。**PCA 保留的是方差，不是独立实验已确认的可重复信号**，因此不能排除所有测量噪声、非线性、未测成分或跨培养批次失配的可能性。属+参考模型汇总表现较高，也不能据此认定菌属是神经差异的因果来源。\n\n这不是神经/化学谁整体更好的检验；不能把这里的新菌株留块预测 R² 与其他动物内/动物间重建 R² 直接比较。两个 rank 的选择已披露，无独立验证或显著性宣称。本检查完成后停止，没有继续挑选 rank、神经元、化学特征或分类器。\n'''
    (L/'latent_chemistry_findings.md').write_text(findings)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(basis_records).groupby('rank').training_variance_fraction.agg(['min','median','max']).to_string())


if __name__=='__main__':main()
