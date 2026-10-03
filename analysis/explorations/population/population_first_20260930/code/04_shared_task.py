"""One fixed, shared taxonomy task for neural and chemical representations.

Observation: a strain x acquisition block, neural mean over animals in that block.
Target: genus, fixed eligibility >=5 unique strains and >=3 blocks, irrespective
of model performance. Outer and inner CV hold out complete blocks and purge every
validation strain ID from training. No animals or held-out strains cross a fold.
Eight-class balanced ridge classifier with the same five penalty candidates.
This tests strain-level taxonomy generalization, not behavioral utility/causality.
"""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
TABLES, LOGS, FIGURES = OUT/'tables', OUT/'logs', OUT/'figures'
LAMBDAS = np.array([1e-4, 10**-2.5, .1, 10**.5, 100.])
MODELS = ['reference_only', 'chemical_log2fc', 'neural_raw', 'neural_cell_scaled']
NEURONS = ['ASK','ADL','ASI','AWA','AWB','ASG','ADF','ASH','ASJ','ASEL','ASER','AWCON','AWCOFF']


def markdown_table(data, digits=4):
    """Small report table without an optional formatting dependency."""
    def fmt(value):
        return f'{value:.{digits}f}' if isinstance(value, (float,np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(map(str,data.columns))+' |',
                      '| '+' | '.join(['---']*len(data.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in data.itertuples(index=False,name=None)])


def strain_weights(data):
    """Each strain has total weight one, split over its observed blocks."""
    return 1 / data.groupby('sample_id').sample_id.transform('size').to_numpy()


def training_weights(data):
    """Each genus has equal total weight; within genus each strain is equal."""
    w = strain_weights(data)
    counts = data.drop_duplicates('sample_id').genus.value_counts()
    w /= data.genus.map(counts).to_numpy()
    return w / w.sum()


def score_predictions(data):
    w = strain_weights(data)
    temp = data.assign(weight=w, success=w*data.correct.to_numpy())
    recalls = temp.groupby('genus').success.sum() / temp.groupby('genus').weight.sum()
    return float(recalls.mean()), recalls


def fit_predict(x, y, data, xtest, penalty, model):
    w = training_weights(data)
    x_mean, y_mean = w @ x, w @ y
    center = x - x_mean
    scale = np.ones(x.shape[1])
    if model == 'neural_cell_scaled':
        # Same scalar for the five bins of a neuron. Estimated only on training
        # strain/block means, without using held-out means or labels.
        variance = w @ center**2
        cell_scale = np.sqrt(variance.reshape(13,5).mean(axis=1))
        scale = np.repeat(np.maximum(cell_scale, 1e-12),5)
    a = center / scale / np.sqrt(x.shape[1])
    atest = (xtest - x_mean) / scale / np.sqrt(x.shape[1])
    aw = a * np.sqrt(w[:,None])
    yw = (y-y_mean) * np.sqrt(w[:,None])
    coefficient = aw.T @ np.linalg.solve(aw@aw.T + penalty*np.eye(len(aw)), yw)
    return atest@coefficient + y_mean, scale


def split(data, block):
    valid = np.flatnonzero(data.date.eq(block))
    validation_ids = set(data.iloc[valid].sample_id)
    train = np.flatnonzero(~data.date.eq(block) & ~data.sample_id.isin(validation_ids))
    assert not set(data.iloc[train].sample_id) & validation_ids
    assert block not in set(data.iloc[train].date)
    return train, valid


def main():
    for path in [TABLES,LOGS,FIGURES]:
        path.mkdir(parents=True,exist_ok=True)
    animal_path = TABLES/'aligned_neural_animal_5bins.parquet'
    chemical_path = TABLES/'aligned_chemical_log2fc_paired.csv'
    taxonomy_path = TABLES/'aligned_taxonomy_paired.csv'
    reference_path = TABLES/'aligned_chemical_reference_groups_paired.csv'
    animal = pd.read_parquet(animal_path)
    neural = animal.groupby(level=['sample_id','date']).mean().sort_index()
    chemical = pd.read_csv(chemical_path,index_col=0)
    taxonomy = pd.read_csv(taxonomy_path,index_col=0)
    references = pd.read_csv(reference_path,index_col=0)
    design = neural.index.to_frame(index=False)
    design['genus'] = design.sample_id.map(taxonomy.genus_clean)
    coverage = design.groupby('genus').agg(n_strains=('sample_id','nunique'),n_blocks=('date','nunique'))
    coverage['eligible'] = coverage.n_strains.ge(5)&coverage.n_blocks.ge(3)
    coverage.to_csv(TABLES/'shared_task_genus_coverage.csv')
    design = design.loc[design.genus.isin(coverage.index[coverage.eligible])].reset_index(drop=True)
    assert design.sample_id.nunique() == 74 and design.genus.nunique() == 8
    features = [f'{n}__{a:02d}_{a+5:02d}' for n in NEURONS for a in range(0,25,5)]
    neural_values = neural.loc[pd.MultiIndex.from_frame(design[['sample_id','date']]),features].to_numpy()
    chemical_values = chemical.loc[design.sample_id].to_numpy()
    assert np.isfinite(neural_values).all() and np.isfinite(chemical_values).all()
    labels = sorted(design.genus.unique())
    code = dict(zip(labels, range(len(labels))))
    y_code = design.genus.map(code).to_numpy()
    y = np.eye(len(labels))[y_code]
    # Known metadata categories only, without response-based feature selection.
    # An unseen-in-training reference dummy has zero coefficient automatically.
    reference_values = pd.get_dummies(references.loc[design.sample_id,'reference_group'],dtype=float).to_numpy()
    x_models = {'neural_raw':neural_values,'neural_cell_scaled':neural_values,
                'chemical_log2fc':chemical_values,'reference_only':reference_values}
    predictions, choices, fold_records, scaling_records = [],[],[],[]
    for block in sorted(design.date.unique()):
        train, test = split(design,block)
        train_design = design.iloc[train].reset_index(drop=True)
        test_design = design.iloc[test].reset_index(drop=True)
        train_animals = animal.index.to_frame(index=False).query('date != @block')
        train_animals = train_animals.loc[train_animals.sample_id.isin(set(train_design.sample_id))]
        test_animals = animal.index.to_frame(index=False).query('date == @block')
        test_animals = test_animals.loc[test_animals.sample_id.isin(set(test_design.sample_id))]
        ta = set(zip(train_animals.date,train_animals.worm_key))
        va = set(zip(test_animals.date,test_animals.worm_key))
        assert not ta&va
        fold_records.append(dict(outer_block=block,level='outer',inner_block='',
            train_strains=train_design.sample_id.nunique(),test_strains=test_design.sample_id.nunique(),
            train_blocks=train_design.date.nunique(),test_blocks=1,
            train_animals=len(ta),test_animals=len(va),shared_strains=0,shared_animals=0,
            train_ids=';'.join(sorted(train_design.sample_id.unique())),
            test_ids=';'.join(sorted(test_design.sample_id.unique())),
            train_genera=';'.join(sorted(train_design.genus.unique()))))
        inner_folds = []
        for inner_block in sorted(train_design.date.unique()):
            ti,vi=split(train_design,inner_block)
            inner_folds.append((ti,vi))
            fold_records.append(dict(outer_block=block,level='inner',inner_block=inner_block,
                train_strains=train_design.iloc[ti].sample_id.nunique(),test_strains=train_design.iloc[vi].sample_id.nunique(),
                train_blocks=train_design.iloc[ti].date.nunique(),test_blocks=1,
                shared_strains=0,shared_animals=0,
                train_ids=';'.join(sorted(train_design.iloc[ti].sample_id.unique())),
                test_ids=';'.join(sorted(train_design.iloc[vi].sample_id.unique())),
                train_genera=';'.join(sorted(train_design.iloc[ti].genus.unique()))))
        for model in MODELS:
            x = x_models[model]
            scores=[]
            for penalty in LAMBDAS:
                rows=[]
                for ti,vi in inner_folds:
                    pred,_=fit_predict(x[train[ti]],y[train[ti]],train_design.iloc[ti],x[train[vi]],penalty,model)
                    temporary=train_design.iloc[vi].copy()
                    temporary['correct']=pred.argmax(axis=1)==y_code[train[vi]]
                    rows.append(temporary)
                score,_=score_predictions(pd.concat(rows,ignore_index=True))
                scores.append(score)
            # Fixed ties favor greater shrinkage; no held-out outcomes used.
            best = np.flatnonzero(np.isclose(scores,max(scores),rtol=0,atol=1e-12))[-1]
            penalty = LAMBDAS[best]
            scores_test,scale=fit_predict(x[train],y[train],train_design,x[test],penalty,model)
            for index, candidate in enumerate(LAMBDAS):
                choices.append(dict(model=model,outer_block=block,penalty=candidate,inner_macro_recall=scores[index],selected=index==best))
            if model=='neural_cell_scaled':
                for n,s in zip(NEURONS,scale.reshape(13,5)[:,0]):
                    scaling_records.append(dict(outer_block=block,neuron_class=n,training_scale=s))
            predicted=scores_test.argmax(axis=1)
            for j,(_,row) in enumerate(test_design.iterrows()):
                prediction=dict(model=model,outer_block=block,**row.to_dict(),
                                predicted_genus=labels[predicted[j]],correct=bool(predicted[j]==y_code[test[j]]),penalty=penalty)
                prediction.update({f'score__{label}':float(scores_test[j,k]) for k,label in enumerate(labels)})
                predictions.append(prediction)
    pred=pd.DataFrame(predictions)
    pred['strain_weight']=0.0
    summaries,per_genus,drop_records=[],[],[]
    for model,rows in pred.groupby('model',sort=False):
        pred.loc[rows.index,'strain_weight']=strain_weights(rows)
        accuracy,recalls=score_predictions(rows)
        for genus,recall in recalls.items():
            per_genus.append(dict(model=model,genus=genus,recall=recall,
                                  n_strains=int(coverage.loc[genus,'n_strains']),n_blocks=int(coverage.loc[genus,'n_blocks'])))
        drop=[]
        for omitted in sorted(rows.date.unique()):
            subset=rows.loc[~rows.date.eq(omitted)]
            drop_score,_=score_predictions(subset)
            drop.append(drop_score)
            drop_records.append(dict(model=model,omitted_block=omitted,macro_recall=drop_score))
        summaries.append(dict(model=model,macro_recall=accuracy,
            accuracy_equal_strain=float(np.average(rows.correct,weights=strain_weights(rows))),
            n_strains=rows.sample_id.nunique(),n_genus=rows.genus.nunique(),n_blocks=rows.date.nunique(),
            n_strain_blocks=len(rows),leave_one_block_out_min=min(drop),leave_one_block_out_max=max(drop)))
    pred.to_csv(TABLES/'shared_task_predictions.csv',index=False)
    pd.DataFrame(choices).to_csv(TABLES/'shared_task_inner_choices.csv',index=False)
    pd.DataFrame(fold_records).to_csv(TABLES/'shared_task_fold_audit.csv',index=False)
    pd.DataFrame(scaling_records).to_csv(TABLES/'shared_task_training_scales.csv',index=False)
    pd.DataFrame(drop_records).to_csv(TABLES/'shared_task_drop_block_scores.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(TABLES/'shared_task_summary.csv',index=False)
    genus_result=pd.DataFrame(per_genus)
    genus_result.to_csv(TABLES/'shared_task_genus_results.csv',index=False)
    for model,rows in pred.groupby('model'):
        confusion=pd.crosstab(rows.genus,rows.predicted_genus,values=rows.strain_weight,aggfunc='sum').reindex(index=labels,columns=labels).fillna(0)
        confusion.to_csv(TABLES/f'shared_task_confusion_{model}.csv')
    comparisons=[]
    for first,second in [('neural_raw','chemical_log2fc'),('neural_cell_scaled','chemical_log2fc'),
                         ('neural_cell_scaled','neural_raw'),('chemical_log2fc','reference_only')]:
        a=pred.loc[pred.model.eq(first)].sort_values(['sample_id','date'])
        b=pred.loc[pred.model.eq(second)].sort_values(['sample_id','date'])
        assert np.array_equal(a[['sample_id','date']],b[['sample_id','date']])
        difference=summary.set_index('model').loc[first,'macro_recall']-summary.set_index('model').loc[second,'macro_recall']
        drops=pd.DataFrame(drop_records).pivot(index='omitted_block',columns='model',values='macro_recall')
        comparisons.append(dict(first=first,second=second,macro_recall_difference=difference,
            first_only_correct=int((a.correct.to_numpy()&~b.correct.to_numpy()).sum()),
            second_only_correct=int((~a.correct.to_numpy()&b.correct.to_numpy()).sum()),
            difference_drop_block_min=float((drops[first]-drops[second]).min()),
            difference_drop_block_max=float((drops[first]-drops[second]).max())))
    pd.DataFrame(comparisons).to_csv(TABLES/'shared_task_comparisons.csv',index=False)

    fig,ax=plt.subplots(figsize=(7.0,4.2))
    display={'reference_only':'Reference identity\n4 metadata categories','chemical_log2fc':'Chemical log₂FC\n380 features',
             'neural_raw':'Neural population\n13 neurons × 5 bins','neural_cell_scaled':'Neural population\ntraining-scaled neurons'}
    colors={'reference_only':'#888888','chemical_log2fc':'#7C597D','neural_raw':'#227A91','neural_cell_scaled':'#4D9573'}
    summary=summary.set_index('model')
    for position,model in enumerate(MODELS):
        point=summary.loc[model]
        lo,hi=point.leave_one_block_out_min,point.leave_one_block_out_max
        ax.hlines(position,lo,hi,color=colors[model],lw=3,alpha=.6)
        ax.scatter(point.macro_recall,position,c=colors[model],s=65,zorder=3)
        ax.text(point.macro_recall+.024,position+.14,f'{point.macro_recall:.1%}',fontsize=10,color=colors[model])
    ax.axvline(1/len(labels),color='0.55',ls='--',lw=1)
    ax.text(1/len(labels)+.01,3.43,'Uniform chance: 12.5%',fontsize=8,color='0.4')
    ax.set(yticks=range(len(MODELS)),yticklabels=[display[m] for m in MODELS],
           xlabel='Genus-balanced recall on held-out strains',xlim=(0,1),ylim=(-.45,3.65))
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Same target and held-out samples: 74 strains, 8 genera',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.text(.5,.015,'Lines: range after omitting one held-out block; not confidence intervals.',ha='center',fontsize=8,color='0.35')
    for suffix in ['png','pdf']:
        fig.savefig(FIGURES/f'shared_task_taxonomy.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    inputs=[animal_path,chemical_path,taxonomy_path,reference_path]
    parameters=dict(eligibility={'minimum_strains':5,'minimum_acquisition_blocks':3},
        n_classes=len(labels),classes=labels,penalty_candidates=LAMBDAS.tolist(),
        outer_split='Leave one complete acquisition block out; purge its strain IDs from all training blocks',
        inner_split='Same leave-block and strain-ID purge on each outer training set',
        neural_transform='Feature centering for ridge intercept; raw or training-only per-neuron pooled residual SD; divide sqrt(65)',
        chemical_transform='Unstandardized log2FC; feature centering for ridge intercept; divide sqrt(380)',
        reference_transform='Four known metadata categories one-hot encoded; centered ridge intercept, divide sqrt(4); unseen training category coefficient equals zero',
        training_weights='Genus balanced, strain equal within genus, acquisition blocks equal within strain',
        evaluation_weights='Genus balanced, strain equal within genus, acquisition blocks equal within strain',
        missing_values='None in included strain-block means; no imputation',
        uncertainty='Range after dropping each held-out acquisition block from fixed predictions; descriptive sensitivity, not confidence intervals or refits',
        input_manifest=[{'path':str(p.relative_to(OUT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs])
    (LOGS/'shared_task_parameters.json').write_text(json.dumps(parameters,indent=2,ensure_ascii=False))
    findings='''# 同一菌属任务的群体神经与化学表征比较\n\n此辅助任务不是本轮科学主线，也不是神经或化学测量“总体好坏”的判定。菌属是独立元数据中的外部标签，但并非行为、感知或神经功能真值。\n\n固定选择至少 5 株、至少 3 个采集块的菌属，不按模型表现挑选，得到 8 属、74 株、80 个菌株–采集块观测。神经输入为每次采集中动物等权的 13×5 原始 ΔF/F₀ 均值；化学输入为相同菌株完整 380 维 log₂FC。二者使用完全相同的目标、训练/测试行与 5 个惩罚候选的岭分类器。神经原单位及训练内按神经元缩放并列；化学保持原 log₂FC 尺度，不 z-score。\n\n每次整块留出，随后从所有训练块中清除所有测试菌株。超参数选择只在外层训练内再次执行同样的留块与菌株清除。没有将单个动物或重复菌株分给同一折的训练与测试。每菌株评估总权重为 1，跨块重复按块等权；最终每属等权汇总。八分类均匀机会水平 12.5%，但机会线不是显著性检验。\n\n## 实际结果\n\n'''
    findings+=markdown_table(summary.reset_index())+'\n\n'
    findings+='增加的必要反证保持同一划分、权重及调参：仅用 4 个 medium reference 数值组标签进行分类，其宏召回率为 '
    findings+=f"{summary.loc['reference_only','macro_recall']:.2%}；完整化学谱为 {summary.loc['chemical_log2fc','macro_recall']:.2%}。因此仅凭参考组标签不能复现完整化学谱的菌属辨别，但这不排除培养条件与化学组成的其他共同变化。\n\n"
    findings+='逐属结果：\n\n'+markdown_table(genus_result.pivot(index='genus',columns='model',values='recall').reset_index(),3)+'\n\n'
    findings+='''图中点为宏平均召回率；横线为固定预测中依次删除一个留出块后的范围，非置信区间，不包含重新训练不确定性。无假设检验或 FDR 宣称。`shared_task_fold_audit.csv` 逐折保存测试/训练菌株ID。\n\n局限：74 株中各属仅 5–29 株，属在块间不均匀；9 个外层块不提供精确泛化误差。不同属、培养基参考和其他菌株条件可共同影响化学分类；神经测量噪声、动物数、缺失动物内细胞和刺激序列亦影响神经均值。本任务没有神经相对化学的先验胜出要求，也不把高菌属分类率等同于更能预测神经差异。\n\n已实际完成全部外层/内层拟合、折内无菌株及动物交叉断言、相同测试行验证、权重与逐属汇总。没有尝试其他分类器、其他菌属门槛或挑选神经元来提高表现。\n'''
    (LOGS/'shared_task_findings.md').write_text(findings)
    print(summary.to_string())
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__':
    main()
