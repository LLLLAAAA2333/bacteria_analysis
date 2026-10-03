# 固定ADF–ASH对比与局部化学状态：数值模型

本目录执行根[固定方案](../PROTOCOL.md)中的模型分支。主响应事先固定为 **unit_ADF−unit_ASH**，保留全部29株Bacteroides、162项化学注释、16个记录species及六个taxonomy flags，未按动物重复覆盖排除任何菌株。没有重新选择神经方向、搜索单分子、改变聚类cut或扩展模型族。

在12个全29化学候选状态中，训练Pearson r²最大的状态为 **L03，14项报告注释、13个Mass-column families**。全29表观r=0.529798、r²=0.280686，拟合式为：

`predicted(unit_ADF − unit_ASH) = 0.034077 + 0.212949 × chemical_state_score`

这里x为成员训练z值的family等权组合，未再除以组合自身SD；一单位x不是一SD，也不是某个分子的浓度或剂量。L03是当前一次拟合的局部统计组合，不是已验证通路，不能靠跨折L编号认定成员相同。

## 全流程held-out结果

每折都只使用训练菌株重新计算化学均值/样本SD、相关/聚类/成员及family权重，再按固定主响应训练r²选一个状态并拟合截距和单斜率。29次删一株及16次留记录species均完成；每种方式各得到29个held-out预测。

表中“改善”均为 `1 − SSE_model / SSE_training_mean_baseline`，正值表示相对于对应训练均值基线的平方误差减少；不是普通样本内R²。

| 响应 | 删一株误差改善 | 留记录species误差改善 |
|---|---:|---:|
| 固定主响应：unit_ADF−unit_ASH | **23.09%** | **19.96%** |
| 原始系数ADF−ASH | 23.34% | 21.39% |
| pre-gate单位profile的ADF−ASH | 23.56% | 20.46% |
| unit_ADF单独响应 | 13.09% | 11.12% |
| unit_ASH单独响应 | 25.38% | 21.44% |
| 完整13坐标 | 4.16% | 3.55% |

主响应RMSE：删株模型 **0.29316**、训练均值基线 **0.33429**；留记录species模型 **0.30165**、基线 **0.33718**。分别有20/29、19/29株的主响应平方误差小于基线；全部菌株误差仍保留，不能把平均改善当成每株均有效。

原始对比、pre-gate对比及13个单位神经坐标**全部沿用该折按主响应选中的同一个化学状态**，没有各自重新选状态。13维结果是同一化学分数分别拟合13条截距/斜率后的合计误差，预测向量未重新单位化；它与之前“化学状态→PC1→神经向量”的模型结构不同。

单独ADF和ASH的全29斜率分别为+0.091975、−0.120974。这是各自线性拟合的方向；差值增加本身不能证明两者必然一增一减，更不意味着所有菌株都按相同剂量规律变化。连续性、rank thirds、元信息组成和极端点影响由同轮diagnostics分支检查，本目录不代替这些判断。

## 候选及成员敏感性

所有12个全29候选都在`tables/full_candidate_results.csv`中，含实际成员、训练r/r²、截距/斜率及选择标记；每个候选的29株分数也完整保存。相同r²以排序后的成员名字tuple打破平局，没有额外搜索规则。

full29所选14项为：Indole-3-carboxylic acid、Indoxyl sulfate、Indole-3-carboxaldehyde、3-Methoxytyramine、p-Cresol、N-Acetylmethionine、N-Formyl-methionine、N-Acetylphenylalanine、2-Hydroxyphenethylamine、N-Acetylleucine、Adenosine-5′-diphosphate(ADP)、Adenosine 2',3'-cyclic phosphate、Cytidine 5'-diphosphate（CDP）、2'-Deoxycytidine 5'-monophosphate(dCMP)。成员异质，不能仅凭个别注释将整个组命名为某条功能通路。

按实际成员集合衡量，25/29个删株fold和13/16个留species fold与full29完全一致，两种方案的成员Jaccard中位数均为1。但少数fold改变了组或成员：

| 删除对象 | 所选成员数 | 与full29所选组Jaccard |
|---|---:|---:|
| A021 | 59 | 0.141 |
| A022 | 11 | 0.786 |
| A023 | 53 | 0.117 |
| A041 | 55 | 0.131 |
| 记录B. fluxus | 13 | 0.929 |
| 记录B. kribbi | 57 | 0.092 |
| 记录B. stercoris | 56 | 0.094 |

未因这些变化移除fold或固定full29成员代替训练发现。全部fold的候选、成员、权重、训练分数和OLS参数保留。若一个训练集没有合格化学状态，则按方案预测各响应训练均值；本次45fold中这种fallback未发生，独立边界检查验证了其行为。

## 输入与代数检查

化学/单位神经表示/元信息来自`reports/exploration_chemical_pattern_direct_report_20261003/tables`的五个指定CSV；原始SNR-gated系数来自`reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv`。所有来源按唯一strain ID与固定13神经列对齐。

逐项核对 `raw_coefficients / ||raw_coefficients_13||₂ = saved_unit_profiles`，最大绝对差 **9.71×10⁻¹⁷**。因此主目标确为 `(raw_ADF−raw_ASH)/||raw_13||₂`，分母涉及全部13神经元。

全29及每折均用同一状态同时拟合ADF、ASH和固定对比，检查 `pred(unit_ADF)−pred(unit_ASH)=pred(primary contrast)`，最大绝对差 **2.22×10⁻¹⁶**。没有事后重新单位化预测13向量破坏该恒等式。

## 文件与调用

| 文件 | 内容 |
|---|---|
| `tables/cohort_targets.csv` | 29株元信息、flags、原始范数、主/aux目标、13单位坐标 |
| `tables/full_candidate_results.csv`、`full_candidate_scores.csv` | 全12候选及所有成员/分数 |
| `tables/selected_full_state_scores.csv` | 选中分数、完整目标、全29预测及基线，供diagnostics使用 |
| `tables/selected_full_state_members.csv` | 选中组元信息、训练均值/SD及权重 |
| `tables/full_selected_response_models.csv` | 同一状态对主/aux/13坐标的全部OLS参数 |
| `tables/heldout_predictions.csv` | 两种方案合计58行held-out预测及各目标训练均值基线 |
| `tables/heldout_target_errors.csv`、`heldout_vector_errors.csv` | 逐目标及13维完整误差 |
| `tables/pooled_performance.csv`、`fold_performance.csv` | pooled及逐fold指标 |
| `tables/full_fit_performance.csv` | 单独标明full_apparent的训练指标 |
| `tables/fold_selection_stability.csv`、`fold_selected_member_jaccard.csv` | 每折与full29、折间实际成员Jaccard |
| `tables/fold_selected_member_frequency.csv` | 分删除方案的162项成员入选频率 |
| `parameters/full_cohort.json`、其余45个JSON | 每次完整训练ID、化学参数/全部状态、选择及所有响应模型 |
| `fold_manifest.json`、`summary.json`、`manifest.json` | train/test映射、根报告汇总、来源/API/代码哈希 |

`pooled_performance.csv`中的标量目标RMSE为`sqrt(SSE/n)`。`unit_profile_13d`行的同名RMSE是每株预测向量L2误差的均方根，即`sqrt(total13D_SSE/n)`；另有明确的per-coordinate RMSE列`sqrt(total13D_SSE/(13n))`。误差改善由相同目标/维数的SSE相除。

`code/fit_fixed_contrast.py`提供Notebook可调用的`fit_selected_state`、`predict_state`、`fit_ols`和`run_analysis(repo_root, out=None)`。它只读取冻结的纯化学API，不修改旧文件。主动重算必须指定新的out目录；已有核心结果时拒绝覆盖。未创建或执行Notebook，也未安装依赖。

自身`code/verify_saved_model.py`从原始输入独立回放全部46次拟合（full29加45fold），用numpy最小二乘重建所有响应预测、基线与参数，核对所有候选择优、完整训练尺度/权重、51行汇总指标、1035个成员Jaccard、代数恒等式、源哈希、空状态fallback及防覆盖。结果 **PASS**，最大绝对差 **3.25×10⁻¹⁵**。其回放使用保存的成员集合；独立从头聚类的全45fold复核由同轮verification分支另外完成。

## 证据范围

当前预设候选族对固定ADF–ASH对比呈现正的held-out误差改善，但这不是独立确认研究。目标依据同一队列前序神经可靠性结果被优先考虑；此前全图谱估计的模板及神经表示保持冻结，未在fold内重新拟合。LOO与留species是重叠的样本级估计，也不能代替独立动物或培养重复。记录species/date和六个分类注释不保证分类或批次完全可控。当前结果不支持化合物效力、因果机制、行为意义或整个13维神经组合已被充分解释的结论。
