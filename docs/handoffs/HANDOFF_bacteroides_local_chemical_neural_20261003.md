# 交接：Bacteroides局部化学状态与ADF−ASH响应

更新日期：2026-10-03。实际工作仓库为 `/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis`。桌面任务当前显示的 `/Users/llllaaaa/.codex/worktrees/7470/bacteria_analysis` 不是本轮数据和结果所在目录。

**当前值得继续的结果：在29株Bacteroides内部，一个含14项注释的化学共变组与ADF−ASH相对配置有关。逐株、整菌种留出后的预测平方误差分别比训练均值基线低23.1%和20.0%。高化学状态的神经偏移较清楚，低、中状态接近；下一步建议解释并精简这个化学组合。**

本文件接续 [此前化学score预测交接](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/HANDOFF_chemical_score_neural_prediction_20261003.md)，覆盖其后的属内/属间比较、Bacteroides局部探索、神经可靠性检查及固定ADF−ASH模型。旧文档保留历史记录，当前入口以本文件和最新结果为准。本次只整理交接和后续建议，没有执行新的化学模型，也没有创建其他任务。

## 用户现在希望解决什么

用户希望找到相对明显、有连续变化支持、化学含义容易解释的局部关系，接受从样本较多的菌属内部入手。当前研究范围是Bacteroides的全部29株，不给只有2–3株的菌属另建模型，也不因某个菌种样本少而自动删掉其在这29株中的成员。

用户认可借鉴论文的思路：提取有明确含义的响应参数，再寻找简单的化学解释。当前资料缺少同一刺激的剂量梯度，化学报告也不是实际神经刺激液的定量测量，因此不照搬sensitivity或EC50。

最新明确偏好是“我们不要强调日期的影响”。后续把记录日期保留为元信息和已知测量限制，不把日期问题作为主结论、默认阻碍或下一个主要分析项目。主线继续放在化学状态的组成、ADF/ASH各自如何变化，以及简单解释能保留多少现有信号。

ADF−ASH是优先候选读出，完整13个神经元继续保留。用户没有要求寻找更多神经目标、扩展模型族或开展实验。本文末尾的下一任务内容是建议，尚未执行。

## 已完成的工作怎样收敛到这里

| 阶段 | 已得到的结果 | 结果入口 |
|---|---|---|
| 第1、2步：属内与属间并排比较 | 属内通常更近，但不同属的内部异质性不一。Bacteroides属内/属间距离中位数比，化学0.688、神经0.372；两种距离定义不同，不能横向比较比值大小。 | [属内/属间报告](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_genus_within_between_20261003/README.md) |
| 第3、4步：化学与神经分别探索属间规律 | 两边独立分析13个多株属、90株。化学有15个共变组；神经有属相关的组合方向。它们是各自的描述，尚不能据此建立跨模态机制。 | [独立分析报告](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_genus_patterns_independent_20261003/README.md) |
| Bacteroides内部主方向 | PC1、PC2分别解释33.60%、31.24%的神经方差，合计64.84%。PC1主要涉及AWB，PC2主要涉及ADF与ASH。预定PC1化学模型的完整13维留出改善为−2.92%、−4.33%。 | [初次局部模型](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_local_model_20261003/README.md) |
| 神经结构与重复性 | 二维平面对删株/删菌种较稳定，单PC1会明显旋转。动物分半支持优先检查ADF−ASH；没有由此认定只需要两个神经元。 | [神经可靠性报告](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_neural_reliability_20261003/README.md) |
| 固定ADF−ASH后的化学模型 | 主目标在查看新化学关联前固定，沿用原局部共变组方法，完整执行45个训练内选组的留出折，得到本交接的正向局部结果。 | [当前总报告](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/README.md) |

不同阶段的预测目标、队列和模型结构有变化，误差改善不能直接作同一任务的性能排名。旧三项score、旧70/36划分、旧M11、属间C组、初次PC1模型的L06，都不等于当前L03。

## 当前数据和模型定义

### 队列与神经目标

全部29株覆盖16个记录菌种。六株原分类备注均保留：A006、A025、A026、A040、A041、A044。标记意味着来源有待核对，不自动意味着坏样本。

主响应为：

```text
y = unit_ADF − unit_ASH
  = (a_ADF − a_ASH) / sqrt(sum(a_neuron² over all 13 neurons))
```

`a` 是保存的、有符号且经过SNR门槛处理的时间模板系数。主响应描述相对神经配置，分母涉及全部13个神经元。预定对照为未单位化的 `a_ADF − a_ASH`，另检查同模板pre-gate单位化差值。模板系数不是原始钙曲线，正负也不能直接翻译为兴奋/抑制。

固定神经列顺序为：ASK、ADL、ASI、AWA、AWB、ASG、ADF、ASH、ASJ、ASEL、ASER、AWCON、AWCOFF。当前模型没有重估共享时间模板，也没有对预测出的13维向量再次单位化。

此前动物分半时，完整13维支持每次为13–21株，中位数20株。ADF−ASH两半相关中位数0.862，同株/异株RMS比中位数0.406。100次划分高度重叠，不能当成100个独立实验。A001、A002、A005、A006、A007、A008、A010、A026从未满足该分半支持规则，但全部保留在当前29株化学模型中。支持不足不等于数据错误。

二维平面的删株、删菌种最大主角度中位数为2.87°、4.73°；较小动物半样本独立重建平面的最大主角度中位数为50.82°。两者样本支持和问题不同，不能相互替代。这也是优先采用固定可读参数、保留二维图作背景的原因。

### 化学输入与分组

化学沿用新报告筛选出的106×162完整面板，从中按菌株ID取29株。162项来自原380项注释中全106株均有有限正值、且重算QC RSD≤0.30的条目。报告单位为ng/mL，输入变换为 `log2(c / (1 ng/mL))`；不加1、不插补、不恢复旧reference或旧fold change。

每个训练集内执行同一套规则：

1. 每项按训练均值和样本SD标准化，SD≤1e−12的常量排除。
2. 用Pearson距离 `1−r`、average linkage和固定切割距离0.5分组。
3. 合格组至少含3项注释、3个Mass-column family。组合分数先在family内等权，再在family间等权。
4. 按固定主目标的训练Pearson r²选一个组，以截距和一条斜率拟合。精确平手按排序后的成员名称打破。

组分数不再除以自身SD，因此一单位分数不等于一SD。family用于降低报告注释冗余的权重，不代表已经确认的独立化合物身份。完整29株共得到12个候选；L03只是在这一拟合中的局部编号，跨折要比较实际成员，不能只看组号。

纯化学函数来自 [local_chemical_axes.py](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_local_model_20261003/chemical/code/local_chemical_axes.py)：`fit_axes(train_log_frame, feature_metadata)`、`transform_axes(heldout_log_frame, fitted)`。

## 可以交给下一任务的主要结果

### 获选化学状态与预测表现

全29株选中14项注释、13个family的L03。完整数据描述模型为：

```text
预测的相对ADF−ASH = 0.034077 + 0.212949 × chemical_state_score
```

Pearson r=0.5298，Spearman rho=0.6069，表观R²=0.2807。全数据相关来自择优选组，留出结果另行计算。

| 响应 | 逐株留出平方误差改善 | 整记录菌种留出平方误差改善 |
|---|---:|---:|
| 主目标：unit ADF−ASH | **23.09%** | **19.96%** |
| 未单位化模板系数ADF−ASH | 23.34% | 21.39% |
| 同模板pre-gate单位化ADF−ASH | 23.56% | 20.46% |
| unit ADF单独坐标 | 13.09% | 11.12% |
| unit ASH单独坐标 | 25.38% | 21.44% |
| 同一化学分数预测全部13坐标 | 4.16% | 3.55% |

改善统一为 `1 − SSE_model / SSE_training_mean_baseline`，按各折留出预测汇总。主目标RMSE分别为0.2932对基线0.3343、0.3016对基线0.3372。23.1%不是RMSE下降，也不是全部神经方差解释比例。

逐株29折、整记录菌种16折，每折均从训练化学数据重算尺度、相关、分组、成员、权重和主目标择优，然后预测留出株。辅助目标和所有13神经坐标沿用该折主目标选出的状态，各自只拟合截距和斜率。两个验证方案各覆盖29株，共58行留出预测。

25/29个逐株折、13/16个菌种折选中与全数据完全相同的14项成员。少数折会换到较大的其他组，已全部纳入误差统计，没有删掉不稳定折。没有训练折触发无合格组的均值预测兜底。

### 高状态的偏移比均匀连续梯度更清楚

在全数据选中轴上按化学分数、再按菌株ID排序，使用预定的10/10/9株分段：

| 化学状态 | 株数 / 记录菌种数 | unit ADF−ASH均值 | 中位数 |
|---|---:|---:|---:|
| 低 | 10 / 8 | −0.147 | −0.141 |
| 中 | 10 / 8 | −0.089 | −0.110 |
| 高 | 9 / 8 | +0.372 | +0.407 |

低、中接近，高状态偏移更明显，各段仍有个体重叠。切点只用化学排名，但所用化学轴已由全数据神经响应选出，因此分段属于已选关系的描述，不能当作独立验证，也没有证明阈值机制。

同一化学状态跨一个观测IQR时，拟合的unit ADF变化为+0.087、unit ASH为−0.114，差值合计+0.201。未单位化ADF/ASH分量的同轴描述斜率也一正一负；后者是补充分量核对，没有各自重新选组。这支持保留两个神经元分别展示的相对趋势，不能推出每株都如此或存在因果拮抗。完整13维结果表明，这条关系主要解释局部对比。

固定全数据化学轴后逐一删株，r为0.466–0.626；逐一删菌种，r为0.446–0.773，方向均未反转。删B. stercoris后斜率从0.213增至0.450，说明定量尺度受低端几株影响。此检查只针对固定轴，不替代重新选组的留出验证。10个有重复株的记录菌种内去均值，23株描述性r=0.637。

记录条件与重复测量的限制已有单独报告，后续按用户要求简要保留即可。现有模型是条件于固定神经表示的内部留出检查，尚无独立培养/神经实验确认，不作剂量、因果或行为意义结论。

### 14项的化学含义仍需整理

完整成员为：

| 便于查读的结构类别 | 报告中的注释名称 |
|---|---|
| 吲哚相关 | Indole-3-carboxylic acid；Indoxyl sulfate；Indole-3-carboxaldehyde |
| 酚/胺相关 | 3-Methoxytyramine；p-Cresol；2-Hydroxyphenethylamine |
| N-acetyl/formyl氨基酸衍生物 | N-Acetylmethionine；N-Formyl-methionine；N-Acetylphenylalanine；N-Acetylleucine |
| 核苷酸 | Adenosine-5′-diphosphate(ADP)；Adenosine 2′,3′-cyclic phosphate；Cytidine 5′-diphosphate(CDP)；2′-Deoxycytidine 5′-monophosphate(dCMP) |

这张表按名称作阅读分组，不是已执行的新模型候选，也不是已确认通路；精确名称与来源Class字段以[成员表](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/model/tables/selected_full_state_members.csv)为准。

Indole-3-carboxylic acid与Indole-3-carboxaldehyde共享一个Mass-column family，各占1/26权重；其余12项各占1/13。模型虽然只有一个分数，但化学组成仍混合多类注释。不能据目前的相关判断哪一项有独立作用。

## 下一任务建议：解释化学组成，再做有限精简

建议下一任务围绕一个具体问题：**这14项中，哪些只是共同变化的冗余标记，能否用少数有明确组成的成分或子组，保留当前ADF−ASH关系？** 重点是减少化学解释的含糊，不要求把预测分数继续提高。

可以先完成成员核对与分解：检查报告名称、Mass/RT/column、family归属和分类注释；在化学数据内描述哪些成员共同变化、哪些能代表其余成员。结合完整29株散点，检查高化学状态的变化由哪些成分共同支持，以及低、中段是否本来就难以区分。任何注释证据不足的条目都明确标出，不把同family直接当成同一分子，也不强行选出致效物。

随后提出少量精简候选。优先考虑基于化学类别或共变覆盖的代表选择、等权分数与预定类别的删组检查。保留当前完整14项作为参照，主目标继续固定，ADF、ASH和完整13神经元仍按同一状态显示。这里不预先决定某一项或某一类别必须入选，也不枚举所有单项、两两组合或为了使散点变平滑而筛菌株。

后续验证需要区分两种问题：

- **解释已发现的L03。** 固定这14项再做精简，结果可称为条件于既有组合的探索性解释。即使在同29株里重做交叉验证，也不能声称整个发现过程没有使用留出株，因为14项已通过全29株神经响应被选中。
- **检验精简方法的预测表现。** 若下一任务需要这个结论，外层仍用逐株/整菌种留出，每折从原162项开始重新分组、按固定ADF−ASH选状态，再执行提前写好的精简规则。凡依据神经响应选择子组、成分数或参数的步骤，都放在外层训练集内，必要时用内层验证。比较完整组与精简组在同一批外层留出预测上的误差，并保留实际成员变化。这个结果检验的是精简流程，仍不是最终固定名单的新样本确认。

有限检查后，如果少数成分仍能保留方向与相近预测增益，就把它们作为更简单的候选状态；如果精简后关系明显减弱，就保留较宽的共变状态解释，不继续扩大模型搜索。高状态偏移目前值得描述，但尚不足以指定阈值或启动非线性拟合。

建议交付：一张含14项身份与冗余关系的表、一张完整29株的化学成员与13神经元图、少量精简候选及选择理由。若实际执行预测比较，再附同口径的完整组/精简组留出表和参数。英文图中文字、完整个体支持和小型Notebook可调用函数沿用现有规范。

可直接用于下一任务的提示：

> 阅读本交接，继续Bacteroides 29株的局部解释。先核对当前14项化学组合的注释、冗余与类别，判断能否形成更简单、含义清楚的化学状态。保持相对ADF−ASH为主目标，未单位化差值作对照，保留全部29株及13神经元。重点解释高化学状态的神经偏移，不以日期分析为主线。先提出有限、可解释的精简规则，明确条件性解释与完整流程验证的区别；如比较预测能力，在训练折内完成所有相关选择。保留当前完整组合为参照，不扩大到全体化合物的无约束搜索，不因展示效果删株，不宣称因果或sensitivity。

## 从哪些文件接手

最新结果根目录记为R：

```text
R = /Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003
D = /Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/tables
Python = /Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/.pixi/envs/default/bin/python
```

| 文件 | 用途 |
|---|---|
| [当前总报告](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/README.md) | 结果、主图、解释范围 |
| [冻结方案](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/PROTOCOL.md) | 主目标、分组参数、45折流程；原文与模型快照保持一致 |
| [复核后澄清](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/REVIEW_NOTES.md) | 分段的条件性、原尺度分量核对、指标口径 |
| `R/model/tables/cohort_targets.csv` | 29株目标、全部13个unit坐标、原尺度ADF/ASH系数、菌种、日期、flags |
| `R/model/tables/raw_coefficients_29x13.csv` | 全部29株、13神经元的未单位化模板系数 |
| `R/model/tables/chemical_log2_29x162.csv`、`feature_metadata_162.csv` | 当前队列完整化学输入与注释 |
| `R/model/tables/selected_full_state_members.csv` | 14项原名、family、注释、均值/SD/权重 |
| `R/model/tables/selected_full_state_scores.csv` | 当前化学分数、目标、全数据拟合 |
| `R/model/tables/full_candidate_results.csv`、`full_candidate_scores.csv` | 全12候选，避免只读赢家 |
| `R/model/tables/heldout_predictions.csv`、`pooled_performance.csv` | 58行留出预测和所有目标的池化表现 |
| `R/model/parameters/full_cohort.json`及其余45个JSON | 每次训练名单、所有候选、成员、尺度、权重、选择及OLS参数 |
| `R/model/tables/fold_selection_stability.csv` | 逐折实际成员稳定性 |
| `R/diagnostics/tables/ordered_strains_and_thirds.csv` | 按化学分数排序的全部个体与10/10/9分段 |
| `R/diagnostics/tables/fixed_axis_associations.csv`、`unnormalized_adf_ash_descriptions.csv` | 同轴全部13坐标和未单位化ADF/ASH分量描述 |
| [可复用模型函数](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/model/code/fit_fixed_contrast.py) | `fit_selected_state`、`predict_state`、`fit_ols`、`run_analysis(repo_root, out=None)` |
| [现有Notebook只读单元](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/code/notebook_cells.py) | 读取保存结果和图，不重新拟合 |
| [独立复核记录](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/verification.json) | 数值、来源、图像与防覆盖检查 |

源输入为D中的 `fresh_chemical_log2.csv`、`fresh_feature_metadata.csv`、`neural_unit_coefficients.csv`、`neural_pre_gate_unit_coefficients.csv`、`sample_context.csv`，加上 `/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv`。模型目录已经保存按ID对齐的当前队列副本，可从这些小表开始，不必重读原始钙成像数据。

现有三张图均为PNG与SVG，图注在 `R/diagnostics/figures/CAPTIONS.md`：

- [化学关系与留出预测](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/diagnostics/figures/01_chemical_relationship_and_heldout.png)
- [ADF/ASH分量、分段与13坐标变化](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/diagnostics/figures/02_adf_ash_thirds_and_all13_effects.png)
- [全部29株、13神经元及菌种/日期](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_bacteroides_adf_ash_chemical_20261003/diagnostics/figures/03_chemical_ordered_all13_context.png)

## 执行与复核状态

独立复核从源输入重建了完整29株拟合及全部45个留出折，包括聚类、权重、主目标择优、16个响应拟合、预测和均值基线，最大数值差为4.83×10⁻¹³。诊断重建最大差为5.00×10⁻¹⁶，三张图均已检查。来源哈希、原方案快照、输入ID对齐、`预测ADF − 预测ASH = 预测主差值`恒等式及防覆盖检查通过。

当前已有足够保存结果供下一任务读取。研究问题和执行范围仍按用户当次要求推进，遵守[AGENTS.md](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/AGENTS.md)。如需重算，写入新的结果目录；原始数据、现有报告、Notebook和冻结方案保持不变。不要执行整个Notebook或回滚工作区已有修改。当前没有提交Git、创建PR或安装依赖。
