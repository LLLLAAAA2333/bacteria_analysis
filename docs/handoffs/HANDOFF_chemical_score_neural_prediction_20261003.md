# 交接：化学共同变化与神经组合预测

更新日期：2026-10-03。实际工作仓库为 `/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis`，本轮数据、脚本和结果都保存在这里。不要把桌面任务的 `/Users/llllaaaa/.codex/worktrees/c0eb/bacteria_analysis` 工作树误当成结果目录。

本文件接续 [原菌属与神经化学交接](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/HANDOFF_genus_neural_chemical_20261003.md)。原文件保留历史背景，其中旧 reference、旧化学距离和 `log2(value + 1)` 等方法不再是本轮入口。当前探索已完成正向预测、反向预测、精简神经输入及受限非线性比较；用户最后要求把这些工作写成交接文档，尚未指定新的分析任务。

## 当前结果可以说到什么程度

从原始化学报告重新筛选后，找到 Glucaric acid、Lumichrome、Vitamin B1 三项共同升降的候选，用一个化学共同得分 score 概括。化学共变清楚，完整神经组合的逐株对应较弱。用化学得分预测全部 13 类神经元，留出平方误差相对训练均值基线仅降低 2.1%。

反过来用神经组合预测化学得分，全部 13 类输入的误差降低为 12.5%；训练内筛出的 5 类输入为 23.3%。精简模型与全部 13 类的直接比较区间仍跨零，优势尚未稳定成立。随后在相同 5 类输入流程内加入两两交互或加性弯曲，留出表现均未改善。

这些结果支持保留化学 score 和候选神经组合用于描述与探索。现有证据尚不足以支持可靠的单株化学定量读出、普遍的属内规则或神经机制解释。以下所有留出结果都来自同一批已被多次查看的 36 株，不能写成新的独立验证。

## 用户确定的方向与展示偏好

用户最初希望从菌株远近及菌属背景寻找突破口，随后把问题收敛为：识别跨多个菌株重复出现的化学共同变化，并描述对应的神经组合变化。当前并未把原交接中的五步菌属问题逐项跑完，也未确定最终 Figure 5。

- 用户明确拒绝无法解释的 A050、A250、A306、ref12 旧化学分母，以及基于它们的旧 FC。本轮直接读原报告，重新计算化学面板、候选和得分。
- 保留全部 13 类神经元的组合视角，主图不回到挑选单个分子对应单个神经元的案例。
- 化学 score 作为多化合物共同变化的连续刻度，用户认为直观，可以保留。
- 主图要简单，以热图呈现。全部 13 类神经元保留，可把关联幅度较大的行放上面；这种排序不等于因果重要性。
- 训练／留出对比只作补充。单株热图看不清趋势后，已改为化学得分区间的观测均值主图，单株差异另行保留。
- 用户注意到反向预测趋近水平线，要求检查精简输入和有限的交互／弯曲扩展。这两轮比较均已完成，不继续自动扩大模型搜索。

遵守 [AGENTS.md](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/AGENTS.md)：研究者决定问题和执行范围，Notebook 优先、小函数、图中文字为英文。当前已有足够的保存结果供读取和作图，无需重跑模板、整个 Notebook 或完整探索流程。

## 数据从哪里来

当前结果根目录记为 `R`：

```text
R = /Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003
```

### 化学从原报告重新筛选

原始文件为 [metabolism_raw_data.xlsx](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/data/metabolism_raw_data.xlsx)，读取 `all` sheet 的 380 个化学注释、样本列与 `QC-1` 至 `QC-39`。当前神经面板对应的 106 株均可对齐，其 380 × 106 化学区域有 3,962 个缺失值。

先要求 106 株中均有有限正值，得到 173 项；再用 QC 可用观测重新计算 `sample SD(ddof=1) / mean`，要求至少两次 QC 且 RSD ≤ 0.30，最终保留 162 项。实际保留项各有 36 至 39 个 QC 值，重算 RSD 与报告值的差异约为浮点精度。162 是重新筛选的结果，没有导入旧面板名单。

报告单位已核对为 ng/mL。程序仅从同一工作簿的 `A_vs_B` 读取 `Name` 和 `unit` 元数据，没有用其中的对照比较结果。另行源文件审计确认 `A_vs_B`、`C_vs_D` 的样本／QC 数值与 `all` 按名称对齐后一致。不能继续沿用早期“单位未知”的说明。

变换为 `log2(reported concentration / (1 ng/mL))`，不加 1、不插补、不计算旧 FC、不按 reference 分组中心化。QC 审计及原缺失保存于 `R/tables/fresh_feature_audit.csv`、`chemical_report_all_380.csv`；新的完整化学面板在 `fresh_chemical_log2.csv`。

分类信息来自 [GM300_bacteria_species_summary.xlsx](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/data/GM300_bacteria_species_summary.xlsx) 的 `Axxx_species_mapping`。106 株覆盖 29 属，其中 13 属至少两株。菌属与记录日期用于描述，没有用于划分训练／留出、形成候选或筛除样本。

### 神经沿用已有模板表示

直接来源是 [strain_coefficients.csv](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv) 和 [condition_metrics.csv](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/tables/condition_metrics.csv)。本轮没有从钙成像记录重新拟合模板。

原列顺序为：ASK、ADL、ASI、AWA、AWB、ASG、ADF、ASH、ASJ、ASEL、ASER、AWCON、AWCOFF。每类一个 0–40 s、8 个时间 bin 的共享模板，系数有正负号。当前门槛为 individual SNR ≥ 0.5、每条件至少两只动物，动物等权、日期等权；未过门槛系数置零，真正缺失保留 NaN。当前 106 × 13 表全部有限。

每株完整系数向量除以其 L2 范数，得到 `R/tables/neural_unit_coefficients.csv`。这种表示保留相对组合与符号，去掉整组共同增益；数值不是神经元贡献百分比，符号也不能直接翻译为兴奋／抑制。后续精简输入只取其中部分坐标，没有重新归一化，分母仍由原 13 项构成，不能据此推断只测五类神经元就足够。

初次沿用旧 reference 的探索保存在 `reports/exploration_chemical_pattern_recurrence_20261003`，只作历史记录。其候选、指标和参考组不进入当前结论。当前来源哈希见 [source_manifest.json](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/source_manifest.json)。

## 化学候选和 score 怎样得到

用随机种子 `20261003`，对 106 株作不分层的 70／36 划分，完整名单保存在 [frozen_candidate.json](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/frozen_candidate.json)。后面所有比较均沿用这份名单。

在 70 株发现组内，以 `1 - |Spearman rho|` 作平均连接聚类，切割距离 0.30。候选至少包含三个化学注释、三个 Mass-column family，PC1 方差占比至少 0.50，共得到 11 个候选。化学 log 值按发现组均值和样本标准差标准化，再用 family 数量的平方根倒数加权、拟合 PC1，并把得分调整为发现组标准差 1。family 仅用于减少重复注释的权重，不代表经过验证的分子身份。

候选用发现组五折的完整 13 维神经预测误差排序。每折重拟合化学尺度、PC1 和回归，候选成员固定于外层 70 株的化学聚类。这个交叉验证用于选候选，不能解释为完整发现流程的独立验证。

选中 M11，包含三个正向共同变化的注释：

```text
score = 0.3631725848573887 × z(Glucaric acid)
      + 0.3669124361541364 × z(Lumichrome)
      + 0.3739966873278723 × z(Vitamin B1)
```

这里的 `z` 始终指 log 浓度按 70 株发现组的均值与标准差转换。三者跨三个 Mass-column family、两个色谱 column；发现组 PC1 方差占比 82.02%，两两 Spearman 中位数在发现组／留出组为 0.747／0.876，留出组最小值为 0.801。它们是共同变化的报告注释，尚不能称为同一条生化通路。

全 106 株的冻结得分在 [sample_scores.csv](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/tables/sample_scores.csv)，三种化学的训练尺度 z 值在 [selected_chemical_standardized.csv](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/tables/selected_chemical_standardized.csv)。主图另做全样本显示中心化，不会改变这里的预测目标。

## 正向与反向预测的结果不能混用

### 化学得分预测全部 13 项神经系数

对每个神经元拟合 `predicted_coefficient[j] = a[j] + b[j] × score`，用前 70 株拟合后固定参数。基线把每个留出菌株都预测为发现组平均 13 维神经向量。预测向量不再次单位化。

36 株、全部 13 项系数的平方误差和 SSE 从基线 **16.443948** 降至 **16.098634**，降低 **2.10%**。化学得分与固定神经投影的相关为 0.509，发现／留出斜率方向 cosine 为 0.803；这些指标均不能翻译成预测准确率或完整神经变异的解释比例。

部分方向重复出现：ADF、ASH、AWB 的系数随 score 增加而上升，AWCON 下降。ADF 两组斜率较接近，AWCON、ASH 在留出组减弱，多个较小分量反号。属内对应不一致：可描述的六属共 24 株，属内相关约 0.244，三个属为正；不能称为普遍的属内规律。相同模板上未置零系数的敏感性结果更弱，完整向量预测仅改善约 0.65%。

精确数值见 [正向 results.json](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/results.json)，方法和其他描述性检查见 [direct-report README](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/README.md)。

### 神经组合预测化学得分

反向问题为 `predicted_score = intercept + sum(weight[j] × neural_unit_coefficient[j])`，另行训练岭回归。岭回归仍是线性模型；它不是正向回归的数学逆，也不支持神经驱动化学的因果解释。

反向基线对每株都预测前 70 株的平均化学得分，其留出 SSE 为 **44.420863**。表中误差降低统一定义为 `1 - model_SSE / mean_baseline_SSE`，与正向 2.1% 的目标、分母不同，不能直接比大小。

| 反向模型 | 岭参数 alpha | 训练五折 SSE | 留出 SSE | 相对化学均值基线的误差降低 | 留出相关 r |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 13 类线性输入 | 100 | 58.399 | 38.858735 | 12.52% | 0.3766 |
| 精简 5 类线性输入 | 10 | 58.745 | 34.080247 | 23.28% | 0.4926 |
| 5 类输入加两两交互 | 100 | 53.636 | 37.433353 | 15.73% | 0.4386 |
| 5 类输入加性样条曲线 | 100 | 61.858 | 37.343184 | 15.93% | 0.4209 |

相对均值基线的条件 95% 区间，全部 13 类为 −4.72% 至 28.55%，精简 5 类为 5.55% 至 42.09%。两者各自对均值的区间，不能代替直接比较两个模型。

直接比较的分母改为对应参照模型的 SSE：

| 直接比较 | 误差降低 | 条件 95% 区间 | 单株误差较小的数量 |
| --- | ---: | --- | ---: |
| 精简 5 类相对全部 13 类 | 12.30% | −2.03% 至 26.72% | 23／36 |
| 交互相对精简线性 | −9.84% | −36.83% 至 9.23% | 13／36 |
| 加性曲线相对精简线性 | −9.57% | −24.43% 至 3.19% | 14／36 |

负值表示误差增加。区间跨零不证明模型等效，也不证明它们一定更差。单株数量只表示误差谁更小，不是准确率。精简模型的点估计更好，训练内部表现却与全部 13 类接近，优势尚未稳定。非线性比较中，训练数据选择了交互模型，这个优势没有延续到 36 株留出结果；保留这一记录，不能倒改成训练选择了线性模型。

最早的反向检查还用同一个 alpha 预测三项化学的标准化 log 值，误差降低分别为 Glucaric acid 7.82%、Lumichrome 14.30%、Vitamin B1 9.02%，三项合计为 10.70%。共同得分由这三项线性组合得到，不能把这些结果当作四次独立支持。

## 三轮反向比较的固定范围

所有输入尺度都在各训练折内拟合，岭参数网格固定为 `[0.01, 0.1, 1, 10, 100, 1000]`。五折使用 `KFold(shuffle=True, random_state=20261004)`，按池化验证 SSE 选参数。目标 score 保持原定义，不在每折变成不同的化学轴。

全部 13 类反向模型用 `StandardScaler + Ridge`。共同得分是主目标，三种成分作为补充目标共用主目标选出的 alpha。

精简比较只试 3 类或 5 类。每个训练折内按与 score 的绝对 Pearson 相关排序，常量列排后，精确平手保留原列序；筛选与标准化均只看该折训练株。精简模型联合选择输入数和 alpha，全 13 类对照单独选择 alpha。只把一个训练选中的精简模型带到 36 株，与全 13 类比较，没有根据多个精简候选的留出结果挑赢家。

最终五类为 **AWCON、ADF、ASH、AWB、ASER**。AWCON、ADF、ASH 在五个训练折均入选，AWB、ASER 各三次；平均两两 Jaccard 为 0.624。训练折高度重叠，这些频数只描述筛选的敏感性，不能解释成独立重复、生物学重要性或入选概率。

非线性比较固定历史选定的输入数 5，每折仍重新筛选，不把全 70 株选出的名单直接带进折内验证。只比较三种模型族，每族独立选一个 alpha：

| 模型族 | 固定变换 |
| --- | --- |
| 线性 | 五项输入标准化后接岭回归，复现上轮精简模型。 |
| 交互 | 标准化输入，保留五个主效应和十个两两乘积，扩展后再次按训练数据标准化，共 15 项；没有平方或更高阶项。 |
| 弯曲 | 每项输入作独立二次 B 样条，每项三个训练范围内等距节点，去掉隐含常数后每项三个基函数，共 15 项，再标准化接岭回归；没有跨神经元交互。 |

样条设置为 `SplineTransformer(n_knots=3, degree=2, knots='uniform', include_bias=False, extrapolation='linear')`。超出训练范围沿边界线性外推，没有根据留出值改节点或截断。最终留出中 ADF 有两株、ASER 有两株超出各自训练范围，记录见对应目录的 `input_range_audit.csv`。

这属于低复杂度加性样条岭回归，没有使用显式曲率惩罚的完整 GAM。GLM 本身也不自动加入特定神经元交互；对连续可负的 score，Gaussian identity 链接在相同惩罚下不会改变原线性形式。

非线性一轮只运行 18 个模型族／alpha 候选、90 个折拟合。协议先保存，训练选定的三个模型及模型族先冻结，再计算 36 株结果。没有因留出结果不佳扩大节点数、换筛选方法、重划分样本或继续尝试新模型。

所有反向区间均使用随机种子 `20261005`、5,000 次配对菌株重采样，每次对比较模型使用同一组 36 株索引，保持预测不变。它们只描述固定目标、训练模型和划分下的波动，没有计入训练／特征选择的不确定性、共享动物或菌属依赖，也未作为多重比较后的确认性显著性检验。

## 主图怎样组织

目前主图草稿为 [05_grouped_story.png](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/figures/05_grouped_story.png)。全 106 株按冻结化学 score 排序，等数量分为五组，株数为 22、21、21、21、21。每列是相同一组菌株的观测均值，上方三种化学、下方全部 13 类神经元。

- 神经均值减去该神经元在全部 106 株的平均 unit 系数。13 行共用一个色阶，不按各行标准差放大。
- 化学颜色是 log 浓度按全部 106 株均值和样本标准差计算的显示 z 值。这与预测目标的 70 株尺度区分清楚，显示变换不会重定义 score。
- 神经行按发现组正向斜率绝对值降序：AWCON、ADF、ASH、AWB、ASK、ASER、AWA、ASEL、ASG、ADL、ASI、ASJ、AWCOFF。
- 组间等宽表示株数接近，不表示化学得分距离相等。分组只依赖化学，没有用神经结果调边界或筛株，也没有拿拟合值代替观测均值。
- 化学渐变部分来自排序方式；平均减少了视觉杂乱，也隐藏了单株重叠和反向变化。主图只能讲平均趋势，不能把它当逐株一致性或新增预测证据。

分组、神经中心值、具体均值与显示参数分别存于 `R/figures/grouped_story_strain_groups.csv`、`grouped_story_neural_center.csv`、`grouped_story_chemical_means.csv`、`grouped_story_neural_means.csv`、`grouped_story_display_parameters.json`。主图与单株补充图均覆盖完整范围，但色阶限值不同，不能直接比较两张图的颜色深浅。

| 图 | 用途 |
| --- | --- |
| [05 分组均值热图](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/figures/05_grouped_story.png) | 当前主图草稿，完整 13 类神经元，描述全样本平均趋势。 |
| [05 全部单株热图](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/figures/05_grouped_story_individuals.png) | 保留同一排序及分组边界，展示全部 106 株的差异。 |
| [04 神经变化方向对照](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/figures/04_neural_pattern_summary.png) | 70／36 两组的 13 个斜率估计；连接线不是置信区间，仅作补充。 |
| [06 留出预测说明](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/figures/06_holdout_prediction_explained.png) | 解释正向预测的学习、冻结、预测、核对，以及 2.1% 的均值基线。 |
| [全部 13 类反向预测](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/reverse_prediction_20261003/reverse_prediction.png) | 化学共同得分实测／预测散点。 |
| [精简输入比较](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/reduced_input_comparison_20261003/reduced_input_comparison.png) | 同坐标比较全部 13 类和精简 5 类，保留配对区间。 |
| [三种模型预测对照](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/nonlinear_input_comparison_20261003/nonlinear_prediction_comparison.png) | 线性、交互、加性曲线的同坐标散点。 |
| [额外预测收益](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/nonlinear_input_comparison_20261003/nonlinear_paired_error_comparison.png) | 两种扩展直接相对精简线性的误差降低和配对区间。 |

上述图均有同名 SVG 和外置英文图注。早期 01／02 图、03 单株热图及排序前备份仍保留，主图入口优先使用 05。

## 接手时怎样读取和复现

Python 为 `/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/.pixi/envs/default/bin/python`，已有 numpy、pandas、scipy、matplotlib、scikit-learn；本轮使用的 scikit-learn 为 1.9.0，无需安装依赖。

[notebook_cells.py](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/code/notebook_cells.py) 已整理只读加载与显示单元，依次包含主图、补充图、反向、精简和非线性结果。它不会自动重跑训练。

| 数值入口 | 分析函数 |
| --- | --- |
| [正向结果](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/results.json) | `R/code/direct_report_recurrence.py`：`run_discovery(root, out)`、`run_holdout(root, out)`。 |
| [反向结果](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/reverse_prediction_20261003/results.json) | `R/code/decode_chemical_pattern.py`：`run_check(report, output)`。 |
| [精简结果](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/reduced_input_comparison_20261003/results.json) | `R/code/compare_reduced_neural_inputs.py`：`run_comparison(report, output)`。 |
| [非线性结果](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_chemical_pattern_direct_report_20261003/nonlinear_input_comparison_20261003/results.json) | `R/code/compare_nonlinear_neural_inputs.py`：`run_comparison(report, output)`。 |

各子目录保留冻结模型、训练 CV、逐株留出预测、参数与验证记录。精简和非线性目录另有 `protocol.json`、训练折成员和排序表；这些协议是本次运行前记录的范围，不是正式预注册研究。

仅修改图时用现有绘图函数，避免调用分析函数：`plot_grouped_story.make_figures(R)`、`plot_holdout_explainer.make_figure(R)`、`decode_chemical_pattern.make_prediction_figure(反向目录)`、`plot_reduced_input_comparison.make_figure(精简目录)`、`plot_nonlinear_input_comparison.make_figures(非线性目录)`。

重算分析应在用户提出新的执行要求后进行，并使用新输出目录。反向、精简、非线性函数拒绝覆盖已有目录，初始候选与留出函数也有冻结结果保护。表按菌株 ID 对齐，不能假定行序一致。

原始数据和旧结果未改。已检查来源哈希、重现旧线性预测，并由只读复核者从原输入独立重算 CV、预测和区间；非线性变换还用 NumPy／SciPy 重建，含样条越界延伸。相关记录保存在各目录的 `verification.json`、`independent_verification.json` 或 `focused_verification.json`。

## 继续解释时保留的边界

当前还不能确认预测有限主要来自样本量、化学与神经实验的匹配程度、神经表征丢失的信息，还是其他因素。化学材料来自独立培养，报告浓度不是实测神经刺激 aliquot 浓度；去掉旧 reference 也没有证明培养、介质、批次等影响不存在。

完整性筛选用过全部 106 株的化学可用性，神经共享模板沿用已有全图谱估计，候选及输入数已在历史探索中选择。因此不能把局部训练内无泄漏等同于整个流程的端到端独立验证。随机菌株划分还允许同一菌属出现在训练与留出中，未证明能推广到新菌属。

精简筛法按边际相关排名，可能丢掉单独相关很弱、只在组合中有信息的神经元。本次交互和弯曲的阴性结果只覆盖筛出的五维输入与规定的两种扩展。特征展开也改变了岭惩罚的结构，预测差异不能直接解释为某个生物学交互成立或不成立。

本轮没有得到继续复杂化模型的收益依据，也没有证明不存在神经化学对应。用户尚未授权重划分、换 score、再选化学候选、搜索更多模型、改变神经表示或重新拟合原始记录。接手后先按用户的新问题确定范围，不自动继续搜索。

本次交接仅新增本文件，没有执行新的科学分析或提交 Git。仓库已有未提交的 AGENTS.md、Notebook 02／03、HMDS 模块等修改，以及未跟踪的 reports、tests 和脚本；它们不是本次交接产生的待清理内容，不要回滚。主图仍是草稿，是否用于最终论文／poster 由用户决定。
