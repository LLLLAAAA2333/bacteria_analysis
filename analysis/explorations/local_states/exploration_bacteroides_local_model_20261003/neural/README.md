# Bacteroides 29株内部的主要神经变化

已保留全部29株和13类神经元，按预先指定的神经PC1提取局部变化方向。**PC1解释33.60%的属内unit-profile方差，PC2解释31.24%；两者接近，当前数据不支持一个明显占优、对删株始终稳定的单一轴。** 前两维合计64.84%，前四维合计87.93%。这些比例描述神经组合本身的变化，不是化学解释度。

本目录只处理神经侧；没有读取化学数值、据化学关系挑PC或拟合跨模态模型。

## 主要方向与完整方差谱

![PC1 direction and complete variance spectrum](figures/01_neural_pc1_and_variance.png)

对29×13的unit系数按神经元减去Bacteroides内均值，不按神经元SD标准化，然后做一次SVD。各PC符号固定为最大绝对loading的坐标为正。因此当前正PC1方向主要是**AWB相对系数升高，同时ADF、ASER、ASEL、AWA等相对系数下降**。PC2则主要表现为ADF升高、ASH降低（loading分别+0.705、−0.636），其方差份额几乎与PC1等大。这些是完整组合中的坐标变化；所有13类都保留在图和表中，不意味着AWB单独解释神经反应，更不意味着兴奋、抑制或分子敏感性。

| 神经元 | PC1 loading | PC2 loading |
|---|---:|---:|
| ASK | +0.110771 | -0.106583 |
| ADL | -0.009081 | +0.016291 |
| ASI | +0.033113 | -0.058992 |
| AWA | -0.127414 | -0.153547 |
| AWB | +0.924102 | +0.127194 |
| ASG | +0.016408 | +0.027493 |
| ADF | -0.251966 | +0.705031 |
| ASH | -0.100536 | -0.635531 |
| ASJ | +0.049015 | -0.134623 |
| ASEL | -0.131755 | +0.010781 |
| ASER | -0.145802 | +0.052085 |
| AWCON | -0.017987 | -0.056554 |
| AWCOFF | +0.033848 | +0.138801 |

每株PC1分数为 `(unit_profile − Bacteroides_mean) @ loading_PC1`。分数高低只有在这个中心与方向定义下有意义。PC1重建为`mean + score × loading`，重建后不重新单位化；剩余66.40%的总体平方变异仍在其他方向中。完整13个PC及分数均保存，PC1是本次预先固定的化学解释目标，而非已经证明的唯一生物变化轴。

## 全部菌株、菌种与完整记录日期

![All 29 strains ordered by PC1](figures/02_neural_strains_by_pc1.png)

行按神经PC1从低到高排列，颜色为unit系数减去属内均值，共用±0.6色阶。图中日期全部为2026年；多日期菌株显示完整集合，没有选其中一个日期。`*`保留源记录的taxonomy note，不据此删株。图中直接显示`B. species`名称；保存表仍提供[完整物种名称与记录映射](tables/species_mapping.csv)。

PC1低端的A048、A045、A040都记录于20260429；4株标记为B. thetaiotaomicron的菌株都在PC1正侧，且都记录于20260520。这些记录提示物种、菌株和日期的覆盖不独立；本轮未把日期当作PCA参数，也未把日期结构归因于物种或记录条件。日期只是记录日期标签，不代表已经确认的培养批次。

29株对应16个物种标签、5个日期、35个strain×date成员关系；6株有两个日期。6个源taxonomy flag为A006、A025、A026、A040、A041、A044，全部保留。已直接核对[分类原表](../../../data/GM300_bacteria_species_summary.xlsx)的`Axxx_species_mapping` sheet：上述记录的`note1`为“?”或“？”，A044为“？污染？”。这保留了原始疑问，不确认污染或分类错误；[字段证据及来源哈希](verification/taxonomy_text_provenance.json)已保存。物种标签的分组稳定性检查仍按当前记录标签进行。

## 单轴稳定性是当前主要限制

固定做29次leave-one-strain-out和16次leave-recorded-species-out，每次仅在训练子集重新估计神经均值和PCA。下面cosine为删样本后PC1与全29株PC1的绝对loading余弦；取绝对值只消除任意符号，并不把两个不同方向视作相同。全数据方向只用于比较与显示符号，不进入训练PCA。

| 检查 | 方向cosine中位数 | 最低cosine | 最低值对应删去的记录 |
|---|---:|---:|---|
| 留一株 | 0.9757 | 0.0924 | A048 |
| 留一物种标签 | 0.9496 | 0.2301 | B. salyersiae：A010、A048 |

另有：留出A001后cosine为0.277；留出B. vulgatus后为0.303；留出4株B. thetaiotaomicron后为0.506。PC1、PC2方差接近与这种方向不稳定相符：样本集合变化后，第一方向可能明显转动。高的中位数不能掩盖这些低值。没有因其影响大而删除A048等记录，也没有选择某个删株结果作为主轴。

这使后续化学解释必须同时考虑“解释了什么固定方向”与“该方向是否稳定”。全数据PC1分数可作观察性描述；预测评估若需要学习神经方向，应在训练折重算均值与方向，并说明如何处理轴的不稳定。这里的稳定性结果本身不属于化学预测性能。

## 同模板pre-gate检查与幅度描述

对同29株pre-gate unit表示做同样PCA：PC1解释32.62%，与主PC1的方向余弦0.9602，对齐符号后逐株score Pearson为0.9518。这个固定的gate检查显示平均方向相近，但不能消除删株/删种的方向不稳定。它沿用相同神经模板，没有重新拟合原始曲线。

原始已gate系数的L2范数`coefficient_norm`原样保存为幅度描述；与原有norm快照最大差2.22×10⁻¹⁶。它不替代主unit表示。unit已经去掉整体gain，因此此分析的“升降”是组合坐标的相对变化，不是整体响应强弱，也不是实际刺激浓度改变时的敏感性。

神经系数依赖全图谱已有共享模板，动物、trial和记录日期存在重复结构。本轮是固定已保存表示上的内部检查，不是新动物、新培养或新菌株的独立实验验证。

## 文件与复现

- [预先方案](PROTOCOL.md)、[来源SHA-256](source_manifest.json)、[参数与精确顺序](parameters.json)、[图参数](plot_parameters.json)。
- [逐株全部PC分数及元数据](tables/strain_scores.csv)、[全部13个loading向量](tables/neural_loadings.csv)、[完整方差谱](tables/pca_spectrum.csv)、[属内均值](tables/mean_unit_profile.csv)。
- [PC1重建](tables/pc1_reconstructed_profiles.csv)、[残差](tables/pc1_residual_profiles.csv)、[原unit系数](tables/neural_unit_profiles.csv)。
- [45次稳定性检查](tables/pc1_stability.csv)、[各次对齐loading](tables/stability_aligned_loadings.csv)、[留出株训练中心投影](tables/heldout_pc1_projections.csv)。
- [pre-gate方向与score检查](tables/pre_gate_sensitivity_summary.csv)、[35个完整日期成员关系](tables/strain_date_membership.csv)、[物种代码映射](tables/species_mapping.csv)。
- [数值独立核验](verification/numerical_verification.json)、[视觉检查](verification/visual_review.md)。

默认在现有Notebook中只读加载，下面调用不拟合、不写文件：

```python
import importlib.util
from pathlib import Path
repo = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
report = repo / 'reports/exploration_bacteroides_local_model_20261003/neural'
spec = importlib.util.spec_from_file_location('bacteroides_neural', report / 'code/local_neural.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.load_saved_results(report)
result['spectrum']
```

仅需精确重算时，使用 `module.run_analysis(repo, out=repo / 'reports/bacteroides_neural_recompute_new')`，目标必须是没有保存科学结果的新目录。函数拒绝覆盖已有tables或核心metadata。只重画时使用`module.draw_figures(report, fresh_figure_directory)`，同样拒绝覆盖已存在的图，不重算统计。

独立核验使用协方差矩阵的eigendecomposition重新获得PCA，未调用主分析函数；并核对所有13个loading、29株全部scores、重建/残差、45次稳定性、pre-gate以及物种/完整日期/flag对齐。全部通过，最大绝对数值差4.45×10⁻¹⁴。两张PNG均已打开视觉检查，SVG同时保留。
