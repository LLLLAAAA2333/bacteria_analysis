# 交接：从菌属内与菌属间差异探索化学和神经组合的关系

更新日期：2026-10-03。仓库：`/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis`。

本轮只整理交接文档。用户将在下一个 session 沿下面的思路探索，目前尚未执行新的属内／属间分析，也未确定 Figure 5。前几个 poster 图的表示、重复性、RDM 和 embedding 沿用已有结果，技术细节见 [前一份交接文档](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/HANDOFF_poster_20261001.md)。

## 1. 用户确定的新问题

以下五步保留用户原文：

1. 同菌属间的化学差异大吗
2. 同菌属间的神经差异大吗
3. 如果上两点符合同菌属间差异一般比较小，那么不同菌属间化学差异有什么规律
4. 如果3符合化学差异确实有一定潜在规律，神经差异有规律吗
5. 如果3 4 都成立，则可以建立神经差异对应的意义

用户关心的是怎样通过绘图发现可以描述、反复出现的规律，尤其是完整神经元组合的变化。用户不希望用报告措辞代替数据探索，也不希望继续围绕少数 example 或单个化合物对应单个神经元组织主图。化合物共同变化、散点分散和报告缺值，使后一种解释难以成立。

当前工作属于 Figure 5 的探索材料。是否能形成主图、主图表达什么，以及是否进入下一步分析，仍由用户决定。遵守 [AGENTS.md](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/AGENTS.md)：Notebook 优先、使用小函数、图中文字为英文、保留单位和证据边界，不自行启动整套后续分析。

## 2. 最新决定与仍未确定的方法

用户明确认为“同神经记录日期集合”没有必要，新探索不再把它作为样本对的入选条件。这不等于已经证明日期没有影响；日期元数据仍应保留。

之前要求同菌属，是为了固定分类背景，查看可比化学距离下的神经差异。新问题把菌属本身作为研究对象，需要同时考察属内和属间，不能继续只保留同属 pairs。

用户没有明确决定新的跨 reference 比较规则。上轮曾讨论把 reference 显示为分组信息，并检查规律是否与 reference 重叠；这只是方法建议，不能写成已确认参数。也没有确定新的距离定义、菌属最少株数、汇总权重、检验方法、降维方法或输出页数。已有表示和距离可以作为起点，但不能把旧四象限的 25% 阈值自动当成新问题的判定标准。

讨论中提出过以下解释和绘图建议，尚未执行，也不替代用户的五步顺序：

- “属内差异小”需要参照，可以比较属内与属间距离的分布和重叠程度。两种距离的尺度不同，不直接比较化学距离与神经距离的数值大小。
- 按菌属保留分布与株数，避免只显示属均值或把所有 pairs 混合。某个属内部差异较大，并不自动排除其他属存在稳定结构；若遇到这种情况，先展示结果再与用户讨论。
- 可先用布局对应的化学图和神经图回答前两个问题；若有继续探索的依据，再用同一菌属顺序的两张属间距离矩阵查看关系，并检查具体的化学共同变化与完整神经组合。各自排序后都出现色块，不足以证明两边结构对应。
- 第 3、4 步分别成立后，还需直接检查两种关系是否对应。例如，化学上 A 属接近 B 属、远离 C 属，神经是否也保留这种关系。两边各自能区分菌属，并不保证它们区分的是同一种结构。
- 即使属间结构对应，也需区分它与属内菌株关系。可以讨论神经组合反映了部分菌属相关化学结构，但不能据此写成化合物驱动机制、菌株功能、行为意义或技术优于 LC–MS。

## 3. 本 session 为什么转向菌属问题

最初从四种关系切入：神经近／化学近、神经远／化学远、化学近／神经远、化学远／神经近。用户选择两种距离各自最近与最远 25%，并检查 20%／30%。用户随后澄清，应逐对绘图整理为 PDF，先看具体差异，不生成类别总结或最终 Figure 5。

之后做过单化合物与单神经元的候选关联检查，也做过严格背景下的 anchor 比较。用户指出，后者主要看到相对于同一个 anchor，两个化学距离接近、神经距离也多接近；每个案例变化的物质与神经元不同，难以概括规律。

这轮讨论确认了旧图的解释限制：两个化学距离接近，是 anchor 比较的筛选条件；B 的神经距离小于 C，是绘图排序。二者都不是发现。化学距离大小相近，也没有保证变化的是同一组化合物或同一变化方向。逐页变化的 top 18 适合检查案例，但不提供固定的跨案例比较坐标。

### 已交付的探索 PDF

| 文件 | 内容与当前用途 |
| --- | --- |
| [四类关系逐对图册](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/output/pdf/neural_chemical_pairs_exploration.pdf) | 1,479 页：说明页及 1,478 对，每对展示 13 类神经元系数／unit 系数、380 项化学散点和该对 top 18。原报告缺值单独标记。属于检查材料。 |
| [神经元与化合物候选关联](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/output/pdf/neural_compound_relation_candidates.pdf) | 13 页：说明页及 12 个候选。用户已明确不希望回到单分子对应单神经元的主图解释。 |
| [严格背景与 anchor 比较](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/output/pdf/neural_chemical_matched_context.pdf) | 124 页：说明、100 对散点、单个未匹配的神经远案例、全部 121 个 anchor 比较。旧筛选下结果仍有效，但不是新探索的样本全集。 |

第一份 PDF 中，化合物若在至少两种关系的 top 18 中出现，名字标红并加粗。规则覆盖 160 个化合物、26,371／26,604 次显示，范围很宽；红色仅方便查找，不代表统计显著或神经关联。说明与生成入口见 [四类图册 README](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_pair_quadrants_20261002/README.md)。

### 100 pairs 与 121 个 anchor 比较的旧规则

106 株的全部不重复菌株对为 5,565 对。同 reference 且神经记录日期集合完全相同，剩 343 对；再加同菌属，剩 100 对。100 对散点没有按化学或神经距离筛选，其中神经近 82 对、中间 17 对、神经远 1 对。

121 个 anchor 比较另要求三株 A、B、C 共同满足上述背景，并使 A–B 与 A–C 的化学距离相对差不超过 10%：

```text
2 × |d(A,B) − d(A,C)| / [d(A,B) + d(A,C)] ≤ 0.10
```

每个比较把神经距离较小的 partner 排为 B、较大的排为 C，不要求一近一远。121 个比较覆盖 31 株、25 个 anchor、6 个 reference／日期／菌属组合；112 个比较来自 Bacteroides，其中 79 个来自同一个 reference／日期组合。相同菌株反复出现，不能当 121 次独立重复。

在这一严格范围内，没有化学距离匹配的神经近／神经远对照。唯一神经远 pair 是 A291／A296，其化学距离 0.9643、神经距离 0.8795，没有同背景第三株可形成 anchor 对照。它仍只是一个案例。

100 对散点的分界线取自全部 5,565 对：化学近 ≤1.203471654、化学远 ≥1.646742924；神经近 ≤0.282589914、神经远 ≥0.700326207。图上 1.75 是横轴刻度，不是阈值。化学分界线是 complete162 原报告距离的新分位数，不能与第一份图册的旧 380 项距离分类混用。

完整方法与记录见 [严格比较 README](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_matched_pair_context_20261003/README.md) 和 [参数](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_matched_pair_context_20261003/parameters.json)。下个 session 不应直接调用该目录的 runner 来执行新问题，它会重新施加旧限制。

## 4. 当前神经表示与化学数据

### 神经表示是 106 × 13 的有符号模板系数

13 类细胞按当前表顺序为 ASK、ADL、ASI、AWA、AWB、ASG、ADF、ASH、ASJ、ASEL、ASER、AWCON、AWCOFF。

- 每类细胞一个跨菌株共享的时间模板，使用 0–40 s 的 8 个 5-s bin。加权非中心化 SVD 拟合，模板 RMS 为 1、绝对值最大 bin 定为正。因此 coefficient 是该模板的有符号系数，单位为 ΔF/F₀。
- 当前主设置为 individual SNR ≥0.5、每条件至少 2 只动物。trials 先在动物内平均，再对动物等权；菌株跨日期时日期等权。未通过 SNR 的系数置零，真正缺失保留 NaN。当前菌株表 106 × 13 全部有限，106 个向量均非零。
- unit coefficient 将一株的完整 13 维向量除以 `sqrt(sum(coefficient**2))`。它不使系数之和等于 1，也不是细胞贡献百分比。
- 神经距离为 `1 − cosine`。整组系数同乘正数，例如全部翻倍，神经距离不变；相对幅度与符号变化会改变向量方向。单个系数的符号相对于各自模板，不能直接翻译为兴奋或抑制。
- 在相同共享模板、各模板等 RMS、相同 bin 支持下，系数 cosine 等于完整模板重建曲线展平后的 cosine。它不等于含模型残差的原始曲线 cosine，时间形状偏离可能被系数表示遗漏。
- 表中的 `raw_coefficient` 是在该门槛模板上投影但尚未置零的值。它不同于另行拟合的 unfiltered 模型。已保存不筛选及其他 SNR 门槛的比较，可在需要时复用。

现有验证没有证明 SNR 筛选提高了预测或样本分离能力。bootstrap valid fraction 是距离有效计算的比例，不是置信度，也不是某个生物模式成立的概率。更完整的模型定义、验证数值和显示约定在前一份 handoff 中。

### 化学距离有两个版本，需明确区分

原报告有 380 个注释化合物，其中部分菌株／化合物单元格缺失。缺失不能直接解释为不存在或低于检出限。旧 log₂FC 构造曾将缺失填零再加 1，按各组 reference 作分母；因此旧 380 项距离和 top 18 可能受到缺值处理的明显影响。

最近的严格比较使用既有 complete162 面板：要求全部 106 株的原报告值存在且 QCRSD ≤0.30，共 162 项，样本侧均为有限正值。它是完整报告的固定子集，不是完整代谢谱，也没有要求所有 reference 源列完整。计算为：

```text
x[s, k] = log2(original_report_value[s, k] + 1)
d_chemical(a, b) = sqrt(mean_k((x[a, k] − x[b, k])**2))
```

此处没有插补样本缺值，1 是固定原报告单位下的 pseudocount。同 reference 内，log₂FC 的共同分母在两株差值中抵消；complete162 的原报告距离与同 reference 的 log₂FC 距离已核对一致，最大误差约 2.44e−15。这个等价性不能直接推广到跨 reference 的菌株对。去掉日期门槛也不解决跨 reference 的化学可比性问题。

四个 reference 名字来自旧 FC 的分母来源，经数值重建核对：

| reference | 分母来源 | 当前神经数据中的株数 |
| --- | --- | ---: |
| A050 | 原表 A050 列 | 25 |
| A250 | 原表 A250 列 | 15 |
| A306 | 原表 A306 列 | 26 |
| ref12 | A051–A057 和 A088–A092 共 12 列的均值，沿用旧缺值处理 | 40 |

reference 不是 anchor，也不是神经记录日期。现有元数据尚未可靠确认这些 ID 各自对应的具体培养基、空白或实验批次，不应直接叫“批次”。化学来自独立培养材料，没有测量实际用于神经记录的刺激 aliquot；化学报告量也不能直接当作神经元实际接触的浓度。

## 5. 下个 session 的数据入口

优先复用已保存输入，无需为了新作图重跑模板、bootstrap、embedding 或整个 Notebook。各表用菌株 ID 对齐，不能假定行顺序相同。

| 文件 | 用途 |
| --- | --- |
| [当前菌株神经系数](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/tables/strain_coefficients.csv) | 当前主表示，106 × 13。以其索引确定共同菌株范围。 |
| [日期条件指标](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/tables/condition_metrics.csv) | `strain`、`block`、`cell`、`coefficient`、`raw_coefficient`、动物数等。日期信息仍保留。 |
| [完整样本背景快照](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_matched_pair_context_20261003/tables/sample_context.csv) | 全部 106 株的 `strain, reference, dates, genus, species, coefficient_norm` 等；并未限于 100 对或 31 株。 |
| [unit 神经系数快照](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_matched_pair_context_20261003/tables/neural_unit_coefficients.csv) | 完整 106 × 13 的 L2 归一化表示。 |
| [complete162 原报告 log 快照](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_matched_pair_context_20261003/tables/chemical_log2_report_plus1.csv) | 完整 106 × 162，不是 FC。 |
| [全部 pair 的两种距离及背景](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_matched_pair_context_20261003/tables/all_pair_catalogue.csv) | 全部 5,565 对，含 `chemical_distance`、`neural_distance`、背景及旧表示对照。新问题不要过滤 `strict_context`。 |
| [原报告化学值](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/tables/aligned_chemical_report_values_all.csv) | 更大样本范围 × 380，需按当前 106 株索引取行；保留源缺失。 |
| [化合物元数据](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/tables/aligned_chemical_metadata.csv) | `previous_complete_162_eligible` 标识固定 162 项，另含 QC 与化学注释。 |
| [旧 log₂FC](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/tables/aligned_chemical_log2fc_paired.csv) | 106 株 × 380 项，注意原有缺值与 reference 处理。 |
| [分类信息](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/tables/aligned_taxonomy_paired.csv) | `genus_clean`、`species_clean`。同属不等于同种。 |
| [reference 映射](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/tables/aligned_chemical_reference_groups_paired.csv) | `reference_group` 及分母重建误差。 |
| [既有表示敏感性与配对覆盖](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_pair_quadrants_20261002/tables/pair_catalogue.csv) | 全 5,565 对的主神经距离、不筛选／其他门槛／原曲线对照、bootstrap 有效比例。这里的 `chemical` 是旧 380 项距离。 |

### 已核对的菌属覆盖

106 株共 29 个菌属，其中 13 属至少 2 株，其余 16 属各 1 株。单株属不能估计属内离散，但是否纳入属间展示尚未决定。

| 菌属 | 株数 |
| --- | ---: |
| Bacteroides | 29 |
| Bifidobacterium | 11 |
| Pediococcus | 7 |
| Enterococcus | 6 |
| Limosilactobacillus | 6 |
| Escherichia | 5 |
| Lactobacillus | 5 |
| Streptococcus | 5 |
| Bacillus | 4 |
| Lactiplantibacillus | 4 |
| Lacticaseibacillus | 3 |
| Leuconostoc | 3 |
| Megasphaera | 2 |

菌属与 reference 有明显覆盖重叠。例如 Bacteroides 的 29 株中 23 株在 A050、6 株在 ref12；Bacillus 的 4 株全在 A250；Escherichia 的 5 株全在 ref12。直接按 pairs 数汇总还会放大多株菌属的权重。需在新分析中说明比较范围和权重，不能把共享菌株的 pairs 当成独立生物重复。

## 6. 可以参考的旧分析，不能当作当前结果

[旧菌属分类任务](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/logs/shared_task_findings.md) 在固定 8 属、74 株上，化学分类宏召回率为 72.25%，神经原单位表示为 49.99%，仅 reference 为 29.85%。这提示两种测量各自含部分菌属信息。

该任务神经输入是 13 类细胞 × 5 个时间窗的 65 维原始均值，化学是 380 项 log₂FC，使用留记录块并从训练中清除测试菌株的分类任务。它不等于当前 13 维模板系数的几何结构，也不回答 complete162 下属内与属间距离有何规律。旧任务的“至少 5 株、至少 3 个记录块”仅是当时条件，不应机械搬入本次探索。参数见 [旧分类参数](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/logs/shared_task_parameters.json)。

[旧化学潜变量预测](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_first_20260930/logs/latent_chemistry_findings.md) 中，化学加 reference 对旧群体神经坐标的留出预测 R² 约 0.02–0.04，低于 genus 加 reference 的约 0.09–0.12。它提醒我们保留化学解释不足的可能性，不能当作新问题已被否定，也不应在下次 session 自动重做更复杂模型。

[旧群体探索报告](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/population_exploration_20260930/REPORT.md) 还保留属间和属内响应组合的案例、动物层面的检查及化学共同变化探索。查看时注意它使用的时间表示与当前模板系数不同。个别案例可供回查，尚未形成用户认可的 Figure 5／6 主图。

## 7. 执行状态

本轮仅查看已有文档、代码和元数据，核对覆盖并新建本文件。没有执行新的属内／属间科学分析，没有改原始数据、Notebook、既有结果、PDF 或前一份 handoff，没有提交 Git。

工作区在本轮前已有未提交修改，包括 AGENTS.md、Notebook 02／03、HMDS 模块，以及未跟踪的 reports、tests 和脚本。继续工作时不要回滚这些内容。仓库 Python 为 `/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/.pixi/envs/default/bin/python`，已有 numpy、pandas、scipy、matplotlib；读取现有数据与常规绘图无需安装依赖。

下个 session 应从用户的五个问题和全部 106 株的现有输入接手，先确定当次要画的比较及方法，再按用户要求执行。当前没有一套已确认、可自动跑完五步的方案。
