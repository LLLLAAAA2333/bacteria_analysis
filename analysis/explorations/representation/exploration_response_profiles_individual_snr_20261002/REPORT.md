# Individual SNR 0.5 exploration — 2026-10-02

按本次要求，将响应筛选改回 individual，并将主门槛由 1 降为 **0.5**。原始 notebook、输入数据和前两版探索均保留。该门槛是预先指定的探索设置，没有根据 chemical 相关性或验证结果调优。

最新图稿：5-bin profile 使用 notebook 02 的独立热图布局，模型示意与模板/系数另成一图；split-half 恢复 notebook 的矮宽直方图和图外图例。HMDS 的 2D、3D 均已扩展到全部 **106 株 / 5565 对**，分别附神经与化学 Shepard diagram；不再用 80% bootstrap 有效率排除样本。响应筛选、模板系数、重复性、RDM 和 bootstrap 数值保持原样，只重新拟合全部样本的神经 2D、3D。点色保持原始 chemical PCo1 → turbo 映射，RDM 使用 RdBu_r。

## 为什么 trial 版本过滤更多

逐项匹配 1456 个 strain × recording date × neuron 条目后，两种算法的平均响应和信号功率完全相同，变化主要来自方差分母：trial 方差 / individual 方差的中位数为 **3.00**，四分位范围为 **2.06–4.44**。trial 总波动中，动物内 trial 波动所占比例的中位数为 **72.5%**。

individual 版本先在每只动物内平均 trials，再评估这些动物平均曲线之间的波动；trial 版本还计入了平均前的动物内波动。因此，相同数值的 SNR 门槛并不代表相同筛选强度。trial 波动可能同时包含测量噪声、适应、顺序效应和漂移，不能全部解释为测量噪声。trial 数量增加带来的偏差校正变化反而有利于提高 SNR，本次是方差增加的影响占主导。

统一最低 2 只动物的入选范围后：

| 筛选单位和门槛 | 保留条目 | 比例 |
| --- | ---: | ---: |
| Trial ≥ 1 | 254 / 1456 | 17.4% |
| Individual ≥ 1 | 538 / 1456 | 37.0% |
| **Individual ≥ 0.5** | **969 / 1456** | **66.6%** |

旧 individual 报告中的 518 项还使用了最低 3 只动物的范围，不能直接与这张表混用。本版沿用最近 trial 版的最低 2 只动物范围；若仍要求 3 只，individual ≥ 0.5 会保留 940 项。

可核查的逐项比较：[snr_definition_comparison.csv](tables/snr_definition_comparison.csv)；汇总与方差分解：[snr_definition_audit.json](snr_definition_audit.json)。这些是描述性条目计数，不是独立生物学重复的数量。

## 本版定义

- 每只动物内 trials 等权平均；同一菌株、日期内动物等权。SNR 使用 **0–40 s 未分 bin 的动物平均曲线**。
- 令 μ 为动物平均响应，P = meanₜ(μ²)，V = meanₜ(var across animals, ddof=1)，n 为动物数。SNR = sqrt(max(P − V/n, 0) / V)。这是基于重复间波动的响应稳定性指标，不是刺激前基线噪声比。
- 至少 2 只动物且 SNR ≥ 0.5 才保留；有足够数据但未通过筛选的系数置零。缺失或不足 2 只动物保留 NaN，不当成零响应。零值代表分析决策，不证明生理无响应。
- 每类细胞使用共享的 **0–40 s、8 个 5-s bin** 模板，RMS 归一化为 1；系数有正负号，单位为 ΔF/F₀。每个门槛均重新拟合模板。
- Profile 保留 notebook 的 **0–25 s、5 个 5-s bin** 展示范围，显示系数 × 模板的前 5 bins；模型拟合、重复性和 RDM 均使用完整 0–40 s。
- 跨日期的菌株表示按日期等权平均。不同日期的筛选状态分别保留，不强制合成一个二元决定。
- 敏感性设置为不筛选及 0.25、0.5、0.75、1。全数据筛选和模板仅用于描述；验证中的筛选与模板均只用训练动物重拟合。

全部参数及来源哈希：[representation_parameters.json](representation_parameters.json)。

## 更新图稿

| 图 | 文件 | 内容与范围 |
| --- | --- | --- |
| 筛选状态 | [Heatmap](figures/00_filter_status_heatmap.png) | 106 株 × 13 类细胞，每个菌株均有标签；红色保留、蓝色置零、白色跨日期混合、灰色不可用 |
| 门槛对照 | [Thresholds](figures/00b_individual_snr_thresholds.png) | 各细胞在 0.25 / 0.5 / 0.75 / 1 下的保留比例 |
| 响应结构 | [5-bin profile](figures/01_response_profile_5bin.png) | 恢复 notebook 的聚类树、细胞分块、刺激结束虚线和彩色细胞标签；仍显示当前模型重建的 5-bin 响应 |
| 模型表示 | [Model](figures/01b_response_model.png) | 上方为曲线压缩示意，下方为 0–40 s 模板和有符号系数矩阵 |
| 重复性 | [Repeatability](figures/02_repeatability_distribution.png) | 100 次动物独立分半；恢复矮宽密度直方图、上方图例和原配色，数值与尾部完整保留 |
| 表示检验 | [Validation](figures/02b_representation_validation.png) | 相同有效分半和细胞支持下的对照，以及留一动物预测 |
| 整体比较 | [RDM](figures/03_neural_chemical_rdm.png) | 相同 106 株顺序，含所有 5565 个样本对 |
| 2D 空间展示 | [2D HMDS + Shepard](figures/03b_neural_chemical_hmds_2d.png) | 上方为神经与化学圆盘，下方分别为 Shepard；两侧均为全部 106 株 / 5565 对 |
| 3D 空间展示 | [3D HMDS 三视图 + Shepard](figures/03c_neural_chemical_hmds_3d.png) | 神经与化学各一行，前三列为同一三维坐标的 XY / XZ / YZ 正交投影，右列为 Shepard；均为 106 株 / 5565 对 |

响应热图、系数图和 RDM 使用 **RdBu_r**，神经 RDM 范围 0–2、化学范围 0–实际最大值。HMDS 点使用 notebook 03 原始 **turbo** 配色，直接按样本 ID 复用存档 `color_hex`；颜色条使用同一 4096 级连续 palette、完整 106 株参考的线性范围 −2.6673737721171715 至 2.7364307161881327，不重新归一化，也不以零为中心。每张图另有同名 SVG。[外部图注](figures/captions.txt)记录筛选规则、显示饱和及覆盖范围。此前的 81 株图保存在 `figures/previous_81_sample_hmds/`；旧文件 `03b_neural_chemical_hmds.png` 是更早的不带 Shepard 版本，当前使用上表的两张全样本图。

主门槛在日期层面保留 969 项、置零 487 项。合并为 106 × 13 的筛选图后，909 格所有日期保留，454 格所有日期置零，15 格不同日期状态混合。ASG 保留 **15/112** 个日期条件；这不能单独证明其少数保留响应具有特异性。

## 验证结果及边界

在相同细胞、相同有效分半的对照中，同菌株平均 cosine 为：筛选后 **0.8180**、不筛选的模板表示 **0.8546**、原始 8-bin 表示 **0.8106**。分半时仍要求每半至少 2 只动物；只有 1 只动物的条目无法估计 individual 方差，保持不可用。主图同菌株条目有 73–100 次有效分半；169 个不同菌株对没有有效分半。重复划分和共享菌株的样本对不能当成独立重复。

留一动物预测以未处理的留出动物 8-bin 曲线为目标，三种方法在共同可预测支持上计分：6985/7063 条动物曲线，覆盖 49 只动物；78 条因训练动物不足排除。

| 表示 | 等权汇总 MSE (ΔF/F₀)² |
| --- | ---: |
| Individual SNR 0.5 + template | 0.037790 |
| Unfiltered template | 0.036289 |
| Zero prediction | 0.094151 |

主设置的预测误差仍比不筛选高 **4.14%**。因此，该门槛能减少展示中的低稳定性条目，但现有检验不支持“筛选改善去噪或预测”的表述。[比较结果](comparison_results.json)保留完整指标。

### Split-half 直方图与原 notebook 的差异

原 notebook 内嵌 Panel B 的均值标签是同菌 0.78、不同菌 0.40；本次分析为 0.82、0.44。最初重绘版的直方图偏高（宽高比 1.12）；现在已恢复原 notebook 的矮宽布局（1.74）、上方图例、原红蓝填充色和稀疏刻度。没有重算 split，没有改动输入数组、均值、计数或密度。bin 宽仍为 0.05、两组分别归一化、填充透明度 0.32，均值虚线不变。横轴保留 −1–1，展示真实低值尾部，纵轴从零开始且未截断峰值。视觉改善来自布局和配色，并不代表统计分离度增加。旧高版图保存在 `figures/previous_tall_repeatability/`；数值一致性检查见 [repeatability_display_parameters.json](figures/repeatability_display_parameters.json)。

现有 matched-support 缓存中，三个表示使用相同细胞及有效分半，描述性对照如下：

| 表示 | 同菌平均 cosine | 不同菌平均 cosine | 均值差 |
| --- | ---: | ---: | ---: |
| 原始 8-bin | 0.8106 | 0.4117 | 0.3990 |
| 模板，不筛选 | 0.8546 | 0.4618 | 0.3928 |
| 模板，individual SNR ≥ 0.5 | 0.8180 | 0.4427 | 0.3753 |

从不筛选模板到筛选模板，同菌分布的标准差由约 0.126 增至 0.210；平均相似度低于 0.5 的菌株由 3 株增至 8 株。说明低值尾部增加是现有结果中的真实差异，不只是排版。门槛附近的响应在两半中可能一边通过、一边置零，加上分别重拟合模板，可能增加半样本不稳定性；未保存逐折细胞门槛决定，不能把这一机制当成逐项验证的结论。对照估计的是筛选及相应模板重拟合的联合影响。

原 notebook 与新分析也不是同一计算条件：Notebook 02 的已保存准备输出明确写着动物矩阵 `(607, 104)`、`13 × 8 bins`，对应 0–40 s 的 8 个 5-s bins；因此这张旧 Panel B 不能解释为五 bin 基准。旧图输出路径指向 Windows，当前本地未找到其 split 数值矩阵。原代码按菌株分别混合日期后分动物、每半至少 1 只、至少 1 类完整共同细胞；新版按日期分全局动物身份、每半至少 2 只、日期等权、至少 4 类共同细胞，并各半独立筛选/拟合。因此，新旧分布差异不能单独归因为时间 bin 数或 SNR。

神经与 chemical RDM 的描述性相关为 Pearson **0.2615**、Spearman **0.2133**；不筛选模板表示分别为 0.2399、0.1953。这里不以重叠的 5565 个样本对作为独立观测计算普通显著性，也不根据相关数值选择门槛。

## HMDS 覆盖与检查

神经输入为 sqrt(2 × cosine distance)。1000 次按日期整只动物联合 bootstrap，每次重算筛选和模板；固定原始条件、日期及样本对的细胞支持，不用缺失后的不同集合替代原定义。距离方差是 bootstrap 样本方差，不除以 bootstrap 次数。

全部 106 株均有非零 profile，所有 5565 对全数据距离及 bootstrap 方差均可计算。上一版用至少 80% 有效 draws 的额外门槛，只保留 3072 对、81 株；这不表示其余 25 株无效，也不表示样本间差异过大而无法拟合。现在移除这项额外覆盖排除，保留全部 106 株 / 5565 对。现有距离及方差逐项复用，无补值，无重做 bootstrap。

每对有效 draws 为 229–998/1000 次，中位数 807；1287 对低于 50%。覆盖作为质量记录保留于 [样本表](hmds_full106/neural_sample_coverage.csv)及[配对表](hmds_full106/neural_pair_coverage_qc.csv)。这些方差是“固定细胞/日期支持仍完整且距离可定义”条件下的样本方差；不能解释为全部抽样的无条件不确定性，低覆盖配对的位置不应视为与高覆盖配对同等可靠。完整范围与方法记录见 [scope.json](hmds_full106/scope.json)。

全部样本的神经 2D、3D 均重新拟合并通过局部收敛与数值梯度检查，联合梯度最大值分别为 2.07 × 10⁻⁹、4.47 × 10⁻⁸。3D 使用当前完整 2D 解升维后的多个起点和局部扰动。Chemical 的 2D、3D 继续复用经输入 RDM 核对的完整 106 株拟合，现在也展示全部 106 株。

| Shepard 展示范围 | 2D relative RMSE | 3D relative RMSE |
| --- | ---: | ---: |
| Neural：全部 5565 个样本对 | 0.1393 | 0.1029 |
| Chemical：全部 5565 个样本对 | 0.0903 | 0.0478 |

每个领域的 2D、3D 使用完全相同的 Shepard 样本对；纵轴为模型双曲距离换算回输入单位，未使用圆盘/球坐标的欧氏距离。Relative RMSE = norm(fitted − input) / norm(input)，当前全部样本均展示，chemical 指标也与完整拟合汇总一致。Chemical 2D 固定 λ = 10，3D 估计 λ，因此其误差变化不能纯粹归因于维度增加；神经两种维度都估计 λ。

两侧使用同一套固定 chemical PCo1 颜色。方向任意，圆盘/球半径不解释为生物学层级或置信度；两侧坐标距离不可直接比较。当前诊断：[2D](hmds_full106/2d/result.json)、[3D](hmds_full106/3d/result.json)。旧的 81 株数值结果在 `hmds/` 与 `hmds3d/` 中原样保留。

### 3D 三视图

当前按神经、化学分为两行，每行依次显示 XY、XZ、YZ 正交投影及 Shepard diagram。三幅投影直接使用同一组已保存的三维坐标，范围均为 −1.04–1.04，横纵轴等比例；圆周是单位球的投影边界。本次只调整展示，没有重新拟合、旋转坐标或选择相机角度；全部 106 株、点色、拟合距离及 Shepard 数值保持不变。

这些投影不属于另做的二维拟合，投影点间距也不代表双曲距离；两域方向任意且未对齐，Shepard 仍使用完整三维拟合距离。[核验记录](three_view_verification.json)记录六组实际绘制坐标、点色和 Shepard 数组与输入完全一致。上一版单相机图及源代码存于 `figures/previous_single_view_3d/`；更早的固定视角图存于 `figures/previous_3d_view/`。历史[视角对照图](figures/3d_view_candidates.png)与[选角记录](figures/hmds_3d_view_selection.json)保留备查，当前三视图不使用该选角结果。

## 可复现性与核验

11 项针对性合成测试已通过，覆盖 individual 方差、阈值传递、缺失与零值处理及比较支持。结果文件另经独立复核；原始输入和 3 个 notebook 的运行前后哈希一致，见 [verification.json](verification.json)。

本次图稿修订的来源、参数及原数据/2D 拟合保留检查另见 [figure_revision_verification.json](figure_revision_verification.json)、[profile 参数](figures/profile_display_parameters.json)和[比较图参数](figures/comparison_display_parameters.json)。

最新点色更正单独记录于 [embedding_color_revision_verification.json](embedding_color_revision_verification.json)：106 个原参考颜色逐项核对通过，108 个已有分析、拟合和 notebook 文件哈希保持一致。

代码和重运行入口见 [README](README.md)。本目录只记录本轮 individual SNR 探索，不修改既有正式图或 notebook。
