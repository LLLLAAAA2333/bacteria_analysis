# Trial-SNR 版本：profile、筛选状态、重复性、RDM 与 HMDS

2026-10-01。本目录是针对六项修改的新探索版本；原 Notebook、正式 Figure 4/5 和上一版 individual-SNR 探索均保留。

## 图的入口

| 请求 | 当前结果 |
| --- | --- |
| 按 trial 计算 SNR | 使用 31,616 条 class-trial 曲线，先核对其均值与既有 animal 缓存一致；[方法参数](representation_parameters.json) |
| profile 恢复五个 bin | [5-bin profile](figures/01_response_profile_5bin.png)：13 类细胞各占五个 5 s bin，展示 0–25 s；系数 × 模板拟合仍用 0–40 s |
| 明确过滤了哪些 neuron–bacteria | [完整筛选状态热图](figures/00_filter_status_heatmap.png)：106 个菌株全部标名；[逐菌株×细胞状态](tables/filter_state_by_strain_cell.csv)、[逐日期判定与 trial 数](tables/condition_metrics.csv) |
| RDM 配套 HMDS | [RDM](figures/03_neural_chemical_rdm.png)及[匹配样本的 HMDS](figures/03b_neural_chemical_hmds.png) |
| 恢复重复性分布图 | [重复性](figures/02_repeatability_distribution.png)：右侧恢复原 Notebook 的密度直方图，bin 宽 0.05、填充加轮廓、均值虚线 |
| 配色一致 | 全部图使用 `RdBu_r`；分类状态与曲线颜色也从同一色图取色。不同量纲保留各自色标单位与范围 |

另有[门槛敏感性](figures/00b_trial_snr_thresholds.png)和[重复性/留出预测对照](figures/02b_representation_validation.png)。每张图同时保存可编辑 SVG，[外部图注](figures/captions.txt)记录具体边界。

## 本轮 trial-SNR 的明确含义

本轮按“同一 neuron–bacteria 条件内重复 trial 的响应与散布”实现。这里仍是**重复 trial 间的经验信号/离散比**，没有切换成每条 trial 相对自身刺激前基线的 SNR，也没有逐条剔除低响应 trial 后抬高平均响应。

不同动物的 trial 数为 1–8，不相等。为保持既有幅度定义，每只动物等权，其内部 trials 均分这份权重。一个条件内有 m 只动物，动物 a 有 rₐ 个 trials，则每条 trial 权重为 wₐᵣ=1/(m·rₐ)。令 μ(t) 为这个加权平均曲线，q=Σw²：

```text
P = mean_t[μ(t)²]
V = mean_t[Σ w·(y_trial(t) − μ(t))²] / (1 − q)
C = P − V·q
trial SNR = sqrt(max(C, 0) / V)
```

因此，SNR 直接使用 trial 曲线的二阶矩，而不是只使用 animal 平均曲线之间的方差。V 包括 trial 内部重复和动物之间的差异；它不是只在每只动物内计算的方差。1/q 仅是权重的等效条目数，不能解释成独立生物样本数。上述加权修正沿用经验门槛形式；由于 trials 嵌套在动物内，不能将 C 宣称为严格无偏信号功率或显著性检验。

主门槛仍暂设 1，并保存 0.5、1.5、2 和不筛选对照。刺激前噪声作为辅助量保留。充分观测且不通过的条件系数置零；原始均值、trial 数、animal 数、置零前系数及残差均保留。

全数据的覆盖门槛恢复为原模板分析的至少 2 只动物，同时要求至少 3 条 trials；两半/留出训练允许 1 只动物，但仍要求至少 3 条 trials。全数据 1,456 个条件 × 细胞条目均满足覆盖，实际总 trial 数为 9–34。上一版采用至少 3 只动物而排除的 39 个条目，在这里恢复，因此新旧版本的变化不应全部归因于 SNR 分母改变。模板/幅度的比较对照使用本轮相同覆盖。

重复性、留出检查和 HMDS bootstrap 仍按**整只动物**划分或重采样，其所有 trials 一起移动；没有把嵌套 trials 当成独立动物验证。

## 筛选状态热图怎样读

- 红色：该菌株–细胞在所有可评价日期都通过。
- 蓝色：所有可评价日期都未通过，条件系数置零。
- 白色：不同日期有保留也有置零；最后菌株幅度仍按日期等权平均。
- 灰色：无可评价日期。本轮全数据没有这种条目。

左右两半接续同一个菌株顺序，每个菌株 ID 都能查到。106×13 个组合中有 **230 个保留、1,134 个全部置零、14 个跨日期不一致**。按条件 × 细胞计，则是 254/1,456 保留，1,202/1,456 置零（82.6%）。ADL、ASI、ASG 在门槛 1 下均无保留条件，因而没有可识别的模板；不能把它们解读为实测曲线始终为零。

| 门槛 | 保留条件 × 细胞 | 置零条件 × 细胞 |
| --- | ---: | ---: |
| 不筛选 | 1,456 | 0 |
| 0.5 | 691 | 765 |
| 1 | 254 | 1,202 |
| 1.5 | 81 | 1,375 |
| 2 | 21 | 1,435 |

这些数量说明 trial 门槛仍需视为探索参数；本轮没有根据 chemical 相关或图面稀疏程度选取门槛。

## 重复性与 RDM

重复性右图已恢复密度分布，不使用累积分布。它展示 100 次独立半组处理后，每个样本对的平均有效余弦相似度；两个分布各自归一化。同样本有 104 个可评价均值，其平均值为 0.640；同一支持上的不筛选模板对照为 0.838。所有有效划分数量单独导出，零向量相似性保持未定义。

对 49 只留出动物的 7,063 条原始八 bin 曲线，等权聚合 MSE 为：不筛选模板 0.03485、trial 门槛 1 为 0.04428，后者高约 27.1%。这里没有对留出的真实响应清零。当前门槛依然不能称为已经验证的降噪改进。

全数据有 9 株全零 profile，使神经余弦 RDM 有定义的样本对从 5,565 减为 4,656。RDM 保留完整 106×106 布局，未定义条目显示灰色。这些有效对上，筛选神经与 chemical RDM 的 Pearson/Spearman 为 0.240/0.203；同样 4,656 对上的不筛选模板为 0.265/0.224。没有报告把重叠样本对当独立观测的 p 值。

Profile 的五个 bin 只用于与 Notebook 一致的展示，重复性、RDM 和 HMDS 使用完整 0–40 s 的八 bin 模板表示；没有把五个 bin 拉伸成 40 s。

## HMDS 的范围与核验

神经 HMDS 沿用 Notebook 的输入定义：将神经 cosine distance 转成 chord distance，即 sqrt(2×(1−cosine))。在日期内联合重采样整只动物 1,000 次，每次重新估计 trial 筛选和模板；保持原有条件、日期及每对细胞支持。方差取有效 draws 的样本方差，不除以 draws 数量。继续要求至少 80% draws 有效。

实际可用于 HMDS 的网络包含一个 **67 株、1,937 条边**的连通集合，其余 39 株没有满足标准的边：9 株来自全数据零 profile，30 株来自 bootstrap 覆盖不足。完整名单在[neural_sample_coverage.csv](hmds/neural_sample_coverage.csv)。没有为这些样本补距离或降低覆盖门槛。

两幅 HMDS 图展示同样 67 个菌株。神经嵌入在这 67 株上重新拟合；chemical 图截取已验证的完整 106 株嵌入中的相同样本，其输入 RMS RDM 与当前缓存最大差异为 1.78×10⁻¹⁵，因而无需重新拟合。两图同一菌株使用同一个冻结 chemical PCo1 数值着色，色图统一为 `RdBu_r`；颜色不是神经聚类标签。

神经拟合收敛与数值梯度检查均通过，联合几何梯度为 1.02×10⁻⁸，relative RMSE 为 0.1470。chemical 原二维拟合也已通过检查。保留原来的双曲等距居中，没有额外收缩圆盘坐标；点靠近边界不表示生物学层级或置信度。两图的坐标方向和欧氏点间距不能直接当成跨模态对应。[拟合结果与来源](hmds/result.json)和各自 Shepard 数值均已保存。

初次对全部 106 株调用拟合时，原方法正确拒绝了不连通网络；其初始诊断保存在 `hmds_initial_disconnected/`。最终有效结果均位于 `hmds/`。

## 文件完整性

12 项针对 trial 权重、嵌套留出、缺失支持和 bootstrap 的合成测试通过。trial 均值与原有 animal 曲线、原始 1 s 到八 bin 的缓存转换、原 Figure 4 无筛选拟合均经过数值核对。原 Notebook 和输入文件哈希未变，见[verification.json](verification.json)。本轮没有运行或修改原 Notebook。
