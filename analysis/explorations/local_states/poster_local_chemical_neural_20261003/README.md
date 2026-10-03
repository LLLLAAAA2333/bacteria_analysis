# Poster 最后一张图：局部化学状态与神经响应组合

已完成可用于 poster 排版的主图，覆盖 **40 株、22 个记录菌种、全部 13 类神经元**。主旨是：在局部菌群中，神经响应组合可以获得具体的化学状态含义；不同局部关系涉及的神经分量不同。

![Poster figure](figures/poster_final.png)

[可编辑 SVG](figures/poster_final.svg) · [矢量 PDF](figures/poster_final.pdf) · [高分辨率 PNG](figures/poster_final.png) · [英文图注](figures/CAPTIONS.md)

主图为 12 × 9 inch，PNG 为 3840 × 2880；SVG 保留可编辑文字，PDF 嵌入字体。图中无日期着色。原始数据、已有报告和 Notebook 保持不变。

## 两列分别说明什么

**Bacteroides（29 株）保留交接中的 14 项化学状态及固定 ADF−ASH。** 高化学状态对应 ADF−ASH 向正侧偏移，下方同时展示全部 13 类神经元的观测变化。没有重新挑选样本或化学成员，也没有把两神经元对比改写成所有神经元都有关。

**Bifidobacterium（11 株）是本轮新增的完整神经组合案例。** 同一套化学共变分组规则形成 12 个候选，按全部 13 维响应的总拟合平方误差选择一个 21 项、20 个 Mass-column family 的状态。较明显的相对变化涉及 AWA、AWCOFF、ADF、ASH；其中留出误差改善主要来自 AWCOFF 和 AWA。主图散点的神经投影由同批 11 株拟合，轴标签已明确写出 fitted，不能把它当成独立验证。热图使用实际观测均值，绝非预测值。

每属内部按化学分数分成三个尽量等大的组，分别为 10/10/9 和 4/4/3 株。热图减去属内各神经元均值，共用 ±0.30 的 unit-coefficient 色阶，不逐行 z-score。单株差异、全部化学成员和全部神经元另见 [Bacteroides 个体图](figures/support_individuals_bacteroides.png)、[Bifidobacterium 个体图](figures/support_individuals_bifidobacterium.png)。

## 同口径检查及没有隐藏的阴性结果

改善均为 `1 − SSE_model / SSE_training_mean`，不是 RMSE 降幅、准确率或总神经方差解释比例。

| 范围与预测目标 | 逐株留出 | 整记录菌种留出 |
|---|---:|---:|
| 既有 Bacteroides：固定 ADF−ASH | +23.09% | +19.96% |
| 既有 Bacteroides：沿同一选组流程辅助预测全部 13 维 | +4.16% | +3.55% |
| **新增 Bifidobacterium：按全部 13 维选组并预测全部 13 维** | **+15.87%** | **+18.22%** |
| 新增 Bacteroides：按全部 13 维选组并预测全部 13 维 | −4.83% | −15.71% |

新 Bacteroides 流程选中 Methylsuccinic acid、Asparagine、Tyrosine 三项，内部留出未优于均值基线。因此它没有替代既有主图锚点，所有候选、参数和阴性结果完整保留。两种主图神经读出不同，不能用 23.1% 与 15.9% 排名。

新增分析按样本覆盖预先限定 `genus n ≥ 10`，得到上述两个属，没有探索任意菌株子集。每个留出折重新估计化学尺度、聚类、成员、权重、选组及回归。Bifidobacterium 的完整成员名单随留出变化，中位 Jaccard 约 0.77，不能称为固定 21 项签名被验证。其四个主要变化分量在未单位化的 gated 模板系数中也保持同向；同化学分数的 pre-gate 神经方向与主表示的 cosine 约 0.985。

这些是条件于既有神经表示及化学QC的内部检查，既往已多次查看该数据。小样本均值有波动，尤其 Bifidobacterium 高组仅三株；不同坐标并非均有信息。化学报告来自独立培养材料，不是实测刺激液。这些边界限制定量推广和因果解释，但不妨碍将图定位为“神经表征具有局部化学可解释性”。记录条件保留在元信息中。

## 文件与复现

- [冻结的有限扩展范围](PROTOCOL.md)、[完整新分析摘要](analysis_summary.json)、[来源哈希](source_manifest.json)。
- `tables/` 与 `parameters/`：新流程在两个属的完整结果，包含未用于主图的 Bacteroides 三项状态。
- [主图数据与来源说明](figure_data/lineage.json)：主图明确组合旧 Bacteroides L03 和新 Bifidobacterium L02；`figure_data/` 保存准确的绘图分数、成员和观测均值。不要混用 `tables/strain_scores.csv` 中的新 Bacteroides 状态。
- [绘图函数](code/plot_poster.py) 只读保存结果；[展示数据整理](code/prepare_display.py) 不重跑模型；[分析函数](code/analyze_population.py) 仅供明确需要重算时调用，已有结果有防覆盖检查。
- [Notebook 可复制单元](code/notebook_cells.py) 只读主图和支持材料；没有新建或执行整个 Notebook。
- [复核记录](verification.json)：独立复核两个完整拟合及全部 62 折，3,908 项数值比较的最大绝对差为 3.55 × 10⁻¹⁵；另核对主图来源、分组、实际均值及导出布局。
