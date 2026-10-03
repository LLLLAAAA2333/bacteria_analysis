# 属内与属间化学／神经差异：第1、2步

日期：2026-10-03。实际仓库：`/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis`。

本次仅完成用户要求的第1、2步，并排比较同属与异属距离。没有开展属间化学模式发现、神经模式对应、预测训练或第3–5步。输入和旧结果保持原样。

## 图与读法

主图：[`figures/01_within_between_by_genus.png`](figures/01_within_between_by_genus.png)，另有可编辑文字的 SVG。完整英文图注：[`figures/01_within_between_by_genus_caption.txt`](figures/01_within_between_by_genus_caption.txt)。

两面板采用同一菌属顺序。蓝色为属内，橙色为该属到其他属的距离。大圆点为中位数，粗线为25%–75%分位，细线为10%–90%分位；浅蓝散点保留全部属内配对。线段是距离分布范围，不是置信区间。化学与神经的横轴定义不同，不能按数值大小判断哪种测量更紧密。

106株覆盖29属；13个多株属共90株在图中逐属展示，其余16个单株属仅参与属间参照。全部5,565个不重复菌株对中，同属配对561个。每个目标属的属间参照包含其余28属，各外属总权重相同；这描述的是“先等概率抽取一个外属，再抽取菌株”的距离分布。Bacteroides的406个属内配对不会被合并到一个跨属总体分布中主导结论。

## 描述性结果

总体可见属内距离偏小的趋势，但不能把所有菌属都视为内部紧密、化学或神经近似相同的单位。

下表比值均为该模态内的“属内距离中位数／属间距离中位数”；1表示两个中位数相同。它不是解释度、预测准确率或显著性。两模态距离变换不同，比值也不用于跨模态比较紧密程度。

| Genus | Strains | Within pairs | Chemical ratio | Neural ratio |
| --- | ---: | ---: | ---: | ---: |
| Bacteroides | 29 | 406 | 0.688 | 0.372 |
| Bifidobacterium | 11 | 55 | 0.692 | 0.912 |
| Pediococcus | 7 | 21 | 0.637 | 0.668 |
| Enterococcus | 6 | 15 | 0.975 | 0.821 |
| Limosilactobacillus | 6 | 15 | 0.475 | 0.017 |
| Escherichia | 5 | 10 | 0.778 | 0.336 |
| Lactobacillus | 5 | 10 | 0.996 | 0.451 |
| Streptococcus | 5 | 10 | 0.682 | 0.980 |
| Bacillus | 4 | 6 | 0.836 | 0.754 |
| Lactiplantibacillus | 4 | 6 | 0.450 | 0.797 |
| Lacticaseibacillus | 3 | 3 | 0.317 | 0.346 |
| Leuconostoc | 3 | 3 | 0.599 | 0.309 |
| Megasphaera | 2 | 1 | 0.735 | 0.140 |

主参照下两模态均有13/13属的中位数比值低于1，但这个方向计数不能表述为“13属都有明显或显著分离”。Lactobacillus的化学比值0.996、Streptococcus的神经比值0.980几乎等于1。Bifidobacterium的神经分布也有较大重叠。相对而言，样本最多的Bacteroides在两个面板中均显示清楚的属内距离偏小趋势。

低中位数还可能掩盖内部异质性。Limosilactobacillus有15个神经属内配对，其中10个距离约0.002–0.028，另外5个约0.950–0.972；不能凭比值0.017说整个属均高度相似。Lactiplantibacillus的神经分布也很宽。Megasphaera只有一个属内配对，无法据此估计稳定的属内分布。

另保存 `P(属内距离 < 属间距离) + 0.5 × P(相等)`，采用相同属间权重，仅用于描述分布重叠。例如Lactobacillus化学为0.532、Lactiplantibacillus神经为0.567，说明中位数比值不能替代全分布检查。该概率不是p值或分类准确率。

## 方法与有限检查

- 化学沿用当前新筛选的106×162表：原报告ng/mL浓度的 `log2(c / (1 ng/mL))`。距离是两株162项log差的均方根。无旧reference、加1、缺失插补或三项score。注释等权，未按跨株方差标准化、未按family重加权，因此变化幅度更大的注释贡献更多。
- 神经沿用106×13个有符号、SNR门槛处理后的unit系数，距离为 `1 - cosine`。表示保留组合方向并去掉整体共同增益；未重拟合模板，未扣除重复测量噪声。
- 所有表按菌株ID对齐；行顺序仅由株数和菌属名称决定。没有根据化学或神经结果挑选菌属，也没有恢复同日期／旧reference筛选。
- 属内配对等权；属间每个外属等权，外属内配对等权。分位数使用加权经验CDF反函数；累计权重恰达0.5时，中位数取相邻观测的中点，与普通偶数样本中位数一致。其他分位数不插值。
- 共享菌株的pairs不是独立重复，未进行把pairs视为独立样本的检验，也未给出假定独立的置信区间。

两个预先限定的检查仍只针对第1、2步：

1. **外属参照仅限多株属**：每个目标属改与其他12个多株属比较，仍各外属等权。两模态各12/13属中位数比值低于1；Lactobacillus化学为1.034，Streptococcus神经为1.052。总体方向保留，但这两项边界结果依赖参照范围。
2. **同模板未置零的神经系数**：使用已保存pre-gate unit表，不重新拟合模板。13/13属中位数比值仍低于1。它检查置零步骤，不等于独立重拟合的unfiltered模型，也未评估所有神经表示。

QC表保留每株原神经系数范数、非零坐标数、日期和每细胞记录动物数范围；这些动物数按细胞跨日期汇总，不应跨细胞相加当作独立动物数。原系数范数约0.289–2.906，非零坐标3–12，日期数1–2。图中的距离分布没有剥离神经测量误差；化学QC重复也不是菌株独立培养重复。

培养、介质、物种、记录日期等因素没有通过本次分层比较被全部控制。化学材料来自独立培养，并非实测神经刺激aliquot。本次既不证明化学导致神经变化，也未检验两种属间结构是否对应。

## 可复用入口与验证

| 文件 | 内容 |
| --- | --- |
| `code/notebook_cells.py` | Notebook可复制的只读加载、图像显示和结果单元 |
| `code/compare_genus_distances.py` | `run_analysis(root, out)`；要求新结果目录 |
| `code/plot_genus_comparison.py` | `make_figure(out)`；仅读取结果重新绘图 |
| `code/verify_genus_distances.py` | `check_results(root, out)`；独立NumPy复核 |
| `tables/genus_summary.csv` | 13属距离中位数、比值及分布比较概率 |
| `tables/distribution_summary.csv` | 全部分位数、均值、极值与配对数 |
| `tables/pair_catalogue.csv` | 全部5,565对的两模态距离及pre-gate距离 |
| `tables/comparison_membership_weights.csv` | 每属比较使用的全部配对和权重 |
| `tables/genus_coverage.csv` | 全29属覆盖与单株属处理 |
| `tables/sample_context_qc.csv` | 对齐分类、日期及神经QC信息 |
| `tables/sensitivity_external_multistrain.csv` | 多株外属参照检查 |
| `tables/sensitivity_neural_pre_gate.csv` | 同模板未置零检查 |
| `protocol.json`、`results.json` | 固定方法、输入哈希与描述性汇总 |
| `verification.json` | 实际运行的数值复核结果 |

使用仓库既有 `.pixi/envs/default/bin/python`；没有安装依赖或执行整个Notebook。独立复核从源表直接重建距离、计数、权重、中位数和分布比较概率，另由只读复核者核对两项敏感性结果。

初稿采用经验CDF的下中位数；方法复核后统一改为常规偶数中位数兼容约定，避免小样本/双峰分布的中心点被置于下端。初稿保存于 `audit/initial_lower_median`，用于追溯；最终图和表均为修正后结果。距离、样本、权重、其他分位数和分布比较概率没有变化。
