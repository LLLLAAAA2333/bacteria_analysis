# 第3步：菌属间化学组合的独立描述

本分析仅使用化学表和菌株分类，未读取神经数值，也未借用神经排序、筛选或结果。第3步与第4步独立推进；本报告不进行第5步对应关系检验。

**结果：多个菌属在若干化学组合上有在当前已测菌株中反复出现的方向差异，同时保留明显的属内异质性。** 固定规则得到15个描述性共变组，覆盖139/162项报告注释，余下23项完整保留。聚类本身总能形成分组，因此“15组”不是存在15个天然功能模块的证据；还必须结合单株方向、成员方向、原始幅度及删株敏感性阅读。

这些新组命名为 **C01–C15**，仅适用于本轮按13属化学均值定义的分析，不能与历史M编号或三项score混用。

![Chemical module centers](figures/01_chemical_module_centers.png)

红/蓝表示高于/低于13属等权参考的标准化均值。每属先取菌株log2均值；每项再用13属中心的样本SD标准化。模块分数先在Mass-column family内等权，再在family间等权。色值无单位，不是浓度或log-fold change。属顺序来自全部162项化学中心；共变组顺序也只用化学数据。所有15组和13属均展示，不以支持率隐藏结果。

## 多株支持与不一致的部分

- 株数最多的 **Bacteroides (n=29)** 同时存在不同方向的组合：C02（14项，含p-Cresyl sulfate、Glucaric acid、Azelaic acid）较高，27/29株同向；C04（9项，含Hydroxypropionic acid、Homoserine、Threonine，也含GABA和Lactate）较高，28/29株同向。C07（7项，含2-Hydroxycaproic acid、Hydroxyisocaproic acid、Phenyllactic acid）较低，28/29株同向；C08（6项，含Asymmetric dimethylarginine、Aspartic acid、Oxalic acid）较低，27/29株同向。四个组合的全部成员属均值均与组合方向一致，逐一删株后的属中心均保持方向。
- **Escherichia (n=5)** 的C05（8项，代表Glyceraldehyde、Methylmalonic acid、Succinic acid）较高，而C06（7项，代表CDP、Lysine、7-methylguanosine）较低；两组均5/5株同向，全部成员属中心也同向。**Enterococcus (n=6)** 的C11（Pentadecanoic acid、Tyrosine、Glucosamine、D-Mannosamine）较低，6/6株同向。
- **Bifidobacterium (n=11)** 的C15（三项N-Acetylglutamic acid、N-Acetylaspartic acid、N-Alpha-acetyllysine）较高，11/11株同向，全部成员属中心同向，逐一删株后保持方向。这只是报告注释构成的观测组合，不解释为已验证的生物通路。
- 同时有许多不一致单元。195个属×组单元中，100个满足预先约定的描述条件：至少75%菌株和family加权后至少75%成员中心同向，且所有删株中心保留方向；其中77个的|标准化组均值|≥0.5。这不是显著性统计、独立验证或天然分类标准，其余95个单元仍完整显示。
- **属间共变不等于属内共变。** C04跨family成员相关的中位数在13属中心间为0.598，去属均值后的属内残差相关仅0.084；C12分别为0.544和0.031。相反，C09与C10的属内相关中位数分别为0.724、0.719。不能把所有组一概解释为逐株共同变化。

## 全部15组的可查读标签

前三注释固定按与本组13属profile的相关性由高到低列出，仅作查读标签；完整成员在下方及表内。最高/最低属只表示该组均值在13属中的极值，括号为同方向株数/总株数，不能把极值自动视为稳定规律。

| 组 | 注释 / family | 前三代表注释 | 均值最高属：同向株数 | 均值最低属：同向株数 | 属间 / 属内相关中位数 |
|---|---:|---|---|---|---|
| C01 | 51 / 42 | Leucine；Isoleucine；Tryptophan | Enterococcus: 5/6 | Leuconostoc: 3/3 | 0.678 / 0.415 |
| C02 | 14 / 14 | p-Cresyl sulfate；Glucaric acid；Azelaic acid | Bacteroides: 27/29 | Limosilactobacillus: 6/6 | 0.775 / 0.390 |
| C03 | 11 / 11 | S-Carboxymethyl-cysteine；4-Pyridoxic acid；O-Succinyhomoserine | Lacticaseibacillus: 3/3 | Leuconostoc: 2/3 | 0.613 / 0.468 |
| C04 | 9 / 9 | Hydroxypropionic acid；Homoserine；Threonine | Bacteroides: 28/29 | Megasphaera: 2/2 | 0.598 / 0.084 |
| C05 | 8 / 8 | Glyceraldehyde；Methylmalonic acid；Succinic acid | Escherichia: 5/5 | Pediococcus: 7/7 | 0.646 / 0.317 |
| C06 | 7 / 7 | Cytidine 5'-diphosphate（CDP）；Lysine；7-methylguanosine | Bifidobacterium: 11/11 | Escherichia: 5/5 | 0.632 / 0.225 |
| C07 | 7 / 7 | 2-Hydroxycaproic acid；Hydroxyisocaproic acid；Phenyllactic acid | Lactiplantibacillus: 4/4 | Bacteroides: 28/29 | 0.763 / 0.469 |
| C08 | 6 / 6 | Asymmetric dimethylarginine；Aspartic acid；Oxalic acid | Pediococcus: 7/7 | Bacteroides: 27/29 | 0.760 / 0.314 |
| C09 | 5 / 5 | Aminoadipic acid；Indoxyl sulfate；p-Cresol | Enterococcus: 4/6 | Bacillus: 4/4 | 0.706 / 0.724 |
| C10 | 5 / 5 | Anserine；Methylguanidine；Indole-3-methyl acetate | Pediococcus: 7/7 | Enterococcus: 4/6 | 0.817 / 0.719 |
| C11 | 4 / 4 | Tyrosine；D-Mannosamine；Glucosamine | Megasphaera: 1/2 | Enterococcus: 6/6 | 0.745 / 0.435 |
| C12 | 3 / 3 | 2,6-Diaminopimelic acid；Pterin；Sebacic acid | Bacillus: 4/4 | Limosilactobacillus: 6/6 | 0.544 / 0.031 |
| C13 | 3 / 3 | Cystathionine；3-Methylhistidine；Taurine | Lactiplantibacillus: 4/4 | Limosilactobacillus: 6/6 | 0.765 / 0.651 |
| C14 | 3 / 3 | Acetylglycine；N-Acetylserine；Acetylcholine | Lactiplantibacillus: 4/4 | Leuconostoc: 3/3 | 0.684 / 0.700 |
| C15 | 3 / 3 | N-Acetylglutamic acid；N-Acetylaspartic acid；N-Alpha-acetyllysine | Bifidobacterium: 11/11 | Escherichia: 5/5 | 0.666 / 0.409 |

相关汇总仅使用跨family成员对；逐株和属内残差相关中每属总权重相同，每株权重1/(13×该属株数)。所有成员对的完整相关表也保留。平均连接法的0.5距离切割不保证模块内每一对相关都≥0.5；例如C01存在少数负相关成员对。

## 逐株与完整面板支持

![All strain scores](figures/02_chemical_modules_all_strains.png)

每列一株，属内按菌株ID排列，宽度反映株数；数值/成员/参考与主图完全相同。共用主图±3色限，超限单株会饱和；精确值保存在`strain_module_scores.csv`。

![Direction fractions](figures/03_module_strain_direction.png)

每个百分比的分母是该属全部菌株，方向是相对此处13属等权参考的正负，零值仍在分母。这里的“同向”不表示化合物被刺激而上调/下调，更不表示相对培养基空白的生成/消耗。

完整162项中心（含23项未分组）见[支持图04](figures/04_all_162_chemical_centers.png)。原始log2均值、相对于等属参考的log2差、每株原值与组内分布全部保留；标准化可能放大小的属间差，不能根据颜色深浅比较不同分子的绝对浓度变化。各组成员跨属均值范围的中位数在C01约0.989 log2单位、C04约4.514 log2单位，说明相似的标准化色彩并不意味着相同原尺度幅度。

## 输入、方法和证据边界

输入为当前新筛选的106×162 log2浓度表，保留所有13个至少两株的属，共90株。16个单株属只列覆盖，不用于定义或选择模式。使用完整162项当前面板，没有三项score、加1、补零、重跑QC筛选或预测模型。

化学模块以13属中心的Pearson相关距离1-r做average linkage，固定切割0.5，要求≥3注释且≥3不同Mass-column family；没有比较其他阈值。各family内可能存在无法独立解释的报告注释，本轮通过family等权减少重复注释支配，仍不能保证全部注释独立。

LOSO逐一移除每株，重算该属中心和13属等权参考，固定模块及标准化尺度。它检验观测中心的条件敏感性，未检验模块发现稳定性，也不是独立验证。目标属仍参与参考；同一份数据定义并描述模块。n=2或3属即使100%同向也证据有限。菌株数不平衡、属内物种结构、培养/测量差异及缺少独立化学培养重复均限制结论。这里描述的是当前完整且QC合格的162项面板，不能推广为全部代谢物，也没有功能或机制归因。

## 主要文件

| 文件 | 内容 |
|---|---|
| `tables/feature_genus_mean_log2_162x13.csv` | 所有162项×13属的原始log2均值 |
| `tables/feature_genus_log2_difference_162x13.csv` | 相对13属等权参考的原始log2差 |
| `tables/feature_genus_standardized_162x13.csv` | 所有标准化属中心 |
| `tables/strain_log2_90x162.csv`、`strain_standardized_90x162.csv` | 全部90株逐项值 |
| `tables/genus_feature_raw_log2_distribution.csv`、`genus_feature_standardized_distribution.csv` | 每属每项的n、SD、分位数、范围 |
| `tables/module_members.csv`、`module_representative_annotations.csv` | 全成员、family权重及代表标签排序 |
| `tables/genus_module_centers.csv`、`strain_module_scores.csv`、`genus_module_distribution.csv` | 各组属均值、逐株分数及组内分布 |
| `tables/module_direction_consistency.csv`、`module_leave_one_strain_out.csv` | 全部方向条件与1350个删株结果 |
| `tables/module_pair_correlations.csv`、`module_summary.csv` | 属间、逐株、去属均值后的相关诊断 |
| `tables/feature_reference_scale_metadata.csv` | 每项参考、原尺度SD/范围、元信息 |
| `orders.json`、`manifest.json`、`protocol.md`、`verification.json` | 精确顺序/成员、哈希/版本、方案和复核 |

分布表的positive/negative计数总是针对该表所保存尺度的0；原始log2表的0对应1 ng/mL，不是等属参考。相对于等属参考的方向应查standardized表或module_direction_consistency表。

## 运行与复核

默认在既有Notebook中使用上一层`code/notebook_cells.py`的只读显示入口查看已保存结果，不重新执行计算。需要精确重算时导入本目录小函数入口并指定新的输出目录：

```python
from pathlib import Path
import importlib.util
repo = Path("/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis")
path = repo / "reports/exploration_genus_patterns_independent_20261003/chemical/code/analyze_chemical_patterns.py"
spec = importlib.util.spec_from_file_location("chemical_patterns", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.run_analysis(repo, out=repo / "reports/exploration_genus_patterns_independent_20261003/chemical_recompute_new")
```

已有核心结果时入口拒绝覆盖；请为每次主动重算使用新的out路径。使用仓库既有`.pixi/envs/default/bin/python`，没有安装依赖或运行整个Notebook。

独立重算脚本`code/verify_chemical_patterns.py`未导入分析函数，直接从源表重建中心、参考、尺度、family权重、模块成员、全部排序、单株分数、方向计数、加权相关，以及1350个显式删株中心。结果 **PASS**，最大绝对误差4.34e-14；输入哈希不变。主图和逐株图已人工视觉检查。输出保护在数值计算后追加，仅改入口安全，manifest保留其说明及计算时/最终代码哈希。

## 完整组成员

**C01**：Conjugated linoleic acids (CLA)；Petroselinic acid；Indole-3-carboxylic acid；Phenylethylamine；Erucic Acid；Elaidic Acid；Lignoceric Acid；Linoleic acid；Linoelaidic acid；Oleic acid；Palmitic acid；Stearic acid；Arachidic Acid；N-Acetylmethionine；Glycine；Beta-Alanine；Alanine；Methylsuccinic acid；Creatine；3-Hydroxybutyric acid；Alloisoleucine；Alpha-aminobutyric acid；Carnosine；Histidine；Quinolinic Acid；Vitamin B2；Nicotinic acid(VB3)；Isoleucine；Leucine；Trimethylamine；Betaine；Choline；Creatinine；Carnitine；Hydroxyproline；Methionine；Norleucine；Phenylalanine；Proline；Tryptophan；Sarcosine；Hydroxypyruvic acid；Glucuronolactone；1-Methylhistidine；3-Hydroxymethylglutaric acid；Adenosine 2',3'-cyclic phosphate；Phosphorylcholine；Histamine；N-Formylglycine；Ophthalmic acid；Trigonelline。

**C02**：Azelaic acid；Vitamin B7；3-Methoxytyramine；p-Cresyl sulfate；Lumichrome；2-Hydroxyphenethylamine；Glucaric acid；Arginine；Vitamin B1；Nicotinuric acid；Trimethyllysine；O-Phosphoethanolamine；5-Aminopentanoic acid；Glycerol 3-phosphate。

**C03**：Erythronic acid；Glyceric acid；Citrulline；Glycolic acid；Allantoin；Pipecolic acid；O-Succinyhomoserine；4-Pyridoxic acid；4-Acetamidobutanoic acid；Cysteic acid；S-Carboxymethyl-cysteine。

**C04**：Gamma-Aminobutyric acid；Hydroxypropionic acid；Quinic acid；Pantothenic acid(VB5)；Lactate；Threonine；Homoserine；cis-4-Hydroxy-D-proline；Beta-Glycerophosphoric acid。

**C05**：Methylmalonic acid；Glyceraldehyde；Succinic acid；N-Acetylalanine；4-Trimethylammoniobutanoic acid；Aniline-2-sulfonate；p-Aminobenzoic acid；Hypoxanthine。

**C06**：Adenosine-5′-diphosphate(ADP)；Lysine；Serine；7-methylguanosine；Acetylcarnitine；Cytidine 5'-diphosphate（CDP）；2'-Deoxycytidine 5'-monophosphate(dCMP)。

**C07**：2-Hydroxy-3-methylbutyric acid；2-Hydroxycaproic acid；p-Hydroxymandelic acid；Phenyllactic acid；2-Hydroxy-4-(methylthio)butanoic acid；Hydroxyisocaproic acid；Galacturonic acid。

**C08**：4-Hydroxybenzaldehyde；Oxalic acid；Asparagine；Aspartic acid；Glutamic acid；Asymmetric dimethylarginine。

**C09**：Indoxyl sulfate；p-Cresol；Aminoadipic acid；Homocitrulline；5-Methoxytryptamine。

**C10**：Indole-3-methyl acetate；Anserine；Gama-glutamylalanine；Methylguanidine；meso-Tartaric acid。

**C11**：Pentadecanoic acid；Tyrosine；Glucosamine；D-Mannosamine。

**C12**：Sebacic acid；Pterin；2,6-Diaminopimelic acid。

**C13**：Taurine；Cystathionine；3-Methylhistidine。

**C14**：Acetylglycine；Acetylcholine；N-Acetylserine。

**C15**：N-Acetylaspartic acid；N-Alpha-acetyllysine；N-Acetylglutamic acid。

