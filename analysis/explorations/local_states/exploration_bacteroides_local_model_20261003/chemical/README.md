# Bacteroides 内部的独立化学候选轴

化学侧仅使用29株Bacteroides的162项log2浓度与注释/分类信息，重新定义局部组合；未读取神经数值、神经排序、模型结果或历史模块成员。神经方向、单轴选择和交叉验证由父分析独立负责。

**按预先固定规则得到12个局部候选轴L01–L12，覆盖128项注释，34项未分组。** 全部162项在这29株中均非数值常量。L编号仅属于当前一次拟合；既不继承此前C组，也不能按相同编号直接对应另一训练折的成员。

![Local chemical axes](figures/01_local_chemical_axes.png)

每列是一株，每行是一个局部化学组。红/蓝表示相对本29株均值的标准化组合分数，不表示实际浓度或log-fold change。每项按训练株均值和样本SD标准化，再按family等权、family内成员等权求组分数；行列顺序都由化学数据确定。图保留全部12组和全部29株，未按神经关系筛选。

局部数据并不都呈平滑分布：A021在L05为+4.260、L07为−4.547，A044在L04为+3.461。这里只指出化学分数的单株极端，不能由此推断异常原因或神经意义。部分full29组合可能依赖少数菌株，因此其成员和尺度不能直接用于held-out预测。

## 全部候选轴

代表注释固定按与自身组分数的Pearson相关从高到低列前三名，供查读成员，未用作功能或通路命名。相关中位数仅汇总跨Mass-column family成员对；原尺度SD列用于提醒不同组的标准化色彩不代表相同绝对变化。

| 轴 | 注释 / family | 前三代表注释 | 跨family相关中位数 | 成员原log2 SD中位数 |
|---|---:|---|---:|---:|
| L01 | 47 / 41 | Alloisoleucine；Tryptophan；Betaine | 0.753 | 0.397 |
| L02 | 19 / 19 | 3-Methylhistidine；Methylguanidine；Cystathionine | 0.765 | 0.606 |
| L03 | 14 / 13 | p-Cresol；N-Acetylmethionine；N-Formyl-methionine | 0.608 | 0.497 |
| L04 | 11 / 7 | Stearic acid；Arachidic Acid；Erucic Acid | 0.835 | 0.410 |
| L05 | 6 / 6 | 2-Hydroxycaproic acid；Hydroxyisocaproic acid；Phenyllactic acid | 0.752 | 0.725 |
| L06 | 6 / 6 | Sebacic acid；Methionine；Leucyl-Glycine | 0.648 | 0.538 |
| L07 | 5 / 5 | Arginine；Carnitine；Acetylcarnitine | 0.799 | 0.894 |
| L08 | 5 / 5 | Asymmetric dimethylarginine；Oxalic acid；N-Acetylaspartic acid | 0.669 | 0.334 |
| L09 | 5 / 5 | N-Alpha-acetyllysine；Gamma-Aminobutyric acid；Lactate | 0.631 | 0.500 |
| L10 | 4 / 4 | Ophthalmic acid；Aminoadipic acid；Serine | 0.742 | 0.389 |
| L11 | 3 / 3 | 3-Hydroxymethylglutaric acid；Cysteic acid；Glycolic acid | 0.583 | 0.316 |
| L12 | 3 / 3 | Asparagine；Methylsuccinic acid；Tyrosine | 0.559 | 0.552 |

平均连接法的固定距离0.5切割不保证组内每对相关都≥0.5。相关和组本身均来自同一份化学数据，不构成外部验证；没有对不同cut、其他距离、PCA或额外descriptor作搜索。

## 训练折API

`code/local_chemical_axes.py`提供无文件读写的纯函数：

```python
fitted = fit_axes(train_log_frame, feature_metadata)
train_scores = fitted["train_scores"]
heldout_scores = transform_axes(heldout_log_frame, fitted)
```

每次fit仅以传入训练行重新估计每项均值/样本SD、相关、分组及family权重。完整数据列需要保留；不能从full29已分好的成员中挑选，再称为训练折发现。transform只套用该fit的参数，保持输入行顺序和训练L列顺序，不重新聚类或标准化。无合格组时返回n×0 DataFrame，不生成替代轴。至少需要两株训练样本；数值常量定义为SD≤1e−12。

返回字典包含`means`、`scales`（全部特征的Series）、`source_features`、`retained_features`、`excluded_features`、`train_ids`、按L编号有序的`module_members`、各成员索引的`score_weights`、`feature_order`、`parameters`和`train_scores`。metadata可把metabolite作为列或索引；所有输入ID必须唯一、数值有限、family非空。

## 来源与表格

来源仅为`reports/exploration_chemical_pattern_direct_report_20261003/tables/`中的三个文件：fresh_chemical_log2.csv、fresh_feature_metadata.csv、sample_context.csv。来源按strain和metabolite ID对齐，SHA256保存在manifest。保持原log2(c / 1 ng/mL)，没有+1、补零或重新挑选面板。

29株源记录包含16个species标签；6株taxonomy_note非空：A006、A025、A026、A040、A041、A044。它们均保留，taxonomy_flag仅指该字段非空，不解释为确定错误。species、dates和原始taxonomy_note逐株原样导出；dates是源context记录，未据此认定化学培养/测量批次。

| 文件 | 内容 |
|---|---|
| `tables/chemical_log2_29x162.csv`、`chemical_standardized_29x162.csv` | 全部162项原始/标准化逐株值 |
| `tables/bacteroides_context_29.csv` | species、dates、taxonomy_note和派生flag |
| `tables/local_feature_scales_metadata.csv` | 每项训练均值、样本SD、原log2分位数/范围和注释 |
| `tables/local_module_members.csv` | 所有成员、family权重与原尺度幅度 |
| `tables/local_axis_scores_29.csv`、`local_axis_scores_with_context.csv` | 全部12轴×29株分数 |
| `tables/local_module_pair_correlations.csv`、`local_module_summary.csv` | 全部组内相关及汇总 |
| `tables/local_module_representatives.csv`、`local_module_annotation_composition.csv` | 代表注释排序与注释类别 |
| `tables/ungrouped_features.csv`、`constant_features.csv` | 未分组/常量注释（常量表为空） |
| `orders.json`、`manifest.json`、`protocol.md`、`verification.json` | 成员与排序、来源/代码哈希、固定方案、复核 |

## 重现与复核

使用既有`.pixi/envs/default/bin/python`。`code/run_local_chemical_analysis.py`内的`run_analysis(repo_root, out=None)`用于主动重算描述性输出，若输出目录已有核心结果即拒绝覆盖；重新计算需显式指定新的out目录。现有结果可直接显示主图/读取表格，未创建或执行Notebook。

`code/verify_local_chemical_axes.py`从源表独立重建原始尺度、相关/模块分区、family权重、所有分数、完整覆盖、排序和分类字段；再检验真实28株训练与held-out变换、常量/无轴、行列重排、输入/fit对象不变、最低family数和缺失列报错。**PASS**，最大绝对误差5.67e-15。主图已视觉检查；入口拒绝覆盖已检查。

full29轴只用于描述和核对；它们在预测中的含义取决于父分析从训练折重新发现、选轴及held-out检验的结果。本报告不判断哪个轴最能解释神经，也不报告预测有效性。属内物种结构、培养/测量差异、报告注释冗余、面板选择和29株样本量限制了进一步解释；这里没有通路或因果结论。

## 完整成员

**L01**：Ricinoleic acid；Phenylethylamine；Erythronic acid；Glucaric acid；Acetylglycine；Glycine；Beta-Alanine；Alanine；Creatine；2-Hydroxybutyric acid；3-Hydroxybutyric acid；Alpha-Hydroxyisobutyric acid；Alloisoleucine；Alpha-aminobutyric acid；Quinolinic Acid；Homocitrulline；Vitamin B2；Nicotinic acid(VB3)；Pantothenic acid(VB5)；Isoleucine；Leucine；Acetylcholine；Trimethylamine；Betaine；Choline；Creatinine；Hydroxyproline；Norleucine；Phenylalanine；Threonine；Tryptophan；N-Acetylalanine；Sarcosine；Homoserine；Hydroxypyruvic acid；5-Methoxytryptamine；1-Methylhistidine；4-Pyridoxic acid；Phosphorylcholine；N-Acetylglutamic acid；Aniline-2-sulfonate；N-Formylglycine；Trigonelline；p-Aminobenzoic acid；Pterin；Diethanolamine；Hypoxanthine。

**L02**：Azelaic acid；Indoleacetic acid；p-Cresyl sulfate；Pentadecanoic acid；Indole-3-methyl acetate；Hydroxypropionic acid；Quinic acid；Anserine；Taurine；Allantoin；Gama-glutamylalanine；Glucosamine；Methylguanidine；Deoxyuridine；4-Acetamidobutanoic acid；Cystathionine；D-Mannosamine；5-Aminopentanoic acid；3-Methylhistidine。

**L03**：Indole-3-carboxylic acid；Indoxyl sulfate；Indole-3-carboxaldehyde；3-Methoxytyramine；p-Cresol；N-Acetylmethionine；N-Formyl-methionine；N-Acetylphenylalanine；2-Hydroxyphenethylamine；N-Acetylleucine；Adenosine-5′-diphosphate(ADP)；Adenosine 2',3'-cyclic phosphate；Cytidine 5'-diphosphate（CDP）；2'-Deoxycytidine 5'-monophosphate(dCMP)。

**L04**：Conjugated linoleic acids (CLA)；Petroselinic acid；Erucic Acid；Elaidic Acid；Lignoceric Acid；Linoleic acid；Linoelaidic acid；Oleic acid；Palmitic acid；Stearic acid；Arachidic Acid。

**L05**：2-Hydroxy-3-methylbutyric acid；2-Hydroxycaproic acid；p-Hydroxymandelic acid；Phenyllactic acid；Hydroxyisocaproic acid；4-Trimethylammoniobutanoic acid。

**L06**：Sebacic acid；cis-Aconitic acid；Methionine；Leucyl-Glycine；Proline；Glucose 1-phosphate。

**L07**：Citrulline；Arginine；Carnitine；Acetylcarnitine；Trimethyllysine。

**L08**：Oxalic acid；Aspartic acid；Glutamic acid；N-Acetylaspartic acid；Asymmetric dimethylarginine。

**L09**：Gamma-Aminobutyric acid；Lactate；O-Succinyhomoserine；N-Alpha-acetyllysine；S-Carboxymethyl-cysteine。

**L10**：Aminoadipic acid；Serine；Galacturonic acid；Ophthalmic acid。

**L11**：Glycolic acid；3-Hydroxymethylglutaric acid；Cysteic acid。

**L12**：Methylsuccinic acid；Asparagine；Tyrosine。

