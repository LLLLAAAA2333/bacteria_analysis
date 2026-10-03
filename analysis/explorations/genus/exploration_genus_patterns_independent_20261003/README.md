# 第3、4步：化学与神经的属间规律，独立分析

日期：2026-10-03。实际工作仓库：`/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis`。

用户要求第3、4步互不影响、同步推进。本轮分别完成化学组合探索与神经组合描述。两边都使用13个多株属、90株；纳入规则仅来自菌属覆盖，16个单株属另行保留覆盖记录。两边使用各自的数据定义组合、参考、排序与检查，不交换化学模块、神经坐标选择或排列顺序，没有执行第5步对应分析。

这里的“独立”指分析互不作为对方的筛选条件，不表示独立培养实验、新样本验证或两个统计独立的数据集。

## 第3步：化学

![Chemical-only module profiles](chemical/figures/01_chemical_module_centers.png)

162项化学注释中，按预先固定的一种相关距离与切割规则，整理出15组共同变化组合，覆盖139项；23项未分组注释仍保留在完整表和支持图中。C01–C15是本轮新定义的编号，与旧三项化学score的M11不同。

每种化合物先计算13属的平均log2浓度，以13属等权均值为参考，再除以13属均值的样本标准差。模块用属均值间Pearson相关的average linkage，距离`1-r`、切割距离0.5，每组至少3项注释且至少3个Mass-column family。组合分数先family内等权、再family间等权。主图颜色是这种标准化组合分数，不是ng/mL或log2倍数变化；原始log2浓度差完整保存。

当前样本内的具体例子：

- **Bacteroides** 的C02、C04平均较高，分别有27/29、28/29株与此方向一致；C07、C08平均较低，分别为28/29、27/29株。它表现为不同化学组合的高低差异，不能简化为所有化学物质一起增加或减少。
- **C07** 包括2-Hydroxycaproic acid、Hydroxyisocaproic acid、Phenyllactic acid等7项注释；**C08** 包括Aspartic acid、Glutamic acid、Oxalic acid等6项。**C02** 包括Glucaric acid、Azelaic acid、p-Cresyl sulfate等14项；**C04** 包括Hydroxypropionic acid、Homoserine、Threonine、Gamma-Aminobutyric acid、Lactate等9项。这些名称描述成员，不把一组命名为已验证通路。
- **Escherichia** 的C05较高、C06较低，各有5/5株同向。C05含Glyceraldehyde、Methylmalonic acid、Succinic acid等8项；C06含Lysine、7-methylguanosine、CDP等7项。**Enterococcus** 的C11较低，6/6株同向；其4项成员为Pentadecanoic acid、Tyrosine、Glucosamine、D-Mannosamine。

这些方向均相对于本轮13属等权参考，不意味着该属特有，也不意味着每个模块成员在每株中都一致。整个模块分数同向与各成员同向分别保存在表中。

属间共变尤其不能直接解释成属内共变。例如C04跨family成员对的相关中位数，在13属均值间为0.598，去除各属均值后的属内残差中仅0.084。模块是在属均值上形成的，这些相关来自同一数据，不是独立验证。固定模块和尺度的删一株检查也未验证模块发现本身的稳定性。

[化学完整说明与成员](chemical/README.md) · [90株化学组合图](chemical/figures/02_chemical_modules_all_strains.png) · [逐属方向一致性](chemical/tables/module_direction_consistency.csv) · [全部162项属均值log2差](chemical/tables/feature_genus_log2_difference_162x13.csv)

## 第4步：神经

![Neural-only genus profiles](neural/figures/01_neural_genus_centered_profiles.png)

每个格子是该属的平均unit系数减去13属等权参考。全部13类神经元保留，不逐行z-score、不把属均值重新单位化，排序仅依据神经属均值的Euclidean差异。红、蓝表示相对参考的有符号系数偏移，不能翻译为兴奋／抑制、贡献百分比或整体反应强弱。

当前样本内存在获得多株支持的组合方向，也有平均图掩盖个体不一致的情况：

- **Bacteroides** 相对参考主要表现为ADF +0.243、ASH +0.184、AWA +0.092。每次删去一株、重估其余同属中心及13属参考后，29/29个留出株的完整13维偏移与其余同属平均方向同向。逐坐标同号株数分别为26/29、23/29、23/29；整体同向不意味着每个坐标都一致。
- **Limosilactobacillus** 主要表现为AWCON +0.761、ADF −0.276、AWCOFF +0.238，但完整留出组合同向为5/6株，A249是例外。
- **Pediococcus、Streptococcus** 的完整留出组合同向仅4/7、3/5株。删去一株后平均方向仍相近，不等于每株符合该平均模式。

“完整留出组合同向”是两个相对参考的13维向量cosine大于0，属于本数据内的描述性对齐，不是分类准确率、属特异性证明、新的实验重复或全流程独立验证。相同模板pre-gate检查中，属中心方向cosine为0.928–0.997，但个体最低为0.520，因此不能概括成所有个体或坐标都稳健。

[神经完整说明与13属摘要](neural/README.md) · [90株神经组合图](neural/figures/02_neural_all_strains_centered_profiles.png) · [留出一株检查](neural/figures/03_neural_leave_one_strain_out.png) · [逐属完整表](neural/tables/genus_summary.csv)

## 使用与解释范围

- 两张主图的菌属顺序独立确定，化学与神经的色阶定义也不同；位置或颜色不能用于直接判断对应。
- 当前结果描述这90株中的属相关组合差异，不能把2–3株支持的小属结果当作普遍规律。种组成、培养材料、日期和神经表示误差未全部控制。
- 化学材料来自独立培养，并非实测神经刺激aliquot；本轮没有赋予神经差异化学、行为或机制意义。
- 现有Notebook可复制 [`code/notebook_cells.py`](code/notebook_cells.py) 只读加载两边主图、个体图与留出检查；不会重新执行分析。
- 两边各自保存方案、来源哈希、完整表、代码及独立数值复核。另由只读复核者独立检查来源隔离、中心／尺度／权重、模块成员、排序、LOSO与敏感性结果。总入口的来源／队列核对见 `integration_verification.json`。

没有修改原始数据、已有Notebook或前两步结果，也没有提交Git。
