# 第4步：独立观察不同菌属的神经组合差异

本轮神经侧已经完成。90株、13个至少两株的菌属中，存在可描述的属平均神经组合差异，其中一些方向获得多数菌株支持；也有属内不一致，使平均图无法代表所有菌株。这是当前样本内的描述性结果，不是已经证明的普遍菌属规律。

本分析只读取保存的神经 unit 系数、同模板 pre-gate 系数、菌株分类/日期和神经支持审计。化学结果未参与样本挑选、参考构建、排序、坐标选择或解释；未进行第5步对应分析。

## 先看完整组合

![All 13 genus centers and all 13 neural coordinates](figures/01_neural_genus_centered_profiles.png)

每个格子是“属内菌株 unit 向量的均值 − 13属等权参考”。参考包含目标属，每属权重1/13；大属不会因株数多而主导参考。保留全部13类神经元，属中心未再次单位化，未对行做z-score。红/蓝只代表相对该参考的有符号模板系数偏移，不代表兴奋/抑制、响应占比或整体响应强弱。属中心长度同时保留属内方向集中程度；属中心Euclidean距离因此是mean-profile差异，不能称为纯方向距离。

属顺序只由神经中心的Euclidean差异、average linkage和optimal leaf ordering决定，树只用于排列，没有切出人为类别。每株原始unit系数以及绝对属均值也保存在表中，可区分“原始系数为负”和“相对参考偏低”。

## 当前样本中能描述的模式

- **Bacteroides** 的主要相对偏移为ADF +0.243、ASH +0.184、AWA +0.092。三个坐标分别有26/29、23/29、23/29株与属平均偏移同号。每次移除一株并重算参考后，29/29个留出菌株的完整13维偏移与其余菌株中心的余弦均大于0；中位数0.619，最低0.285。这说明该参考下的平均组合方向获多株支持，不能解释成每个坐标都一致或能够识别该属。
- **Bifidobacterium** 同样有ASH偏高的属平均，但ADF偏低、AWA偏高（−0.212、+0.202、+0.192分别为ADF、AWA、ASH）。对应坐标同号株数为10/11、8/11、8/11；完整组合的留出同向株数为9/11。它与Bacteroides的区别体现为组合中的不同坐标，并非整体强弱。内部仍有明显异质性。
- **Limosilactobacillus** 的属中心偏移较大，主要为AWCON +0.761、ADF −0.276、AWCOFF +0.238，完整留出组合同向5/6株；A249余弦为−0.090，和其他5株的主要方向不一致。其余5株余弦接近1，所以只报一个属均值会隐去这个例外。
- **Escherichia与Lactobacillus** 在ASH坐标的相对偏移呈相反方向，分别为−0.333与+0.368；各自5/5株的ASH方向与属均值同号，完整留出组合同样各5/5为正。两属在其他坐标也有差别，但株数只有5，不能据此宣称普遍的属特征。
- **Pediococcus与Streptococcus** 需要尤其保留株层信息：完整留出组合同向仅4/7和3/5。Pediococcus虽有AWCON +0.374的属均值，只有4/7株该坐标同号。删除任意一株后的属中心方向仍比较稳定，不能替代“每株都符合中心”的检查。
- **Megasphaera、Leuconostoc、Lacticaseibacillus** 可见较集中的组合，但分别只有2、3、3株。Megasphaera删除一株后仅剩一株，不能视为可靠的属内重复验证。

上述例子采用固定的“各属绝对偏移最大的3个坐标”来帮助阅读。分析和主图从未舍弃其他坐标。所有属同等保留在下表与完整图中。

| 菌属 | n | 最大3个坐标偏移（相对参考） | 完整留出组合同向 | 偏移长度 | 属内RMS散布 |
|---|---:|---|---:|---:|---:|
| Megasphaera | 2 | ASH +0.423、ASEL +0.404、ADL +0.360 | 2/2 | 0.767 | 0.211 |
| Leuconostoc | 3 | ASK +0.445、AWCON -0.302、AWA -0.247 | 3/3 | 0.683 | 0.328 |
| Lacticaseibacillus | 3 | AWCOFF -0.319、AWCON -0.282、ASH -0.150 | 3/3 | 0.521 | 0.361 |
| Escherichia | 5 | ASH -0.333、ADF +0.199、AWB +0.198 | 5/5 | 0.501 | 0.479 |
| Lactiplantibacillus | 4 | ASH -0.250、ADF +0.183、ASEL -0.113 | 3/4 | 0.368 | 0.633 |
| Bacillus | 4 | AWCON -0.328、ASEL -0.276、AWB +0.179 | 4/4 | 0.517 | 0.535 |
| Bacteroides | 29 | ADF +0.243、ASH +0.184、AWA +0.092 | 29/29 | 0.344 | 0.425 |
| Streptococcus | 5 | ADF +0.167、ASK -0.157、ASJ -0.131 | 3/5 | 0.329 | 0.618 |
| Enterococcus | 6 | AWB +0.184、ASH +0.137、AWCON -0.097 | 5/6 | 0.290 | 0.577 |
| Lactobacillus | 5 | ASH +0.368、ADF -0.100、AWB -0.076 | 5/5 | 0.409 | 0.451 |
| Bifidobacterium | 11 | ADF -0.212、AWA +0.202、ASH +0.192 | 9/11 | 0.398 | 0.675 |
| Pediococcus | 7 | AWCON +0.374、ASK +0.152、AWB -0.131 | 4/7 | 0.445 | 0.607 |
| Limosilactobacillus | 6 | AWCON +0.761、ADF -0.276、AWCOFF +0.238 | 5/6 | 0.909 | 0.523 |

“完整留出组合同向”指：留出菌株和剩余同属菌株的13维偏移向量余弦 > 0，分母为有定义的留出余弦数（本次全都有定义）。参考在每次留出后重新计算。“逐坐标同号”用固定完整参考，是另一项描述：单个坐标高于/低于参考的方向是否与属均值一致。sign(0)=0，只有符号完全相同计入；零符号没有方向意义。本轮169个属中心坐标没有恰好为零的偏移。二者均非p值、分类准确率或独立验证。

## 株层差异与固定敏感性检查

[完整90株热图](figures/02_neural_all_strains_centered_profiles.png)保留主图属顺序，属内再按神经相似性排列。其色阶为±1.0，属中心图为±0.8，不能跨图以颜色深浅直接比较系数大小。[留出一株的完整组合图](figures/03_neural_leave_one_strain_out.png)将“留出株是否同向”和“剩余属中心是否改变方向”分开显示。

固定一次leave-one-strain-out检查中，属中心方向余弦最低值介于0.803与0.998，但这只表示删除一株后的条件均值稳定性；属中心和留出菌株未使用独立实验模板，剩余参考也来自本数据。因此明确称为descriptive held-out alignment，不称为新增实验重复、泛化验证或预测性能。全部实际属中心偏移长度为0.290–0.909，未触发预先固定的0.05小向量解释保护；0.05不是生物阈值，也不决定保留哪些结果。

[同模板pre-gate检查](figures/04_neural_pre_gate_sensitivity.png)保留主图顺序与共同色阶，每个版本使用自己的等属参考。属中心偏移向量与主分析的余弦为0.928–0.997，说明平均组合的大方向大体保留。部分较小坐标会变号，单株一致性也明显更低：最小单株余弦为0.520（Bifidobacterium），因此不能表述为“所有菌株和坐标完全稳健”。此检查只比较既有SNR gate，不包含模板拟合、培养差异和抽样误差。

## 解释边界

- unit表示移除了整体gain；此处的规律是神经组合的相对结构，不能推断整体反应强弱。
- SNR gate生成的0不能当作没有生物反应；相同模板的敏感性也不是全流程重拟合。
- 属内菌株不是新增独立培养重复。多株属的数量为2–29不等，某些属的全部菌株集中在一个神经记录日期（Bacillus、Lacticaseibacillus、Megasphaera），同属种组成和日期仍可能影响模式。本轮未分离属、种、培养与记录日期的因果贡献。
- 原始覆盖包含29属106株；16个单株属仅记入覆盖表，不定义参考、不用于属内重复或挑选模式。
- 这里只完成第4步的独立描述。没有赋予通路、机制、行为或化学对应意义，也没有检查两种模态是否保留相同属间结构。

## 文件与复现

- [预先记录的方案](PROTOCOL.md)；[来源与SHA-256](source_manifest.json)；[参数、版本和确切顺序](parameters.json)。
- [逐属完整摘要](tables/genus_summary.csv)、[169个属×神经元完整统计](tables/genus_neuron_summary.csv)、[90次留出记录](tables/leave_one_strain_out.csv)、[同模板pre-gate敏感性](tables/pre_gate_sensitivity_summary.csv)。
- [分析成员与日期](tables/main_membership.csv)、[16个单株属覆盖](tables/singleton_coverage_only.csv)、[神经支持审计](tables/neural_qc_audit.csv)。
- [全部78个属间均值差向量](tables/all_genus_pair_contrasts.csv)保留13坐标。距离只表示未再归一化mean-profile差异。
- [图注](figures/CAPTIONS.md)、[数值独立复算](verification/numerical_verification.json)、[图像检查](verification/visual_review.md)。

默认在现有Notebook中使用总目录的 `code/notebook_cells.py` 只读显示入口；它读取已保存结果，不重新计算或覆盖本轮科学结果。无需新建Notebook或安装依赖。

只有确需精确重算时，才调用下面的小函数并指定一个**尚未产生结果的新输出目录**：

```python
import importlib.util
from pathlib import Path
repo = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
script = repo / 'reports/exploration_genus_patterns_independent_20261003/neural/code/analyze_neural.py'
spec = importlib.util.spec_from_file_location('independent_genus_neural', script)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
result = analysis.run_analysis(repo, out=repo / 'reports/neural_step4_recompute_new')
result['genus_summary']
```

`run_analysis` 在目标已存在tables或核心结果时抛出 `FileExistsError`，保护已保存结果。只需重画时使用 `analysis.redraw_saved_figures(saved_result_dir, fresh_figure_output_dir)`；它只读保存的神经表，向另一个新目录输出图，不重算统计或改写原结果。`code/verify_neural.py` 可对原结果独立核验。

独立核验没有调用主分析函数，直接从输入CSV重算reference、全部中心/逐坐标分布、90次LOSO、敏感性与78个属间差值；全部通过，最大绝对数值差4.44×10⁻¹⁶。4张PNG均已打开人工视觉检查，SVG同时提供以便后续排版。
