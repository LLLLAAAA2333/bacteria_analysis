# 无基因组条件下的代谢足迹嵌入思路

日期：2026-05-29
项目语境：线虫对细菌培养上清代谢组的神经响应

## 核心立场

我们要构建的对象不应该是一个高维化学丰度空间，而应该是一个
**观察到的代谢足迹**（observed metabolic footprint）：细菌相对于空白培养基消耗了哪些代谢物、积累了哪些代谢物，以及这些变化如何组成可解释的模块。

这个区分非常重要，因为当前没有基因组数据。没有基因组数据时，我们不能说某一类细菌“具有”某条代谢通路，或者某条通路在遗传/酶学层面被激活。我们能说的是一个更谨慎、但对本项目更有用的命题：实际测到的培养基变化呈现出某种带方向的生化足迹，这种足迹可以通过化学类别、反应邻近性、通路成员关系和经验共变结构来组织。

因此，这个 embedding 应该回答：

> 哪些可解释的代谢足迹模块区分了不同细菌样本？这些模块化后的化学空间，是否比原始高维代谢物距离更好地对应线虫神经响应结构？

它不应该回答：

> 每种细菌具体编码了哪些代谢通路，或者哪些通路在细胞内因果激活？

后一个问题需要基因组、转录组、同位素流量追踪、酶学实验或靶向验证。

## 为什么 `log2FC + Euclidean` 不是合适的化学空间

fold-change 本身是合理的。它把绝对丰度转成相对于空白培养基的上调/下调，这和神经响应的实验对比是匹配的：线虫响应的是被细菌改变后的样本，而不是孤立的绝对培养基化学组成。

问题在 Euclidean。

在 200 多个代谢物构成的空间里，欧几里得距离默认每个代谢物轴都是独立的、等权的、正交的、可以直接相加比较的。但这对代谢组并不成立。代谢物之间有前体-产物关系、通路共属关系、化学类别相似性、共同的缺失/QC 行为，以及真实生物调控导致的相关变化。有些代谢物可能对线虫感知没有意义，但会主导距离；另一些代谢物单独变化不大，却作为协同模块有明确生物含义。

所以这里要明确拆开：

- **Fold-change**：合理的刺激层预处理。
- **Full-space Euclidean distance**：不合理的生物几何。

好的 embedding 应保留 fold-change 的实验对比含义，但替换掉 full-dimensional Euclidean 这个距离结构。

## 文献给出的启发

### Exometabolomics 把问题定义为培养基转化

微生物 exometabolomics，也叫 metabolic footprinting，通常比较 spent medium 和未接种/空白培养基。核心观测是：某个代谢物在微生物生长后是增加、减少，还是不变。Web of Microbes 就是围绕这个逻辑建立的：微生物通过相对 control environment 的生产和消耗模式，与环境代谢物相连。

这支持本项目目前对 fold-change 的解释。当前矩阵不是一个随意变换过的丰度表，而是一个细菌 effect-versus-medium 表。

参考来源：

- [Web of Microbes: curated microbial exometabolomics database](https://pmc.ncbi.nlm.nih.gov/articles/PMC6134592/)
- [Extracellular Microbial Metabolomics: The State of the Art](https://pmc.ncbi.nlm.nih.gov/articles/PMC5618328/)

### 通路方法有帮助，但声明必须降级

MetaboAnalyst、MSEA、MetPA、mummichog 和 PAPi 都是在尝试把单个代谢物列表转成更有生物意义的功能结构。它们共同的思想是：代谢物变化如果能按已知 metabolite set、pathway 或 network context 来组织，就比单个代谢物更容易解释。

MetaboAnalyst 的 pathway analysis 结合了 enrichment 和 topology；mummichog 利用代谢网络结构，从高通量代谢组特征中推断可能的 network activity；PAPi 对本项目尤其相关，因为它从代谢物丰度构建 pathway activity profile，并且明确讨论了 extracellular metabolic footprinting，即先用 uncultured medium 对 spent medium 做归一化。

但是这些方法的默认语言对本项目来说往往过强。Pathway activity score 可以作为有用的摘要，但没有基因组数据时，更适合称为 **pathway footprint score**，而不是 pathway activity 或 pathway capacity。也就是说，一个 pathway-like 信号只能说明这条通路相关的已测代谢物在培养基中共同改变；它不能证明细菌编码、使用或调控了完整通路。

参考来源：

- [MetaboAnalyst 6.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223798/)
- [MetaboAnalyst 2.0 / MSEA / MetPA overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC3394314/)
- [PAPi: Pathway Activity Profiling](https://academic.oup.com/bioinformatics/article/26/23/2969/221025)
- [mummichog: Predicting Network Activity from High Throughput Metabolomics](https://pmc.ncbi.nlm.nih.gov/articles/PMC3701697/)

### 化学本体和经典通路同样重要

经典 pathway map 对代谢组的覆盖是不均匀的。很多被检测到的代谢物不能干净地放入 canonical pathway diagram，而一些与感知、生物活性或细菌生态相关的化学家族，可能用 chemical class 描述比用 pathway 名称更自然。

ChemRICH 正是作为经典 biochemical pathway mapping 的替代或补充而提出的。它的前提对本项目很重要：代谢物不是统计独立的，chemical similarity 或 chemical class 有时比 pathway map 更能稳定地组织代谢组。ClassyFire 和 ChemOnt 提供了可计算的化学分类层级，包括 SuperClass、Class、SubClass、DirectParent 等。

这和本项目直接相关，因为当前原始 workbook 已经有化学分类字段。这些类别不只是画图标签；它们可以成为 embedding 的一层结构。

参考来源：

- [ChemRICH](https://pmc.ncbi.nlm.nih.gov/articles/PMC5673929/)
- [ChemRICH project page](https://barupal.github.io/ChemRICH/)
- [ClassyFire / ChemOnt](https://pmc.ncbi.nlm.nih.gov/articles/PMC5096306/)

### 反应网络提供前体-产物结构

Pathway membership 告诉我们两个代谢物是否属于同一条命名通路。Reaction database 提供的是更局部的关系：一个代谢物是否可以通过已知生化反应转化成另一个代谢物，或者两个代谢物是否共享反应邻域。

MetaCyc、KEGG REACTION/RCLASS 和 Rhea 在这里很有用。它们提供的是关于反应、化合物和通路的参考知识。在没有基因组数据的条件下，这一层应该被当作 **通用生化邻接图**（universal biochemical adjacency graph），而不是 organism-specific metabolic model。一个 substrate-product edge 说明两个代谢物在生化上邻近；它不说明当前细菌一定能完成这个转化。

这个反应层的价值在于：它可以避免 embedding 把前体和产物当作完全无关的独立轴。

参考来源：

- [MetaCyc 2019 update](https://pmc.ncbi.nlm.nih.gov/articles/PMC6943030/)
- [Rhea reaction knowledgebase](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728268/)
- [KEGG REACTION / RCLASS](https://www.kegg.jp/kegg/reaction/reaction.html)

### 经验共变提供数据集特异结构

知识库是不完整的。测量数据本身还包含另一类结构：哪些代谢物会在不同细菌样本之间共同上升或共同下降。代谢组网络分析文献强调了几类互补网络：已知生化网络、谱图相似网络、association/correlation network，以及 multi-layer network。

对本项目来说，经验共变有用，但必须谨慎解释。它不应压过生化知识，因为相关性可能来自 taxonomy、batch、共同缺失、QC 或培养基共同效应。但它可以帮助发现本数据集中真实存在、而 pathway annotation 没有完整捕捉到的模块。

参考来源：

- [Networks and Graphs Discovery in Metabolomics](https://pubmed.ncbi.nlm.nih.gov/35350714/)
- [Network-based strategies in metabolomics interpretation](https://pubmed.ncbi.nlm.nih.gov/32380880/)
- [DNEA: data-driven network analysis of metabolomics data](https://pmc.ncbi.nlm.nih.gov/articles/PMC11657348/)

## 合适的概念框架

最适合本项目的框架可以叫：

**taxonomy-aware observed metabolic-footprint embedding**
即：带细菌分类信息约束的、基于观测代谢足迹的化学空间嵌入。

这个名字包含四个边界：

- **taxonomy-aware**：细菌类别标签用于解释和验证，不用于证明遗传通路能力。
- **observed**：embedding 基于实际测得的代谢物，不基于推断的 genome-scale metabolism。
- **metabolic-footprint**：核心信号是相对空白培养基的 spent-medium transformation。
- **embedding**：先把样本投影到低维、可解释的模块空间，再计算距离。

## Embedding 应该表达什么

每个细菌样本应表示为一组模块层面的生产和消耗模式。

在代谢物层，最自然的基本量是 signed `log2FC`：

- 正值：相对空白培养基的积累或分泌
- 负值：相对空白培养基的消耗或减少
- 零值：无明显变化，或 missing-to-one 修正后的中性值

在模块层，不应该把正负变化简单压成一个无方向的幅度。生产和消耗有不同生物意义。例如，氨基酸消耗可能代表营养利用，而有机酸积累可能代表 overflow metabolism。它们不应该仅仅因为属于相关通路就在一个总分里互相抵消。

因此，一个模块至少应有两个概念上分开的摘要：

- **production footprint**：模块内协调的正向 fold-change
- **depletion footprint**：模块内协调的负向 fold-change

真正有用的对象不是：

```text
sample = [metabolite_1, metabolite_2, ..., metabolite_250]
```

而更接近：

```text
sample = [
  amino_acid_depletion,
  amino_acid_accumulation,
  purine_nucleoside_depletion,
  purine_nucleoside_accumulation,
  indole_aromatic_depletion,
  indole_aromatic_accumulation,
  organic_acid_depletion,
  organic_acid_accumulation,
  ...
]
```

这是生物学压缩，不是普通的数学降维。

## 什么可以算作模块

好的模块不应只由单一证据来源定义。多个关系层相互支持时，模块最有说服力。

### 化学类别模块

这些模块来自 ClassyFire/ChemOnt 式层级，或当前 workbook 中已有字段：

- SuperClass
- Class
- SubClass
- DirectParent

这些模块化学上连贯，通常容易解释。它们对感知问题尤其有用，因为 chemical class 往往比宽泛 pathway name 更自然地对应物理化学性质、气味相关性质或潜在生物活性。

弱点：chemical class 不等于 metabolic pathway。它描述的是结构，不是生物转化。

### 通路成员模块

这些模块来自 KEGG、MetaCyc、HMDB、SMPDB 或 MetaboAnalyst 式映射。

当问题接近代谢功能时，它们很有用，例如：

- amino acid metabolism
- purine metabolism
- TCA-related organic acids
- aromatic amino acid derivatives
- bile acid / steroid-related chemistry

弱点：pathway database 覆盖不均匀，通路边界是人工定义且高度重叠的；没有基因组数据时，也不能证明 pathway capacity。

### 反应邻域模块

这些模块来自 KEGG REACTION/RCLASS、Rhea、MetaCyc 等反应数据库。一个模块可以是由已知生化转化连接起来的局部 compound neighborhood。

这一层对前体-产物关系很重要。如果 compound A 和 compound B 是直接或近距离反应邻居，embedding 不应该把它们当作完全无关的高维坐标。

弱点：通用反应邻接不是物种特异的；反应数据库可能包含当前细菌并不相关的路线。

### 经验共变模块

这些模块来自实际测量的 fold-change 矩阵。即使 annotation 不完整，跨样本共同变化的代谢物也可能形成一个模块。

它可用于发现数据集特异的 footprint：

- 区分某类细菌的代谢物集合
- 反复出现的生产/消耗模式
- 在 pathway database 中被拆散、但在数据中共同变化的代谢物

弱点：共变不是因果。它可能反映 taxonomy、date、missingness、QC 或共同培养基效应。它应该作为支持层，而不是唯一的生物定义。

## 最合适的结构：多层模块

embedding 不应该只选择一种 ontology，然后丢掉其他信息。更合理的结构是 multi-layer：

```text
metabolite
  -> chemical class memberships
  -> pathway memberships
  -> reaction neighbors
  -> empirical co-change module
```

当一个模块在多层结构中都连贯时，它更可信。例如：

- 一组 purine nucleosides 在化学上连贯
- 同时映射到 nucleotide/purine 相关通路
- 同时形成反应邻域 cluster
- 同时在细菌样本中共同变化
- 同时随细菌类别或神经响应模式变化

这比一个模块仅仅因为 RSA 高而被选中要强得多。

## 距离应如何重新定义

目标不是在原始化学矩阵中找一个更好的 full-space metric。目标是先把样本投影到一个有生物组织方式的 footprint space，再比较样本。

这样会产生几种不同的距离含义。

### 模式距离

这个距离问的是两个细菌是否有相似的 signed module pattern：

```text
它们是否生产和消耗相同的模块？
```

这更接近模块分数上的 correlation 或 cosine distance，而不是单代谢物层面的 Euclidean distance。

### 活跃模块重叠

这个距离问的是两个细菌是否扰动了相同模块，而不太关心精确幅度：

```text
它们是否触及了代谢空间中的同一批区域？
```

这更接近 weighted Jaccard 或 overlap distance。如果线虫对某种 footprint type 的存在敏感，而不是对精确浓度幅度敏感，这种距离可能更合适。

### 方向性足迹距离

这个距离保留生产和消耗方向：

```text
相同模块是否被消耗？相同模块是否被积累？
```

这一点很重要。一个细菌消耗氨基酸，另一个细菌积累氨基酸衍生物，即使都涉及同一个 broad pathway，在生态意义上也不一定相似。

### 网络扩散距离

这个距离允许变化在反应或化学类别邻域中局部扩散：

```text
这些样本是否改变了生化空间中相邻的区域？
```

这很有吸引力，因为 metabolite annotation 和 measurement 都不完整。如果一个样本积累某个代谢物，另一个样本积累它的直接产物或近邻化学类，embedding 应该识别出部分相似性。

关键是 diffusion 必须局部且可解释。它应该在已知生化邻域内平滑，而不是把一切模糊成一个泛泛的低维云。

## 细菌类别标签的角色

细菌类别标签很有价值，但不应该直接定义 embedding。

它们可以帮助回答：

- 模块足迹在同一细菌类别内是否稳定？
- 哪些模块是 category-specific？
- 哪些模块跨 taxonomy 仍然稳定？
- 神经 geometry 跟踪的是 bacterial taxonomy 本身，还是与 taxonomy 部分相关的 metabolic footprint？

它们不应用来声称：

- 这一类细菌遗传上具有 pathway X
- 这一类细菌因果激活 pathway X
- 一个 taxonomy-enriched module 自动就是神经刺激维度

最有力的解释会是一个区分：

> 神经响应不是单纯按 taxonomy 排列，也不是按 full chemical distance 排列，而是更接近观察到的代谢足迹模块。

这会支持一个更强的科学命题：线虫表征的是细菌代谢对环境造成的化学后果，而不只是细菌分类身份。

## 为什么这比 PCA 更合适

PCA 对可视化有用，但如果作为主 embedding，它解决的是错误问题。PCA 寻找最大方差轴，却不知道哪些方差与感觉相关、与通路连贯、或具有生物解释性。

在本项目中，大方差可能来自：

- 少数极端 fold-change
- missing/non-detect 处理方式
- 宽泛丰度尺度
- 高度冗余的代谢物家族
- 技术或样本 block 结构

这里提出的 embedding 由生化组织方式约束，而不是由神经响应监督，也不是由最大方差驱动。这正是本项目需要的中间路线：

- 不是纯 unsupervised PCA
- 不是 neural-supervised feature mining
- 不是 high-dimensional Euclidean distance
- 不是 genome-based pathway inference

它是一个 knowledge-guided、observed-footprint representation。

## 为什么这比单纯 taxonomy-class model 更好

一个 taxonomy-class model 可以说：

```text
Purine nucleosides 与神经 geometry 有关联。
```

这有用，但不完整。它把化学类别当成最终答案。更深的问题是，这个类别是否属于一个更大的 footprint：

- medium nucleosides 是否被消耗？
- downstream nucleobases 是否积累？
- 是否是 purine salvage 或 breakdown footprint？
- 是否和 amino acid 或 organic acid 变化共同出现？
- 它是 bacterial-category marker，还是 neural-relevant chemical module？

因此，embedding 应允许 taxonomy class 成为模块，但也要把这些模块放入 reaction、pathway 和 empirical co-change 语境中。

## 合适的科学表述

可以这样表述：

> 我们用细菌在培养基中留下的观察到的代谢足迹来表示每个细菌样本。由于缺少基因组数据，我们不推断物种特异的代谢通路能力。相反，我们利用化学分类、通路成员关系、反应网络邻近性和经验共变结构，把测得的 fold-change 代谢物组织成可解释的足迹模块。随后，我们检验线虫神经响应是否比原始高维代谢物距离更好地对齐这种模块层面的代谢足迹空间。

这个表述既强，也诚实。

它说明：

- fold-change 在生物学上合理
- 高维 Euclidean distance 不合理
- pathway knowledge 有用，但不是因果证明
- taxonomy labels 帮助解释，但不能替代化学
- 目标是理解神经系统是否表征观察到的细菌代谢后果

## 主要 caveats

### 代谢物身份质量

所有 pathway、reaction 和 ontology mapping 都依赖 metabolite identity。错误名称或同分异构体歧义会制造假模块。当 identity confidence 较低时，chemical class 往往比精确 reaction mapping 更稳健。

### 缺失值和 non-detect

missing-to-one 策略很重要，因为 raw missing 如果被当作低值 fold-change，可能看起来像极端 depletion。任何 embedding 都应该区分稳健观测到的 fold-change 和 missingness-driven signal。

### 通路覆盖偏差

Primary metabolism 的注释通常比 secondary metabolism 或感觉相关代谢物更好。只用 pathway embedding 可能会抹掉缺少清晰 pathway membership、但生物上重要的化合物。

### 模块重叠

代谢物常属于多个 pathway 或 class。强行把每个代谢物分配到唯一模块太僵硬。应允许重叠，但解释时必须避免把同一个信号重复计为多个独立发现。

### Taxonomy 不是机制

如果某个 footprint module 在某类细菌中富集，这说明它可能是 class-associated metabolic phenotype。它不证明遗传或酶学机制。

## 最终综合

最合理的方向不是在原始化学矩阵中寻找单一更好的距离指标，而是重新定义化学空间。

原始化学矩阵维度太高，且代谢物之间依赖太强，不适合 Euclidean geometry。Fold-change 修正了实验对比，但没有修正几何。Pathway database 提供生物分组，但如果被当作 organism-specific activity 就会过度声明。Chemical ontology 提供稳健结构，但不提供代谢方向。Reaction network 提供局部生化邻接，但不提供物种能力。Empirical co-change 捕捉数据集特异结构，但不是因果。

因此，合适的 embedding 是一个组合结构：

```text
signed fold-change
  -> chemical/pathway/reaction/co-change modules
  -> production and depletion footprint scores
  -> module-level sample embedding
  -> interpretable chemical-neural comparison
```

这给项目一个更强的科学对象：

> 线虫神经活动可能表征的是细菌代谢足迹的低维投影，而不是细菌 taxonomy 本身，也不是完整 metabolome。

这条路径最符合当前数据、缺少基因组信息的边界，以及项目真正想解决的科学问题。
