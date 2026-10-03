# Poster可用图与文字

推荐主图：[PDF](figures/poster_neighborhood_atlas.pdf) / [SVG](figures/poster_neighborhood_atlas.svg) / [PNG预览](figures/poster_neighborhood_atlas.png)。同一图中显示化学近邻、细胞贡献和群体分离，不再挑A022/A023作全体代表。SVG文字可编辑，PDF为矢量图。

推荐标题：

**Population calcium responses reveal diversity within chemical neighborhoods**

中文：**群体钙响应刻画化学邻域内的表型多样性**

推荐中心句：

**Chemically neighboring strains span a range of population calcium-response differences, with the contributing neuron classes varying across strain pairs.**

中文：**化学近邻菌株之间呈现不同程度的群体钙响应差异，差异所处的神经元组合也随菌株对改变。**

## 主图短图注

Chemical neighbors were selected from matched, same-genus candidate sets containing at least three strains, using all 380 log₂ fold-change features (28 pairs, 40 strains, 38 animals). Rows are ordered by neural separation. Left: chemical profile difference; center: neuron contributions; right: mean cross-animal inner product of paired calcium-response differences, excluding animal self-products. Neurons use fixed catalogue-derived scales. Contributions sum to the population score: 26 pairs have 13 eligible classes and two have 10; crosses mark fewer than three paired animals. Lines show delete-one-animal ranges, not confidence intervals. Negative estimates are retained; small estimates do not establish equivalent responses. Colors distinguish same-species and different-species pairs. Estimates describe the recorded protocol; chemistry was measured in separate culture batches.

## 如有第二张图的空间

[147对背景图 PDF](figures/poster_chemical_neural_landscape.pdf) / [SVG](figures/poster_chemical_neural_landscape.svg)：用灰色全体比较衬托28对化学近邻，表达近邻内部的连续差异，不做全局回归或“神经比化学好”的排名。

短图注：

All 147 matched within-genus strain comparisons are shown, with 28 chemistry-selected neighbor pairs highlighted. Chemical difference is RMS Δlog₂FC over 380 features. Neural separation uses paired animal responses across 13 neuron classes where coverage permits. Pairs share strains and animals and are not independent replicates. Lines are delete-one-animal sensitivity ranges. Same-species and different-species neighbors overlap; no near/far threshold or equivalence claim is imposed.

## 可选补充，不建议塞进主图

[种内/种间倾向 PDF](figures/poster_species_tendency.pdf)使用7个同时包含两类比较的匹配集合。5/7组原始分数均值差为负，等组平均−0.469。每条线只连接该组点值与零，不是误差线。标题已限定为倾向；不能写“species解释了化学近邻的全部例外”。

[原始钙单位附图](figures/poster_neighborhood_atlas_raw.pdf)用于回应尺度问题。具体最大细胞名单不稳健，主图不能去掉cell-scaled标注，也不能将8类曾贡献最大的细胞写成8条独立生物通路。

取舍建议：主图支持“群体表型补充化学邻域信息”；species图支持有限规律。共同报告化学条目的关联尚不适合进入poster中心结论。19/28删动物后正值也不应写成19个显著发现。
