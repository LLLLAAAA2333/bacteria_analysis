# 固定 ADF−ASH 对比与所选化学状态：诊断结果

这次局部分析找到一个有方向、具有留出预测增益的候选关系，但还不能称为已经建立“明显、连续、化学解释简单”的模型。主要限制是低、中化学组响应重叠，高分组才明显抬升；所选状态含多类注释，且日期、菌种和动物重复的限制仍存在。本目录只诊断 `model/` 已保存的选择和预测，没有重新选化学轴或神经目标。

全部 29 株、16 个记录菌种、8 种原记录日期集合和 6 个 taxonomy flag 均保留。此前无法进入完整动物分半检查的 8 株也没有删除。响应是完整 13 维系数单位化后的 ADF−ASH，表示相对配置；raw 指未作 L2 归一化的模板系数，不是原始钙曲线。

## 关系及留出结果

全数据选中的状态为本次局部编号 L03：14 个化学注释、13 个 Mass-column family。其 score 是成员 log2 浓度标准化后，先在 family 内等权、再在 family 间等权的平均；不是某一个化合物剂量，也不是总浓度。

全 29 株 Pearson r=0.530、Spearman ρ=0.607；表观 R²=0.281。化学 score IQR 为 0.944（−0.380 至 0.564），对应拟合 unit ADF−ASH 改变 +0.201。全数据斜率为 +0.213；散点仍有明显偏离拟合线。

| 检查 | 主模型 RMSE | 训练均值基线 RMSE | 相对基线 SSE 改善 |
|---|---:|---:|---:|
| 留一菌株，29 个预测 | 0.293 | 0.334 | +23.1% |
| 留一记录菌种，29 个预测/16 折 | 0.302 | 0.337 | +20.0% |

这些数字来自模型端：每折均在训练株内重算化学标准化、聚类、成员、权重、轴选择和 OLS。它们不同于本目录的“固定全数据轴删除诊断”。辅助 raw 对比的留株/留种改善为 +23.3%/+21.4%，同模板 pre-gate unit 为 +23.6%/+20.5%。不能据此抹去本批数据中先前选择 ADF−ASH 的过程；神经模板也来自原完整数据，因此这些是条件于现有表示的探索性预测检查。

![Full fit and held-out predictions](figures/01_chemical_relationship_and_heldout.png)

## 低、中、高分的实际支持

在已经按全样本主响应择优选出的化学轴上，按 score、再按 strain ID 排序，固定分为 10/10/9 株。切点本身只使用 score 排名，但轴选择已经使用响应，所以这些 thirds 仍是同数据的条件性描述，不能当作独立验证。每一株见 `tables/ordered_strains_and_thirds.csv`。

| 化学 rank third | n | ADF−ASH 均值 | 中位数 | 最小–最大 | 菌种数 | 日期集合数 |
|---|---:|---:|---:|---:|---:|---:|
| LOW | 10 | −0.147 | −0.141 | −0.646–0.331 | 8 | 5 |
| MID | 10 | −0.089 | −0.110 | −0.372–0.101 | 8 | 5 |
| HIGH | 9 | +0.372 | +0.407 | −0.041–0.680 | 8 | 6 |

低、中两组的中心和个体范围明显重叠，高组总体更高，但同样不是每一株都高。高组跨 8 个菌种、6 种日期集合，不能直接说成某一个菌种或单一日期的分类；这些覆盖也不足以排除菌种/日期混杂。全数据支持正的秩关系，却不证明沿化学轴均匀、单调的生物剂量响应。没有为改善展示而去掉 A023、B. stercoris 或 taxonomy flag 株。

## ADF 和 ASH 是否同时改变

在相同固定化学轴上，两个 unit 坐标的群体拟合方向相反，ADF 上升、ASH 下降，ASH 的相关幅度更强。它们是跨菌株平均关系，并非每株内部的干预变化。

| 坐标 | Pearson r | Spearman ρ | 每化学 IQR 拟合变化 |
|---|---:|---:|---:|
| unit ADF | +0.401 | +0.534 | +0.087 |
| unit ASH | −0.573 | −0.621 | −0.114 |
| 未单位化 ADF 模板系数 | +0.377 | +0.520 | +0.085 |
| 未单位化 ASH 模板系数 | −0.369 | −0.318 | −0.069 |

未单位化的两坐标也有相反斜率，因此这个方向并非仅由 L2 分母引入。不过后两行只是同轴的描述性分量检查，没有另选目标、没有单独验证为新发现；系数符号也不是兴奋/抑制结论。未单位化数值与 unit 数值单位不同，不能直接比较变化大小。

所有 13 个坐标在相同化学 IQR 上的拟合变化均保存，主变化确实集中于 ADF/ASH。对完整 13 维向量的留株/留种 SSE 改善仅为 +4.2%/+3.5%，说明这条化学轴对应的是局部对比，而非充分解释整个神经组合。

![ADF and ASH components, individual thirds, all-neuron effects](figures/02_adf_ash_thirds_and_all13_effects.png)

## 端点、菌种和日期的影响

固定全数据 x 后，29 次单株删除的 Pearson 为 0.466–0.626、斜率为 0.184–0.313，方向均为正。16 次菌种删除的 Pearson 为 0.446–0.773、斜率为 0.168–0.450，表明方向持续存在，但定量斜率对菌种构成有影响。

A023 位于化学低端，占 x 平方离差的 34.4%；删去它时 r 从 0.530 升至 0.626、斜率从 0.213 升至 0.313。删去高端 A049 时 r=0.466，仍为正。删去整个 B. stercoris（A021/A022/A023）时斜率为 0.450。因而不能说正关系由一个端点单独制造，也不能忽略低端这几株对梯度大小的影响。所有删除只作敏感性描述，主结果仍含全部 29 株。

| 去组均值的描述 | 保留组数 | 保留菌株数 | Σ(n组−1) | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| 同记录菌种，组内至少 2 株 | 10 | 23 | 13 | 0.637 | 0.626 |
| 相同原始记录日期集合，组内至少 2 株 | 4 | 25 | 21 | 0.352 | 0.453 |

singleton 组在此描述中没有组内差异，分别有 6 个菌种组、4 个日期集合组被略去，主分析并未删去它们。去均值仅减掉该标签组的 x/y 平均值；菌种、日期、培养和动物结构没有被完整分离。因此第二行不能称为批次校正，也不能把 23/25 个残差当作新增加的独立重复证据。

## 这个化学状态包含什么

它由不同注释共同升降构成，不能直接改名为单一通路。依据已保存的 Class 注释：

| 注释类别 | 注释数 | Mass-column family 数 | 总 score 权重 |
|---|---:|---:|---:|
| Carboxylic acids and derivatives | 4 | 4 | 30.8% |
| Indoles and derivatives | 2 | 1 | 7.7% |
| Organic sulfuric acids and derivatives | 1 | 1 | 7.7% |
| Organonitrogen compounds | 1 | 1 | 7.7% |
| Phenols | 2 | 2 | 15.4% |
| Purine nucleotides | 2 | 2 | 15.4% |
| Pyrimidine nucleotides | 2 | 2 | 15.4% |

成员包括 Indole-3-carboxylic acid、Indole-3-carboxaldehyde、Indoxyl sulfate、3-Methoxytyramine、p-Cresol、四种 N-acetyl/formyl 氨基酸衍生物、2-Hydroxyphenethylamine 和四种核苷酸注释。两项 indole 注释共享一个 Mass-column family，因此各占该 family 一半权重。完整名称、RT、质量通道、注释和权重见 `tables/selected_state_members.csv`。目前最直接的化学意义是“这些异质成分的共变状态较高时，ADF 相对 ASH 的模板权重通常较高”；尚不能归因于其中某种成分或简单机制。

![All strains, 13 neurons, species and dates](figures/03_chemical_ordered_all13_context.png)

## 文件、复算与检查

`summary.json` 保存主要描述数值。`tables/fixed_axis_associations.csv` 保存主/辅助对比和全 13 坐标，`unnormalized_adf_ash_descriptions.csv` 保存未单位化分量。`rank_third_summary.csv` 与 `rank_third_species_date_composition.csv` 保留分组支持；`fixed_axis_deletion_influence.csv` 保留全部删除；`within_group_*` 表保留去均值前后个体、组分母和结果。`heldout_predictions.csv` 与 `pooled_performance.csv` 是模型端输出的完整复制，没有重新拟合。图注见 `figures/CAPTIONS.md`。

已有 Notebook 可调用只读入口：

```python
from pathlib import Path
import importlib.util
repo = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
result = repo / 'reports/exploration_bacteroides_adf_ash_chemical_20261003/diagnostics'
spec = importlib.util.spec_from_file_location('fixed_axis_diagnostics', result / 'code/fixed_axis_diagnostics.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
saved = diagnostics.load_saved(result)
display(saved['ordered_strains'])
```

明确重算时调用 `diagnostics.run_diagnostics(repo, out=new_empty_result_directory)`；模型输出仍是唯一输入。绘图由 `code/plot_fixed_axis.py::plot_saved(result_dir, out=new_empty_figure_directory)` 仅读取结果表。两者拒绝覆盖已有科学结果。没有新建或运行 Notebook。

输入路径和 SHA-256 见 `source_manifest.json`。从同一固定 x 直接重构模型端的主目标拟合值，最大差为 1.13×10⁻¹⁶；29 株顺序、raw/unit 恒等式、13 坐标和 metadata 对齐已检查。进一步独立复核保存在上一级 `verification/`；本地数值/保护检查与实际图像检查保存在 `diagnostics/verification/`。
