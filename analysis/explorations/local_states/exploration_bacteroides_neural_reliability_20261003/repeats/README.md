# Bacteroides：固定神经表示下的重复记录检查

本轮只评估神经记录的二维位置及两个预先固定指标的重复性，不建立化学模型。分析保留原始 29 株的覆盖记录、菌种、日期和 taxonomy flag；没有按响应结果选择 split 或删株。所有输入哈希见 `source_manifest.json`，分析约定见 `PROTOCOL.md`。

## 可以据此判断什么

在有共同完整观测的动物分半中，ADF−ASH 的重复差异通常小于不同菌株差异；AWB 较弱。固定二维平面中的位置有可重现成分，但重复散布并不小。若各半样本重新估计二维平面，其第二个主角度很大，说明半样本下平面的两个方向并未都稳定恢复。这与完整数据删除一株/一个种的稳定性是不同问题，应分别报告。

仅有 6 株的跨日结果更不均一：ADF−ASH 的跨日差异小于这 6 株平均位置间的差异，AWB 的跨日差异则明显更大。完整 13 维和固定二维的跨日变化，均接近或超过这 6 株均值之间的差异。因此不能把完整神经位置称为已经得到跨日重现，也不能将结果概括成一个统一的“可靠/不可靠”标签。

## 覆盖和独立单位

- 输入为 29 株、5 个记录日期、35 个菌株×日期条件，以及 30 个不同 `date|worm_key` 动物 ID。219 个菌株×日期×动物暴露记录不能视为 219 只独立动物；同一动物可能有多个菌株/神经元记录。缓存中的 trial 已在动物内平均。
- 重用原来保存的全部 100 个按日期平衡、整只动物分组的 A/B assignment。100 次分半互相重叠，并不是 100 次独立实验。下文分位数只是分半敏感性范围，不是置信区间。
- 每个日期×神经元要求两半分别至少 2 只动物。在此支持条件下，两半使用完全相同的可用日期，按每个神经元等日期权重平均。缺失或 n<2 保留 NaN；有足够观测但 SNR 不通过才置零。
- 同时有完整 13 个共同观测坐标且两半非零的菌株数，每次 split 为 13–21 株，中位数 20。合计 21 株至少参与过一次。A001、A002、A005、A006、A007、A008、A010、A026 从未有完整分半支持；这表示无法评价其完整向量重复性，不能解释成这些菌株更不可靠。
- 每次 split 在所有原记录日期均有完整支持的数量为 9–15，中位数 15。主分析使用共同可用日期交集，是条件性重复检查；它与原全日期 29 株表示的估计对象不同，不能直接作为其噪声或预测上限。

完整覆盖表为 `tables/animal_strain_coverage.csv`（29×100）及 `animal_per_strain_coverage_summary.csv`。共同条件的两半 gate 不一致比例中位数为 27.9%（5th–95th：25.0%–31.3%）；这也是保留同模板 pre-gate 对照的原因。

## 动物分半：重复散布与菌株差异并排

模板冻结为原完整数据的 13×8 模板，两半分别重估 SNR。固定二维位置用原 29 株的均值和 PC1/PC2 loading 投影。没有对半样本单独 z-score 或归一化 PC score。单位向量为完整 13 维系数除以自身 L2 范数，保留各坐标的符号；这些系数不是兴奋/抑制分类，也不是占比。

同一 split 的同一完整菌株集合上：

`same_RMS = sqrt(mean_i ||A_i − B_i||²)`

`between_RMS = sqrt(mean_(i≠j) ||A_i − B_j||²)`，包含全部有序异株对。

下表为 SNR-gated 100 次分半的中位数；最后一列是逐 split 比值的分位数，因此“比值中位数”不必等于前两列中位数的商。不同指标的 RMS 尺度不可直接互比。

| 表示/指标 | 同株 RMS | 异株 RMS | 比值中位数 | 比值 5th–95th |
|---|---:|---:|---:|---:|
| ADF−ASH，unit | 0.218 | 0.534 | 0.406 | 0.265–0.641 |
| AWB，unit | 0.186 | 0.289 | 0.688 | 0.385–0.950 |
| 固定 PC1/PC2 位置 | 0.244 | 0.497 | 0.506 | 0.376–0.683 |
| 完整 13 维 unit | 0.420 | 0.689 | 0.609 | 0.520–0.687 |

ADF−ASH 的两半 Pearson 中位数为 0.862，AWB 为 0.601，但相关性只是辅助量；上表的幅度对照更直接反映重复散布。同模板 pre-gate 的四个比值依次为 0.386、0.616、0.437、0.632。该对照保持相同菌株支持和模板，仅取消 SNR 置零，没有重新估计未筛选模板。

作为固定诊断，还保存原始完整记录日期集合标签相同的异株参考，每次有 78–108 个有序对。这只是 **same-original-date-label reference**：逐神经元的实际共享日期交集仍可能不同，不能称为日期/批次校正后的信号。

![Animal splits: repeat scatter and between-strain differences](figures/01_animal_repeat_vs_between.png)

两半在同一完整菌株集合上分别中心化、重新估计 PCA 后，二维平面的较小/较大主角度中位数为 19.8°/50.8°，较大角度的 5th–95th 为 24.4°–86.3°。Pre-gate 对应为 20.8°/65.1°。这检验半样本平面方向的恢复，不能与固定平面中的位置差混为同一指标。

![Half-plane orientations and complete-profile coverage](figures/03_half_planes_and_coverage.png)

## 六株跨日期：保留每一对

以下日期均为 2026 年，顺序为早→晚；`source_note_check` 只按现存标签保留，不解释为已确认的分类错误。

| 菌株 | 菌种 | 日期 | taxonomy flag |
|---|---|---|---|
| A011 | Bacteroides caccae | 04-29 → 05-20 | 无 |
| A013 | Bacteroides uniformis | 04-14 → 06-01 | 无 |
| A014 | Bacteroides uniformis | 04-14 → 06-01 | 无 |
| A024 | Bacteroides nordii | 04-14 → 06-01 | 无 |
| A025 | Bacteroides ovatus | 03-31 → 04-14 | source_note_check |
| A044 | Bacteroides coprophilus | 04-14 → 04-29 | source_note_check |

所有日期条件的 13 个系数完整，单位化前后数值均保存。跨日同株 RMS 是六个晚−早差的 RMS；异株参考是六株各自两日平均表示之间的 15 个唯一配对差的 RMS。两者测量精度不同，小样本也不是随机重复子集，因此下列比值仅为描述，不是可靠性系数。记录日期不同不能证明独立培养重复，跨日差还包括动物和日期等多种来源。

| SNR-gated 指标 | 跨日同株 RMS | 六株均值间 RMS | 描述比值 |
|---|---:|---:|---:|
| ADF−ASH，unit | 0.352 | 0.524 | 0.672 |
| AWB，unit | 0.398 | 0.146 | 2.734 |
| 固定 PC1/PC2 位置 | 0.499 | 0.455 | 1.098 |
| 完整 13 维 unit | 0.630 | 0.555 | 1.136 |

Pre-gate 比值依次为 0.667、2.760、1.094、1.087，未改变这一描述性结论。

![Six strains recorded on two dates](figures/02_cross_date_pairs.png)

## 未单位化系数辅助检查

所有动物辅助指标使用与主指标相同的 joint-complete13 菌株集合，不另外筛选神经元或菌株。这里的“原始系数”指未作 L2 归一化的模板系数，并非原始钙曲线；其单位及范数属于 ΔF/F0 系数空间。不要把这些 RMS 与 unit RMS 的绝对大小直接比较。

| SNR-gated 辅助指标 | 动物分半比值中位数 | 跨日同株 RMS | 六株均值间 RMS | 跨日描述比值 | 跨日 Pearson |
|---|---:|---:|---:|---:|---:|
| 原始 ADF−ASH | 0.425 | 0.173 | 0.425 | 0.407 | 0.978 |
| 原始 AWB | 0.737 | 0.216 | 0.085 | 2.529 | −0.684 |
| 完整系数 L2 范数 | 0.534 | 0.150 | 0.263 | 0.571 | 0.773 |

AWB 的跨日不一致在未单位化系数中仍存在，不能全部归于单位化。原始 ADF−ASH 的跨日相关很高，但 A011、A014、A044 仍从正值跨到负值；A024/A025 两株较高而其余四株较低，也会驱动相关。它支持把 ADF−ASH 保留为有依据的后续候选，尚不证明绝对响应稳定或已验证连续梯度。逐株原始变化见 `date_auxiliary_pair_changes.csv`，不能只引用相关系数。

## 文件和复算

`summary.json` 是整合入口；`tables/animal_split_metrics.csv` 保留每次 split、representation、指标、共同菌株数与配对分母。`animal_split_condition_support.csv` 保留逐日期×神经元的两半动物数、SNR、gate 状态和共享支持；`animal_half_profiles.csv` 保留每半所有系数、unit 向量、范数、投影和固定指标。`animal_half_plane_angles.csv`、`date_profiles.csv`、`date_pair_changes.csv` 与 `date_summary.csv` 分别保存平面和跨日结果。四个 `*auxiliary*` 表保存未单位化指标。图注见 `figures/CAPTIONS.md`。

现有 Notebook 可只读调用，不必新建或运行整个 Notebook：

```python
from pathlib import Path
import importlib.util

repo = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
result = repo / 'reports/exploration_bacteroides_neural_reliability_20261003/repeats'
spec = importlib.util.spec_from_file_location('repeat_reliability', result / 'code/repeat_reliability.py')
repeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeat)
saved = repeat.load_saved_results(result)  # read only
display(saved['metrics'])
```

精确重算需在明确授权后指定一个新的空输出目录：

```python
fresh = repo / 'reports/exploration_bacteroides_neural_reliability_recompute/repeats'
repeat.run_analysis(repo, out=fresh)
```

绘图入口为 `code/plot_repeats.py::plot_saved(result_dir, out=new_empty_figure_dir)`，仅读取已保存表。已有结果/图目录会拒绝覆盖。没有新建 Notebook，没有读取化学值，没有编辑原始数据或旧结果。

验证包含：冻结模板完整条件重建最大绝对误差 1.25×10⁻¹⁶；独立代码从原动物缓存重建 split 0 和 99 的支持、SNR、gate、日期汇总、归一化、投影、配对分母与 RMS，并用协方差特征分解复核平面角度，最大误差 2.42×10⁻¹³（角度）；来源哈希及 metadata 对齐均通过。`verification/independent_numerical_checks.json` 保留结果。三张图均已实际打开检查布局，见 `verification/visual_review.md`。
