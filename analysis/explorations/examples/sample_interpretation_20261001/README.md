# 从具体样本解释响应：poster panels

本轮按用户授权执行有限样本分析和成图。主组为 A021/A022/A023，支持组为 A007/A010。复用已核验的动物级曲线、分窗值、整动物留出模板和化学对齐表；没有重跑完整模型、完整数据探索或 Notebook，没有修改原始数据或旧产物。

## 可直接用于 poster 的图

- [两页矢量 PDF](figures/poster_panels.pdf)：第 1 页为三株主图，第 2 页为时间平均的局部边界例子。
- 第 1 页：[PNG](figures/strain_response_poster.png)、[可编辑文字 SVG](figures/strain_response_poster.svg)。
- 第 2 页：[PNG](figures/timing_cancellation_poster.png)、[可编辑文字 SVG](figures/timing_cancellation_poster.svg)。
- [英文外置图注](captions.txt)。完整数值及适用边界见下文，不在图内堆叠说明。
- [五页支持材料](figures/supporting_evidence.pdf)：全 13 类三株曲线、全细胞配对差值、模板分解、全 13 类支持组曲线、化学口径与候选展示。

主图的核心信息是：同种菌株在不同细胞中的相对响应排序不同。化学距离提供组成背景；当前图没有建立化学到神经的映射。第二页的核心信息是：原本整体差向量不稳定的 A007/A010，在 AWB 的特定时间窗仍有一致的相反方向差异，时间平均会掩盖它。

## 实际观察及边界

三株均标注 B. stercoris、20260601、reference A050，共 6 只动物；AWA、AWB、ADF、ASER、AWCON 仅 5 只。逐细胞三株的动物集合完全一致，比较没有通过改变动物支持制造排序。刺激期为 0–10 s；主比较为既有 0–25 s，分为 0–10 和 10–25 s。25–40 s 保留为预先确定的晚期支持，不按效果更换窗口。

### 三株主图

10–25 s 原单位配对差值如下；计数是方向描述，不是显著性检验。

| Cell | 对比 | 均值差 ΔF/F₀ | 与均值同方向的动物 |
| --- | --- | ---: | ---: |
| AWA | A022 − A021 | −0.2306 | 5/5 |
| AWA | A023 − A021 | −0.3301 | 5/5 |
| AWA | A023 − A022 | −0.0995 | 4/5 |
| ASH | A022 − A021 | +0.4024 | 6/6 |
| ASH | A023 − A021 | +0.2479 | 6/6 |
| ASH | A023 − A022 | −0.1545 | 6/6 |
| AWCON | A023 − A022 | +0.1369 | 5/5 |

加入 A021 后，AWA 与 ASH 的相对菌株排序不同。ASK 也保留在支持资料中：A022 高于 A021/A023，而后两株的均值接近；这不证明后两株响应等价。

AWCON 的时间过程值得保留：A022 的 0–10 和 10–25 s 均为 5/5 动物负值，A023 的 10–25 s 为 5/5 正值。A023−A021 的均值差从 0–10 s 的 −0.0967 变为 10–25 s 的 +0.1234，两阶段分别有 4/5 同号；**同一动物同时满足早负晚正的是 3/5**，不能称所有动物都出现排序反转。

AWA 的 A021 在 10–25 s 均值为正，但实际仅 2/5 动物为正；AWCON 的 A021 早期正均值也不稳。因此主图保留每只动物的点和配对线，结论落在相对差异，不能把均值正负升级为一致激活/抑制。细胞之间原始 GCaMP 幅度的数值大小也不是放电强度或功能重要性排名。

### A007/A010 的局部时程反例

两株来自 20260520，同一 reference，动物组与主组不同。全 13 类细胞覆盖为 3–7 只。AWB 有 5 只共同动物：

| Window | A010 − A007 均值差 ΔF/F₀ | 动物方向 |
| --- | ---: | --- |
| 10–15 s | +0.1518 | 5/5 正 |
| 20–25 s | −0.2275 | 5/5 负 |
| 0–25 s | −0.0107 | 3 正、2 负 |

均值接近零不等于无响应或响应等价；该例也没有推翻此前全群体差向量不稳定的观察。这是观察同一个 pair 的更具体时间过程，不是独立验证。全五个预定时间窗均在图中展示，例子由结果事后选择。核钙信号不用于推断放电潜伏期或精确神经动力学。

### 化学组成背景

三株共同报告集合为 322 项。这个固定集合与旧 A022/A023 单独共同报告的 332 项不同。

| Pair | 全 380 项 RMS | 固定共同 322 项 RMS | 固定 QC 子集 286 项 RMS | 单侧缺失对全谱平方距离的贡献 |
| --- | ---: | ---: | ---: | ---: |
| A021 / A022 | 2.904 | 1.315 | 1.237 | 82.3% |
| A021 / A023 | 2.881 | 1.427 | 1.374 | 79.2% |
| A022 / A023 | 1.794 | 0.644 | 0.599 | 88.9% |

主图只显示固定 322 项距离；其余口径在支持图中。A022/A023 在这些口径下相对更近，但不能称化学相同。322 项中报告 FC 差在两倍以内的比例分别为 79.2%、75.8%、93.5%；这描述继承上游填零/+1处理后的报告量，不是刺激浓度的倍数。

支持图给出六个按化学跨度与 QC 选择的条目，完整排行保留在表中，没有按神经结果挑分子。MSI/鉴定置信等级未提供，化学来自独立培养批次；没有用三株筛选神经相关分子，也不做受体、分子因果或神经优于化学的主张。

## 方法与可复核数据

- 原曲线先在动物内平均 trial，使用原 ΔF/F₀，不额外中心化、单位长度归一化或平滑。主图曲线为原 1 s 采样点均值；带为逐时间点 SEM，非置信区间。配对图线只连接同一动物，黑色短线是均值。
- 差值统一为 `sample_second - sample_first`，由同日期、同 animal_id 的响应配对。主组三株及支持组分别计算，不跨采集块合并推断。缺失记录保留，作图仅排除缺测值，不填零。
- 所有 13 类细胞均计算和保存；主图 AWA/ASH/AWCON 是事后说明性选择，分别展示相对排序、幅度和阶段信息。它们不构成预先定义的细胞组合。全 13 类动物曲线和差值放支持材料。
- 留出模板分解：对某动物的配对差曲线 d，读取排除此动物、同窗口同细胞的既有模板 t，计算 α=(d·t)/(t·t)，r=d−αt。用同一个 t 分解其他配对动物的均值差。平均时间维的两部分交叉内积和等于原配对差的交叉内积；负值保留。
- α 使用被检验动物的实际曲线，属于描述性投影，不是预测。模板外分量含偏移和其他形状差异，不能自动当作神经时程机制。留出折训练高度重叠，不作为独立重复或方差解释率；不计算确认性 p 值、FDR 或生物学方差比例。
- 删一动物均值范围是影响检查，不是置信区间。样本、细胞、窗口、共享端点的 pair 和 trial 都不是额外独立动物。
- 神经反应与刺激顺序的影响尚未实验分离；日期仅为采集块代理。这里描述的是已有协议与样本条件下的响应，不能推广为新培养批次的纯菌株效应。

主要表：

| 表 | 内容 |
| --- | --- |
| `neural_curves.csv` / `neural_observations.csv` | 原动物曲线和 5 s 分窗值，含缺失 |
| `neural_animal_stages.csv` / `neural_paired_stages.csv` | 动物级阶段响应与配对差 |
| `neural_pair_stage_summary.csv` / `neural_pair_window_summary.csv` | 均值、方向、覆盖、删动物范围 |
| `neural_group_coverage.csv` | 三株/两株共同覆盖 |
| `neural_oof_projection_*.csv` | 留出模板、动物级分解及一致性描述 |
| `chemical_pair_summary.csv` / `chemical_missing_contributions.csv` | 共同、全谱、QC口径与缺失贡献 |
| `chemical_primary_profiles_all380.csv` | 三株全部已对齐化学值和报告状态 |
| `chemical_primary_common_ranked.csv` / `chemical_candidate_display.csv` | 固定共同集合完整排行和六项展示 |
| `poster_*.csv` | 两张主图使用的精确数值 |

## 复现与核验

在仓库根目录按需运行；也可在现有 Notebook 中用 `%run` 调用相同文件：

```bash
.pixi/envs/default/bin/python reports/sample_interpretation_20261001/code/neural_analysis.py
.pixi/envs/default/bin/python reports/sample_interpretation_20261001/code/chemical_context.py
.pixi/envs/default/bin/python reports/sample_interpretation_20261001/code/poster_figures.py
.pixi/envs/default/bin/python reports/sample_interpretation_20261001/code/verify.py
```

仅改图时只需执行绘图脚本及核验脚本。重复运行覆盖本目录同名产物。未新增依赖、Notebook、CLI框架、提交或PR。

神经输入重聚合、配对、投影正交及分量加和检查在 `logs/neural_verification.json`；化学口径与输入指纹在 `logs/chemical_verification.json`。两张主图及五页支持 PDF 经 Poppler 渲染检查。最终主图数据、PDF页数和旧文件保护记录见 `logs/verification.json`；330个原文件的基线指纹位于 `logs/source_baseline.json`。
