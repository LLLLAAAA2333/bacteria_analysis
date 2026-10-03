# Poster 与神经响应探索交接

更新至 2026-10-02。本文件保留原名，汇总本 session 的讨论、分析和改图状态。仓库根目录为 `/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis`。

当前已完成 **individual SNR ≥ 0.5 的模板幅度表示、全 106 样本比较，以及 3D HMDS 三视图**。最新绘图请求已交付，本次只整理交接。第6张图的科学问题仍未确定，原 notebook 和旧 Figure 4/5 尚未迁移到新表示。接手后根据用户的新请求继续，不自行启动下一轮分析。

## 用户希望课题回答什么

用户把课题定位为 response atlas / sensor 技术研究，不以解释肠道细菌与线虫的生理机制为主要目标。收束需要展示神经响应测量能提供什么有用的信息，以及这种技术为什么值得做。化学与神经的整体对应较弱，不能靠强调两者相关，或直接声称优于 LC–MS，来完成这一论证。

曾将图5扩展成化学近邻对 × 细胞幅度差异热图，统一了 0–40 s 模板幅度。用户明确认为它不能用于 poster：六对菌及日期标签难以理解、看不到主要规律、独立样本不足，而且只是图5的扩展。**不要把这张检查图重新当成已确定的 Figure 6。** 候选的化学接近程度不同，少量近邻也不能证明化学等价或 LC–MS 无法区分。

讨论过不同细胞是否提供互补的样本区分信息。用户认为可以探索，但 profile 已显示部分结构，补一个系统总结仍不足以收束项目。细胞响应是否对应可解释的化学成分组、菌株功能，以及图5少数化学差异是否对应神经差异，均未获得可靠结果；此前广泛拟合不理想，不能写成机制或功能结论。

用户确定的组织顺序是：响应结构与整体 profile，接着重复性，再做神经与化学 RDM 的整体比较，之后的具体分析方向待定。原 Figure 4 的模型简化内容应提前，与 Figure 1 的整体 profile 并列。先在 exploration 中实现，暂不改原 Jupyter Notebook；当前文件序号是探索输出编号，不代表最终 poster 编号。

## 从这里接手

| 入口 | 用途 |
| --- | --- |
| [AGENTS.md](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/AGENTS.md) | 研究者控制、执行范围和科学作图约定 |
| [当前 REPORT](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/REPORT.md) | 最新参数、结果、图稿及解释边界 |
| [当前 README](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/README.md) | 代码、缓存依赖、仅重画与全量重跑的区别 |
| [表示参数](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/representation_parameters.json) | 筛选、聚合、模板定义及来源哈希 |
| [外部图注](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/captions.txt) | 各图的方法与限制，图中文字维持英文 |

下文 `current/` 指 `reports/exploration_response_profiles_individual_snr_20261002/`，只是路径简写，不是新增目录。

## 当前神经表示的定义

数据范围为 **106 样本、112 个菌株 × 日期条件、13 类细胞、49 只动物、7063 条完整动物曲线**。筛选在菌株 × 日期 × 细胞层面进行，不整株删除，也不整类删除 ASG。

- trials 先在每只动物内等权平均，再用 **0–40 s 未分 bin 的动物平均曲线**估计 SNR。同一条件内动物等权，跨日期汇总菌株时日期等权。
- 最低 **2 只动物**，主门槛 **individual SNR ≥ 0.5**。令 μ 为条件平均曲线，P = meanₜ(μ²)，V = meanₜ(var across animal means, ddof=1)，n 为动物数，则 SNR = sqrt(max(P − V/n, 0) / V)。它度量重复间稳定性，不是刺激前基线噪声比；baseline 指标仅作辅助，没有最低 trial 数 gate。
- 有足够动物但未通过筛选的系数置零；缺失或不足 2 只动物保持 NaN。置零代表分析决策，不证明生理无响应，也不是删除某只动物。
- 每类细胞拟合一个共享的 **0–40 s、8 个 5-s bin** 模板。用保留条件均值的加权非中心化 SVD，菌株等权、菌株内可用日期等权。模板 RMS = 1，绝对值最大 bin 定为正，系数带符号，单位为 ΔF/F₀。
- Profile 展示模板重建的前 **5 bins，即 0–25 s**；建模和比较仍使用完整 0–40 s。RDM 用 13 类细胞的系数计算 1 − cosine；split-half 比较各半模板重建的八 bin 响应，不能直接比较由不同模板得到的两半系数。
- 已保存不筛选及 0.25、0.5、0.75、1 的敏感性结果，各门槛会重新拟合模板。全数据筛选和模板用于描述，验证中只用训练动物重拟合。门槛没有按化学相关性优化。表中的 raw_coefficient 是投影到该门槛模板后未置零的值，不是独立不筛选模型的系数。

主设置在 1456 个日期条件 × 细胞条目中保留 **969**、置零 **487**。合并为 106 × 13 的筛选图后，909 格各日期均保留、454 格各日期均置零、15 格日期间混合。ASG 保留 15/112 个日期条件，不能单独证明响应特异性。逐项记录见 `current/tables/condition_metrics.csv`、`filter_state_by_strain_cell.csv`。

用户曾要求按 trial 筛选，看到过滤过多后改回 individual，并允许降低门槛。统一最低 2 只动物后，trial ≥ 1 保留 254/1456，individual ≥ 1 保留 538/1456，individual ≥ 0.5 保留 969/1456。两种算法的均值及信号功率相同，trial 方差 / individual 方差的中位数为 3.00；trial 总波动中动物内成分的中位比例为 72.5%。同数值门槛不是同筛选强度，动物内波动也不能全部命名为测量噪声。旧 individual 报告的 518 个保留条目还用了最低 3 只动物，不能混用。核查见 `current/snr_definition_audit.json`。

## 当前图稿

各图均有同名 PNG 和 SVG。以下是当前文件，旧图不要按文件序号误认成最新版。

| 内容 | 当前 PNG | 显示约定 |
| --- | --- | --- |
| 筛选状态 | [00_filter_status_heatmap](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/00_filter_status_heatmap.png) | 全 106 × 13，区分保留、置零、日期混合、不可用 |
| 门槛敏感性 | [00b_individual_snr_thresholds](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/00b_individual_snr_thresholds.png) | 各细胞在不同门槛下的保留比例 |
| 整体 profile | [01_response_profile_5bin](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/01_response_profile_5bin.png) | notebook 02 的聚类树、细胞分块、刺激结束虚线、彩色细胞标签；5-bin 展示 |
| 模型表示 | [01b_response_model](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/01b_response_model.png) | 曲线压缩示意，另列完整模板和有符号系数矩阵 |
| 重复性 | [02_repeatability_distribution](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/02_repeatability_distribution.png) | 动物独立分半，右侧为矮宽密度直方图，不用累积分布 |
| 表示检验 | [02b_representation_validation](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/02b_representation_validation.png) | 相同支持下的分半对照及留一动物预测，作为检查材料 |
| RDM 比较 | [03_neural_chemical_rdm](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/03_neural_chemical_rdm.png) | 同一批 106 样本、5565 对，含距离 hexbin 汇总 |
| 2D HMDS | [03b_neural_chemical_hmds_2d](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/03b_neural_chemical_hmds_2d.png) | 上方神经与化学圆盘，下方各自 Shepard diagram |
| 3D HMDS | [03c_neural_chemical_hmds_3d](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/exploration_response_profiles_individual_snr_20261002/figures/03c_neural_chemical_hmds_3d.png) | **最新：神经、化学各一行，XY / XZ / YZ 三投影及右侧 Shepard** |

响应热图、系数图和 RDM 用 **RdBu_r**。用户所说恢复 notebook 配色，后来明确指 **embedding 中点的颜色**，不是把 RDM 改成 magma。点色按样本 ID 复用 notebook 03 保存的 chemical PCo1 → **turbo** 的 `color_hex`；完整 106 样本参考范围为 −2.6673737721171715 至 2.7364307161881327，线性归一化，不以零为中心，也不按子集重新缩放。颜色条用 256 个 turbo 节点、4096 级连续插值。

精确颜色来源是 `output/jupyter-notebook/chemical_hmds_20260928_230847_298095/color_reference/aid_to_chemical_color.csv` 及同目录 `color_parameters.json`，当前显示记录在 `current/figures/comparison_display_parameters.json`。

重复性恢复了原红蓝配色、上方图例和矮宽布局，整图 12.8 × 6.4 inch，直方图轴宽高比约 1.74。bin 宽 0.05，两组各自密度归一化，横轴保留 −1–1，纵轴从零开始且不截峰。布局调整没有重算分半、删除尾部或改变实际分离度。

## 验证结果必须保留的边界

同一有效分半和共同细胞支持上的描述性结果如下。重复分半和共享样本的配对不能当成独立生物重复。

| 表示 | 同菌平均 cosine | 不同菌平均 cosine | 均值差 |
| --- | ---: | ---: | ---: |
| 原始 8-bin | 0.8106 | 0.4117 | 0.3990 |
| 模板，不筛选 | 0.8546 | 0.4618 | 0.3928 |
| 模板，individual SNR ≥ 0.5 | 0.8180 | 0.4427 | 0.3753 |

实际分半 100 次，seed = **20261001**，不要误用函数默认值。新版按日期分全局动物身份、每半至少 2 只、日期等权、至少 4 类完整共同细胞，并分别重拟合门槛与模板。主图有 106 个同菌条目和 5396 个不同菌对，另 169 对缺少有效分半。

原 notebook 的分半规则不同，保存的准备输出还是 13 × 8 bins、0–40 s；不能因为 profile 画五 bin，就把原重复性称为五 bin 基准。原内嵌图均值约为 0.78/0.40，原 split 数值缓存未在本地找到，新旧分布差异不能单独归因为 SNR 或 bin 数。

留一动物预测在共同支持的 6985/7063 条曲线上，筛选模板 MSE = 0.037790，不筛选模板 = 0.036289，零预测 = 0.094151。主门槛比不筛选模板误差高约 **4.14%**。当前可以说筛选减少了展示中的低稳定性条目，但**已有检验不支持筛选提高去噪、预测或样本分离能力**。门槛附近两半筛选不同可能增加不稳定性，尚未保存逐折筛选决定，不能写成已验证原因。

神经与化学 RDM 的描述性 Pearson = 0.2615、Spearman = 0.2133；不筛选模板分别为 0.2399、0.1953。没有把 5565 个重叠配对当独立观测计算普通显著性，也未据此选择门槛。详细数值见 `current/comparison_results.json`。

## HMDS 已扩展到全部 106 样本

先前只显示 81 样本，是神经 bootstrap 有效 draws ≥ 80% 的额外图覆盖规则造成的，保留了 3072 个神经配对。其余 25 样本已参与 profile、重复性和完整 RDM，并未被整套分析丢弃，也不是组间差异太大而不能拟合。

用户要求全部显示后，已移除该覆盖排除：神经 2D、3D 用全部 **106 样本 / 5565 对**重新拟合并检查收敛，化学复用核对过输入的全 106 样本拟合。原始距离和 bootstrap 方差直接复用，无补值、无重做 bootstrap。当前结果在 `current/hmds_full106/2d/` 和 `3d/`；旧 `hmds/`、`hmds3d/` 的神经结果及匹配显示范围是 81 样本版本，不能误当当前结果。

神经 HMDS 输入为 sqrt(2 × cosine distance)，化学为现有 380 个 log₂FC 特征的 RMS 差。1000 次 bootstrap，seed = 20261001，按日期整只动物联合抽样，每次重新筛选、拟合模板，并固定原始条件、日期和配对细胞支持。距离方差为 bootstrap 样本方差，不除以抽样次数。

每对有效 draws 为 229–998/1000，中位数 807，1287 对低于 50%。这些是固定支持仍完整且距离可定义条件下的方差，不能解释为无条件抽样不确定性；全部显示也不代表每个点同等可靠。覆盖表及原 80% mask 保留在 `current/hmds_full106/`，范围见 `scope.json`。

| 全部 5565 对的 Shepard relative RMSE | 2D | 3D |
| --- | ---: | ---: |
| Neural | 0.1393 | 0.1029 |
| Chemical | 0.0903 | 0.0478 |

两种维度使用同一配对集合，Shepard 的 fitted distance 是双曲距离换回输入单位，不能用圆盘或球内欧氏距离替代。这些是拟合误差，不是交叉验证成绩。化学 2D 固定 λ = 10，3D 估计 λ，误差变化不能纯归因于维度；神经两维度均估计 λ。两维度的局部收敛与数值梯度检查已通过，详见各维度 `result.json` 和 `current/hmds_full106/verification.json`。

最新 3D 图直接取保存坐标的 XY、XZ、YZ，各投影 106 点、相同尺度、equal aspect，范围 ±1.04。它们来自同一个三维拟合；两域方向任意且未对齐，投影间距不代表双曲距离，半径也不解释成生物学层级或置信度。此次改图没有重新拟合或旋转坐标，Shepard 保留原三维距离。

旧单视角图及源代码存于 `current/figures/previous_single_view_3d/`，更早的固定视角图存于 `previous_3d_view/`。候选相机角度、选角代码及 `current/hmds_3d_viewer.html` 离线查看器仍保留；当前静态三视图不读取旧选角结果。

## 代码与复现范围

本轮代码在 `current/code/`，原 notebook 继续保留。

| 文件 | 作用 |
| --- | --- |
| `individual_representation.py` | individual SNR、门槛、模板、系数 |
| `individual_comparisons.py` | 动物分半、留一动物、距离及整只动物 bootstrap |
| `individual_profile_display.py` | notebook 风格 profile、独立模型图 |
| `individual_repeatability_display.py` | 矮宽密度直方图和原配色 |
| `individual_comparison_display.py` | RDM、2D HMDS、3D 正交三视图和 Shepard |
| `individual_hmds_full.py` | 全 106 样本 HMDS 拟合，仅需重新分析时使用 |
| `refresh_individual_figures.py` | 读取已保存结果重画，更新图注及显示参数 |
| `run_individual_exploration.py` | 全量分析，会重做 bootstrap 与拟合，不能为改版式而运行 |

仅当用户要求重画时，可在仓库 `.pixi/envs/default/bin/python` 环境调用：

```python
from pathlib import Path
import sys

root = Path('/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis')
out = root / 'reports/exploration_response_profiles_individual_snr_20261002'
sys.path.insert(0, str(out / 'code'))
from refresh_individual_figures import refresh_individual_figures

# 覆盖当前选定图稿与图注，复用已有分析和 hmds_full106 拟合。
refresh_individual_figures(out)
```

无界面绘图设 `MPLBACKEND=Agg`。只改一张图时可调用对应 helper，但要同步图注、参数并保留旧版。全量 runner 要求新的输出目录，已有拟合目录有保护检查。当前目录依赖早期 loader、响应缓存和 notebook HMDS 模块，不是可独立搬走运行的数据包，依赖及哈希以 README、`representation_parameters.json` 为准。

已保存核验包括 11 项针对性合成测试、输入和三个 notebook 哈希、全 106 样本拟合检查、点色逐样本核对、直方图原数组及密度一致性。最新 `current/three_view_verification.json` 位于报告根目录，记录六组实际坐标和点色、两组 Shepard 数组完全一致，232 个受保护文件未变。另独立解析最终 SVG，六图各 106 点，颜色匹配，无裁切。各阶段 verification 对应当时输出，最新显示状态以三视图记录及 `figures/comparison_display_parameters.json` 为准。

## 保留的旧 Figure 4/5 与近邻探索

原 Figure 4 是 A021/A022/A023 × 七类细胞的原始曲线、模板、幅度示意。原 Figure 5 是 **A231/A232**，包含化学散点、两株 × 七类细胞幅度及 AWB、ASH、AWA 代表曲线。文件名虽带 `draft`，内容仍是已交付的该版本，本轮重组尚未改写。

| 历史内容 | 文件入口 |
| --- | --- |
| Figure 4 图稿 | [response_process_draft.png](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/response_process_draft_20261001/response_process_draft.png)，同目录有 PDF、SVG、图注和参数 |
| Figure 4 代码 | [response_structure_process.py](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/notebook/response_structure_process.py) |
| Figure 5 图稿 | [sample_comparison_draft.png](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/sample_comparison_draft_20261001/sample_comparison_draft.png)，同目录有 PDF、SVG、图注和参数 |
| Figure 5 代码 | [sample_comparison_poster.py](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/notebook/sample_comparison_poster.py) |
| 六对近邻幅度探索 | [REPORT](/Volumes/myssd/phd/bacteria_analysis/bacteria_analysis/reports/poster_neighborhood_amplitudes_20261001/REPORT.md)，保留为检查材料，用户已否定其作为 Figure 6 主图 |

旧 Figure 4/5 共用七细胞模板及 ±0.6 幅度色标，图5代码读取图4参数。AWCON 仅 n=2，只做描述；代表曲线用动物均值 ± SEM，保留原始 1 s 采样，重建按 5 s bin。AWA 限于早期方向差异，不能写整条曲线翻转；AWB 有共同模板时程偏离，近似系数不能写成响应等价。用户不希望主图堆 individual 曲线、动物差异点或大块覆盖统计。

近邻探索在既有 147 对可比菌株上统一了 0–40 s 幅度，主范围为五对化学候选加 A022/A023 参照，没有重拟合旧模板。其七细胞范围、最低 3 只配对动物及化学共同报告掩码，与当前全 106 样本分析不同，不能混用。图上的点是删除任一动物后正能量的影响检查，不代表显著性；多个候选共享日期和动物，不能当六次独立验证。代码在 `notebook/neighborhood_amplitudes.py` 与 `neighborhood_amplitudes_display.py`。

旧化学选例中 A250 reference 报告缺失影响尚未核清，化学来自独立培养批次，实际刺激 aliquot 未测，顺序也未建立随机化或平衡。少量差异不能归因于特定化合物、菌株功能、行为或化学之外的独立信息。

更早选例和方法仍在 `reports/poster_current_goal.md`、`poster_methods_notes_20261001.md` 及 `figure5_chemistry_first_20261001/selection_parameters.json`。A022/A023、A040/A041 图5备份也保留。它们记录历史，布局、主参数及后续方向以本交接和用户最新指令为准。

## 接手时不要混用的版本

- `reports/exploration_response_profiles_20261001/` 是最早的表示探索；`exploration_response_profiles_trial_snr_20261001/` 是 trial 版。当前使用 `exploration_response_profiles_individual_snr_20261002/`。
- `current/figures/previous_81_sample_hmds/`、`previous_tall_repeatability/` 及旧单视角目录是历史图。旧 `03b_neural_chemical_hmds.png` 无当前完整 Shepard，交付时使用带 `_2d`、`_3d` 的文件。
- 第6张图仍待用户确定。细胞互补性、化学成分解释、菌株功能映射及新技术优势验证只是讨论过的候选，未授权自动接着跑，现有证据不足的地方应明确保留。
- 工作区已有修改及未跟踪文件，包括 `AGENTS.md`、notebook 02/03、HMDS 模块、`reports/`、`tests/` 等，不是全部由本 session 产生。原 notebook 在本轮探索前后的哈希未变，不等于 Git 工作区干净。继续前检查 `git status --short`，不要回滚已有工作。
- 本次 handoff 更新只读现有结果并整理文档，没有重新运行分析、改动数据或提交 Git。
