# Poster 工作区

按四组结果组织：**响应图谱 → 可重复性 → 全体菌株的化学–神经比较 → 局部化学状态对应的神经配置**。方法条附在第一组旁边。Notebook 是当前分析的主入口；共享函数接收明确的数据参数，不依赖旧脚本的动态导入。

先运行 **[00_preparation](notebooks/00_preparation.ipynb)** 准备图 1 的输入。它从已审核的个体平均曲线重新计算 SNR、模板、系数、未筛选曲线距离排序、五箱显示和阈值敏感性，导出到 `poster/data/prepared/responses/`。Notebook 01 明确读取这些文件；旧报告只作为核对基准。这里的起点已经是 ΔF/F₀ 个体曲线缓存，尚未将原始荧光预处理迁入 poster。

| 主图 | Notebook | 当前导出 | 同一本 notebook 中的探索图 |
|---|---|---|---|
| 1 · 106 株 × 13 类神经元响应图谱 | [01_response_atlas](notebooks/01_response_atlas.ipynb) | [图谱](figures/main/fig01_response_atlas.png) · [方法条](figures/main/fig01_method_strip.png) | 观测与重建、完整模板及幅度、覆盖、SNR 敏感性 |
| 2 · 同菌株响应的可重复性 | [02_repeatability](notebooks/02_repeatability.ipynb) | [Split-half 分布](figures/main/fig02_repeatability.png) | 完整相似度矩阵、拆分支持、匹配表示对照、留一动物预测 |
| 3 · 全体菌株的化学–神经比较 | [03_global_chemical_neural](notebooks/03_global_chemical_neural.ipynb) | [380 项 log2FC：双距离矩阵与配对分布](figures/main/fig03_global_chemical_neural.png) | 162 项浓度替代版及其单对计算、属内/属间距离；380 项 HMDS 支持图；旧样本对历史图 |
| 4 · 局部化学状态与神经配置 | [04_local_chemical_states](notebooks/04_local_chemical_states.ipynb) | [局部关系与 13 神经元组均值](figures/main/fig04_local_chemical_states.png) | 状态成员共变、单株贡献、个体散点、IQR 效应、逐神经元留出结果、一个训练折 |

导出图、计算表和准备缓存只保留在本地，不纳入 Git。全部新导出均有同名 PNG、SVG、PDF；文件位于 [figures/main](figures/main/) 和 [figures/supporting](figures/supporting/)。PNG 便于预览，SVG/PDF 用于后续排版。

## 如何使用

从本仓库根目录或 `poster/notebooks/` 打开 notebook，选择项目 Pixi 环境的 Python kernel，执行 **Restart Kernel and Run All**。源码 notebook 保存代码和叙述；完整已执行副本在本地 `reports/notebooks/poster/`。新 checkout 需要另外提供 [本地输入清单](docs/local_inputs.csv) 列出的缓存。

```bash
# 首次准备或更新准备实现后：重建输入，再画图 1
pixi run python poster/scripts/execute_notebooks.py 00 01

# 在仓库根目录：执行四本当前 notebook，输出副本写入 reports/notebooks/poster/
# 默认使用已准备的输入，不隐式重跑 00
pixi run python poster/scripts/execute_notebooks.py

# 只重新执行其中一本
pixi run python poster/scripts/execute_notebooks.py 04

# 检查关键科学计算约定
pixi run test

# 手动执行 notebook 后，保留本地输出副本并清理待提交 notebook
pixi run clean-notebooks
```

日常修改：问题、参数、解释和少量探索代码写在 notebook；多个图共用的计算或绘图函数放入 `src/poster_analysis/`。修改后重跑对应 notebook，再查看新导出。执行器默认运行四本绘图 notebook；指定 `00` 时才运行准备。执行输出与运行记录保存到忽略的 `reports/`，不会写回源码 notebook。它不会发现并运行旧 notebook。

## 本次整理中的科学边界

- 图 1 读取 00 按既有 individual-SNR ≥ 0.5、最少两只动物规则准备的表示。图 1 本身不重拟合模板；灰色缺失与筛选后的零有不同含义。
- 图 2 的 split-half 和 LOAO 仍读取已验证缓存；它们涉及训练子集内重新拟合，不能从全样本模板直接推导。00 的覆盖范围是本次列出的图 1 准备文件，不是全部原始数据和验证分析的总流水线。
- 图 3 主图恢复 **380 项 log2 fold-change**：106 株、5,565 个唯一菌株对，描述性 Pearson r = 0.262、Spearman ρ = 0.213。保留原版神经距离色标 0–2 和 RdBu_r 计数 hexbin（gridsize=27）。[162 项浓度替代版](figures/supporting/fig03_global_chemical_neural_162concentration.png) 留在支持图中（r = 0.175、ρ = 0.185；神经色标 0–1.2、Blues hexbin，gridsize=25）。共享菌株的配对不能视为独立样本，均不据此报告普通配对相关检验的 p 值。
- 两套化学表示来自同一批原始化学测量，但特征筛选和预处理不同：380 项采用缺失填 0、加 1、按参考组计算 log2FC；162 项要求完整、正浓度及 QC RSD ≤ 0.30，使用 log2 浓度。Notebook 03 先展示 380 项主图，再保留 162 项计算和探索；图 4 仍使用 162 项浓度。
- 图 4 保留 Bacteroides 的固定 ADF−ASH / 14 项状态，并展示 Bifidobacterium 的 13 维目标 / 21 项状态。它们覆盖 40 株；下方显示全部 13 个神经元的实测组均值，不能据此声称每个神经元均有关联。失败的 Bacteroides 全 13 维扩展和训练折选择过程保留在 notebook 中。
- 全 106 株 HMDS 的化学定义与恢复后的 380 项主图一致，仍作为支持探索；旧 A231/A232 示例采用早期 7 神经元表示，保留为历史参考。

图 3 主图继续使用 `fig03_global_chemical_neural.*`；主计算表为 `fig03_matched_pairs.csv`、`fig03_descriptive_correspondence.json`、`fig03_display_parameters.json`、`fig03_sources.csv`。162 项替代版的计算记录使用 `fig03_162concentration_` 前缀，避免两种表示混用。

详细入口：[图与探索目录](docs/figure_catalogue.md) · [英文图注](docs/captions.md) · [目录设计与函数分工](docs/architecture.md) · [历史脚本索引](docs/script_inventory.csv) · [交接记录](../docs/handoffs/README.md)。
