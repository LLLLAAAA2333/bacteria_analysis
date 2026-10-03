# Poster 工作区整理交接

> 这是 2026-10-03 的历史交接。目录与 Git 保存范围已在后续整理中调整，当前规则见 [仓库结构](../repository_layout.md)；下文保留当时记录。

当前入口为 `poster/README.md`；每张主图一本 notebook，相关探索图在同册。后续直接维护 notebook 与 `poster/src/poster_analysis/`，不继续往旧 notebook 末尾追加整段流水线。

补充准备入口：`00_preparation.ipynb` 从已审核的个体 ΔF/F₀ 平均曲线计算 SNR、五个固定阈值下的模板/系数、菌株汇总、raw-profile 排序和显示表，写入 `poster/data/prepared/responses/`；01 已改为读取该目录。历史 NPZ 和表只用于外部一致性核对。00 不等于从原始荧光开始的完整流水线，也不重跑 split-half、LOAO 或化学预处理。

## 当前图组

1. `01_response_atlas.ipynb`：106 株 × 13 神经元，显示五个时间箱；旁附曲线→分箱→模板×系数方法条。包含覆盖、完整模板/系数、重建误差示例、SNR 敏感性。
2. `02_repeatability.ipynb`：同菌株/异菌株 split-half 相似度分布；包含完整矩阵、有效拆分、匹配表示与 LOAO 缓存汇总。
3. `03_global_chemical_neural.ipynb`：主图恢复 380 项 log2FC，106 株、5,565 个唯一菌株对，Pearson r=0.2615119485、Spearman ρ=0.2133169581；保留原版神经距离色标 0–2、RdBu_r 计数 hexbin（gridsize=27）。162 项浓度版保留为支持图（r≈0.17517、ρ≈0.18546；神经色标 0–1.2、Blues hexbin、gridsize=25）。Notebook 先主版、后替代版及其探索。全 106 株 HMDS 与主图使用同一 380 项化学定义，仍属支持探索；旧 A231/A232 的 7 神经元图仍是历史参考。
4. `04_local_chemical_states.ipynb`：旧 Bacteroides 14 项 L03 / 固定 ADF−ASH，及 Bifidobacterium 21 项 L02 / 全 13 维目标。重建已接受 `figure_data/`，保留失败的 Bacteroides 3 项 L12，展示一个训练内重建折，完整留出指标读缓存并独立汇总。

## 代码与数据

- 共享模块 `atlas.py`、`repeatability.py`、`global_comparison.py`、`local_states.py` 接收显式数据/路径；`chemical_axes.py` 保留已经验证的纯化学算法。
- `paths.py` 统一来源别名，notebook 根据仓库标记定位，不绑定机器的卷名。
- 新图在 `poster/figures/main/` 和 `supporting/`，三种格式；计算表与输入指纹在 `poster/tables/`。
- 图 3 主图沿用 `main/fig03_global_chemical_neural.*`；`fig03_matched_pairs.csv`、`fig03_descriptive_correspondence.json`、`fig03_display_parameters.json`、`fig03_sources.csv` 对应 380 项主版。162 项替代图为 `supporting/fig03_global_chemical_neural_162concentration.*`，计算记录使用 `fig03_162concentration_` 前缀。
- 两版化学表示来自同一批测量：380 项采用缺失填 0、加 1、按参考组计算 log2FC；162 项采用完整、正浓度及 QC RSD ≤ 0.30 筛选后的 log2 浓度。图 4 仍使用 162 项浓度，不因图 3 恢复而切换输入。
- 旧报告和旧 notebook 原位保留。当前 notebook 不导入旧运行脚本。00 按固定规则重算全样本模板和阈值敏感性；原始成像、HMDS 或完整留出搜索仍不重跑。
- 四份历史根 handoff 移到本目录并保留原文，旧位置是相对符号链接；历史内容不是当前分析定义。

## 科学解释

主线是响应表示有可重复结构，整体化学–神经距离对应较弱，而局部化学状态能为部分菌株的神经配置提供意义。5,565 个配对共享菌株，相关系数只作描述，不报告普通配对相关检验的 p 值。不要将全数据监督选组后的散点当作独立验证，不声称全部神经元都相关，也不要求各属共享同一规律。跨记录条件信息只用于既有汇总、支持和内部验证；不作为 poster 的主题。

## 复现与核对

从仓库根执行 `pixi run python poster/scripts/execute_notebooks.py`，四本分别使用新的 kernel，写回执行输出。用 `PYTHONPATH=poster/src pixi run python -m unittest discover -s poster/tests -v` 检查关键科学约定。

本次执行与文件保护检查记录在 `poster/docs/notebook_execution.json`、`verification.json`；图中计算值的核对记录在各 `fig*_checks*` 表。复制代码到别处时仍需携带来源表列出的本地报告数据。
