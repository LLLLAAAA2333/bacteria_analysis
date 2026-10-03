# 复现本轮分析

先看2026-09-30更新的`REVIEW.md`，再按需查阅`REPORT.md`中的探索记录。现有图已降级为概览、候选检查和诊断材料，不再指定研究主图；PNG、PDF、图注、数据表和脚本仍保留。数值均来自实际运行，但通过运行和复核不代表研究解释已完成。代码只读项目源数据，只写本目录。既有Notebook和未提交改动均保留。

## 环境与输入

使用项目现有Pixi Python 3.11环境，准确包版本、平台、Git HEAD及运行前工作区状态见`logs/environment.json`，依赖锁为项目`pixi.lock`。无需安装新包。输入路径、大小、SHA-256见`logs/input_manifest.json`；主要输入是`data/106bac.parquet`及四个xlsx文件。parquet是已处理ΔF/F₀入口，不含原始荧光或F₀拟合过程。

## 运行

在项目根目录执行以下命令；约几分钟，耗时随机器变化。各脚本也可从Notebook用`%run`调用，本轮不改Notebook。顺序中03与02本身独立，但04/05/06使用之前生成的表。

```bash
.pixi/envs/default/bin/python reports/exploration_20260929/code/01_prepare.py
.pixi/envs/default/bin/python reports/exploration_20260929/code/02_neural_overview.py
.pixi/envs/default/bin/python reports/exploration_20260929/code/03_chemical_prepare.py
.pixi/envs/default/bin/python reports/exploration_20260929/code/04_phenotype_checks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/default/bin/python reports/exploration_20260929/code/05_chemical_tests.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .pixi/envs/default/bin/python reports/exploration_20260929/code/06_asparagine_check.py
.pixi/envs/default/bin/python reports/exploration_20260929/code/07_verify.py
```

`code/run_analysis.py`是同一命令序列的简短运行记录器，可替代上述逐条执行；不接受额外参数。会重建本目录派生表和图，覆盖本轮旧派生版本，不改源数据。报告文字不会自动改写；改参数后必须重新检查报告数字。

## 参数和单位

- 刺激起始为0秒，持续[0,10)秒；观察范围[-5,40)秒，每点1秒。响应单位为存储的ΔF/F₀，无量纲。
- 双侧只在同trial同volume内平均；trial再在同动物同菌株内平均。动物ID为`(date,worm_key)`，不是单独`worm_key`。
- 主神经窗口：stim[0,10)、post[10,30)、full[0,40)；保存late[30,40)与8个5秒窗，但未将窗口当生物学重复。
- 全景随机种子20260929，500次共享动物划半、1000次日期内同步动物重采样；现象检查种子2026092904，10000次边际动物重采样。06的参数写在其代码和日志中。
- 化学主变换为`log2(合作方报告值 + 1)`；+1沿用数值尺度惯例，不是LOD。主面板162项，QC RSD≤0.30且106株全非缺失；248项的扩展面板在每折只用训练菌株中位数插补。缺失原因未知，不一概叫未检测到。
- 化学整体水平为主面板162项log值的样本中位数，不是总分子浓度。旧FC仅作敏感性。神经/化学共同单位为菌株ID，不能推断为同培养批次。
- 日期留一每折排除所有六个跨日菌株，使训练测试菌株无交叉；评价相对响应。没有确认性显著检验，没有把筛选后的区间当作选择校正区间。

## 输出与运行状态

- `tables/`保存trial、动物、日期、菌株层级，以及图用表和完整筛选/反证结果；`trial_curves.parquet`的行是菌株×日期×动物×类神经元×trial，45列是相对时间。
- `logs/run_status.json`记录本轮最终命令、返回码和耗时；`logs/verification.json`记录输入完整性、抽样独立重算、投影正交性、交叉验证与交付文件检查。
- `logs/INDEPENDENT_REVIEW.md`记录独立审查发现和修复。各分析的JSON、日志及`RESEARCH_LOG.md`保留未支持的路线。
- 本轮没有执行：原始荧光重提取/F₀重拟合、GCaMP型号分层（缺元数据）、刺激顺序随机化检验、LC–MS生物学重复误差估计（无对应重复）、确认性分子因果检验、独立实验验证、多变量高复杂度模型。

浏览文献仅检索公开方法资料，没有上传项目数据，也未调用额外付费服务。

## 2026-09-30交付重评

本次只修正摘要、报告和证据定位，未重新运行分析，未修改代码、表或图。此前`verification.json`中的`core_figures`字段以及独立审查中的“核心图/主结论”用语记录的是上一版交付，不再代表当前推荐；原运行日志不改写。旧版四份文档的逐字快照保存在`logs/pre_reassessment_20260930/`。修订及不变文件的校验见`logs/reassessment_20260930.json`。
