# 复现与检阅入口

先读[REVIEW.md](REVIEW.md)，具体证据、图注、样本范围与解释边界在[REPORT.md](REPORT.md)。本轮只写入`reports/population_exploration_20260930/`，不改原始数据、旧输出或Notebook。

## 输入与环境

复用上一轮独立核验的动物／trial曲线、化学报告值、QC及分类信息。复用的是处理数据，不继承上一轮科学结论。主要输入：

- 神经源：`data/106bac.parquet`。本轮直接由源表独立重建图2的16条动物曲线，核对绘图数据。
- 神经派生：`reports/exploration_20260929/tables/animal_metrics.csv`、`animal_curves.parquet`、`trial_curves.parquet`、`trial_design.csv`。
- 化学源：`data/metabolism_raw_data.xlsx`、`matrix.xlsx`、`GM300_bacteria_species_summary.xlsx`、`current_samples.xlsx`。
- 化学派生：上一轮`tables/chemical_log.csv`、`chemical_raw.csv`、`chemical_feature_metadata.csv`、`chemical_reference_groups.csv`、`taxonomy.csv`。

准确路径、大小及SHA-256在[输入清单](logs/input_manifest.json)；处理入口`01_prepare.py`、`03_chemical_prepare.py`的旧版代码也已校验。项目现有Pixi环境，无新增依赖。版本、Python可执行文件、Git HEAD和工作区状态在[环境信息](logs/environment.json)，依赖锁为项目`pixi.lock`。

## 一条命令复现

在项目根目录运行：

```bash
.pixi/envs/default/bin/python reports/population_exploration_20260930/code/run_analysis.py
```

所有脚本也可单独用这个Python执行，或在Notebook中`%run`，按01到10顺序；本轮未新建Notebook。运行器将BLAS线程限制为1，逐个记录命令、退出码和耗时到[run_status.json](logs/run_status.json)。它只重建本轮派生表和图，不自动改报告文字；修改参数后需要重新检查文字数字。

## 代码与研究问题

|脚本|作用|
|---|---|
|[01_prepare.py](code/01_prepare.py)|输入版本与环境核查，确认原数据未改变|
|[02_population_patterns.py](code/02_population_patterns.py)|全13类组合、共同增益反证、同动物菌属对照|
|[03_temporal_patterns.py](code/03_temporal_patterns.py)|宽阶段组合与首轮／后续检查，记录未保留的时程路线|
|[04_chemical_groups.py](code/04_chemical_groups.py)|不读神经结果的化学组、成员、质量及共变诊断|
|[05_population_information.py](code/05_population_information.py)|整只动物留出的组合与幅度对照|
|[06_information_checks.py](code/06_information_checks.py)|共同细胞、去AWCON、顺序趋势和首后trial转移|
|[07_population_chemistry.py](code/07_population_chemistry.py)|全实验块嵌套划分的化学组合预测|
|[08_chemical_neighbors.py](code/08_chemical_neighbors.py)|固定k=3局部非线性反证及邻居身份|
|[09_synthesis.py](code/09_synthesis.py)|原单位／菌株均值评分、主图直接对比、图2及补充图|
|[10_verify.py](code/10_verify.py)|源表重算、预测／权重／划分／输入完整性与交付检查|

主要随机种子按脚本记录：2026093002（300次动物划分、4,000次动物bootstrap）、2026093004（200次化学模块bootstrap）、2026093005（2,000次固定预测动物bootstrap）、2026093009（显示抖动）、2026093010（数值不变性检查）。其余确定性步骤不依赖随机抽样。

神经单位为存储ΔF/F₀，时间单位秒；刺激[0,10)，后期[10,30)，同支持平均[0,30)，完整[0,40)。每个动物的所有刺激、细胞和阶段共同划分，trial先平均。化学变换log₂(报告值+1)，+1是数值约定，不是检测限。七个窄组是化学探索结果，预测中训练内重做筛选；162项QC完整面板固定于模型前。岭惩罚候选0.01、0.1、1、10；九外折／八内折。近邻固定k=3，不按神经结果调k。

## 图和对应数据

主图仅两张：[响应组合与共同动物](figures/explore_population_combinations.png)、[同属两个菌株的配比](figures/02_within_genus_composition.png)，均有同名PDF。主图不显示日期或把日期当生物类别。

补充：[逐动物信息对照](figures/S01_population_information.png)、[Bifidobacterium全13类阶段探索](figures/explore_temporal_bifidobacterium.png)。后者用于检查选择过程，不作为精细时程编码的主证据。所有图注和数据表链接均在报告对应段落。

## 状态与未完成项

最终完整复现的10个脚本全部退出0，总耗时67.72秒。26个输入校验不变；从源parquet独立重建图2的16条曲线，最大绝对误差2.22×10⁻¹⁶；预测、权重、训练测试分隔及文档链接检查通过。实际运行记录以[run_status.json](logs/run_status.json)及[verification.json](logs/verification.json)为准；独立审查在[independent_review.md](logs/independent_review.md)。数值检查通过不代表独立生物学验证。

未执行原始荧光／F₀重新提取、匹配培养样本补测、受体或纯品操纵、刺激顺序随机化实验、独立新动物批次验证、行为效价推断。化学模型仅评价指定的刺激期及40秒目标，没有穷尽非线性模型、时间表示或未测代谢物。未上传未公开数据、未发布结果、未调用额外付费服务。
