# 检阅与复现

先读[REVIEW.md](REVIEW.md)，再看[五细胞原曲线](figures/response_patterns_asi_asj.png)。[REPORT.md](REPORT.md)包含科学判断、反证、覆盖、图注和表格入口。这里是独立新一轮输出，不覆盖前两轮结果。

## 运行

在项目根目录，用已有pixi环境执行：

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .pixi/envs/default/bin/python reports/atlas_extension_20260930/run_analysis.py
```

脚本由自身路径定位项目，可以逐个执行，也可在已有Notebook中用`%run`调用；未新增或重构Notebook。完整运行只覆盖本目录内同名输出，不修改原始数据或既有结果。

| 顺序 | 代码 | 实际用途 |
|---|---|---|
| 1 | [00_provenance.py](code/00_provenance.py) | 校验既有输入与环境 |
| 2 | [01_atlas_roles.py](code/01_atlas_roles.py) | 全13细胞加删、三态、四/六细胞与同属/首后trial反证 |
| 3 | [02_response_patterns.py](code/02_response_patterns.py) | 原曲线、78细胞对探索、留动物五细胞模式及反例 |
| 4 | [04_targeted_chemistry.py](code/04_targeted_chemistry.py) | 每目标调参的162/248面板预测 |
| 5 | [03_chemical_profiles.py](code/03_chemical_profiles.py) | 化学类别两种表示、直接属内拟合及核验 |
| 6 | [05_chemical_checks.py](code/05_chemical_checks.py) | 同属、删株/块、首后trial和共同ASI/ASJ目标 |
| 7 | [06_within_genus_chemistry.py](code/06_within_genus_chemistry.py) | 162面板训练目标也限定属内的受限后续分析 |
| 8 | [07_population_bridge.py](code/07_population_bridge.py) | 同一神经高低分组的化学预测重建 |
| 9 | [08_verify.py](code/08_verify.py) | 调用03的既有模型贡献分解，统一核验输入、数字、覆盖与文档链接 |

编号保留并行工作顺序，实际依赖如表。03的`interpret_bridge()`需要07先完成，08会调用它；它只分解已有预测，不重新拟合。03与04的基线是独立实现，统一参数后逐行一致。

各分析脚本已经分别实际运行成功；最后统一验证结果见[verification.json](logs/verification.json)，运行状态见[run_status.json](logs/run_status.json)。提供的完整串行入口未为重复计时而再次执行全部模型，逐个执行的成功不能冒称全入口已经端到端运行。若重新运行，入口会自动写实际退出码和用时；任何失败立即停止并保留日志。本轮没有尚未运行却用于报告的模型或图。

## 输入、参数与结果

- 必需输入主要位于`reports/exploration_20260929/tables/`：动物指标、动物曲线、trial曲线、taxonomy、chemical_log、feature_metadata与reference_groups；用于算术核对的前轮预测位于`reports/population_exploration_20260930/tables/`。29个继承/本轮输入的路径、大小及SHA256见[输入清单](logs/input_manifest.json)，包含原始文件的既有追溯关系。若输入变动，00会明确失败，不悄悄使用旧结论。
- 环境版本和Python可执行文件见[environment.json](logs/environment.json)。使用已有numpy/pandas/scipy/matplotlib/pyarrow等；没有安装依赖或调用外部付费服务。无需联网运行。
- 窗口stim `[0,10)`、post `[10,30)`、探索full `[0,40)`，1秒采样；所有数值来自存储ΔF/F₀。三态阈值0、0.05、0.1；化学岭参数0.01、0.1、1、10、100。随机种子固定在脚本和`*_methods.json`；重采样2,000次仅作固定预测下的描述范围。
- `tables/`保留逐观察预测、动物配对结果、训练参数选择、完整失败结果和图表数据。`figures/`提供PNG与矢量PDF；核心图以原曲线为主，贡献审计和化学预测均有明确适用范围。
- `logs/`保留各路方法、运行输出、自查、独立审查和最终文件校验。原始数据及前轮输入校验一致；本轮输出清单见[output_manifest.json](logs/output_manifest.json)。

## 未完成与禁止外推

未做独立实验验证、化合物干预、精确动力学拟合或刺激制备来源的外部核实。没有把钙信号当放电、把同动物trial当生物学重复、把LC–MS报告当实际暴露浓度，也没有声称日期/菌株混杂已被模型解决。主结果的动物、菌株覆盖各不相同，具体身份留在表中；图中没有日期编码。

分析路线及被否定的解释见[RESEARCH_LOG.md](RESEARCH_LOG.md)。
