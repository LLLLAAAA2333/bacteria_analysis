# 复现入口：化学邻域内的群体响应

先读 [REVIEW.md](REVIEW.md)，详细方法、限制及结果在 [REPORT.md](REPORT.md)。本轮仅写入本目录，沿用已核验的输入，不重跑或修改Notebook、不重建原始预处理。

## 运行

从项目根目录执行，使用现有Pixi环境，无新增依赖：

```bash
.pixi/envs/default/bin/python reports/poster_neighborhoods_20260930/code/01_pair_signal.py
.pixi/envs/default/bin/python reports/poster_neighborhoods_20260930/code/02_pair_context.py
.pixi/envs/default/bin/python reports/poster_neighborhoods_20260930/code/03_neuron_patterns.py
.pixi/envs/default/bin/python reports/poster_neighborhoods_20260930/code/04_poster_figures.py
.pixi/envs/default/bin/python reports/poster_neighborhoods_20260930/code/05_verify.py
```

01→02/03→04→05。重复运行会更新本轮同名输出。代码也可用Notebook中的`%run`调用；没有新增Notebook或CLI框架。01–04均已实际执行并成功；04在修改色标和覆盖图注后再次执行。05的实际状态与检查计数见 [verification.json](logs/verification.json)，独立复核见 [independent_review.md](logs/independent_review.md)。

分析本身没有随机拟合、抽样或置换；05从原trial表独立抽查首次／后续转移的随机种子为 **2026093003**。环境版本、SHA256和时间记录在 [environment.json](logs/environment.json)、[input_manifest.json](logs/input_manifest.json)、[output_manifest.json](logs/output_manifest.json)。已有项目环境由项目根目录`pixi.toml`、`pixi.lock`锁定，本轮不安装依赖。

## 输入与分析单位

| 输入（相对项目根目录） | 用途 |
| --- | --- |
| `reports/population_first_20260930/tables/aligned_neural_animal_5bins.parquet` | 607动物×菌株行；106株、49动物；13类×5窗 |
| 同目录 `neural_value_pairs.csv` | 147个同菌属、同reference、同采集目录pair；既有近邻规则 |
| 同目录 `population_structure_loadings.csv` | 仅核对原固定每细胞尺度，不用其latent axes |
| 同目录 `aligned_chemical_log2fc_paired.csv` | Notebook03对应的106×380 log₂FC，不再加1、删列或标准化 |
| 同目录 `aligned_chemical_report_observed_paired.parquet`、`aligned_chemical_metadata.csv` | 原报告缺失标记、QC信息；不等同分子不存在或已知检出限 |
| `reports/exploration_20260929/tables/trial_curves.parquet` | 首次／后续trial和刺激segment位置；0–25秒五窗 |

神经观测为同动物两株的trial平均钙响应差，动物身份为采集块+worm_key。不同动物差值之间的乘积不含自身平方项；主分数按细胞平均。试次、时间窗、神经元、共享端点的pair不是额外独立生物重复。本轮不做独立pair的OLS/p值或确认性FDR。

固定参数：13类细胞同时进入；5个5秒窗从0到25秒（刺激0–10秒）；每cell×pair≥3只动物；缺失不填零；负估计保留。主图为化学预选近邻中候选组≥3株的28对，26对完整13类、2对10类。每细胞缩放是全目录定义的描述性坐标，不是独立训练或功能重要性权重。03转移检查的选择和尺度则排除整只留出动物。

## 输出对应关系

| 代码 | 核心输出 | 科学用途 |
| --- | --- | --- |
| `01_pair_signal.py` | `pair_signal_*`表和日志 | 群体分离程度、逐动物差值、删动物范围、原单位及覆盖敏感性 |
| `02_pair_context.py` | `pair_context_*`表和日志 | 组内species、化学共同报告部分、刺激位置的有限规律检查；删整株／整块 |
| `03_neuron_patterns.py` | `neuron_patterns_*`表和日志 | 贡献集中度、最大细胞尺度敏感性、留动物首次／后续转移 |
| `04_poster_figures.py` | `poster_*`数据表；PNG/PDF/SVG | 群体差异谱、147对背景、species倾向、原单位附图 |
| `05_verify.py` | 清单、环境和验证日志 | 数值恒等式、输入未变、输出完整性、trial重构、文档链接 |

主图全部值可从 [poster_neighborhood_rows.csv](tables/poster_neighborhood_rows.csv) 和 [poster_cell_contributions.csv](tables/poster_cell_contributions.csv)重绘。逐动物原始尺度差值保存在 [pair_signal_animal_differences.csv](tables/pair_signal_animal_differences.csv)。日期只作数据溯源与设计分组，不在poster主图编码。

## 实际检查与未完成边界

已检查：147对贡献求和、公式独立重算、整个动物排除、训练/测试分离、13类/共同10类覆盖、原始单位与排除本pair的尺度、刺激位置、species与化学覆盖门槛、删整株/整块敏感性，以及PNG图面和向量输出。独立审阅还重构了全部20,498条转移细胞值及1,776个选择折。

未执行新的培养/成像、实际刺激aliquot化学复测、跨日期独立验证或打乱顺序实验；没有分子因果结论或神经优于化学的预测任务。旧目录输出清单和原始输入的SHA256复核记录在05日志中。实验限制不等于这批数据不能用于群体表型描述。
