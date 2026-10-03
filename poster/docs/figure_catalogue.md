# 当前主图与相关探索图

以下文件名均相对 `poster/figures/`；同名 PNG、SVG、PDF 在本地生成，不提交 Git。以四本 notebook 的来源表与数值检查为准，旧报告中的图名不会自动指代当前版本。

图 1 的输入先由 [00_preparation](../notebooks/00_preparation.ipynb) 从已审核的个体曲线计算，写入 `poster/data/prepared/responses/`。图谱、模板、SNR 和排序的历史报告用于核对；下表中的原来源仍用于追溯方法及独立验证。

| 图组 | 主图文件 stem | 对应支持图 stem（均在 supporting/） |
|---|---|---|
| 01 | `main/fig01_response_atlas`、`main/fig01_method_strip` | `fig01_observed_vs_reconstructed_example`、`fig01_templates_and_amplitudes`、`fig01_condition_support`、`fig01_snr_sensitivity` |
| 02 | `main/fig02_repeatability` | `fig02_full_similarity_matrix`、`fig02_split_support`、`fig02_matched_representation_controls`、`fig02_heldout_prediction` |
| 03 | `main/fig03_global_chemical_neural`（380 项 log2FC） | `fig03_global_chemical_neural_162concentration`；162 项版的 `fig03_pair_calculation_example`、`fig03_pooled_within_between`、`fig03_genus_balanced_distances` |
| 04 | `main/fig04_local_chemical_states` | `fig04_bifidobacterium_member_covariance`、`fig04_example_score_contributions`、`fig04_individuals_bacteroides`、`fig04_individuals_bifidobacterium`、`fig04_neural_iqr_effects`、`fig04_bifidobacterium_leaveout_by_neuron` |

## 图 3 两种表示的文件约定

两版来自同一批化学测量，区别在特征选择与预处理。Notebook 03 先计算和展示 380 项主图，再保留 162 项浓度替代版的计算及探索；图 4 继续使用 162 项浓度。

| 内容 | 380 项 log2FC 主图 | 162 项浓度替代版 |
|---|---|---|
| Pearson r / Spearman ρ | 0.2615119485 / 0.2133169581 | 0.17517 / 0.18546 |
| 神经距离色标 | 0–2 | 0–1.2，超过上界饱和显示 |
| 配对计数 hexbin | RdBu_r，gridsize=27 | Blues，gridsize=25 |
| 图文件 | `main/fig03_global_chemical_neural.*` | `supporting/fig03_global_chemical_neural_162concentration.*` |
| 配对与统计记录（相对 `poster/tables/`） | `fig03_matched_pairs.csv`、`fig03_descriptive_correspondence.json` | `fig03_162concentration_matched_pairs.csv`、`fig03_162concentration_descriptive_correspondence.json` |
| 显示参数与来源记录 | `fig03_display_parameters.json`、`fig03_sources.csv` | 对应文件使用 `fig03_162concentration_` 前缀 |

两版均含 106 株、5,565 个唯一配对。相关系数为描述性指标；共享菌株的配对并非独立样本，不报告普通配对相关检验的 p 值。

## 来源与当前用途

旧报告名称通过 [catalogue.csv](../../analysis/explorations/catalogue.csv) 映射到分组后的缓存与源码目录。

| 原报告名称与文件 | 整理后的入口 | 用途 |
|---|---|---|
| `exploration_20260929/tables/animal_curves.parquet`、`animal_metrics.csv`、`trial_curves.parquet`；`response_structure_20260930/data/observations.parquet` | Notebook 00；`preparation.py` | 从个体均值真实重建响应表示；trial 刺激前样本和审核分箱只用于输入核对 |
| `exploration_response_profiles_individual_snr_20261002` | Notebook 01、02；`atlas.py`、`repeatability.py` | 当前神经表示、图谱、重复性、匹配对照和参数敏感性 |
| `response_structure_20260930` 等早期响应结构 | Notebook 01 方法说明；原实现保留 | 模板模型与表示的历史来源；不在新 notebook 中重拟合 |
| `population_first_20260930/tables/aligned_chemical_log2fc_paired.csv` | Notebook 03 主图 | 380 项 log2FC：缺失填 0、加 1、按参考组计算；106 株全局比较 |
| `exploration_chemical_pattern_direct_report_20261003` | Notebook 03 替代版、04 主图 | 162 项 log2 浓度、注释 metadata、13 维神经单位向量；完整、正浓度、QC RSD ≤ 0.30 |
| `exploration_genus_within_between_20261003` | Notebook 03 的 162 项替代版；`global_comparison.py` | 162 项浓度距离矩阵核对、属内/属间背景，不作为 380 项主图的化学距离 |
| `exploration_genus_patterns_independent_20261003` | Notebook 03 的背景小节 | 两侧各自的属模式；不把独立拟合的模块解释成跨模态对应 |
| `exploration_bacteroides_local_model_20261003/chemical` | Notebook 04；`chemical_axes.py` | 保留经验证的纯化学训练/变换算法 |
| `exploration_bacteroides_adf_ash_chemical_20261003` | Notebook 04；`local_states.py` | Bacteroides 固定 ADF−ASH 目标及历史留出结果 |
| `poster_local_chemical_neural_20261003` | Notebook 04 | `figure_data/` 是已接受主图；`tables/` 还含失败的 Bacteroides 3 项全向量模型，二者不能混用 |
| `exploration_bacteroides_neural_reliability_20261003` | 历史脚本索引、原 README | 神经可靠性、PC1 与子空间探索，作为方法背景，不增加主图 |

## 保留但不放进当前主图的内容

- 全 106 株 HMDS 的化学侧采用 380 项 log2FC，与恢复后的图 3 主图定义一致；Notebook 03 仍将其作为支持探索并注明来源，没有重拟合 HMDS。
- A231/A232 旧样本对图采用早期 7 神经元表示和另一套化学输入。Notebook 03 用于回顾选图过程；不能与当前两张化学–神经图的数值直接拼接。
- 更早的 population、neighborhood、pair-quadrants、全样本预测等探索的源码保留在 `analysis/explorations/`，结果保留在本地 `reports/`；没有因为新故事而删除负面结果。完整实现位置见 [脚本索引](script_inventory.csv)。

主版面保留四组结果和方法条即可。重复性完整矩阵、参数敏感性、模型留出明细更适合 notebook 内的解释与交流，不必都挤入 poster。
