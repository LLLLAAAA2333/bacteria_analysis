# 历史探索归档

这里按主题保存 32 项历史探索的 Python 源码、协议和文字记录。对应的表格、拟合参数、核对结果、图与 PDF 保留在本地 `reports/<主题>/<报告名>/`，与源码分开存放。

历史文件保留迁移前的原始字节，包括当时的路径、环境约定、输出目录、文档链接和源码哈希。这些脚本不能直接当作当前入口运行；如果要继续某项探索，需要先按当前目录结构移植所需脚本，再检查输入、输出和方法约束。当前分析入口见 [notebooks/](../../notebooks/)、[src/bacteria_analysis/](../../src/bacteria_analysis/) 和 [poster/](../../poster/)。

[catalogue.csv](catalogue.csv) 记录每项探索的原路径、源码目录和本地产物目录，所有路径均相对于仓库根目录。[source_manifest.csv](source_manifest.csv) 记录实际迁移的源文件及其 SHA-256，可用于检查原始内容是否完整。目录内原有文档中的相对链接和绝对路径保持当时写法，查找现存文件时以这两份清单为准。部分报告只留下本地产物，对应源码目录中的说明会标出这种情况。

本地产物链接在缺少历史缓存的 checkout 中可能不可用，目录缺失不代表需要重新运行分析。

## population

早期总体探索，包含输入核对、响应图谱、群体结构和化学预测。

- `exploration_20260929`：[源码与文档](population/exploration_20260929/) · [本地产物](../../reports/population/exploration_20260929/)
- `atlas_extension_20260930`：[源码与文档](population/atlas_extension_20260930/) · [本地产物](../../reports/population/atlas_extension_20260930/)
- `population_exploration_20260930`：[源码与文档](population/population_exploration_20260930/) · [本地产物](../../reports/population/population_exploration_20260930/)
- `population_first_20260930`：[源码与文档](population/population_first_20260930/) · [本地产物](../../reports/population/population_first_20260930/)

## representation

响应表示及其展示，包含各版 SNR 定义、模板、可靠性比较和响应结构草图。

- `exploration_response_profiles_20261001`：[源码与文档](representation/exploration_response_profiles_20261001/) · [本地产物](../../reports/representation/exploration_response_profiles_20261001/)
- `exploration_response_profiles_trial_snr_20261001`：[源码与文档](representation/exploration_response_profiles_trial_snr_20261001/) · [本地产物](../../reports/representation/exploration_response_profiles_trial_snr_20261001/)
- `exploration_response_profiles_individual_snr_20261002`：[源码与文档](representation/exploration_response_profiles_individual_snr_20261002/) · [本地产物](../../reports/representation/exploration_response_profiles_individual_snr_20261002/)
- `response_structure_20260930`：[源码与文档](representation/response_structure_20260930/) · [本地产物](../../reports/representation/response_structure_20260930/)
- `response_structure_display_20260930`：[源码与文档](representation/response_structure_display_20260930/) · [本地产物](../../reports/representation/response_structure_display_20260930/)
- `response_structure_display_sketch_20260930`：[源码与文档](representation/response_structure_display_sketch_20260930/) · [本地产物](../../reports/representation/response_structure_display_sketch_20260930/)
- `response_structure_poster_20261001`：[源码与文档](representation/response_structure_poster_20261001/) · [本地产物](../../reports/representation/response_structure_poster_20261001/)
- `response_process_draft_20261001`：[源码与文档](representation/response_process_draft_20261001/) · [本地产物](../../reports/representation/response_process_draft_20261001/)

## chemical_neural

化学与神经关系探索，包含邻域、配对背景、化学模式及跨模态预测。

- `chemical_neighborhood_focus_20260930`：[源码与文档](chemical_neural/chemical_neighborhood_focus_20260930/) · [本地产物](../../reports/chemical_neural/chemical_neighborhood_focus_20260930/)
- `exploration_neural_compound_relations_20261002`：[源码与文档](chemical_neural/exploration_neural_compound_relations_20261002/) · [本地产物](../../reports/chemical_neural/exploration_neural_compound_relations_20261002/)
- `exploration_pair_quadrants_20261002`：[源码与文档](chemical_neural/exploration_pair_quadrants_20261002/) · [本地产物](../../reports/chemical_neural/exploration_pair_quadrants_20261002/)
- `exploration_matched_pair_context_20261003`：[源码与文档](chemical_neural/exploration_matched_pair_context_20261003/) · [本地产物](../../reports/chemical_neural/exploration_matched_pair_context_20261003/)
- `exploration_chemical_pattern_recurrence_20261003`：[源码与文档](chemical_neural/exploration_chemical_pattern_recurrence_20261003/) · [本地产物](../../reports/chemical_neural/exploration_chemical_pattern_recurrence_20261003/)
- `exploration_chemical_pattern_direct_report_20261003`：[源码与文档](chemical_neural/exploration_chemical_pattern_direct_report_20261003/) · [本地产物](../../reports/chemical_neural/exploration_chemical_pattern_direct_report_20261003/)

## genus

属内与属间结构，包含两种模态分别开展的模式分析和距离比较。

- `exploration_genus_patterns_independent_20261003`：[源码与文档](genus/exploration_genus_patterns_independent_20261003/) · [本地产物](../../reports/genus/exploration_genus_patterns_independent_20261003/)
- `exploration_genus_within_between_20261003`：[源码与文档](genus/exploration_genus_within_between_20261003/) · [本地产物](../../reports/genus/exploration_genus_within_between_20261003/)

## local_states

局部化学状态与神经响应，包含 Bacteroides 模型、固定对比、可靠性和局部展示。

- `exploration_bacteroides_local_model_20261003`：[源码与文档](local_states/exploration_bacteroides_local_model_20261003/) · [本地产物](../../reports/local_states/exploration_bacteroides_local_model_20261003/)
- `exploration_bacteroides_adf_ash_chemical_20261003`：[源码与文档](local_states/exploration_bacteroides_adf_ash_chemical_20261003/) · [本地产物](../../reports/local_states/exploration_bacteroides_adf_ash_chemical_20261003/)
- `exploration_bacteroides_neural_reliability_20261003`：[源码与文档](local_states/exploration_bacteroides_neural_reliability_20261003/) · [本地产物](../../reports/local_states/exploration_bacteroides_neural_reliability_20261003/)
- `poster_local_chemical_neural_20261003`：[源码与文档](local_states/poster_local_chemical_neural_20261003/) · [本地产物](../../reports/local_states/poster_local_chemical_neural_20261003/)

## examples

历史展示与样本例子，包含邻域图、候选例子筛选和样本比较的多个版本。

- `poster_neighborhoods_20260930`：[源码与文档](examples/poster_neighborhoods_20260930/) · [本地产物](../../reports/examples/poster_neighborhoods_20260930/)
- `poster_neighborhood_amplitudes_20261001`：[源码与文档](examples/poster_neighborhood_amplitudes_20261001/) · [本地产物](../../reports/examples/poster_neighborhood_amplitudes_20261001/)
- `figure5_chemistry_first_20261001`：[源码与文档](examples/figure5_chemistry_first_20261001/) · [本地产物](../../reports/examples/figure5_chemistry_first_20261001/)
- `figure5_example_selection_20261001`：[源码与文档](examples/figure5_example_selection_20261001/) · [本地产物](../../reports/examples/figure5_example_selection_20261001/)
- `sample_interpretation_20261001`：[源码与文档](examples/sample_interpretation_20261001/) · [本地产物](../../reports/examples/sample_interpretation_20261001/)
- `sample_comparison_draft_20261001`：[源码与文档](examples/sample_comparison_draft_20261001/) · [本地产物](../../reports/examples/sample_comparison_draft_20261001/)
- `sample_comparison_backup_A022_A023_20261001`：[源码与文档](examples/sample_comparison_backup_A022_A023_20261001/) · [本地产物](../../reports/examples/sample_comparison_backup_A022_A023_20261001/)
- `sample_comparison_backup_A040_A041_20261001`：[源码与文档](examples/sample_comparison_backup_A040_A041_20261001/) · [本地产物](../../reports/examples/sample_comparison_backup_A040_A041_20261001/)
