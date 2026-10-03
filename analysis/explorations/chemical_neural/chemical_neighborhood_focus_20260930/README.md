# 检阅与复现

先读 [REVIEW.md](REVIEW.md)。本轮的新入口已经收束为“化学近邻内部的直接响应表型”。主图为 [A022/A023](figures/chemical_neighbor_A022_A023.png)；[A007/A010](figures/chemical_neighbor_A007_A010.png)为同规则化学预选反例。两张原时程图为附图，所有图均有PNG/PDF。群体轴图与分类比较不再列为本轮主图，前轮文件完整保留。

## 实际执行

三个脚本均已在现有Pixi环境实际运行成功；化学审计包含有限敏感性计算，响应审计复用并独立回算既有留动物结果，绘图脚本逐点回算原曲线。没有重训化学/神经分类器，没有筛新分子、安装包或改原始数据。

在项目根目录依次执行：

    .pixi/envs/default/bin/python reports/chemical_neighborhood_focus_20260930/code/chemical_neighbor_audit.py
    .pixi/envs/default/bin/python reports/chemical_neighborhood_focus_20260930/code/response_evidence_audit.py
    .pixi/envs/default/bin/python reports/chemical_neighborhood_focus_20260930/code/01_plot_complement.py
    .pixi/envs/default/bin/python reports/chemical_neighborhood_focus_20260930/code/02_verify.py

也可在现有Notebook用 %run 分步调用。所有输出写到本轮独立目录；重跑会覆盖该目录中的同名结果，若需保存本次版本应先复制。未新增或修改Notebook。

## 输入与参数

输入主要为 reports/population_first_20260930/tables 下的化学、神经、pair及逐动物预测表。化学审计还使用其全299株参考谱、原报告值与元数据；响应图由此前animal_curves.parquet做独立验证。继承数据的原始来源、FC重建和Notebook逐值核对见[上一轮输入说明](../population_first_20260930/logs/alignment_methods.md)。

主化学口径固定106株×380项log₂FC；近邻选择在同菌属、同reference及同刺激目录内进行。扩大到化学全库及共同报告条目，仅检验已选实例的身份；没有更换主面板。伪计数0.1和10也是诊断，单位是合作实验室报告值的原任意单位，未将这些值当作可比较的绝对分子量。

神经固定13类×5个5秒窗，0–25秒。主图用该范围的动物内时间均值差，附图显示−5至24秒；留动物方向检查沿用五窗，未随本轮结果改窗。重复单位为动物，身份由采集块与worm_key共同识别。所有新脚本确定性执行，不引入随机筛选。

文件SHA-256和输出清单见[最终验证](logs/verification.json)、[输入清单](logs/input_manifest.json)、[输出清单](logs/output_manifest.json)。环境版本在[绘图验证](logs/figure_verification.json)，使用仓库现有pixi.lock。没有遗留失败的重点计算；没有进行新的动物实验、刺激aliquot化学重测或顺序随机化验证。

## 核查与数据对应

- 主图化学点：[example_chemical_points.csv](tables/example_chemical_points.csv)；包含两株FC及各自原报告是否有值。
- 主图动物点：[example_paired_response_points.csv](tables/example_paired_response_points.csv)；附图原曲线：[example_response_curves.csv](tables/example_response_curves.csv)。
- 全45对的覆盖和例外：[response_audit_pair_directions.csv](tables/response_audit_pair_directions.csv)；[删除影响](tables/response_audit_deletions.csv)。
- 化学定义的稳定性：[chemical_neighbor_audit_example_ranks.csv](tables/chemical_neighbor_audit_example_ranks.csv)、[敏感性](tables/chemical_neighbor_audit_sensitivity.csv)。

独立[化学审计](logs/chemical_neighbor_audit.md)和[响应证据审计](logs/response_evidence_audit.md)分别核对邻近定义、全群体向量和图中原始数值。这些是同一数据的实现与稳健性检查，不是独立生物学验证。详细限制和每张图的样本数、单位及误差含义见[REPORT.md](REPORT.md)。
