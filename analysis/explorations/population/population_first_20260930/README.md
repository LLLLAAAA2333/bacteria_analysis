# 复现与检阅入口

先读 [REVIEW.md](REVIEW.md)，判断与图注见 [REPORT.md](REPORT.md)，选择过程见 [RESEARCH_LOG.md](RESEARCH_LOG.md)。全部新增文件位于本目录；原始数据、Notebook与前轮结果没有改写。

## 实际执行状态

00–05均已实际运行，完成输入逐值核对、200次整动物拆分、留动物预测、嵌套整块留出和固定低维反证。06为交付前的数值、划分、权重、文件与校验检查。各步骤日志位于 logs；最终状态见 [verification.json](logs/verification.json)。

此次按步骤运行与修订，并非只提供未执行代码。run_analysis.py 是把相同脚本顺序调用的便利入口，尚未再从头重复运行一遍；所有组成分析已执行。最终修订中，01重绘权重附图并补充主图读图说明，02更新图与核查，03仅重新计算删除影响的权重；这些受影响部分已经运行，不需要重复不变的模型拟合。

04首次在计算结束后的Markdown格式化遇到缺少可选tabulate包；已去除依赖并全程重跑成功。未安装新包。没有遗留失败的重点分析；没有执行新的生物实验、实际刺激aliquot化学测量、受体机制实验或独立外部验证。

## 运行

在项目根目录使用现有Pixi环境：

    .pixi/envs/default/bin/python reports/population_first_20260930/run_analysis.py

也可依次执行下表的小脚本，或在现有Notebook中用 %run 调用。脚本读取相对于自身的位置，不依赖当前工作目录。重新运行只会替换本轮目录的同名输出；请先复制本目录，若希望保留当前交付版本。

仅检查交付文件：

    .pixi/envs/default/bin/python reports/population_first_20260930/code/06_verify.py

| 顺序 | 脚本 | 输出与用途 |
| --- | --- | --- |
| 00 | [00_align_inputs.py](code/00_align_inputs.py) | aligned_*表；原始神经流程与复用表逐值比较，FC及元数据审计 |
| 01 | [01_population_structure.py](code/01_population_structure.py) | population_structure_*表与3张图；完整群体结构、留动物实测对照 |
| 02 | [02_neural_value.py](code/02_neural_value.py) | neural_value_*表与3张图；化学预选近邻、逐动物差异及反证 |
| 03 | [03_population_chemistry.py](code/03_population_chemistry.py) | population_chemistry_*表；完整65维群体的化学预测 |
| 04 | [04_shared_task.py](code/04_shared_task.py) | shared_task_*表与1张附图；同一菌属任务的公平对照 |
| 05 | [05_latent_chemistry.py](code/05_latent_chemistry.py) | latent_chemistry_*表；训练内rank3/5群体目标反证 |
| 06 | [06_verify.py](code/06_verify.py) | 最终验证、输入与输出SHA-256清单 |

## 输入、环境及参数

00使用 data/106bac.parquet、data/matrix.xlsx、data/metabolism_raw_data.xlsx、data/GM300_bacteria_species_summary.xlsx 和当前 notebook/03_chemical_neuron_bacteria.ipynb。为避免无必要重建，01/02复用 reports/exploration_20260929/tables 下已核对的animal_curves、trial_curves、taxonomy、chemical_reference_groups及chemical_legacy_logfc。这些都是复现所需的本地依赖；不能只复制本轮目录而省略它们。

输入路径、文件大小及SHA-256见 [input_manifest.json](logs/input_manifest.json)；本轮开始记录的8个源文件还保留在 [alignment_inputs.json](logs/alignment_inputs.json)。附加复用表与此前交付清单比对。最终产物清单见 [output_manifest.json](logs/output_manifest.json)，不包含自身、流式运行日志、缓存及会变化的run_status.json。

环境为Python 3.11.16，使用仓库pixi.toml/pixi.lock。numpy、pandas、scipy、pyarrow、openpyxl、matplotlib及独立复拟用scikit-learn的实际版本见 [verification.json](logs/verification.json) 与 [alignment_environment.json](logs/alignment_environment.json)。未访问外部分析服务。

神经固定13×5坐标，五窗为[0,5)…[20,25)秒，单位ΔF/F₀；8窗40秒仅用于敏感性。化学固定106×380个log₂FC，无再次加1或逐特征标准化。输入中上游填零/伪计数、缺失细胞处理和同培养条件但独立培养批次的对应程度，详见REPORT。

随机种子：00为20260930；01为2026093001，200次拆分；02为2026093022，2,000次固定预测整动物bootstrap。03–05为确定性划分/线性代数，无随机特征选择。03/05的共享λ候选为0.001、0.01、0.1、1、10、100；04为0.0001、10⁻²·⁵、0.1、10⁰·⁵、100。未按测试成绩改变这些候选。

## 审核与图表

主图只有两张，各自回答一个问题；其余为对应附图。PNG用于检阅，PDF用于放大；同前缀表保存动物/菌株层的数据，REPORT提供具体链接与误差定义。图中不以日期编码主结论，支持集合仍在数据表中可追踪。

独立复核包括 [群体结构](logs/independent_population_review.md)、[化学近邻](logs/independent_neural_value_review.md)、[完整化学预测](logs/independent_review.md) 及 [低维化学反证](logs/independent_latent_review.md)。部分审计采用独立公式或scikit-learn复拟，没有把同数据复核称作独立生物学验证。

本目录不是新分析框架。需要调整参数时，修改相应小脚本并重新运行该步骤及其下游，另存输出以保留发现过程。
