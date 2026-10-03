# 目录设计与维护

借鉴 [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) 对 notebooks、可导入源代码、图表产物和文档的分离；按本项目规模放在单独的 `poster/`，没有引入模板框架或新环境。Notebook 的用途参考 [Jupyter Notebook 文档](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)：同一阅读顺序内结合叙述、公式、可执行计算和输出。

```text
poster/
  README.md
  notebooks/               # 一张主图一册；叙述和分析的权威入口
                           # 00_preparation 准备图 1 的响应表示输入
  src/poster_analysis/      # 可复用函数；导入无文件写入或拟合
  data/prepared/            # 从个体曲线缓存计算的派生输入；不放原始测量
  scripts/                 # 显式运行 notebook、生成目录索引的小工具
  figures/main/            # 本地四组主图与方法条，PNG/SVG/PDF；不提交
  figures/supporting/      # 同一 notebook 导出的相关探索图
  tables/                  # 图中数值、检查值和来源指纹
  docs/                    # 图注、方法与本地输入清单
docs/handoffs/              # 交接原文；根目录无副本或链接
notebooks/                  # 三本通用分析 notebook
src/bacteria_analysis/      # 通用分析模块
analysis/explorations/      # 按主题保存的历史源码与协议
tests/analysis/、tests/poster/ # 统一测试入口
reports/                   # 按主题分组的本地生成结果，不提交
```

## 函数与 notebook 的边界

| 模块 | 可复用职责 | Notebook 内保留的过程 |
|---|---|---|
| `paths.py`、`constants.py`、`style.py` | 解析仓库相对路径、来源别名、统一样式和导出 | 来源表、参数、输出目的地 |
| `preparation.py` | 读取及核对个体曲线缓存、SNR/加权 SVD 拟合、生成表示和显示表 | SNR 手算、单神经元加权 SVD、阈值循环、排序规则、显式保存和历史一致性核对 |
| `atlas.py` | 载入已保存数组，有限值条件汇总，图谱/模板/方法条布局 | 示例分箱、系数乘模板、显式条件汇总、旧表一致性 |
| `repeatability.py` | 读取缓存、唯一配对提取、分布与支持图布局 | 对角线/上三角构造、密度、支持计数、LOAO 等权汇总 |
| `global_comparison.py` | 对齐菌株、化学 RMS 和神经余弦距离、双矩阵及属距离图；按参数控制色标与 hexbin | 380 项 log2FC 主图计算；162 项浓度替代版的单对手算、全矩阵、唯一配对和距离背景分析 |
| `chemical_axes.py` | 仅在训练行上标准化、化学分组与变换 | 展示均值/SD、组规模、family 权重、单个分数计算 |
| `local_states.py` | 状态选择、OLS、三等分、观察组均值、训练折、局部图布局 | 显式设计矩阵、候选比较、选中成员贡献、失败模型、缓存留出误差 |

可复用函数不从磁盘隐式获取神经标签，不写历史目录，也不导入历史 `analysis/explorations/` 归档脚本。保存图函数限制输出到 `poster/figures/`。探索中只使用一次的几行绘图留在 notebook，便于顺着问题阅读和修改。

## 图 3 的主版与替代版

Notebook 03 先展示 380 项 log2FC 主版，再展示 162 项浓度替代版及相关探索。两者使用同一批原始化学测量和相同神经表示，但化学预处理不同：380 项为缺失填 0、加 1、按参考组计算 log2FC；162 项为完整、正浓度、QC RSD ≤ 0.30 筛选后的 log2 浓度。图 4 继续使用后者。

主版沿用 `fig03_global_chemical_neural.*` 图名和 `fig03_matched_pairs.csv`、`fig03_descriptive_correspondence.json`、`fig03_display_parameters.json`、`fig03_sources.csv` 表名。替代图位于 `supporting/fig03_global_chemical_neural_162concentration.*`，对应计算记录使用 `fig03_162concentration_` 前缀。色标和 hexbin 参数分别记录：主版神经色标 0–2、RdBu_r、gridsize=27；替代版 0–1.2、Blues、gridsize=25。两版不共用一份来源或显示参数记录。

全 106 株 HMDS 与主版使用同一 380 项化学定义，但仍放在支持探索；早期 A231/A232 的 7 神经元图仅作历史参考。

## 历史实现怎样处理

当前分析模块在 `src/bacteria_analysis/` 和 `poster/src/poster_analysis/`。历史探索的 Python 源码、协议和讨论已按六个主题迁入 `analysis/explorations/`，原始字节与路径记录在该目录的清单中；生成结果按相同主题移到 `reports/`。旧脚本包含当时的路径与运行环境，作为方法档案保存；要继续某项旧探索时需移植其依赖，当前 notebook 不调用它们。

Handoff 仅保留在 `docs/handoffs/`，根目录不保留兼容链接。旧 notebook 备份保留在本地 `archive/notebooks/`，不纳入当前版本。

Notebook 使用仓库相对路径，`report_path` 从 [catalogue.csv](../../analysis/explorations/catalogue.csv) 查找分组后的本地缓存。复制项目时应另外保留 [local_inputs.csv](local_inputs.csv) 中列出的文件；Git 不保存图表、派生数据或模型缓存。旧运行记录中的哈希描述当时源码，没有为目录迁移而伪造重新执行记录。

`00_preparation` 是明确的再计算步骤：从已审核个体 ΔF/F₀ 曲线生成图 1 的输入，而不是复制旧结果。`01` 只接受带完整输出指纹 manifest 的准备目录，准备文件缺失或变动会提示重跑 00，不会静默退回历史输入。该检查验证已发布文件的完整性；更新上游数据或准备算法后仍需主动重跑 00。旧显示表保留为外部核对基准。Split-half/LOAO 与化学输入的上游流程仍在历史报告中，未被声称已经迁移到 00。

## 运行与更新

1. 修改 notebook 中的参数、叙述、单次探索，或修改共享模块。
2. 用项目 kernel 从头执行受影响的 notebook。执行器为每本建立独立 kernel，将完整输出保存到本地 `reports/notebooks/poster/`；源码 notebook 保持无输出。
3. 查看 PNG 的标签、布局和色标，同时保留 SVG/PDF。
4. 用 `pixi run test` 检查相关测试。手动执行后，用 `pixi run clean-notebooks` 保存输出副本并清理源码。

生成 notebook 时使用了 Jupyter skill 的 experiment scaffold。生成过程的临时构建脚本不作为第二套维护源；以后直接维护 notebook 和共享函数，避免内容双份漂移。
