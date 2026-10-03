# 仓库目录与保存规则

本次整理将分析入口、复用代码、历史记录和生成结果分开。原始数据与科学结果的内容没有修改，整理不触发全数据分析。

```text
notebooks/                         三本通用分析 notebook，Git 保存代码与叙述
src/bacteria_analysis/             共用分析模块
poster/
  notebooks/                      00 准备输入，01–04 对应四组主图
  src/poster_analysis/             poster 共用函数
  scripts/                        显式执行与来源索引
  docs/                           图注、方法、源代码与本地输入清单
  figures/、tables/、data/         本地生成物，不提交
analysis/explorations/
  population/                     早期总体探索
  representation/                 响应表示、模板与 SNR
  chemical_neural/                 化学–神经整体关系
  genus/                          属内与属间模式
  local_states/                   局部状态、模型与可靠性
  examples/                       历史展示和样本例子
tests/analysis/、tests/poster/      统一测试入口
docs/handoffs/                     历史交接；不在根目录保留链接
reports/                          同主题分组的本地产物、执行记录
archive/notebooks/                本地旧 notebook 快照
```

三个分析 notebook 使用 `src/bacteria_analysis/` 包；poster 仍使用自己的 `poster/src/poster_analysis/`。这里没有把历史运行脚本包装成新的批处理流水线。

历史报告的 283 个源码和文档按原始字节迁移，校验值见 [source_manifest.csv](../analysis/explorations/source_manifest.csv)。旧脚本及交接中的路径属于历史记录，通过 [catalogue.csv](../analysis/explorations/catalogue.csv) 查找现在的来源与结果位置。要继续某项旧探索时，需要先移植该脚本所需的路径；当前 notebook 不导入这些旧脚本。

JSON 拟合参数、验证日志、预测表、图、PDF 和二进制缓存留在本地。版本库仅保存科学方法源码、协议、说明与路径/输入指纹清单。新 checkout 需要单独提供本地输入，不能仅凭代码复现所有历史计算。

Notebook 源码和已执行副本分开：手动运行后用 `pixi run clean-notebooks` 先保存完整副本，再去除待提交 notebook 的输出。poster 执行脚本直接将输出写到 `reports/notebooks/poster/`。这次只核对目录、字节、代码语法、测试与读取路径，没有重跑科学分析。
