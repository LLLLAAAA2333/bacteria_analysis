# Bacterial stimuli: neural responses and chemical composition

当前 poster 从 **[poster/README.md](poster/README.md)** 开始：一本 preparation notebook，四本主图 notebook，每本包含对应的探索图和解释。

| 路径 | 内容 | Git |
|---|---|---|
| `notebooks/` | 测量检查、重复性、化学–神经探索的三本分析入口 | 代码与叙述 |
| `src/bacteria_analysis/` | 分析 notebook 共用的计算和绘图模块 | 提交 |
| `poster/notebooks/`、`poster/src/` | 当前 poster 的叙述、计算和绘图实现 | 提交 |
| `analysis/explorations/` | 六个主题下的历史探索源码、协议和讨论 | 原样归档并提交 |
| `tests/analysis/`、`tests/poster/` | 两组测试，统一运行 | 提交 |
| `docs/handoffs/` | 交接记录；根目录不保留副本或兼容链接 | 提交 |
| `reports/<主题>/` | 历史计算结果、缓存、运行参数与核对记录 | 仅本地 |
| `reports/notebooks/` | 已执行 notebook 副本，包含完整输出 | 仅本地 |
| `poster/figures/`、`poster/tables/`、`poster/data/` | 当前导出图表和 preparation 结果 | 仅本地 |
| `archive/notebooks/` | 旧 notebook 与脚本备份 | 仅本地 |
| `data/`、`output/`、`tmp/` | 原始输入、其它生成结果和临时文件 | 仅本地 |

历史探索与本地结果都按 `population`、`representation`、`chemical_neural`、`genus`、`local_states`、`examples` 分组，名称与路径映射见 [探索目录](analysis/explorations/README.md) 和 [catalogue.csv](analysis/explorations/catalogue.csv)。历史脚本保留当时的路径与源码指纹，属于方法记录；当前维护入口是 `notebooks/`、`src/` 和 `poster/`。

## 运行

环境由 `pixi.toml`、`pixi.lock` 定义。从仓库根目录运行：

```bash
pixi run lab                 # 打开 notebook
pixi run test                # 统一运行 analysis 和 poster 测试
pixi run clean-notebooks     # 先保存本地已执行副本，再清除源码 notebook 输出
```

`poster/scripts/execute_notebooks.py` 将成功运行的 notebook 写入 `reports/notebooks/poster/`，源码 notebook 不写入执行输出。手动运行 Jupyter 后，提交前执行 `pixi run clean-notebooks`。

Git 不保存图、PDF、派生数据或模型缓存。新 checkout 需要另外复制原始数据或已审核缓存；poster 的本地输入清单及 SHA-256 见 [local_inputs.csv](poster/docs/local_inputs.csv)。运行 `00_preparation` 可从这些缓存准备图 1 输入，它不包含原始荧光预处理或所有历史模型搜索。

完整目录说明见 [repository_layout.md](docs/repository_layout.md)。忽略规则不会删除文件；本地科学结果应另行备份。
