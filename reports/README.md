# 本地结果与缓存

本目录只保留生成产物，不存放活动源码。除本说明外，整个目录都不纳入 Git。

历史结果按六类分组：`population/`、`representation/`、`chemical_neural/`、`genus/`、`local_states/`、`examples/`。每个子目录继续使用原报告名称。对应的源码与协议在 [analysis/explorations/](../analysis/explorations/README.md)，旧路径、新源码路径、新结果路径见 [catalogue.csv](../analysis/explorations/catalogue.csv)。

- `notebooks/analysis/`、`notebooks/poster/` 保存完整已执行 notebook；`notebooks/before_layout_change/` 保留整理前的原始执行副本。
- `poster/` 保存当前运行与整理核对记录。`poster/verification_before_layout_change/` 是旧核对记录，里面的路径和哈希描述当时状态。
- 当前 poster 的图、表和准备结果仍分别写入仓库的 `poster/figures/`、`poster/tables/`、`poster/data/`，也仅保留本地。

本地输入目录不是可随意清空的缓存：部分文件是保存下来的科学记录。迁移机器时按 [poster 输入清单](../poster/docs/local_inputs.csv) 复制所需文件，并独立备份原始数据与历史结果。
